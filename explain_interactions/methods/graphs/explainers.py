from typing import Dict, List, Union, NewType, Set
from explain_interactions.datamodels import (
    InstanceExplanation,
    TokenizedInstance,
    Instance,
    TxTokenizedInstance,
    ExplanationPhrase,
    Phrase,
    Token,
)
from explain_interactions.handlers import Handler
from networkx import Graph
from explain_interactions.methods.graphs.extract import GraphExtractor
from explain_interactions.methods.graphs.louvain import louvain_communities
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.registry import EXPL_METHOD, register, GRAPH_EXTRACTOR_REGISTRY
from explain_interactions.methods import Explainer
from tqdm import tqdm

Community = NewType("community", Set[int])  # graph nodes are integer indices


class AttentionGraphExplainer(Explainer):
    """
    For an instance, we have the attention weights, get a graph out of that.
    """

    def __init__(self, handler: Handler, scorer: Scorer, graph_extractor_params: Dict, **kwargs):
        super().__init__(handler=handler, scorer=scorer, **kwargs)
        self.graph_extractor: GraphExtractor = GRAPH_EXTRACTOR_REGISTRY[graph_extractor_params["name"]](
            **{**graph_extractor_params, **{"handler": self.handler}}
        )

    def generate_communities(self, graph: Graph) -> List[Community]:
        """
        A community is a set of nodes. Subclasses will implement this
        :param graph:
        :return:
        """
        raise NotImplementedError("subclass needs to implement this")

    def generate_phrases(self, tokens: List[Token], tok_indices: List[int]) -> List[Phrase]:
        """
        Generate phrases from tokens
        :param tokens:
        :param tok_indices:
        :return:
        """

        def create_phrase(_curr_phrase_tok_indices: List[int]) -> Phrase:
            return Phrase(
                start_token_index=_curr_phrase_tok_indices[0],
                end_token_index=_curr_phrase_tok_indices[-1],
                text=" ".join([x.text for x in tokens[_curr_phrase_tok_indices[0] : _curr_phrase_tok_indices[-1] + 1]]),
            )

        phrases = []
        current_phrase_tok_indices = []
        for i in range(len(tok_indices) - 1):
            current_phrase_tok_indices.append(tok_indices[i])
            if tok_indices[i + 1] != tok_indices[i] + 1:  # we are no longer in a phrase, write this out
                phrases.append(create_phrase(current_phrase_tok_indices))
                current_phrase_tok_indices = []
        if current_phrase_tok_indices:
            if current_phrase_tok_indices[-1] + 1 == tok_indices[-1]:  # need to add the last token to a phrase
                current_phrase_tok_indices.append(tok_indices[-1])
                phrases.append(create_phrase(current_phrase_tok_indices))
            else:
                phrases.append(create_phrase(current_phrase_tok_indices))
                phrases.append(create_phrase([tok_indices[-1]]))
        else:
            phrases.append(create_phrase([tok_indices[-1]]))
        return phrases

    def create_phrase_pairs_from_communities(
        self, instance: TxTokenizedInstance, communities: List[Community]
    ) -> List[ExplanationPhrase]:
        """
        A community is a list of nodes
        :param instance:
        :param communities:
        :return:
        """
        all_phrase_pairs = []
        for index, community in enumerate(communities):
            if len(community) < 2:
                continue
            part1_tok_indices = [index for index, x in enumerate(instance.part1_map) if all([y in community for y in x])]
            # Sagnik: this is a little tricky, we are including a token if any of its subwords of it does not belong in
            #  a community. Should we change it to `all`?
            part2_tok_indices = [index for index, x in enumerate(instance.part2_map) if all([y in community for y in x])]
            if not part1_tok_indices or not part2_tok_indices:
                continue
            part1_phrases = self.generate_phrases(instance.part1_tokens, part1_tok_indices)
            part2_phrases = self.generate_phrases(instance.part2_tokens, part2_tok_indices)
            for phrase_par1 in part1_phrases:
                for phrase_part2 in part2_phrases:
                    all_phrase_pairs.append(
                        ExplanationPhrase(
                            instance_idx=instance.idx,
                            part1=phrase_par1,
                            part2=phrase_part2,
                            score=0.0,
                        )
                    )
        return all_phrase_pairs

    @staticmethod
    def visualize_communities(instance: TxTokenizedInstance, communities: List[Community]):
        """
        print the communities for an instance in an understandable way
        :param instance:
        :param communities:
        :return:
        """
        print(
            f"[instance idx]: {instance.idx}, [part1]: {instance.part1}, [part2]: {instance.part2}, [label]: {instance.label_index}"
        )
        for index, community in enumerate(communities):
            if len(community) < 2:  # individual nodes
                continue
            print("-" * 30)
            # a community: 1, 2, 3, 20, 10, 11, 12
            # part1 token map: [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15, 16],
            # [17]]
            # part2 token map: [[19], [20], [21], [22]]
            part1_tok_indices = [index for index, x in enumerate(instance.part1_map) if all([y in community for y in x])]
            # Sagnik: this is a little tricky, we are including a token if any of its subwords of it does not belong in
            #  a community. Should we change it to `all`?
            part2_tok_indices = [index for index, x in enumerate(instance.part2_map) if all([y in community for y in x])]
            print(f"[comm {index}]")
            print(f"[part1 tokens]: [{[x.text for i, x in enumerate(instance.part1_tokens) if i in part1_tok_indices]}]")
            print(f"[part2 tokens]: [{[x.text for i, x in enumerate(instance.part2_tokens) if i in part2_tok_indices]}]")
        print("=" * 30)

    def generate_expls_from_communities(
        self, instance: TxTokenizedInstance, communities: List[Community], expl_type: str = "phrase"
    ) -> InstanceExplanation:
        """
        A community is a list of nodes. The first
        :param instance:
        :param communities:
        :param expl_type:
        :return:
        """
        if expl_type == "phrase":
            return InstanceExplanation(
                instance_idx=instance.idx,
                expl_tokens=[],
                expl_phrases=self.create_phrase_pairs_from_communities(instance, communities),
            )
        else:
            raise NotImplementedError("only phrase pairs are supported")

    def run(self, instances: Union[List[Instance], List[TokenizedInstance]], **kwargs) -> List[InstanceExplanation]:
        """
        each instance must have an InstanceExplanations already defined.
        :param instances:
        :param kwargs:
        :return:
        """
        super().run(instances)
        instance_communities = {
            instance.idx: self.generate_communities(instance_graph.graph)
            for instance, instance_graph in zip(
                self.tokenized_instances, self.graph_extractor.run(self.tokenized_instances).data
            )
        }
        _tok_instances = {tok_instance.idx: tok_instance for tok_instance in self.tokenized_instances}
        # visualize if needed
        # [
        #     self.visualize_communities(_tok_instances[idx], communities=communities)
        #     for idx, communities in instance_communities.items()
        # ]
        instance_expls = {
            idx: self.generate_expls_from_communities(_tok_instances[idx], communities)
            for idx, communities in instance_communities.items()
        }
        # score the explanations before returning
        return self.scorer.run(
            instance_explanations=[
                (instance, instance_expls[instance.idx])
                for instance in tqdm(self.tokenized_instances, desc=f"scoring explanations, {self.scorer}")
            ]
        )


@register(_type=EXPL_METHOD, _name="louvain_community")
class LouvainCommunitiesExplainer(AttentionGraphExplainer):
    def __init__(self, handler: Handler, scorer: Scorer, graph_extractor_params: Dict, **kwargs):
        super().__init__(handler=handler, scorer=scorer, graph_extractor_params=graph_extractor_params, **kwargs)

    def generate_communities(self, graph: Graph) -> List[Community]:
        """
        A community is a set of nodes with a score
        :param graph:
        :return:
        """
        return louvain_communities(graph)
