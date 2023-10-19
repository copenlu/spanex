# flake8: noqa
from typing import List, Union
from explain_interactions.datamodels import Instance, InstanceExplanation, TokenizedInstance
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.handlers import Handler


class Explainer:
    def __init__(self, handler: Handler, scorer: Scorer, **kwargs):
        self.handler = handler
        self.scorer = scorer

    def tokenize(self, instances: List[Instance]) -> List[TokenizedInstance]:
        return self.handler.tokenizer.tokenize_tx(instances)

    @staticmethod
    def filter(tokenized_instances: List[TokenizedInstance]) -> List[TokenizedInstance]:
        """
        This method is supposed to filter out some instances that can cause problem downstream.
        :param tokenized_instances:
        :return:
        """

        def part_contains_space(instance: TokenizedInstance) -> bool:
            """
            if a part of the instance contains a whitespace after spacy tokenization, tx tokenization gets screwed.
            remove them.
            :param instance:
            :return:
            """
            return any([x.text.strip() == "" for x in instance.part1_tokens]) or any(
                [x.text.strip() == "" for x in instance.part2_tokens]
            )

        filter_fns = [part_contains_space]
        return [x for x in tokenized_instances if not any([filter_fn(x) for filter_fn in filter_fns])]

    def run(self, instances: Union[List[Instance], List[TokenizedInstance]], *args, **kwargs) -> List[InstanceExplanation]:
        """
        Produce a list of explanations for a list of inputs. All explanation methods will implement this.
        :param instances:
        :param args:
        :param kwargs:
        :return: each instance can produce multiple explanations, but the explanation phrase has a key instance_idx that assigns
        it to an instance.
        """
        tokenized_instances = (
            self.tokenize(instances) if not hasattr(instances[0], "part1_tokens") else instances
        )  # we need to tokenize this
        self.tokenized_instances = self.filter(tokenized_instances)
        print(f"{self.__class__} filtered down from {len(tokenized_instances)} to {len(self.tokenized_instances)}")


from .annotation import AnnotationExplainer
from .alternate_explainer import RandomExplainerPhrase, PartPhrase
from .graphs import LouvainCommunitiesExplainer
