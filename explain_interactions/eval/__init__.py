"""
There are two types of evaluations
================================
PerturbationEvaluation
-----------
Init
    - handler
__call__
    Input:
        - instance
        - explanation
    Output:
        - float


ComparisonEvaluation
--------------
Init
    - pass
Input
    - instance
    - real_explanation
    - random_explanation(s)
Output
    - bool
"""
# flake8: noqa
from typing import Dict, List, Union, NewType, Tuple
from explain_interactions.datamodels import (
    Instance,
    InstanceExplanation,
    ComparativeExplanation,
    ExplanationToken,
    ExplanationPhrase,
    TxTokenizedInstance,
)
from explain_interactions.eval.perturbations import Perturber
from explain_interactions.handlers import Handler
from collections import defaultdict

Scalar = NewType("Scalar", Union[int, float])


class Eval:
    def __init__(self, handler: Handler, **kwargs):
        self.handler = handler
        self.split_explanation_params = kwargs.get("split_explanation_params", False)
        self.token_or_phrase = kwargs.get("token_or_phrase", "phrase")
        self.sep = "$"

    def split_explanations(
        self, _instances: List[Instance], inst_explanations: List[InstanceExplanation], split_params: List[str]
    ) -> Dict[str, List[InstanceExplanation]]:
        """
        which params the explanations will be split upon, default instance_idx
        :param _instances:
        :param inst_explanations:
        :param split_params:
        :return:
        """

        def get_key(
            _inst_expl: InstanceExplanation, _expl_phrase_or_token: Union[ExplanationToken, ExplanationPhrase], label_index: int
        ):
            _key = _inst_expl.instance_idx
            if "ann" in split_params:
                _key += f"{self.sep}{_expl_phrase_or_token.annotator}"
            else:
                _key += f"{self.sep}NA"
            if "relation" in split_params:
                _key += f"{self.sep}{_expl_phrase_or_token.relation}"
            else:
                _key += f"{self.sep}NA"
            if "level" in split_params:
                _key += f"{self.sep}{_expl_phrase_or_token.level}"
            else:
                _key += f"{self.sep}NA"
            if "label_index" in split_params:
                _key += f"{self.sep}{label_index}"
            else:
                _key += f"{self.sep}NA"
            return _key

        _instances: Dict[str, Instance] = {x.idx: x for x in _instances}
        d_phrase = defaultdict(list)
        for inst_expl in inst_explanations:
            for expl_phrase in inst_expl.expl_phrases:
                d_phrase[
                    # f"{inst_expl.instance_idx}{self.sep}{expl_phrase.annotator}{self.sep}{expl_phrase.relation}{self.sep}{expl_phrase.level}{self.sep}{_instances[inst_expl.instance_idx].label_index}"
                    get_key(inst_expl, expl_phrase, _instances[inst_expl.instance_idx].label_index)
                ].append(expl_phrase)
        d_token = defaultdict(list)
        for inst_expl in inst_explanations:
            for expl_token in inst_expl.expl_tokens:
                d_token[
                    # f"{inst_expl.instance_idx}{self.sep}{expl_token.annotator}{self.sep}{expl_token.label}{self.sep}{expl_token.level}{self.sep}{_instances[inst_expl.instance_idx].label_index}"
                    get_key(inst_expl, expl_token, _instances[inst_expl.instance_idx].label_index)
                ].append(expl_token)
        _d = defaultdict(list)
        for k, v_phrase in d_phrase.items():
            instance_idx = k.split(self.sep)[0]
            _d[self.sep.join(k.split(self.sep)[1:])].append(
                InstanceExplanation(instance_idx=instance_idx, expl_phrases=v_phrase, expl_tokens=d_token[k])
            )
        return _d


class PerturbationEval(Eval):
    def __init__(self, handler: Handler, perturber: Perturber, **kwargs):
        self.perturber = perturber
        self.top_k = kwargs.get("top_k", -1)
        super().__init__(handler=handler, **kwargs)

    def perturb(self, orig: Instance, explanation: InstanceExplanation, **kwargs) -> List[Instance]:
        """
        Given an explanation of type token/chunk, create a perturbed dataset from that explanation.
        For example, when the underlying tokenizer is HF based, replace the explanation tokens by MASK.
        The output of this method must be consumable by the predict method.
        :param orig:
        :param explanation:
        :return:
        """
        return [x[1] for x in self.perturb_with_num(orig, explanation, **kwargs)]

    def perturb_with_num(self, orig: Instance, explanation: InstanceExplanation, **kwargs) -> List[Tuple[int, Instance]]:
        """
        Given an explanation of type token/chunk, create a perturbed dataset from that explanation.
        For example, when the underlying tokenizer is HF based, replace the explanation tokens by MASK.
        This method also returns the number of tokens that has been changed in each perturbed instance, we will
        use this for averaging later on.
        :param orig:
        :param explanation:
        :return:
        """
        return self.perturber.run(orig, explanation, **kwargs)

    def predict(self, instances: List[TxTokenizedInstance]) -> Scalar:
        pass

    def _run(
        self,
        _instances: Dict[str, TxTokenizedInstance],
        _explanations: List[InstanceExplanation],
        ann: str = "NA",
        level: str = "NA",
        relation: str = "NA",
        class_label: str = "NA",
    ) -> float:
        pass

    def run(
        self, instances: List[Instance], explanations: List[InstanceExplanation], **kwargs
    ) -> Union[float, Dict[str, float]]:
        """
        Run the evaluation algorithm on a list of explanations of the form [instance, explanation].
        `original` must have the same type as each instance of the output of the `perturb` method.
        IOW, `[original]+ perturb(*)` should be consumable by `predict`.
        If we don't have to split by  the explanation type, just return a float. Else, return a dictionary of floats.
        **Note**: The implementation assumes that the `predict` method uses a SequentialSampler.
        :param instances:
        :param explanations:
        :return:
        """

        non_tok_instances = instances
        instances: Dict[str, TxTokenizedInstance] = {
            instance.idx: instance for instance in self.handler.tokenizer.tokenize_tx(instances=instances)
        }
        assert len(instances) == len(explanations)
        split_explanations_params = kwargs.get("split_explanation_params", self.split_explanation_params)
        if split_explanations_params is not None:
            explanations: Dict[str, List[InstanceExplanation]] = self.split_explanations(
                _instances=non_tok_instances, inst_explanations=explanations, split_params=split_explanations_params
            )
            results = {
                k: self._run(
                    _instances=instances,
                    _explanations=v,
                    ann=k.split(self.sep)[0],
                    level=k.split(self.sep)[2],
                    relation=k.split(self.sep)[1],
                    class_label=k.split(self.sep)[3],
                )
                for k, v in explanations.items()
            }
        else:
            results = self._run(_instances=instances, _explanations=explanations)
        return results


class ComparisonEval(Eval):
    def __init__(self, handler: Handler, **kwargs):
        super(ComparisonEval, self).__init__(handler=handler, **kwargs)

    def run(self, comparative_explanations: List[ComparativeExplanation], *args, **kwargs) -> Union[float, Dict]:
        """
        In what percentage of cases is the original explanation better than the random explanation for the same instance?
        :param comparative_explanations:
        :param args:
        :param kwargs:
        :return:
        """


from .auc import AUC, PhaHF
from .auc_norm import AUCNormalized, PhaHFHFNormalized
from .comparison import BaseComparisonEval
