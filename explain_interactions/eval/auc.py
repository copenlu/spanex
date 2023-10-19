from typing import Dict, List
import numpy as np
from explain_interactions.datamodels import TxTokenizedInstance, InstanceExplanation, InstanceOutput
from explain_interactions.eval.perturbations import Perturber
from tqdm import tqdm
from explain_interactions.registry import register, EVAL_METHOD
from explain_interactions.eval import PerturbationEval
from explain_interactions.handlers import Handler


@register(_type=EVAL_METHOD, _name="auc")
class AUC(PerturbationEval):
    """
    See discussions on Overleaf. For manual explanations, we can not rank explanations, so we can only produce a
    comprehensiveness/sufficiency score
    """

    def __init__(self, handler: Handler, perturber: Perturber, **kwargs):
        super().__init__(handler=handler, perturber=perturber, **kwargs)

    def predict(self, instances) -> np.ndarray:
        """
        Get the logits of the given inputs (B x num_label), and choose the ones that correspond to the predicted class index
        in the original instance.
        Example: input: [this is good, this MASK good], logits: [[0.8, 0.2], [0.3, 0.6]], output = [0.8, 0.3]
        as the predicted class label is 0
        :param instances: Union[List[Instance], List[TokenizedInstance], [TokenizedTxInstance]]. Assumes the first instance
        is the original one, and the rest is perturbed.
        :return:
        """
        model_output_instances: List[InstanceOutput] = self.handler.predict(instances=instances)
        orig_pred_class_idx = model_output_instances[0].pred_class_index
        return np.array([moi.probs[orig_pred_class_idx] for moi in model_output_instances])

    def _run(
        self,
        _instances: Dict[str, TxTokenizedInstance],
        _explanations: List[InstanceExplanation],
        ann: str = "NA",
        level: str = "NA",
        relation: str = "NA",
        class_label: str = "NA",
    ) -> float:
        _results = []
        for inst_expl in tqdm(
            _explanations,
            desc=f"Eval: {self.__class__}, ann: {ann}, level: {level}, relation: {relation}, " f"class_label: {class_label}",
        ):
            instance = _instances[inst_expl.instance_idx]
            perturbed = self.perturb(instance, inst_expl, top_k=self.top_k)
            confs = self.predict(instances=[instance] + perturbed)
            assert len(confs) == 2  # batch size is 2
            _results.append(confs[0] - confs[1])
        return np.mean(_results)


@register(_type=EVAL_METHOD, _name="pha-hf-phrase")
class PhaHF(PerturbationEval):
    """
    post-hoc accuracy
    """

    def __init__(self, handler: Handler, perturber: Perturber, **kwargs):
        super().__init__(handler=handler, perturber=perturber, **kwargs)

    def predict(self, instances) -> int:
        """
        If the predicted class label of the perturbed instance does not differ from the predicted class label of the original
        instance, return 1 else return 0.
        This is the same as computing the preserved accuracy of a model for the perturbed instances in the dataset
        Example: input: [this is good, this good], logits: [[0.8, 0.2], [0.3, 0.7]], output = 0 as the predicted class label has
        changed.
        :param instances: Union[List[Instance], List[TokenizedInstance], [TokenizedTxInstance]]. Assumes the first instance
        is the original one, and the rest is perturbed.
        :return:
        """
        model_output_instances: List[InstanceOutput] = self.handler.predict(instances=instances)
        assert len(model_output_instances) == 2
        if model_output_instances[0].pred_class_index == model_output_instances[1].pred_class_index:
            return 1
        return 0

    def _run(
        self,
        _instances: Dict[str, TxTokenizedInstance],
        _explanations: List[InstanceExplanation],
        ann: str = "NA",
        level: str = "NA",
        relation: str = "NA",
        class_label: str = "NA",
    ) -> float:
        _results = []
        for inst_expl in tqdm(
            _explanations,
            desc=f"Eval: {self.__class__}, ann: {ann}, level: {level}, relation: {relation}, " f"class_label: {class_label}",
        ):
            instance = _instances[inst_expl.instance_idx]
            perturbed = self.perturb(instance, inst_expl, top_k=self.top_k)
            _results.append(self.predict(instances=[instance] + perturbed))
        return np.mean(_results)
