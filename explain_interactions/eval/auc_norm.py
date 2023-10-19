"""
For annotation/random-phrase/part-phrase explanations, we don't have the concept of ``top-k'' phrase pairs as we
don't generate these explanations. Therefore, we would probably want to normalize the AUC comprehensiveness and
sufficiency scores in some way as they do depend on a) the number of phrase pairs, b) number of tokens in each phrase pair.
IOW, the evaluations are explanation length dependent. A simple normalization is to divide the scores by the number of
perturbed tokens.
"""
from typing import Dict, List, Union, NewType
import numpy as np
from explain_interactions.datamodels import TxTokenizedInstance, InstanceExplanation
from explain_interactions.eval.perturbations import Perturber
from tqdm import tqdm
from explain_interactions.registry import register, EVAL_METHOD
from explain_interactions.eval.auc import AUC, PhaHF
from explain_interactions.handlers import Handler

Scalar = NewType("Scalar", Union[int, float])


@register(_type=EVAL_METHOD, _name="auc-norm")
class AUCNormalized(AUC):
    """
    See discussions on Overleaf. For manual explanations, we can not rank explanations, so we can only produce a
    comprehensiveness/sufficiency score
    """

    def __init__(self, handler: Handler, perturber: Perturber, **kwargs):
        super().__init__(handler=handler, perturber=perturber, **kwargs)

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
            num_tokens_changed_perturbed = self.perturb_with_num(instance, inst_expl, top_k=self.top_k)
            num_tokens_changed = [x[0] for x in num_tokens_changed_perturbed]
            perturbed = [x[1] for x in num_tokens_changed_perturbed]
            assert len(num_tokens_changed) == 1
            if num_tokens_changed[0] == 0:
                # this should not ideally happen, but in one SNLI instance for a phrase pair, somehow the explanation
                # is the entire thing (probably because the ann didn't annotate anything), we need to remove this data point from all data files.
                # HFInstance(part1='One tan girl with a wool hat is running and leaning over an object, while another person in a wool hat is sitting on the ground.', part2='A man watches his daughter leap', idx='186fdbb5f6332894d0610b95cde78fe2', label_index=1
                # InstanceExplanation(instance_idx='186fdbb5f6332894d0610b95cde78fe2', expl_tokens=[], expl_phrases=[ExplanationPhrase(instance_idx='186fdbb5f6332894d0610b95cde78fe2', score=0.5, part1=Phrase(text='One tan girl with a wool hat is running and leaning over an object , while another person in a wool hat is sitting on the ground .', start_token_index=0, end_token_index=27), part2=Phrase(text='A man watches his daughter leap', start_token_index=0, end_token_index=5), level='high', relation='Dangler-SYSTEM-HYPOTHESIS', annotator='1')])
                continue
            confs = self.predict(instances=[instance] + perturbed)
            assert len(confs) == 2  # batch size is 2
            _results.append((confs[0] - confs[1]) / num_tokens_changed[0])
        return np.mean(_results)


@register(_type=EVAL_METHOD, _name="pha-hf-phrase-norm")
class PhaHFHFNormalized(PhaHF):
    """
    Return logit diff/num_tokens_perturbed. See base class for detailed explanation.
    """

    def __init__(self, handler: Handler, perturber: Perturber, **kwargs):
        super().__init__(handler=handler, perturber=perturber, **kwargs)

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
            num_tokens_changed_perturbed = self.perturb_with_num(instance, inst_expl, top_k=self.top_k)
            num_tokens_changed = [x[0] for x in num_tokens_changed_perturbed]
            assert len(num_tokens_changed) == 1
            if num_tokens_changed[0] == 0:
                # this should not ideally happen, but in one SNLI instance for a phrase pair, somehow the explanation
                # is the entire thing (probably because the ann didn't annotate anything), we need to remove this data point from all data files.
                # HFInstance(part1='One tan girl with a wool hat is running and leaning over an object, while another person in a wool hat is sitting on the ground.', part2='A man watches his daughter leap', idx='186fdbb5f6332894d0610b95cde78fe2', label_index=1
                # InstanceExplanation(instance_idx='186fdbb5f6332894d0610b95cde78fe2', expl_tokens=[], expl_phrases=[ExplanationPhrase(instance_idx='186fdbb5f6332894d0610b95cde78fe2', score=0.5, part1=Phrase(text='One tan girl with a wool hat is running and leaning over an object , while another person in a wool hat is sitting on the ground .', start_token_index=0, end_token_index=27), part2=Phrase(text='A man watches his daughter leap', start_token_index=0, end_token_index=5), level='high', relation='Dangler-SYSTEM-HYPOTHESIS', annotator='1')])
                continue
            perturbed = [x[1] for x in num_tokens_changed_perturbed]
            _results.append(self.predict(instances=[instance] + perturbed) / num_tokens_changed[0])
        return np.mean(_results)
