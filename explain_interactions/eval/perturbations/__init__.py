# flake8: noqa
from explain_interactions.datamodels import Instance, InstanceExplanation
from typing import List, Tuple


class Perturber:
    """
    Take an input, an explanation and perturb it to produce one or multiple instances
    """

    def __init__(self, **kwargs):
        pass

    def run(self, orig: Instance, instance_explanation: InstanceExplanation, **kwargs) -> List[Tuple[int, Instance]]:
        """
        Given an explanation of type token/phrase, create one/multiple perturbed instances from that explanation.
        For example, when the underlying tokenizer is HF based, replace the explanation tokens by MASK.
        We will also return the number of tokens that has been changed in each perturbed instance.
        :param orig:
        :param instance_explanation:
        :return:
        """


from .hf import CompHFPerturberPhrase, SuffHFPerturberPhrase
