# flake8:noqa
from explain_interactions.handlers import Handler
from typing import List, Tuple
from explain_interactions.datamodels import Instance, InstanceExplanation
from dataclasses import dataclass
from explain_interactions.datamodels import TxTokenizedInstance, InstanceOutput, ExplanationPhrase, ExplanationToken


@dataclass
class InstExplOutput:
    instance_idx: str
    tokenized_instance: TxTokenizedInstance
    instance_output: InstanceOutput


@dataclass
class InstExplTokenOutput(InstExplOutput):
    explanation_token: ExplanationToken


@dataclass
class InstExplPhraseOutput(InstExplOutput):
    explanation_phrase: ExplanationPhrase


class Scorer:
    """
    Given an input and an Explanation, produce scores.
    """

    def __init__(self, handler: Handler, **kwargs):
        self.handler = handler
        self.tokenizer = self.handler.tokenizer if self.handler is not None else None

    def run(self, instance_explanations: List[Tuple[Instance, InstanceExplanation]]) -> List[InstanceExplanation]:
        """
        produce scores from
        :param instance_explanations:
        :return:
        """
        pass


from .attention import BaseAttentionWeightScorerPhrase
from .base import BaseScorer
