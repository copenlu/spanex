from typing import List, Tuple
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.handlers import Handler
from explain_interactions.datamodels import Instance, InstanceExplanation, ExplanationToken, ExplanationPhrase
from explain_interactions.registry import register, EXPL_SCORER

DEFAULT_SCORE = 0.5


@register(_type=EXPL_SCORER, _name="base")
class BaseScorer(Scorer):
    def __init__(self, handler: Handler, **kwargs):
        super().__init__(handler, **kwargs)
        self.default_score = DEFAULT_SCORE

    def add_default_score_token(self, _inputs: List[ExplanationToken]) -> List[ExplanationToken]:
        return [ExplanationToken(**{**_input.__dict__, **{"score": self.default_score}}) for _input in _inputs]

    def add_default_score_phrase(self, _inputs: List[ExplanationPhrase]) -> List[ExplanationPhrase]:
        return [ExplanationPhrase(**{**_input.__dict__, **{"score": self.default_score}}) for _input in _inputs]

    def run(self, instance_explanations: List[Tuple[Instance, InstanceExplanation]]) -> List[InstanceExplanation]:
        output = []
        for _, inst_expl in instance_explanations:
            output.append(
                InstanceExplanation(
                    instance_idx=inst_expl.instance_idx,
                    expl_tokens=self.add_default_score_token(inst_expl.expl_tokens),
                    expl_phrases=self.add_default_score_phrase(inst_expl.expl_phrases),
                )
            )
        return output
