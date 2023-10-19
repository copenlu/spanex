from explain_interactions.eval import ComparisonEval
from typing import List
from explain_interactions.datamodels import ComparativeExplanation
import numpy as np
from explain_interactions.registry import register, EVAL_METHOD

TOP_K = 5


@register(_type=EVAL_METHOD, _name="comparison-eval-base")
class BaseComparisonEval(ComparisonEval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, comparative_explanations: List[ComparativeExplanation], *args, **kwargs):
        num_better = 0
        top_k = kwargs.get("top_k", TOP_K)
        for comparative_expl in comparative_explanations:
            avg_score_orig = np.mean([x.score for x in comparative_expl.base.expl_phrases[:top_k]])
            avg_score_random = np.mean([x.score for x in comparative_expl.alternate.expl_phrases[:top_k]])
            if avg_score_orig > avg_score_random:
                num_better += 1
        return (num_better / len(comparative_explanations)) * 100
