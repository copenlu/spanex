from typing import Tuple, List
import numpy as np
from explain_interactions.datamodels import (
    Instance,
    InstanceExplanation,
    ExplanationPhrase,
    TxTokenizedInstance,
    Phrase,
    InstanceOutput,
)
from explain_interactions.methods.expl_scorers import Scorer, InstExplPhraseOutput
from explain_interactions.handlers import Handler
from explain_interactions.registry import register, EXPL_SCORER
from collections import defaultdict


class AttentionWeightScorerPhrase(Scorer):
    """
    Explanations are of type ExplanationPhrase, and the scores are generated from attention weights.
    """

    def __init__(self, handler: Handler, **kwargs):
        super().__init__(handler, **kwargs)
        self.layer = kwargs.get("layer")
        self.head = kwargs.get("head")

    def generate_output(self, instance_explanations: List[Tuple[Instance, InstanceExplanation]]) -> List[InstExplPhraseOutput]:
        _instance_inputs = {instance.idx: instance for instance, _ in instance_explanations}
        if hasattr(instance_explanations[0][0], "part1_map"):
            _tokenized_instances = _instance_inputs
        else:
            _tokenized_instances = {
                k: v for k, v in zip(_instance_inputs.keys(), self.tokenizer.tokenize_tx(list(_instance_inputs.values())))
            }
        _instance_outputs = {
            k: v for k, v in zip(_instance_inputs.keys(), self.handler.predict(list(_tokenized_instances.values())))
        }
        _instance_explanations = {instance.idx: explanation.expl_phrases for instance, explanation in instance_explanations}
        output = []
        for k, _instance_output in _instance_outputs.items():
            for expl in _instance_explanations[k]:
                output.append(
                    InstExplPhraseOutput(
                        instance_idx=_instance_inputs[k].idx,
                        tokenized_instance=_tokenized_instances[k],
                        instance_output=_instance_output,
                        explanation_phrase=expl,
                    )
                )
        return output

    def _run(self, instance_expl_output: InstExplPhraseOutput) -> float:
        pass

    def run(self, instance_explanations: List[Tuple[Instance, InstanceExplanation]]) -> List[InstanceExplanation]:
        inst_expl_outputs: List[InstExplPhraseOutput] = self.generate_output(instance_explanations)
        results = defaultdict(list)
        for inst_expl_output in inst_expl_outputs:
            results[inst_expl_output.instance_idx].append(
                ExplanationPhrase(
                    instance_idx=inst_expl_output.instance_idx,
                    part1=inst_expl_output.explanation_phrase.part1,
                    part2=inst_expl_output.explanation_phrase.part2,
                    score=self._run(inst_expl_output),
                    level=inst_expl_output.explanation_phrase.level,
                    relation=inst_expl_output.explanation_phrase.relation,
                    annotator=inst_expl_output.explanation_phrase.annotator,
                ),
            )
        return [InstanceExplanation(instance_idx=k, expl_tokens=[], expl_phrases=v) for k, v in results.items()]


@register(_type=EXPL_SCORER, _name="base-attn_weight-phrase")
class BaseAttentionWeightScorerPhrase(AttentionWeightScorerPhrase):
    """
    The most simple way to generate score from attention weights.
    """

    def __init__(self, handler: Handler, **kwargs):
        super().__init__(handler=handler, **kwargs)

    @staticmethod
    def get_tx_tokens_from_phrase(tx_tokenized_instance: TxTokenizedInstance, phrase: Phrase, part: str):
        output = []
        if part == "part1":
            for index, val in enumerate(tx_tokenized_instance.part1_map):
                if phrase.start_token_index <= index <= phrase.end_token_index:
                    output.extend(val)
        elif part == "part2":
            for index, val in enumerate(tx_tokenized_instance.part2_map):
                if phrase.start_token_index <= index <= phrase.end_token_index:
                    output.extend(val)
        else:
            raise RuntimeError("part must be part 1 or 2")
        return output

    def _run(self, instance_expl_output: InstExplPhraseOutput) -> float:
        instance_output = instance_expl_output.instance_output
        self.layer = self.layer if self.layer != -1 else len(instance_output.att_weights) - 1
        self.head = self.head if self.head != -1 else len(instance_output.att_weights[0]) - 1
        attention_weight = (
            instance_output.att_weights[self.layer][self.head].numpy().astype(float)
            if type(instance_output) == InstanceOutput
            else instance_output.att_weights.numpy().astype(float)
        )  # T x T
        expl_tx_token_indices_part_1 = self.get_tx_tokens_from_phrase(
            instance_expl_output.tokenized_instance, instance_expl_output.explanation_phrase.part1, "part1"
        )
        expl_tx_token_indices_part_2 = self.get_tx_tokens_from_phrase(
            instance_expl_output.tokenized_instance, instance_expl_output.explanation_phrase.part2, "part2"
        )
        weights = []
        for part1_index in expl_tx_token_indices_part_1:
            for part2_index in expl_tx_token_indices_part_2:
                weights.append(attention_weight[part1_index, part2_index] + attention_weight[part2_index, part1_index])
        return np.mean(weights) / 2
