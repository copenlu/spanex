from explain_interactions.eval.perturbations import Perturber
from explain_interactions.datamodels import Token, TokenMap, HFInstance, Phrase, ExplanationPhrase, InstanceExplanation
from explain_interactions.handlers import Handler
import torch
from typing import List, Tuple, Optional
from copy import deepcopy
from explain_interactions.registry import register, PERTURBATION_METHOD

REPLACE_TOKEN = "[MASK]"
REPLACE_TOKEN_RB = "<mask>"


class HFPerturberPhrase(Perturber):
    """
    The explanations are of type phrase. Perturb the original instance with multiple explanations and return 1 perturbed
    instance.
    """

    def __init__(self, handler: Handler, **kwargs):
        super().__init__(**kwargs)
        self.replace_token = REPLACE_TOKEN_RB if "roberta" in handler.tokenizer.name else REPLACE_TOKEN
        self.replace_token_id = handler.tokenizer.vocab[self.replace_token]

    def replace_input_ids_text_phrase(
        self,
        text: List[Token],
        input_ids: torch.Tensor,
        phrases: List[Phrase],
        token_map: TokenMap,
        replace_token: str,
        replace_token_id: int,
    ) -> Tuple[List[Token], torch.Tensor]:
        pass

    def perturb_by_phrase(
        self, orig: HFInstance, phrase_explanations: List[ExplanationPhrase], **kwargs
    ) -> List[Tuple[int, HFInstance]]:
        top_k = kwargs.get("top_k", -1)
        phrase_explanations = sorted(phrase_explanations, key=lambda x: x.score, reverse=True)
        phrase_explanations = phrase_explanations if top_k == -1 else phrase_explanations[: top_k + 1]
        part1_phrases = [phrase_expl.part1 for phrase_expl in phrase_explanations]
        part2_phrases = [phrase_expl.part2 for phrase_expl in phrase_explanations]
        part1_replaced_tokens, input_ids = self.replace_input_ids_text_phrase(
            text=orig.part1_tokens,
            input_ids=orig.input_ids,
            phrases=part1_phrases,
            token_map=orig.part1_map,
            replace_token=self.replace_token,
            replace_token_id=self.replace_token_id,
        )
        part2_replaced_tokens, input_ids = self.replace_input_ids_text_phrase(
            text=orig.part2_tokens,
            input_ids=input_ids,
            phrases=part2_phrases,
            token_map=orig.part2_map,
            replace_token=self.replace_token,
            replace_token_id=self.replace_token_id,
        )
        count = lambda _l: len([x for x in _l if x.text == self.replace_token])  # noqa: E731
        num_replaced_tokens = count(part1_replaced_tokens) + count(part2_replaced_tokens)
        orig = HFInstance(
            input_ids=input_ids,
            token_type_ids=orig.token_type_ids,
            attention_mask=orig.attention_mask,
            part1_map=orig.part1_map,
            part2_map=orig.part2_map,
            part1_tokens=part1_replaced_tokens,
            part2_tokens=part2_replaced_tokens,
            part1=" ".join([x.text for x in part1_replaced_tokens]),
            part2=" ".join([x.text for x in part2_replaced_tokens]),
            idx=orig.idx,
            label_index=orig.label_index,
        )
        return [(num_replaced_tokens, orig)]

    def run(self, orig: HFInstance, instance_explanation: InstanceExplanation, **kwargs) -> List[Tuple[int, HFInstance]]:
        return self.perturb_by_phrase(orig, instance_explanation.expl_phrases, **kwargs)


@register(_name="comp-hf-phrase", _type=PERTURBATION_METHOD)
class CompHFPerturberPhrase(HFPerturberPhrase):
    """
    The explanations are of type phrase. Perturb the original instance with multiple explanations and return 1 perturbed
    instance. This is required for computing AUC comprehensiveness scores.
    The perturbed instance consists of original tokens - tokens from the explanations (the explanation tokens are replaced with
    the MASK token)
    """

    def __init__(self, handler: Handler, **kwargs):
        super().__init__(handler, **kwargs)

    def replace_input_ids_text_phrase(
        self,
        text: List[Token],
        input_ids: torch.Tensor,
        phrases: List[Phrase],
        token_map: TokenMap,
        replace_token: str,
        replace_token_id: int,
    ) -> Tuple[List[Token], torch.Tensor]:
        _input_ids = deepcopy(input_ids)
        _text = [x.text for x in text]
        for phrase in phrases:
            for token_index in range(phrase.start_token_index, phrase.end_token_index + 1):
                _text[token_index] = replace_token
                to_be_replaced_ids = token_map[token_index]
                for _index in to_be_replaced_ids:
                    _input_ids[_index] = replace_token_id
        _tokens = []
        _start_char_index = 0
        for t in _text:
            _tokens.append(Token(text=t, start_char_index=_start_char_index))
            _start_char_index = _start_char_index + len(t) + 1  # adjust for space
        return _tokens, _input_ids


@register(_name="suff-hf-phrase", _type=PERTURBATION_METHOD)
class SuffHFPerturberPhrase(HFPerturberPhrase):
    """
    The explanations are of type phrase. Perturb the original instance with multiple explanations and return 1 perturbed
    instance. This is required for computing AUC sufficiency scores.
    The perturbed instance consists only of the tokens from the explanations (the rest is replaced by the MASK token)
    The replacement is a bit tricky. For eg:, consider the following scenario:
    part1_tokens: ['One', 'tan', 'girl', 'with', 'a', 'wool', 'hat', 'is', 'running', 'and', 'leaning', 'over', 'an', 'object', ',', 'while', 'another', 'person', 'in', 'a', 'wool', 'hat', 'is', 'sitting', 'on', 'the', 'ground', '.']
    part2_tokens: ['A', 'boy', 'runs', 'into', 'a', 'wall']
    phrase pair explanations: [('One tan girl', 'A boy'), ('girl', 'boy'), ('leaning', 'runs'), ('running', 'runs')]
    we can not iteratively (iterating over the explanations) mask out the non explanation tokens, because after the first iteration,
    part1_tokens have everything masked out except the first three tokens. When the second explanation comes in, tokens "One", and "tan" gets masked as well.
    So we have to be a bit careful. after the first iteration, unmask.
    """

    def __init__(self, handler: Handler, **kwargs):
        super().__init__(handler, **kwargs)

    @staticmethod
    def merge_overlapping(boundaries: List[Tuple[int, int]]):
        """
        [(1, 3), (2, 4), (7, 8), (6,9), (9, 10), (15, 17)] -> [(1, 4), (6, 10), (15, 17)]
        :param boundaries:
        :return:
        """

        def merge(_current: Tuple[int, int], _next: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            """
            :param _current:
            :param _next:
            :return:
            """
            if _next[0] > _current[1] + 1:
                return None
            else:
                return min(_current[0], _next[0]), max(_current[1], _next[1])

        boundaries = sorted(boundaries, key=lambda x: x[0])
        current = boundaries[0]
        merged_all = []
        for next_ in boundaries[1:]:
            result = merge(current, next_)
            if result is not None:
                current = result
            else:
                merged_all.append(current)
                current = next_
        merged_all.append(current)
        return merged_all

    @staticmethod
    def within(_input: int, boundaries: List[Tuple[int, int]]):
        for boundary in boundaries:
            if boundary[0] <= _input <= boundary[1]:
                return True
        return False

    def replace_input_ids_text_phrase(
        self,
        text: List[Token],
        input_ids: torch.Tensor,
        phrases: List[Phrase],
        token_map: TokenMap,
        replace_token: str,
        replace_token_id: int,
    ) -> Tuple[List[Token], torch.Tensor]:
        _input_ids = deepcopy(input_ids)
        _text = [x.text for x in text]
        do_not_mask_boundaries = self.merge_overlapping(
            [(phrase.start_token_index, phrase.end_token_index) for phrase in phrases]
        )
        for token_index, _ in enumerate(text):
            if not self.within(token_index, do_not_mask_boundaries):
                _text[token_index] = replace_token
                to_be_replaced_ids = token_map[token_index]
                for _index in to_be_replaced_ids:
                    _input_ids[_index] = replace_token_id
        _tokens = []
        _start_char_index = 0
        for t in _text:
            _tokens.append(Token(text=t, start_char_index=_start_char_index))
            _start_char_index = _start_char_index + len(t) + 1  # adjust for space
        return _tokens, _input_ids
