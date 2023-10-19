from typing import Dict, List, Optional, Union
from explain_interactions.datamodels import (
    InstanceExplanation,
    Token,
    TokenWithIndex,
    Phrase,
    TokenizedInstance,
    ExplanationPhrase,
    Instance,
)
import random
from explain_interactions.handlers import Handler
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.registry import EXPL_METHOD, register
from explain_interactions.methods import Explainer
from tqdm import tqdm
import numpy as np

DEFAULT_NUM_EXPLS = 3
NUM_PHRASE_PAIRS = "num_phrase_pairs"
NUM_PHRASES_HYP = "num_phrases_hypothesis"
NUM_PHRASES_PREM = "num_phrases_premise"
SNLI = "snli"
FEVER = "fever"


def token_inside_expl_phrase(index_: int, part: str, existing_explanation: InstanceExplanation) -> bool:
    if part == "part1":
        for phrase in existing_explanation.expl_phrases:
            if phrase.part1.start_token_index <= index_ <= phrase.part1.end_token_index:
                return True
        return False
    elif part == "part2":
        for phrase in existing_explanation.expl_phrases:
            if phrase.part2.start_token_index <= index_ <= phrase.part2.end_token_index:
                return True
        return False
    else:
        raise RuntimeError("part must be part 1 or 2")


def sample_from_discrete_dist(dataset_name, sampling_field) -> int:
    dists = {
        SNLI: {
            NUM_PHRASE_PAIRS: {"probs": [0.57, 0.3, 0.1, 0.03], "vals": [1, 2, 3, 4]},
            NUM_PHRASES_HYP: {
                "probs": [0.67, 0.16, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02],
                "vals": [1, 2, 3, 4, 5, 6, 7, 8],
            },
            NUM_PHRASES_PREM: {"probs": [0.81, 0.06, 0.05, 0.04, 0.01, 0.01, 0.02], "vals": [1, 2, 3, 4, 5, 6, 7]},
        },
        FEVER: {
            NUM_PHRASE_PAIRS: {"probs": [0.84, 0.14, 0.01, 0.01], "vals": [1, 2, 3, 4]},
            NUM_PHRASES_HYP: {
                "probs": [0.84, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01],
                "vals": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            },
            NUM_PHRASES_PREM: {"probs": [0.86, 0.07, 0.02, 0.015, 0.015, 0.02], "vals": [1, 2, 3, 4, 5, 6]},
        },
    }
    elements = dists[dataset_name][sampling_field]["vals"]
    probabilities = dists[dataset_name][sampling_field]["probs"]
    return np.random.choice(elements, 1, p=probabilities)[0]


def get_unused_tokens(part_tokens: List[Token], part: str, existing_explanation: InstanceExplanation) -> List[TokenWithIndex]:
    return [
        TokenWithIndex(text=token.text, start_char_index=token.start_char_index, start_token_index=index)
        for index, token in enumerate(part_tokens)
        if not token_inside_expl_phrase(index, part, existing_explanation)
    ]


def get_all_tokens(part_tokens: List[Token]) -> List[TokenWithIndex]:
    return [
        TokenWithIndex(text=token.text, start_char_index=token.start_char_index, start_token_index=index)
        for index, token in enumerate(part_tokens)
    ]


def select_phrases(_phrases: List[Phrase], _phrase_lengths: List[int]) -> List[Phrase]:
    """
    For random explanations, we want to select phrases that have lengths similar to the original phrases.
    I am skipping this for now, because this seems more complicated than I had imagined.
    :param _phrases:
    :param _phrase_lengths:
    :return:
    """
    return _phrases
    # if len(_phrases) <= len(_phrase_lengths):
    #     return _phrases
    # selected_phrases = []
    # _phrase_lengths_ =  set(_phrase_lengths)
    # _phrases_ = deepcopy(_phrases)
    # for _phrase in _phrases:
    #     if (_phrase.end_token_index - _phrase.start_token_index) in _phrase_lengths_:
    #         _phrase_lengths_.remove((_phrase.end_token_index - _phrase.start_token_index))
    #         _phrases_.remove(_phrase)
    #         selected_phrases.append(_phrase)
    # if len(selected_phrases) < len(_phrase_lengths):
    #     selected_phrases = selected_phrases + _phrases_[:(len(_phrase_lengths) - len(selected_phrases))]
    # return selected_phrases


def create_phrases(_tokens: List[TokenWithIndex], phrase_lengths: List[int] = [], max_phrase_length: int = 3) -> List[Phrase]:
    phrases = []
    current_token_index = _tokens[0].start_token_index
    current_phrase = [_tokens[0]]
    for _token in _tokens:
        if _token.start_token_index == current_token_index + 1 and len(current_phrase) < max_phrase_length:
            current_phrase.append(_token)
            current_token_index = _token.start_token_index
        else:
            phrases.append(
                Phrase(
                    text=" ".join([x.text for x in current_phrase]),
                    start_token_index=current_phrase[0].start_token_index,
                    end_token_index=current_phrase[-1].start_token_index,
                )
            )
            current_token_index = _token.start_token_index
            current_phrase = [_token]
    phrases.append(
        Phrase(
            text=" ".join([x.text for x in current_phrase]),
            start_token_index=current_phrase[0].start_token_index,
            end_token_index=current_phrase[-1].start_token_index,
        )
    )
    phrases = select_phrases(_phrases=phrases, _phrase_lengths=phrase_lengths)
    return phrases


def create_explanation_phrases(
    phrases_part_1: List[Phrase], phrases_part_2: List[Phrase], instance_idx: str, default_score: float = 0.5
):
    expl_phrases = []
    for index_part_1, phrase_part_1 in enumerate(phrases_part_1):
        for index_part_2, phrase_part_2 in enumerate(phrases_part_2):
            expl_phrases.append(
                ExplanationPhrase(
                    instance_idx=f"{instance_idx}-{index_part_1}-{index_part_2}",
                    part1=phrase_part_1,
                    part2=phrase_part_2,
                    score=default_score,
                )
            )
    return expl_phrases


class AlternateExplainer(Explainer):
    """
    For an instance, we have an explanation (possibly from manual annotations), generate an alternate explanation.
    """

    def __init__(self, handler: Handler, scorer: Scorer, **kwargs):
        super().__init__(handler=handler, scorer=scorer, **kwargs)

    def generate_alternate_explanations(
        self, instance: TokenizedInstance, existing_explanation: Optional[InstanceExplanation] = None, **kwargs
    ) -> InstanceExplanation:
        pass

    def run(self, instances: Union[List[Instance], List[TokenizedInstance]], **kwargs) -> List[InstanceExplanation]:
        """
        each instance must have an InstanceExplanations already defined.
        :param instances:
        :param kwargs:
        :return:
        """
        super().run(instances)
        instance_expls_orig: Dict[str, InstanceExplanation] = {
            expl.instance_idx: expl for expl in kwargs.get("instance_explanations", [])
        }
        instance_expls_alternate = (
            {
                instance.idx: self.generate_alternate_explanations(
                    instance=instance, existing_explanation=instance_expls_orig[instance.idx]
                )
                for instance in tqdm(self.tokenized_instances, desc=f"generating explanations {self.__class__}")
            }
            if instance_expls_orig
            else {
                instance.idx: self.generate_alternate_explanations(instance=instance)
                for instance in tqdm(self.tokenized_instances, desc=f"generating explanations {self.__class__}")
            }
        )
        # score the random explanations before returning
        return self.scorer.run(
            instance_explanations=[
                (instance, instance_expls_alternate[instance.idx])
                for instance in tqdm(self.tokenized_instances, desc=f"scoring explanations, {self.scorer}")
            ]
        )


@register(_type=EXPL_METHOD, _name="random-phrase")
class RandomExplainerPhrase(AlternateExplainer):
    """
    Choose some phrases randomly from first part, some phrases randomly from the second part.
    """

    def __init__(self, handler: Handler, scorer: Scorer, **kwargs):
        super().__init__(handler=handler, scorer=scorer, **kwargs)
        self.expl_from_unused_phrases = kwargs.get("expl_from_unused_phrases", False)
        self.dataset_name = kwargs["dataset_name"]
        assert self.dataset_name in [SNLI, FEVER], f"Sampling stats available only for {SNLI} and {FEVER}"

    def generate_alternate_explanations(
        self, instance: TokenizedInstance, existing_explanation: Optional[InstanceExplanation] = None, **kwargs
    ) -> InstanceExplanation:
        """
        :param instance:
        :param existing_explanation:
        :return:
        """
        if existing_explanation is not None:
            expl_part_1_phrase_lengths = [
                (x.part1.end_token_index - x.part1.start_token_index) for x in existing_explanation.expl_phrases
            ]
            expl_part_2_phrase_lengths = [
                (x.part2.end_token_index - x.part2.start_token_index) for x in existing_explanation.expl_phrases
            ]
        else:
            expl_part_1_phrase_lengths, expl_part_2_phrase_lengths = [], []
        if self.expl_from_unused_phrases:
            assert existing_explanation is not None
            tokens_to_generate_expl_from_part1 = get_unused_tokens(
                instance.part1_tokens, part="part1", existing_explanation=existing_explanation
            )
            tokens_to_generate_expl_from_part2 = get_unused_tokens(
                instance.part2_tokens, part="part2", existing_explanation=existing_explanation
            )
        else:
            tokens_to_generate_expl_from_part1 = get_all_tokens(instance.part1_tokens)
            tokens_to_generate_expl_from_part2 = get_all_tokens(instance.part2_tokens)

        # create some phrases out of these tokens.
        phrases_part_1 = create_phrases(
            _tokens=tokens_to_generate_expl_from_part1,
            phrase_lengths=expl_part_1_phrase_lengths,
            max_phrase_length=sample_from_discrete_dist(self.dataset_name, sampling_field=NUM_PHRASES_PREM),
        )
        phrases_part_2 = create_phrases(
            _tokens=tokens_to_generate_expl_from_part2,
            phrase_lengths=expl_part_2_phrase_lengths,
            max_phrase_length=sample_from_discrete_dist(self.dataset_name, sampling_field=NUM_PHRASES_HYP),
        )
        random.shuffle(phrases_part_1)
        random.shuffle(phrases_part_2)
        # we need to produce these many phrase explanations
        num_phrase_pairs = (
            len(existing_explanation.expl_phrases)
            if existing_explanation is not None
            else sample_from_discrete_dist(dataset_name=self.dataset_name, sampling_field=NUM_PHRASE_PAIRS)
        )
        instance_idx = instance.idx
        expl_phrases = create_explanation_phrases(
            phrases_part_1=phrases_part_1, phrases_part_2=phrases_part_2, instance_idx=instance_idx
        )
        random.shuffle(expl_phrases)

        return InstanceExplanation(instance_idx=instance_idx, expl_tokens=[], expl_phrases=expl_phrases[:num_phrase_pairs])


@register(_type=EXPL_METHOD, _name="part-phrase")
class PartPhrase(AlternateExplainer):
    """
    This will generate phrase explanations where for each explanation, part i (i=1 or 2) is from an annotation explanation and
    part j (j=1 or 2) is generated from the part j (premise/hypothesis) tokens.
    """

    def __init__(self, handler: Handler, scorer: Scorer, part: str, **kwargs):
        super().__init__(handler=handler, scorer=scorer, **kwargs)
        self.part = part
        assert self.part in ["part1", "part2"]
        self.dataset_name = kwargs["dataset_name"]
        assert self.dataset_name in [SNLI, FEVER], f"Sampling stats available only for {SNLI} and {FEVER}"

    def generate_alternate_explanations(
        self, instance: TokenizedInstance, existing_explanation: Optional[InstanceExplanation] = None, **kwargs
    ) -> InstanceExplanation:
        # we need to produce these many phrase explanations
        # num_phrase_pairs = (
        #     len(existing_explanation.expl_phrases)
        #     if existing_explanation is not None
        #     else sample_from_discrete_dist(dataset_name=self.dataset_name, sampling_field=NUM_PHRASE_PAIRS)
        # )
        # the auc comprehensive and sufficiency scores are dependent on the explanation lengths, both on the
        # number of phrase pairs and the number of tokens in a phrase pair. Since we don't have a concept of "top-k"
        # here, we need to make sure that the number of phrase pairs reflect the actual distribution. Remember, in
        # the annotation explanations, we will slice the explanations along the relation label, which is why we can not
        # simply use the number of annotation phrase pairs here.
        num_phrase_pairs = sample_from_discrete_dist(dataset_name=self.dataset_name, sampling_field=NUM_PHRASE_PAIRS)
        instance_idx = instance.idx
        tokens_to_generate_expl_from = (
            get_all_tokens(instance.part2_tokens) if self.part == "part1" else get_all_tokens(instance.part1_tokens)
        )
        if self.part == "part1":
            expl_phrase_lengths = (
                [(x.part2.end_token_index - x.part2.start_token_index) for x in existing_explanation.expl_phrases]
                if existing_explanation is not None
                else []
            )
            existing_phrases = [
                x.part1 for x in existing_explanation.expl_phrases if x.annotator == "0" and "SYSTEM" not in x.relation
            ]
        else:
            expl_phrase_lengths = (
                [(x.part1.end_token_index - x.part1.start_token_index) for x in existing_explanation.expl_phrases]
                if existing_explanation is not None
                else []
            )
            existing_phrases = [
                x.part2 for x in existing_explanation.expl_phrases if x.annotator == "0" and "SYSTEM" not in x.relation
            ]
        sampling_field = NUM_PHRASES_PREM if self.part == "part1" else NUM_PHRASES_HYP
        new_phrases = create_phrases(
            _tokens=tokens_to_generate_expl_from,
            phrase_lengths=expl_phrase_lengths,
            max_phrase_length=sample_from_discrete_dist(dataset_name=self.dataset_name, sampling_field=sampling_field),
        )
        phrases_part_1 = existing_phrases if self.part == "part1" else new_phrases
        phrases_part_2 = existing_phrases if self.part == "part2" else new_phrases
        expl_phrases = create_explanation_phrases(
            phrases_part_1=phrases_part_1, phrases_part_2=phrases_part_2, instance_idx=instance_idx
        )
        random.shuffle(expl_phrases)
        return InstanceExplanation(instance_idx=instance_idx, expl_tokens=[], expl_phrases=expl_phrases[:num_phrase_pairs])
