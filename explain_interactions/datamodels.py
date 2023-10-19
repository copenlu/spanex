from dataclasses import dataclass
from typing import List, Optional, NewType
import torch

TokenMap = NewType("TokenMap", List[List[int]])

"""
inputs
"""


@dataclass
class Token:
    text: str
    start_char_index: int


@dataclass
class TokenWithIndex(Token):
    start_token_index: int  # position of this token in a sentence


@dataclass
class Phrase:
    text: str
    start_token_index: int
    end_token_index: int


@dataclass
class Instance:
    part1: str
    part2: str
    idx: str
    label_index: int


@dataclass
class TokenizedInstance(Instance):
    part1_tokens: List[Token]
    part2_tokens: List[Token]


@dataclass
class TxTokenizedInstance(TokenizedInstance):
    part1_map: TokenMap  # part1_text: ['donald', 'trump']. tokenization: ['don', '#ald', 'trump'].
    # part_1_map = [[0, 1], 2]
    part2_map: TokenMap


@dataclass
class HFInstance(TxTokenizedInstance):
    """
    Subclass of tokenized transformer instance for HF models.
    """

    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    # roberta doesn't have this, so we will pass in a dummy tensor of shape 1 x len(input_ids)
    attention_mask: torch.Tensor


"""
explanations
"""


@dataclass
class ExplanationBase:
    instance_idx: str
    score: float


@dataclass
class ExplanationToken(ExplanationBase):
    part1: Token  # tokens from the first part
    part2: Token  # tokens from the second part
    level: str = "NA"  # when from annotation, we should know the level, i.e. whether it's upper/lower level (premise/pleaf)
    label: str = "NA"  # when from annotation, the label, i.e., antonym, synonym, hypernym-p-2-h etc.
    annotator: str = "NA"  # when from annotation, the annotator id


@dataclass
class ExplanationPhrase(ExplanationBase):
    part1: Phrase  # phrase from the first part
    part2: Phrase  # phrase from the second part
    level: str = "NA"  # when from annotation, we should know the level, i.e. whether it's upper/lower level (premise/pleaf)
    relation: str = "NA"  # when from annotation, the relation, i.e., antonym, synonym, hypernym-p-2-h etc.
    annotator: str = "NA"  # when from annotation, the annotator id


@dataclass
class InstanceExplanation:
    """
    Holds all explanations for an instance, both token level and phrase level.
    """

    instance_idx: str
    expl_tokens: List[ExplanationToken]
    expl_phrases: List[ExplanationPhrase]


@dataclass
class InstanceTextExplanation:
    instance_idx: str
    label_index: int
    part1: str
    part2: str
    expl_phrases: List[ExplanationPhrase]


@dataclass
class ComparativeExplanation:
    """
    a dataclass to make the comparison evaluation easier. There is a base explanation: which is supposed to be something like
    explanations derived from manual annotations/ some algorithmic process and an alternate explanation: which is supposed to be
    something like explanations created randomly.
    """

    instance_idx: str
    base: InstanceExplanation
    alternate: InstanceExplanation


"""
outputs
"""


@dataclass
class InstanceOutput:
    """
    This is very similar to SequenceClassifierOutput from HF, except the batch axis is removed.
    """

    instance_idx: str
    pred_class_label: Optional[str]
    pred_class_index: int
    logits: torch.FloatTensor  # (num_classes, 1)
    probs: torch.FloatTensor
    hidden_states: Optional[List[torch.FloatTensor]]
    # List of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape
    # (sequence_length, hidden_size). If there are 12 layers, the List contains 13 such tensors.
    att_weights: Optional[List[torch.FloatTensor]]
    # Tuple of torch.FloatTensor (one for each layer) of shape (num_heads, sequence_length,
    # sequence_length). If there are 12 layers, the List contains 12 such tensors.


@dataclass
class InstanceOutputLayerHead:
    """
    InstanceOutput for a specific layer and head: we will not output the hidden states, and we will output the attention
    weights for a specific layer and head.
    """

    instance_idx: str
    pred_class_label: Optional[str]
    pred_class_index: int
    logits: torch.FloatTensor  # (num_classes, 1)
    probs: torch.FloatTensor
    att_weights: torch.FloatTensor  # (sequence_length, sequence_length)
