from typing import List, Tuple, Union

# import torch
from transformers import AutoTokenizer
import spacy
from tqdm import tqdm
from explain_interactions.datamodels import Instance, HFInstance, TokenMap, TokenWithIndex, TokenizedInstance
from explain_interactions.tokenizers import Tokenizer
from explain_interactions.registry import register, TOKENIZER

ROBERTA_TOKENIZER = "RoBertaTokenizer"
BERT_TOKENIZER = "BertTokenizer"


def spacy_hf_mapping(spacy_tokens, hf_offsets, hf_offsets_pointer) -> Tuple[int, TokenMap]:
    """
    Warning: don't use this, I haven't tested thoroughly.
    Find the mapping between spacy tokens and hf tokens.
    text: 'donald trump', spacy_tokens: ['donald', 'trump']. hf_tokens: ['don', '#ald', 'trump']. map = [[0, 1], 2]
    Use this method if the hf tokenization was done with is_split_into_words = True.
    Returns the map and the index to which the hf_offsets array has been scanned.
    :param spacy_tokens:
    :param hf_offsets:
    :param hf_offsets_pointer:
    :return:
    """
    spacy_hf_map = [[] for _ in range(len(spacy_tokens))]
    for hf_index in range(hf_offsets_pointer, len(hf_offsets)):
        hf_offset = hf_offsets[hf_index]
        last_hf_index = hf_index
        if hf_offset[0] == 0 and hf_offset[1] == 0:  # we have reached the SEP token
            return last_hf_index, spacy_hf_map
        for spacy_index, spt in enumerate(spacy_tokens):
            if hf_offset[0] >= spt.start and hf_offset[1] <= spt.end:
                spacy_hf_map[spacy_index].append(hf_index)


def spacy_hf_mapping_is_split_into_words(
    hf_offsets, hf_offsets_pointer, tokenizer_type, tokenizer_max_length=512
) -> Tuple[int, TokenMap]:
    """
    Find the mapping between spacy tokens and hf tokens.
    text: 'donald trump', spacy_tokens: ['donald', 'trump']. hf_tokens: ['don', '#ald', 'trump']. map = [[0, 1], 2]
    Use this method if the hf tokenization was done with is_split_into_words = True.
    Returns the map and the index to which the hf_offsets array has been scanned.
    :param hf_offsets:
    :param hf_offsets_pointer:
    :param tokenizer_max_length:
    :param tokenizer_type
    :return:
    """

    def has_hit_sep(_hf_offset, _next_hf_offset):
        if tokenizer_type == BERT_TOKENIZER:
            return _hf_offset[0] + _hf_offset[1] == 0
        elif tokenizer_type == ROBERTA_TOKENIZER:
            return _hf_offset[0] + _hf_offset[1] + _next_hf_offset[0] + _next_hf_offset[1] == 0

    spacy_hf_map = []
    current_token_map = []
    # hf_offsets look like:
    # [(0, 0), (0, 5), (0, 8), (0, 3),  (3, 7), (0, 0), (0, 1), (0, 9),   (9, 12),  (12, 15), (0, 16), (0, 0), (0, 0)]
    # [SEP,    tok1,   tok2,   tok3#p1, tok3#p2, SEP,   tok4,    tok5#p1,  tok5#p2,  tok5#p3,  tok4,   SEP, ...]
    hf_index = hf_offsets_pointer
    while hf_index < len(hf_offsets) - 1:
        hf_offset = hf_offsets[hf_index]
        # if hf_offset[0] == 0 and hf_offset[1] == 0:  # we have hit a sep token
        #     return hf_index, spacy_hf_map
        next_hf_offset = hf_offsets[hf_index + 1]
        if has_hit_sep(_hf_offset=hf_offset, _next_hf_offset=next_hf_offset):
            return hf_index, spacy_hf_map
        if next_hf_offset[0] == 0:  # the next offset is **not** the part of this token
            current_token_map.append(hf_index)
            spacy_hf_map.append(current_token_map)
            current_token_map = []
        else:
            current_token_map.append(hf_index)
        hf_index += 1
    # we have already seen the penultimate hf_offset. Now, the last token must be (0, 0), i.e., a SEP token. if not,
    # raise an error
    hf_index = hf_index + 1 if hf_index < len(hf_offsets) - 1 else hf_index
    if hf_index > tokenizer_max_length:
        raise RuntimeError("the data should not have token lengths > model max length")
    elif hf_offsets[hf_index][0] != 0 or hf_offsets[hf_index][1] != 0:
        raise RuntimeError("reached the end of the seq w/o seeing the SEP token")
    elif current_token_map:  # add the last non-empty current token map to the spacy_hf_map
        spacy_hf_map.append(current_token_map)
    return hf_index, spacy_hf_map


class ShortHFTokenizer(Tokenizer):
    """
    Handling long sequences by truncating part 2 (IOW, there's no stride).
    """

    @staticmethod
    def set_up_parser():
        _nlp = spacy.load("en_core_web_md")
        config = {"punct_chars": None}
        _nlp.add_pipe("sentencizer", config=config)
        return _nlp

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nlp = self.set_up_parser()
        self.hf_tokenizer_type = kwargs.get("hf_tokenizer_type", BERT_TOKENIZER)
        self.call_params = {
            "truncation": "only_second",
            "return_offsets_mapping": True,
            "is_split_into_words": True,
        }

    def tokenize_str(self, _input: str) -> List[TokenWithIndex]:
        return [
            TokenWithIndex(text=x.text, start_char_index=x.idx, start_token_index=idx) for idx, x in enumerate(self.nlp(_input))
        ]

    def tokenize(self, instances: List[Instance]) -> List[TokenizedInstance]:
        def _tokenize(_instance: Instance) -> TokenizedInstance:
            return TokenizedInstance(
                part1=_instance.part1,
                part2=_instance.part2,
                idx=_instance.idx,
                label_index=_instance.label_index,
                part1_tokens=self.tokenize_str(_instance.part1),
                part2_tokens=self.tokenize_str(_instance.part2),
            )

        return [_tokenize(_instance=instance) for instance in instances]

    def tokenize_tx(self, instances: Union[List[Instance], List[TokenizedInstance]]) -> List[HFInstance]:
        tokenized_instances: List[TokenizedInstance] = (
            instances if hasattr(instances[0], "part1_tokens") else self.tokenize(instances)
        )
        be = self.hf_tokenizer(
            [[token.text for token in tokenized_instance.part1_tokens] for tokenized_instance in tokenized_instances],
            [[token.text for token in tokenized_instance.part2_tokens] for tokenized_instance in tokenized_instances],
            **self.call_params
        )
        assert len(instances) == len(be.offset_mapping)
        hf_data = []
        if self.hf_tokenizer_type == ROBERTA_TOKENIZER:
            token_type_ids = []
            for input_ids in be.input_ids:
                token_type_ids.append([0] * len(input_ids))
        else:
            token_type_ids = be.token_type_ids
        for (tokenized_instance, hf_offsets, input_ids, token_type_ids, attention_mask,) in tqdm(
            zip(
                tokenized_instances,
                be.offset_mapping,
                be.input_ids,
                token_type_ids,
                be.attention_mask,
            ),
            desc="Converting dataset to HFInstances",
        ):
            if len(input_ids) > self.hf_tokenizer.model_max_length:
                print("can not use an input that is longer than model max length")
                continue
            last_hf_index, part1_map = spacy_hf_mapping_is_split_into_words(
                hf_offsets, hf_offsets_pointer=1, tokenizer_type=self.hf_tokenizer_type
            )
            # the first token is CLS
            if self.hf_tokenizer_type == ROBERTA_TOKENIZER:
                last_hf_index += 1  # roberta has an extra sep token
            _, part2_map = spacy_hf_mapping_is_split_into_words(
                hf_offsets, hf_offsets_pointer=last_hf_index + 1, tokenizer_type=self.hf_tokenizer_type
            )
            # We ended at a SEP token
            hf_data.append(
                HFInstance(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    part1_map=part1_map,
                    part2_map=part2_map,
                    **tokenized_instance.__dict__
                )
            )
        return hf_data


@register(_type=TOKENIZER, _name="bert-small-snli")
class PrajjwalBertSmall(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-small-snli"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        self.hf_tokenizer.model_max_length = 128
        self.vocab = self.hf_tokenizer.vocab


@register(_type=TOKENIZER, _name="bert-base-uncased-snli")
class BertBaseUnCasedSNLI(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-base-uncased-snli"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-base-uncased")
        self.vocab = self.hf_tokenizer.vocab


@register(_type=TOKENIZER, _name="bert-base-cased-snli")
class BertBaseCasedSNLI(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-base-cased-snli"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-base-cased")
        self.vocab = self.hf_tokenizer.vocab


@register(_type=TOKENIZER, _name="bert-large-uncased-snli")
class BertLargeUnCasedSNLI(BertBaseUnCasedSNLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-large-uncased-snli"


@register(_type=TOKENIZER, _name="bert-large-cased-snli")
class BertLargeCasedSNLI(BertBaseCasedSNLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-large-cased-snli"


@register(_type=TOKENIZER, _name="roberta-base-snli")
class RobertaBaseSNLI(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, hf_tokenizer_type=ROBERTA_TOKENIZER)
        self.name = "roberta-base-snli"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("pepa/roberta-base-snli", add_prefix_space=True)
        self.vocab = self.hf_tokenizer.vocab


@register(_type=TOKENIZER, _name="roberta-small-snli")
class RobertaSmallSNLI(RobertaBaseSNLI):
    """
    TODO: The roberta-small model might have been trained with a different tokenizer than Roberta.
    ```
    In [84]: t = AutoTokenizer.from_pretrained("pepa/roberta-base-snli")

    In [85]: len(t.vocab)
    Out[85]: 50265

    In [86]: t = AutoTokenizer.from_pretrained("pepa/roberta-small-snli")

    In [87]: len(t.vocab)
    Out[87]: 32000

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "roberta-small-snli"


@register(_type=TOKENIZER, _name="roberta-large-snli")
class RobertaLargeSNLI(RobertaBaseSNLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "roberta-large-snli"


@register(_type=TOKENIZER, _name="bert-base-uncased-fever")
class BertBaseUnCasedFever(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-base-uncased-fever"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-base-uncased-fever")
        self.vocab = self.hf_tokenizer.vocab
        self.call_params.pop("truncation")


@register(_type=TOKENIZER, _name="bert-base-cased-fever")
class BertBaseCasedFever(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-base-cased-fever"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-base-cased-fever")
        self.vocab = self.hf_tokenizer.vocab
        self.call_params.pop("truncation")


@register(_type=TOKENIZER, _name="bert-large-uncased-fever")
class BertLargeUnCasedFever(BertBaseUnCasedFever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-large-uncased-fever"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-large-uncased-fever")


@register(_type=TOKENIZER, _name="bert-large-cased-fever")
class BertLargeCasedFever(BertBaseCasedFever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "bert-large-cased-fever"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-large-cased-fever")


@register(_type=TOKENIZER, _name="roberta-base-fever")
class RobertaBaseFever(ShortHFTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, hf_tokenizer_type=ROBERTA_TOKENIZER)
        self.name = "roberta-base-fever"
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/roberta-base-fever", add_prefix_space=True)
        self.vocab = self.hf_tokenizer.vocab
        self.call_params.pop("truncation")


@register(_type=TOKENIZER, _name="roberta-large-fever")
class RobertaLargeFever(RobertaBaseFever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hf_tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/roberta-large-fever", add_prefix_space=True)
        self.name = "roberta-large-fever"
