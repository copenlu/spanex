from explain_interactions.datamodels import HFInstance
from explain_interactions.dataloaders import EIDataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers.tokenization_utils_base import PaddingStrategy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from explain_interactions.registry import register, DATA_LOADER


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    This is modified from transformers code. See the definitions there.
    """

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [
                {"input_ids": x["input_ids"], "token_type_ids": x["token_type_ids"], "attention_mask": x["attention_mask"]}
                for x in features
            ],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
        )
        first = features[0]
        for k, v in first.items():
            if k in ["input_ids", "attention_mask", "token_type_ids"]:
                continue
            elif isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif not isinstance(v, str):
                batch[k] = torch.tensor([f[k] for f in features])
            else:
                batch[k] = [f[k] for f in features]
        return batch


class HfEIDataLoader(EIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.max_length = kwargs.get("max_length")
        self.sampler = SequentialSampler

    def __call__(self, instances: List[HFInstance], *args, **kwargs) -> DataLoader:
        data = [
            {
                "attention_mask": _instance.attention_mask,
                "input_ids": _instance.input_ids,
                "token_type_ids": _instance.token_type_ids,
                "idx": _instance.idx,
                "y": _instance.label_index,
            }
            for _instance in instances
        ]
        return DataLoader(
            dataset=data,
            sampler=self.sampler(data),
            batch_size=self.batchsz,
            collate_fn=DataCollatorWithPadding(self.tokenizer, padding="longest", max_length=self.tokenizer.model_max_length),
        )


@register(_name="bert-small-snli", _type=DATA_LOADER)
class PrajjwalBertSmall(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")


@register(_name="bert-base-uncased-snli", _type=DATA_LOADER)
class BertBaseUnCasedSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-base-uncased")


@register(_name="bert-base-cased-snli", _type=DATA_LOADER)
class BertBaseCasedSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-base-cased")


@register(_name="bert-large-uncased-snli", _type=DATA_LOADER)
class BertLargeUnCasedSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-large-uncased")


@register(_name="bert-large-cased-snli", _type=DATA_LOADER)
class BertLargeCasedSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_bert-large-cased")


@register(_name="roberta-small-snli", _type=DATA_LOADER)
class RobertaSmallSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("pepa/roberta-small-snli")


@register(_name="roberta-base-snli", _type=DATA_LOADER)
class RobertaBaseSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("pepa/roberta-base-snli")


@register(_name="roberta-large-snli", _type=DATA_LOADER)
class RobertaLargeSNLI(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("pepa/roberta-large-snli")


@register(_name="roberta-base-fever", _type=DATA_LOADER)
class RobertaBaseFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/roberta-base-fever")


@register(_name="roberta-large-fever", _type=DATA_LOADER)
class RobertaLargeFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/roberta-large-fever")


@register(_name="bert-base-cased-fever", _type=DATA_LOADER)
class BertBaseCasedFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-base-cased-fever")


@register(_name="bert-base-uncased-fever", _type=DATA_LOADER)
class BertBaseUnCasedFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-base-uncased-fever")


@register(_name="bert-large-cased-fever", _type=DATA_LOADER)
class BertLargeCasedFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-large-cased-fever")


@register(_name="bert-large-uncased-fever", _type=DATA_LOADER)
class BertLargeUnCasedFever(HfEIDataLoader):
    def __init__(self, batchsz, **kwargs):
        super().__init__(batchsz, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("sagnikrayc/bert-large-uncased-fever")
