# flake8: noqa
from typing import Dict, List, Union
from explain_interactions.datamodels import Instance, TxTokenizedInstance, InstanceOutput
from explain_interactions.dataloaders import EIDataLoader
from explain_interactions.tokenizers import Tokenizer
import torch


class Handler:
    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: EIDataLoader, id_2_label: Dict[int, str], **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.id_2_label = id_2_label
        self.head = kwargs.get("head", None)
        self.layer = kwargs.get("layer", None)

    def load(self, **kwargs):
        """
        load from configs
        :param kwargs:
        :return:
        """
        pass

    def predict(self, instances: Union[List[Instance], List[TxTokenizedInstance]], **kwargs) -> List[InstanceOutput]:
        """
        Subclasses should override.
        :param instances:
        :param kwargs:
        :return:
        """


from explain_interactions.handlers.hf import (
    HFHandler,
    HeadLayerAttention,
    ClfWeightHeadImportance,
    RandomHeadImportance,
)
