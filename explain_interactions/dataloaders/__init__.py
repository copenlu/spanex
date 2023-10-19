# flake8: noqa
"""
Takes a list of tokenized instances and creates an iterator containing batches that a model can consume
"""
from explain_interactions.datamodels import TxTokenizedInstance
from torch.utils.data import DataLoader
from typing import List


class EIDataLoader:
    """
    base class for explain_interaction dataloader.
    """

    def __init__(self, batchsz: int, **kwargs):
        self.batchsz = batchsz

    def __call__(self, instances: List[TxTokenizedInstance], *args, **kwargs) -> DataLoader:
        pass


from .hf import (
    PrajjwalBertSmall,
    BertBaseUnCasedSNLI,
    BertBaseCasedSNLI,
    BertLargeUnCasedSNLI,
    BertLargeCasedSNLI,
    RobertaSmallSNLI,
    RobertaBaseSNLI,
    RobertaLargeSNLI,
    BertBaseUnCasedFever,
    BertBaseCasedFever,
    BertLargeUnCasedFever,
    BertLargeCasedFever,
    RobertaBaseFever,
    RobertaLargeFever,
)
