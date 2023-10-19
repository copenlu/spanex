# flake8: noqa
from typing import List
import torch.nn
from transformers.tokenization_utils_base import BatchEncoding
from explain_interactions.datamodels import InstanceOutput


class Model(torch.nn.Module):
    def predict(self, batch: BatchEncoding) -> List[InstanceOutput]:
        pass


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


from explain_interactions.models.hf import (
    PrajjwalBertSmallSNLILocal,
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

from explain_interactions.models.scalarmix_for_attnhead import BertSmallSNLILocalScalarMix
