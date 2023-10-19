# flake8: noqa
from typing import List
from explain_interactions.datamodels import Instance, TokenizedInstance, TxTokenizedInstance, TokenWithIndex


class Tokenizer:
    def __init__(self, **kwargs):
        pass

    def tokenize_str(self, _input: str) -> List[TokenWithIndex]:
        pass

    def tokenize(self, instances: List[Instance]) -> List[TokenizedInstance]:
        pass

    def tokenize_tx(self, instances: List[Instance]) -> List[TxTokenizedInstance]:
        pass


from explain_interactions.tokenizers.hf_spacy import (
    PrajjwalBertSmall,
    RobertaBaseSNLI,
    RobertaLargeSNLI,
    RobertaSmallSNLI,
    BertBaseUnCasedSNLI,
    BertBaseCasedSNLI,
    BertLargeCasedSNLI,
    BertLargeUnCasedSNLI,
    BertBaseUnCasedFever,
    BertBaseCasedFever,
    BertLargeUnCasedFever,
    BertLargeCasedFever,
    RobertaBaseFever,
    RobertaLargeFever,
)
