# flake8: noqa
from typing import Any, List, Union
from explain_interactions.datamodels import Instance


class Dataset:
    def load(self, *args, **kwargs) -> Union[List[Instance], List[Any]]:
        """
        subclasses must implement this
        :param args:
        :param kwargs:
        :return:
        """


from .hf_datasets import SnliHF, ESnliHF, MNliHF, MNliHFBP, FeverCopenluHF
from .local import HFInstanceLocalDataset, FromAnnotation
