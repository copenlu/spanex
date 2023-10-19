"""
return instances from saved pt files.
"""
import json
import random
from typing import List
import torch
from tqdm import tqdm
from explain_interactions.datasets import Dataset
from explain_interactions.registry import register, DATASET
from explain_interactions.datamodels import Instance, HFInstance

SEED = 25


@register(_type=DATASET, _name="hf-instance-local")
class HFInstanceLocalDataset(Dataset):
    def __init__(self, file_loc: str, **kwargs):
        self.file_loc = file_loc

    def load(self, *args, **kwargs) -> List[HFInstance]:
        return torch.load(self.file_loc)


@register(_type=DATASET, _name="from-annotation")
class FromAnnotation(Dataset):
    """
    read the SNLI/fever data from the annotation file. we also know that the lengths are less than 512.
    """

    def __init__(self, file_loc: str, phase: str, **kwargs):
        self.data = {phase: []}
        self.label_2_id = kwargs["label_2_id"]
        for line in tqdm(open(file_loc)):
            _line = json.loads(line)
            self.data[phase].append(
                Instance(
                    part1=_line["premise"],
                    part2=_line["hypothesis"],
                    idx=_line["id"],
                    label_index=self.label_2_id[_line["label"]],
                )
            )

    def load(self, phase, *args, **kwargs) -> List[Instance]:
        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(self.data[phase]))
        num_points_needed = kwargs.get("num_points_needed", -1)
        randomize = kwargs.get("randomize", False)
        _data = [x for x in self.data[phase][start:end]]
        if num_points_needed != -1 and len(_data) > num_points_needed:
            _data = _data[:num_points_needed]
        if randomize:
            random.Random(SEED).shuffle(_data)
        return _data


@register(_type=DATASET, _name="fever-local")
class FeverDatasetLocal(Dataset):
    @staticmethod
    def _replace(_input, replacements):
        for k, v in replacements.items():
            _input = _input.replace(k, v)
        return _input

    def __init__(self, file_loc: str, phase: str, **kwargs):
        label_2_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        replacements = {"-LRB-": "(", "-LSB-": "(", "-RRB-": ")", "-RSB-": ")"}
        self.data = {phase: []}
        for line in tqdm(open(file_loc)):
            _d = json.loads(line)
            self.data[phase].append(
                Instance(
                    part2=self._replace(_d["claim"], replacements),
                    part1=self._replace(
                        " ".join([f"[source: {x[0]}] {x[2]}" for x in _d["evidence"]]),
                        replacements,
                    ),
                    idx=_d["original_id"],
                    label_index=label_2_id[_d["label"]],
                )
            )

    def load(self, phase, *args, **kwargs) -> List[Instance]:
        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(self.data[phase]))
        num_points_needed = kwargs.get("num_points_needed", -1)
        if "existing_annotation_ids_file" in kwargs:
            existing_annotation_ids = [
                x for x in open(kwargs["existing_annotation_ids_file"]) if x
            ]  # see <project home>/data/datasets/README.md
        else:
            existing_annotation_ids = []
        randomize = kwargs.get("randomize", False)
        _data = [x for x in self.data[phase][start:end] if x.idx not in existing_annotation_ids]
        if num_points_needed != -1 and len(_data) > num_points_needed:
            _data = _data[:num_points_needed]
        if randomize:
            random.Random(SEED).shuffle(_data)
        return _data
