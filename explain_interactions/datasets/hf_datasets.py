from typing import List
from datasets import load_dataset
from explain_interactions.datasets import Dataset
from explain_interactions.datamodels import Instance
from explain_interactions.registry import register, DATASET
from collections import defaultdict
import random

SEED = 30


class NliHF(Dataset):
    def load(self, phase, *args, **kwargs) -> List[Instance]:
        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(self.data[phase]))
        randomize = kwargs.get("randomize", False)
        instances = [
            Instance(
                part1=datum["premise"],
                part2=datum["hypothesis"],
                idx=str(f"{self.name}-{phase}-{index}"),
                label_index=int(datum["label"]),
            )
            for index, datum in enumerate(list(self.data[phase]))
        ]
        if kwargs.get("id_2_label") is not None:
            instances = [x for x in instances if x.label_index in kwargs["id_2_label"]]
        if randomize:
            random.Random(SEED).shuffle(instances)
        return instances[start:end]


@register(_type=DATASET, _name="mnli-hfd")
class MNliHF(NliHF):
    def __init__(self, **kwargs):
        self.name = "mnli"
        self.data = load_dataset("glue", "mnli")


@register(_type=DATASET, _name="mnli-hfd-bp")
class MNliHFBP(NliHF):
    """
    The binary parse version of MNLI data
    """

    def __init__(self, **kwargs):
        self.name = "mnli-bp"
        self.data = load_dataset("multi_nli")

    def load(self, phase, *args, **kwargs) -> List[Instance]:
        num_instances = kwargs.get("num_instances", len(self.data[phase]))
        randomize = kwargs.get("randomize", False)
        reject_genres = kwargs.get("reject_genres", ["facetoface", "telephone"])
        bp_or_normal = (
            ("premise_binary_parse", "hypothesis_binary_parse")
            if kwargs.get("use_binary_parse", False)
            else ("premise", "hypothesis")
        )
        prompt_blocks = defaultdict(list)
        for datum in list(self.data[phase]):
            if datum["genre"] not in reject_genres:
                prompt_blocks[datum["promptID"]].append(
                    Instance(
                        part1=datum[bp_or_normal[0]],
                        part2=datum[bp_or_normal[1]],
                        idx=str(f"{self.name}-{datum['pairID']}"),
                        label_index=int(datum["label"]),
                    )
                )
        for v in prompt_blocks.values():
            v.sort(key=lambda x: x.idx)
        prompt_blocks = list(prompt_blocks.values())
        if randomize:
            random.Random(SEED).shuffle(prompt_blocks)
        instances = []
        index = 0
        for prompt_block in prompt_blocks:
            for instance in prompt_block:
                if index == num_instances:
                    break
                instances.append(instance)
                index += 1
        return instances[:num_instances]


@register(_type=DATASET, _name="snli-hfd")
class SnliHF(NliHF):
    def __init__(self, **kwargs):
        self.name = "snli"
        self.data = load_dataset("snli")


@register(_type=DATASET, _name="esnli-hfd")
class ESnliHF(NliHF):
    def __init__(self, **kwargs):
        self.name = "esnli"
        self.data = load_dataset("esnli")


@register(_type=DATASET, _name="fever-copenlu-hf")
class FeverCopenluHF:
    def __init__(self, **kwargs):
        self.name = "fever-copenlu"
        self.data = load_dataset("copenlu/fever_gold_evidence")
        self.data = self.data.map(self.make_evidence_text).filter(
            lambda x: len(x["claim"].split()) + len(x["evidence_text"].split()) < 420
        )
        self.label_2_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

    @staticmethod
    def make_evidence_text(example):
        example["evidence_text"] = " ".join([" ".join(sent[0].split("_")) + " " + sent[-1] for sent in example["evidence"]])
        return example

    def load(self, phase, *args, **kwargs) -> List[Instance]:
        start = kwargs.get("start", 0)
        end = kwargs.get("end", len(self.data[phase]))
        randomize = kwargs.get("randomize", False)
        instances = [
            Instance(
                part1=datum["evidence_text"],
                part2=datum["claim"],
                idx=datum["id"],
                label_index=self.label_2_id[datum["label"]],
            )
            for index, datum in enumerate(list(self.data[phase]))
        ]
        if kwargs.get("id_2_label") is not None:
            instances = [x for x in instances if x.label_index in kwargs["id_2_label"]]
        if randomize:
            random_seed = kwargs.get("random_seed", SEED)
            random.Random(random_seed).shuffle(instances)
        return instances[start:end]
