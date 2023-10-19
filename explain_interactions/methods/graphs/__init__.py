import json
from dataclasses import dataclass
from networkx import Graph
from typing import List
from tqdm import tqdm
from networkx.readwrite import adjacency_data, adjacency_graph


@dataclass
class InstanceGraph:
    instance_idx: str
    graph: Graph
    metadata: str


@dataclass
class InstanceGraphs:
    """
    a bunch of instance graphs
    """

    data: List[InstanceGraph]

    def save(self, loc: str):
        """
        serialize to a jsonl file
        :param loc:
        :return:
        """
        print(f"saving graphs to {loc}")
        with open(loc, "w") as wf:
            for datum in tqdm(self.data):
                d = {"instance_idx": datum.instance_idx, "metadata": datum.metadata, "graph": adjacency_data(datum.graph)}
                wf.write(json.dumps(d) + "\n")

    @classmethod
    def load(cls, loc: str):
        """
        :param loc:
        :return:
        """
        print(f"loading graphs from {loc}")
        _cls = cls([])
        for line in open(loc):
            datum = json.loads(line)
            _cls.data.append(
                InstanceGraph(
                    instance_idx=datum["instance_idx"], metadata=datum["metadata"], graph=adjacency_graph(datum["graph"])
                )
            )
        return _cls


from .explainers import LouvainCommunitiesExplainer  # noqa
