"""
Take a model and data instances and extract multiple graphs out of it.
"""
from networkx import Graph, DiGraph
from explain_interactions.methods.graphs import InstanceGraph, InstanceGraphs
from tqdm import tqdm
from typing import List, Union
from explain_interactions.datamodels import TxTokenizedInstance, InstanceOutput, InstanceOutputLayerHead
from explain_interactions.handlers import Handler
from explain_interactions.registry import register, GRAPH_EXTRACTOR


class GraphExtractor:
    """
    Extract InstanceGraph for a dataset
    """

    def __init__(self, handler: Handler, *args, **kwargs):
        self.handler = handler

    def run(self, *args, **kwargs) -> InstanceGraphs:
        pass


@register(_type=GRAPH_EXTRACTOR, _name="attention-single-layer-single-head")
class AttentionGraphExtractor(GraphExtractor):
    def __init__(self, handler, **kwargs):
        super().__init__(handler)
        layer = kwargs.get("layer")
        head = kwargs.get("head")
        self.layer = layer
        self.head = head

    def convert_instance_to_graph(
        self, instance: TxTokenizedInstance, instance_output: Union[InstanceOutput, InstanceOutputLayerHead], **kwargs
    ) -> Graph:
        """
        convert a [seq_len x seq_len] torch tensor to a graph
        :param instance:
        :param instance_output:
        :return:
        """
        g = DiGraph()
        data = (
            instance_output.att_weights[self.layer][self.head].numpy().astype(float)
            if type(instance_output) == InstanceOutput
            else instance_output.att_weights.numpy().astype(float)
        )
        # convert to numpy here, will be helpful to serialize
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                g.add_edge(i, j, weight=data[i, j])
                g.add_edge(j, i, weight=data[j, i])
        return g

    def run(self, instances: List[TxTokenizedInstance], *args, **kwargs) -> InstanceGraphs:
        output_instances = self.handler.predict(instances)
        self.layer = self.layer if self.layer != -1 else len(output_instances[0].att_weights) - 1
        self.head = self.head if self.head != -1 else len(output_instances[0].att_weights[0]) - 1
        results = []
        for instance, output_instance in tqdm(zip(instances, output_instances), desc="Extracting graphs"):
            results.append(
                InstanceGraph(
                    instance_idx=output_instance.instance_idx,
                    graph=self.convert_instance_to_graph(instance=instance, instance_output=output_instance),
                    metadata=f"attn-layer-model-{kwargs.get('model', 'UNK')}-{self.layer}-head-{self.head}",
                )
            )
        return InstanceGraphs(data=results)


@register(_type=GRAPH_EXTRACTOR, _name="attention-bipartite-single-layer-single-head")
class AttentionBiPartiteGraphExtractor(AttentionGraphExtractor):
    def __init__(self, handler, **kwargs):
        super().__init__(handler, **kwargs)

    def convert_instance_to_graph(
        self, instance: TxTokenizedInstance, instance_output: Union[InstanceOutput, InstanceOutputLayerHead], **kwargs
    ) -> Graph:
        """
        convert a [seq_len x seq_len] torch tensor to a graph
        :param instance
        :param instance_output:
        :return:
        """
        g = DiGraph()
        g.adjacency()
        data = (
            instance_output.att_weights[self.layer][self.head].numpy().astype(float)
            if type(instance_output) == InstanceOutput
            else instance_output.att_weights.numpy().astype(float)
        )
        assert data.shape[0] == data.shape[1]  # data should be a symmetric matrix
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                g.add_edge(i, j, weight=0.0)
                g.add_edge(j, i, weight=0.0)
        part1_start = instance.part1_map[0][0]
        part1_end = instance.part1_map[-1][-1]
        part2_start = instance.part2_map[0][0]
        part2_end = instance.part2_map[-1][-1]
        for i in range(part1_start, part1_end):
            for j in range(part2_start, part2_end):
                g.remove_edge(i, j)
                g.remove_edge(j, i)
                g.add_edge(i, j, weight=data[i, j])
                g.add_edge(j, i, weight=data[j, i])
        return g
        # ^ at this point, g is essentially a bipartite graph, but has the same number of nodes as
        # #([CLS][PART1 HF TOKENS][SEP][PART2 HF TOKENS][SEP]), with edges between parts of internal nodes being 0
