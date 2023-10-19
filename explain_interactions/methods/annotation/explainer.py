"""
An explainer takes a list of instances and produces a list(list of explanations) for it. See __init__.py
"""
from typing import Dict, List, Tuple
from explain_interactions.methods import Explainer
from explain_interactions.methods.annotation.read_annotation_output import AnnotationDataProcessor, AnnotationDatum
from explain_interactions.datamodels import (
    Instance,
    InstanceExplanation,
    TokenizedInstance,
)
from explain_interactions.methods.expl_scorers import Scorer
from explain_interactions.handlers import Handler
from explain_interactions.registry import register, EXPL_METHOD
import serpyco
import json
from tqdm import tqdm


@register(_type=EXPL_METHOD, _name="annotation")
class AnnotationExplainer(Explainer):
    def __init__(self, handler: Handler, scorer: Scorer, expl_file: str, labels: Dict, **kwargs):
        super().__init__(handler=handler, scorer=scorer, **kwargs)
        ann_data_processor = AnnotationDataProcessor(self.handler.tokenizer)
        input_ser = serpyco.Serializer(AnnotationDatum)
        input_data = [input_ser.load(json.loads(x)) for x in tqdm(open(expl_file), desc="reading annotation data")]
        output_data: List[Tuple[TokenizedInstance, InstanceExplanation]] = [
            ann_data_processor.process_to_instance_explanation(datum, labels=labels)
            for datum in tqdm(input_data, desc="processing annotation data")
        ]
        self.all_annotation_explanations: Dict[str, InstanceExplanation] = {
            x.instance_idx: x for x in self.scorer.run(output_data)
        }

    def run(self, instances: List[Instance], *args, **kwargs) -> List[InstanceExplanation]:
        super().run(instances)
        return [self.all_annotation_explanations[instance.idx] for instance in self.tokenized_instances]
