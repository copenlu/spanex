"""
run as python read_annotation_output.py --input ../../../data/snli_agreement.jsonl --output ../../../data/snli_agreement_processed.jsonl
"""
from typing import Dict, List, Tuple
from dataclasses import dataclass
from explain_interactions.datamodels import (
    InstanceExplanation,
    ExplanationPhrase,
    Phrase,
    TokenizedInstance,
    Token,
    TokenWithIndex,
)
from explain_interactions.tokenizers import Tokenizer
from itertools import chain

DEFAULT_SCORE = 1.0
EXCLUDE_RELATIONS = ["151166_35_38_14_17_low", "151166_38_41_17_20_low", "151166_107_110_14_17_low", "151166_110_113_17_20_low"]


@dataclass
class AnnotationRelation:
    labels: List[str]
    annotators: List[int]
    level: str
    premise_text: str
    hypothesis_text: str
    start_premise: int
    start_hypothesis: int
    end_premise: int
    end_hypothesis: int
    id: str


@dataclass
class AnnotationDatum:
    id: str
    label: str
    hypothesis: str
    premise: str
    relations: List[AnnotationRelation]
    original_id: str


@dataclass
class ProcessedRelation:
    level: str
    label: str
    annotator: int
    premise: List[TokenWithIndex]
    hypothesis: List[TokenWithIndex]
    id: str


@dataclass
class ProcessedDatum:
    id: str
    label: str
    premise_tokens: List[TokenWithIndex]
    hypothesis_tokens: List[TokenWithIndex]
    relations: List[ProcessedRelation]


class AnnotationDataProcessor:
    def __init__(self, tokenizer: Tokenizer, **kwargs):
        self.tokenizer = tokenizer  # this tokenizer must have a method called tokenize_str that produces str -> TokenWithIndex

    @staticmethod
    def inside(token: Token, start: int, end: int):
        return token.start_char_index >= start and token.start_char_index + len(token.text) <= end

    def process_relation(
        self, relation: AnnotationRelation, premise_tokens: List[TokenWithIndex], hyp_tokens: List[TokenWithIndex]
    ) -> List[ProcessedRelation]:
        def modify_label(label_: str) -> str:
            if label_.startswith("Synonym-SYSTEM"):
                return "Synonym"
            return label_

        if relation.id in EXCLUDE_RELATIONS:
            print(f"excluding relation {relation.id}")
            return []
        rel_premise_tokens = [
            x for x in premise_tokens if self.inside(x, start=relation.start_premise, end=relation.end_premise)
        ]
        rel_hyp_tokens = [x for x in hyp_tokens if self.inside(x, start=relation.start_hypothesis, end=relation.end_hypothesis)]
        if not rel_premise_tokens:
            raise RuntimeError(f"for relation {relation.id}, premise tokens could not be found")
        if not rel_hyp_tokens:
            raise RuntimeError(f"for relation {relation.id}, hypothesis tokens could not be found")
        return [
            ProcessedRelation(
                label=modify_label(_label),
                level=relation.level,
                annotator=_ann,
                premise=rel_premise_tokens,
                hypothesis=rel_hyp_tokens,
                id=relation.id,
            )
            for _label, _ann in zip(relation.labels, relation.annotators)
        ]

    def process(self, annotation_datum: AnnotationDatum) -> ProcessedDatum:
        premise_tokens = self.tokenizer.tokenize_str(annotation_datum.premise)
        hypothesis_tokens = self.tokenizer.tokenize_str(annotation_datum.hypothesis)
        processed_relations = list(
            chain(
                *[
                    self.process_relation(relation, premise_tokens=premise_tokens, hyp_tokens=hypothesis_tokens)
                    for relation in annotation_datum.relations
                ]
            )
        )
        return ProcessedDatum(
            id=annotation_datum.id,
            label=annotation_datum.label,
            premise_tokens=premise_tokens,
            hypothesis_tokens=hypothesis_tokens,
            relations=processed_relations,
        )

    @staticmethod
    def get_phrase(_tokens: List[TokenWithIndex]) -> Phrase:
        return Phrase(
            text=" ".join([x.text for x in _tokens]),
            start_token_index=_tokens[0].start_token_index,
            end_token_index=_tokens[-1].start_token_index,
        )

    def process_to_instance_explanation(
        self, annotation_datum: AnnotationDatum, labels: Dict[str, int]
    ) -> Tuple[TokenizedInstance, InstanceExplanation]:
        processed_datum = self.process(annotation_datum)
        tokenized_instance = TokenizedInstance(
            part1=annotation_datum.premise,
            part2=annotation_datum.hypothesis,
            idx=annotation_datum.id,
            part1_tokens=processed_datum.premise_tokens,
            part2_tokens=processed_datum.hypothesis_tokens,
            label_index=labels[annotation_datum.label],
        )
        explanations = InstanceExplanation(
            instance_idx=processed_datum.id,
            expl_tokens=[],
            expl_phrases=[
                ExplanationPhrase(
                    instance_idx=annotation_datum.id,
                    part1=self.get_phrase(relation.premise),
                    part2=self.get_phrase(relation.hypothesis),
                    score=DEFAULT_SCORE,
                    level=relation.level,
                    relation=relation.label,
                    annotator=str(relation.annotator),
                )
                for relation in processed_datum.relations
            ],
        )
        return tokenized_instance, explanations
