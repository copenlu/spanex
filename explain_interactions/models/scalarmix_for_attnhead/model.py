from typing import Union
from explain_interactions.models.scalarmix_for_attnhead.scalarmix import ScalarMix
from explain_interactions.registry import MODEL_REGISTRY
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch


class ClassifierWithAttnHeadDistribution(nn.Module):
    def __init__(self, orig_model: Union[BertForSequenceClassification, RobertaForSequenceClassification], num_attn_heads):
        super().__init__()
        self.orig_model = orig_model
        if type(self.orig_model) == BertForSequenceClassification:
            self.intermediate = self.orig_model.bert.pooler.dense
            self.orig_model_clf = self.orig_model.classifier
        elif type(self.orig_model) == RobertaForSequenceClassification:
            self.intermediate = self.orig_model.classifier.dense
            self.orig_model_clf = self.orig_model.classifier.out_proj
        else:
            raise RuntimeError("only bert and roberta model is supported")
        for param in self.orig_model.parameters():
            param.requires_grad = False
        self.num_labels = self.orig_model.num_labels
        self.num_attn_heads = num_attn_heads
        self.mixer = ScalarMix(mixture_size=self.num_attn_heads)
        assert not self.orig_model_clf.in_features % num_attn_heads
        self.hidden_dim = self.orig_model_clf.in_features // num_attn_heads
        self.clf = nn.Linear(in_features=self.hidden_dim, out_features=self.orig_model_clf.out_features)

    def get_most_important_attn_head(self, head_importance):
        """
        head importance is supposed to be a number between 0 and num_attn_heads - 1, 0 being the most important.
        also can pass -1 to get the least important head
        :param head_importance:
        :return:
        """
        scalar_mix_weights = [x.item() for x in self.mixer.scalar_parameters]
        _, top_indices = torch.topk(torch.tensor(scalar_mix_weights), k=self.num_attn_heads)
        return top_indices[head_importance].item()

    def forward(
        self, input_ids, attention_mask, labels=None, token_type_ids=None, output_attentions=False, output_hidden_states=False
    ):
        prev_model_output = self.orig_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        prev_model_cls = prev_model_output.hidden_states[-1][:, 0, :]
        prev_model_cls = self.intermediate(prev_model_cls)
        prev_model_cls = prev_model_cls.split(self.hidden_dim, dim=1)
        prev_model_scalar_mixed = self.mixer(prev_model_cls)
        logits = self.clf(prev_model_scalar_mixed)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        attentions = prev_model_output.attentions if output_attentions else None
        hidden_states = prev_model_output.hidden_states if output_hidden_states else None
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)

    @classmethod
    def load(cls, model_dir: str, base_model_name: str, num_attn_heads: int):
        state_dict = torch.load(f"{model_dir}/pytorch_model.bin")
        orig_model = MODEL_REGISTRY[base_model_name].load()
        c = cls(orig_model, num_attn_heads)
        c.mixer = ScalarMix(
            mixture_size=c.num_attn_heads,
            initial_scalar_parameters=[state_dict[f"mixer.scalar_parameters.{i}"] for i in range(c.num_attn_heads)],
        )
        c.clf.weight = nn.Parameter(state_dict["clf.weight"])
        c.clf.bias = nn.Parameter(state_dict["clf.bias"])
        return c
