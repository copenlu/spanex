import torch
from tqdm import tqdm  # noqa
from typing import Dict, List
from explain_interactions.models import get_device
from explain_interactions.handlers import Handler
from explain_interactions.datamodels import (
    Instance,
    TokenizedInstance,
    TxTokenizedInstance,
    HFInstance,
    InstanceOutput,
    InstanceOutputLayerHead,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from explain_interactions.tokenizers import Tokenizer
from explain_interactions.dataloaders import EIDataLoader
from explain_interactions.registry import register, HANDLER, TOKENIZER_REGISTRY, MODEL_REGISTRY, DATA_LOADER_REGISTRY
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification


@register(_type=HANDLER, _name="hf")
class HFHandler(Handler):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        data_loader: EIDataLoader,
        id_2_label: Dict[int, str],
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_loader=data_loader,
            id_2_label=id_2_label,
            **kwargs,
        )

    @classmethod
    def load(cls, **kwargs):
        tokenizer_params = kwargs["tokenizer_params"]
        tokenizer = TOKENIZER_REGISTRY[tokenizer_params["name"]](**tokenizer_params)
        model_params = kwargs["model_params"]
        model_name = model_params.pop("name")
        model = MODEL_REGISTRY[model_name].load(**model_params)
        data_loader_params = kwargs["data_loader_params"]
        data_loader = DATA_LOADER_REGISTRY[data_loader_params["name"]](**data_loader_params)
        id_2_label = kwargs.pop("id_2_label")
        return cls(model=model, tokenizer=tokenizer, data_loader=data_loader, id_2_label=id_2_label, **kwargs)

    def tokenize_if_needed(self, instances: List) -> List[TxTokenizedInstance]:
        assert type(instances[0]) in [Instance, TokenizedInstance, TxTokenizedInstance, HFInstance]
        if type(instances[0]) == Instance or type(instances[0]) == TokenizedInstance:
            return self.tokenizer.tokenize_tx(instances=instances)
        else:
            return instances

    def _convert_classifier_output_large_batch(
        self, batch_output: SequenceClassifierOutput, batch_ids: List[str], _
    ) -> List[InstanceOutput]:
        """
        Sometimes we just want to predict using a model and do not care about the attention weights. We should
        be able to use large batches then. In this case, we should use this function in predict
        :param batch_output:
        :param batch_ids:
        :param batch_attn_mask:
        :return:
        """
        bo_logits = batch_output.logits
        assert len(batch_ids) == bo_logits.shape[0]
        return [
            InstanceOutput(
                instance_idx=instance_id,
                logits=instance_logit.detach().cpu(),
                probs=torch.softmax(instance_logit, dim=0).detach().cpu(),
                hidden_states=None,
                att_weights=None,
                pred_class_index=torch.argmax(instance_logit).item(),
                pred_class_label=self.id_2_label[torch.argmax(instance_logit).item()],
            )
            for instance_id, instance_logit in zip(batch_ids, bo_logits)
        ]

    def _convert_classifier_output(
        self, batch_output: SequenceClassifierOutput, batch_ids: List[str], batch_attn_mask
    ) -> List[InstanceOutput]:
        def adjust_length_hidden_states(_instance_hidden_states: List[torch.FloatTensor], attn_mask):
            return [x[: torch.count_nonzero(attn_mask)] for x in _instance_hidden_states]

        def adjust_length_attn_weights(_instance_attn_weights: List[torch.FloatTensor], attn_mask):
            return [x[:, : torch.count_nonzero(attn_mask), : torch.count_nonzero((attn_mask))] for x in _instance_attn_weights]

        bo_logits = batch_output.logits
        bo_hidden_states = torch.stack(batch_output.hidden_states, dim=1)
        bo_attn_weights = torch.stack(batch_output.attentions, dim=1)
        assert len(batch_ids) == bo_logits.shape[0] == bo_hidden_states.shape[0] == bo_attn_weights.shape[0]
        return [
            InstanceOutput(
                instance_idx=instance_id,
                logits=instance_logit.detach().cpu(),
                probs=torch.softmax(instance_logit, dim=0).detach().cpu(),
                hidden_states=adjust_length_hidden_states(instance_hidden_state.detach().cpu(), attn_mask),
                att_weights=adjust_length_attn_weights(instance_att_weights.detach().cpu(), attn_mask),
                pred_class_index=torch.argmax(instance_logit).item(),
                pred_class_label=self.id_2_label[torch.argmax(instance_logit).item()],
            )
            for instance_id, instance_logit, instance_hidden_state, instance_att_weights, attn_mask in zip(
                batch_ids, bo_logits, bo_hidden_states, bo_attn_weights, batch_attn_mask
            )
        ]

    def predict(self, instances: List[Instance], *args, **kwargs) -> List[InstanceOutput]:
        """
        The input is either a list of instances, or tokenized datum, which for the purposes of this class is HFInstance.
        If the input is not of type instance, tokenize it first.
        Else,
        :param instances:
        :param args:
        :param kwargs:
        :return:
        """
        if not instances:
            return []
        instances = self.tokenize_if_needed(instances)
        all_outputs = []
        # for batch in self.data_loader(instances):
        for batch in tqdm(
            self.data_loader(instances),
            desc=f"predicting with batch size {self.data_loader.batchsz}",
        ):
            _batch = {k: v.to(get_device()) for k, v in batch.items() if k not in ["idx", "y"]}
            batch_ids = batch["idx"]
            with torch.no_grad():
                batch_output: SequenceClassifierOutput = self.model(
                    **_batch, output_hidden_states=False, output_attentions=False
                )
                all_outputs.extend(
                    self._convert_classifier_output_large_batch(batch_output, batch_ids, _batch["attention_mask"])
                )
        return all_outputs


@register(_type=HANDLER, _name="hf-head-layer")
class HeadLayerAttention(HFHandler):
    """
    This is a special case of HFHandler, where we only want to return the hidden states and attention weights
    for a particular layer and head.
    """

    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: EIDataLoader, id_2_label: Dict[int, str], **kwargs
    ):
        super().__init__(model, tokenizer, data_loader, id_2_label, **kwargs)
        self.head_importance = kwargs.get("head_importance", 0)  # if we want the most important head, set this to 0,
        # the least important head is -1
        print(f"producing results for {self.head_importance}th most important head")
        assert self.layer is not None, f"layer must be specified for the handler {self.__class__}"

    def _convert_classifier_output(
        self, batch_output: SequenceClassifierOutput, batch_ids: List[str], batch_attn_mask, **kwargs
    ) -> List[InstanceOutputLayerHead]:
        """
        :param batch_output:
        :param batch_ids:
        :param batch_attn_mask:
        :return:
        """

        def adjust_length_attn_weights(_instance_attn_weights, attn_mask):
            return _instance_attn_weights[: torch.count_nonzero(attn_mask), : torch.count_nonzero((attn_mask))]

        bo_logits = batch_output.logits
        _bo_attn_weights = batch_output.attentions[self.layer]
        # we can specify the head in two ways: for all instances in the batch, we can use one head, or for each
        # instance we can specify a different head.
        batch_heads = kwargs["batch_heads"]
        bo_attn_weights = (
            _bo_attn_weights[torch.arange(_bo_attn_weights.size(0)), batch_heads].detach().cpu()
        )  # see https://stackoverflow.com/a/67951672/1555818
        assert len(batch_ids) == bo_logits.shape[0] == bo_attn_weights.shape[0]
        return [
            InstanceOutputLayerHead(
                instance_idx=instance_id,
                logits=instance_logit.detach().cpu(),
                probs=torch.softmax(instance_logit, dim=0).detach().cpu(),
                att_weights=adjust_length_attn_weights(instance_att_weights, attn_mask),
                pred_class_index=torch.argmax(instance_logit).item(),
                pred_class_label=self.id_2_label[torch.argmax(instance_logit).item()],
            )
            for instance_id, instance_logit, instance_att_weights, attn_mask in zip(
                batch_ids, bo_logits, bo_attn_weights, batch_attn_mask
            )
        ]

    def get_heads(self, batch_output: SequenceClassifierOutput) -> torch.Tensor:
        """
        :returns a tensor of shape (batch_size, ) containing the head to use for each instance in the batch.
        :param batch_output:
        :return:
        """
        assert self.head is not None, f"head must be specified for the handler {self.__class__}"
        return torch.tensor([self.head] * batch_output.logits.shape[0])

    def predict(self, instances: List[Instance], *args, **kwargs) -> List[InstanceOutputLayerHead]:
        """
        :param instances:
        :param args:
        :param kwargs:
        :return:
        """
        if not instances:
            return []
        all_outputs = []
        instances = self.tokenize_if_needed(instances)
        for batch in self.data_loader(instances):
            _batch = {k: v.to(get_device()) for k, v in batch.items() if k not in ["idx", "y"]}
            batch_ids = batch["idx"]
            with torch.no_grad():
                batch_output: SequenceClassifierOutput = self.model(**_batch, output_hidden_states=True, output_attentions=True)
                all_outputs.extend(
                    self._convert_classifier_output(
                        batch_output, batch_ids, _batch["attention_mask"], batch_heads=self.get_heads(batch_output)
                    )
                )
        return all_outputs


@register(_type=HANDLER, _name="hf-head-layer-random-head")
class RandomHeadImportance(HeadLayerAttention):
    """
    Generate a random head for each instance in the batch.
    """

    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: EIDataLoader, id_2_label: Dict[int, str], **kwargs
    ):
        super().__init__(model, tokenizer, data_loader, id_2_label, **kwargs)

    def get_heads(self, batch_output: SequenceClassifierOutput):
        batch_size, max_head, seq_len, seq_len = batch_output.attentions[self.layer].shape
        heads = torch.randint(high=max_head, size=(1, batch_size)).squeeze()
        return heads


@register(_type=HANDLER, _name="hf-head-layer-head-importance")
class ClfWeightHeadImportance(HeadLayerAttention):
    """
    Get the most important head for each instance by looking at the classifier on top of CLS.
    """

    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: EIDataLoader, id_2_label: Dict[int, str], **kwargs
    ):
        super().__init__(model, tokenizer, data_loader, id_2_label, **kwargs)
        self.cls_clf_mat = self.get_cls_classifier_matrix().T
        print(f"classification matrix shape: {self.cls_clf_mat.shape}")

    def get_cls_classifier_matrix(self) -> torch.Tensor:
        """
        Get the classifier matrix for the CLS token.
        :return:
        """
        model = self.model
        if isinstance(model, BertForSequenceClassification):
            print("using bert")
            return torch.matmul(model.bert.pooler.dense.weight, (model.classifier.weight.T + model.classifier.bias))
        elif isinstance(model, RobertaForSequenceClassification):
            print("using roberta")
            assert model.roberta.pooler is None, "pooler inside a roberta model is not supported"
            return torch.matmul(
                (model.classifier.dense.weight + model.classifier.dense.bias),
                (model.classifier.out_proj.weight.T + model.classifier.out_proj.bias),
            )
        else:
            raise NotImplementedError(f"model {model} not supported")

    def get_heads(self, batch_output: SequenceClassifierOutput):
        """
        What is the predicted class for this instance?
        :param batch_output:
        :return:
        """
        bo_logits = batch_output.logits
        batch_size, num_heads, _, _ = batch_output.attentions[self.layer].shape
        pred_class_indices = torch.argmax(bo_logits, dim=1)
        most_important_heads = torch.zeros(batch_size, dtype=torch.long)
        for instance_index, pred_class_index in enumerate(pred_class_indices):
            instance_cls_clf_vec = self.cls_clf_mat[pred_class_index]
            batch_output_hidden_states = batch_output.hidden_states[self.layer]
            cls_token_repr = batch_output_hidden_states[instance_index, 0, :]
            assert (
                cls_token_repr.shape == instance_cls_clf_vec.shape
            ), f"something is wrong, check the shapes: {cls_token_repr.shape} vs {instance_cls_clf_vec.shape}"
            # let's say the cls token is [-5, -4, 3, 2, 1] and the classifier vector is [-2, 1, -7, 5, -1]
            # if cls[i] < 0, clf[i] < 0 means a positive contribution, and clf[i] > 0 means a negative contribution.
            # so, before we look at the total contribution over a sequence, we need to make sure that the signs are
            # consistent.
            instance_cls_clf_vec = instance_cls_clf_vec * torch.where(cls_token_repr > 0, 1, -1)
            # now, let's look at the contribution for each head from the classifier weights.
            assert (
                cls_token_repr.shape[0] % num_heads == 0
            ), "something is wrong, cls token repr should be divisible by num heads"
            head_size = cls_token_repr.shape[0] // num_heads
            split_instance_cls_clf_vec = instance_cls_clf_vec.reshape(num_heads, head_size)
            _, top_indices = torch.topk(torch.sum(split_instance_cls_clf_vec, dim=1), k=num_heads)
            most_important_heads[instance_index] = top_indices[self.head_importance]
        return most_important_heads


@register(_type=HANDLER, _name="hf-head-importance-scalar-mix")
class HeadImportanceScalarMix(HeadLayerAttention):
    """
    Get the head importance from the scalar mix layer in the model. The model is supposed to have a scalar mix layer
    """

    def __init__(
        self, model: torch.nn.Module, tokenizer: Tokenizer, data_loader: EIDataLoader, id_2_label: Dict[int, str], **kwargs
    ):
        super().__init__(model, tokenizer, data_loader, id_2_label, **kwargs)
        self.most_important_head = model.get_most_important_attn_head(head_importance=self.head_importance)

    def get_heads(self, batch_output: SequenceClassifierOutput):
        batch_size, max_head, seq_len, seq_len = batch_output.attentions[self.layer].shape
        heads = torch.full(size=(1, batch_size), fill_value=self.most_important_head).squeeze()
        return heads
