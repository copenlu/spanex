from transformers import AutoModelForSequenceClassification
from explain_interactions.models import get_device
from explain_interactions.registry import register, MODEL
from explain_interactions.utils import conv_path


@register(_type=MODEL, _name="bert-small-snli")
class PrajjwalBertSmallSNLILocal:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/prajjwal-bert-small"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-base-uncased-snli")
class BertBaseUnCasedSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("boychaboy/SNLI_bert-base-uncased", **kwargs).to(get_device())


@register(_type=MODEL, _name="bert-base-cased-snli")
class BertBaseCasedSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("boychaboy/SNLI_bert-base-cased", **kwargs).to(get_device())


@register(_type=MODEL, _name="bert-large-uncased-snli")
class BertLargeUnCasedSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("boychaboy/SNLI_bert-large-uncased", **kwargs).to(
            get_device()
        )


@register(_type=MODEL, _name="bert-large-cased-snli")
class BertLargeCasedSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("boychaboy/SNLI_bert-large-cased", **kwargs).to(get_device())


@register(_type=MODEL, _name="roberta-small-snli")
class RobertaSmallSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("pepa/roberta-small-snli", **kwargs).to(get_device())


@register(_type=MODEL, _name="roberta-base-snli")
class RobertaBaseSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("pepa/roberta-base-snli", **kwargs).to(get_device())


@register(_type=MODEL, _name="roberta-large-snli")
class RobertaLargeSNLI:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("pepa/roberta-large-snli", **kwargs).to(get_device())


@register(_type=MODEL, _name="bert-base-uncased-fever")
class BertBaseUnCasedFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/bert-base-uncased-fever", **kwargs).to(
            get_device()
        )


@register(_type=MODEL, _name="bert-base-cased-fever")
class BertBaseCasedFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/bert-base-cased-fever", **kwargs).to(get_device())


@register(_type=MODEL, _name="bert-large-uncased-fever")
class BertLargeUnCasedFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/bert-large-uncased-fever", **kwargs).to(
            get_device()
        )


@register(_type=MODEL, _name="bert-large-cased-fever")
class BertLargeCasedFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/bert-large-cased-fever", **kwargs).to(
            get_device()
        )


@register(_type=MODEL, _name="roberta-base-fever")
class RobertaBaseFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/roberta-base-fever", **kwargs).to(get_device())


@register(_type=MODEL, _name="roberta-large-fever")
class RobertaLargeFever:
    @classmethod
    def load(cls, **kwargs):
        return AutoModelForSequenceClassification.from_pretrained("sagnikrayc/roberta-large-fever", **kwargs).to(get_device())
