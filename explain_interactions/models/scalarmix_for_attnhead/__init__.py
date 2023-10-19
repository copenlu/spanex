from explain_interactions.models.scalarmix_for_attnhead.model import ClassifierWithAttnHeadDistribution
from explain_interactions.registry import MODEL, register
from explain_interactions.utils import conv_path
from explain_interactions.models import get_device


@register(_type=MODEL, _name="bert-small-snli-scalar-mix")
class BertSmallSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/bertsmall-sm/checkpoint-137344"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-base-cased-snli-scalar-mix")
class BertBaseCasedSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/bertbc-sm/checkpoint-137344"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-base-uncased-snli-scalar-mix")
class BertBaseUncasedSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/bertbu-sm/checkpoint-137344"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-large-cased-snli-scalar-mix")
class BertLargeCasedSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/bertlc-sm/checkpoint-103008"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-large-uncased-snli-scalar-mix")
class BertLargeUnCasedLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/bertlu-sm/checkpoint-103008"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="roberta-base-snli-scalar-mix")
class RobertaBaseSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/rb-sm/checkpoint-137344"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="roberta-large-snli-scalar-mix")
class RobertaLargeSNLILocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/snli/rl-sm/checkpoint-103008"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-base-cased-fever-scalar-mix")
class BertBaseCasedFeverLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/bertbc-sm/checkpoint-57048"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-base-uncased-fever-scalar-mix")
class BertBaseUncasedFeverLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/bertbu-sm/checkpoint-57048"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-large-cased-fever-scalar-mix")
class BertLargeCasedFeverLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/bertlc-sm/checkpoint-57048"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="bert-large-uncased-fever-scalar-mix")
class BertLargeUnCasedFeverScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/bertlu-sm/checkpoint-57048"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="roberta-base-fever-scalar-mix")
class RobertaBaseFeverLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/rb-sm/checkpoint-57044"), **kwargs
        ).to(get_device())


@register(_type=MODEL, _name="roberta-large-fever-scalar-mix")
class RobertaLargeFeverLocalScalarMix:
    @classmethod
    def load(cls, **kwargs):
        return ClassifierWithAttnHeadDistribution.load(
            conv_path("${EXPL_INTR_HOME}/trained-models/fever/scalar-mix/rl-sm/checkpoint-57044"), **kwargs
        ).to(get_device())
