"""
A manual explanation is of the form: ExplanationPhrase: where Phrase: {"text", "start_token_index", "end_token_index",
"score"}. A comparison baseline takes an annotated instance, and if the manual explanation does not have a score, it
derives a score from the network (possibly from the attention weight). The same scoring fn. is used for a random explanation.
Our goal is to see if the manual explanation indeed has a higher score than the random one.
"""
# flake8: noqa
from explain_interactions.methods.annotation.explainer import AnnotationExplainer
