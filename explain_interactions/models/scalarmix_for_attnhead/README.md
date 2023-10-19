## Scalar mix for attn head

Given a trained AutoModelForSeqClassification, since we do not really know which attn head is the most important, we are
using two strategies:
1. Looking directly at the classifier layers of the model. See handlers `hf-head-layer-head-importance` for this strategy.
2. Training a separate model that will take the cls token output of the original model, use a scalar mix layer (of size num_attn_head) to combine the cls token split results. The scalar mix layer of this model should tell us which head was the most important one. This package contains code for that.
