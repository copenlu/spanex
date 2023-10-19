import torch

from explain_interactions.datasets.local import HFInstanceLocalDataset
from explain_interactions.registry import TOKENIZER_REGISTRY
from explain_interactions.dataloaders.hf import HfEIDataLoader

BATCHSZ = 8
data = HFInstanceLocalDataset(
    "../../data/hf-data/mnli/prajjwal-bert-small-mnli/"
).load(phase="validation_matched")
print("data loaded")
tokenizer = TOKENIZER_REGISTRY["prajjwal-bert-small-mnli"]()
print("tokenizer loaded")
dataloader = HfEIDataLoader(batchsz=BATCHSZ, tokenizer=tokenizer)
for batch in dataloader(data):
    shapes = [v.shape for v in batch.values() if type(v) == torch.Tensor]
    assert len(set([x[0] for x in shapes])) == 1
