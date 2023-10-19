from dataclasses import dataclass


@dataclass
class ModelInitParams:
    expl_intr_model: str
    num_attn_heads: int


@dataclass
class TrainParams:
    batch_size: int
    lr: float
    epochs: int
    output_path: str
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01


@dataclass
class Model:
    init_params: ModelInitParams


@dataclass
class ExpConfig:
    seed: int
    hf_tokenizer: str
    hf_dataset: str
    model: Model
    train_params: TrainParams
