from dataclasses import dataclass
from typing import List

@dataclass
class MaskConfig:
    model: str
    attn_heads: int = 0
    mlps: int = 0

@dataclass
class DataConfig:
    seed: int
    name: str
    path: str
    workers:int
    val_size: float

@dataclass
class TrainerConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    checkpoint_path: str
    results_path: str


@dataclass
class DiffMaskConfig:
    seed: int
    mask: MaskConfig
    data: DataConfig
    trainer: TrainerConfig
