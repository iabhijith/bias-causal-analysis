from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    name: str
    path: str
    workers:int
    batch_size: int
    val_size: float

@dataclass
class CMAConfig:
    seed: int
    model: str
    attn_heads: int
    results_path: str
    data: DataConfig

