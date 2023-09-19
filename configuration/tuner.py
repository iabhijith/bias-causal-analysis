from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    name: str
    components_path: str
    components: str

@dataclass
class DataConfig:
    name: str
    path: str
    workers:int
    val_size: float

@dataclass
class TunerConfig:
    epochs: int
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    checkpoint_path: str
    results_path: str


@dataclass
class MitigationConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    tuner: TunerConfig
