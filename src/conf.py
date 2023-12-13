from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str
    model_dir: str
    sub_dir: str


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    mid_dim: int
    dropout_rate: float
    edge_dropout_rate: float
    norm_type: Literal["layer", "batch"]
    conv_type: str
    conv_params: dict[str, Any]


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int


@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    val_batch_size: int
    num_workers: int


@dataclass
class OptimizerConfig:
    lr: float


@dataclass
class SchedulerConfig:
    num_warmup_steps: int


@dataclass
class WeightConfig:
    exp_name: str
    run_name: str


@dataclass
class PrepareDataConfig:
    dir: DirConfig
    n_splits: int


@dataclass
class PrepareDegConfig:
    dir: DirConfig
    k: int


@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    fold: int
    k: int
    batch_size: int
    num_workers: int
    dir: DirConfig
    model: ModelConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig


@dataclass
class InferenceConfig:
    exp_name: str
    phase: Literal["val", "test"]
    seed: int
    train_exp_name: str
    train_run_name: str
    batch_size: int
    num_workers: int
    use_amp: bool
    dir: DirConfig
    dataset: DatasetConfig
