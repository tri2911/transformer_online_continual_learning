from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class StreamConfig:
    n_tasks: int = 100
    examples_per_task: int = 5_000
    classes_per_task: int = 10
    seed: int = 42
    cifar_mean: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
    cifar_std: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 100
    feature_dim: int = 512
    d_model: int = 256
    n_blocks: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    context_window: int = 512
    chunk_size: int = 50


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    replay_streams: int = 20
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_max_norm: float = 1.0
    device: str = "cpu"
    max_steps: int = 2_000
    log_every: int = 25
    eval_every: int = 250
    rolling_window: int = 500
    seeds: Sequence[int] = field(default_factory=lambda: (42, 123, 456))
    features_cache: str = "data/features_cache.pt"
    use_feature_cache: bool = True
