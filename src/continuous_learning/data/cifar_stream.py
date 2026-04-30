from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from continuous_learning.config import StreamConfig


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    classes: np.ndarray
    seed: int


class CIFARTaskStream:
    """Constructs paper-style tasks for CIFAR-100 continual streams."""

    def __init__(
        self,
        labels: np.ndarray,
        config: StreamConfig,
        num_total_classes: int = 100,
        features_cache: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.config = config
        self.num_total_classes = num_total_classes
        self.features_cache = features_cache
        self.device = device
        self._root_rng = np.random.default_rng(config.seed)
        self._class_to_indices = {
            cls: np.flatnonzero(self.labels == cls) for cls in range(num_total_classes)
        }
        self.task_specs = self._build_task_specs()
        self._use_cache = False
        self._feat_bank: torch.Tensor | None = None
        self._label_bank: torch.Tensor | None = None
        if features_cache:
            cache = torch.load(features_cache, map_location="cpu")
            self._feat_bank = cache["features"].contiguous()
            self._label_bank = cache["labels"].contiguous()
            self._use_cache = True
            print(f"[CIFARTaskStream] Using pre-extracted features: {tuple(self._feat_bank.shape)}")

    def _build_task_specs(self) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        for task_id in range(self.config.n_tasks):
            classes = self._root_rng.choice(
                self.num_total_classes,
                size=self.config.classes_per_task,
                replace=False,
            )
            seed = int(self._root_rng.integers(0, 2**31 - 1))
            specs.append(TaskSpec(task_id=task_id, classes=classes, seed=seed))
        return specs

    def sample_indices_for_task(self, task_spec: TaskSpec) -> np.ndarray:
        """Samples one task worth of examples with replacement, per paper."""
        task_rng = np.random.default_rng(task_spec.seed)
        pool = np.concatenate([self._class_to_indices[int(c)] for c in task_spec.classes])
        if pool.size == 0:
            raise ValueError(f"No indices found for classes {task_spec.classes.tolist()}")
        sampled = task_rng.choice(
            pool,
            size=self.config.examples_per_task,
            replace=True,
        )
        return sampled.astype(np.int64)

    def build_task_index_stream(self) -> list[np.ndarray]:
        return [self.sample_indices_for_task(task_spec) for task_spec in self.task_specs]

    def get_cached_features(self, indices: np.ndarray) -> torch.Tensor:
        if not self._use_cache or self._feat_bank is None:
            raise RuntimeError("Feature cache is not enabled for this stream.")
        idx = torch.as_tensor(indices, dtype=torch.long)
        return self._feat_bank[idx].to(self.device)

    def has_feature_cache(self) -> bool:
        return self._use_cache

    @property
    def cached_labels(self) -> torch.Tensor | None:
        return self._label_bank


class CIFARStream(CIFARTaskStream):
    """Compatibility alias with cache-related constructor args."""

    pass


def cifar100_transforms(config: StreamConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=config.cifar_mean, std=config.cifar_std),
        ]
    )


def load_cifar100(
    root: str,
    config: StreamConfig,
    train: bool = True,
    download: bool = True,
) -> datasets.CIFAR100:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            category=Warning,
        )
        return datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=cifar100_transforms(config),
        )


def load_feature_cache(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(path, map_location="cpu")
    return cache["features"].contiguous(), cache["labels"].contiguous()
