from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continuous_learning.config import ModelConfig


def reset_probability(chunk_size: int, current_step: int) -> float:
    if current_step <= 0:
        raise ValueError("current_step must be >= 1")
    return min(1.0, float(chunk_size) / float(current_step))


@dataclass
class ReplayStreamReader:
    """Maintains one replay stream state and chunk sampling."""

    task_stream: list[np.ndarray]
    model_config: ModelConfig
    seed: int
    current_task: int = 0
    position: int = 0
    allow_reset: bool = True

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.current_task >= len(self.task_stream):
            self.current_task = 0
        self.last_chunk_info: dict[str, int | bool] | None = None

    def maybe_reset(self, current_step: int) -> None:
        if not self.allow_reset:
            return
        p_reset = reset_probability(self.model_config.chunk_size, current_step)
        if self._rng.random() < p_reset:
            self.current_task = int(self._rng.integers(0, len(self.task_stream)))
            self.position = 0

    def next_chunk(self, current_step: int) -> np.ndarray:
        self.maybe_reset(current_step)
        task_id_before = int(self.current_task)
        task_indices = self.task_stream[task_id_before]

        start = self.position
        stop = start + self.model_config.chunk_size
        wrapped = False
        if stop <= len(task_indices):
            chunk = task_indices[start:stop]
            self.position = stop
        else:
            head = task_indices[start:]
            tail_size = stop - len(task_indices)
            tail = task_indices[:tail_size]
            chunk = np.concatenate([head, tail])
            self.current_task = (self.current_task + 1) % len(self.task_stream)
            self.position = tail_size
            wrapped = True

        self.last_chunk_info = {
            "task_id": task_id_before,
            "position_start": int(start),
            "position_stop": int(stop),
            "wrapped": wrapped,
        }

        return chunk.astype(np.int64)
