from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import torch


def instantaneous_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def rolling_mean(values: deque[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_recovery_window_accuracy(
    anchor_recovery: dict[str | int, list[float]],
    *,
    first_n_examples: int = 100,
) -> float:
    """Mean accuracy over the first N examples of every task.

    This isolates adaptation quality at task switches instead of being dominated
    by steady-state within-task memorization.
    """
    if first_n_examples <= 0:
        raise ValueError("first_n_examples must be > 0")

    values: list[float] = []
    for curve in anchor_recovery.values():
        if not curve:
            continue
        values.extend(float(x) for x in curve[:first_n_examples])
    if not values:
        return 0.0
    return sum(values) / len(values)


@dataclass
class OnlineMetricTracker:
    rolling_window: int = 500
    instant_history: list[float] = field(default_factory=list)
    rolling_buffer: deque[float] = field(default_factory=deque)
    per_task_correct: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_task_total: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    switch_step: list[int] = field(default_factory=list)

    def update(self, task_id: int, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        instant = instantaneous_accuracy(logits, targets)
        self.instant_history.append(instant)
        self.rolling_buffer.append(instant)
        if len(self.rolling_buffer) > self.rolling_window:
            self.rolling_buffer.popleft()

        preds = logits.argmax(dim=-1)
        correct = int((preds == targets).sum().item())
        total = int(targets.numel())
        self.per_task_correct[task_id] += correct
        self.per_task_total[task_id] += total
        task_avg = self.per_task_correct[task_id] / max(self.per_task_total[task_id], 1)

        return {
            "instant_accuracy": instant,
            "rolling_accuracy": rolling_mean(self.rolling_buffer),
            "task_average_accuracy": task_avg,
        }

    def mark_task_switch(self, global_step: int) -> None:
        self.switch_step.append(global_step)

    def accuracy_vs_task_number(self) -> dict[int, float]:
        return {
            task_id: self.per_task_correct[task_id] / max(self.per_task_total[task_id], 1)
            for task_id in sorted(self.per_task_total.keys())
        }

    def recovery_speed(self, window: int = 50) -> dict[int, float]:
        """Mean instant accuracy in the first N steps after each task switch."""
        out: dict[int, float] = {}
        for i, start in enumerate(self.switch_step):
            segment = self.instant_history[start : start + window]
            out[i] = 0.0 if not segment else sum(segment) / len(segment)
        return out
