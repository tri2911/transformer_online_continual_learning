from __future__ import annotations

import torch
import torch.nn as nn


class OnlineSGDHead(nn.Module):
    """Chunk-wise linear classifier baseline (no temporal cache)."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        caches: list[object] | None = None,
    ) -> tuple[torch.Tensor, list[object]]:
        del labels, caches
        logits = self.classifier(features)
        return logits, []
