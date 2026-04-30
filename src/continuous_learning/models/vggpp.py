from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)


class VGGPlusPlusFeatureExtractor(nn.Module):
    """
    VGG-inspired extractor for online CIFAR, matching paper constraints:
    - GroupNorm instead of BatchNorm
    - 10% dropout
    - deeper than the EMNIST extractor
    """

    def __init__(self, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvGNAct(3, 64, dropout),
            ConvGNAct(64, 64, dropout),
            nn.MaxPool2d(2),
            ConvGNAct(64, 128, dropout),
            ConvGNAct(128, 128, dropout),
            nn.MaxPool2d(2),
            ConvGNAct(128, 256, dropout),
            ConvGNAct(256, 256, dropout),
            nn.MaxPool2d(2),
            ConvGNAct(256, 512, dropout),
            ConvGNAct(512, 512, dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        h = self.proj(h)
        return F.normalize(h, p=2, dim=-1)
