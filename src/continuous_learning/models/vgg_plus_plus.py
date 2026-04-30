from __future__ import annotations

import torch
import torch.nn as nn


class _ConvGNReLUDrop(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return self.drop(x)


class VGGPlusPlusJoint(nn.Module):
    """Appendix-D style VGG++ encoder for joint continual-learning training."""

    def __init__(self, output_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvGNReLUDrop(3, 64, dropout=dropout),
            _ConvGNReLUDrop(64, 64, dropout=dropout),
            nn.MaxPool2d(2),
            _ConvGNReLUDrop(64, 128, dropout=dropout),
            _ConvGNReLUDrop(128, 128, dropout=dropout),
            nn.MaxPool2d(2),
            _ConvGNReLUDrop(128, 256, dropout=dropout),
            _ConvGNReLUDrop(256, 256, dropout=dropout),
            nn.MaxPool2d(2),
            _ConvGNReLUDrop(256, 512, dropout=dropout),
            _ConvGNReLUDrop(512, 512, dropout=dropout),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.project = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.project(h)
