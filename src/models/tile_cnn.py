"""Custom CNN for tile-based litter classification, built from scratch.

Architecture: a stack of convolution–pooling blocks followed by a dense
classification head. Each block is Conv3x3 → BatchNorm → ReLU → MaxPool2,
halving spatial resolution while widening channels. Global average pooling
bridges to the head, which keeps the parameter count small and makes the
network input-size tolerant. Output is a single logit for binary
litter/no-litter classification (used with BCEWithLogitsLoss).
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU -> MaxPool2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TileCNN(nn.Module):
    """Stacked conv-pool blocks + dense head, single-logit output."""

    def __init__(
        self,
        conv_channels: list[int] = (32, 64, 128, 256, 256),
        dense_units: int = 128,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in conv_channels:
            blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch,)."""
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
