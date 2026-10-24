"""1D ResNet baseline for PTB-XL multi-label ECG classification.

Architecture follows He et al. (2016) "Deep Residual Learning for Image Recognition",
adapted to 1D temporal signals. The model receives ECG of shape [B, 12, 5000] and
produces sigmoid probabilities over 5 diagnostic super-classes.

Used as a strong CNN baseline for comparison against the proposed ViT model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    """Two-layer residual block with 1D convolutions and BatchNorm."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """1D ResNet-18 style network for multi-label ECG classification.

    Returns sigmoid probabilities (consistent with ViT and AsymmetricLoss).
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.in_channels = base_filters

        # Stem
        self.conv1 = nn.Conv1d(
            in_channels, base_filters,
            kernel_size=15, stride=2, padding=7, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(base_filters,     layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [BasicBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, 5000]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.sigmoid(self.fc(x))


def load_resnet1d(checkpoint: str | Path, device: torch.device) -> ResNet1D:
    """Instantiate a ResNet1D and load weights from a checkpoint file."""
    model = ResNet1D().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
