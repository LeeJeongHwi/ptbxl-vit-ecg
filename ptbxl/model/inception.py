"""InceptionTime baseline for PTB-XL multi-label ECG classification.

Reference:
    Ismail Fawaz, H., Lucas, B., Forestier, G., et al.
    "InceptionTime: Finding AlexNet for Time Series Classification."
    Data Mining and Knowledge Discovery 34, 1936-1962 (2020).

Architecture:
    - 6 Inception blocks (default 128 filters each)
    - Residual connections every 3 blocks
    - Bottleneck layer reduces channels before parallel multi-kernel convs
    - Global Average Pooling + Linear classifier
    - Sigmoid output for multi-label classification

Used as a strong sequence-CNN baseline (time-series SOTA architecture).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class InceptionModule(nn.Module):
    """One Inception block: parallel 1D convs at multiple kernel sizes + maxpool branch."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (10, 20, 40),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1

        # Bottleneck: reduce channels before expensive multi-kernel convs
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels,
                kernel_size=1, stride=1, padding=0, bias=False,
            )
            conv_in = bottleneck_channels
        else:
            conv_in = in_channels

        # Three parallel convs with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                conv_in, n_filters,
                kernel_size=k, stride=1, padding=k // 2, bias=False,
            )
            for k in kernel_sizes
        ])

        # Maxpool branch followed by 1x1 conv
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(
            in_channels, n_filters,
            kernel_size=1, stride=1, padding=0, bias=False,
        )

        # Output: 4 branches × n_filters
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T]
        z = self.bottleneck(x) if self.use_bottleneck else x

        # Three multi-kernel convs from bottleneck output
        out_convs = [conv(z) for conv in self.convs]

        # Maxpool branch operates on original input (not bottleneck)
        out_pool = self.maxpool_conv(self.maxpool(x))

        # Match temporal length (multi-kernel padding may shift by 1 due to even kernel sizes)
        target_len = min(o.shape[-1] for o in out_convs + [out_pool])
        out_convs = [o[..., :target_len] for o in out_convs]
        out_pool = out_pool[..., :target_len]

        out = torch.cat(out_convs + [out_pool], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """InceptionTime network for multi-label ECG classification on PTB-XL.

    Returns sigmoid probabilities (consistent with ViT, ResNet1D, BiLSTM, and AsymmetricLoss).
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        n_filters: int = 32,
        depth: int = 6,
        bottleneck_channels: int = 32,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.use_residual = use_residual

        # Per-block output channels = 4 branches × n_filters
        block_out = n_filters * 4

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_in = in_channels if i == 0 else block_out
            self.blocks.append(
                InceptionModule(
                    in_channels=block_in,
                    n_filters=n_filters,
                    bottleneck_channels=bottleneck_channels,
                )
            )

        # Residual shortcut every 3 blocks (after blocks 2 and 5)
        if use_residual:
            self.shortcut_in = nn.ModuleList([
                nn.Conv1d(in_channels, block_out, kernel_size=1, bias=False),
                nn.Conv1d(block_out, block_out, kernel_size=1, bias=False),
            ])
            self.shortcut_bn = nn.ModuleList([
                nn.BatchNorm1d(block_out),
                nn.BatchNorm1d(block_out),
            ])
            self.shortcut_relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_out, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, 5000]
        residual_input = x

        for i, block in enumerate(self.blocks):
            x = block(x)
            # Residual every 3 blocks (i = 2 and 5)
            if self.use_residual and (i + 1) % 3 == 0:
                shortcut_idx = (i + 1) // 3 - 1
                shortcut = self.shortcut_in[shortcut_idx](residual_input)
                shortcut = self.shortcut_bn[shortcut_idx](shortcut)
                # Match temporal lengths
                target_len = min(x.shape[-1], shortcut.shape[-1])
                x = x[..., :target_len] + shortcut[..., :target_len]
                x = self.shortcut_relu(x)
                residual_input = x

        x = self.gap(x).flatten(1)
        return self.sigmoid(self.fc(x))


def load_inception(checkpoint: str | Path, device: torch.device) -> InceptionTime:
    """Instantiate an InceptionTime and load weights from a checkpoint file."""
    model = InceptionTime().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
