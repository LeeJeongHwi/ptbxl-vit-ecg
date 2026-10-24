"""Bidirectional LSTM baseline for PTB-XL multi-label ECG classification.

Uses a small 1D convolutional stem to downsample the 5000-sample signal before
feeding it to a stacked BiLSTM. Returns sigmoid probabilities, consistent with
the ViT model and AsymmetricLoss criterion.

Used as a sequential-model baseline for comparison against the proposed ViT.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class BiLSTM_ECG(nn.Module):
    """BiLSTM over CNN-extracted features for multi-label ECG classification."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # 1D CNN stem to downsample 5000 -> ~156
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, 5000]
        x = self.feature_extractor(x)        # [B, 128, ~156]
        x = x.permute(0, 2, 1)                # [B, ~156, 128]
        lstm_out, _ = self.lstm(x)            # [B, ~156, hidden*2]
        x = lstm_out.mean(dim=1)              # global temporal average -> [B, hidden*2]
        return self.sigmoid(self.classifier(x))


def load_bilstm(checkpoint: str | Path, device: torch.device) -> BiLSTM_ECG:
    """Instantiate a BiLSTM_ECG and load weights from a checkpoint file."""
    model = BiLSTM_ECG().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
