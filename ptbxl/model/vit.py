"""Vision Transformer model for PTB-XL multi-label ECG classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for imbalanced multi-label classification.

    Expects sigmoid probabilities from the model, not raw logits.
    """

    def __init__(self, gamma_neg: float = 2, gamma_pos: float = 0, clip: float = 0.05, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xs_pos, xs_neg = x, 1 - x
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        with torch.no_grad():
            pt = xs_pos * y + xs_neg * (1 - y)
            gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            weight = torch.pow(1 - pt, gamma)
        return -(loss * weight).mean()


class ViTEmbeddings(nn.Module):
    """Patch embedding with CLS token and positional encoding for 1-D ECG signals."""

    def __init__(
        self,
        in_channel: int,
        emb_size: int,
        patch_size: int,
        signal_length: int = 5000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embeddings = nn.Sequential(
            nn.Conv1d(in_channel, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e p -> b p e"),
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(1, signal_length // patch_size + 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(x)
        cls_token = repeat(self.cls_token, "() n e -> b n e", b=x.shape[0])
        return self.dropout(torch.cat([cls_token, embeddings], dim=1) + self.positions)


class MLP(nn.Module):
    """Feed-forward sublayer: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, emb_size: int = 768, dropout: float = 0.35, mlp_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StochasticDepth(nn.Module):
    """Drop entire residual paths during training for regularization (DropPath)."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class EncoderBlock(nn.Module):
    """Transformer encoder block: pre-norm multi-head attention + MLP with stochastic depth."""

    def __init__(
        self,
        emb_size: int = 768,
        num_heads: int = 6,
        f_dropout: float = 0.35,
        dropout: float = 0.2,
        sd_survive: float = 0.0,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout, bias=True, batch_first=True)
        self.lnorm_b = nn.LayerNorm(emb_size)
        self.lnorm_a = nn.LayerNorm(emb_size)
        self.mlp = MLP(emb_size, dropout=f_dropout, mlp_dim=mlp_dim)
        self.sd = StochasticDepth(sd_survive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.lnorm_b(x)
        attn_out, _ = self.attn(normed, normed, normed, average_attn_weights=False)
        x = x + self.sd(attn_out)
        return x + self.sd(self.mlp(self.lnorm_a(x)))

    def get_attention_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw attention weight tensor (batch, heads, seq, seq)."""
        normed = self.lnorm_b(x)
        _, weight = self.attn(normed, normed, normed, average_attn_weights=False)
        return weight


class ViT(nn.Module):
    """Vision Transformer for multi-label ECG classification on PTB-XL.

    The defaults correspond to the final proposed configuration (patch size 20 samples,
    embedding dimension 256, 4 heads, encoder depth 6, MLP dimension 512). Stochastic
    depth uses a linearly increasing drop-path schedule from 0 (first block) to
    drop_path_max (last block).
    """

    def __init__(
        self,
        in_channel: int = 12,
        patch_size: int = 20,
        emb_size: int = 256,
        num_heads: int = 4,
        n_classes: int = 5,
        depth: int = 6,
        mlp_dim: int = 512,
        signal_length: int = 5000,
        emb_dropout: float = 0.1,
        drop_path_max: float = 0.3,
    ) -> None:
        super().__init__()
        sd_rates = np.linspace(0, drop_path_max, depth)
        self.embeddings = ViTEmbeddings(in_channel, emb_size, patch_size, signal_length=signal_length, dropout=emb_dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(emb_size=emb_size, num_heads=num_heads, mlp_dim=mlp_dim, sd_survive=float(sd_rates[i]))
            for i in range(depth)
        ])
        self.classification_head = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.classification_head(x[:, 0]))

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention weights of the final encoder block."""
        x = self.embeddings(x)
        for block in self.blocks[:-1]:
            x = block(x)
        return self.blocks[-1].get_attention_scores(x)


def load_model(checkpoint: str | Path, device: torch.device) -> ViT:
    """Instantiate a ViT and load weights from a checkpoint file."""
    model = ViT().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model
