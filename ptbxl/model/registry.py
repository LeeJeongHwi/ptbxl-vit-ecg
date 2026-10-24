"""Model registry: pick a model class by name, used by training and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from ptbxl.config import VIT_CONFIG
from ptbxl.model.bilstm import BiLSTM_ECG, load_bilstm
from ptbxl.model.inception import InceptionTime, load_inception
from ptbxl.model.resnet1d import ResNet1D, load_resnet1d
from ptbxl.model.vit import ViT, load_model as load_vit

# Map of model-name -> (constructor, loader) callable pair
MODEL_REGISTRY: dict[str, tuple[Callable[[], nn.Module], Callable[..., nn.Module]]] = {
    "vit":       (ViT,           load_vit),
    "resnet1d":  (ResNet1D,      load_resnet1d),
    "bilstm":    (BiLSTM_ECG,    load_bilstm),
    "inception": (InceptionTime, load_inception),
}


def list_models() -> list[str]:
    """Return the registered model names."""
    return list(MODEL_REGISTRY.keys())


def build_model(name: str) -> nn.Module:
    """Instantiate a model by name with its default configuration."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list_models()}"
        )
    ctor, _ = MODEL_REGISTRY[name]
    return ctor()


def build_vit(cfg: Mapping[str, Any] = VIT_CONFIG) -> ViT:
    """Instantiate the Vision Transformer from a configuration dict.

    The dict uses the keys of VIT_CONFIG: patch (samples), depth, heads, embed, mlp.
    """
    return ViT(
        in_channel=12,
        patch_size=int(cfg["patch"]),
        emb_size=int(cfg["embed"]),
        num_heads=int(cfg["heads"]),
        n_classes=5,
        depth=int(cfg["depth"]),
        mlp_dim=int(cfg["mlp"]),
        signal_length=5000,
    )


def load_checkpoint(name: str, checkpoint: str | Path, device: torch.device) -> nn.Module:
    """Load a checkpoint for the given model name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list_models()}"
        )
    _, loader = MODEL_REGISTRY[name]
    return loader(checkpoint, device)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
