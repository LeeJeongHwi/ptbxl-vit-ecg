"""Constants and shared configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

CLASS_MAP: dict[str, int] = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
LABEL_NAMES: list[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Architecture of the final proposed Vision Transformer (patch size in samples,
# encoder depth, attention heads, embedding dimension, MLP hidden dimension).
VIT_CONFIG: dict[str, int] = {"patch": 20, "depth": 6, "heads": 4, "embed": 256, "mlp": 512}

Record = Mapping[str, Any]
DiagnosticLookup = Mapping[str, Union[str, Mapping[str, Any], Any]]
PreprocessFn = Callable[[Sequence[float]], Sequence[float]]


@dataclass
class PipelineConfig:
    """All pipeline parameters in one place.

    data_root is required for ML operations; pass Path(".") when using only
    pure data utilities (split_by_folds, window_signal, run_data_pipeline, etc.).
    """

    data_root: Path = Path(".")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("npy_data")
    model_dir: Path = Path("models")
    model_name: str = "vit"          # "vit" | "resnet1d" | "bilstm" | "inception"
    sampling_rate: int = 500
    train_folds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple[int, ...] = (9,)
    test_folds: tuple[int, ...] = (10,)
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 70
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    num_workers: int = 0
    threshold: float = 0.7143
    seed: int = 42
    rebuild_cache: bool = False
    early_stopping_patience: int = 10
    class_map: dict[str, int] = field(default_factory=lambda: CLASS_MAP.copy())
    # Optional windowing used by run_data_pipeline
    window_size: Optional[int] = None
    stride: Optional[int] = None

    def __post_init__(self) -> None:
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
