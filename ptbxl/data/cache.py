"""Cache I/O for preprocessed numpy split arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ptbxl.config import PipelineConfig
from ptbxl.data.dataset import PTBXLDataset, collect_labeled_arrays


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    """Return canonical paths for the six cached numpy split files."""
    return {n: cache_dir / f"{n}.npy" for n in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")}


def save_arrays(cache_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    """Persist split arrays so later runs skip WFDB loading."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(cache_dir / f"{name}.npy", value)


def load_arrays(
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the six cached split arrays produced by save_arrays."""
    paths = cache_paths(cache_dir)
    return tuple(np.load(paths[n]) for n in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"))


def load_or_build_arrays(
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single entry-point for data: return cached arrays or build from PTB-XL records."""
    paths = cache_paths(config.cache_dir)
    if not config.rebuild_cache and all(p.exists() for p in paths.values()):
        return load_arrays(config.cache_dir)

    train_set = PTBXLDataset(config.data_root, config.train_folds, sampling_rate=config.sampling_rate)
    val_set = PTBXLDataset(config.data_root, config.val_folds, sampling_rate=config.sampling_rate)
    test_set = PTBXLDataset(config.data_root, config.test_folds, sampling_rate=config.sampling_rate)

    x_train, y_train = collect_labeled_arrays(train_set)
    x_val, y_val = collect_labeled_arrays(val_set)
    x_test, y_test = collect_labeled_arrays(test_set)

    save_arrays(config.cache_dir, {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    })
    return x_train, y_train, x_val, y_val, x_test, y_test
