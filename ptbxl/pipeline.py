"""Top-level ML pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from ptbxl.config import PipelineConfig
from ptbxl.data.cache import load_or_build_arrays
from ptbxl.evaluation.metrics import evaluate_model
from ptbxl.model.registry import load_checkpoint
from ptbxl.training.trainer import make_dataloader, set_seed, train_model
from ptbxl.visualization.plots import plot_confusion_heatmap, plot_losses


def ensure_dirs(config: PipelineConfig) -> None:
    """Create output, cache, and model directories before any file writes."""
    for d in (config.output_dir, config.cache_dir, config.model_dir):
        d.mkdir(parents=True, exist_ok=True)


def save_config(config: PipelineConfig) -> Path:
    """Persist the resolved run configuration as JSON for reproducibility."""
    data = asdict(config)
    for key in ("data_root", "output_dir", "cache_dir", "model_dir"):
        data[key] = str(data[key])
    path = config.output_dir / "pipeline_config.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_pipeline(
    config: PipelineConfig,
    *,
    train: bool = True,
    evaluate: bool = True,
) -> dict[str, Any]:
    """End-to-end pipeline: data preparation, training, test evaluation and artifact export.

    Pass train=False with a pre-trained checkpoint in config.model_dir to skip training.
    The test set is evaluated at config.threshold; use sweep_and_evaluate.py to select
    the threshold on the validation set first.
    """
    set_seed(config.seed)
    ensure_dirs(config)
    save_config(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x_train, y_train, x_val, y_val, x_test, y_test = load_or_build_arrays(config)

    train_loader = make_dataloader(x_train, y_train, config.batch_size, config.num_workers, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, config.eval_batch_size, config.num_workers, shuffle=False)
    test_loader = make_dataloader(x_test, y_test, config.eval_batch_size, config.num_workers, shuffle=False)

    model: Optional[nn.Module] = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    if train:
        model, train_losses, val_losses = train_model(config, train_loader, val_loader, device)

    results: dict[str, Any] = {}
    if evaluate:
        best_ckpts = sorted(config.model_dir.glob("best_*.pth"), key=lambda p: p.stat().st_mtime)
        if model is not None:
            # evaluate last (in-memory) model + best checkpoint if it exists
            checkpoints_to_eval = [None] + (best_ckpts[-1:] if best_ckpts else [])
        else:
            all_ckpts = sorted(config.model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
            if not all_ckpts:
                raise FileNotFoundError(
                    f"No .pth checkpoints in {config.model_dir}; run with train=True first."
                )
            checkpoints_to_eval = all_ckpts

        all_results = []
        for ckpt in checkpoints_to_eval:
            eval_model = model if ckpt is None else load_checkpoint(config.model_name, ckpt, device)
            ckpt_name = "trained" if ckpt is None else ckpt.stem
            metrics = evaluate_model(eval_model, test_loader, config.threshold, device)
            print(f"\n[{ckpt_name}] ExactMatch={metrics['exact_match_ratio']:.4f} | HammingLoss={metrics['hamming_loss']:.4f}")
            all_results.append({"checkpoint": ckpt_name, "metrics": metrics})

        results["checkpoints"] = all_results
        if train_losses:
            results["loss_plot"] = str(plot_losses(train_losses, val_losses, config.output_dir))
            results["confusion_plot"] = str(
                plot_confusion_heatmap(all_results[-1]["metrics"]["confusion_matrix"], config.output_dir)
            )

    return results
