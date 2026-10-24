"""One-factor-at-a-time ablation of the Vision Transformer hyperparameters.

Starting from the baseline configuration (patch 20, depth 4, heads 4, embedding 256),
each of patch size, encoder depth, number of heads and embedding dimension is varied in
turn while the other three are held fixed. Every configuration is trained once with the
same protocol as all other experiments (folds 1-8 / 9 / 10, AdamW, asymmetric loss,
cosine annealing, batch 16, learning rate 1e-4, weight decay 0.05, early stopping with
patience 10), the classification threshold is selected on the validation fold by
maximising the exact match ratio, and the test fold is evaluated once at that threshold.
The configuration with the highest validation EMR is the final proposed model.

Usage (from the repository root, virtual environment activated):
    python run_ablation.py --data-root ptb_xl_dataset
    python run_ablation.py --data-root ptb_xl_dataset --skip-completed

Outputs:
    results/ablation/ablation_summary.json       one entry per configuration
    results/ablation/<name>/<name>_val_sweep.csv validation threshold sweep
    models/ablation/<name>/best_<name>.pth       checkpoint with the lowest validation loss
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ptbxl.config import LABEL_NAMES, PipelineConfig
from ptbxl.data.cache import load_or_build_arrays
from ptbxl.evaluation.metrics import confusion_metrics, evaluate_model, threshold_sweep
from ptbxl.model.registry import build_vit, count_parameters
from ptbxl.training.trainer import make_dataloader, set_seed, train_loop

BASELINE = {"patch": 20, "depth": 4, "heads": 4, "embed": 256, "mlp": 512}

ABLATIONS: list[dict[str, Any]] = [
    {"name": "baseline_check", **BASELINE},
    {"name": "patch_10",  "patch": 10, "depth": 4, "heads": 4, "embed": 256, "mlp": 512},
    {"name": "patch_40",  "patch": 40, "depth": 4, "heads": 4, "embed": 256, "mlp": 512},
    {"name": "patch_50",  "patch": 50, "depth": 4, "heads": 4, "embed": 256, "mlp": 512},
    {"name": "depth_2",   "patch": 20, "depth": 2, "heads": 4, "embed": 256, "mlp": 512},
    {"name": "depth_6",   "patch": 20, "depth": 6, "heads": 4, "embed": 256, "mlp": 512},
    {"name": "heads_2",   "patch": 20, "depth": 4, "heads": 2, "embed": 256, "mlp": 512},
    {"name": "heads_8",   "patch": 20, "depth": 4, "heads": 8, "embed": 256, "mlp": 512},
    {"name": "embed_128", "patch": 20, "depth": 4, "heads": 4, "embed": 128, "mlp": 512},
    {"name": "embed_512", "patch": 20, "depth": 4, "heads": 4, "embed": 512, "mlp": 512},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("ptb_xl_dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/ablation"))
    p.add_argument("--model-root", type=Path, default=Path("models/ablation"))
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-thresholds", type=int, default=50)
    p.add_argument("--skip-completed", action="store_true",
                   help="Skip configurations that already have an entry in the summary JSON.")
    return p.parse_args()


def run_single_ablation(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    arrays: tuple,
    device: torch.device,
) -> dict[str, Any]:
    """Train one configuration, select the threshold on validation, evaluate the test fold once."""
    name = cfg["name"]
    print(f"\n{'#' * 90}")
    print(f"#  Ablation: {name}  |  patch={cfg['patch']} depth={cfg['depth']} "
          f"heads={cfg['heads']} embed={cfg['embed']} mlp={cfg['mlp']}")
    print(f"{'#' * 90}\n")

    x_train, y_train, x_val, y_val, x_test, y_test = arrays
    train_loader = make_dataloader(x_train, y_train, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size=32, num_workers=0, shuffle=False)
    test_loader = make_dataloader(x_test, y_test, batch_size=32, num_workers=0, shuffle=False)

    model_dir = args.model_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    model = build_vit(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Trainable params: {n_params:,}")

    train_start = time.time()
    result = train_loop(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.early_stopping_patience,
        checkpoint=model_dir / f"best_{name}.pth",
        desc=name,
    )
    elapsed = time.time() - train_start
    print(f"\nTraining done. Best val loss {result.best_loss:.4f} at epoch {result.best_epoch}. "
          f"Time: {elapsed/60:.2f} min")

    model.load_state_dict(torch.load(result.checkpoint, map_location=device))
    model.eval()

    # threshold selection on the validation fold
    print("\nThreshold sweep on validation...")
    thresholds = np.linspace(0, 1, args.num_thresholds)
    val_df = threshold_sweep(model, val_loader, thresholds, device)
    val_df.to_csv(out_dir / f"{name}_val_sweep.csv", index=False)
    val_emr = val_df.groupby("threshold")["exact_match_ratio"].first()
    best_threshold = float(val_emr.idxmax())
    best_val_emr = float(val_emr.max())
    print(f"  Best threshold on validation: {best_threshold:.4f}  (val EMR = {best_val_emr:.4f})")

    # single evaluation of the test fold at the selected threshold
    print(f"\nTest evaluation at threshold={best_threshold:.4f}...")
    test_metrics = evaluate_model(model, test_loader, best_threshold, device)
    per_class = []
    for label, cm in zip(LABEL_NAMES, test_metrics["confusion_matrix"]):
        m = confusion_metrics(np.asarray(cm))
        m["label"] = label
        per_class.append(m)

    summary = {
        "name": name,
        "config": {k: v for k, v in cfg.items() if k != "name"},
        "trainable_params": n_params,
        "training_time_min": elapsed / 60,
        "best_epoch": result.best_epoch,
        "best_val_loss": float(result.best_loss),
        "best_threshold_on_validation": best_threshold,
        "validation_emr_at_best_threshold": best_val_emr,
        "test_metrics_at_best_threshold": {
            "threshold": best_threshold,
            "exact_match_ratio": test_metrics["exact_match_ratio"],
            "hamming_loss": test_metrics["hamming_loss"],
            "per_class": per_class,
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
    }
    print(f"\n>>> {name}: Test EMR = {test_metrics['exact_match_ratio']:.4f}  "
          f"Hamming = {test_metrics['hamming_loss']:.4f}\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def load_existing_results(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    try:
        return json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
    except Exception:
        return []


def save_results(out_path: Path, results: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"baseline": BASELINE, "results": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ablation configs: {len(ABLATIONS)}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ablation_summary.json"

    config = PipelineConfig(data_root=args.data_root, cache_dir=args.cache_dir, num_workers=0)
    print("Loading cached arrays...")
    arrays = load_or_build_arrays(config)
    print(f"  Train: {arrays[0].shape}  Val: {arrays[2].shape}  Test: {arrays[4].shape}\n")

    results = load_existing_results(summary_path)
    completed = {r["name"] for r in results}
    if args.skip_completed and completed:
        print(f"Resuming. Already completed: {sorted(completed)}\n")

    overall_start = time.time()
    for idx, cfg in enumerate(ABLATIONS, start=1):
        if args.skip_completed and cfg["name"] in completed:
            print(f"[{idx}/{len(ABLATIONS)}] {cfg['name']}: skipped (already done)")
            continue
        print(f"\n[{idx}/{len(ABLATIONS)}] Running ablation: {cfg['name']}")
        result = run_single_ablation(cfg, args, arrays, device)
        # save after every configuration so that a partial run is not lost
        results = [r for r in results if r["name"] != cfg["name"]]
        results.append(result)
        save_results(summary_path, results)
        print(f"  Updated summary: {summary_path}")

    total_elapsed = time.time() - overall_start
    print(f"\nAll ablations done in {total_elapsed/60:.2f} min")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
