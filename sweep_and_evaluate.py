"""Validation-based threshold selection and single test evaluation for one checkpoint.

  1. sweep 50 thresholds in [0, 1] on the validation fold
  2. keep the threshold that maximises the exact match ratio
  3. evaluate the held-out test fold once at that threshold
  4. save the validation sweep (CSV) and the test metrics (JSON)

Usage (from the repository root, virtual environment activated):
    python sweep_and_evaluate.py --model vit --checkpoint models/ablation/depth_6/best_depth_6.pth --output-dir results/vit
    python sweep_and_evaluate.py --model resnet1d --checkpoint <path to .pth> --output-dir results/resnet1d
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ptbxl.config import LABEL_NAMES, PipelineConfig
from ptbxl.data.cache import load_or_build_arrays
from ptbxl.evaluation.metrics import confusion_metrics, evaluate_model, threshold_sweep
from ptbxl.model.registry import load_checkpoint
from ptbxl.training.trainer import make_dataloader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("ptb_xl_dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    p.add_argument("--model", type=str, required=True,
                   choices=("vit", "resnet1d", "bilstm", "inception"))
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to the .pth checkpoint")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save the sweep CSV and final metrics JSON")
    p.add_argument("--num-thresholds", type=int, default=50)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # data (cached arrays)
    config = PipelineConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    print("\nLoading cached arrays...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_or_build_arrays(config)
    print(f"  Train: {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    val_loader = make_dataloader(x_val, y_val,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers, shuffle=False)
    test_loader = make_dataloader(x_test, y_test,
                                  batch_size=args.eval_batch_size,
                                  num_workers=args.num_workers, shuffle=False)

    # model
    print(f"\nLoading checkpoint...")
    model = load_checkpoint(args.model, args.checkpoint, device)

    # step 1: threshold sweep on the validation fold
    print(f"\n{'='*70}")
    print(f"  Step 1: Threshold sweep on VALIDATION set "
          f"({args.num_thresholds} points)")
    print(f"{'='*70}")
    thresholds = np.linspace(0, 1, args.num_thresholds)
    t0 = time.time()
    val_sweep_df = threshold_sweep(model, val_loader, thresholds, device)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    val_csv_path = args.output_dir / f"{args.model}_val_sweep.csv"
    val_sweep_df.to_csv(val_csv_path, index=False)
    print(f"  Saved: {val_csv_path}")

    # best threshold by EMR (a record-level metric, identical across the per-class rows of a threshold)
    val_emr = val_sweep_df.groupby("threshold")["exact_match_ratio"].first()
    best_threshold = float(val_emr.idxmax())
    best_val_emr = float(val_emr.max())
    print(f"\n  Best threshold on validation: {best_threshold:.4f}")
    print(f"  Validation EMR at best threshold: {best_val_emr:.4f}")

    # step 2: single evaluation of the test fold
    print(f"\n{'='*70}")
    print(f"  Step 2: Final TEST evaluation at threshold={best_threshold:.4f}")
    print(f"{'='*70}")
    test_metrics = evaluate_model(model, test_loader, best_threshold, device)
    print(f"\n  Test EMR:          {test_metrics['exact_match_ratio']:.4f}")
    print(f"  Test Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    # per-class metrics from the confusion matrices
    per_class = []
    for label, cm in zip(LABEL_NAMES, test_metrics["confusion_matrix"]):
        m = confusion_metrics(np.asarray(cm))
        m["label"] = label
        per_class.append(m)

    # save
    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "num_thresholds_swept": args.num_thresholds,
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
    out_json = args.output_dir / f"{args.model}_final_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_json}")

    # summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({args.model})")
    print(f"{'='*70}")
    print(f"  Best threshold (val):     {best_threshold:.4f}")
    print(f"  Validation EMR:           {best_val_emr:.4f}")
    print(f"  Test EMR:                 {test_metrics['exact_match_ratio']:.4f}")
    print(f"  Test Hamming Loss:        {test_metrics['hamming_loss']:.4f}")
    print(f"\n  Per-class (TEST):")
    print(f"  {'Label':<8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for pc in per_class:
        print(f"  {pc['label']:<8} {pc['accuracy']:>8.4f} "
              f"{pc['precision']:>8.4f} {pc['recall']:>8.4f} {pc['f1']:>8.4f}")


if __name__ == "__main__":
    main()
