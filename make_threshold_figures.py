"""Confusion matrices and threshold-sweep curves of the proposed Vision Transformer (Figures 2-5).

All four figures refer to the final configuration (encoder depth 6) trained in the
ablation study, under the fixed protocol (folds 1-8 / 9 / 10, threshold selected on the
validation fold).

  Figure 2  per-class confusion matrices on the test fold at the selected threshold
  Figure 3  per-class accuracy on the validation fold as a function of the threshold
  Figure 4  per-class precision (dashed) and recall (solid) on the validation fold
  Figure 5  per-class F1 score on the validation fold

In Figures 3-5 each curve is drawn up to the last threshold at which the class is still
predicted (recall >= 0.01); beyond that point the model predicts no positive and precision
and F1 are undefined.

Usage (from the repository root, virtual environment activated):
    python make_threshold_figures.py
    python make_threshold_figures.py --out-dir figures

Inputs: results/ablation/ablation_summary.json and results/ablation/depth_6/depth_6_val_sweep.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
DPI = 300
SWEEP_FIGSIZE = (12, 8)
SWEEP_GRID = (2, 3)      # last cell hidden
MIN_RECALL = 0.01        # curves are cut where recall falls below this value


def load_confusion_matrices(summary_path: Path, name: str = "depth_6"):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    entry = next(r for r in data["results"] if r["name"] == name)
    return entry["test_metrics_at_best_threshold"]["confusion_matrix"]


def setup_axes(n_rows: int, n_cols: int):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=SWEEP_FIGSIZE)
    return fig, axes.flatten()


def valid_mask(sub: pd.DataFrame) -> np.ndarray:
    """Rows up to and including the last threshold with recall >= MIN_RECALL."""
    recall = sub["recall"].fillna(0).to_numpy()
    valid_idx = np.where(recall >= MIN_RECALL)[0]
    mask = np.zeros(len(sub), dtype=bool)
    if len(valid_idx):
        mask[: valid_idx[-1] + 1] = True
    return mask


def make_figure2(cms, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes = axes.flatten()
    for i, (label, cm) in enumerate(zip(LABELS, cms)):
        cm_arr = np.array(cm)             # sklearn layout: [[TN, FP], [FN, TP]]
        TN, FP = cm_arr[0]
        FN, TP = cm_arr[1]
        display = np.array([[TP, FN], [FP, TN]])   # actual positive on top, predicted positive on the left
        im = sns.heatmap(
            display, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"],
            ax=axes[i], cbar=False, annot_kws={"size": 14},
            linewidths=0.5, linecolor="white",
            square=True,
        )
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(im.collections[0], cax=cax)
        cbar.outline.set_visible(False)
        axes[i].set_title(f"{label} Confusion Matrix", fontsize=13)
        axes[i].set_xlabel("Predicted", fontsize=11)
        axes[i].set_ylabel("Actual", fontsize=11)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=90)
    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def make_sweep_figure(df: pd.DataFrame, out: Path, kind: str) -> None:
    fig, axes = setup_axes(*SWEEP_GRID)
    for i, label in enumerate(LABELS):
        sub = df[df["label"] == label].sort_values("threshold").reset_index(drop=True)
        sub = sub[valid_mask(sub)]
        ax = axes[i]
        if kind == "accuracy":
            ax.plot(sub["threshold"], sub["accuracy"], linewidth=2)
            ax.set_ylabel("Accuracy", fontsize=11)
        elif kind == "precision_recall":
            ax.plot(sub["threshold"], sub["precision"], linestyle="--", color="red", linewidth=2, label="Precision")
            ax.plot(sub["threshold"], sub["recall"], linestyle="-", color="tab:blue", linewidth=2, label="Recall")
            ax.set_ylabel("Value", fontsize=11)
            ax.legend(loc="lower right", fontsize=10)
        else:
            ax.plot(sub["threshold"], sub["f1"], color="tab:blue", linewidth=2)
            ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.4)
    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("results/ablation/ablation_summary.json"))
    ap.add_argument("--sweep", type=Path, default=Path("results/ablation/depth_6/depth_6_val_sweep.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    for p in (args.summary, args.sweep):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run from the repository root after run_ablation.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sweep)
    make_figure2(load_confusion_matrices(args.summary), args.out_dir / "Figure2_confusion_matrices.png")
    make_sweep_figure(df, args.out_dir / "Figure3_accuracy_validation.png", "accuracy")
    make_sweep_figure(df, args.out_dir / "Figure4_precision_recall_validation.png", "precision_recall")
    make_sweep_figure(df, args.out_dir / "Figure5_f1_validation.png", "f1")


if __name__ == "__main__":
    main()
