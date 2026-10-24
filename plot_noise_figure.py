#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure: test-set metric versus SNR under synthetic noise, one panel per noise type.

The panels are laid out in a 2 x 3 grid (Gaussian, baseline wander, powerline 50 Hz,
powerline 60 Hz, muscle artifact, composite) and are read from the aggregated
noise_summary.csv files written by run_noise_robustness.py; no model is evaluated.

Usage (from the repository root, virtual environment activated):
    python plot_noise_figure.py
    python plot_noise_figure.py --metric f1_HYP --out results/noise_robustness/fig_noise_hyp.png
    python plot_noise_figure.py --metric d_emr     # degradation relative to the clean signal

Outputs: <out>.png (300 dpi) and <out>.pdf (default results/noise_robustness/fig_noise_robustness.png)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_multiseed import MODEL_DISPLAY, MODEL_NAMES                       # noqa: E402

# categorical colours (colour-blind safe) with a marker as a second encoding
MODEL_STYLE = {
    "vit_depth6":    dict(color="#2a78d6", marker="o"),
    "resnet1d":      dict(color="#eb6834", marker="s"),
    "bilstm":        dict(color="#1baf7a", marker="^"),
    "inceptiontime": dict(color="#eda100", marker="D"),
}
METRIC_LABEL = {
    "emr": "Test EMR", "hamming": "Test Hamming loss", "macro_f1": "Test macro-F1",
    "d_emr": "ΔEMR vs. clean", "d_macro_f1": "Δmacro-F1 vs. clean",
}
for lab in ("NORM", "MI", "STTC", "CD", "HYP"):
    METRIC_LABEL[f"f1_{lab}"] = f"Test {lab} F1"
    METRIC_LABEL[f"d_f1_{lab}"] = f"Δ{lab} F1 vs. clean"


def load_summary(path: Path, metric: str) -> Dict[Tuple[str, Optional[float]], Dict[str, Tuple[float, float]]]:
    """-> {(noise_type, snr or None): {model: (mean, std)}}"""
    out: Dict[Tuple[str, Optional[float]], Dict[str, Tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["metric"] != metric:
                continue
            snr = None if r["noise_type"] == "clean" or r["snr_db"] in ("", "None") else float(r["snr_db"])
            out.setdefault((r["noise_type"], snr), {})[r["model"]] = (float(r["mean"]), float(r["std"]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir50", type=Path, default=Path("results/noise_robustness"),
                    help="aggregated results with all noise types (powerline at 50 Hz)")
    ap.add_argument("--dir60", type=Path, default=Path("results/noise_robustness_60hz"),
                    help="aggregated results of the 60 Hz powerline run (panel skipped if missing)")
    ap.add_argument("--metric", default="emr", choices=sorted(METRIC_LABEL))
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    s50 = load_summary(args.dir50 / "noise_summary.csv", args.metric)
    s60 = load_summary(args.dir60 / "noise_summary.csv", args.metric) if (args.dir60 / "noise_summary.csv").exists() else {}
    clean = s50.get(("clean", None), {})
    delta = args.metric.startswith("d_")

    panels: List[Tuple[str, Dict, str]] = [
        ("gaussian", s50, "Gaussian white noise"),
        ("baseline_wander", s50, "Baseline wander (0.1–0.5 Hz)"),
        ("powerline", s50, "Powerline interference (50 Hz)"),
        ("powerline", s60, "Powerline interference (60 Hz)"),
        ("muscle", s50, "Muscle artifact (20–150 Hz)"),
        ("composite", s50, "Composite (all four)"),
    ]
    panels = [p for p in panels if any(k[0] == p[0] for k in p[1])]
    if not panels:
        print("no data found")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(1, min(args.ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.0 * nrows), sharey=True, dpi=300, squeeze=False)
    letters = "abcdefghij"
    for k, (nt, src, title) in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        snrs = sorted({key[1] for key in src if key[0] == nt and key[1] is not None}, reverse=True)
        xs = list(range(len(snrs) + (0 if delta else 1)))
        for m in MODEL_NAMES:
            pts = ([] if delta else [clean.get(m)]) + [src.get((nt, s), {}).get(m) for s in snrs]
            if all(p is None for p in pts):
                continue
            mean = np.array([p[0] if p else np.nan for p in pts])
            sd = np.array([p[1] if p else 0.0 for p in pts])
            st = MODEL_STYLE[m]
            ax.fill_between(xs, mean - sd, mean + sd, color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=st["color"], marker=st["marker"], markersize=4, linewidth=1.6,
                    markeredgecolor="white", markeredgewidth=0.7, label=MODEL_DISPLAY[m])
        if delta:
            ax.axhline(0, color="#9a9a96", linewidth=0.8, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(([] if delta else ["clean"]) + [f"{s:g}" for s in snrs], fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.text(-0.02, 1.07, f"({letters[k]})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
        if k // ncols == nrows - 1 or k + ncols >= len(panels):
            ax.set_xlabel("SNR (dB)", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel(f"{METRIC_LABEL[args.metric]} (mean ± SD, 5 seeds)", fontsize=8)
        if k == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    for k in range(len(panels), nrows * ncols):          # hide unused cells
        axes[k // ncols][k % ncols].set_visible(False)
    fig.tight_layout()
    out = args.out or (args.dir50 / f"fig_noise_robustness{'' if args.metric == 'emr' else '_' + args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] {out}  ({len(panels)} panels, {nrows}x{ncols})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
