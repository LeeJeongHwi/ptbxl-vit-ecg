#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figure: one test record, one lead, clean and with every synthetic noise type at a given SNR.

The noise realisation is exactly the one evaluated by run_noise_robustness.py (same seeding),
so the figure shows the signals the models actually received. Rows: clean, Gaussian, baseline
wander, powerline (one row per --powerline-hz), muscle artifact, composite; all rows share the
same vertical scale. Amplitudes are in mV, the units of the preprocessed signals.

Usage (from the repository root, virtual environment activated):
    python plot_noise_examples.py                       # record 0, lead II, 0 dB, first 5 s
    python plot_noise_examples.py --index 17 --snr-db 5 --t0 2 --dur 4

Outputs: <out>.png and <out>.pdf (default results/noise_robustness/noise_examples_<SNR>dB.png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptbxl.config import LABEL_NAMES, LEAD_NAMES, PipelineConfig          # noqa: E402
from ptbxl.data.cache import load_or_build_arrays                          # noqa: E402
from run_noise_robustness import NOISE_TYPES, make_noise                   # noqa: E402

ROW_TITLE = {
    "gaussian": "Gaussian white noise",
    "baseline_wander": "Baseline wander (0.1–0.5 Hz)",
    "muscle": "Muscle artifact (20–150 Hz)",
    "composite": "Composite (all four)",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("ptb_xl_dataset"))
    ap.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    ap.add_argument("--index", type=int, default=0, help="test-set record index (fold 10 order)")
    ap.add_argument("--lead", default="II", help="lead name (I, II, ..., V6) or 0-based index")
    ap.add_argument("--snr-db", type=float, default=0.0)
    ap.add_argument("--noise-seed", type=int, default=2026, help="must match run_noise_robustness.py")
    ap.add_argument("--powerline-hz", type=float, nargs="+", default=[50.0, 60.0])
    ap.add_argument("--t0", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--dur", type=float, default=5.0, help="window length (s)")
    ap.add_argument("--units", default="mV", help="y-axis unit label (the preprocessed signals are in mV)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    config = PipelineConfig(data_root=args.data_root, cache_dir=args.cache_dir, num_workers=0)
    fs = float(config.sampling_rate)
    _, _, _, _, x_test, y_test = load_or_build_arrays(config)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    lead = int(args.lead) if str(args.lead).isdigit() else LEAD_NAMES.index(args.lead)
    i = args.index
    rec = x_test[i]
    labels = [LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if y_test[i][j] > 0.5] or ["no label"]
    print(f"record #{i}  labels={labels}  lead={LEAD_NAMES[lead]}  SNR={args.snr_db:g} dB")

    rows = [("Clean", rec[lead].copy(), None)]
    for nt in NOISE_TYPES:
        hz_list = args.powerline_hz if nt == "powerline" else [50.0]
        for hz in hz_list:
            # same seeding as run_noise_robustness.main, hence the same noise realisation
            rng = np.random.default_rng([args.noise_seed, NOISE_TYPES.index(nt), int(round(args.snr_db * 100))])
            noise = make_noise(nt, x_test, args.snr_db, fs, hz, rng)
            n = noise[i, lead]
            del noise
            snr_rec = 10.0 * np.log10(rec[lead].var() / max(n.var(), 1e-12))
            title = f"Powerline interference ({hz:g} Hz)" if nt == "powerline" else ROW_TITLE[nt]
            rows.append((title, rec[lead] + n, snr_rec))
            print(f"  {title:32s} realised SNR for this record/lead: {snr_rec:6.2f} dB")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s0, s1 = int(args.t0 * fs), int(min((args.t0 + args.dur) * fs, rec.shape[-1]))
    t = np.arange(s0, s1) / fs
    ymax = max(np.abs(sig[s0:s1]).max() for _, sig, _ in rows) * 1.05
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7.2, 1.05 * n_rows + 0.7), sharex=True, sharey=True, dpi=200)
    for ax, (title, sig, snr_rec) in zip(np.atleast_1d(axes), rows):
        ax.plot(t, sig[s0:s1], color="#1f1f1e", linewidth=0.6)
        tag = title if snr_rec is None else f"{title}  —  SNR {args.snr_db:g} dB"
        ax.text(0.005, 0.95, tag, transform=ax.transAxes, fontsize=8, va="top", ha="left",
                color="#0b0b0b", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_ylim(-ymax, ymax)
        ax.set_yticks([-round(ymax * 0.6, 1), 0, round(ymax * 0.6, 1)])
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.supylabel(f"Lead {LEAD_NAMES[lead]} amplitude ({args.units})", fontsize=8)
    fig.suptitle(f"Test record #{i} ({', '.join(labels)}), lead {LEAD_NAMES[lead]}: clean vs. synthetic noise at "
                 f"{args.snr_db:g} dB SNR", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = args.out or Path("results/noise_robustness") / f"noise_examples_{args.snr_db:g}dB.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[plot] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
