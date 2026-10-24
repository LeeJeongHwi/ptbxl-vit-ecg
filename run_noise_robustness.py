#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Robustness of the trained models to synthetic noise added to the test signals.

No model is retrained. Every checkpoint written by run_multiseed.py
    results/multiseed/<model>/seed_<s>/best_<model>_seed<s>.pth      (4 architectures x 5 seeds)
is evaluated on the held-out PTB-XL test fold (fold 10) after synthetic noise has been added
to the preprocessed signals at controlled signal-to-noise ratios. Each model uses the
classification threshold selected on its own clean validation fold (stored in result.json);
the threshold is not re-tuned on noisy data.

Noise types (fs = 500 Hz, 10 s records, 12 leads):
    gaussian         white Gaussian noise
    baseline_wander  low-frequency sinusoid, frequency ~ U(0.1, 0.5) Hz, random phase per record and lead
    powerline        mains interference, 50 Hz sinusoid (PTB-XL was recorded in Germany; 60 Hz with --powerline-hz 60)
    muscle           white noise band-limited to 20-150 Hz (mask in the discrete Fourier domain)
    composite        the four types above, each contributing one quarter of the total noise power
The SNR (dB) is defined per record and lead from the AC power (variance) of the clean lead:
    P_noise = Var(x_lead) / 10^(SNR/10)
The noise of each (noise type, SNR) condition is generated once with a fixed random seed, so
that every checkpoint is evaluated on the identical noisy test set.

Metrics: EMR, Hamming loss, per-class F1 and macro-F1 (same functions as in the clean
evaluation) and the degradation Delta = metric(noisy) - metric(clean) of the same seed,
aggregated as mean +- sample SD over seeds; the degradation of the ViT is compared with that
of each baseline by Welch's t-test on the per-seed values.

Usage (from the repository root, virtual environment activated):
    python run_noise_robustness.py --smoke-test     # 200 test records, 10 dB only
    python run_noise_robustness.py                  # 5 noise types x 5 SNRs x 20 checkpoints
    python run_noise_robustness.py --powerline-hz 60 --noise-types powerline --out-dir results/noise_robustness_60hz
    python run_noise_robustness.py --aggregate-only

Outputs (default --out-dir results/noise_robustness):
    noise_raw.json            every (model, seed, noise type, SNR) evaluation
    noise_summary.csv         long format: model, noise type, SNR, metric, n, mean, std, min, max
    noise_per_seed.csv        one row per (model, seed, condition)
    noise_table.md            EMR and macro-F1 (mean +- SD) and Delta-EMR per noise type and SNR
    noise_stats.md            Delta-EMR of the ViT versus each baseline per condition (Welch's t-test)
    noise_curves.png / .pdf   EMR versus SNR, one panel per noise type
    runs.log
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ptbxl.config import LABEL_NAMES, VIT_CONFIG, PipelineConfig          # noqa: E402
from ptbxl.data.cache import load_or_build_arrays                           # noqa: E402
from ptbxl.evaluation.metrics import confusion_metrics, evaluate_model      # noqa: E402
from ptbxl.model.registry import build_model as registry_build_model, build_vit   # noqa: E402
from ptbxl.training.trainer import make_dataloader                          # noqa: E402
from run_multiseed import (                                                 # noqa: E402
    MODEL_DISPLAY, MODEL_NAMES, REGISTRY_NAME, _welch, _cohen_d, _stats, fmt_ms,
)

LABELS: List[str] = list(LABEL_NAMES)
NOISE_TYPES: List[str] = ["gaussian", "baseline_wander", "powerline", "muscle", "composite"]
NOISE_DISPLAY = {
    "gaussian": "Gaussian white noise",
    "baseline_wander": "Baseline wander (0.1–0.5 Hz)",
    "powerline": "Powerline interference (50 Hz)",
    "muscle": "Muscle artifact (20–150 Hz)",
    "composite": "Composite (all four)",
}
DEFAULT_SNR_DB: List[float] = [30.0, 20.0, 10.0, 5.0, 0.0]
CLEAN = "clean"           # label used for the noise-free reference condition

# categorical colours (colour-blind safe) with a marker as a second encoding
MODEL_STYLE = {
    "vit_depth6":    dict(color="#2a78d6", marker="o"),
    "resnet1d":      dict(color="#eb6834", marker="s"),
    "bilstm":        dict(color="#1baf7a", marker="^"),
    "inceptiontime": dict(color="#eda100", marker="D"),
}


# ---- noise generation ----
def _scale_to_snr(x: np.ndarray, n: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise n so that, per record and lead, Var(n) = Var(x) / 10^(snr/10)."""
    p_sig = x.var(axis=-1, keepdims=True)                      # (N, 12, 1) AC power of the clean lead
    p_noise_target = p_sig / (10.0 ** (snr_db / 10.0))
    p_n = n.var(axis=-1, keepdims=True)
    scale = np.sqrt(p_noise_target / np.maximum(p_n, 1e-12))
    scale[p_sig <= 0] = 0.0                                    # flat lead -> no noise
    return (n * scale).astype(np.float32)


def _sinusoid(rng: np.random.Generator, shape: Tuple[int, int, int], fs: float,
              f_lo: float, f_hi: float) -> np.ndarray:
    N, C, L = shape
    t = (np.arange(L, dtype=np.float32) / fs)[None, None, :]
    f = rng.uniform(f_lo, f_hi, size=(N, C, 1)).astype(np.float32)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, C, 1)).astype(np.float32)
    return np.sin(2.0 * np.pi * f * t + phi, dtype=np.float32)


def _bandlimited_white(rng: np.random.Generator, shape: Tuple[int, int, int], fs: float,
                       f_lo: float, f_hi: float, chunk: int = 256) -> np.ndarray:
    """White noise band-limited to [f_lo, f_hi] by a mask in the discrete Fourier domain."""
    N, C, L = shape
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    mask = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float32)
    out = np.empty(shape, dtype=np.float32)
    for s in range(0, N, chunk):
        w = rng.standard_normal((min(chunk, N - s), C, L), dtype=np.float32)
        spec = np.fft.rfft(w, axis=-1) * mask
        out[s:s + w.shape[0]] = np.fft.irfft(spec, n=L, axis=-1).astype(np.float32)
    return out


def make_noise(kind: str, x: np.ndarray, snr_db: float, fs: float, powerline_hz: float,
               rng: np.random.Generator) -> np.ndarray:
    """Return noise (same shape as x) already scaled to the requested per-lead SNR."""
    shape = x.shape
    if kind == "gaussian":
        n = rng.standard_normal(shape, dtype=np.float32)
    elif kind == "baseline_wander":
        n = _sinusoid(rng, shape, fs, 0.1, 0.5)
    elif kind == "powerline":
        n = _sinusoid(rng, shape, fs, powerline_hz, powerline_hz)
    elif kind == "muscle":
        n = _bandlimited_white(rng, shape, fs, 20.0, 150.0)
    elif kind == "composite":
        # each component carries 1/4 of the total noise power -> +6.02 dB per component
        parts = [make_noise(k, x, snr_db + 10.0 * math.log10(4.0), fs, powerline_hz, rng)
                 for k in ("gaussian", "baseline_wander", "powerline", "muscle")]
        return np.sum(parts, axis=0, dtype=np.float32)
    else:
        raise ValueError(f"unknown noise type {kind!r}")
    return _scale_to_snr(x, n, snr_db)


def realised_snr_db(x: np.ndarray, n: np.ndarray) -> float:
    """Mean realised SNR over records and leads (written to the log as a check)."""
    p_sig = x.var(axis=-1)
    p_n = n.var(axis=-1)
    ok = (p_sig > 0) & (p_n > 0)
    return float(np.mean(10.0 * np.log10(p_sig[ok] / p_n[ok])))


# ---- checkpoints ----
def discover_checkpoints(ckpt_root: Path, models: Sequence[str], seeds: Optional[Sequence[int]]
                         ) -> List[Dict[str, Any]]:
    found = []
    for m in models:
        for rdir in sorted((ckpt_root / m).glob("seed_*")):
            seed = int(rdir.name.split("_")[1])
            if seeds and seed not in seeds:
                continue
            res = rdir / "result.json"
            ckpt = rdir / f"best_{m}_seed{seed}.pth"
            if not (res.exists() and ckpt.exists()):
                print(f"[warn] incomplete run dir skipped: {rdir}")
                continue
            r = json.loads(res.read_text(encoding="utf-8"))
            found.append(dict(model=m, seed=seed, ckpt=ckpt,
                              threshold=float(r["best_threshold_on_validation"]),
                              clean_ref=r["test_metrics_at_best_threshold"]))
    return found


def load_model(model_name: str, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = build_vit(VIT_CONFIG) if model_name == "vit_depth6" else registry_build_model(REGISTRY_NAME[model_name])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


# ---- evaluation ----
def evaluate(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, thr: float, device: torch.device,
             batch_size: int, num_workers: int) -> Dict[str, Any]:
    loader = make_dataloader(x, y, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    tm = evaluate_model(model, loader, thr, device)
    per_class = []
    for lab, cm in zip(LABELS, tm["confusion_matrix"]):
        m = {k: float(v) for k, v in confusion_metrics(np.asarray(cm)).items()}
        m["label"] = lab
        per_class.append(m)
    f1s = {f"f1_{pc['label']}": pc["f1"] for pc in per_class}
    return dict(emr=float(tm["exact_match_ratio"]), hamming=float(tm["hamming_loss"]),
                macro_f1=float(np.mean([pc["f1"] for pc in per_class])), **f1s,
                per_class=per_class,
                confusion_matrix=[[[int(v) for v in row] for row in np.asarray(cm).tolist()]
                                  for cm in tm["confusion_matrix"]])


def cond_key(noise_type: str, snr: Any) -> str:
    return f"{noise_type}@{snr}"


# ---- aggregation ----
METRICS = ["emr", "hamming", "macro_f1"] + [f"f1_{lab}" for lab in LABELS]


def aggregate(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    """raw rows -> per (model, condition) stats incl. deltas vs the clean condition of the same seed."""
    clean = {(r["model"], r["seed"]): r for r in raw if r["noise_type"] == CLEAN}
    rows = []
    for r in raw:
        row = dict(r)
        ref = clean.get((r["model"], r["seed"]))
        for k in METRICS:
            row[f"d_{k}"] = (r[k] - ref[k]) if ref else float("nan")
        rows.append(row)
    conds = []
    for r in rows:
        c = (r["noise_type"], r["snr_db"])
        if c not in conds:
            conds.append(c)
    # order: clean first, then by NOISE_TYPES order and decreasing SNR
    conds.sort(key=lambda c: (0 if c[0] == CLEAN else 1 + NOISE_TYPES.index(c[0]),
                              -(c[1] if isinstance(c[1], (int, float)) else 1e9)))
    models = [m for m in MODEL_NAMES if any(r["model"] == m for r in rows)]
    table: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m in models:
        table[m] = {}
        for nt, snr in conds:
            sub = [r for r in rows if r["model"] == m and r["noise_type"] == nt and r["snr_db"] == snr]
            if not sub:
                continue
            entry = {"n": len(sub), "seeds": sorted(r["seed"] for r in sub)}
            for k in METRICS:
                entry[k] = _stats([r[k] for r in sub])
                entry[f"d_{k}"] = _stats([r[f"d_{k}"] for r in sub if not math.isnan(r[f"d_{k}"])])
            table[m][cond_key(nt, snr)] = entry
    return dict(models=models, conditions=[list(c) for c in conds], table=table, rows=rows)


def write_outputs(summary: Dict[str, Any], raw: List[Dict[str, Any]], out_dir: Path, log) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "noise_raw.json").write_text(json.dumps(raw, indent=1, default=float), encoding="utf-8")

    with (out_dir / "noise_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "display_name", "noise_type", "snr_db", "metric", "n", "mean", "std", "min", "max"])
        for m in summary["models"]:
            for nt, snr in summary["conditions"]:
                e = summary["table"][m].get(cond_key(nt, snr))
                if not e:
                    continue
                for k in METRICS + [f"d_{k}" for k in METRICS]:
                    s = e[k]
                    w.writerow([m, MODEL_DISPLAY[m], nt, snr, k, s["n"], f"{s['mean']:.6f}", f"{s['std']:.6f}",
                                f"{s['min']:.6f}", f"{s['max']:.6f}"])

    # per-seed table
    rows = summary["rows"]
    if rows:
        keys = ["model", "seed", "noise_type", "snr_db", "realised_snr_db", "threshold"] + METRICS + [f"d_{k}" for k in METRICS]
        with (out_dir / "noise_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # ---- markdown tables ----
    n_seeds = max((e["n"] for m in summary["models"] for e in summary["table"][m].values()), default=0)
    lines = [f"**Noise robustness.** Test-set EMR (mean ± SD over {n_seeds} seeds) under synthetic noise at "
             f"decreasing SNR. Threshold per seed = the clean-validation-selected threshold (not re-tuned). "
             f"ΔEMR = EMR(noisy) − EMR(clean) of the same seed.\n"]
    noise_types_present = []
    for c in summary["conditions"]:
        if c[0] != CLEAN and c[0] not in noise_types_present:
            noise_types_present.append(c[0])
    for nt in noise_types_present:
        snrs = [c[1] for c in summary["conditions"] if c[0] == nt]
        lines.append(f"\n**{NOISE_DISPLAY[nt]}** — EMR (ΔEMR vs clean)\n")
        hdr = ["Model", "Clean"] + [f"{s:g} dB" for s in snrs]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for m in summary["models"]:
            t = summary["table"][m]
            cl = t.get(cond_key(CLEAN, None))
            cells = [MODEL_DISPLAY[m], fmt_ms(cl["emr"], 4) if cl else "–"]
            for s in snrs:
                e = t.get(cond_key(nt, s))
                cells.append(f"{fmt_ms(e['emr'], 4)} ({e['d_emr']['mean']:+.4f})" if e else "–")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(f"\n**{NOISE_DISPLAY[nt]}** — macro-F1 / HYP F1\n")
        hdr = ["Model", "Clean macro-F1"] + [f"{s:g} dB macro-F1" for s in snrs] + ["Clean HYP F1"] + [f"{s:g} dB HYP F1" for s in snrs]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        for m in summary["models"]:
            t = summary["table"][m]
            cl = t.get(cond_key(CLEAN, None))
            cells = [MODEL_DISPLAY[m], fmt_ms(cl["macro_f1"], 3) if cl else "–"]
            cells += [fmt_ms(t[cond_key(nt, s)]["macro_f1"], 3) if cond_key(nt, s) in t else "–" for s in snrs]
            cells += [fmt_ms(cl["f1_HYP"], 3) if cl else "–"]
            cells += [fmt_ms(t[cond_key(nt, s)]["f1_HYP"], 3) if cond_key(nt, s) in t else "–" for s in snrs]
            lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "noise_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- degradation of the ViT versus each baseline ----
    st = ["**Degradation comparison — Proposed ViT vs. each baseline** (per-seed ΔEMR = EMR(noisy) − EMR(clean); "
          "Welch's two-sided t-test on ΔEMR; positive Δ(ViT − baseline) = ViT degrades less. With five seeds "
          "per architecture these tests have low power and are reported descriptively.)\n",
          "| Noise | SNR | Baseline | ViT ΔEMR | Baseline ΔEMR | Δ(ViT − baseline) | Cohen's d | p |",
          "|---|---|---|---|---|---|---|---|"]
    rows = summary["rows"]
    if "vit_depth6" in summary["models"]:
        for nt, snr in summary["conditions"]:
            if nt == CLEAN:
                continue
            a = [r["d_emr"] for r in rows if r["model"] == "vit_depth6" and r["noise_type"] == nt and r["snr_db"] == snr]
            for m in summary["models"]:
                if m == "vit_depth6":
                    continue
                b = [r["d_emr"] for r in rows if r["model"] == m and r["noise_type"] == nt and r["snr_db"] == snr]
                if not a or not b:
                    continue
                t, p = _welch(a, b)
                d = _cohen_d(a, b)
                st.append("| " + " | ".join([
                    NOISE_DISPLAY[nt], f"{snr:g} dB", MODEL_DISPLAY[m], fmt_ms(_stats(a), 4), fmt_ms(_stats(b), 4),
                    f"{np.mean(a) - np.mean(b):+.4f}", f"{d:+.2f}" if not math.isnan(d) else "–",
                    f"{p:.3f}" if not math.isnan(p) else "–"]) + " |")
    (out_dir / "noise_stats.md").write_text("\n".join(st) + "\n", encoding="utf-8")

    try:
        plot_curves(summary, out_dir)
        log(f"[plot] {out_dir / 'noise_curves.png'}")
    except Exception as e:  # plotting is optional
        log(f"[plot] skipped ({type(e).__name__}: {e})")


def plot_curves(summary: Dict[str, Any], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = [nt for nt in NOISE_TYPES if any(c[0] == nt for c in summary["conditions"])]
    if not types:
        return
    fig, axes = plt.subplots(1, len(types), figsize=(3.4 * len(types), 3.4), sharey=True, dpi=150)
    axes = np.atleast_1d(axes)
    for ax, nt in zip(axes, types):
        snrs = sorted({c[1] for c in summary["conditions"] if c[0] == nt}, reverse=True)
        xs = list(range(len(snrs) + 1))                       # 0 = clean, then decreasing SNR
        for m in summary["models"]:
            t = summary["table"][m]
            pts = [t.get(cond_key(CLEAN, None))] + [t.get(cond_key(nt, s)) for s in snrs]
            mean = np.array([p["emr"]["mean"] if p else np.nan for p in pts])
            sd = np.array([p["emr"]["std"] if p else 0.0 for p in pts])
            style = MODEL_STYLE[m]
            ax.fill_between(xs, mean - sd, mean + sd, color=style["color"], alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=style["color"], marker=style["marker"], markersize=4.5, linewidth=1.8,
                    label=MODEL_DISPLAY[m], markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["clean"] + [f"{s:g}" for s in snrs], fontsize=8)
        ax.set_xlabel("SNR (dB)", fontsize=9)
        ax.set_title(NOISE_DISPLAY[nt], fontsize=9)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Test EMR (mean ± SD over seeds)", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_curves.png")
    fig.savefig(out_dir / "noise_curves.pdf")
    plt.close(fig)


# ---- cli / main ----
def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("ptb_xl_dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    p.add_argument("--ckpt-root", type=Path, default=Path("results/multiseed"))
    p.add_argument("--out-dir", default="results/noise_robustness")
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="default: every seed found under --ckpt-root")
    p.add_argument("--noise-types", nargs="+", default=NOISE_TYPES, choices=NOISE_TYPES)
    p.add_argument("--snr-db", nargs="+", type=float, default=DEFAULT_SNR_DB)
    p.add_argument("--noise-seed", type=int, default=2026, help="RNG seed for noise generation (fixed across models)")
    p.add_argument("--powerline-hz", type=float, default=50.0)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="re-evaluate conditions already in noise_raw.json")
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="first 200 test records, SNR 10 dB only, 2 noise types -> results/noise_robustness_smoke")
    args = p.parse_args(argv)
    if args.smoke_test:
        if args.out_dir == "results/noise_robustness":
            args.out_dir = "results/noise_robustness_smoke"
        args.snr_db = [10.0]
        args.noise_types = ["gaussian", "composite"] if args.noise_types == NOISE_TYPES else args.noise_types[:2]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    NOISE_DISPLAY["powerline"] = f"Powerline interference ({args.powerline_hz:g} Hz)"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "noise_raw.json"
    log_path = out_dir / "runs.log"

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    raw: List[Dict[str, Any]] = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []

    if not args.aggregate_only:
        device = torch.device(args.device)
        log(f"\n##### run_noise_robustness.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  device={device}  "
            f"types={args.noise_types}  snr={args.snr_db}  noise_seed={args.noise_seed}  "
            f"powerline_hz={args.powerline_hz:g}" + ("  [SMOKE TEST]" if args.smoke_test else ""))

        ckpts = discover_checkpoints(args.ckpt_root, args.models, args.seeds)
        if not ckpts:
            log(f"[error] no checkpoints under {args.ckpt_root}")
            return 1
        log(f"checkpoints: {len(ckpts)}  -> " + ", ".join(f"{c['model']}:s{c['seed']}" for c in ckpts))

        config = PipelineConfig(data_root=args.data_root, cache_dir=args.cache_dir, num_workers=args.num_workers)
        fs = float(config.sampling_rate)
        log("Loading cached arrays...")
        _, _, _, _, x_test, y_test = load_or_build_arrays(config)
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)
        if args.smoke_test:
            x_test, y_test = x_test[:200], y_test[:200]
        log(f"  test {x_test.shape}  fs={fs:g} Hz")

        log("Loading models...")
        models = {(c["model"], c["seed"]): load_model(c["model"], c["ckpt"], device) for c in ckpts}

        done = {(r["model"], r["seed"], r["noise_type"], r["snr_db"]) for r in raw}
        conditions: List[Tuple[str, Any]] = [(CLEAN, None)] + [(nt, s) for nt in args.noise_types for s in args.snr_db]
        t_all = time.time()
        for ci, (nt, snr) in enumerate(conditions, 1):
            pending = [c for c in ckpts if args.force or (c["model"], c["seed"], nt, snr) not in done]
            if not pending:
                log(f"[{ci}/{len(conditions)}] {nt} {snr}: all checkpoints done, skipped")
                continue
            if nt == CLEAN:
                x_cond, rsnr = x_test, float("inf")
            else:
                rng = np.random.default_rng([args.noise_seed, NOISE_TYPES.index(nt), int(round(snr * 100))])
                noise = make_noise(nt, x_test, snr, fs, args.powerline_hz, rng)
                rsnr = realised_snr_db(x_test, noise)
                x_cond = (x_test + noise).astype(np.float32)
                del noise
            log(f"\n[{ci}/{len(conditions)}] {nt} @ {snr} dB  (realised {rsnr:.2f} dB)  elapsed {(time.time() - t_all) / 60:.1f} min")
            for c in pending:
                m = models[(c["model"], c["seed"])]
                res = evaluate(m, x_cond, y_test, c["threshold"], device, args.eval_batch_size, args.num_workers)
                row = dict(model=c["model"], seed=c["seed"], noise_type=nt, snr_db=snr, realised_snr_db=rsnr,
                           threshold=c["threshold"],
                           powerline_hz=(args.powerline_hz if nt == "powerline" else None), **res)
                raw = [r for r in raw if (r["model"], r["seed"], r["noise_type"], r["snr_db"]) !=
                       (c["model"], c["seed"], nt, snr)]
                raw.append(row)
                note = ""
                if nt == CLEAN and not args.smoke_test:
                    ref = c["clean_ref"]["exact_match_ratio"]
                    note = "  [matches result.json]" if abs(ref - res["emr"]) < 1e-9 else f"  [!! result.json EMR {ref:.4f}]"
                log(f"    {c['model']:14s} s{c['seed']}  EMR {res['emr']:.4f}  HL {res['hamming']:.4f}  "
                    f"macroF1 {res['macro_f1']:.3f}  HYP {res['f1_HYP']:.3f}{note}")
            raw_path.write_text(json.dumps(raw, indent=1, default=float), encoding="utf-8")
            del x_cond
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not raw:
        log("[aggregate] nothing to aggregate")
        return 1
    # label the powerline condition with the frequency actually used (stored per row; raw files
    # without the field fall back to --powerline-hz, so pass it with --aggregate-only when it differs)
    hz_in_rows = {r.get("powerline_hz") for r in raw if r["noise_type"] == "powerline"} - {None}
    if hz_in_rows:
        NOISE_DISPLAY["powerline"] = f"Powerline interference ({', '.join(f'{h:g}' for h in sorted(hz_in_rows))} Hz)"
    summary = aggregate(raw)
    write_outputs(summary, raw, out_dir, log)
    log("\n" + (out_dir / "noise_table.md").read_text(encoding="utf-8"))
    log(f"[aggregate] {len(raw)} evaluations -> {out_dir}/noise_summary.csv, noise_per_seed.csv, "
        f"noise_table.md, noise_stats.md, noise_curves.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
