"""Multi-seed comparison of the proposed Vision Transformer with the three baselines.

Each architecture (ViT, ResNet1D, BiLSTM, InceptionTime) is trained with several random
seeds under the identical protocol used throughout: folds 1-8 / 9 / 10 of PTB-XL, AdamW
(learning rate 1e-4, weight decay 0.05), cosine annealing, asymmetric loss, batch size 16,
up to 70 epochs with early stopping (patience 10), the classification threshold selected
on the validation fold by maximising the exact match ratio (50 thresholds in [0, 1]) and a
single evaluation of the test fold at that threshold. Test-set EMR, Hamming loss and
per-class F1 are then aggregated over seeds as mean +- sample standard deviation, and the
ViT is compared with each baseline by Welch's t-test on the per-seed values.

Usage (from the repository root, virtual environment activated):
    python run_multiseed.py --smoke-test        # quick end-to-end check on a small subset
    python run_multiseed.py                     # 4 architectures x seeds 42-46
    python run_multiseed.py --seeds 42 43 44
    python run_multiseed.py --models vit_depth6 bilstm
    python run_multiseed.py --aggregate-only    # rebuild the tables from existing result.json files

Outputs (default --out-dir results/multiseed):
    <model>/seed_<s>/best_<model>_seed<s>.pth   checkpoint with the lowest validation loss
    <model>/seed_<s>/result.json                metrics of that run
    <model>/seed_<s>/val_sweep.csv              validation threshold sweep
    multiseed_raw.json, multiseed_summary.json/.csv, multiseed_per_seed.csv,
    multiseed_table.md, multiseed_stats.md, runs.log

A (model, seed) pair whose result.json already exists is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats

from ptbxl.config import LABEL_NAMES, VIT_CONFIG, PipelineConfig
from ptbxl.data.cache import load_or_build_arrays
from ptbxl.evaluation.metrics import confusion_metrics, evaluate_model, threshold_sweep
from ptbxl.model.registry import build_model as registry_build_model, build_vit, count_parameters
from ptbxl.training.trainer import make_dataloader, set_seed, train_loop

LABELS: List[str] = list(LABEL_NAMES)

PROTOCOL = dict(
    epochs=70,
    patience=10,
    batch_size=16,
    eval_batch_size=32,
    lr=1e-4,
    weight_decay=0.05,
    num_thresholds=50,
    loss="AsymmetricLoss (gamma_neg 2, gamma_pos 0, clip 0.05)",
    scheduler="CosineAnnealingLR(T_max=epochs, eta_min=1e-6)",
    early_stopping="validation loss, strict improvement",
)

MODEL_NAMES: List[str] = ["vit_depth6", "resnet1d", "bilstm", "inceptiontime"]
REGISTRY_NAME = {"resnet1d": "resnet1d", "bilstm": "bilstm", "inceptiontime": "inception"}
MODEL_DISPLAY = {
    "vit_depth6": "Proposed ViT (depth = 6)",
    "resnet1d": "ResNet1D",
    "bilstm": "BiLSTM",
    "inceptiontime": "InceptionTime",
}
DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------------
# seeding and model construction
# ---------------------------------------------------------------------------
def seed_everything(seed: int, strict: bool = False) -> None:
    """Seed python, numpy and torch and make cuDNN deterministic."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        # full determinism; some kernels (e.g. the cuDNN LSTM backward pass) do not support it
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def build_model(name: str) -> torch.nn.Module:
    if name == "vit_depth6":
        return build_vit(VIT_CONFIG)
    if name in REGISTRY_NAME:
        return registry_build_model(REGISTRY_NAME[name])
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


# ---------------------------------------------------------------------------
# one (model, seed) run
# ---------------------------------------------------------------------------
def run_one(model_name: str, seed: int, arrays: tuple, args, device: torch.device, log) -> Dict[str, Any]:
    run_dir = Path(args.out_dir) / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        log(f"[skip] {model_name} seed={seed} (result.json exists; use --force to redo)")
        return json.loads(result_path.read_text(encoding="utf-8"))

    log(f"\n=== {model_name}  seed={seed} ===")
    x_train, y_train, x_val, y_val, x_test, y_test = arrays

    # seed, then loaders, then model, so that every run consumes the RNG in the same order
    seed_everything(seed, strict=args.strict_deterministic)
    train_loader = make_dataloader(x_train, y_train, batch_size=args.batch_size,
                                   num_workers=args.num_workers, shuffle=True)
    val_loader = make_dataloader(x_val, y_val, batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers, shuffle=False)
    test_loader = make_dataloader(x_test, y_test, batch_size=args.eval_batch_size,
                                  num_workers=args.num_workers, shuffle=False)
    model = build_model(model_name).to(device)
    n_params = count_parameters(model)
    log(f"    trainable params: {n_params:,}")

    tag = f"{model_name}_seed{seed}"
    t0 = time.time()
    trained = train_loop(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, checkpoint=run_dir / f"best_{tag}.pth", desc=tag,
    )
    elapsed_min = (time.time() - t0) / 60.0
    model.load_state_dict(torch.load(trained.checkpoint, map_location=device))
    model.eval()
    log(f"    best val loss {trained.best_loss:.4f} @ epoch {trained.best_epoch}  ({elapsed_min:.1f} min)")

    # threshold selected on the validation fold
    thresholds = np.linspace(0, 1, args.num_thresholds)
    val_df = threshold_sweep(model, val_loader, thresholds, device)
    val_df.to_csv(run_dir / "val_sweep.csv", index=False)
    val_emr = val_df.groupby("threshold")["exact_match_ratio"].first()
    best_threshold = float(val_emr.idxmax())
    best_val_emr = float(val_emr.max())

    # single evaluation of the test fold
    test_metrics = evaluate_model(model, test_loader, best_threshold, device)
    per_class = []
    for label, cm in zip(LABELS, test_metrics["confusion_matrix"]):
        m = {k: float(v) for k, v in confusion_metrics(np.asarray(cm)).items()}
        m["label"] = label
        per_class.append(m)
    cms_int = [[[int(v) for v in row] for row in np.asarray(cm).tolist()]
               for cm in test_metrics["confusion_matrix"]]
    val_metrics = evaluate_model(model, val_loader, best_threshold, device)

    result = {
        "name": model_name,
        "display_name": MODEL_DISPLAY[model_name],
        "seed": seed,
        "config": (dict(VIT_CONFIG) if model_name == "vit_depth6" else {"registry": REGISTRY_NAME[model_name]}),
        "protocol": {k: v for k, v in PROTOCOL.items()},
        "trainable_params": n_params,
        "training_time_min": elapsed_min,
        "best_epoch": trained.best_epoch,
        "best_val_loss": float(trained.best_loss),
        "best_threshold_on_validation": best_threshold,
        "validation_emr_at_best_threshold": best_val_emr,
        "validation_hamming_at_best_threshold": float(val_metrics["hamming_loss"]),
        "test_metrics_at_best_threshold": {
            "threshold": best_threshold,
            "exact_match_ratio": float(test_metrics["exact_match_ratio"]),
            "hamming_loss": float(test_metrics["hamming_loss"]),
            "per_class": per_class,
            "confusion_matrix": cms_int,
        },
        "meta": {
            "torch": torch.__version__, "device": str(device),
            "strict_deterministic": args.strict_deterministic, "smoke_test": args.smoke_test,
            "n_train": int(len(x_train)), "n_val": int(len(x_val)), "n_test": int(len(x_test)),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    if args.no_save_ckpt:
        trained.checkpoint.unlink(missing_ok=True)

    f1s = "  ".join(f"{pc['label']} {pc['f1']:.3f}" for pc in per_class)
    log(f"    -> thr {best_threshold:.4f} | val EMR {best_val_emr:.4f} | "
        f"test EMR {test_metrics['exact_match_ratio']:.4f}  HL {test_metrics['hamming_loss']:.4f} | {f1s}")

    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# aggregation over seeds
# ---------------------------------------------------------------------------
METRIC_KEYS: List[Tuple[str, str]] = [("test_emr", "Test EMR"), ("test_hamming", "Hamming Loss")] + \
                                     [(f"f1_{lab}", f"{lab} F1") for lab in LABELS]


def flatten_result(r: Dict[str, Any]) -> Dict[str, Any]:
    tm = r["test_metrics_at_best_threshold"]
    flat: Dict[str, Any] = dict(
        model=r["name"], seed=r["seed"], params=r["trainable_params"],
        best_thr=r["best_threshold_on_validation"], val_emr=r["validation_emr_at_best_threshold"],
        best_epoch=r["best_epoch"], best_val_loss=r["best_val_loss"],
        test_emr=tm["exact_match_ratio"], test_hamming=tm["hamming_loss"],
    )
    for pc in tm["per_class"]:
        flat[f"f1_{pc['label']}"] = pc["f1"]
        flat[f"precision_{pc['label']}"] = pc["precision"]
        flat[f"recall_{pc['label']}"] = pc["recall"]
    return flat


def load_all_results(out_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for p in sorted(out_dir.glob("*/seed_*/result.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    return results


def _stats(vals: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=float)
    n = int(a.size)
    return dict(n=n, mean=float(a.mean()) if n else float("nan"),
                std=float(a.std(ddof=1)) if n > 1 else 0.0,
                min=float(a.min()) if n else float("nan"), max=float(a.max()) if n else float("nan"))


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [flatten_result(r) for r in results]
    models = [m for m in MODEL_NAMES if any(r["model"] == m for r in rows)]
    summary: Dict[str, Any] = {"std_definition": "sample std (ddof=1) over seeds", "models": {}}
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        seeds = sorted(r["seed"] for r in mrows)
        entry: Dict[str, Any] = dict(display_name=MODEL_DISPLAY[m], seeds=seeds, n_seeds=len(seeds),
                                     params=int(np.median([r["params"] for r in mrows])),
                                     best_thresholds=[r["best_thr"] for r in mrows], metrics={})
        keys = ["val_emr", "test_emr", "test_hamming"] + \
               [f"{k}_{lab}" for lab in LABELS for k in ("f1", "precision", "recall")] + \
               ["best_thr", "best_epoch", "best_val_loss"]
        for key in keys:
            entry["metrics"][key] = _stats([r[key] for r in mrows])
        summary["models"][m] = entry
    summary["rows"] = rows
    return summary


def fmt_ms(s: Dict[str, Any], nd: int = 3) -> str:
    if s["n"] == 0 or math.isnan(s["mean"]):
        return "–"
    return f"{s['mean']:.{nd}f} ± {s['std']:.{nd}f}" if s["n"] > 1 else f"{s['mean']:.{nd}f}"


def _best_model(summary: Dict[str, Any], key: str) -> str:
    lower_is_better = key == "test_hamming"
    items = [(m, e["metrics"][key]["mean"]) for m, e in summary["models"].items()
             if not math.isnan(e["metrics"][key]["mean"])]
    if not items:
        return "–"
    return (min if lower_is_better else max)(items, key=lambda t: t[1])[0]


def _welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's two-sided t-test; returns (t, p), or NaNs when fewer than two values per group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def _cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def stats_report(summary: Dict[str, Any]) -> str:
    rows = summary["rows"]
    if "vit_depth6" not in summary["models"]:
        return "ViT (vit_depth6) results not present; no comparison computed.\n"
    out = ["**Seed-level comparison of the proposed ViT with each baseline** "
           "(Welch's two-sided t-test on the per-seed test metrics; Cohen's d with pooled SD; "
           "positive Δ = ViT higher). With five seeds per architecture these tests have low power "
           "and are reported descriptively.\n"]
    hdr = ["Baseline", "Metric", "ViT mean ± SD", "Baseline mean ± SD", "Δ (ViT − baseline)", "Cohen's d", "t", "p"]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    vit = [r for r in rows if r["model"] == "vit_depth6"]
    for m in summary["models"]:
        if m == "vit_depth6":
            continue
        base = [r for r in rows if r["model"] == m]
        for key, pretty in METRIC_KEYS:
            a, b = [r[key] for r in vit], [r[key] for r in base]
            t, p = _welch(a, b)
            d = _cohen_d(a, b)
            out.append("| " + " | ".join([
                MODEL_DISPLAY[m], pretty, fmt_ms(_stats(a), 4), fmt_ms(_stats(b), 4),
                f"{np.mean(a) - np.mean(b):+.4f}", f"{d:+.2f}" if not math.isnan(d) else "–",
                f"{t:+.2f}" if not math.isnan(t) else "–", f"{p:.3f}" if not math.isnan(p) else "–"]) + " |")
    return "\n".join(out) + "\n"


def write_outputs(summary: Dict[str, Any], results: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_raw.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (out_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    with (out_dir / "multiseed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "display_name", "metric", "n", "mean", "std", "min", "max"])
        for m, e in summary["models"].items():
            for k, s in e["metrics"].items():
                w.writerow([m, e["display_name"], k, s["n"], f"{s['mean']:.6f}", f"{s['std']:.6f}",
                            f"{s['min']:.6f}", f"{s['max']:.6f}"])

    rows = summary["rows"]
    if rows:
        with (out_dir / "multiseed_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in sorted(rows, key=lambda r: (MODEL_NAMES.index(r["model"]), r["seed"])):
                w.writerow(r)

    seeds_all = sorted({s for e in summary["models"].values() for s in e["seeds"]})
    lines = [f"**Test-set performance over {len(seeds_all)} random seeds "
             f"({', '.join(map(str, seeds_all))}); mean ± sample SD.** For every seed the classification "
             f"threshold was selected on the validation set (50 values in [0, 1], maximum EMR) and the "
             f"test set was evaluated once at that threshold.\n"]
    header = ["Model", "Params", "n seeds", "Best Thr. (mean, range)", "Val EMR"] + [p for _, p in METRIC_KEYS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for m, e in summary["models"].items():
        mt = e["metrics"]
        thr = mt["best_thr"]
        thr_txt = f"{thr['mean']:.4f}" + (f" ({thr['min']:.4f}–{thr['max']:.4f})" if thr["min"] != thr["max"] else "")
        row = [e["display_name"], f"{e['params'] / 1e6:.2f} M", str(e["n_seeds"]), thr_txt,
               fmt_ms(mt["val_emr"], 4), fmt_ms(mt["test_emr"], 4), fmt_ms(mt["test_hamming"], 4)] + \
              [fmt_ms(mt[f"f1_{lab}"], 3) for lab in LABELS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Best mean per column: " + "; ".join(
        f"{pretty} → {MODEL_DISPLAY.get(_best_model(summary, key), '–')}" for key, pretty in METRIC_KEYS))
    (out_dir / "multiseed_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "multiseed_stats.md").write_text(stats_report(summary), encoding="utf-8")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("ptb_xl_dataset"))
    p.add_argument("--cache-dir", type=Path, default=Path("npy_data"))
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--out-dir", default="results/multiseed")
    p.add_argument("--epochs", type=int, default=PROTOCOL["epochs"])
    p.add_argument("--patience", type=int, default=PROTOCOL["patience"])
    p.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size"])
    p.add_argument("--eval-batch-size", type=int, default=PROTOCOL["eval_batch_size"])
    p.add_argument("--lr", type=float, default=PROTOCOL["lr"])
    p.add_argument("--weight-decay", type=float, default=PROTOCOL["weight_decay"])
    p.add_argument("--num-thresholds", type=int, default=PROTOCOL["num_thresholds"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strict-deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms(True); may fail for the cuDNN LSTM")
    p.add_argument("--force", action="store_true", help="re-run (model, seed) even if result.json exists")
    p.add_argument("--no-save-ckpt", action="store_true", help="delete the per-seed checkpoint after evaluation")
    p.add_argument("--aggregate-only", action="store_true", help="skip training; rebuild tables from result.json")
    p.add_argument("--smoke-test", action="store_true",
                   help="real pipeline on a small subset: 2 models x 2 seeds x 2 epochs -> results/multiseed_smoke")
    args = p.parse_args(argv)
    if args.smoke_test:
        if args.out_dir == "results/multiseed":
            args.out_dir = "results/multiseed_smoke"
        args.epochs, args.patience = min(args.epochs, 2), 2
        args.seeds = args.seeds[:2]
        args.models = ["vit_depth6", "bilstm"] if args.models == MODEL_NAMES else args.models[:2]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runs.log"

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    if not args.aggregate_only:
        device = torch.device(args.device)
        log(f"\n##### run_multiseed.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  device={device}  "
            f"models={args.models}  seeds={args.seeds}  epochs={args.epochs}  patience={args.patience}"
            + ("  [SMOKE TEST]" if args.smoke_test else ""))
        if not args.smoke_test and (args.epochs != PROTOCOL["epochs"] or args.patience != PROTOCOL["patience"]
                                    or args.batch_size != PROTOCOL["batch_size"]
                                    or args.weight_decay != PROTOCOL["weight_decay"]):
            log("[warn] epochs/patience/batch/weight-decay differ from the reference protocol")

        config = PipelineConfig(data_root=args.data_root, cache_dir=args.cache_dir, num_workers=args.num_workers)
        log("Loading cached arrays...")
        arrays = load_or_build_arrays(config)
        if args.smoke_test:
            xtr, ytr, xva, yva, xte, yte = arrays
            arrays = (xtr[:256], ytr[:256], xva[:96], yva[:96], xte[:96], yte[:96])
        log(f"  train {arrays[0].shape}  val {arrays[2].shape}  test {arrays[4].shape}")

        total, k = len(args.models) * len(args.seeds), 0
        t_all = time.time()
        for seed in args.seeds:            # seed-major: a partial run still yields all models for each seed
            for m in args.models:
                k += 1
                log(f"\n[{k}/{total}]  elapsed {(time.time() - t_all) / 60:.0f} min")
                try:
                    run_one(m, seed, arrays, args, device, log)
                except Exception as e:     # log and continue; the failed run can be re-launched later
                    log(f"[error] {m} seed={seed}: {type(e).__name__}: {e}")
                    if args.smoke_test:
                        raise
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    results = load_all_results(out_dir)
    if not results:
        log("[aggregate] no result.json files found")
        return 1
    summary = aggregate(results)
    write_outputs(summary, results, out_dir)
    log("\n" + (out_dir / "multiseed_table.md").read_text(encoding="utf-8"))
    log(f"[aggregate] {len(results)} runs -> {out_dir}/multiseed_summary.json, .csv, "
        f"multiseed_per_seed.csv, multiseed_table.md, multiseed_stats.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
