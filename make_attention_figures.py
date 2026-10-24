"""Attention maps of the proposed Vision Transformer for representative test records.

For each illustrated diagnostic category one record of the held-out test fold is chosen by a
fixed rule: among the true positives of that category, the record with the highest predicted
probability for that category; for conduction disturbance the choice is restricted to records
annotated with complete left bundle branch block (CLBBB) and complete right bundle branch block
(CRBBB), respectively. For every chosen record the script

  1. re-computes EMR and Hamming loss of the whole test fold at the operating threshold and
     compares them with the stored result of the checkpoint,
  2. maps the test-set index to the PTB-XL ecg_id, SCP codes and clinical report,
  3. extracts the attention from the class token to the 250 patch tokens of the final encoder
     block (one value per head, no interpolation) and tabulates how concentrated each head is
     and where the highest-weighted 0.04-s windows fall relative to the nearest R peak
     (lead II; neurokit2 if installed, otherwise a scipy peak detector),
  4. renders the overlays: the leads in which the finding is conventionally read, all 12 leads,
     a diagnostic version with R peaks and a 0.2-s grid, one panel per head plus their mean,
     and a composite of the per-head panels of all records.

Usage (from the repository root, virtual environment activated):
    python make_attention_figures.py --data-root ptb_xl_dataset
    python make_attention_figures.py --data-root ptb_xl_dataset --fig-dir figures/attention
    python make_attention_figures.py --indices 1304 2062 2154 1362      # explicit test-set indices
    python make_attention_figures.py --mode head --heads 2 2 3 3        # one head per figure (0-based)
    python make_attention_figures.py --data-root ptb_xl_dataset --list-candidates CLBBB
    python make_attention_figures.py --no-figs

The figures are drawn 20 in wide and are meant to be printed at --print-width-cm; label, tick
and line sizes are given in printed points and scaled accordingly.

Outputs (default --out results/attention; figures in --fig-dir, default the same folder):
    attention_report.md / attention_report.json, attention_<idx>.npy (heads x 250),
    <Fig>_<tag>_<idx>_<display>_<leads>.png, <Supp>_<tag>_<idx>_<display>_12lead.png,
    diag_<Fig>_<tag>_<idx>_<display>.png, S5_<Fig>_allheads_<lead>.png, S5_allheads_composite.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

PATCH = 20                  # samples per patch (0.04 s at 500 Hz)
FS = 500
SIG_LEN = 5000
N_PATCH = SIG_LEN // PATCH  # 250
DEFAULT_THRESHOLD = 35 / 49  # 0.7143, selected on the validation fold for the proposed model

# (figure name, category used for the selection rule, SCP-code restriction, leads shown,
#  lead used for the per-head panel)
FIGURES = [
    dict(name="Fig6", category="MI",  scp=None,    leads=["V1", "V2", "V3", "V4"], heads_lead="V2"),
    dict(name="Fig7", category="CD",  scp="CLBBB", leads=["V5", "V6"],             heads_lead="V6"),
    dict(name="Fig8", category="CD",  scp="CRBBB", leads=["V1", "V2", "V3"],       heads_lead="V1"),
    dict(name="Fig9", category="HYP", scp=None,    leads=["V4", "V5", "V6"],       heads_lead="V5"),
]
SUPP = {"Fig6": "S1", "Fig7": "S2", "Fig8": "S3", "Fig9": "S4"}
SCP_SUFFIX = {"CLBBB": "LBBB", "CRBBB": "RBBB"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_data_root(cli_root: Optional[str], cfg_root: Path) -> Optional[Path]:
    candidates = []
    if cli_root:
        candidates.append(Path(cli_root))
    candidates += [Path(cfg_root), Path("ptb_xl_dataset"), Path("ptb-xl"), Path("data"), Path("data/ptb-xl")]
    for c in candidates:
        if (c / "ptbxl_database.csv").exists() and (c / "scp_statements.csv").exists():
            return c
    return None


def detect_rpeaks(sig: np.ndarray, fs: int = FS) -> tuple[np.ndarray, str]:
    """R-peak positions (samples) with neurokit2 when available, otherwise a scipy fallback."""
    try:
        import neurokit2 as nk  # type: ignore

        _, info = nk.ecg_peaks(sig.astype(float), sampling_rate=fs)
        r = np.asarray(info["ECG_R_Peaks"], dtype=int)
        if len(r) >= 3:
            return r, "neurokit2"
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] neurokit2 unavailable or failed ({exc}); using the scipy fallback (less reliable)")
    from scipy.signal import find_peaks

    a = np.abs(sig - np.median(sig))
    r, _ = find_peaks(a, distance=int(0.3 * fs), height=np.percentile(a, 98))
    return np.asarray(r, dtype=int), "scipy-fallback"


def classify_offset(ms: float) -> str:
    """Coarse ECG segment of a window centre from its offset to the nearest R peak (narrow-QRS heuristic)."""
    if -70 <= ms <= 70:
        return "QRS (within 70 ms of R)"
    if -220 <= ms < -70:
        return "PR / P wave"
    if 70 < ms <= 160:
        return "J-point / ST"
    if 160 < ms <= 420:
        return "T wave"
    return "TP / isoelectric"


def norm_entropy(a: np.ndarray) -> float:
    p = a / a.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(a)))


def select_record(probs: np.ndarray, y_true: np.ndarray, category: str, meta, scp: Optional[str],
                  min_likelihood: float, label_names: list[str]) -> tuple[int, int]:
    """Highest predicted probability among the true positives of `category` (optionally restricted
    to records carrying the SCP code `scp` with at least `min_likelihood`). Returns (index, n candidates)."""
    ci = label_names.index(category)
    mask = y_true[:, ci] == 1
    if scp:
        if meta is None:
            sys.exit(f"the {scp} restriction needs the PTB-XL metadata; pass --data-root or use --indices")
        has = np.array([(scp in codes) and float(codes[scp]) >= min_likelihood for codes in meta.scp_codes], dtype=bool)
        mask &= has
    idx = np.where(mask)[0]
    if len(idx) == 0:
        sys.exit(f"no candidate record for {category} ({scp})")
    best = idx[int(np.argmax(probs[idx, ci]))]
    return int(best), int(len(idx))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="folder with ptbxl_database.csv and scp_statements.csv")
    ap.add_argument("--cache-dir", default="npy_data")
    ap.add_argument("--checkpoint", default="models/ablation/depth_6/best_depth_6.pth")
    ap.add_argument("--reference", default="results/ablation/ablation_summary.json",
                    help="ablation summary whose depth_6 entry gives the expected test EMR / Hamming loss")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--indices", type=int, nargs=4, default=None,
                    help="test-set indices for the four figures (skips the selection rule)")
    ap.add_argument("--heads", type=int, nargs=4, default=None, help="head index per figure, 0-based (for --mode head)")
    ap.add_argument("--mode", choices=["mean", "head"], default="mean",
                    help="vector shown in the overlays: mean of the heads, or the single head given by --heads")
    ap.add_argument("--min-likelihood", type=float, default=50.0,
                    help="minimum SCP-code likelihood for the CLBBB / CRBBB restriction")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--rpeak-lead", default="II")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--out", default="results/attention")
    ap.add_argument("--fig-dir", default=None, help="folder for the figures (default: --out)")
    ap.add_argument("--list-candidates", default=None, metavar="SCP",
                    help="rank the test records carrying this SCP code by the predicted probability of its superclass and exit")
    ap.add_argument("--print-width-cm", type=float, default=18.0)
    ap.add_argument("--label-pt", type=float, default=9.0)
    ap.add_argument("--tick-pt", type=float, default=8.0)
    ap.add_argument("--line-pt", type=float, default=0.5)
    ap.add_argument("--time-axis", action="store_true", help="add a time axis (0-10 s) to the bottom row")
    ap.add_argument("--heads-row-in", type=float, default=0.9, help="row height (in) of the per-head composite")
    ap.add_argument("--heads-dpi", type=int, default=400, help="resolution of the per-head composite")
    ap.add_argument("--dpi", type=int, default=600, help="resolution of the single-record figures")
    args = ap.parse_args()

    import torch
    from ptbxl.config import LABEL_NAMES, LEAD_NAMES, PipelineConfig
    from ptbxl.data.cache import load_or_build_arrays
    from ptbxl.model.registry import build_vit
    from ptbxl.visualization.plots import (
        attention_style, extract_attention, plot_attention_heads, plot_attention_heads_grid, plot_attention_leads,
    )

    style = attention_style(args.print_width_cm, args.label_pt, args.tick_pt, args.line_pt, args.time_axis)
    print(f"[style] print width {args.print_width_cm} cm -> scale x{style['scale']:.2f}: label {style['label']:.0f} pt, "
          f"line {style['lw']:.2f} pt (figure units); time axis = {style['time_axis']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    data: dict[str, Any] = {"mode": args.mode, "threshold": args.threshold, "figures": {}}

    def say(line: str = "") -> None:
        print(line)
        report.append(line)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    say(f"# Attention report (display = {args.mode})\n\nDevice: {device}\n")

    # ---- data -------------------------------------------------------------
    cfg = PipelineConfig(data_root=Path(args.data_root) if args.data_root else Path("."), cache_dir=Path(args.cache_dir))
    _, _, _, _, x_te, y_te = load_or_build_arrays(cfg)
    say("## 1. Test arrays\n")
    say(f"- x_test shape = {x_te.shape}, y_test shape = {y_te.shape}")
    say(f"- x_test value range = [{x_te.min():.4f}, {x_te.max():.4f}] mV")
    assert x_te.shape[1:] == (12, SIG_LEN), "unexpected signal length or number of leads"

    # ---- model and whole-test-set check -----------------------------------
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    model = build_vit().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        xt = torch.from_numpy(x_te).float()
        for s in range(0, len(xt), 64):
            probs.append(model(xt[s:s + 64].to(device)).cpu().numpy())
    probs = np.concatenate(probs)
    pred = (probs >= args.threshold).astype(np.float32)
    emr = float((pred == y_te).all(axis=1).mean())
    hl = float((pred != y_te).mean())
    say(f"\n## 2. Whole test set at threshold {args.threshold:.4f}\n")
    say(f"- EMR = {emr:.4f}, Hamming loss = {hl:.4f}")
    ref_path = Path(args.reference)
    if ref_path.exists():
        try:
            ref = next(r for r in json.loads(ref_path.read_text(encoding="utf-8"))["results"] if r["name"] == "depth_6")
            tm = ref["test_metrics_at_best_threshold"]
            ok = abs(emr - tm["exact_match_ratio"]) < 1e-6 and abs(hl - tm["hamming_loss"]) < 1e-6
            say(f"- stored result of the depth_6 run: EMR = {tm['exact_match_ratio']:.4f}, Hamming loss = {tm['hamming_loss']:.4f} "
                f"-> {'match' if ok else 'MISMATCH (cached arrays or checkpoint differ from the stored run)'}")
            data["reference_match"] = ok
        except Exception as exc:  # noqa: BLE001
            say(f"- reference not usable ({exc})")
    data["whole_test_set"] = {"emr": emr, "hamming": hl}

    # ---- metadata mapping ------------------------------------------------
    meta = scp_df = None
    data_root = find_data_root(args.data_root, cfg.data_root)
    say("\n## 3. Test index to ecg_id mapping\n")
    if data_root is None:
        say("- ptbxl_database.csv not found; pass --data-root to report ecg_id and SCP codes")
    else:
        import pandas as pd
        from ptbxl.data.dataset import PTBXLDataset

        ds = PTBXLDataset(data_root, cfg.test_folds, sampling_rate=cfg.sampling_rate)
        meta = ds.y[ds.y.class_ids.map(len) > 0]           # same filter as collect_labeled_arrays
        scp_df = pd.read_csv(data_root / "scp_statements.csv", index_col=0)
        say(f"- fold-10 records = {len(ds.y)}, with at least one superclass label = {len(meta)}, x_test = {len(x_te)}")
        if len(meta) != len(x_te):
            say("- record counts differ; the mapping cannot be trusted and is not used")
            meta = None
        else:
            lab = np.zeros((len(meta), 5), dtype=np.float32)
            for i, ids in enumerate(meta.class_ids):
                lab[i, ids] = 1.0
            n_match = int((lab == y_te).all(axis=1).sum())
            say(f"- labels of the metadata match y_test for {n_match} of {len(meta)} records")
            if n_match != len(meta):
                meta = None

    # ---- optional candidate listing ----------------------------------------
    if args.list_candidates:
        if meta is None:
            sys.exit("--list-candidates requires the PTB-XL metadata (--data-root)")
        code = args.list_candidates.upper()
        if code not in scp_df.index:
            sys.exit(f"unknown SCP code: {code}")
        cls_name = str(scp_df.loc[code, "diagnostic_class"])
        ci = LABEL_NAMES.index(cls_name)
        cands = []
        for i in range(len(meta)):
            codes = meta.iloc[i].scp_codes
            if code in codes and float(codes[code]) >= args.min_likelihood and y_te[i, ci] == 1:
                exact = bool(((probs[i] >= args.threshold).astype(np.float32) == y_te[i]).all())
                cands.append((float(probs[i, ci]), i, exact))
        cands.sort(reverse=True)
        say(f"\n## Candidates with {code} (likelihood >= {args.min_likelihood:g}), true {cls_name}, sorted by P({cls_name})\n")
        say(f"- {len(cands)} candidates in the test set\n")
        say("| rank | test idx | ecg_id | P(class) | exact match | true labels | scp_codes (likelihood >= 50) | report |")
        say("|---|---|---|---|---|---|---|---|")
        for r, (pcls, i, exact) in enumerate(cands[:12], 1):
            row = meta.iloc[i]
            tl = [LABEL_NAMES[c] for c in range(5) if y_te[i, c] == 1]
            cd = ", ".join(f"{k}({v:g})" for k, v in row.scp_codes.items() if float(v) >= 50)
            say(f"| {r} | {i} | {row.name} | {pcls:.3f} | {'yes' if exact else 'no'} | {'+'.join(tl)} | {cd} | "
                f"{str(row.report)[:70]} |")
        (out_dir / f"candidates_{code}.md").write_text("\n".join(report), encoding="utf-8")
        print(f"\nSaved: {(out_dir / f'candidates_{code}.md').resolve()}")
        return

    # ---- record selection --------------------------------------------------
    say("\n## 4. Record selection\n")
    chosen: list[tuple[dict, int]] = []
    for k, spec in enumerate(FIGURES):
        if args.indices:
            idx, n_cand = int(args.indices[k]), None
            say(f"- {spec['name']}: test idx {idx} (given on the command line)")
        else:
            idx, n_cand = select_record(probs, y_te, spec["category"], meta, spec["scp"], args.min_likelihood, LABEL_NAMES)
            restr = f", restricted to {spec['scp']} (likelihood >= {args.min_likelihood:g})" if spec["scp"] else ""
            say(f"- {spec['name']}: highest P({spec['category']}) among the {n_cand} true-{spec['category']} records"
                f"{restr} -> test idx {idx}")
        chosen.append((spec, idx))

    # ---- per-record attention ---------------------------------------------
    rl = LEAD_NAMES.index(args.rpeak_lead)
    heads_panels = []
    for k, (spec, idx) in enumerate(chosen):
        fig_name = spec["name"]
        head = args.heads[k] if args.heads else None
        ecg = x_te[idx].T                                      # (5000, 12)
        y_true = y_te[idx]
        p = probs[idx]
        true_set = [LABEL_NAMES[c] for c in range(5) if y_true[c] == 1]
        pred_set = [LABEL_NAMES[c] for c in range(5) if p[c] >= args.threshold]
        tag = "_".join(true_set) + (f"_{SCP_SUFFIX.get(spec['scp'], spec['scp'])}" if spec["scp"] else "")
        say(f"\n## {fig_name} — test idx {idx}, {tag}, display = {args.mode}"
            + (f" (head {head + 1})" if args.mode == "head" and head is not None else "") + "\n")
        entry: dict[str, Any] = {"idx": idx, "tag": tag, "category": spec["category"], "scp": spec["scp"],
                                 "display": args.mode}
        say(f"- true labels: {true_set}  |  predicted at {args.threshold:.4f}: {pred_set}  |  "
            f"{'exact match' if true_set == pred_set else 'not an exact match'}")
        say("- probabilities: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(LABEL_NAMES, p)))
        entry.update(true=true_set, pred=pred_set, probs={n: float(v) for n, v in zip(LABEL_NAMES, p)})

        if meta is not None:
            row = meta.iloc[idx]
            say(f"- ecg_id = {row.name}  ({row.filename_hr})  | age {row.age}, sex {row.sex}")
            codes = row.scp_codes if isinstance(row.scp_codes, dict) else {}
            lines = []
            for c, lik in codes.items():
                if c in scp_df.index:
                    s = scp_df.loc[c]
                    lines.append(f"    - {c} (likelihood {lik}): class={s.diagnostic_class}, "
                                 f"subclass={s.diagnostic_subclass}, {s.description}")
                else:
                    lines.append(f"    - {c} (likelihood {lik}): not in scp_statements")
            say("- scp_codes:\n" + "\n".join(lines))
            say(f"- report: {row.report}")
            entry.update(ecg_id=int(row.name), scp_codes={k2: float(v) for k2, v in codes.items()}, report=str(row.report))

        cls_to_patch = extract_attention(model, torch.from_numpy(x_te[idx]).float(), device)   # (heads, 250)
        assert cls_to_patch.shape[1] == N_PATCH, f"{cls_to_patch.shape[1]} patch tokens, expected {N_PATCH}"
        np.save(out_dir / f"attention_{idx}.npy", cls_to_patch)

        rpeaks, method = detect_rpeaks(ecg[:, rl])
        if len(rpeaks) > 1:
            say(f"- R peaks on lead {args.rpeak_lead}: n = {len(rpeaks)} ({method}); mean RR = {np.mean(np.diff(rpeaks)) / FS:.3f} s")
        else:
            say("- R peaks: fewer than 2 found")

        def n_qrs(vec: np.ndarray) -> int:
            if not len(rpeaks):
                return -1
            cen = np.argsort(-vec)[:8] * PATCH + PATCH / 2
            offs = np.array([(c - rpeaks[np.argmin(np.abs(rpeaks - c))]) / FS * 1000 for c in cen])
            return int((np.abs(offs) <= 70).sum())

        say("\n| vector | max | max / mean | patches > 2x mean | top-8 share | norm. entropy | top-8 within 70 ms of R |")
        say("|---|---|---|---|---|---|---|")
        head_rows = []
        vecs = [(f"head {h + 1}", cls_to_patch[h]) for h in range(cls_to_patch.shape[0])]
        vecs.append(("mean of heads", cls_to_patch.mean(axis=0)))
        for name, v in vecs:
            mv = float(v.mean())
            sh = float(v[np.argsort(-v)[:8]].sum() / v.sum())
            en = norm_entropy(v)
            say(f"| {name} | {v.max():.4f} | {v.max() / mv:.1f} | {int((v > 2 * mv).sum())} | {sh:.1%} | {en:.3f} | {n_qrs(v)}/8 |")
            head_rows.append({"vector": name, "max": float(v.max()), "n_over_2x": int((v > 2 * mv).sum()),
                              "top8_share": sh, "entropy": en, "top8_qrs": n_qrs(v)})
        entry["head_table"] = head_rows

        if args.mode == "mean" or head is None:
            a = cls_to_patch.mean(axis=0)
            disp_label, disp_tag = "mean of the heads", "headmean"
        else:
            a = cls_to_patch[head]
            disp_label, disp_tag = f"Head {head + 1}", f"head{head + 1}"
        mean_w = float(a.mean())
        topk = np.argsort(-a)[: args.topk]
        say(f"\n- displayed vector = {disp_label}: mean patch weight = {mean_w:.4f}, max = {a.max():.4f} "
            f"({a.max() / mean_w:.1f}x mean); patches > 2x mean = {int((a > 2 * mean_w).sum())}/{N_PATCH}; "
            f"top-{args.topk} carry {a[topk].sum() / a.sum():.1%}")
        say("- note: the segment column assumes a narrow QRS; in bundle branch block an offset of +50 to +100 ms may "
            "still lie within the widened QRS, and with short RR intervals 'TP' may be the T wave of the preceding beat")
        say(f"\n| rank | patch | t (s) | weight | x mean | nearest R (ms) | segment (heuristic) |")
        say("|---|---|---|---|---|---|---|")
        rows = []
        for r, pch in enumerate(topk, 1):
            c = pch * PATCH + PATCH / 2
            if len(rpeaks):
                off = float((c - rpeaks[np.argmin(np.abs(rpeaks - c))]) / FS * 1000)
                region, off_s = classify_offset(off), f"{off:+.0f}"
            else:
                off, region, off_s = float("nan"), "n/a", "n/a"
            say(f"| {r} | {pch} | {pch * PATCH / FS:.2f}–{(pch + 1) * PATCH / FS:.2f} | {a[pch]:.4f} | "
                f"{a[pch] / mean_w:.1f} | {off_s} | {region} |")
            rows.append({"patch": int(pch), "t0": pch * PATCH / FS, "t1": (pch + 1) * PATCH / FS,
                         "weight": float(a[pch]), "x_mean": float(a[pch] / mean_w),
                         "nearest_R_ms": off, "segment": region})
        entry.update(top=rows, rpeaks=rpeaks.tolist(), rpeak_method=method)
        say("\n- per-head top-5 patches (t in s): " + "  |  ".join(
            f"head {h + 1}: " + ", ".join(f"{q * PATCH / FS:.2f}" for q in np.argsort(-cls_to_patch[h])[:5])
            for h in range(cls_to_patch.shape[0])))

        heads_lead = LEAD_NAMES.index(spec["heads_lead"])
        heads_panels.append((f"lead {spec['heads_lead']}  —  record of Figure {fig_name[3:]}", ecg[:, heads_lead], cls_to_patch))
        if not args.no_figs:
            leads = spec["leads"]
            lead_idx = [LEAD_NAMES.index(l) for l in leads]
            base = f"{tag}_{idx}_{disp_tag}"
            f1 = plot_attention_leads(ecg, a, lead_idx, LEAD_NAMES, fig_dir / f"{fig_name}_{base}_{leads[0]}-{leads[-1]}.png",
                                      style, dpi=args.dpi)
            f2 = plot_attention_leads(ecg, a, list(range(12)), LEAD_NAMES, fig_dir / f"{SUPP[fig_name]}_{base}_12lead.png",
                                      style, dpi=args.dpi)
            f3 = plot_attention_leads(ecg, a, lead_idx, LEAD_NAMES, out_dir / f"diag_{fig_name}_{base}.png", style,
                                      title=f"{fig_name} — idx {idx} — {disp_label} — green: R peaks ({method}), grid 0.2 s",
                                      rpeaks=rpeaks, grid=True, dpi=200)
            f4 = plot_attention_heads(ecg, cls_to_patch, heads_lead, LEAD_NAMES,
                                      fig_dir / f"S5_{fig_name}_allheads_{spec['heads_lead']}.png", style, dpi=args.dpi)
            say(f"\n- figures: {f1.name}, {f2.name}, {f3.name}, {f4.name}  (folder: {fig_dir})")
        data["figures"][fig_name] = entry

    if not args.no_figs and heads_panels:
        grid_style = dict(style)
        f5 = plot_attention_heads_grid(heads_panels, fig_dir / "S5_allheads_composite.png", grid_style,
                                       row_in=args.heads_row_in, dpi=args.heads_dpi)
        say(f"\n- composite of the per-head panels: {f5.name}  (folder: {fig_dir})")

    (out_dir / "attention_report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "attention_report.json").write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    print(f"\nReport: {(out_dir / 'attention_report.md').resolve()}")


if __name__ == "__main__":
    main()
