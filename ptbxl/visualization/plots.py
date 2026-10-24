"""Visualization utilities: loss curves, confusion heatmap, attention overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ptbxl.config import LABEL_NAMES
from ptbxl.model.vit import ViT


def plot_losses(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_dir: Path,
) -> Path:
    """Save a train/validation loss curve and return its path."""
    path = output_dir / "Train_loss.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="validation")
    ax.set_title("Train / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_heatmap(
    confusions: Sequence[Sequence[Sequence[int]]],
    output_dir: Path,
    label_names: Sequence[str] = LABEL_NAMES,
) -> Path:
    """Save per-class multi-label confusion matrices as a single figure."""
    path = output_dir / "cfmt.png"
    fig = plt.figure(figsize=(8, 8))
    for idx, (label, conf) in enumerate(zip(label_names, confusions), start=1):
        c = np.asarray(conf)
        tn, fp = c[0]
        fn, tp = c[1]
        ax = fig.add_subplot(3, 2, idx)
        sns.heatmap(
            [[int(tp), int(fn)], [int(fp), int(tn)]],
            annot=True,
            cmap="Blues",
            fmt="g",
            ax=ax,
        )
        ax.set_title(f"{label} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Attention overlays
# ---------------------------------------------------------------------------
# Each patch token summarises one window of `patch_size` samples of all leads, so the
# class-token attention is a function of time only. It is drawn as uniform bands of one
# patch width (no interpolation), identical in every lead, with the colour scale of each
# map normalised to its own minimum and maximum.

def extract_attention(model: ViT, signal: torch.Tensor, device: torch.device) -> np.ndarray:
    """Class-token attention of the final encoder block to the patch tokens.

    signal: tensor of shape (leads, samples). Returns an array (heads, n_patches); the
    attention from the class token to itself is dropped.
    """
    model.eval()
    with torch.no_grad():
        attns = model.get_last_selfattention(signal.unsqueeze(0).to(device))   # (1, heads, seq, seq)
    return attns[0, :, 0, 1:].detach().cpu().numpy()


def attention_style(
    print_width_cm: float = 18.0,
    label_pt: float = 9.0,
    tick_pt: float = 8.0,
    line_pt: float = 0.5,
    time_axis: bool = False,
    fig_width_in: float = 20.0,
) -> dict:
    """Font and line sizes for the attention figures.

    The figures are drawn `fig_width_in` inches wide and reduced to `print_width_cm` in
    the document, so the sizes given in printed points are multiplied by the reduction
    factor before drawing.
    """
    scale = fig_width_in / (print_width_cm / 2.54)
    return dict(scale=scale, label=label_pt * scale, tick=tick_pt * scale, lw=line_pt * scale,
                frame=0.3 * scale, time_axis=time_axis, fig_width=fig_width_in)


def _draw_row(ax, sig: np.ndarray, attn_patch: np.ndarray, label: str, style: dict, patch_size: int,
              fs: int, rpeaks: Optional[np.ndarray] = None, grid: bool = False,
              label_fs: Optional[float] = None, time_axis: bool = False) -> None:
    L = len(sig)
    heat = np.repeat(attn_patch, patch_size)[np.newaxis, :]
    lo, hi = float(sig.min()), float(sig.max())
    ax.imshow(heat, aspect="auto", cmap="Reds", alpha=0.6, extent=(0, L, lo, hi),
              interpolation="nearest", vmin=float(attn_patch.min()), vmax=float(attn_patch.max()), zorder=0)
    ax.plot(np.arange(L), sig, linewidth=style["lw"], color=(0, 0, 0.7), zorder=2)
    if grid:
        for p in range(0, L + 1, patch_size * 5):
            ax.axvline(p, color="gray", lw=0.2 * style["scale"], alpha=0.5, zorder=1)
        if rpeaks is not None:
            for r in rpeaks:
                ax.axvline(r, color="green", lw=0.5 * style["scale"], ls="--", zorder=1)
    ax.set_xlim(0, L)
    ax.set_ylim(lo, hi)
    fsz = style["label"] if label_fs is None else label_fs
    # horizontal lead label: one text line high, so it fits even in a low row
    ax.set_ylabel(label, fontsize=fsz, rotation=0, ha="right", va="center", labelpad=6 * style["scale"])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(style["frame"])
    if time_axis:
        ax.set_xticks(np.arange(0, L + 1, fs))
        ax.set_xticklabels([str(t) for t in range(0, L // fs + 1)])
        ax.tick_params(axis="x", labelsize=min(style["tick"], fsz), length=4 * style["scale"], width=style["frame"])
        ax.set_xlabel("Time (s)", fontsize=fsz)
    else:
        ax.set_xticks([])


def _fit_label(out: Path, row_heights_in: Sequence[float], style: dict) -> float:
    """Shrink the lead label so that it fits into the lowest row; returns the font size used."""
    fsz = style["label"]
    min_row_pt = min(row_heights_in) * 72.0
    if fsz > 0.9 * min_row_pt:
        fsz = 0.9 * min_row_pt
        print(f"  [note] {out.name}: label reduced from {style['label']:.0f} to {fsz:.0f} pt "
              f"(about {fsz / style['scale']:.1f} pt when printed); lowest row {min(row_heights_in):.2f} in")
    return fsz


def plot_attention_leads(ecg: np.ndarray, attn_patch: np.ndarray, lead_idx: Sequence[int],
                         lead_names: Sequence[str], out: Path, style: dict, *, patch_size: int = 20,
                         fs: int = 500, title: Optional[str] = None, rpeaks: Optional[np.ndarray] = None,
                         grid: bool = False, dpi: int = 600) -> Path:
    """Overlay one attention vector on a subset of leads (one row per lead).

    ecg: array (samples, leads); attn_patch: array (n_patches,). Row heights are
    proportional to the peak-to-peak amplitude of each lead.
    """
    n = len(lead_idx)
    amps = [float(np.ptp(ecg[:, i])) for i in lead_idx]
    ratios = [a / amps[0] if amps[0] > 0 else 1.0 for a in amps]
    height = 15 if n == 12 else 1.4 * n + 0.8
    fig, axs = plt.subplots(n, 1, figsize=(style["fig_width"], height), gridspec_kw={"height_ratios": ratios},
                            squeeze=False)
    fig.subplots_adjust(hspace=0.0)
    label_fs = _fit_label(out, [height * r / sum(ratios) for r in ratios], style)
    for k, (ax, i) in enumerate(zip(axs[:, 0], lead_idx)):
        _draw_row(ax, ecg[:, i], attn_patch, lead_names[i], style, patch_size, fs, rpeaks, grid,
                  label_fs=label_fs, time_axis=style["time_axis"] and k == n - 1)
    if title:
        fig.suptitle(title, fontsize=12 * style["scale"], y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_attention_heads_grid(panels: Sequence[tuple[str, np.ndarray, np.ndarray]], out: Path, style: dict, *,
                              patch_size: int = 20, row_in: float = 0.9, dpi: int = 400) -> Path:
    """Several per-head panels stacked in one figure.

    panels: sequence of (title, lead signal (samples,), cls_to_patch (heads, n_patches)).
    Each panel shows one row per head plus their mean, each row with its own colour scale,
    and is labelled (a), (b), ... followed by its title.
    """
    n_rows = panels[0][2].shape[0] + 1
    gap = 0.9
    height = len(panels) * n_rows * row_in + (len(panels) - 1) * gap + 0.6
    fig = plt.figure(figsize=(style["fig_width"], height))
    outer = fig.add_gridspec(len(panels), 1, hspace=gap / (n_rows * row_in), top=1 - 0.5 / height, bottom=0.1 / height)
    for p, (title, sig, cls_to_patch) in enumerate(panels):
        inner = outer[p].subgridspec(n_rows, 1, hspace=0.0)
        rows = [(f"Head {h + 1}", cls_to_patch[h]) for h in range(cls_to_patch.shape[0])] + \
               [("Mean", cls_to_patch.mean(axis=0))]
        for k, (lab, vec) in enumerate(rows):
            ax = fig.add_subplot(inner[k])
            _draw_row(ax, sig, vec, lab, style, patch_size, fs=500)
            if k == 0:
                ax.set_title(f"({'abcdefghij'[p]})  {title}", loc="left", fontsize=style["label"],
                             pad=6 * style["scale"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_attention_heads(ecg: np.ndarray, cls_to_patch: np.ndarray, lead: int, lead_names: Sequence[str],
                         out: Path, style: dict, *, patch_size: int = 20, fs: int = 500,
                         title: Optional[str] = None, dpi: int = 600) -> Path:
    """One lead, one row per attention head plus their mean; each row uses its own colour scale."""
    H = cls_to_patch.shape[0]
    rows = [(f"Head {h + 1}", cls_to_patch[h]) for h in range(H)] + [("Mean", cls_to_patch.mean(axis=0))]
    fig, axs = plt.subplots(len(rows), 1, figsize=(style["fig_width"], 1.4 * len(rows) + 0.8), squeeze=False)
    fig.subplots_adjust(hspace=0.0)
    for k, (ax, (lab, vec)) in enumerate(zip(axs[:, 0], rows)):
        _draw_row(ax, ecg[:, lead], vec, lab, style, patch_size, fs,
                  time_axis=style["time_axis"] and k == len(rows) - 1)
    if style["time_axis"]:
        axs[0, 0].set_title(f"Lead {lead_names[lead]}", loc="left", fontsize=style["label"], pad=6 * style["scale"])
    else:
        axs[-1, 0].set_xlabel(f"lead {lead_names[lead]}", fontsize=style["label"])
    if title:
        fig.suptitle(title, fontsize=12 * style["scale"], y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
