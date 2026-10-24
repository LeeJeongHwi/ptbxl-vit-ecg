"""Evaluation metrics: per-threshold predictions, classification report, threshold sweep."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    multilabel_confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ptbxl.config import LABEL_NAMES


def safe_divide(numerator: float, denominator: float) -> float:
    """Division with zero-denominator guard; returns 0.0 when denominator is 0."""
    return float(numerator / denominator) if denominator else 0.0


def confusion_metrics(conf: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 from a 2×2 confusion matrix."""
    tn, fp = float(conf[0, 0]), float(conf[0, 1])
    fn, tp = float(conf[1, 0]), float(conf[1, 1])
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "accuracy": safe_divide(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
    }


def predict_labels(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and threshold sigmoid probabilities to binary predictions."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Predict (threshold={threshold:.2f})"):
            probs = model(x.to(device)).cpu().numpy()
            preds.extend((probs > threshold).astype(int))
            targets.extend(y.cpu().numpy().astype(int))
    return np.asarray(preds), np.asarray(targets)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
    label_names: Sequence[str] = LABEL_NAMES,
) -> dict[str, Any]:
    """Print and return classification metrics at the given threshold."""
    predictions, targets = predict_labels(model, loader, threshold, device)
    confusion = multilabel_confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=list(label_names), zero_division=0)
    print(report)
    return {
        "threshold": threshold,
        "exact_match_ratio": float(accuracy_score(targets, predictions)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
    }


def threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
    label_names: Sequence[str] = LABEL_NAMES,
) -> pd.DataFrame:
    """Evaluate per-class metrics across many thresholds and return a tidy DataFrame."""
    rows = []
    for threshold in thresholds:
        predictions, targets = predict_labels(model, loader, float(threshold), device)
        for label, conf in zip(label_names, multilabel_confusion_matrix(targets, predictions)):
            row: dict[str, Any] = {"threshold": float(threshold), "label": label}
            row.update(confusion_metrics(conf))
            row["hamming_loss"] = float(hamming_loss(targets, predictions))
            row["exact_match_ratio"] = float(accuracy_score(targets, predictions))
            rows.append(row)
    return pd.DataFrame(rows)
