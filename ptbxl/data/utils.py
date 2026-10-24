"""Pure data utilities — no heavy ML dependencies required at import time."""

from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from ptbxl.config import CLASS_MAP, DiagnosticLookup, PreprocessFn, PipelineConfig, Record


def parse_scp_codes(value: Union[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Parse an scp_codes value that may be a string repr or already a mapping."""
    if isinstance(value, Mapping):
        return value
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("scp_codes must parse to a mapping")
    return parsed


def _diagnostic_class(entry: Union[str, Mapping[str, Any], Any]) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        value = entry.get("diagnostic_class")
        return str(value) if value is not None else None
    value = getattr(entry, "diagnostic_class", None)
    return str(value) if value is not None else None


def aggregate_diagnostic(
    scp_codes: Union[str, Mapping[str, Any]],
    diagnostic_lookup: DiagnosticLookup,
) -> list[str]:
    """Map SCP codes to sorted diagnostic superclass labels using a lookup table."""
    labels: set[str] = set()
    for code in parse_scp_codes(scp_codes):
        if code in diagnostic_lookup:
            label = _diagnostic_class(diagnostic_lookup[code])
            if label:
                labels.add(label)
    return sorted(labels)


def labels_to_multihot(
    labels: Iterable[str],
    class_map: Mapping[str, int] = CLASS_MAP,
) -> list[int]:
    """Convert diagnostic label strings to a fixed-length multi-hot vector."""
    target = [0] * len(class_map)
    for label in labels:
        if label not in class_map:
            raise KeyError(f"unknown diagnostic label: {label}")
        target[class_map[label]] = 1
    return target


def split_by_folds(
    records: Iterable[Record],
    train_folds: Iterable[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    test_folds: Iterable[int] = (10,),
) -> tuple[list[Record], list[Record]]:
    """Partition records by strat_fold into train and test lists."""
    train_set = set(train_folds)
    test_set = set(test_folds)
    train: list[Record] = []
    test: list[Record] = []
    for record in records:
        fold = int(record["strat_fold"])
        if fold in train_set:
            train.append(record)
        elif fold in test_set:
            test.append(record)
    return train, test


def minmax_scale(signal: Sequence[float]) -> list[float]:
    """Min-max normalize a 1-D signal to [0, 1]; returns zeros for constant signals."""
    if not signal:
        return []
    values = [float(v) for v in signal]
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def window_signal(
    signal: Sequence[float],
    window_size: int,
    stride: Optional[int] = None,
) -> list[list[float]]:
    """Slide a window over a signal and return a list of fixed-length windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be positive")
    values = list(signal)
    return [values[i : i + window_size] for i in range(0, len(values) - window_size + 1, stride)]


def preprocess_signal(
    signal: Sequence[float],
    preprocessors: Optional[Sequence[PreprocessFn]] = None,
) -> list[float]:
    """Apply a sequence of preprocessing functions to a signal in order."""
    processed: Sequence[float] = signal
    for fn in preprocessors or ():
        processed = fn(processed)
    return [float(v) for v in processed]


def build_targets(
    records: Iterable[Record],
    diagnostic_lookup: DiagnosticLookup,
    class_map: Mapping[str, int] = CLASS_MAP,
) -> list[dict[str, Any]]:
    """Annotate records with diagnostic_superclass and multi-hot target vectors."""
    rows: list[dict[str, Any]] = []
    for record in records:
        labels = aggregate_diagnostic(record["scp_codes"], diagnostic_lookup)
        rows.append({
            **dict(record),
            "diagnostic_superclass": labels,
            "target": labels_to_multihot(labels, class_map),
        })
    return rows


def _apply_preprocessors(
    row: Record,
    preprocessors: Sequence[PreprocessFn],
    config: PipelineConfig,
) -> dict[str, Any]:
    if "signal" not in row:
        return dict(row)
    processed = preprocess_signal(row["signal"], preprocessors)
    result = {**dict(row), "signal": processed}
    if config.window_size:
        result["windows"] = window_signal(processed, config.window_size, config.stride)
    return result


def run_data_pipeline(
    records: Iterable[Record],
    diagnostic_lookup: DiagnosticLookup,
    config: Optional[PipelineConfig] = None,
    preprocessors: Optional[Sequence[PreprocessFn]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pure data pipeline: label extraction, fold split, optional signal preprocessing.

    Does not require PTB-XL files or ML libraries at runtime.
    """
    cfg = config or PipelineConfig()
    rows = build_targets(records, diagnostic_lookup, cfg.class_map)
    train, test = split_by_folds(rows, cfg.train_folds, cfg.test_folds)
    if preprocessors:
        train = [_apply_preprocessors(row, preprocessors, cfg) for row in train]
        test = [_apply_preprocessors(row, preprocessors, cfg) for row in test]
    return {"train": train, "test": test}
