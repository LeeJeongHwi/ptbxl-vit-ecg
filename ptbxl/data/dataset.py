"""PTB-XL dataset loader and array collection utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb
from scipy.signal import butter, lfilter, medfilt
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import pywt
except ImportError:
    pywt = None

import pandas as pd

from ptbxl.config import CLASS_MAP


class PTBXLDataset(Dataset):
    """PTB-XL metadata and WFDB signal loader with multi-label encoding."""

    def __init__(
        self,
        data_root: Path | str,
        folds: Sequence[int],
        class_map: dict[str, int] | None = None,
        sampling_rate: int = 500,
        verbose: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.folds = tuple(folds)
        self.class_map = class_map or CLASS_MAP
        self.sampling_rate = sampling_rate
        self.verbose = verbose

        y = pd.read_csv(self.data_root / "ptbxl_database.csv", index_col="ecg_id")
        y = y.loc[y.strat_fold.isin(self.folds)].copy()
        agg_df = pd.read_csv(self.data_root / "scp_statements.csv", index_col=0)
        self.agg_df = agg_df[agg_df.diagnostic == 1]
        y.scp_codes = y.scp_codes.apply(ast.literal_eval)
        y["diagnostic_superclass"] = y.scp_codes.apply(self._aggregate_diagnostic)
        y["class_ids"] = y.diagnostic_superclass.apply(self._map_class_ids)
        self.y = y

        if self.verbose:
            print("unique superclasses =", self.agg_df.diagnostic_class.unique())
            print("unique folds =", self.y.strat_fold.unique())

    def _aggregate_diagnostic(self, y_dic: dict[str, float]) -> list[str]:
        return sorted({
            self.agg_df.loc[key].diagnostic_class
            for key in y_dic
            if key in self.agg_df.index
        })

    def _map_class_ids(self, class_labels: Iterable[str]) -> list[int]:
        return [self.class_map[label] for label in class_labels if label in self.class_map]

    def _read_signal(self, data_path: Path) -> np.ndarray:
        signal, _ = wfdb.rdsamp(str(data_path))
        return np.asarray(signal)

    def _remove_baseline(self, ecg: np.ndarray) -> np.ndarray:
        """Baseline wander removal with two cascaded median filters, applied per lead.

        The kernel lengths are 0.3 s and 0.7 s rounded up to the next odd number of
        samples (151 and 351 samples at 500 Hz); the estimated baseline is subtracted.
        """
        k1 = int(0.3 * self.sampling_rate) | 1  # ensure odd
        k2 = int(0.7 * self.sampling_rate) | 1
        out = np.empty_like(ecg)
        for i in range(ecg.shape[1]):
            baseline = medfilt(medfilt(ecg[:, i], k1), k2)
            out[:, i] = ecg[:, i] - baseline
        return out

    @staticmethod
    def lowpass_filter(fs: int, low: float, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Build a low-pass Butterworth filter for powerline noise removal."""
        return butter(order, low / (0.5 * fs), btype="low")

    @staticmethod
    def highpass_filter(fs: int, high: float, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Build a high-pass Butterworth filter for baseline drift removal."""
        return butter(order, high / (0.5 * fs), btype="high")

    def powerline_noise_filtering(self, data: np.ndarray, low: float = 45.0, order: int = 5) -> np.ndarray:
        """Apply low-pass filter to suppress powerline interference."""
        b, a = self.lowpass_filter(self.sampling_rate, low, order)
        return lfilter(b, a, data)

    def wavelet_baseline(self, x: np.ndarray, level: int = 8, wavelet: str = "sym4") -> np.ndarray:
        """Remove baseline wander via wavelet soft-thresholding."""
        if pywt is None:
            raise ImportError("Install PyWavelets (pywt) to use wavelet_baseline().")
        coeffs = pywt.wavedec(x, wavelet, level=level)
        coeffs[1:] = (pywt.threshold(c, 1.2, mode="soft") for c in coeffs[1:])
        return x - pywt.waverec(coeffs, wavelet)[: len(x)]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        row = self.y.iloc[idx]
        labels = np.zeros(len(self.class_map), dtype=np.float32)
        labels[row.class_ids] = 1.0
        if self.sampling_rate == 100:
            filename = row.filename_lr
        elif self.sampling_rate == 500:
            filename = row.filename_hr
        else:
            raise ValueError(f"Unsupported sampling_rate: {self.sampling_rate}")
        ecg = self._read_signal(self.data_root / filename)          # (samples, leads), mV
        ecg = self._remove_baseline(ecg)                             # median-filter baseline removal first
        ecg = self.powerline_noise_filtering(ecg.T).T                # then 45 Hz low-pass filter
        return {"ecg": ecg.T.astype(np.float32), "class": labels}    # (leads, samples); no amplitude scaling


def collect_labeled_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Collect ECG and label arrays from a PTBXLDataset, dropping empty-label samples."""
    xs, ys = [], []
    for sample in tqdm(dataset, desc="Collect records"):
        if np.sum(sample["class"]):
            xs.append(sample["ecg"])
            ys.append(sample["class"])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
