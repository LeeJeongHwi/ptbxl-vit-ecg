# ptbxl-vit-ecg

Code and result files for the article

> S. Oh, J. Lee and D.-K. Kim, *Explainable Vision Transformer for Multi-Label 12-Lead ECG Classification* (under review).

A Vision Transformer classifies 10-second 12-lead ECG records of the PTB-XL dataset into the five
diagnostic superclasses (NORM, MI, STTC, CD, HYP) as a multi-label problem, without beat
segmentation: the 12 x 5000 signal is cut into 250 non-overlapping patches of 20 samples (0.04 s),
embedded by a one-dimensional convolution and processed by six Transformer encoder blocks. The
attention from the class token to the patch tokens is used to show which 0.04-s windows the model
weighted most. The model is compared with three baselines (ResNet1D, BiLSTM, InceptionTime) under
one protocol, with five random seeds per architecture and an evaluation under synthetic noise.

## Repository layout

```
ptbxl/                      package: data loading and preprocessing, models, training loop, metrics, plots
run_ablation.py             one-factor-at-a-time ablation of the ViT; trains the final model (depth 6)
run_multiseed.py            four architectures x five seeds, mean +- SD, Welch's t-tests
run_noise_robustness.py     evaluation of the 20 trained models under synthetic noise
sweep_and_evaluate.py       threshold selection on the validation fold + test evaluation of one checkpoint
make_threshold_figures.py   Figures 2-5 (confusion matrices, threshold sweeps)
make_attention_figures.py   Figures 6-9 and Supplementary Figures S1-S5 (attention maps)
plot_noise_figure.py        Supplementary Figure S6 (metric vs. SNR)
plot_noise_examples.py      Supplementary Figure S7 (noisy signal examples)
results/                    result files of the reported experiments (see below)
```

## Environment

Python 3.11, PyTorch 2.1.0 (CUDA 12.1 build), scikit-learn 1.3.2, NumPy 1.26.0, pandas 2.1.3,
SciPy 1.11.4, einops 0.7.0; exact versions are pinned in `pyproject.toml` / `uv.lock`.
The experiments were run on Windows 11 with a single NVIDIA RTX 4090.

```
# with uv
uv sync
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS

# or with pip
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

`neurokit2` is optional; `make_attention_figures.py` uses it for R-peak detection in its tables and
falls back to a simple SciPy peak detector when it is not installed (the figures do not depend on it).

## Data

Download PTB-XL 1.0.3 from PhysioNet (https://physionet.org/content/ptb-xl/1.0.3/) and place it in
`ptb_xl_dataset/` so that `ptb_xl_dataset/ptbxl_database.csv`, `ptb_xl_dataset/scp_statements.csv`
and `ptb_xl_dataset/records500/` exist. The 500 Hz records are used.

The first script that needs the data builds a cache of preprocessed arrays in `npy_data/`
(`x_train.npy`, ..., `y_test.npy`): records are read with `wfdb`, the baseline is estimated per lead
with two cascaded median filters (151 and 351 samples, i.e. 0.3 s and 0.7 s) and subtracted, and a
fifth-order Butterworth low-pass filter with a 45 Hz cut-off is applied. No amplitude scaling is
applied; the arrays are in mV. Records without a diagnostic superclass label are dropped.
Split: folds 1-8 training (17,084 records), fold 9 validation (2,146), fold 10 test (2,158).

## Protocol

All training runs share the same settings: AdamW (learning rate 1e-4, weight decay 0.05), cosine
annealing (T_max 70, eta_min 1e-6), asymmetric loss (gamma_neg 2, gamma_pos 0, clip 0.05), batch size
16, at most 70 epochs with early stopping on the validation loss (patience 10), checkpoint with the
lowest validation loss. For every trained model the classification threshold is chosen on the
validation fold as the value among 50 equally spaced thresholds in [0, 1] that maximises the exact
match ratio (EMR), and the test fold is evaluated once at that threshold. Model selection in the
ablation study uses the validation EMR only.

## Reproducing the experiments

Run everything from the repository root with the virtual environment activated. Approximate
training times are for one RTX 4090.

| Step | Command | Output | Reported in |
|---|---|---|---|
| Ablation (10 configurations, about 1 h; trains the final model) | `python run_ablation.py --data-root ptb_xl_dataset` | `results/ablation/`, `models/ablation/<name>/best_<name>.pth` | Table 5; Table 2, Figures 2-5 and Supplementary Tables S1-S2 (depth_6 run) |
| Multi-seed comparison (4 architectures x seeds 42-46, about 2 h) | `python run_multiseed.py` | `results/multiseed/` | Table 3, Supplementary Table S3 |
| Noise robustness, 50 Hz mains | `python run_noise_robustness.py` | `results/noise_robustness/` | Table 4, Supplementary Tables S4-S6 |
| Noise robustness, 60 Hz mains | `python run_noise_robustness.py --powerline-hz 60 --noise-types powerline --out-dir results/noise_robustness_60hz` | `results/noise_robustness_60hz/` | Table 4, Supplementary Tables S4-S6 |
| Figures 2-5 | `python make_threshold_figures.py --out-dir figures` | `figures/Figure2_...png` ... | Figures 2-5 |
| Attention maps | `python make_attention_figures.py --data-root ptb_xl_dataset --fig-dir figures/attention` | `results/attention/`, `figures/attention/` | Figures 6-9, Supplementary Figures S1-S5 |
| Noise figures | `python plot_noise_figure.py` and `python plot_noise_examples.py` | `results/noise_robustness/fig_noise_robustness.png`, `.../noise_examples_0dB.png` | Supplementary Figures S6, S7 |

Notes

* `run_multiseed.py` and `run_noise_robustness.py` resume: runs whose result files exist are skipped;
  `--aggregate-only` rebuilds the tables (`*_table.md`, `*_stats.md`, CSV/JSON) from the stored per-run
  results without any computation, which reproduces the tables of the paper from the files in
  `results/`.
* The seed-42 ViT run of `run_multiseed.py` reproduces the depth_6 run of the ablation
  (same validation loss, threshold and test metrics).
* `make_attention_figures.py` selects the illustrated records by a fixed rule (the true positive of
  each category with the highest predicted probability; for CD restricted to records annotated with
  CLBBB and CRBBB). With the cache built as described above and the depth_6 checkpoint it selects
  test indices 1304, 2062, 2154 and 1362 (PTB-XL ecg_id 12169, 20120, 21812 and 12585). `--list-candidates CLBBB` ranks the
  candidates of one SCP code; `--indices` draws arbitrary records.
* `sweep_and_evaluate.py --model vit --checkpoint <file> --output-dir <dir>` runs the threshold
  selection and test evaluation for a single checkpoint of any of the four architectures
  (`--model vit | resnet1d | bilstm | inception`). `ptbxl --data-root ptb_xl_dataset --model vit`
  trains and evaluates one model with the default settings.

## Result files

`results/ablation/ablation_summary.json` and `results/ablation/<name>/<name>_val_sweep.csv` hold the
ten ablation runs (trainable parameters, best epoch, validation-selected threshold, validation EMR,
test EMR, Hamming loss, per-class metrics and confusion matrices; and the full validation sweep).
`results/multiseed/` holds `result.json` and `val_sweep.csv` of each of the 20 runs together with the
aggregated tables. `results/noise_robustness/` and `results/noise_robustness_60hz/` hold every noisy
evaluation (`noise_raw.json`, `noise_per_seed.csv`) and the aggregated tables. These files are the
source of Tables 2-5 and Supplementary Tables S1-S6 of the paper.

Trained checkpoints (about 400 MB in total) are not part of the repository; they are available from
the corresponding author on reasonable request. All runs can be reproduced with the commands above.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`. PTB-XL is distributed by PhysioNet under its own license (Creative Commons
Attribution 4.0).
