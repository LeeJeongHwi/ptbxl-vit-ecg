**Degradation comparison — Proposed ViT vs. each baseline** (per-seed ΔEMR = EMR(noisy) − EMR(clean); Welch's two-sided t-test on ΔEMR; positive Δ(ViT − baseline) = ViT degrades less. With five seeds per architecture these tests have low power and are reported descriptively.)

| Noise | SNR | Baseline | ViT ΔEMR | Baseline ΔEMR | Δ(ViT − baseline) | Cohen's d | p |
|---|---|---|---|---|---|---|---|
| Gaussian white noise | 30 dB | ResNet1D | 0.0001 ± 0.0005 | 0.0015 ± 0.0031 | -0.0014 | -0.62 | 0.380 |
| Gaussian white noise | 30 dB | BiLSTM | 0.0001 ± 0.0005 | -0.0002 ± 0.0021 | +0.0003 | +0.18 | 0.785 |
| Gaussian white noise | 30 dB | InceptionTime | 0.0001 ± 0.0005 | -0.0007 ± 0.0020 | +0.0008 | +0.57 | 0.414 |
| Gaussian white noise | 20 dB | ResNet1D | 0.0007 ± 0.0010 | -0.0019 ± 0.0043 | +0.0026 | +0.82 | 0.256 |
| Gaussian white noise | 20 dB | BiLSTM | 0.0007 ± 0.0010 | -0.0040 ± 0.0025 | +0.0047 | +2.51 | 0.009 |
| Gaussian white noise | 20 dB | InceptionTime | 0.0007 ± 0.0010 | -0.0019 ± 0.0015 | +0.0027 | +2.08 | 0.013 |
| Gaussian white noise | 10 dB | ResNet1D | -0.0014 ± 0.0022 | -0.0240 ± 0.0176 | +0.0226 | +1.80 | 0.045 |
| Gaussian white noise | 10 dB | BiLSTM | -0.0014 ± 0.0022 | -0.0939 ± 0.0370 | +0.0925 | +3.53 | 0.005 |
| Gaussian white noise | 10 dB | InceptionTime | -0.0014 ± 0.0022 | -0.0339 ± 0.0263 | +0.0325 | +1.74 | 0.050 |
| Gaussian white noise | 5 dB | ResNet1D | -0.0006 ± 0.0017 | -0.1119 ± 0.0638 | +0.1113 | +2.46 | 0.018 |
| Gaussian white noise | 5 dB | BiLSTM | -0.0006 ± 0.0017 | -0.3674 ± 0.0669 | +0.3668 | +7.76 | 0.000 |
| Gaussian white noise | 5 dB | InceptionTime | -0.0006 ± 0.0017 | -0.1300 ± 0.0540 | +0.1295 | +3.39 | 0.006 |
| Gaussian white noise | 0 dB | ResNet1D | -0.0091 ± 0.0024 | -0.3288 ± 0.1238 | +0.3197 | +3.65 | 0.004 |
| Gaussian white noise | 0 dB | BiLSTM | -0.0091 ± 0.0024 | -0.5491 ± 0.0205 | +0.5400 | +37.02 | 0.000 |
| Gaussian white noise | 0 dB | InceptionTime | -0.0091 ± 0.0024 | -0.3598 ± 0.1121 | +0.3507 | +4.43 | 0.002 |
| Baseline wander (0.1–0.5 Hz) | 30 dB | ResNet1D | 0.0001 ± 0.0005 | 0.0001 ± 0.0009 | +0.0000 | +0.00 | 1.000 |
| Baseline wander (0.1–0.5 Hz) | 30 dB | BiLSTM | 0.0001 ± 0.0005 | -0.0008 ± 0.0016 | +0.0009 | +0.79 | 0.270 |
| Baseline wander (0.1–0.5 Hz) | 30 dB | InceptionTime | 0.0001 ± 0.0005 | -0.0013 ± 0.0010 | +0.0014 | +1.75 | 0.033 |
| Baseline wander (0.1–0.5 Hz) | 20 dB | ResNet1D | -0.0005 ± 0.0017 | -0.0033 ± 0.0034 | +0.0029 | +1.06 | 0.146 |
| Baseline wander (0.1–0.5 Hz) | 20 dB | BiLSTM | -0.0005 ± 0.0017 | -0.0028 ± 0.0017 | +0.0023 | +1.34 | 0.068 |
| Baseline wander (0.1–0.5 Hz) | 20 dB | InceptionTime | -0.0005 ± 0.0017 | -0.0040 ± 0.0038 | +0.0035 | +1.18 | 0.114 |
| Baseline wander (0.1–0.5 Hz) | 10 dB | ResNet1D | -0.0131 ± 0.0061 | -0.0314 ± 0.0178 | +0.0184 | +1.38 | 0.082 |
| Baseline wander (0.1–0.5 Hz) | 10 dB | BiLSTM | -0.0131 ± 0.0061 | -0.0478 ± 0.0122 | +0.0348 | +3.61 | 0.001 |
| Baseline wander (0.1–0.5 Hz) | 10 dB | InceptionTime | -0.0131 ± 0.0061 | -0.0703 ± 0.0145 | +0.0572 | +5.14 | 0.000 |
| Baseline wander (0.1–0.5 Hz) | 5 dB | ResNet1D | -0.0540 ± 0.0156 | -0.1169 ± 0.0442 | +0.0628 | +1.90 | 0.030 |
| Baseline wander (0.1–0.5 Hz) | 5 dB | BiLSTM | -0.0540 ± 0.0156 | -0.2060 ± 0.0345 | +0.1520 | +5.68 | 0.000 |
| Baseline wander (0.1–0.5 Hz) | 5 dB | InceptionTime | -0.0540 ± 0.0156 | -0.2232 ± 0.0392 | +0.1691 | +5.67 | 0.000 |
| Baseline wander (0.1–0.5 Hz) | 0 dB | ResNet1D | -0.2220 ± 0.0285 | -0.3586 ± 0.0455 | +0.1366 | +3.60 | 0.001 |
| Baseline wander (0.1–0.5 Hz) | 0 dB | BiLSTM | -0.2220 ± 0.0285 | -0.4568 ± 0.0193 | +0.2348 | +9.66 | 0.000 |
| Baseline wander (0.1–0.5 Hz) | 0 dB | InceptionTime | -0.2220 ± 0.0285 | -0.3761 ± 0.0436 | +0.1541 | +4.19 | 0.000 |
| Powerline interference (50 Hz) | 30 dB | ResNet1D | 0.0008 ± 0.0005 | 0.0009 ± 0.0035 | -0.0001 | -0.04 | 0.956 |
| Powerline interference (50 Hz) | 30 dB | BiLSTM | 0.0008 ± 0.0005 | 0.0002 ± 0.0036 | +0.0006 | +0.26 | 0.706 |
| Powerline interference (50 Hz) | 30 dB | InceptionTime | 0.0008 ± 0.0005 | 0.0000 ± 0.0017 | +0.0008 | +0.65 | 0.352 |
| Powerline interference (50 Hz) | 20 dB | ResNet1D | -0.0000 ± 0.0010 | -0.0010 ± 0.0055 | +0.0010 | +0.26 | 0.703 |
| Powerline interference (50 Hz) | 20 dB | BiLSTM | -0.0000 ± 0.0010 | -0.0047 ± 0.0030 | +0.0047 | +2.15 | 0.020 |
| Powerline interference (50 Hz) | 20 dB | InceptionTime | -0.0000 ± 0.0010 | -0.0021 ± 0.0008 | +0.0021 | +2.41 | 0.006 |
| Powerline interference (50 Hz) | 10 dB | ResNet1D | -0.0032 ± 0.0018 | -0.0280 ± 0.0189 | +0.0248 | +1.85 | 0.042 |
| Powerline interference (50 Hz) | 10 dB | BiLSTM | -0.0032 ± 0.0018 | -0.0994 ± 0.0373 | +0.0963 | +3.65 | 0.004 |
| Powerline interference (50 Hz) | 10 dB | InceptionTime | -0.0032 ± 0.0018 | -0.0269 ± 0.0132 | +0.0237 | +2.52 | 0.015 |
| Powerline interference (50 Hz) | 5 dB | ResNet1D | -0.0062 ± 0.0024 | -0.1075 ± 0.0499 | +0.1013 | +2.87 | 0.010 |
| Powerline interference (50 Hz) | 5 dB | BiLSTM | -0.0062 ± 0.0024 | -0.3469 ± 0.0543 | +0.3407 | +8.86 | 0.000 |
| Powerline interference (50 Hz) | 5 dB | InceptionTime | -0.0062 ± 0.0024 | -0.1216 ± 0.0605 | +0.1154 | +2.69 | 0.013 |
| Powerline interference (50 Hz) | 0 dB | ResNet1D | -0.0165 ± 0.0052 | -0.3138 ± 0.0991 | +0.2973 | +4.24 | 0.003 |
| Powerline interference (50 Hz) | 0 dB | BiLSTM | -0.0165 ± 0.0052 | -0.5473 ± 0.0163 | +0.5308 | +43.82 | 0.000 |
| Powerline interference (50 Hz) | 0 dB | InceptionTime | -0.0165 ± 0.0052 | -0.3095 ± 0.0720 | +0.2930 | +5.74 | 0.001 |
| Muscle artifact (20–150 Hz) | 30 dB | ResNet1D | -0.0001 ± 0.0009 | 0.0014 ± 0.0029 | -0.0015 | -0.69 | 0.326 |
| Muscle artifact (20–150 Hz) | 30 dB | BiLSTM | -0.0001 ± 0.0009 | -0.0005 ± 0.0027 | +0.0004 | +0.18 | 0.783 |
| Muscle artifact (20–150 Hz) | 30 dB | InceptionTime | -0.0001 ± 0.0009 | -0.0008 ± 0.0016 | +0.0007 | +0.58 | 0.395 |
| Muscle artifact (20–150 Hz) | 20 dB | ResNet1D | -0.0004 ± 0.0006 | 0.0002 ± 0.0042 | -0.0006 | -0.19 | 0.784 |
| Muscle artifact (20–150 Hz) | 20 dB | BiLSTM | -0.0004 ± 0.0006 | -0.0056 ± 0.0030 | +0.0052 | +2.40 | 0.017 |
| Muscle artifact (20–150 Hz) | 20 dB | InceptionTime | -0.0004 ± 0.0006 | -0.0012 ± 0.0013 | +0.0008 | +0.83 | 0.242 |
| Muscle artifact (20–150 Hz) | 10 dB | ResNet1D | 0.0004 ± 0.0017 | -0.0285 ± 0.0200 | +0.0289 | +2.03 | 0.032 |
| Muscle artifact (20–150 Hz) | 10 dB | BiLSTM | 0.0004 ± 0.0017 | -0.1139 ± 0.0397 | +0.1143 | +4.07 | 0.003 |
| Muscle artifact (20–150 Hz) | 10 dB | InceptionTime | 0.0004 ± 0.0017 | -0.0297 ± 0.0224 | +0.0300 | +1.89 | 0.040 |
| Muscle artifact (20–150 Hz) | 5 dB | ResNet1D | -0.0024 ± 0.0014 | -0.1175 ± 0.0645 | +0.1151 | +2.52 | 0.016 |
| Muscle artifact (20–150 Hz) | 5 dB | BiLSTM | -0.0024 ± 0.0014 | -0.4039 ± 0.0597 | +0.4015 | +9.51 | 0.000 |
| Muscle artifact (20–150 Hz) | 5 dB | InceptionTime | -0.0024 ± 0.0014 | -0.1090 ± 0.0454 | +0.1066 | +3.32 | 0.006 |
| Muscle artifact (20–150 Hz) | 0 dB | ResNet1D | -0.0033 ± 0.0038 | -0.3301 ± 0.1239 | +0.3268 | +3.73 | 0.004 |
| Muscle artifact (20–150 Hz) | 0 dB | BiLSTM | -0.0033 ± 0.0038 | -0.5559 ± 0.0176 | +0.5525 | +43.48 | 0.000 |
| Muscle artifact (20–150 Hz) | 0 dB | InceptionTime | -0.0033 ± 0.0038 | -0.3248 ± 0.1015 | +0.3215 | +4.48 | 0.002 |
| Composite (all four) | 30 dB | ResNet1D | -0.0009 ± 0.0003 | 0.0008 ± 0.0030 | -0.0018 | -0.84 | 0.255 |
| Composite (all four) | 30 dB | BiLSTM | -0.0009 ± 0.0003 | 0.0003 ± 0.0019 | -0.0012 | -0.87 | 0.237 |
| Composite (all four) | 30 dB | InceptionTime | -0.0009 ± 0.0003 | -0.0004 ± 0.0015 | -0.0006 | -0.51 | 0.464 |
| Composite (all four) | 20 dB | ResNet1D | 0.0007 ± 0.0024 | 0.0005 ± 0.0035 | +0.0003 | +0.09 | 0.887 |
| Composite (all four) | 20 dB | BiLSTM | 0.0007 ± 0.0024 | -0.0032 ± 0.0025 | +0.0040 | +1.64 | 0.032 |
| Composite (all four) | 20 dB | InceptionTime | 0.0007 ± 0.0024 | -0.0025 ± 0.0012 | +0.0032 | +1.75 | 0.034 |
| Composite (all four) | 10 dB | ResNet1D | -0.0041 ± 0.0039 | -0.0284 ± 0.0183 | +0.0243 | +1.83 | 0.040 |
| Composite (all four) | 10 dB | BiLSTM | -0.0041 ± 0.0039 | -0.0865 ± 0.0277 | +0.0824 | +4.16 | 0.002 |
| Composite (all four) | 10 dB | InceptionTime | -0.0041 ± 0.0039 | -0.0273 ± 0.0172 | +0.0233 | +1.87 | 0.037 |
| Composite (all four) | 5 dB | ResNet1D | -0.0138 ± 0.0025 | -0.1246 ± 0.0643 | +0.1108 | +2.43 | 0.018 |
| Composite (all four) | 5 dB | BiLSTM | -0.0138 ± 0.0025 | -0.3474 ± 0.0607 | +0.3335 | +7.76 | 0.000 |
| Composite (all four) | 5 dB | InceptionTime | -0.0138 ± 0.0025 | -0.0998 ± 0.0292 | +0.0860 | +4.14 | 0.003 |
| Composite (all four) | 0 dB | ResNet1D | -0.0517 ± 0.0028 | -0.3457 ± 0.1104 | +0.2940 | +3.76 | 0.004 |
| Composite (all four) | 0 dB | BiLSTM | -0.0517 ± 0.0028 | -0.5440 ± 0.0189 | +0.4923 | +36.43 | 0.000 |
| Composite (all four) | 0 dB | InceptionTime | -0.0517 ± 0.0028 | -0.2709 ± 0.0641 | +0.2192 | +4.83 | 0.002 |
