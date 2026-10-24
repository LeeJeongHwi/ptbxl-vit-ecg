**Seed-level comparison of the proposed ViT with each baseline** (Welch's two-sided t-test on the per-seed test metrics; Cohen's d with pooled SD; positive Δ = ViT higher). With five seeds per architecture these tests have low power and are reported descriptively.

| Baseline | Metric | ViT mean ± SD | Baseline mean ± SD | Δ (ViT − baseline) | Cohen's d | t | p |
|---|---|---|---|---|---|---|---|
| ResNet1D | Test EMR | 0.6001 ± 0.0089 | 0.6348 ± 0.0084 | -0.0348 | -4.02 | -6.35 | 0.000 |
| ResNet1D | Hamming Loss | 0.1272 ± 0.0011 | 0.1111 ± 0.0021 | +0.0161 | +9.40 | +14.85 | 0.000 |
| ResNet1D | NORM F1 | 0.8401 ± 0.0058 | 0.8652 ± 0.0031 | -0.0251 | -5.37 | -8.49 | 0.000 |
| ResNet1D | MI F1 | 0.6960 ± 0.0126 | 0.7325 ± 0.0035 | -0.0365 | -3.94 | -6.23 | 0.002 |
| ResNet1D | STTC F1 | 0.7378 ± 0.0046 | 0.7613 ± 0.0134 | -0.0235 | -2.35 | -3.72 | 0.014 |
| ResNet1D | CD F1 | 0.6908 ± 0.0094 | 0.7465 ± 0.0140 | -0.0557 | -4.68 | -7.39 | 0.000 |
| ResNet1D | HYP F1 | 0.5380 ± 0.0472 | 0.5537 ± 0.0454 | -0.0157 | -0.34 | -0.54 | 0.607 |
| BiLSTM | Test EMR | 0.6001 ± 0.0089 | 0.6342 ± 0.0057 | -0.0341 | -4.54 | -7.18 | 0.000 |
| BiLSTM | Hamming Loss | 0.1272 ± 0.0011 | 0.1128 ± 0.0011 | +0.0145 | +12.93 | +20.45 | 0.000 |
| BiLSTM | NORM F1 | 0.8401 ± 0.0058 | 0.8631 ± 0.0040 | -0.0229 | -4.60 | -7.27 | 0.000 |
| BiLSTM | MI F1 | 0.6960 ± 0.0126 | 0.7200 ± 0.0068 | -0.0240 | -2.36 | -3.74 | 0.009 |
| BiLSTM | STTC F1 | 0.7378 ± 0.0046 | 0.7703 ± 0.0078 | -0.0324 | -5.10 | -8.06 | 0.000 |
| BiLSTM | CD F1 | 0.6908 ± 0.0094 | 0.7297 ± 0.0187 | -0.0389 | -2.63 | -4.16 | 0.006 |
| BiLSTM | HYP F1 | 0.5380 ± 0.0472 | 0.5418 ± 0.0417 | -0.0038 | -0.09 | -0.14 | 0.896 |
| InceptionTime | Test EMR | 0.6001 ± 0.0089 | 0.6259 ± 0.0063 | -0.0258 | -3.32 | -5.25 | 0.001 |
| InceptionTime | Hamming Loss | 0.1272 ± 0.0011 | 0.1135 ± 0.0020 | +0.0137 | +8.38 | +13.25 | 0.000 |
| InceptionTime | NORM F1 | 0.8401 ± 0.0058 | 0.8608 ± 0.0039 | -0.0207 | -4.19 | -6.62 | 0.000 |
| InceptionTime | MI F1 | 0.6960 ± 0.0126 | 0.7297 ± 0.0186 | -0.0337 | -2.12 | -3.35 | 0.012 |
| InceptionTime | STTC F1 | 0.7378 ± 0.0046 | 0.7681 ± 0.0063 | -0.0303 | -5.51 | -8.71 | 0.000 |
| InceptionTime | CD F1 | 0.6908 ± 0.0094 | 0.7233 ± 0.0127 | -0.0324 | -2.92 | -4.61 | 0.002 |
| InceptionTime | HYP F1 | 0.5380 ± 0.0472 | 0.5105 ± 0.0376 | +0.0275 | +0.64 | +1.02 | 0.340 |
