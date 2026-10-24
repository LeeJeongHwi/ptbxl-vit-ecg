**Test-set performance over 5 random seeds (42, 43, 44, 45, 46); mean ± sample SD.** For every seed the classification threshold was selected on the validation set (50 values in [0, 1], maximum EMR) and the test set was evaluated once at that threshold.

| Model | Params | n seeds | Best Thr. (mean, range) | Val EMR | Test EMR | Hamming Loss | NORM F1 | MI F1 | STTC F1 | CD F1 | HYP F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Proposed ViT (depth = 6) | 3.29 M | 5 | 0.7224 (0.7143–0.7347) | 0.6111 ± 0.0060 | 0.6001 ± 0.0089 | 0.1272 ± 0.0011 | 0.840 ± 0.006 | 0.696 ± 0.013 | 0.738 ± 0.005 | 0.691 ± 0.009 | 0.538 ± 0.047 |
| ResNet1D | 8.74 M | 5 | 0.7143 (0.6735–0.7551) | 0.6342 ± 0.0057 | 0.6348 ± 0.0084 | 0.1111 ± 0.0021 | 0.865 ± 0.003 | 0.732 ± 0.003 | 0.761 ± 0.013 | 0.747 ± 0.014 | 0.554 ± 0.045 |
| BiLSTM | 0.72 M | 5 | 0.7143 (0.6939–0.7347) | 0.6324 ± 0.0045 | 0.6342 ± 0.0057 | 0.1128 ± 0.0011 | 0.863 ± 0.004 | 0.720 ± 0.007 | 0.770 ± 0.008 | 0.730 ± 0.019 | 0.542 ± 0.042 |
| InceptionTime | 0.49 M | 5 | 0.7306 (0.7143–0.7551) | 0.6291 ± 0.0048 | 0.6259 ± 0.0063 | 0.1135 ± 0.0020 | 0.861 ± 0.004 | 0.730 ± 0.019 | 0.768 ± 0.006 | 0.723 ± 0.013 | 0.511 ± 0.038 |

Best mean per column: Test EMR → ResNet1D; Hamming Loss → ResNet1D; NORM F1 → ResNet1D; MI F1 → ResNet1D; STTC F1 → BiLSTM; CD F1 → ResNet1D; HYP F1 → ResNet1D
