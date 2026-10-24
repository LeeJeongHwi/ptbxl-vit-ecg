**Degradation comparison — Proposed ViT vs. each baseline** (per-seed ΔEMR = EMR(noisy) − EMR(clean); Welch's two-sided t-test on ΔEMR; positive Δ(ViT − baseline) = ViT degrades less. With five seeds per architecture these tests have low power and are reported descriptively.)

| Noise | SNR | Baseline | ViT ΔEMR | Baseline ΔEMR | Δ(ViT − baseline) | Cohen's d | p |
|---|---|---|---|---|---|---|---|
| Powerline interference (60 Hz) | 30 dB | ResNet1D | -0.0001 ± 0.0004 | 0.0006 ± 0.0044 | -0.0007 | -0.24 | 0.726 |
| Powerline interference (60 Hz) | 30 dB | BiLSTM | -0.0001 ± 0.0004 | 0.0003 ± 0.0037 | -0.0004 | -0.14 | 0.836 |
| Powerline interference (60 Hz) | 30 dB | InceptionTime | -0.0001 ± 0.0004 | -0.0003 ± 0.0015 | +0.0002 | +0.17 | 0.795 |
| Powerline interference (60 Hz) | 20 dB | ResNet1D | 0.0003 ± 0.0010 | -0.0001 ± 0.0048 | +0.0004 | +0.11 | 0.874 |
| Powerline interference (60 Hz) | 20 dB | BiLSTM | 0.0003 ± 0.0010 | -0.0042 ± 0.0046 | +0.0044 | +1.34 | 0.097 |
| Powerline interference (60 Hz) | 20 dB | InceptionTime | 0.0003 ± 0.0010 | -0.0012 ± 0.0014 | +0.0015 | +1.25 | 0.088 |
| Powerline interference (60 Hz) | 10 dB | ResNet1D | -0.0001 ± 0.0017 | -0.0361 ± 0.0216 | +0.0361 | +2.35 | 0.020 |
| Powerline interference (60 Hz) | 10 dB | BiLSTM | -0.0001 ± 0.0017 | -0.1106 ± 0.0411 | +0.1105 | +3.80 | 0.004 |
| Powerline interference (60 Hz) | 10 dB | InceptionTime | -0.0001 ± 0.0017 | -0.0290 ± 0.0174 | +0.0289 | +2.34 | 0.020 |
| Powerline interference (60 Hz) | 5 dB | ResNet1D | 0.0000 ± 0.0027 | -0.1378 ± 0.0590 | +0.1378 | +3.30 | 0.006 |
| Powerline interference (60 Hz) | 5 dB | BiLSTM | 0.0000 ± 0.0027 | -0.3487 ± 0.0610 | +0.3487 | +8.08 | 0.000 |
| Powerline interference (60 Hz) | 5 dB | InceptionTime | 0.0000 ± 0.0027 | -0.1220 ± 0.0533 | +0.1220 | +3.23 | 0.007 |
| Powerline interference (60 Hz) | 0 dB | ResNet1D | -0.0026 ± 0.0065 | -0.3576 ± 0.0929 | +0.3550 | +5.39 | 0.001 |
| Powerline interference (60 Hz) | 0 dB | BiLSTM | -0.0026 ± 0.0065 | -0.5440 ± 0.0190 | +0.5414 | +38.11 | 0.000 |
| Powerline interference (60 Hz) | 0 dB | InceptionTime | -0.0026 ± 0.0065 | -0.3177 ± 0.0627 | +0.3151 | +7.07 | 0.000 |
