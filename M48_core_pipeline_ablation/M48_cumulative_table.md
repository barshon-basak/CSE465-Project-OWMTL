# M48 — cumulative ablation ladder

Baseline first, one component added per row, on the **corrected official 60/40 patient-independent split** (2,636 test cycles, 47 patients). Every score is recomputed from the source run's committed raw confusion matrix. A rung whose run does not exist yet is `pending` and is **never** filled with a plausible value.

⚠ marks a step smaller than the measured seed band (range 0.0141 over seeds [42, 1, 2]) — a step inside that band is not distinguishable from run-to-run noise.

### Order A — SpecAugment before class weighting

| Config | pretrain | finetune | classweight | specaug | ampnorm | ICBHI | Se | Sp | F1<sub>macro</sub> | Δ vs prev |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| **S0** Baseline — MobileNetV2, random init, frozen, plain CE | ✗ | ✗ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S1** + ImageNet pre-training | ✓ | ✗ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S2** + full fine-tuning | ✓ | ✓ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S3** + SpecAugment | ✓ | ✓ | ✗ | ✓ | ✗ | 0.5497 | 0.4275 | 0.6718 | 0.4294 | — |
| **S4** + Inverse-frequency class-weighted CE | ✓ | ✓ | ✓ | ✓ | ✗ | 0.5602 | 0.4089 | 0.7115 | 0.4141 | +0.0105 ⚠ |
| **S5** + amplitude normalisation — FULL PIPELINE | ✓ | ✓ | ✓ | ✓ | ✓ | **0.5764** | 0.3996 | 0.7532 | 0.4217 | +0.0162 |

### Order B — class weighting before SpecAugment

| Config | pretrain | finetune | classweight | specaug | ampnorm | ICBHI | Se | Sp | F1<sub>macro</sub> | Δ vs prev |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| **S0** Baseline — MobileNetV2, random init, frozen, plain CE | ✗ | ✗ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S1** + ImageNet pre-training | ✓ | ✗ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S2** + full fine-tuning | ✓ | ✓ | ✗ | ✗ | ✗ | `pending` | — | — | — | — |
| **S3** + class-weighted CE | ✓ | ✓ | ✓ | ✗ | ✗ | 0.5200 | 0.4266 | 0.6135 | 0.3985 | — |
| **S4** + SpecAugment | ✓ | ✓ | ✓ | ✓ | ✗ | 0.5602 | 0.4089 | 0.7115 | 0.4141 | +0.0402 |
| **S5** + amplitude normalisation — FULL PIPELINE | ✓ | ✓ | ✓ | ✓ | ✓ | **0.5764** | 0.3996 | 0.7532 | 0.4217 | +0.0162 |

### The order effect, measured

Both orderings above are built from the same runs and differ only in which of the two components is added at rung 4. What each is credited with therefore depends on where it sits, and this is that difference:

| Component | credited in order A | credited in order B | shift |
|---|---:|---:|---:|
| SpecAugment (training split only) | pending | +0.0402 | — |
| Inverse-frequency class-weighted CE | +0.0105 | pending | — |

### Leave-one-out companion (M45, already complete)

Full pipeline minus one component. Reported beside the ladder because the two answer different questions — *worth given everything else* versus *worth given only what came before*.

| Row | Change from full model | ICBHI | Δ vs A0 |
|---|---|---:|---:|
| `A0` | baseline (M22_v2): MobileNetV2 + SpecAugment | 0.5602 | — |
| `A1` | - SpecAugment (M3_v2) | 0.5200 | -0.0402 |
| `A2` | - ImageNet pre-training (random init) | 0.4999 | -0.0603 |
| `A3` | - class-weighted loss (plain CE) | 0.5497 | -0.0105 |
| `A4` | frozen backbone, classifier only | 0.4595 | -0.1007 |
| `A5` | 64 mels instead of 128 | 0.5427 | -0.0175 |
| `A6` | 4 s cycles instead of 8 s | 0.5224 | -0.0378 |
| `P1` | + band-pass filter (50-2000 Hz Butterworth) | 0.5513 | -0.0089 |
| `P2` | + spectral-gating denoising | 0.5248 | -0.0354 |
| `P3` | + per-cycle peak amplitude normalisation | **0.5764** | +0.0162 |
| `P4` | zero-padding instead of cyclic tiling | 0.5291 | -0.0311 |
| `P5` | - per-spectrogram min-max normalisation | 0.5539 | -0.0063 |

Rows P1–P3 **add** a stage the baseline does not have, so their sign reads the other way: a positive delta is a recommendation to adopt.

### Still to run

`S0`, `S1`, `S2` — rows in `m48_gpu_rows.py`, produced by `M48_kaggle_tier_A.ipynb` alongside the seed band. About 18 minutes each; they share the spectrogram cache with every other row, so they add no cache time.

