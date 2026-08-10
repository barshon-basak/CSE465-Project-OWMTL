<h1 align="center">M7 — Deep Ensemble (N = 5)</h1>
<h3 align="center">Member D · Trust, Calibration & Explainability Lead</h3>

<p align="center">
  <em>Open-World Multi-Task Learning for Respiratory Disease Diagnosis</em><br>
  <em>CSE465 — Machine Learning · Group 5 Capstone · North South University</em>
</p>

<p align="center">
  <img alt="Model" src="https://img.shields.io/badge/model-M7-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-complete-brightgreen">
  <img alt="Backbone" src="https://img.shields.io/badge/backbone-MobileNetV2%20(ImageNet)-orange">
  <img alt="Dataset" src="https://img.shields.io/badge/dataset-ICBHI%202017-green">
  <img alt="Runs" src="https://img.shields.io/badge/runs-clean%20%2B%20SpecAugment-lightgrey">
</p>

---

## Overview

**M7 trains the same dual-head model five separate times** with five different random seeds (42–46). This is a *deep ensemble*. The purpose is not to build one stronger classifier — it is to use the **disagreement between the five members** as an epistemic-uncertainty signal: when the five copies disagree about a patient, that patient is likely to have a disease the model was never trained on.

This is the **gold-standard uncertainty baseline** for the project. Member B's cross-task consistency disagreement score (M15/M20) must beat the number established here to support the paper's central claim.

Two complete experiments were run:

| Run | Augmentation | Purpose |
|---|---|---|
| **Clean** | none | Baseline (faculty requirement #3, #4) |
| **Augmented** | SpecAugment | Augmentation ablation (faculty requirement #5) |

---

## Faculty Requirements — Compliance Map

| # | Requirement | Status | Where it is satisfied |
|---|---|---|---|
| 1 | Apply ALL preprocessing techniques for the dataset | ✅ Done | [§1 Preprocessing](#1-preprocessing-pipeline) — 9-stage pipeline |
| 2 | Apply pre-trained DL models, at least one per member | ✅ Done | [§2 Model](#2-model-architecture) — MobileNetV2, ImageNet-pretrained, ×5 members |
| 3 | Accuracy, precision, recall, F1, confusion matrix, model size, # params, training time | ✅ Done | [§3 Metrics](#3-results--full-metric-suite) — all 8 items, per member, both runs |
| 4 | Plot train/val loss and accuracy vs. epochs for ALL applied models | ✅ Done | [§4 Plots](#4-training-curves--confusion-matrices) — 51 figures, all 10 model runs |
| 5 | Apply data augmentation and repeat #3 and #4 | ✅ Done | [§5 Augmentation](#5-augmentation-ablation-specaugment) — full SpecAugment re-run |

---

## 1. Preprocessing Pipeline

Every preprocessing stage applied to the raw ICBHI audio, in execution order. Parameters follow the team's shared protocol (`Model_Training_Protocol.md` §2) so all members' spectrograms are directly comparable.

| # | Stage | Detail |
|---|---|---|
| 1 | **Audio loading** | `torchaudio.load()` on the 920 `.wav` recordings |
| 2 | **Mono conversion** | Multi-channel waveforms averaged across channels |
| 3 | **Resampling** | Any non-16 kHz source resampled to **16 000 Hz** |
| 4 | **Cycle segmentation** | Each recording sliced into individual respiratory cycles using the ICBHI annotation `start`/`end` timestamps → **6 898 cycles** |
| 5 | **Fixed-length padding / trimming** | Every cycle zero-padded or truncated to exactly **8.0 s** (128 000 samples). Mean natural cycle length is 2.7 s |
| 6 | **Log-mel spectrogram** | `MelSpectrogram` → `AmplitudeToDB`. `n_fft=1024`, `hop_length=160` (10 ms), `win_length=400` (25 ms), `n_mels=128` |
| 7 | **Clinical sub-band filtering** | Mel filterbank restricted to **50–2000 Hz** — removes heart/muscle sound below 50 Hz and ambient room noise above 2 kHz |
| 8 | **Per-sample normalization** | Zero-mean / unit-variance standardisation per spectrogram: `(x − μ) / (σ + 1e−6)` |
| 9 | **3-channel replication** | Single-channel spectrogram repeated to `[3, 128, T]` to match the ImageNet-pretrained backbone's expected input |

### Data splitting (patient-independent — non-negotiable)

Cycles from the same patient **never** appear in more than one split. Cycle-randomised splitting causes acoustic-signature leakage and inflates accuracy to ~95% in a way that collapses on deployment.

| Split | Cycles | Notes |
|---|---|---|
| Train | 3 195 | 60% of known-class patients, minus calibration carve-out |
| Calibration | 1 034 | Held out entirely from training; reserved for M11 post-hoc calibration |
| Test | 2 082 | 40% patient-independent hold-out |
| Unknown / OOD | 549 | 19 patients — Bronchiectasis (7), Pneumonia (6), Bronchiolitis (6). **Evaluation only, never trained on** |

Split method: `patient_independent_60_40_with_calibration_carveout`. Disjointness is asserted in code before training begins.

**Class definitions**

- **Known diseases (3):** COPD (64 patients) · Healthy (26) · URTI (14)
- **Unknown diseases (3):** Bronchiectasis · Pneumonia · Bronchiolitis
- **Sound events (4):** Normal (3 642 cycles) · Crackle (1 864) · Wheeze (886) · Both (506)
- **Excluded:** LRTI (n = 2) and Asthma (n = 1) — too few patients to place in either split

---

## 2. Model Architecture

**Pre-trained model used: MobileNetV2, ImageNet-1K weights** (`torchvision.models.mobilenet_v2`) — this satisfies faculty requirement #2.

```
             log-mel spectrogram  [3, 128, 801]
                        │
            ┌───────────▼────────────┐
            │  MobileNetV2 features   │  ImageNet-1K pretrained
            │  (last 3 blocks         │  frozen except final 3 blocks
            │   unfrozen)             │
            └───────────┬────────────┘
                        │  AdaptiveAvgPool2d → 1280-d
            ┌───────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐      ┌──────────────────┐
   │ Sound-event head │      │  Disease head    │
   │ Dropout(0.3)     │      │  Dropout(0.3)    │
   │ Linear(1280→4)   │      │  Linear(1280→3)  │
   └─────────────────┘      └──────────────────┘
   Normal/Crackle/           COPD/Healthy/URTI
   Wheeze/Both
```

**Transfer-learning strategy:** the backbone is frozen except the final 3 feature blocks. With only 104 known-class patients, full fine-tuning of all 3.5 M backbone parameters would overfit immediately. Result: 2 232 839 total parameters, **1 215 047 trainable (54.4%)**.

**Ensemble construction:** 5 members, seeds `[42, 43, 44, 45, 46]`. Each seed independently controls head initialisation and data-loader shuffle order, giving genuinely decorrelated members from the same architecture.

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam, `lr = 1e-4`, `weight_decay = 1e-4` |
| Scheduler | StepLR |
| Batch size | 16 |
| Max epochs | 15 |
| Early stopping | patience 4, monitored on `val_combined_f1` |
| Loss | `cross_entropy(sound) + cross_entropy(disease)`, task weights 1.0 / 1.0 |
| Checkpointing | Every epoch + mid-epoch, mirrored to Google Drive |
| Hardware | Google Colab, Tesla T4, PyTorch 2.11.0+cu128 |

---

## 3. Results — Full Metric Suite

All metrics computed on the **2 082-cycle patient-independent test set** using each member's best checkpoint.

### 3.1 Clean run — sound-event classification (4-class)

| Seed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 42 | 0.5183 | 0.4023 | 0.3697 | 0.3778 |
| 43 | 0.5187 | 0.3717 | 0.3597 | 0.3610 |
| 44 | 0.5101 | 0.3871 | 0.3601 | 0.3602 |
| 45 | 0.5043 | 0.3617 | 0.3521 | 0.3503 |
| 46 | 0.5062 | 0.3921 | 0.3878 | 0.3856 |
| **Mean** | **0.5114** | **0.3830** | **0.3659** | **0.3670** |

**Per-class F1 (seed 42):** Normal 0.633 · Crackle 0.449 · Wheeze 0.240 · Both 0.189

### 3.2 Clean run — disease classification (3-class)

| Seed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 42 | 0.9294 | 0.6191 | 0.5416 | 0.5514 |
| 43 | 0.9270 | 0.6120 | 0.5551 | 0.5595 |
| 44 | 0.9299 | 0.7500 | 0.5497 | 0.5401 |
| 45 | 0.9308 | 0.6259 | 0.5729 | 0.5379 |
| 46 | 0.9328 | 0.5648 | 0.5556 | 0.5354 |
| **Mean** | **0.9300** | **0.6344** | **0.5550** | **0.5449** |

**Per-class disease F1, all seeds (clean)**

| Seed | COPD | Healthy | URTI |
|---|---|---|---|
| 42 | 0.973 | 0.530 | 0.151 |
| 43 | 0.973 | 0.513 | 0.195 |
| 44 | 0.971 | 0.583 | 0.067 |
| 45 | 0.975 | 0.596 | 0.044 |
| 46 | 0.977 | 0.586 | 0.042 |

### 3.3 Confusion matrices

Both raw and row-normalised confusion matrices are stored for **every** member and **both** tasks in `results_M7.json` / `results_M7_aug.json` under `per_member_results[i].{sound_event_metrics,disease_metrics}.confusion_matrix_{raw,normalized}`. Rendered figures are listed in [§4](#4-training-curves--confusion-matrices).

Example — disease confusion matrix, seed 42, clean (rows = true, columns = predicted):

|  | COPD | Healthy | URTI |
|---|---|---|---|
| **COPD** | 1859 | 28 | 3 |
| **Healthy** | 34 | 59 | 14 |
| **URTI** | 38 | 36 | 11 |

### 3.4 Efficiency — model size, parameters, training time

| Metric | Per member | Ensemble (×5) |
|---|---|---|
| Total parameters | 2 232 839 | 11 164 195 |
| Trainable parameters | 1 215 047 | 6 075 235 |
| Model size | 8.75 MB | **43.75 MB** |
| Training time (clean) | see below | **5 050.62 s** (~84 min) |
| Training time (augmented) | see below | **7 069.33 s** (~118 min) |

**Per-member training time**

| Seed | Epochs (clean) | Total (clean) | s/epoch (clean) | Epochs (aug) | Total (aug) | s/epoch (aug) |
|---|---|---|---|---|---|---|
| 42 | 13 | 1 273 s | 97.9 | 15 | 1 628 s | 108.6 |
| 43 | 10 | 1 000 s | 100.0 | 12 | 1 312 s | 109.3 |
| 44 | 8 | 788 s | 98.5 | 10 | 1 108 s | 110.8 |
| 45 | 7 | 692 s | 98.9 | 12 | 1 345 s | 112.1 |
| 46 | 13 | 1 297 s | 99.8 | 15 | 1 676 s | 111.7 |

Epoch counts vary because early stopping (patience 4) fires at different points per seed.

### 3.5 Core deliverable — ensemble-disagreement unknown detection

The disagreement score is the **mean pairwise L2 distance between the five members' disease-head softmax vectors**, aggregated per patient. Evaluated as a binary detector: known test patients = 0, held-out unknown-disease patients = 1.

| Run | AUROC | AUPRC |
|---|---|---|
| Clean | **0.6566** | **0.3968** |
| SpecAugment | **0.6842** | **0.4354** |

**Reference points for interpretation:** AUROC 0.50 = random guessing, 1.00 = perfect separation. AUPRC baseline = the positive-class prevalence, **0.311** (19 unknown / 61 total patients).

So the clean ensemble separates unknown from known patients correctly about **66% of the time** — above chance, but far from clinically usable. This is exactly the bar Member B's cross-task disagreement mechanism must clear at M20.

---

## 4. Training Curves & Confusion Matrices

**51 figures total**, covering all 10 model runs (5 seeds × 2 augmentation conditions).

| Figure type | Count | Filename pattern |
|---|---|---|
| Training vs. validation **loss** vs. epochs | 10 | `M7_seed{42..46}_{clean,aug}_loss_curve.png` |
| Training vs. validation **accuracy** vs. epochs | 10 | `M7_seed{42..46}_{clean,aug}_accuracy_curve.png` |
| Validation **F1** vs. epochs | 10 | `M7_seed{42..46}_{clean,aug}_f1_curve.png` |
| **Disease** confusion matrix | 10 | `M7_seed{42..46}_{clean,aug}_disease_confusion_matrix.png` |
| **Sound-event** confusion matrix | 10 | `M7_seed{42..46}_{clean,aug}_sound_event_confusion_matrix.png` |
| Unknown-disagreement score histogram | 1 | `M7_unknown_disagreement_hist.png` |

Requirement #4 asks for loss and accuracy curves for *all* applied models — with a 5-member ensemble that means 5 curves per run, not one. All five are plotted separately. Full numeric epoch-by-epoch history (`train_loss`, `val_loss`, `train_sound_acc`, `train_disease_acc`, `val_sound_acc`, `val_disease_acc`, `val_sound_f1`, `val_disease_f1`, `val_combined_f1`, `epoch_time_s`, `lr`) is stored in `per_member_results[i].training_history` for independent re-plotting.

---

## 5. Augmentation Ablation (SpecAugment)

**Method:** SpecAugment applied on-the-fly to training spectrograms only — never to validation, test, or unknown/OOD data.

- `FrequencyMasking(freq_mask_param=15)` — masks up to 15 consecutive mel bins
- `TimeMasking(time_mask_param=25)` — masks up to 25 consecutive time frames

Masks are applied **after** log-mel conversion and **before** per-sample normalisation. Everything else — split, architecture, optimizer, seeds, epoch budget — is held constant, so the comparison is a clean single-variable ablation.

### 5.1 Clean vs. augmented — head-to-head

| Metric | Clean | SpecAugment | Δ |
|---|---|---|---|
| Sound-event accuracy | 0.5114 | 0.5352 | **+0.0238** ✅ |
| Sound-event F1 (macro) | 0.3670 | 0.3700 | +0.0030 ✅ |
| Disease accuracy | 0.9300 | 0.9284 | −0.0016 ➖ |
| Disease F1 (macro) | 0.5449 | 0.5260 | **−0.0189** ❌ |
| Unknown-detection AUROC | 0.6566 | 0.6842 | **+0.0276** ✅ |
| Unknown-detection AUPRC | 0.3968 | 0.4354 | **+0.0386** ✅ |
| Total training time | 5 050.62 s | 7 069.33 s | **+40.0%** ❌ |
| Total epochs run | 51 | 64 | +13 |

### 5.2 Augmented — sound-event classification

| Seed | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| 42 | 0.5451 | 0.4070 | 0.3900 | 0.3900 |
| 43 | 0.5389 | 0.3930 | 0.3680 | 0.3700 |
| 44 | 0.5350 | 0.3850 | 0.3690 | 0.3590 |
| 45 | 0.5221 | 0.3710 | 0.3700 | 0.3640 |
| 46 | 0.5346 | 0.3770 | 0.3650 | 0.3670 |
| **Mean** | **0.5352** | **0.3866** | **0.3724** | **0.3700** |

### 5.3 Augmented — disease classification

| Seed | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| 42 | 0.9297 | 0.5550 | 0.5310 | 0.5270 |
| 43 | 0.9308 | 0.5037 | 0.5505 | 0.5254 |
| 44 | 0.9284 | 0.5573 | 0.4967 | 0.5188 |
| 45 | 0.9284 | 0.5040 | 0.5290 | 0.5160 |
| 46 | 0.9250 | 0.5700 | 0.5700 | 0.5430 |
| **Mean** | **0.9284** | **0.5380** | **0.5354** | **0.5260** |

**Per-class disease F1 (augmented)**

| Seed | COPD | Healthy | URTI |
|---|---|---|---|
| 42 | 0.974 | 0.566 | 0.041 |
| 43 | 0.972 | 0.604 | **0.000** |
| 44 | 0.964 | 0.592 | **0.000** |
| 45 | 0.969 | 0.579 | **0.000** |
| 46 | 0.972 | 0.560 | 0.096 |

### 5.4 Verdict

SpecAugment is **not a clean win**. It helps the two things that matter most for the paper's argument — sound-event accuracy (+2.4 pts) and unknown-detection AUROC (+0.028) — and improves the Healthy class. But it **destroys URTI**: four of five seeds score exactly 0.000 F1, meaning those members never correctly identified a single URTI patient. Masking already-scarce spectral evidence from a 14-patient class removes the little signal there was. The cost is also 40% more training time.

**Recommendation:** report both. If a single configuration must be chosen for downstream M11/M14, use the **augmented** ensemble for the disagreement signal and the **clean** ensemble for any per-class disease reporting.

---

## Interpretation — Reading These Numbers Honestly

Headline accuracy is misleading on this dataset. Both test sets are severely imbalanced, so the correct comparison is against a majority-class baseline:

| Task | M7 result | Majority-class baseline | Verdict |
|---|---|---|---|
| Sound-event accuracy | 0.5114 | 0.5437 (always predict "Normal", 1132/2082) | **Below baseline** |
| Disease accuracy | 0.9300 | 0.9078 (always predict "COPD", 1890/2082) | **+2.2 pts only** |
| Unknown detection (AUROC) | 0.6566 | 0.5000 (coin flip) | **+0.157, modest but real** |

This is why **macro-F1 is the metric that should be reported in the paper**, not accuracy. The 93% disease accuracy is almost entirely the COPD class being 91% of the test cycles.

**What is genuinely working:**
- COPD detection is excellent and stable (F1 0.964–0.977 across all 10 runs)
- Seed-to-seed variance is very tight (disease accuracy spans 0.925–0.933; sound accuracy 0.504–0.545) — results are reproducible, not lucky
- The disagreement signal is above chance in both runs and improves with augmentation, which is the expected direction

**What is not working, and why:**
- **URTI (F1 0.00–0.20)** — only 14 patients in the entire dataset. No architecture fixes a class this rare; this is a dataset-level constraint
- **Wheeze / Both (F1 0.19–0.25)** — 12.8% and 7.3% of cycles respectively, and the loss is unweighted (see deviations below)
- **Unknown detection at 0.66 AUROC** — usable as a baseline, not as a product

---

## Known Deviations & Limitations

These are documented deliberately so reviewers and teammates are not surprised.

1. **Backbone substitution.** The project reference specifies M7 should build on the team's M1 from-scratch CNN checkpoint. ImageNet-pretrained MobileNetV2 was used instead, to satisfy faculty requirement #2 (pretrained model per member) and because M7 was trained independently without waiting on the M1 handoff. This is recorded in the `meta.notes` field of both results JSONs.

2. **Unweighted cross-entropy.** `Model_Training_Reference.md` specifies inverse-frequency class-weighted `CrossEntropyLoss` for the backbone-ablation group (M2/M3/M4) to counter ICBHI class imbalance. M7 uses **plain unweighted** cross-entropy on both heads. This likely contributes materially to the Wheeze/Both and URTI collapse and should be either re-run with weighting or explicitly justified in the manuscript.

3. **Small evaluation sample for the headline number.** The AUROC/AUPRC figures are computed over **61 patients** (42 known + 19 unknown). That is a small sample and the resulting AUROC carries wide uncertainty. A bootstrap confidence interval should be added before this number appears in the paper.

4. **Documentation conflict.** `Stage1_to_3_Detailed_Analysis_Report.md` contains a stage diagram labelling M5 and M7 as *"Skipped"*, which contradicts these completed results. One of the two documents is stale and must be reconciled before the M28 Model Zoo appendix is assembled.

5. **`meta.member_name` is still the placeholder `"YOUR_NAME_HERE"`** in both results JSONs. Fill this in before submission or merge.

6. **Disagreement uses the disease head only.** The sound-event head's probability vectors are not included in the disagreement computation. A joint-head variant is a cheap, obvious ablation that has not yet been run.

---

## Reproducibility

```
Platform        : Google Colab
GPU             : NVIDIA Tesla T4
PyTorch         : 2.11.0+cu128
Base seed       : 42
Ensemble seeds  : 42, 43, 44, 45, 46
Dataset         : ICBHI 2017 Respiratory Sound Database
                  (920 recordings, 126 patients, 6898 cycles, ~5.5 h audio)
```

Checkpoints are written every epoch and mirrored to Google Drive, so training survives a Colab disconnect, RAM crash, or session change. Both the member reports and the disagreement scores are cached to disk; re-running the notebook loads them instead of repeating inference. Set `FORCE_RECOMPUTE_DISAGREEMENT = True` to override.

---

## File Manifest

| File | Contents |
|---|---|
| `M7_Deep_Ensemble_3_4.ipynb` | Full training + evaluation notebook (32 cells, 11 sections) |
| `results_M7.json` | Protocol-compliant results, clean run — all 5 members |
| `results_M7_aug.json` | Protocol-compliant results, SpecAugment run — all 5 members |
| `member_reports_clean.json` | Cached per-member evaluation reports (clean) |
| `member_reports_aug.json` | Cached per-member evaluation reports (augmented) |
| `member_seed{42..46}_clean_result.json` | Individual member training artefacts (clean) |
| `member_seed{42..46}_aug_result.json` | Individual member training artefacts (augmented) |
| `disagreement_clean.json` | Per-patient disagreement scores + AUROC/AUPRC (clean) |
| `patient_split.json` | Exact patient-ID assignment for train / calibration / test / unknown |
| `M7_seed*_*.png` | 50 training-curve and confusion-matrix figures |
| `M7_unknown_disagreement_hist.png` | Disagreement-score distribution, known vs. unknown |

Results JSONs follow the shared `results_M<ID>.json` schema for merge into the **M28 Model Zoo & Reporting Appendix**.

---

## Position in the Pipeline

```
              M1 (backbone) ──► M7  Deep Ensemble N=5   ◄── you are here
                                 │
                                 ├──► M8   MC-Dropout variant
                                 ├──► M9   SNGP variant
                                 ├──► M10  Evidential head
                                 │
                                 ▼
                                M11  Post-hoc calibrators
                                 │    (temperature / vector / focal)
                                 ▼
                                M14  Conformal wrapper v1
                                 │
                                 ▼
                                M20  Ensemble disagreement (M7)
                                      vs. cross-task disagreement (Member B)
```

**Next step:** M8 (MC-Dropout) or M9 (SNGP) as the Tier-3 alternative uncertainty signal. M9's selling point is matching M7's uncertainty quality with **one** model and **one** forward pass instead of 5 models and 43.75 MB — that efficiency comparison must be reported explicitly.

---

<p align="center"><sub>Model M7 · Member D · Group 5 OWMTL Capstone · Completed 2026-08-05</sub></p>
