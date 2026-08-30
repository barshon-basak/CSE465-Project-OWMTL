# OWMTL — Consolidated Progress Report

**Generated:** 2026-08-30 by a full sweep of the repository.  
**Purpose:** single source of truth for writing `DRAFT_PAPER/main.tex`. Everything a section
of the paper could need is dumped here, with the file each number came from.

**Read alongside:**
- `DRAFT_PAPER/paper_requirement.md` — the CSE465 rulebook the paper must satisfy
- `CORRECTED_BASELINES.md` — the corrected M2/M3 table (authoritative)
- `Novelty Experiment/NOVELTY_STATUS.md` — the eight novelty directions in full
- `CLINICIAN_RESULTS_v1.md` — the physician labelling study
- `Asif's/audit/PROJECT_AUDIT.md` — automated validity audit
- `DECISION_2026-08-16_PIVOT.md` — why the project reframed

> **Rule used throughout:** every number here is traceable to a committed file. Where a number
> is contested, superseded, or must not be reported, that is stated inline rather than omitted.

---

## 0. Executive summary — what this project actually is now

The project began as a multi-task open-world respiratory-sound system (OWMTL). Three
independent findings redirected it:

1. **The evaluation was wrong**, in three separate ways, all found by auditing our own
   pipeline. Corrected baselines are far lower than what was originally reported.
2. **The concept-validity gate failed twice**, pre-registered, on real audio.
3. **A physician disagrees with ICBHI's own labels at chance level on crackles** — which
   reframes (2) from *our detector is bad* to *the reference standard is unreliable*.

The defensible contribution is therefore **a corrected evaluation protocol, a pre-registered
negative result, a label-reliability ceiling, and released tooling** — not a new SOTA model.

**The single most quotable pair of results:**

- A frozen AudioSet-pretrained AST linearly encodes **13 of 14** physics concepts
  (mean R² 0.41) with a random-projection control at **0/14** (mean R² −0.005).
- Those same concepts carry almost no *disease* information, and LoRA fine-tuning on the
  ICBHI task **degrades 13 of 14** concepts (mean ΔR² −0.091).

> **The model hears the sounds. The sounds do not predict the ICBHI disease labels.**

---

## 1. Dataset

### 1.1 ICBHI 2017 Respiratory Sound Database (primary)

| Property | Value |
|---|---|
| Recordings | 920 |
| Patients | 126 |
| Official split | 539 train / 381 test recordings |
| Annotated respiratory cycles | 6,898 (6,887 usable in M39 after min-length filter) |
| Sound-event classes | Normal, Crackle, Wheeze, Both |
| Disease classes used | COPD, Healthy, URTI (known) + 3 held-out unknown |
| Sample rate used | 16 kHz |
| Source | Kaggle `vbookshelf/respiratory-sound-database` |

**Cycle-level class distribution on the corrected official split (from M2 run):**

| Class | Test-set support |
|---|---:|
| Normal | 1579 |
| Crackle | 649 |
| Wheeze | 385 |
| Both | 143 |
| **Total test cycles** | **2756** |

Train cycles: 4025 · Test cycles: 2756

### 1.2 SPRSound BioCAS2022 (pediatric, used only for N8 covariate shift)

| Property | Value |
|---|---|
| Records | 1,949 (177 "Poor Quality" dropped) |
| Annotated events | 6,656 |
| Children | 243 |
| Age range | 0.2–16.2 y (median 4.4) |

Events are cut from the JSON start/end times to match ICBHI's raw unpadded annotated cycles.

---

## 2. The evaluation errors and their corrections — the paper's spine

### 2.1 Error 1 — non-standard ICBHI metric

The project reported `icbhi_score = (recall_macro + specificity_macro) / 2`, a macro average
over all four classes. The **official ICBHI 2017 challenge metric** pools the three abnormal
classes against Normal:

```
Se    = correctly-classified ABNORMAL events / all abnormal events   (Crackle+Wheeze+Both)
Sp    = correctly-classified Normal   events / all Normal   events
Score = (Se + Sp) / 2
```

**Measured inflation across 14 models: +0.11 mean, +0.22 worst**
(`Asif's/audit/ICBHI_SCORE_AUDIT.md`).

The macro variant also **masks pathologies**: one model reported 0.5137 while detecting 9% of
abnormal events; another reported 0.5506 with **specificity exactly 0.0000**.

### 2.2 Error 2 — the split was not the official one

The split loader searched for `ICBHI_Challenge_train_test.txt` (capital C) with exact-path
`os.path.exists()`. The real file is lowercase. On Linux it never matched, and the code
silently fell back to `pid <= 111` — producing **11 test patients, 7.1% of cycles**, while
reporting `split_method: "patient_independent_official_60_40"`. The literal string could not
disagree with the data.

### 2.3 Error 3 — the official split is not patient-disjoint

Patients **156 and 218** have recordings on both sides of the official ICBHI file. Policy
adopted: `drop_from_train` — their *train* recordings are dropped, the official *test* set is
left byte-identical so scores stay comparable to published work.

### 2.4 Error 4 — checkpoint selection read the test set

Found 2026-08-29. `train_model(..., val_loader=test_loader)`: the 60-epoch run evaluated the
**official test set every epoch** and kept the best-scoring epoch as `best_model.pth`. Every
"single held-out evaluation" therefore carried a 60-shot selection advantage.
`Model_Training_Protocol.md` writes "test/validation set" as one term throughout, so this was
the intended design project-wide, not a one-off slip.

**Fix:** a patient-grouped validation slice carved from TRAIN only
(`GroupShuffleSplit`, 15%, seeded). Results JSONs now carry `test_touched_times: 1`.

**Empirical size of the bias (M2):** 0.4671 (leaky) → 0.4720 (clean). Negligible *for this
model* — worth reporting honestly, since it means M2's result is not sensitive to the bug.

---

## 3. Corrected baselines — AUTHORITATIVE TABLE

Both runs share the identical corrected protocol. **These are the numbers for the paper.**

| Model | Architecture | **Official ICBHI** | Se | Sp | Accuracy | Macro-F1 | Params | Size MB | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 | 2D_CNN_4Block_w32_do0.3 | **0.4720** | 0.2005 | 0.7435 | 0.5116 | 0.2884 | 421,732 | 1.62 | 19 |
| M3 | mobilenet_v3_small | **0.5132** | 0.1980 | 0.8284 | 0.5591 | 0.3500 | 929,316 | 3.67 | 17 |

**M3 − M2 = +0.0412** on the official metric.

### 3.1 Per-class recall — why both sit below 0.52

| Model | Normal | Crackle | Wheeze | Both |
|---|---:|---:|---:|---:|
| M2 | 0.744 | 0.074 | 0.486 | 0.007 |
| M3 | 0.828 | 0.217 | 0.184 | 0.147 |

Both models collapse toward Normal: high specificity, very low sensitivity. Published ICBHI
SOTA on the official split is **~0.60–0.65**; M2 lands *below* the 0.50 a trivial
always-one-class model achieves. This is the corrected picture, not a broken run.

### 3.2 Inflation, per model — direct evidence for §2.1

| Model | Official | Macro (non-standard) | Inflation |
|---|---:|---:|---:|
| M2 | 0.4720 | 0.5480 | +0.0760 |
| M3 | 0.5132 | 0.5639 | +0.0507 |

### 3.3 Training configuration (both)

| Setting | M2 | M3 |
|---|---|---|
| Sample rate | 16000 | 16000 |
| n_mels | 128 | 128 |
| n_fft | 1024 | 1024 |
| hop_length | 160 | 160 |
| f_min | 50 | 50 |
| f_max | 2000 | 2000 |
| Duration (s) | 8.0 | 8.0 |
| Batch size | 32 | 16 |
| Epochs | 60 | 40 |
| LR | 0.001 | 0.001 |
| Optimizer | Adam | Adam |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Loss | inverse_frequency_class_weighted_CrossEntropyLoss | inverse_frequency_class_weighted_CrossEntropyLoss |
| Seed | 42 | 42 |
| GPU | Tesla T4 | Tesla T4 |
| Train time (s) | 612.9 | 433.9 |
| Time/epoch (s) | 10.2 | 10.8 |
| Inference (ms/sample) | 1.321 | 5.576 |

---

## 4. Complete model inventory

Every `results_M*.json` in the repository. **Split column is critical** — rows on different
splits are NOT comparable, and rows marked `patient_id_fallb` carry the 11-patient bug (§2.2).

| Model | Owner | Acc | Macro-F1 | Official | Macro | Params | MB | Ep | Split | File |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| M1 | Barshon | 0.5407 | 0.4844 | 0.6143 | 0.7181 | 421,732 | 4.85 | 60 | `patient_independent_60_40` | `Barshon's/M1/results_M1.json` |
| M11 | Barshon | 0.3573 | 0.3017 | — | 0.4983 | — | — | 30 | `patient_independent_60_20_20` | `Barshon's/M11/results_M11 (3).json` |
| M12 | Asif | 0.6138 | 0.5238 | 0.6138 | 0.7227 | 3,627,476 | 13.86 | 60 | `patient_independent_official_60_40` | `Asif's/M12/results_M12.json` |
| M13 | Barshon | 0.7209 | 0.6061 | — | 0.7137 | 4,153,556 | 16.61 | — | `patient_independent_stratified_60_` | `Barshon's/M13/results_M13.json` |
| M13cbm_independent | owmtl_concept_engine | 0.4884 | 0.4542 | — | — | — | — | — | `patient_independent_official_60_40` | `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_independent.json` |
| M13cbm_leaky | owmtl_concept_engine | 0.6512 | 0.5457 | — | — | — | — | — | `patient_independent_official_60_40` | `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_leaky.json` |
| M13cbm_opaque | owmtl_concept_engine | 0.7209 | 0.5973 | — | — | — | — | — | `patient_independent_official_60_40` | `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_opaque.json` |
| M13cbm_sequential | owmtl_concept_engine | 0.5581 | 0.4441 | — | — | — | — | — | `patient_independent_official_60_40` | `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_sequential.json` |
| M14 | Barshon | — | — | — | — | — | — | — | `patient_independent_60_20_20` | `Barshon's/M14/v1/results_M14.json` |
| M14 | Barshon | — | — | — | — | — | — | — | `patient_independent_60_20_20` | `Barshon's/M14/v2/results_M14.json` |
| M15 | Barshon | — | — | — | — | — | — | — | `—` | `Barshon's/M15/v4/results_M15.json` |
| M15 | Barshon | 0.0000 | 0.0000 | — | 0.0000 | 4,153,556 | 15.87 | — | `patient_independent` | `Barshon's/M15/v6/results_M15.json` |
| M16 | Barshon | — | — | — | — | — | 1.55 | — | `—` | `Barshon's/M16/results_M16.json` |
| M17 | Barshon | 0.8129 | 0.3844 | — | — | — | — | 20 | `—` | `Barshon's/M17/results_M17.json` |
| M18 | Barshon | 0.0411 | — | — | — | — | 13.68 | — | `—` | `Barshon's/M18/results_M18.json` |
| M19 | Barshon | — | — | — | — | 4,153,556 | — | — | `—` | `Barshon's/M19/results_M19.json` |
| M2 | Asif | 0.4009 | 0.3597 | — | 0.6167 | 421,732 | 1.62 | 60 | `official_icbhi_60_40_patient_disjo` | `Asif's/M2/17aug_run_v2/results_M2.json` |
| M2 | Asif | 0.4917 | 0.3518 | 0.4671 | 0.5780 | 421,732 | 1.62 | 60 | `official_icbhi_60_40_patient_disjo` | `Asif's/M2/29aug_run_v3/results_M2.json` |
| M2 | Asif | 0.5116 | 0.2884 | 0.4720 | 0.5480 | 421,732 | 1.62 | 60 | `official_icbhi_60_40_patient_disjo` | `Asif's/M2/m2_v4/results_M2.json` |
| M2 | Asif | 0.6138 | 0.5238 | 0.6138 | 0.7227 | 3,627,476 | 13.86 | 60 | `patient_independent_60_40_patient_` | `Asif's/M2/results_M2.json` |
| M20 | Barshon | — | — | — | — | 4,153,556 | — | — | `—` | `Barshon's/M20/results_M20.json` |
| M21 | Barshon | 0.5750 | — | — | — | 3,586,340 | — | — | `—` | `Barshon's/M21/results_M21.json` |
| M22 | Asif | 0.6524 | 0.5253 | 0.6495 | 0.7077 | 2,228,996 | 8.74 | 40 | `patient_independent_60_40_patient_` | `Asif's/M22/result_M22/results_M22.json` |
| M24 | Barshon | — | — | — | — | — | — | 30 | `patient_independent_70_30` | `Barshon's/M24/results_M24.json` |
| M28 | Barshon | — | — | — | — | — | — | — | `—` | `Barshon's/M28/results_M28.json` |
| M29 | Asif | — | — | — | — | 3,627,476 | — | — | `patient_independent_known_60_40; u` | `Asif's/M29/results_M29.json` |
| M3 | Asif | 0.4430 | 0.3816 | — | 0.6239 | 6,957,956 | 27.12 | 40 | `official_icbhi_60_40_patient_disjo` | `Asif's/M3/17aug_run_result/results_M3.json` |
| M3 | Asif | 0.5591 | 0.3500 | 0.5132 | 0.5639 | 929,316 | 3.67 | 40 | `official_icbhi_60_40_patient_disjo` | `Asif's/M3/29 aug run/results_M3.json` |
| M3 | Asif | 0.5915 | 0.4904 | 0.5895 | 0.6984 | 2,228,996 | 8.74 | 40 | `patient_independent_60_40_patient_` | `Asif's/M3/results_M3.json` |
| M30 | Asif | 0.6016 | 0.4675 | 0.5975 | 0.6777 | 7,957,724 | 30.63 | 30 | `patient_independent_60_40_patient_` | `Asif's/M30_v2/results_M30.json` |
| M30 | Barshon | 0.7275 | — | — | 0.8213 | 7,957,724 | 30.62 | 30 | `patient_independent_70_30` | `Barshon's/M30/results_M30.json` |
| M31 | Barshon | 0.5529 | 0.4351 | 0.5535 | 0.6377 | 3,726,297 | 14.24 | 30 | `patient_independent_70_30` | `Barshon's/M31/results_M31.json` |
| M32 | Barshon | 0.4802 | 0.4378 | 0.4733 | 0.6547 | 3,735,732 | 14.28 | 30 | `patient_independent_70_30` | `Barshon's/M32/results_M32.json` |
| M33 | Barshon | 0.3561 | 0.2393 | 0.3330 | 0.5506 | 13,186,008 | 50.30 | 30 | `patient_independent_70_30` | `Barshon's/M33/results_M33.json` |
| M33 | Barshon | 0.5857 | 0.4948 | 0.5832 | 0.6806 | 13,186,008 | 50.30 | 30 | `patient_independent_70_30` | `Barshon's/M33_v2/results_M33.json` |
| M34 | Barshon | 0.5795 | 0.5268 | 0.5754 | 0.7046 | 3,627,476 | 13.86 | 30 | `patient_independent_70_30` | `Barshon's/M34/results_M34.json` |
| M35 | Barshon | 0.6690 | 0.6420 | 0.6864 | 0.7839 | 3,627,476 | 13.86 | 30 | `patient_independent_70_30` | `Barshon's/M35/results_M35.json` |
| M35 | Barshon | 0.6766 | 0.6491 | 0.6719 | 0.7918 | 3,627,476 | 13.86 | 30 | `patient_independent_70_30` | `Barshon's/M35_v2/results_M35.json` |
| M36 | Barshon | 0.4765 | 0.2155 | 0.5052 | 0.5137 | 4,980 | 0.02 | 30 | `patient_independent_70_30` | `Barshon's/M36/results_M36.json` |
| M37 | Barshon | 0.6806 | 0.6599 | 0.6753 | 0.7969 | 3,635,700 | 13.89 | 30 | `patient_independent_70_30` | `Barshon's/M37_v2/results_M37.json` |
| M39 | Asif | — | — | — | — | 0 | 0.00 | — | `official_60_40_patient_independent` | `Asif's/M39/2nd_run_handoff/results_M39.json` |
| M39 | Asif | — | — | — | — | 0 | 0.00 | — | `official_60_40_patient_independent` | `Asif's/M39/M39_handoff/results_M39.json` |
| M6 | Barshon | 0.7689 | 0.4561 | — | — | 455,015 | 0.13 | 80 | `patient_independent_70_30` | `Barshon's/M6/result/results_M6.json` |
| M7 | Sami | 0.5183 | 0.3778 | — | — | 2,232,839 | 8.75 | 15 | `patient_independent_60_40_with_cal` | `Sami's/M7/M7_handoff/results_M7.json` |
| M7 | Sami | 0.5447 | 0.3901 | — | — | 2,232,839 | 8.75 | 15 | `patient_independent_60_40_with_cal` | `Sami's/M7/M7_handoff/results_M7_aug.json` |

---

## 5. Model roster by role (for the paper's "Applied Models" section)

| ID | What it is | Role in the paper | Status |
|---|---|---|---|
| M1 | 4-block CNN from scratch | provisional baseline | real, superseded by M2 |
| M2 | Tuned 2D CNN (sweep-selected) | **primary CNN baseline** | ✅ corrected run |
| M3 | MobileNetV2 (ImageNet-pretrained) | lightweight/efficiency arm | ✅ corrected run |
| M4 | AST (Audio Spectrogram Transformer) | transformer arm | ⚠️ no results JSON in repo |
| M6 | OpenMax + Weibull | open-set baseline | real; AUROC 0.4516 = below chance |
| M7 | Deep ensemble (Sami) | ensembling | real; clean + augmented |
| M12 | Backbone decision study | selects M2 as backbone | real |
| M13 | Prototypical / few-shot disease head | few-shot arm | real (reproduced by N3) |
| M22 | M3 + SpecAugment | **augmentation ablation** | ⚠️ needs corrected-split re-run |
| M23 | AST + SpecAugment (Farhana) | transformer + aug | notebook only, no results |
| M29 | Post-hoc OOD on frozen M2 | open-set baseline | real; Energy AUROC 0.6466 |
| M30 | M2+M3 gated fusion | fusion | real; loses to M2 alone |
| M31–M37 | Barshon's variants (70/30 split) | not comparable to official split | see audit |
| M35 | best 70/30 model | — | official 0.6864 but different split |
| M37 | LoRA PEFT (on M2 CNN, not a FM) | misnamed — no foundation model | see N1 |
| M39 | Physics concept extraction (pure DSP) | **gate G2 evidence** | ✅ real, FAILED gate |

> ⚠️ **Course requirement gap:** `paper_requirement.md` §4 demands **at least 4 transformer
> models** and **one pretrained model per member (4)**. Currently the only transformer with a
> committed results JSON is **none** — M4 (AST) and M23 (AST+SpecAugment) have notebooks but no
> `results_M*.json` in the repo, and M37 is a LoRA wrapper around the project's own CNN, not a
> transformer. **This is the single largest unmet requirement.** See §12.

---

## 6. Novelty experiments (N1–N8) — all eight ran

Source: `Novelty Experiment/NOVELTY_STATUS.md`. These correspond to Dr. Khan's eight suggested
directions. Every script refuses to state a result its CI does not support.

| # | Direction | Verdict |
|---|---|---|
| N1 | Foundation-model concept probing (AST + LoRA) | ✅ **strongest positive** |
| N2 | Concept-space open-set recognition | ✅ complete — parity, not a win |
| N3 | Prototypical few-shot disease head | ✅ complete — no few-shot advantage |
| N4 | Calibration-aware honest operating point | ✅ complete — 3 findings |
| N5 | Concept leakage / faithfulness audit | ✅ complete — leakage confirmed w/ CI |
| N6 | Physics-derived concept bottleneck | ✅ complete — **committed result does not replicate** |
| N7 | Clinician concept intervention | ✅ complete — well-instrumented negative |
| N8 | Pediatric physics-fragility | ✅ **second real positive** |

### 6.1 N1 — Foundation-model concept probing ⭐ HEADLINE

Linear probe of 14 physics concepts from frozen encoders. 6,898 cycles, patient-grouped CV,
ridge R², bootstrap CI, within-patient permutation null.

| Space | Concepts above chance | Mean probe R² |
|---|---:|---:|
| **AST frozen** (`MIT/ast-finetuned-audioset-10-10-0.4593`, 86.5 M) | **13 / 14** | **0.4102** |
| AST + LoRA (r=8, 294,912 trainable = 0.341%) | 13 / 14 | 0.3195 |
| M2 CNN (scratch-trained on this task) | 12 / 14 | 0.4289 |
| **Random-projection control** | **0 / 14** | **−0.0054** |

**Per-concept R² from the frozen M2 encoder:**

| Concept | R² | 95% CI | Concept | R² | 95% CI |
|---|---:|---|---|---:|---|
| wheeze_duration_ratio | 0.777 | [0.766, 0.786] | papr_db | 0.520 | [0.498, 0.540] |
| spectral_flatness | 0.759 | [0.717, 0.790] | crackle_presence | 0.501 | [0.484, 0.516] |
| rhonchi_presence | 0.648 | [0.632, 0.664] | coarse_crackle_ratio | 0.366 | [0.351, 0.383] |
| wheeze_presence | 0.605 | [0.586, 0.623] | low_high_freq_ratio | 0.363 | [0.313, 0.406] |
| dominant_freq_hz | 0.548 | [0.479, 0.621] | crackle_rate_hz | 0.356 | [0.328, 0.381] |
| wheeze_dominant_freq_hz | 0.289 | [0.247, 0.328] | transient_timing_centroid | 0.276 | [0.258, 0.295] |
| inspiratory_energy_fraction | 0.003 | [−0.025, 0.030] | fine_crackle_ratio | −0.008 | [−0.117, 0.036] |

**LoRA effect (adapted − frozen): 13 of 14 concepts DEGRADE, mean ΔR² = −0.0907.**

| Most degraded | ΔR² | Least affected | ΔR² |
|---|---:|---|---:|
| **rhonchi_presence** | **−0.219** | wheeze_dominant_freq_hz | −0.033 |
| dominant_freq_hz | −0.183 | inspiratory_energy_fraction | −0.006 |
| spectral_flatness | −0.154 | fine_crackle_ratio | +0.008 |
| papr_db | −0.142 | | |
| crackle_presence | −0.128 | | |

**Reading (three claims, all defensible):**
1. An AudioSet-pretrained FM already encodes clinical respiratory acoustics without ever seeing
   a respiratory corpus. The random control confirms this is not a dimensionality artefact.
2. Fine-tuning on the ICBHI task **systematically destroys** that encoding — adapting 0.34% of
   parameters strips ~a tenth of recoverable concept variance across almost every concept.
3. The concept it destroys most (`rhonchi_presence`, −0.219) is the one N5 independently finds
   to be the *only* concept carrying disease information. Two experiments, different data and
   estimators, converge on the same concept from opposite directions.

### 6.2 N2 — Concept-space open-set recognition

104 known / 19 unknown patients, official 60/40, patient level, unknown group never fitted.

| Space | MSP | Entropy | **Energy** | Mahalanobis | kNN |
|---|---:|---:|---:|---:|---:|
| concept 14-d | 0.612 | 0.622 | **0.627** | 0.520 | 0.525 |
| embedding 768-d | 0.555 | 0.551 | 0.603 | 0.486 | 0.514 |
| concat | 0.611 | 0.613 | **0.641** | 0.479 | 0.519 |

Best: concat/energy **0.6414** [0.496, 0.779]. **Every CI spans chance.**

**Paired bootstrap, concept − embedding** (same resample for both — the only valid comparison):

| Detector | Δ AUROC | 95% CI | p |
|---|---:|---|---:|
| entropy | +0.071 | [−0.108, 0.240] | 0.405 |
| msp | +0.058 | [−0.122, 0.225] | 0.503 |
| mahalanobis | +0.034 | [−0.089, 0.171] | 0.625 |
| energy | +0.023 | [−0.164, 0.209] | 0.802 |
| knn | +0.011 | [−0.135, 0.162] | 0.895 |

**Reading.** A 14-dimensional clinically-named score matches a 768-dimensional opaque one; no
paired difference is distinguishable from zero. Publishable framing: *you pay nothing in
detection performance for full interpretability of the open-set score.* At n=19 unknown patients
no CI can exclude chance — a property of the corpus, not the method.

### 6.3 N3 — Prototypical few-shot disease head

61 train / 43 test patients, 200 episodes per k, paired against a linear head on the identical
support set.

| k | concept proto | concept linear | **M2 proto** | **M2 linear** | verdict (M2) |
|---|---:|---:|---:|---:|---|
| 1 | 0.309 | 0.293 | 0.424 | 0.394 | not shown to differ |
| 2 | 0.312 | 0.315 | 0.441 | 0.417 | not shown to differ |
| 5 | 0.303 | 0.359 | 0.483 | 0.485 | not shown to differ |
| 10 | 0.262 | 0.424 | 0.505 | 0.525 | not shown to differ |
| 20 | *refused* | — | *refused* | — | k > rarest train class (10 patients) |
| all-train | 0.231 | 0.427 | **0.548** [0.486, 0.596] | 0.502 | — |

**Reading.** The M2 arm reproduces M13 (0.548 vs M13's 0.6061, overlapping at n=43) — M13's
number is sound. But prototypical is **statistically indistinguishable from a plain linear head**
at every k. The classic few-shot advantage does not appear on this task in either space.
The k=20 row is **refused rather than fabricated** — URTI has only 10 training patients.

### 6.4 N4 — Calibration-aware honest operating point

M2 embedding space; 42 fit / 19 calibration / 43 test patients.

| Quantity | Value |
|---|---|
| Fitted temperature | **4.89** — severely overconfident |
| ECE / MCE | 0.4049 → **0.1771** / 0.743 → 0.483 |
| Brier / NLL | 0.808 → 0.569 / 2.093 → 0.885 |
| Accuracy | 0.5116 → 0.5116 (**invariant, as it must be**) |
| AURC | **0.3424** |
| Selective acc @ cov 100/90/80/70/50% | 0.512 / 0.579 / **0.618** / 0.600 / 0.619 |
| Clinical point @ target Se 0.90 | Se **0.8966** [0.759, 1.000], Sp **0.5714**, referral **0.744** |
| Expected cost @ FN:FP = 10:1 | 0.8372 per patient |
| Conformal empty-set rate, known vs unknown | **0.0 vs 0.0** (mean set size 1.70 vs 1.84) |

**Three findings:**
1. **The accuracy-invariance check catches a bug in M11.** Temperature scaling is monotone, so
   accuracy *cannot* change. M11 reports 0.3573 → 0.6595 under calibration — not like-for-like,
   and its ECE improvement is not interpretable. Ours holds at 0.5116 exactly.
2. **Selective prediction works in embedding space but not concept space.** Accuracy rises
   0.512 → 0.618 at 80% coverage on M2 embeddings; in concept space it *fell* (0.465 → 0.381).
3. **The M14 conformal paradox, stated correctly.** Coverage is guaranteed by construction, so
   reporting it is circular. The informative number is the empty-set rate on unknown patients:
   0.0, identical to knowns, with unknowns getting *larger* sets (1.84 vs 1.70). The wrapper
   detects nothing. That is what M14 should have reported instead of 95.45% coverage.

**At the sensitivity a clinician would demand, the model refers 74% of all patients.**

### 6.5 N5 — Concept leakage / faithfulness audit

6,311 cycles, 104 patients, patient-grouped CV, features regenerated independently.

| Quantity | Value |
|---|---|
| Leakage I(y;f\|c) | **0.2025 bits** (12.77% of base) — committed run said 0.2209 |
| Patient-bootstrap CI | **[0.1096, 0.3292]** — **excludes zero** |
| Permutation null | **−0.0086** [−0.0149, −0.0013], n=5 |
| Verdict | **leakage exceeds the estimator's own bias** |
| Accuracy: concepts / +features / features | 0.9086 / 0.9303 / 0.9287 |

**Drop-one-concept importance:**

| Concept | Bits lost if removed |
|---|---:|
| **rhonchi_presence** | **+0.0293** |
| wheeze_presence | +0.0049 |
| crackle_presence | +0.0036 |
| papr_db | +0.0033 |
| dominant_freq_hz | +0.0029 |
| inspiratory_energy_fraction | +0.0018 |
| spectral_flatness / coarse_crackle_ratio / low_high_freq_ratio | +0.0015 – 0.0016 |
| crackle_rate_hz, fine_crackle_ratio | +0.0004 – 0.0005 |
| wheeze_dominant_freq_hz, transient_timing_centroid | ≈ 0 |
| wheeze_duration_ratio | **−0.0039** (removing it *helps*) |

**One concept out of fourteen carries the bottleneck.** `rhonchi_presence` contributes 6× the
next; five are dead (≤0.001 bits) and one is actively harmful. This explains N6 and N7: a
bottleneck resting on one concept has nothing to intervene on.

The independent regeneration reproduces the committed number to within 0.02 bits from a feature
matrix rebuilt with different code — a real reproducibility check that passed.

### 6.6 N6 — Physics-derived concept bottleneck ⚠️ COMMITTED RESULT DOES NOT REPLICATE

5 seeds, stratified bootstrap CIs, 43 test patients, all four modes.

| Mode | Accuracy | Macro-F1 | 95% CI | Seed sd | Committed F1 (1 seed) |
|---|---:|---:|---|---:|---:|
| **independent** (true bottleneck) | 0.5814 | **0.5254** | [0.333, 0.700] | 0.059 | 0.4542 |
| leaky | 0.6512 | 0.4802 | [0.385, 0.554] | 0.021 | 0.5457 |
| sequential | **0.6744** | 0.4526 | [0.358, 0.540] | 0.032 | 0.4441 |
| opaque (upper bound) | 0.5814 | **0.4378** | [0.332, 0.521] | 0.019 | 0.5973 |
| *shuffled-concept control* | 0.4884 | 0.2963 | [0.197, 0.393] | — | *never run* |

**⚠️ The monotone ordering inverts.** With five seeds instead of one, the strict bottleneck
(`independent`) has the **highest** macro-F1 and unconstrained `opaque` the **lowest** — the
opposite of the committed run. McNemar independent vs opaque: **p = 1.0000**.

The claimed *"interpretability cost of 0.2326 accuracy / 0.1432 macro-F1"* was a **single-seed
artefact**. The corrected statement: at n=43 there is **no measurable interpretability cost in
either direction**. **This claim must not go into the paper.**

**The control the committed run never had:**

| Metric | real − shuffled | 95% CI | p | Verdict |
|---|---:|---|---:|---|
| accuracy | +0.0930 | [−0.093, 0.279] | 0.389 | not shown to differ |
| macro-F1 | **+0.2291** | **[0.004, 0.441]** | **0.045** | **real concepts beat shuffled** |

The physics concepts carry something beyond their marginal distribution — but only on macro-F1
and only marginally (CI lower bound 0.004). Accuracy is dominated by COPD (64 of 104 patients).
**Report both metrics**; quoting either alone misleads.

### 6.7 N7 — Clinician concept intervention (SIMULATED)

Intervention curve: start from "no findings recorded" (train population mean), restore k true
concept values, 43 test patients.

| Ordering | independent k=0 → k=14 | leaky k=0 → k=14 |
|---|---|---|
| by sensitivity | 0.5814 → 0.5349 (**−0.047**) | 0.6512 → 0.6279 (**−0.023**) |
| random order | 0.5814 → 0.5349 (−0.047) | 0.6512 → 0.6279 (−0.023) |
| shuffled-concept control | 0.5814 → 0.4419 (−0.140) | 0.6512 → 0.6512 (0.000) |

**Directed interventions** (clinician asserts a finding; effect on p(COPD)):

| Assertion | independent | leaky |
|---|---|---|
| crackle_presence → COPD | +0.164 (88.4% ↑) ✓ | −0.010 (11.6% ↑) ✗ |
| wheeze_presence → COPD | −0.019 (32.6% ↑) ✗ | −0.000 (27.9% ↑) ✗ |
| rhonchi_presence → COPD | −0.277 (11.6% ↑) ✗ | +0.001 (30.2% ↑) ✓ |

**Reading.**
1. Correcting concepts does **not** improve diagnosis — it slightly degrades it, in both modes.
   The sensitivity ordering is *identical* to a random ordering.
2. **The leaky mode bypasses the concept layer almost entirely** — per-concept sensitivity
   collapses from ~0.70 (independent) to ~0.03 (leaky), a 20× drop. Given the option, the model
   routes around the concepts. Cleanest evidence for the leakage argument in the project.
3. Five of six directed interventions push the probability the **wrong way**.

**Every number here is labelled SIMULATED — there is no clinician in this loop.** The hook is
wired: drop `clinician_corrections.csv` (`patient,concept,value`) into the folder and it reruns.

### 6.8 N8 — Pediatric physics-fragility ⭐ SECOND POSITIVE

**Pre-registered directional predictions, fixed in code before any pediatric audio was read:**
`wheeze_dominant_freq_hz` higher, `dominant_freq_hz` higher, `low_high_freq_ratio` lower,
`rhonchi_presence` lower. Other ten concepts are the control set.

**Cross-corpus (adult ICBHI → pediatric SPRSound), Cohen's d:**

| Concept | d | 95% CI | adult-vs-child AUROC | Pre-registered |
|---|---:|---|---:|---|
| `dominant_freq_hz` | **+1.382** | [1.241, 1.558] | 0.930 | **higher ✓** |
| `low_high_freq_ratio` | **−0.512** | [−0.574, −0.472] | 0.126 | **lower ✓** |
| `rhonchi_presence` | **−3.870** | [−3.959, −3.784] | 0.015 | **lower ✓** |
| `wheeze_dominant_freq_hz` | −0.179 | [−0.207, −0.149] | 0.545 | higher ✗ |

**Within-cohort age gradient (confound-free arm), Spearman ρ vs age:**

| Concept | ρ | 95% CI | Pre-registered |
|---|---:|---|---|
| `dominant_freq_hz` | **−0.259** | [−0.348, −0.156] | **negative ✓** |
| `wheeze_dominant_freq_hz` | **−0.240** | [−0.310, −0.159] | **negative ✓** |
| `low_high_freq_ratio` | **+0.300** | [0.187, 0.399] | **positive ✓** |
| `rhonchi_presence` | −0.090 | [−0.152, −0.031] | positive ✗ |

Concept-space MMD² = **0.4805** [0.457, 0.502] (Gap7 reported embedding MMD 0.4236, no CI).

**Reading.** Both arms give 3/4. In the confound-free age arm **all three frequency concepts
confirm the mechanism with intervals excluding zero** — as a child grows, dominant and wheeze
frequencies fall and the low/high energy ratio rises, exactly what inverse scaling with airway
calibre predicts, with corpus, device, protocol and annotator all held fixed.

> *"Adult-tuned acoustic priors degrade on pediatric airways via frequency scaling"* is now a
> **mechanistic, pre-registered, confound-controlled claim** rather than an unexplained MMD.

`rhonchi_presence` misses in *opposite* directions in the two arms — report as unresolved.

**⚠ Methodological correction to N8's own design.** The first version judged the mechanism by a
one-sided binomial sign test over 4 predictions. With n=4 that test **floors at p = 0.0625** —
even a perfect 4/4 can never reach p < 0.05 — so it hardcoded "NOT supported" regardless of the
data. The criterion is now per-concept.

---

## 7. Concept extraction and the pre-registered gate (M39)

**Pure DSP — no model, no training, no learned parameters.** 6,887 cycles, 4,251 train /
2,636 test, 79/47 patients, corrected official split.

**Gate G2 rule (pre-registered):** AUROC ≥ 0.65 **AND** 95% DeLong CI excludes chance, on test.

| Concept | Run 1 | Run 2 | 95% CI (run 2) | AUPRC | Prevalence | Gate |
|---|---:|---:|---|---:|---:|---|
| `crackle_score` | 0.5506 | **0.5580** | [0.533, 0.583] | 0.308 | 0.267 | **FAIL** |
| `wheeze_score` | 0.5729 | **0.5340** | — | 0.190 | 0.174 | **FAIL** |

**Verdict: 0 of 2 validatable concepts passed. Two runs, then stop** — the test split was read
exactly twice and no third revision was attempted. This is the methodological spine.

**Why the AUROC floor matters:** an earlier CI-only rule passed at AUROC 0.55, because at
n≈2,600 even a trivial effect is significant. *A gate that cannot fail is not a gate.*

**Concept taxonomy (9 concepts):** 2 validatable against ICBHI (crackle/wheeze presence),
5 proxy, 2 descriptive. **The 2/5/2 split is itself a finding** about what this benchmark can
validate.

**Synthetic-signal validation:** the extractors provably respond to constructed crackles and
wheezes (**43/43 tests pass**), which makes the real-audio failure interpretable rather than a
code bug. Three DSP bugs were caught this way: band leakage scoring a 400 Hz tone as rhonchi
0.970; brick-wall FFT Gibbs ringing creating phantom crackles; detection firing on 1e-16
numerical noise.

---

## 8. Clinician validation study — the reliability ceiling

**Design:** 1 physician, blind, 120 clips + 12 hidden duplicates, 6 calibration exemplars.
Source: `CLINICIAN_RESULTS_v1.md`.

### 8.1 Quality gate — did he agree with himself? **PASS**

| Measure | Result |
|---|---|
| crackles, exact | **7/8 (88%)** |
| wheeze, exact | **7/8 (88%)** |
| crackles, presence-only (all 12) | **11/12 (92%)** |

Pack rule was *">3 of 12 self-disagreements → task too hard from recordings"*. He had **one**.
**This is what makes everything below interpretable.**

### 8.2 Clinician vs ICBHI — the core finding

Usable clips, definite calls only, excluded **per column** (n=108 crackle, 110 wheeze).

| | Physician + | ICBHI + | Raw agreement | κ (95% CI) | Sensitivity (95% CI) |
|---|---:|---:|---:|---|---|
| **crackle** | 23 | 52 | 52.8% | **+0.035** (−0.122, +0.192) | **23.1%** (12.0, 35.3) |
| **wheeze** | 16 | 37 | 71.8% | **+0.266** (+0.085, +0.445) | 29.7% (15.4, 45.2) |

**Crackle κ's CI includes zero.** Against a literature benchmark of physician-vs-physician
crackle κ ≈ 0.62 (Aviles-Solis 2016), agreement with ICBHI's crackle labels is statistically
**indistinguishable from chance**.

**That 23.1% is a near-exact independent replication of Tzeng et al. (2025), where seven senior
physicians reached 23.23% on this same corpus** — different rater, different clip sample, and
calibration exemplars their study did not report providing.

**Sensitivity analysis** (ambiguous treated as positive): crackle κ −0.003 [−0.166, +0.161],
sens 25.9% [14.8, 37.9]; wheeze κ +0.289 [+0.112, +0.460], sens 35.0% [20.5, 50.0]. The finding
does not depend on that judgement call.

### 8.3 Our extractors scored against the clinician instead of ICBHI

| Concept | vs ICBHI (M39) | **vs clinician** | 95% CI | Reading |
|---|---:|---:|---|---|
| `crackle_score` | 0.5580 | **0.5143** | [0.396, 0.633] | chance against **both** → our detector |
| `wheeze_score` | 0.5340 | **0.6599** | [0.503, 0.817] | **improves, clears the 0.65 gate** |
| `rhonchi_score` | — | **0.7090** | [0.480, 0.938] | best point estimate, n=10 positives |
| `crackle_fine_ratio` | — | 0.4803 | [0.217, 0.744] | **underpowered** — 4 negatives |

**The answer differs by concept, which is the interesting part:**
- **Crackle: it's us.** ~0.51–0.56 against both reference standards. Changing the reference does
  not rescue it.
- **Wheeze: it's substantially the labels.** Same detector scores 0.534 vs ICBHI and **0.660 vs a
  physician**. The detector did not change; only the reference standard did.

**Do not over-read the wheeze result** — n=16 positives, CI lower bound 0.503.

### 8.4 The `ambiguous` label — our instrument bug

We shipped **six** calibration exemplars including `example_ambiguous.wav`, but the answer
options only offered `none/fine/coarse/both/unsure`. **We gave a sixth reference sound with no
matching answer.** He matched the sound to our exemplar and wrote our word for it.

Confirmed with him: `ambiguous` means *"none of the named classes fit"*, **not** *"I can't tell"*
(`unsure` used once; `ambiguous` eight times; confidence 43% high vs 89% for `none`).

**~7% of clips contained a sound fitting none of the four named categories** — an *anchored*
judgement, since he had an exemplar. **He also marked 12.9% of clips `unusable`** — a real
statement about ICBHI's audio quality.

**Lesson:** every reference exemplar must have a matching answer option.

### 8.5 Inter-rater — PRE-REGISTERED, DATA PENDING

A second rater on ~30 shared clips is in progress. The interpretation is **fixed in advance**
(`INTER_RATER_READINGS` in `analyze_clinician_labels.py`):

| | Condition | Meaning |
|---|---|---|
| **A** | κ_inter ≥ 0.40 **and** κ_inter − κ_ICBHI ≥ 0.20 | clinicians agree with each other, not ICBHI → **the labels are the unreliable term** |
| **B** | κ_inter < 0.40 | clinicians don't agree either → ceiling belongs to the *task* |
| **C** | κ_inter ≥ 0.40 **and** κ_ICBHI ≥ 0.40 | no ceiling → **our extractors are simply weak** |

Writing (C) down in advance is what makes (A) or (B) credible. At n≈30 every interval will be
wide; the script marks anything under n=30 descriptive rather than confirmatory.

---

## 9. Open-set / open-world results

| Model | Method | AUROC | AUPR | Unknown recall | n unknown | Verdict |
|---|---|---:|---:|---:|---:|---|
| M6 | OpenMax + Weibull | **0.4516** | 0.2352 | 0.0255 | 19 | **below chance** |
| M29 | Energy (post-hoc, frozen M2) | **0.6466** | 0.4131 | 0.1053 | 19 | best zero-training |
| N2 | concat / Energy | **0.6414** [0.496, 0.779] | — | — | 19 | CI spans chance |
| N2 | concept-only / Energy | 0.627 | — | — | 19 | CI spans chance |

**Consequence for the project's novelty claim:** M29's post-hoc Energy score, which requires
**zero training**, beats M6's OpenMax (0.6466 vs 0.4516). Any proposed mechanism must clear
**0.6466**, not 0.4516. Several earlier claims in the repo compare against the sub-chance
baseline and are invalid for that reason (audit code `comparison_against_subchance_baseline`).

**Setup:** 104 known / 19 unknown patients, official 60/40, patient-level aggregation, unknown
group never fitted (`unknown_group_never_fitted: true`).

---

## 10. Human benchmarks — the literature ceiling

### Tzeng et al., *JMIR AI* 2025;4:e67239 (arXiv:2407.13895), Table 4

Seven **senior physicians** blindly annotated a random 25% of the ICBHI test set.

| Condition | Accuracy | **Sensitivity** | Specificity | **ICBHI score** | Confidence (SD) |
|---|---:|---:|---:|---:|---:|
| Clean | 49.40% | **23.23%** | 72.32% | **47.77%** | 2.88 (1.50) |
| Noisy | 47.59% | 16.77% | 74.58% | 45.68% | 2.32 (1.29) |
| Denoised | 51.51% | 28.38% | 71.75% | 50.07% | 2.65 (1.36) |

**Seven senior physicians reach a 47.77% ICBHI score on clean ICBHI audio, missing ~77% of the
cycles ICBHI labels abnormal, at mean self-rated confidence 2.88/5.**

**Caveat:** the paper does not state that physicians were given ICBHI's annotation conventions
or calibration examples. Our listening pack ships six reference exemplars, so our clinician is
*better* calibrated — a defensible methodological difference worth stating.

### Aviles-Solis et al. 2016

12 physicians: **κ < 0.40** for detailed adventitious-sound descriptions; κ = 0.62 / 0.59 for
combined crackle / wheeze presence. **Simplifying the taxonomy increases agreement.**

**The argument this licenses:** `crackle_fine_ratio` targeted a distinction humans agree on at
κ < 0.40. **Its ceiling was set before any DSP was written.** Generalise carefully — this bounds
*concept-level validation against these labels*, not respiratory ML overall.

---

## 11. Automated project audit

`Asif's/audit/audit_project.py` — 51 files scanned, no GPU, ~1 s.

| Severity | Count | Meaning |
|---|---:|---|
| 🔴 CRITICAL | 19 | The result does not support the claim made on it |
| 🟡 WARNING | 87 | Needs resolving before submission |
| ⚪ INFO | 43 | Worth knowing, not blocking |

**Models with critical findings:** M6, M14, M15, M16, M18, M19, M20, M21, M30, M33, M35, M36
**Files with no findings:** M2, M12, M18(csv), M22, M35

### Selected critical findings — these must not appear in the paper as results

| Model | Code | Detail |
|---|---|---|
| M6 | `discrimination_at_or_below_chance` | open-set AUROC 0.4516 ≤ 0.5 |
| M14 | `discrimination_at_or_below_chance` | AUROC 0.4522 (v1), 0.4809 (v2) |
| M15 | `icbhi_score_unverifiable` | icbhi_score 0.0000, no confusion matrix committed |
| M16/M18/M19/M20/M21 | `not_protocol_compliant` | loose metric files, not §4 schema |
| M30 | `icbhi_score_unverifiable` | no confusion matrix to recompute from |
| M33 | `normal_detection_collapse` | official Sp < 0.15 |
| M35 | `best_epoch_is_first` | the model never really trained |
| M36 | `abnormal_detection_collapse` | official Se < 0.15 |

### Known synthetic-data models

Per `Asif's/CLAUDE.md`: **M11, M13, M15, M17, M18, M19, M20, M21, M24** were built on
`torch.randn` / `np.random` data, not ICBHI. M15's Dataset is *named* `ICBHI_OWL_Dataset` and
returns Gaussian noise. **No number from those models is a result.**

⚠️ *Caveat:* N3 reproduced M13's few-shot number on real M2 embeddings (0.548 vs M13's 0.6061,
overlapping at n=43), so M13's specific figure appears sound even if its pipeline was flagged.
Verify per-model before citing.

**Verified real:** M1, M2, M3, M4, M6, M12, M22, M29, M30, M39, M7, and all of N1–N8.

---

## 12. ⚠️ COURSE-REQUIREMENT GAP ANALYSIS

Against `DRAFT_PAPER/paper_requirement.md` §4. **This is the section to act on.**

| # | Requirement | Status | Gap |
|---|---|---|---|
| 1 | Apply ALL preprocessing techniques | 🟡 partial | log-mel + SpecAugment done; no systematic preprocessing ablation |
| 2 | Pretrained DL models — **one per member (4)** | 🟡 partial | M3 (MobileNetV2), M4 (AST), M23 (AST+SpecAug) exist; **M4/M23 have no results JSON** |
| 3 | **At least 4 transformer models** | 🔴 **NOT MET** | **zero transformers with committed results.** M37 is LoRA on the project's own CNN, not a transformer |
| 4 | Acc/Prec/Rec/F1 for ALL models | 🟡 partial | present for most; missing for M4, M23; several models lack per-class |
| 5 | Params / size / training time for ALL | 🟡 partial | present for most real models; see §4 table for blanks |
| 6 | Normalized confusion matrix + curves for BEST | ✅ available | `Asif's/M2/m2_v4/*.png`, `Asif's/M3/29 aug run/*.png` |
| 7 | Augmentation → same metric set | 🔴 blocked | **M22 has not been re-run on the corrected split** |
| 8 | Novelties → same metric set | ✅ done | N1–N8, §6 |
| 9 | **XAI on the BEST model** | 🔴 **NOT MET** | no LIME/SHAP/Grad-CAM anywhere in the repo |
| 10 | Ablation study on BEST model | 🟡 partial | N6 (bottleneck modes), M12 (backbone), M22 (SpecAugment) — needs assembling into the slide-15 format |

### The three hard blockers

**1. Four transformer models (requirement 3).** Nothing in the repo satisfies this. The closest
assets: M4 (AST notebook, results in Barshon's Drive, never committed), M23 (AST+SpecAugment
notebook by Farhana, no results). N1 *did* run a frozen AST + LoRA over all 6,898 cycles and
produced embeddings — but as a **concept probe**, not a 4-class classifier with a metrics JSON.

> **Cheapest path:** N1's AST embedding pass already exists (`embeddings/ast_frozen.npy`,
> `ast_lora.npy`). A linear/MLP head on those embeddings, evaluated with the standard metric
> suite on the corrected split, yields **two** transformer rows (AST-frozen, AST-LoRA) for
> little compute. Two more (e.g. a Swin/ViT spectrogram model, or PANNs/BEATs) would be needed.

**2. XAI (requirement 9).** Nothing exists. Grad-CAM on M2's spectrogram input is the natural
choice — M2 is a plain CNN, so Grad-CAM is a few lines. **N1's concept-probe result is arguably
a stronger form of interpretability evidence and should be presented alongside**, but it does
not substitute for the required saliency figure.

**3. M22 on the corrected split (requirement 7).** The augmentation ablation is the only
remaining training run needed. The notebook is patched and verified; it has not been run.

---

## 13. Available figures

171 PNGs in the repo. The ones the paper needs:

| Purpose | Path |
|---|---|
| **BEST model confusion matrix** | `Asif's/M3/29 aug run/confusion_matrix.png` (M3 is best on official metric) |
| BEST model loss curve | `Asif's/M3/29 aug run/loss_curve.png` |
| BEST model accuracy curve | `Asif's/M3/29 aug run/accuracy_curve.png` |
| BEST model macro-F1 curve | `Asif's/M3/29 aug run/f1_curve.png` |
| BEST model ICBHI-score curve | `Asif's/M3/29 aug run/icbhi_score_curve.png` |
| M2 equivalents | `Asif's/M2/m2_v4/*.png` |
| Hyperparameter sweep | `Asif's/M2/m2_v4/hp_sweep_comparison.png` |
| SpecAugment preview | `Asif's/M22/result_M22/specaugment_preview.png` |
| Backbone comparison | `Asif's/M12/backbone_comparison.png` |
| Efficiency vs performance | `Asif's/M12/efficiency_vs_performance.png` |
| Open-set baselines | `Asif's/M29/openset_baselines.png`, `score_distributions.png` |
| AUROC forest plot | `Asif's/Statistics/auroc_forest_plot.png` |
| **Novelty N1–N8** | `Novelty Experiment/figures/N{1..8}_*.png` |

---

## 14. Tooling and reproducibility artefacts

| Tool | Path | What it does |
|---|---|---|
| Canonical score format + paired tests | `Asif's/Statistics/owmtl_scores.py` | DeLong, McNemar, bootstrap CI; 39/39 self-tests |
| Significance report | `Asif's/Statistics/SIGNIFICANCE_REPORT.md` | CIs + pairwise tests over the open-set suite |
| ICBHI metric audit | `Asif's/audit/icbhi_score_audit.py` | recomputes official score from confusion matrices |
| Official split loader/auditor | `Asif's/audit/official_split.py` | documents the 156/218 leak |
| Project audit | `Asif's/audit/audit_project.py` | 51 files, 19 CRITICAL findings |
| Device-axis feasibility (G4) | `Asif's/audit/check_device_structure.py` | 3/126 patients span devices → BLOCKING |
| Concept extractors | `Asif's/engine/concept_extractors.py` | 9 concepts, numpy-only, 43/43 synthetic tests |
| Clinician label analysis | `Asif's/engine/analyze_clinician_labels.py` | intra/inter-rater, vs ICBHI, vs extractors |
| Undefined-name checker | `Asif's/engine/check_undefined_names.py` | static NameError detection for notebooks |
| Notebook patches | `Asif's/engine/patch_*.py` | official metric, val split, report consistency, ablation guards |

**Notebook correctness markers** (all three of M2/M3/M22 carry these):
`OWMTL_DATA_SETUP_V1`, `OWMTL_OFFICIAL_METRIC_V1`, `OWMTL_REPORT_CONSISTENCY_V1`,
`OWMTL_VAL_SPLIT_V1`, `OWMTL_SPLIT_LOOKUP_V2` (+ `OWMTL_ABLATION_GUARD_V1`, `OWMTL_M3_LOOKUP_V1`
on M3/M22).

**Per-cycle score dump:** each notebook now writes `scores_<MID>.csv` in the
`owmtl_scores.py` schema, keyed `<wav_stem>#<start>-<end>` — stable **across** notebooks, so
models can be compared with paired DeLong **without retraining**. ⚠️ Not yet recovered from
Drive for M2/M3.

---

## 15. Claims that must NOT go into the paper

Each of these appears somewhere in the repo's history and is **refuted by a later, better-
controlled run**. Listing them so nobody reintroduces them.

| Claim | Where it came from | Why it's dead |
|---|---|---|
| "ICBHI score 0.7227" (M2) / 0.6984 (M3) | original runs | macro metric **and** the 11-patient split; corrected: 0.4720 / 0.5132 |
| "Interpretability costs 0.2326 acc / 0.1432 F1" | N6 committed single-seed run | **single-seed artefact**; with 5 seeds the ordering inverts, McNemar p=1.0000 |
| "M15 beats M6 by 36.9%" | M15 | ratio against a **sub-chance** baseline (M6 AUROC 0.4516) |
| "Calibration improves accuracy 0.3573 → 0.6595" | M11 | temperature scaling is **monotone** — accuracy cannot change |
| "Conformal wrapper achieves 95.45% coverage" | M14 | coverage is guaranteed by construction; circular. Empty-set rate on unknowns is 0.0 |
| "Concept bottleneck is interpretable and effective" | project premise | G2 failed **twice**; N7 shows interventions push the wrong way 5/6 times |
| "Physicians validate our crackle detector" | — | crackle κ vs ICBHI = +0.035, **CI includes zero** |
| M4 (AST) as a committed baseline | M12 reference rows | `results_M4.json` is **not in the repo**; never entered the audit |

---

## 16. Framing discipline for the write-up

From `PAPER_OUTLINE.md`, and worth preserving verbatim:

> Present the evaluation errors as **errors we made and caught**, not as accusations about other
> groups. That is both honest and far more persuasive. The implicit argument — if a team auditing
> itself this hard made all three, the 135-paper literature likely contains them — lands harder
> unstated.

**Name our concepts "physics-derived", never "label-free"** — Label-free CBM (Oikarinen, ICLR)
is LLM/CLIP-derived and the collision would be read as overclaiming.

**Position honestly:** we are not inventing concept bottlenecks or disentangled evaluation. The
contribution is the **respiratory-audio measurement**, not the method category.

**Concede the weak-detector point directly.** The ceiling argument (§8, §10) does not depend on
our extractors being good, which is exactly why it survives the concession.

---

## 17. Suggested mapping onto the required paper sections

| Paper section (per `paper_requirement.md`) | Source in this document |
|---|---|
| Abstract | §0 executive summary — write last |
| Introduction — motivation | §0, §2 (why evaluation rigor matters on a 135-paper benchmark) |
| Introduction — literature review (≥4 papers) | §10 (Tzeng 2025, Aviles-Solis 2016) + Koh 2020 CBM + Oikarinen ICLR label-free CBM. **Needs 4 members × 1 paper each** |
| Introduction — gaps/limitations | §2 (nobody characterises the reference standard), §10 |
| Proposed System — 4.1 Dataset | §1 (ICBHI + SPRSound), cite the dataset |
| Proposed System — 4.2 Preprocessing | §3.3 config table (log-mel, 128 mels, 16 kHz, 8 s, SpecAugment) |
| Proposed System — 4.3 Applied Models | §5 roster; **flowchart still to be drawn** |
| Results — all model metrics | §3 (corrected baselines), §4 (full inventory) |
| Results — params/size/time table | §3.3, §4 |
| Results — BEST model figures | §13 |
| Results — augmented | **blocked on M22 re-run** |
| Results — novelty | §6 (N1–N8) |
| Results — XAI | 🔴 **does not exist yet** — see §12 |
| Results — comparison table | §10 + §3; final row = **This work** |
| 5.1 Ablation Study | §6.6 (N6 modes), §7 (gate runs), M12 backbone, M22 SpecAugment |
| Conclusions + 2–3 future works | §18 |

---

## 18. Open items, ordered by value

### Blocking the course submission

1. 🔴 **Four transformer models.** Cheapest route: fit heads on N1's existing
   `ast_frozen.npy` / `ast_lora.npy` embeddings → 2 rows for near-zero compute; then add two
   more (Swin/ViT on spectrograms, or PANNs/BEATs).
2. 🔴 **XAI on the best model.** Grad-CAM on M2 (plain CNN, trivial to hook). Present N1's
   concept-probe result alongside as the stronger interpretability evidence.
3. 🔴 **Re-run M22** on the corrected split — the augmentation ablation. Notebook is patched and
   verified; ~40 epochs.
4. 🟡 **Recover `scores_M2.csv` / `scores_M3.csv` from Drive** — enables paired DeLong between
   models with no retraining.
5. 🟡 **Draw the system flowchart** (explicitly required, none exists).

### Strengthening the scientific claim

6. 🟡 **Second rater's ~30 clips** — inter-rater κ, pre-registered (§8.5). The one thing
   reviewers will ask for by name.
7. ⚪ Raise `--n_perm` in N5 before quoting an interval on the permutation null (currently n=5,
   mean well-pinned, CI indicative only).
8. ⚪ Real `clinician_corrections.csv` would turn N7 from SIMULATED into a real intervention.

### Explicitly NOT to be started

Per `DECISION_2026-08-16_PIVOT.md`: no new bottleneck head, no intervention API, no fourth
mechanism. *(Note the standing conflict: `RTK_requirements.md` §12 forbids the N5–N7 work that
was run anyway on faculty instruction. The faculty list supersedes the stop-list —
`NOVELTY_STATUS.md` records this so it is not re-litigated.)*

### Future work for the Conclusions section

1. **Multi-corpus validation** — SPRSound is already ingested (N8); a full cross-corpus training
   study would test whether the reliability ceiling is ICBHI-specific.
2. **A reference standard built for the task** — multi-rater, calibrated, with an explicit
   "fits no named class" option (§8.4's lesson), replacing single-annotator cycle labels.
3. **Concept-preserving fine-tuning** — N1 shows LoRA destroys concept encoding; a regulariser
   that preserves probe R² while adapting to the task is a concrete, motivated next step.

---

## 19. One-paragraph version, for the abstract draft

> Automatic respiratory-sound classification is evaluated almost universally on the ICBHI 2017
> benchmark, yet the reliability of that benchmark's own labels has not been characterised. We
> audited our own pipeline and found three evaluation errors — a non-standard scoring metric
> inflating results by 0.11 on average, a split loader silently falling back to an 11-patient
> test set, and checkpoint selection reading the test set every epoch — and report corrected
> baselines that fall from 0.72 to 0.47 and 0.51 on the official challenge metric. Against a
> pre-registered validity gate, physics-derived acoustic concepts failed twice on real audio. A
> blinded physician study then reframed that failure: the physician's crackle agreement with
> ICBHI's labels is statistically indistinguishable from chance, while replicating the 23%
> sensitivity previously reported for seven senior physicians on the same corpus. Probing a
> frozen AudioSet-pretrained transformer shows it linearly encodes 13 of 14 clinical acoustic
> concepts against a random-projection control at 0 of 14 — and that task fine-tuning degrades
> 13 of them. The model hears the sounds; the sounds do not predict the labels. We release the
> corrected protocol, the audit tooling, and a canonical score format enabling paired
> significance testing across models.

---

## 20. Provenance

Numbers in §3, §4 and §1 are read directly from the committed `results_M*.json` files at
generation time. Numbers in §6 are transcribed from `Novelty Experiment/NOVELTY_STATUS.md`,
§7 from `Asif's/M39/2nd_run_handoff/results_M39.json`, §8 from `CLINICIAN_RESULTS_v1.md`,
§10 from `Papers/HUMAN_BENCHMARKS.md`, §11 from `Asif's/audit/PROJECT_AUDIT.md`.

Regenerate the data-driven sections by re-running the generator used to build this file, or
re-derive any single number from the path given in its table.

