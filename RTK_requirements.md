# `req.md` — CSE465 course implementation requirements vs. this project

**Written:** 2026-08-20 · **Source:** Dr. Khan's "Implementation Requirements (CSE465)" list
**Read with:** `DECISION_2026-08-16_PIVOT.md` (what the paper is now), `Model_Training_Protocol.md`
(§1 hard rules, §2 preprocessing, §3 metrics, §4.1 ablation schema, §5 plots), `Model_Training_Reference.md`
(what every M-number already is).

---

## 0. How to read this document

The faculty list is a **course deliverable checklist**. `PAPER_OUTLINE.md` is the **paper**. They are
not the same thing and must not be merged:

- The course wants a broad model zoo with the standard table (metrics + size + time), augmentation,
  novelties, XAI, ablation.
- The paper's claim after the 2026-08-16 pivot is the *corrected evaluation protocol* + a
  pre-registered negative result. Adding models does not damage that claim — **as long as every new
  number is produced under the corrected split and the official metric.** A course run done on the
  wrong split becomes a liability the audit will flag.

**One rule that governs everything below:** every new run uses `Asif's/audit/official_split.py` and
reports `icbhi_score_official` with `confusion_matrix_raw` committed. No exceptions, no fallbacks.
A notebook that cannot find the split file must raise, not fall back (README, trap #2).

**Group size = 4** (Asif, Barshon, Farhana, Sami) → **X = 4 transformer models**, and ≥4 pre-trained
models, one owned per member.

---

## 1. Requirement-by-requirement status

| # | Requirement | Status today | Gap |
|---|---|---|---|
| 1 | ALL preprocessing techniques | 🟡 Partial — `Model_Training_Protocol.md` §2 fixes SR/mel/window/freq; notebooks do load → pad/crop → log-mel → normalize → class-weighted loss | No band-pass filter, no silence trim, no denoising, no per-recording normalization variant, no documented preprocessing ablation. Nothing tabulated. |
| 2 | Pre-trained DL model, one per member | 🟡 M3 MobileNetV2 (Asif) is committed; M4 AST was run but never exported a JSON. M1/M2 are **trained from scratch** — they do not count. M40–M43 are pre-trained and cover all four members once run | Run `M40_M43_transformers/`; M47 CNN gap-fills stay optional |
| 3 | ≥ X = 4 transformer models | 🟡 **6 of 8 runs done.** `M40_M43_transformers/` — M41 Swin-T **0.5304**/0.5291 and M42 DeiT-S 0.5149/0.4981 trained cleanly (both below the 0.5602 M22_v2 baseline, which is the expected and reportable trade-off). **M40 ViT-B/16 collapsed to all-Normal** in both runs. M43 AST runs on Kaggle (`M43_AST_kaggle.ipynb`) — 1212 patches need 16 GB. Prior state: M4 had no committed JSON, M23 undocumented, M33_v2 is a transformer *head*, M37_v2 applies LoRA to the **M2 CNN** not an AST | Run M43 on Kaggle; decide whether to re-run M40 with warmup or report the collapse |
| 4 | Accuracy / precision / recall / F1 for ALL models | 🟡 Schema exists (§3) and Asif's runs + M31–M37 comply; **17 of 39 results JSONs are missing required fields** | Backfill or re-export (list in §5) |
| 5 | Model size / params / training time for ALL models | 🟡 Same story — present in Asif's runs + M31–M37, absent in M11 / M14 / M15v4 / M17 / M18 / M20 / M24 / M28 / Gap7 / M7 | Backfill |
| 6 | Normalized CM + train/val loss & accuracy vs. epoch for BEST model | 🟡 Curves + CM exist per model; "BEST" is **undefined and currently ambiguous** (M22 = 0.6495 official 60/40; M35 = 0.6864 but on the easier 70/30 split — not comparable) | Fix the definition (§6), then regenerate publication-grade plots |
| 7 | Augmentation on training samples + results | ✅ Strongest item — M22 (SpecAugment, +0.036 official), M24-CB (class-balancing), M7 clean-vs-aug pairs | Just needs one consolidated table + the same treatment on the new transformers |
| 8 | Novelties + results | ✅ Plenty — M31 GradNorm, M32 demographic fusion, M33_v2 temporal transformer, M34 curriculum, M35 physics-informed loss, M37 LoRA, M13 prototypical head, M14 conformal, M15 cross-task, M29/M38 open-set, M39 concept gate | All on the 70/30 split → **not comparable to the backbone table**. Needs the honest split-annotated presentation (§8) |
| 9 | XAI for BEST model | 🔴 **Nothing exists.** Only mentioned in archived plans | Build it (§9) |
| 10 | Ablation study for BEST model | 🟡 §4.1 defines the ablation metadata schema and M12 is a clean backbone *selection*, but no component-wise ablation run exists | Build it (§10) |

---

## 2. New work, at a glance

| ID | What | Owner | Est. |
|---|---|---|---|
| **M40** | ViT-B/16 (ImageNet-21k) on log-mel | Barshon | 1 notebook + run |
| **M41** | Swin-T on log-mel | Farhana | 1 notebook + run |
| **M42** | DeiT-S **or** CvT-13 on log-mel | Sami | 1 notebook + run |
| **M43** | AST re-run, official split, JSON committed (fixes the M4 hole) | Asif | re-run |
| **M44** | XAI pack for the best model | owner of the best model | 1 notebook |
| **M45** | Ablation study for the best model | owner of the best model | 1 notebook |
| **M46** | Preprocessing ablation (6 rows) | Asif | 1 notebook |
| **M47** | Pre-trained CNN gap-fill (ResNet50 / EfficientNet-B0 / DenseNet121), only if a member wants a separate CNN entry | as needed | 1–2 runs |
| **T1** | Results-JSON backfill sweep | Barshon | script, no GPU |
| **T2** | Master table + figure pack | Sami | script |

Everything else in the repo stays where it is. **Do not start new mechanisms** — the pivot document's
stop-list still holds.

---

## 3. Requirement 1 — "ALL preprocessing techniques"

Read this as: *the preprocessing pipeline must be complete, explicit, and shown to matter.* Two
deliverables.

### 3a. One canonical preprocessing module

Write `Asif's/owmtl/preprocessing.py` (or extend what is there) so every new notebook imports the same
function instead of re-implementing the chain. Stages, in order:

1. **Load & resample** → 16 kHz mono (`librosa.load(sr=16000)`).
2. **Band-pass filter** → 4th-order Butterworth 50–2000 Hz (`scipy.signal.filtfilt`, zero-phase).
   *Currently the band limit only happens implicitly via mel `fmin`/`fmax` — make it explicit and ablatable.*
3. **Cycle segmentation** → from the ICBHI annotation files' start/end times. Cycles are the unit of
   classification; **patients** are the unit of splitting.
4. **Duration standardisation** → 8.0 s; pad by cyclic repetition (not zeros — a documented choice),
   centre-crop if longer.
5. **Amplitude normalisation** → per-cycle peak or RMS normalisation, to remove device gain differences.
6. **Denoising** → spectral gating / spectral subtraction; on by default, ablated in M46.
7. **Log-mel** → 128 mels, n_fft 1024, hop 160, win 400, 50–2000 Hz, `power_to_db`.
8. **Per-spectrogram standardisation** → zero-mean / unit-variance, plus the 3-channel replication +
   ImageNet mean/std variant that the pre-trained CNNs and ViTs need.
9. **Label encoding** → 4-class sound event (Normal / Crackle / Wheeze / Both); disease labels at patient level.
10. **Class-imbalance handling** → inverse-frequency weighted CE, held constant across backbones so the
    comparison stays fair (this is why M2/M3/M4 share it).
11. **Split** → `official_split.py`, patients 156 and 218 reassigned to train, patient-independent.

### 3b. M46 — preprocessing ablation

One table, one backbone (the current best), one variable at a time:

| Row | Config | Report |
|---|---|---|
| P0 | Full pipeline | official ICBHI, Se, Sp, macro-F1 |
| P1 | − band-pass filter | Δ vs P0 |
| P2 | − denoising | Δ |
| P3 | − amplitude normalisation | Δ |
| P4 | zero-padding instead of cyclic | Δ |
| P5 | − per-spectrogram standardisation | Δ |

That table *is* the answer to requirement 1, and it doubles as part of requirement 10.

---

## 4. Requirements 2 & 3 — pre-trained models and transformers

### Assignment (one pre-trained model per member, ≥4 transformers total)

| Member | Pre-trained CNN (req. 2) | Transformer (req. 3) |
|---|---|---|
| Asif | M3 MobileNetV2 ✅ done | **M43** AST — re-run on the official split, commit `results_M43.json` |
| Barshon | M47a ResNet50 *(optional)* | **M40** ViT-B/16 |
| Farhana | M47b EfficientNet-B0 *(optional)* | **M41** Swin-T (supersedes / absorbs M23) |
| Sami | M47c DenseNet121 *(optional)* | **M42** DeiT-S or CvT-13 |

Notes:

- The four transformers are themselves pre-trained models, so they satisfy requirement 2 as well.
  M47 is a gap-fill *only if* a member wants a separate CNN entry to defend in the viva.
  **Prefer the minimum path** — four well-run transformers beat eight rushed runs.
- `Barshon's/RESNET & EFFICIENTNET(Not Related to Project)/` is excluded by its own label today. If it
  is re-used, it must be re-run under the protocol and re-labelled, not imported as-is.
- **Expect the transformers to lose.** M4 already showed AST at ~24× the size and ~30× the latency of
  M2 for a worse score — 920 recordings is not enough for a ViT to shine. That is a legitimate,
  reportable finding (the accuracy/compute trade-off), not a failure. Do not tune until it wins.

### Fixed settings for all four transformer runs (so they stay comparable)

- Input: 128×801 log-mel → resize/pad to the model's expected resolution, replicate to 3 channels,
  ImageNet normalisation.
- Same class-weighted CE as M2/M3/M4. Seed 42. Official 60/40 split.
- AdamW, lr 1e-4 (backbone) / 1e-3 (head), cosine schedule, ≤40 epochs, early stop on val macro-F1.
- **Clean run first** (`is_augmented: false`), **then** the SpecAugment run — that pair is exactly what
  requirement 7 needs.
- Log per-epoch and total training time from the loop; params / size via the §3 snippet in the protocol.

---

## 5. Requirements 4 & 5 — metrics for ALL models

The schema already exists; the problem is compliance. Audit of the 39 committed results JSONs:

- **Missing efficiency (params / size / time):** `Gap7`, `M11`, `M14 v1`, `M14 v2`, `M15 v4`, `M16`
  (params), `M17`, `M18`, `M19` (size + time), `M20`, `M21`, `M24`, `M28`, `M7`, `M7_aug`.
- **Missing `confusion_matrix_raw`:** `M29`, `M39` (both runs), Barshon's `M30`, plus all of the above.
- **Missing `icbhi_score_official`:** `M2/17aug_run_v2`, `M3/17aug_run_result`, `M6`, `M13`, `M15 v6`,
  `M29`, plus all of the above.

**T1 — backfill sweep (Barshon), no GPU needed:**

1. `python3 "Asif's/audit/icbhi_score_audit.py" --write` — backfills the official metric wherever a raw
   confusion matrix exists.
2. Files with no committed matrix: re-export from the checkpoint if one exists; if not, mark the row
   **"not recoverable"** in the master table rather than transcribing a number. A score without a matrix
   is precisely the failure mode the protocol was rewritten to stop.
3. Efficiency fields: params and size are recomputable offline from the checkpoint (cheap). Training
   time is **not** recoverable — mark it, do not invent it.
4. Re-run `python3 "Asif's/audit/audit_project.py"` and attach the clean output to the submission.

**T2 — master table (Sami):** extend `Barshon's/M28` to emit one CSV + LaTeX table with columns:

`model · owner · type (scratch / pretrained-CNN / transformer) · split · augmented? · accuracy ·
precision_macro · recall_macro · F1_macro · Se · Sp · icbhi_score_official · params · size_MB ·
s_per_epoch · total_train_s · GPU`

**The split must be its own column.** A 70/30 row and a 60/40 row in one table without that column is
the exact error the paper's §3 is about; the course table must not commit it.

---

## 6. Requirement 6 — the BEST model

**Definition to adopt and write down once:** *best = highest `icbhi_score_official` on the corrected
official 60/40 split, among models with a committed raw confusion matrix.*

Under that rule, **M22 (MobileNetV2 + SpecAugment, 0.6495) is the current best.** M35's 0.6864 is on the
70/30 split and cannot be ranked against it — say so explicitly rather than quietly dropping it.
Re-declare the winner after M40–M43 land.

Deliverables for whichever model wins:

1. **Normalized confusion matrix** — row-normalized 4×4, annotated, ≥150 DPI; commit the raw-count
   version beside it.
2. **Training vs validation loss** vs epoch, best epoch marked.
3. **Training vs validation accuracy** vs epoch, best epoch marked.
4. Keep the existing F1 and ICBHI-score curves — they cost nothing and strengthen the report.

`training_history` is already in the results schema, so all of these regenerate from JSON without retraining.

---

## 7. Requirement 7 — augmentation

Already the project's most defensible result. Consolidate rather than expand:

| Model | Base | Augmentation | Result |
|---|---|---|---|
| M22 | M3 MobileNetV2 | SpecAugment (time + freq masking) | 0.6495 official, **+0.036** over M3 |
| M24-CB | disease head | class balancing (noise / pitch shift / time stretch on Healthy + URTI) | `Barshon's/M24` |
| M7 | deep ensemble | clean vs aug, 5 seeds each | `Sami's/M7/M7_handoff` |
| **new** | each of M40–M43 | SpecAugment, same recipe as M22 | clean-vs-aug Δ per transformer |

Rules (protocol §7): augment **training only**; same seed as the clean run; set `is_augmented: true` and
`augmentation_method`. The deliverable is a paired clean/augmented table with a Δ column — one variable
changed, so the Δ actually means something.

---

## 8. Requirement 8 — novelties

The repo already has more novelty runs than the course asks for. The task is presentation, not production.

| Novelty | Model | Result | Verdict |
|---|---|---|---|
| Physics-informed loss | M35 | 0.6864 official (70/30) | best verified — but M35_v2 regressed (best epoch 1); report both |
| LoRA / PEFT | M37 | 0.6753 with 0.23% of params trained | strong efficiency story |
| GradNorm MTL | M31 | 0.5535 | below baseline |
| Demographic fusion | M32 | 0.4733 | below baseline |
| Temporal transformer | M33_v2 | 0.5832 (M33 collapsed, Sp = 0) | below baseline |
| Curriculum learning | M34 | 0.5754 | below baseline |
| Multistage distillation | M36 | 0.5052 (Se 0.09) | collapsed |
| Prototypical disease head | M13 | patient-F1 0.6061 | real |
| Conformal wrapper | M14 | AUROC 0.4809, 0% unknown detection | negative |
| Cross-task consistency | M15 | AUROC 0.5747 < Energy 0.6466 | negative — the headline that failed |
| Open-set suite | M29 / M38 | Energy AUROC 0.6466 | the bar; CI spans chance at n = 19 |
| Concept-extraction gate | M39 | AUROC 0.55 / 0.53, gate FAIL ×2 | the paper's pre-registered negative result |

Two things to add, both cheap:

- The **split column** on every row (M31–M37 are 70/30; the backbones are 60/40).
- Honest phrasing on the open-set rows: at n = 19 unknown patients every CI crosses 0.5, so write
  "not shown to beat chance", never "X beats Y".

**If time allows, one re-run only:** M35 (physics-informed loss) on the corrected official split. It is
the single number that would change the story if it survives the harder split, and it is one job.

---

## 9. Requirement 9 — XAI (M44), the biggest genuine gap

Nothing exists. Build one notebook, `M44_xai_best_model.ipynb`, in the best model's owner's folder.

**Method by model family:**

- CNN best model (M22 / M3 / M2 lineage): **Grad-CAM** on the last conv block → heatmap over the log-mel,
  overlaid on the spectrogram.
- Transformer best model (M40–M43): **attention rollout** + Grad-CAM on the final block — raw attention
  alone is a weak explanation and reviewers know it.
- Cheap second method for cross-checking either family: **occlusion sensitivity** (slide a time–frequency
  mask, record the score drop). Model-agnostic, ~20 lines.

**What to show — a 4-panel figure per class:** one correctly-classified example each of Normal, Crackle,
Wheeze, Both, showing original spectrogram · Grad-CAM overlay · occlusion map · predicted vs true. Plus at
least one **misclassified** example — graders and reviewers both look for it.

**The one analysis that lifts this above decoration:** the ICBHI annotations give crackle/wheeze event
*times*. Measure whether the Grad-CAM mass lands inside the annotated event window — a "pointing game" hit
rate. That turns a pretty picture into a measured claim, and it connects directly to the paper's
label-reliability argument. Report the hit rate as a number.

**Do not** claim clinical interpretability from it. Given the project's own finding that concept-level
detection against these labels is bounded by label reliability, phrase XAI as *evidence about where the
model looks*, not as clinical validation.

---

## 10. Requirement 10 — ablation study (M45)

Component-wise, one variable at a time, on the best model, on the official split. The metadata block in
`Model_Training_Protocol.md` §4.1 (`ablation_group`, `component_flags`, `loss_weights`) already defines how
to record each row — fill it, do not reinvent it.

Rows, assuming the best model is the M22-style CNN + SpecAugment:

| Row | Change | Isolates |
|---|---|---|
| A0 | Full model | reference |
| A1 | − SpecAugment | augmentation contribution (already have it: M3 vs M22) |
| A2 | − ImageNet pre-training (random init) | value of transfer learning |
| A3 | − class-weighted loss (plain CE) | imbalance handling |
| A4 | frozen backbone, head-only training | how much fine-tuning buys |
| A5 | 64 mels instead of 128 | input-resolution sensitivity |
| A6 | 4 s instead of 8 s cycles | temporal-context sensitivity |

Report every row with `icbhi_score_official`, Se, Sp, macro-F1, params and s/epoch — so the ablation table
also feeds requirement 5. Merge M45 and M46 into **one** ablation section in the report; they are the same
kind of evidence.

**Sanity rule:** if a removal *improves* the score, do not hide it. M35_v2 and M33 are already in the record
as regressions; that consistency is the project's whole credibility.

---

## 11. Suggested order

1. **T1 backfill + audit run** — makes the 30+ existing models countable for requirements 4 and 5.
   Cheapest win in this document, no GPU.
2. **M43 (AST re-run)** — closes the missing-JSON hole and gives transformer #1 legitimately.
3. **M40 / M41 / M42 in parallel** — one member each, same template; clean run, then augmented run.
4. **Re-declare the best model** using §6's rule.
5. **M44 (XAI) + M45 (ablation)** — both depend on step 4.
6. **M46 (preprocessing ablation)** — can run in parallel with step 5.
7. **T2 master table + figure pack** — last, once everything is audit-clean.

---

## 12. What this document does *not* authorise

- No new mechanism. `DECISION_2026-08-16_PIVOT.md`'s stop-list stands: no bottleneck head, no intervention
  API, no concept-leakage work, no fourth idea.
- No third G2 extractor revision. The test split has been read twice; that budget is spent.
- No tuning a transformer until it beats M2. Report the loss honestly instead.
- No number in the course report that is absent from a committed results JSON with a raw confusion matrix.
  The whole point of the corrected protocol is that we do not do that any more.
