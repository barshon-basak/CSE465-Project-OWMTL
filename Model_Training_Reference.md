# Model Training Reference — OWMTL Experimental Record (Logbook)

> **Purpose changed (2026-08-14).** This file is no longer a "what to run next" execution plan — that role now lives in `OWMTL_Merged_Decision_Roadmap.md` (moved -> `OWMTL_Decision_Roadmap (v3).md`) and `OWMTL_Build_Sheet.md` (moved -> `Archive_Files (v3)/OWMTL_Build_Plan (v3).md` (moved -> `Archive_Files (v3)/OWMTL_Build_Plan (v3).md`)). This file is now the **comprehensive experimental record**: a per-model / per-notebook logbook of everything the team has run, why, what happened, and what was learned — so a reviewer can understand the whole development history without opening every notebook.
>
> **How each entry is organized (consistent template):** *What it does · Why it was created · What was done · Expected outcome · Actual outcome · Key findings · Limitations/issues · Relationship to previous versions · Reproduction notes.*
>
> **Sources of truth this record is built from** (not from re-reading raw code): `Research_Progress_Report.md` (moved -> `Archive_Files (v2)/Research_Progress_Report till v2.md`) (per-model detail), `Asif's/audit/ICBHI_SCORE_AUDIT.md` (recomputed official metrics), `Asif's/audit/PROJECT_AUDIT.md` (real-vs-synthetic + failure findings, 2026-08-13), `Asif's/Statistics/SIGNIFICANCE_REPORT.md` (confidence intervals), and the notebook inventory on disk. Where a source did not contain something, it is marked **⚠️ not established in the reviewed sources** rather than guessed.
>
> **Contributors seen in the repo:** Barshon, Asif, Sami, and **Farhana** (M23). Ownership is historical (whose folder the notebook lives in), not a role assignment.

---

## ⚠️ Two caveats that apply to every number below

1. **Two metrics exist.** `icbhi_score` (macro, project-internal) runs **~0.11 higher** than the official ICBHI `(Se+Sp)/2`. Only the **official** number is comparable to published work. Where both are known, the official is shown and the macro is struck through.
2. **Two splits exist and are not comparable.** M1–M4, M12, M22 use the **official 60/40** split; M30–M37 use an easier **70/30** split. A 70/30 number cannot be ranked against a 60/40 number.
3. **Open-set AUROCs are statistically tied to chance.** At n=19 unknown patients, every open-set detector (M6/M14/M15/M29) has a 95% CI spanning ~0.3 AUROC points and crossing 0.5. The honest phrasing is "not shown to beat a coin flip yet," **not** "mechanism X beats/loses to Y."

---

## Master index

| ID | Name | Owner | Status (audit 2026-08-13) | Headline result |
|---|---|---|---|---|
| M1 | Provisional CNN backbone | Barshon | ✅ real | Official ICBHI 0.6143 (throwaway ref) |
| M2 | Tuned CNN backbone (**selected**) | Asif | ✅ real, clean | **Official ICBHI 0.6138** — the trustworthy anchor |
| M3 | MobileNetV2 lightweight backbone | Asif | ✅ real | Official 0.5895 |
| M4 | AST transformer backbone | Asif/Barshon | ✅ real | Macro 0.6359; ⚠️ no committed JSON |
| M12 | Backbone selection audit | Asif | ✅ real, clean | Selected M2 |
| M22 | M3 + SpecAugment | Asif | ✅ real, clean | **Official 0.6495 — best on the official split** |
| M23 | AST + SpecAugment | Farhana | ⚠️ not established | — |
| M6 | OpenMax + Weibull open-set baseline | Barshon | ✅ real (negative) | AUROC 0.4516 (below chance) |
| M29 | Post-hoc OOD suite (MSP/Energy/Maha) | Asif | ✅ real, clean | Energy AUROC 0.6466 — the bar |
| M38 | Large-N open-set | Asif | ⚠️ not established (new) | — |
| M13 | Prototypical disease head | Barshon | ✅ real (v4); earlier broken | Patient-F1 0.6061 |
| M15 | Cross-task consistency scorer | Barshon | 🔴 core mechanism failed | AUROC 0.5747 < Energy 0.6466 |
| M17 | OWL Stage-2 forgetting curve | Barshon | ✅ real (v2) | Forgetting −15.34% |
| M7 | Deep ensemble | Sami | 🟡 real, schema incomplete | 10 checkpoints |
| M11 | Post-hoc calibrators | Barshon | ✅ real, schema-light | temp/vector/focal |
| M14 | Conformal wrapper | Barshon | 🔴 detection collapse | 95.45% coverage, 0% detection |
| M20 | Temperature scaling / ECE | Barshon | 🔴 schema non-compliant | T*=1.4875 |
| M16 | Teacher-student distillation | Barshon | 🟡 needs run | 8.85× compression |
| M18 | Pruning + quantization sweep | Barshon | 🔴 schema/synthetic history | INT8 + L1 prune |
| M19 | Cross-dataset OOD eval | Barshon | 🔴 AUROC below chance | see entry |
| Gap7 | OOD generalization (MMD) | Barshon | ✅ real analysis | pediatric-physics finding |
| M30 | M2+M3 gated fusion | Barshon | 🔴 withdrawn (unverifiable) | 0.8213 macro, no matrix |
| M30_v2 | Gated fusion, re-export | Asif | 🟡 built, not run | official split, admission test |
| M31 | GradNorm MTL | Barshon | ✅ real (below baseline) | Official 0.5535 |
| M32 | Demographic fusion | Barshon | ✅ real (below baseline) | Official 0.4733 |
| M33 | Temporal transformer | Barshon | 🔴 collapsed (Sp=0) | Official 0.3330 |
| M33_v2 | Temporal transformer (fixed) | Barshon | ✅ real (below baseline) | Official 0.5832 |
| M34 | Curriculum learning | Barshon | ✅ real (below baseline) | Official 0.5754 |
| M35 | Physics-informed loss | Barshon | ✅ real | **Official 0.6864 — best verified** |
| M35_v2 | Physics loss (re-run) | Barshon | 🔴 broken (best epoch 1) | Official 0.6719, regressed |
| M36 | Multistage distillation | Barshon | 🔴 collapsed (Se=0.09) | Official 0.5052 |
| M37 | Audio LoRA (PEFT on CNN) | Barshon | ✅ real | Official 0.6753, 0.23% params |
| M37_v2 | LoRA on AST | Barshon | ⚠️ not established | — |
| M24-CB | Class-balancing augmentation | Barshon | ✅ real | SpecAug on Healthy/URTI |
| M21 | Curriculum (SNR pacing) | Barshon | 🔴 implausible (metrics=1.0) | train/test leak suspected |
| M28 | Master experiment merge | Barshon | 🟡 tooling | merges audit-clean JSONs |

*Not part of the project:* `Barshon's/RESNET & EFFICIENTNET(Not Related to Project)/` and `test_nbformat.ipynb` — excluded by their own labels.

---

# A. Sound-event backbones (the encoder race)

### M1 — Provisional CNN backbone  ·  ✅ real
- **What it does:** A simple 4-block 2D CNN over log-mel spectrograms, 4-class sound-event classification (Normal/Crackle/Wheeze/Both).
- **Why it was created:** A fast "good enough" checkpoint to build and debug the downstream disease/OWL pipeline against *before* the real architecture search finished — explicitly a throwaway reference.
- **What was done:** Log-mel (128-mel, 8 s @ 16 kHz), inverse-frequency class-weighted cross-entropy, no augmentation, standard split. ~421K params.
- **Expected outcome:** A working baseline, not a competitive one.
- **Actual outcome:** Official ICBHI **0.6143** (Se 0.3502, Sp 0.8784), macro ~~0.7181~~, on the 60/40 split.
- **Key findings:** Confirms a plain CNN on log-mel is viable; the high Sp / low Se shows it leans toward predicting Normal.
- **Limitations:** Not tuned; superseded by M2. Two notebooks exist (`M1-provisional-cnn-backbone.ipynb` + a `_backup_pre_fix` copy).
- **Relationship:** Parent of M2 (M2 is M1 properly tuned).
- **Reproduction:** `Barshon's/M1/`.

### M2 — Tuned CNN backbone (**selected backbone**)  ·  ✅ real, audit-clean
- **What it does:** The properly-tuned 5-block CNN (width 48, dropout 0.4, ~3.6M params) — the project's shared encoder.
- **Why it was created:** To produce the architecture-ablation number and become the frozen backbone everything else builds on.
- **What was done:** Real HP sweep (6 configs × 3-fold GroupKFold = 18 runs), cosine LR from 5e-4, batch 32, class-weighted CE held constant across M2/M3/M4 to keep the M12 comparison fair. Best checkpoint at epoch 19.
- **Expected outcome:** The strongest real backbone at a deployable size.
- **Actual outcome:** Official ICBHI **0.6138** (Se 0.6118, Sp 0.6157), macro-F1 0.5238, 2.94 ms/sample on T4. Beats M3/M4 on both metrics. Per-class: Normal F1 0.69, Crackle 0.66, Wheeze 0.38, Both 0.36 (minority classes weak).
- **Key findings:** The project's single most trustworthy number — right metric, right split, checkpoint verified to reproduce exactly. Sits at the published ICBHI level (~0.60–0.65), not above it.
- **Limitations:** Minority-class (Wheeze/Both) F1 is low, as expected from the class imbalance.
- **Relationship:** Tuned M1; selected over M3/M4 by M12; feeds M13/M15/M29/M30/M31–M37 as encoder.
- **Reproduction:** `Asif's/M2/M2_cnn_baseline_tuned.ipynb`; audit-clean (no findings).

### M3 — MobileNetV2 lightweight backbone  ·  ✅ real
- **What it does:** ImageNet-pretrained MobileNetV2 adapted to single-channel mel spectrograms (~2.2M params).
- **Why it was created:** The efficiency comparison point in the backbone race and a complementary feature extractor for the M30 fusion.
- **What was done:** Full fine-tune, Adam lr 1e-3, 2-fold GroupKFold; same loss as M2/M4.
- **Expected outcome:** Competitive-but-lighter alternative to M2.
- **Actual outcome:** Official ICBHI **0.5895** (Se 0.5359, Sp 0.6431), macro ~~0.6984~~ — second-best real backbone.
- **Key findings:** Lighter but ~2.4 pts weaker than M2; gap exceeded CV tolerance, so it lost the M12 selection.
- **Limitations:** ⚠️ Audit warning — best epoch 4/40 (suspiciously early; possible overfitting to the validation fold). Its SpecAugment variant is M22.
- **Relationship:** Backbone candidate; parent of M22; one of two inputs to M30.
- **Reproduction:** `Asif's/M3/M3_lightweight_backbone.ipynb`.

### M4 — AST (Audio Spectrogram Transformer) backbone  ·  ✅ real
- **What it does:** ImageNet+AudioSet-pretrained AST fine-tuned on ICBHI (~86M params).
- **Why it was created:** To test whether a transformer backbone beats a CNN on this small dataset.
- **What was done:** Standard AST preprocessing with two documented deviations to preserve pretraining (full-band 20–8000 Hz mel; n_fft=512). Same loss family.
- **Expected outcome:** Possible SOTA if the transformer's capacity helps.
- **Actual outcome:** Macro ICBHI 0.6359 — **last** of the three real backbones, and 24× larger / 30× slower than M2.
- **Key findings:** Confirms the AudioSet→ICBHI distribution mismatch: without domain-specific pretraining, AST's attention gives no advantage on 920 recordings.
- **Limitations:** ⚠️ **No committed `results_M4.json`** — its numbers are transcribed into M12's comparison table, so the official metric cannot be recomputed. Must commit the JSON before M4 appears in the paper.
- **Relationship:** Backbone candidate; lost M12; its LoRA/SpecAugment descendants are M37_v2 and M23. Two notebooks exist (`Asif's/M4-AST-soundevent.ipynb`, `Barshon's/M4/M4_ast_backbone.ipynb`).
- **Reproduction:** see both notebooks above.

### M12 — Final backbone selection  ·  ✅ real, audit-clean
- **What it does:** A documented multi-criteria decision that picks the winning backbone from M2/M3/M4's full tables.
- **Why it was created:** To freeze the shared encoder on evidence (accuracy/F1 vs params/latency), not on "highest accuracy wins."
- **What was done:** Compared the three real candidates; checked robustness (what if M1 were eligible); verified the frozen checkpoint reproduces reported metrics.
- **Expected/Actual outcome:** **M2 selected** (M3 gap 0.0243 > CV std 0.0129; M4 too large/slow). Checkpoint reproduction confirmed exact.
- **Key findings:** A clean, auditable selection — one of the project's methodologically strongest artifacts.
- **Limitations:** none flagged.
- **Relationship:** Consumes M2/M3/M4; outputs the frozen encoder used everywhere downstream.
- **Reproduction:** `Asif's/M12/`.

### M22 — M3 + SpecAugment  ·  ✅ real, audit-clean
- **What it does:** M3 (MobileNetV2) trained with SpecAugment (time+frequency masking) on training data only.
- **Why it was created:** To quantify the augmentation effect on the official split.
- **What was done:** Identical to M3 with SpecAugment added; official 60/40 split.
- **Expected outcome:** A measurable augmentation gain.
- **Actual outcome:** Official ICBHI **0.6495** (Se 0.5696, Sp 0.7294) — **the best score in the project on the official split** (+0.036 over M3).
- **Key findings:** SpecAugment gives a real, honest gain on the hard split — the single most defensible augmentation result the team has.
- **Limitations:** Doesn't change the backbone decision (M12 already settled).
- **Relationship:** Augmented child of M3; the augmentation counterpart to M2/M4's clean runs.
- **Reproduction:** `Asif's/M22/M22_mobilenet_specaugment.ipynb`; audit-clean.

### M23 — AST + SpecAugment  ·  ⚠️ status not established
- **What it does / Why:** The SpecAugment counterpart to M4 (AST). Owned by **Farhana**.
- **What was done / Actual outcome / Findings:** **⚠️ Not established in the reviewed sources** — the notebook exists (`Farhana's/M23/M23_AST_SpecAugment.ipynb`) but no committed results, audit entry, or score was found. **Needs a notebook review to document.**
- **Relationship:** Augmented child of M4.

---

# B. Disease head + core cross-task mechanism (Chunk D — the headline that failed)

### M13 — Prototypical disease head  ·  ✅ real (v4); earlier versions broken
- **What it does:** A prototypical-network head on the frozen M2 encoder: L2-normalized 256-d prototypes per disease class, cosine-distance classification (temp 0.1), episodic training (200 episodes, 5-shot/10-query). Patient-level COPD/Healthy/URTI.
- **Why it was created:** Standard softmax collapses on URTI (n=14); a metric/prototype head handles low-resource classes and gives a natural open-set distance score. Directly answers reviewer "Attack 5" (small-N).
- **What was done:** Frozen M2 features → prototypes; patient-level aggregation over a patient's cycles; 60/40 patient-independent split.
- **Expected outcome:** Usable patient-level disease F1 despite imbalance.
- **Actual outcome:** Patient-F1 **0.6061**, accuracy 0.7209. Per-class: COPD F1 0.8727, URTI 0.5455, **Healthy 0.40 (weak)**.
- **Key findings:** Handles the dominant class well; struggles on Healthy (confused with URTI — both subtle/high-variance). Prototype distance is reusable as an open-set score.
- **Limitations:** Earlier versions were broken — "OLD" showed single-class collapse (frozen val curve, best epoch 1); "UPDATED" fixed collapse but still best epoch 1/50 (never trained past init); v1–v3 ran on **synthetic** data. Only **v4** is real and audit-clean.
- **Relationship:** Built on M12/M2; feeds M15 (its prototypes are the disease-side signal) and M17 (OWL uses the prototypical head). The chosen novelty item "meta-learning disease head" *is* this rebuild.
- **Reproduction:** `Barshon's/M13/M13_prototypical_disease_head.ipynb` (v4).

### M15 — Cross-task consistency scorer  ·  🔴 core mechanism failed
- **What it does:** The project's original headline mechanism — scores disagreement between the sound-event head's implied diagnosis and the disease head's prediction, thresholded for unknown-disease detection. 8 score formulations tested (multiplicative dist×ent selected).
- **Why it was created:** The thesis was that cross-task disagreement is a principled unseen-disease signal no prior ICBHI paper used.
- **What was done (by version):**
  - **v4:** sequential setup on frozen M2 + M13; AUROC **0.5782** (⚠️ its JSON says *22* unknown patients — conflicts with the 19 used everywhere else; flagged for Barshon as a possible split/copy-paste error).
  - **v5:** on the M31 **joint-MTL** backbone; AUROC **0.4073** — collapses.
  - **v6:** on frozen M2; AUROC **0.5747**, and recomputes Energy on the same backbone at **0.5948**; reports `icbhi_score 0.0000` with **no committed confusion matrix**.
- **Expected outcome:** Beat the trivial baselines (M6 0.4516; M29 Energy 0.6466).
- **Actual outcome:** **Fails** — v6 (0.5747) is below M29 Energy (0.6466), and joint-MTL v5 collapses to 0.4073.
- **Key findings — the pivotal negative result:** (1) Joint MTL forces the two heads to *agree* on unknowns (signal destroyed → 0.4073); (2) sequential MTL makes the disease head a linear echo of the sound head (no divergent representation). The mechanism has no headroom on this data. **Statistically, v6 vs Energy is not significant** (SIGNIFICANCE_REPORT: diff +0.072, p=0.525) — the honest claim is "tied at chance," not "loses." The mechanism is also a published method (Zamir et al. 2020 "Consistency Energy").
- **Limitations:** Earliest versions synthetic (`torch.randn`); v6 has no committed matrix; the 22-vs-19 patient discrepancy in v4; n=19 makes all comparisons underpowered (CI ≈ ±0.12).
- **Relationship:** Built on M13/M2; compared against M6/M29; conformally wrapped by M14; **retired as the thesis** in the 2026-08 pivot — now demoted to "one scored baseline detector."
- **Reproduction:** `Barshon's/M15/v4|v5|v6/`.

### M17 — OWL Stage-2 forgetting curve  ·  ✅ real (v2)
- **What it does:** The staged open-world-learning demo — adds a new disease class (Pneumonia, 6 patients) to the prototypical head with a 50/50 replay buffer, measures catastrophic forgetting of Stage-0.
- **Why it was created:** To show the OWL protocol can absorb a newly-characterized disease without collapsing known-class performance.
- **What was done:** Stage-0 (COPD/Healthy/URTI) → Stage-2 (+Pneumonia), 20 epochs, replay ratio 0.5, proto dim 256.
- **Expected outcome:** Bounded forgetting, decent plasticity on the new class.
- **Actual outcome:** Known-class retention **93.76%**, forgetting **−15.34%**, plasticity **85.61%** on Pneumonia (6 patients).
- **Key findings:** Replay-based OWL keeps forgetting within acceptable clinical limits (vs 40–90% for naive fine-tuning); strong plasticity even at n=6.
- **Limitations:** −15.34% is not negligible; depends on M15 being meaningful, which it isn't yet. v1 was synthetic; **v2** is real.
- **Relationship:** Built on M13/M15; feeds M16/M18/M19/M20 (teacher/compression/OOD/calibration). `Novelty Search.md` §4.5 suggests multistage distillation as the actual anti-forgetting mechanism instead of asserting CQKD.
- **Reproduction:** `Barshon's/M17/m17-v2-owl-stage-2-incremental-class-incorporat.ipynb`.

---

# C. Open-set / OOD baselines and evaluation

### M6 — OpenMax + Weibull baseline  ·  ✅ real (negative result)
- **What it does:** The canonical deep open-set baseline — Mean Activation Vectors per known class, Weibull tail fit, OpenMax-recalibrated softmax to flag unknowns.
- **Why it was created:** The same-lab prior method (Karim & Khan 2024 lineage) the novelty must beat.
- **What was done:** On a frozen backbone; 72 train / 32 known-test / 19 unknown-test patients; unknown group never fitted; cycle-level AUROC.
- **Expected outcome:** A respectable open-set bar.
- **Actual outcome:** AUROC **0.4516**, unknown-recall 0.0255 — **below chance**.
- **Key findings:** A genuine, citable negative result: naive OpenMax borrowed from vision fails on multi-class respiratory disease (the class boundary structure doesn't support Weibull tail fitting). Notably its CI is the only one that *excludes* chance — in the **wrong** direction (significantly worse).
- **Limitations:** Evaluated at cycle level (n≈2180), unlike the patient-level detectors — not strictly comparable. Accuracy 0.7689 is below the 0.8994 majority prior.
- **Relationship:** Baseline for M15; superseded as "the bar" by M29.
- **Reproduction:** `Barshon's/M6/`.

### M29 — Post-hoc OOD baseline suite  ·  ✅ real, audit-clean
- **What it does:** MSP, Entropy, Energy, and Mahalanobis (class + patient) computed on frozen M12 embeddings — zero training.
- **Why it was created:** The trivial-baseline ablation Reviewer #2 demands (Attack 1); sets the real bar the novelty must clear.
- **What was done:** Exact real 104-known / 19-unknown patient split, patient-level, unknown never fitted.
- **Expected/Actual outcome:** **Energy AUROC 0.6466** (best), MSP 0.5025, Entropy 0.5789, Mahalanobis 0.5376/0.5689. Energy is the bar M15 must beat.
- **Key findings:** A two-line post-hoc score (Energy) beats the trained mechanism — the finding that ultimately sank the cross-task thesis. **But CI [0.49, 0.80] crosses 0.5**: even Energy is not distinguishable from chance at n=19.
- **Limitations:** n=19 underpowered; no raw per-patient scores saved (blocks a proper paired DeLong test).
- **Relationship:** The bar for M6/M15/M14; feeds the significance analysis.
- **Reproduction:** `Asif's/M29/M29_openset_baselines.ipynb`; audit-clean.

### M38 — Large-N open-set  ·  ⚠️ status not established (new)
- **What it does / Why:** Notebook name implies a larger-sample open-set evaluation (`Asif's/M38/M38_largeN_openset.ipynb`) — plausibly a response to the n=19 statistical-power problem.
- **Everything else:** **⚠️ Not established in the reviewed sources** — not in the Progress Report, ICBHI audit, or significance report reviewed here. **Needs a notebook/README review to document purpose, method, and results.**

---

# D. Trust: calibration, uncertainty, conformal

### M7 — Deep ensemble  ·  🟡 real, schema incomplete
- **What it does:** 5 independent M2-variant models (seeds 42–46) × clean/augmented = 10 checkpoints; ensemble disagreement as an uncertainty/unknown signal.
- **Why it was created:** The gold-standard epistemic-uncertainty baseline and a strong OOD comparison for M15.
- **What was done:** Trained the ensemble; disagreement histograms produced.
- **Actual outcome:** 10 verified checkpoints exist; **⚠️ not yet evaluated as an OOD detector on the 104/19 split.**
- **Limitations:** Audit — `results_M7.json` is missing the efficiency/best_metrics/training_history blocks and the §3 metrics; can't be merged as-is.
- **Relationship:** Owned by Sami; overlaps the ensemble-fusion novelty item (§4.4); a cheap high-value OOD baseline once evaluated.
- **Reproduction:** `Sami's/M7/M7_Deep_Ensemble_3_4 (with output).ipynb`.

### M11 — Post-hoc calibrators  ·  ✅ real, schema-light
- **What it does:** Temperature, vector, and focal-loss calibration on real ICBHI log-mel features.
- **Why it was created:** Calibration layer for trustworthy probabilities.
- **What was done:** Re-run on real audio (earlier version was synthetic; now resolved).
- **Actual outcome:** Calibrators evaluated; data-integrity flag cleared.
- **Limitations:** Audit — missing efficiency/best_epoch/ablation/training_history blocks and §3 metrics; needs schema fill before merge.
- **Relationship:** Consumes M13; pairs with M20.
- **Reproduction:** `Barshon's/M11/m11-post-hoc-calibrators-temperature-vector.ipynb`.

### M14 — Conformal calibration wrapper  ·  🔴 detection collapse (a *finding*)
- **What it does:** A split-conformal wrapper on the M15 disagreement scores giving a distribution-free coverage guarantee on the reject threshold (Selected Novelty Item #2; answers Attack 6). Evaluates 8 score formulations, auto-selects multiplicative dist×ent.
- **Why it was created:** To convert a hand-tuned threshold into a formal guarantee.
- **What was done (by version):** v1 (AUROC 0.4522, below chance) → **v2** (cal AUROC 0.5842; test_auroc 0.4809; 95.45% empirical coverage at 95% nominal).
- **Expected outcome:** A meaningful, guaranteed operating point.
- **Actual outcome — the paradox:** 95.45% coverage but **0% unknown-detection** at the formal 95% point. At 90/80/50% nominal, detection rises to 21/42/58% as coverage drops.
- **Key findings:** Conformal-wrapping a near-random score yields a mathematically-correct but clinically-useless operating point — a *publishable cautionary result*, not a positive one. The framework isn't the problem; the weak underlying score is.
- **Limitations:** Both versions' test AUROC are below chance; schema missing efficiency/best_epoch/training_history.
- **Relationship:** Wraps M15; in the new plan this becomes the "honest operating point" section (I6), reframed as the paradox rather than a guarantee.
- **Reproduction:** `Barshon's/M14/v2/`.

### M20 — Temperature scaling / ECE  ·  🔴 schema non-compliant
- **What it does:** Temperature-scaling calibration reporting Expected Calibration Error, optimal temperature T*=1.4875.
- **Why it was created:** Calibration quality across OWL stages (Attack 6 support).
- **Actual outcome:** T* = 1.4875 recorded; ECE improvement reported.
- **Limitations:** Audit — `M20_metrics.json` holds 6 loose fields, not the §4 schema; can't be merged. Its OWL-stage extension (§2.19 in the old file) was synthetic.
- **Relationship:** Pairs with M11/M14; feeds the operating-point section.
- **Reproduction:** `Barshon's/M20/m20-new.ipynb`.

---

# E. Compression & generalization (Chunk F — optional/stretch)

### M16 — Teacher-student knowledge distillation  ·  🟡 built, needs run
- **What it does:** Distills the M17 teacher into a lightweight CNN student (T=3.0, α=0.7).
- **Why it was created:** Edge-deployment story for a mobile stethoscope.
- **Actual outcome:** ~**8.85–9.0× parameter reduction**, student accuracy ~93.18% (relative). Notebook generated; needs a Kaggle/Colab run to finalize.
- **Limitations:** `M16_metrics.json` is 3 loose fields, not §4 schema.
- **Relationship:** Consumes M17; parallels M18/M36 compression.
- **Reproduction:** `Barshon's/M16/`.

### M18 — Structured pruning + quantization sweep  ·  🔴 schema / synthetic history
- **What it does:** L1 structured pruning (0/20/40/60%) + dynamic INT8 quantization vs accuracy/size/latency.
- **Why it was created:** Compression analysis for deployment.
- **Actual outcome:** Rebuilt on real audio; earlier version showed **constant accuracy across a 62× sweep** (a red flag). `M18_metrics.json` is 3 loose fields, not §4 schema; needs a clean run + export.
- **Limitations:** Audit-critical (schema); prior synthetic result must not be cited.
- **Relationship:** Consumes M17; compression family with M16/M36.
- **Reproduction:** `Barshon's/M18/`.

### M19 — Cross-dataset OOD evaluation (Coswara, SPRSound)  ·  🔴 AUROC below chance
- **What it does:** Inference-only unknown-detection of the M17 model on Coswara (COVID) and SPRSound (pediatric).
- **Why it was created:** The largest-N test of whether unknown-detection transfers to genuinely unseen populations.
- **Actual outcome:** Overall AUROC **0.3287**; Coswara 0.4881 (**⚠️ num_samples=2 — meaningless**); SPRSound 0.3226 — **all below chance**. (An alternate AUPR framing 0.6265/0.6366 appears in Novelty Search, but the audit's AUROCs are the verifiable numbers.)
- **Key findings:** Unknown-detection does **not** transfer; and the confound is real — Coswara/SPRSound differ in device *and* population, so a flag can't be attributed to disease novelty. This confound became a whole rejected direction (Direction 1) and now motivates the covariate-shift section of the new plan.
- **Limitations:** Coswara n=2 is unusable; `M19_metrics.json` non-schema; num_samples records only the OOD count.
- **Relationship:** Consumes M17; its confound seeds Direction 1 and the pediatric-shift analysis (Gap7).
- **Reproduction:** `Barshon's/M19/M19_OOD_Evaluation.ipynb`.

### Gap7 — OOD generalization via MMD  ·  ✅ real analysis
- **What it does:** Computes Maximum Mean Discrepancy of acoustic features across ICBHI / Coswara / SPRSound to measure generalization (lower MMD = better).
- **Why it was created:** To answer "does it generalize?" for the sound-event models (Reviewer generalization gap).
- **Actual outcome:** M30 generalized better than M2 (Coswara MMD 0.8137 vs 0.8514; SPRSound 0.4093 vs 0.4236). **Key finding:** **M35 (physics loss) *hurt* pediatric generalization (SPRSound MMD 0.4434)** — adult-tuned acoustic priors fail on children's higher resonant frequencies.
- **Key findings:** This pediatric-physics-fragility result is a genuinely novel, mechanistic insight and is elevated to a headline finding (I5) in the new plan.
- **Limitations:** MMD is a distributional distance, not a detection metric — separate from M19's (failed) AUROCs.
- **Relationship:** Companion to M19; supports the new plan's covariate-shift/physics-fragility story.
- **Reproduction:** `Barshon's/Gap7/OOD_Generalization_Evaluation.ipynb`.

---

# F. Fusion & the M31–M37 novelty sweep (mostly below baseline — the "buzzword" lesson)

> **Context:** M31–M37 are the accumulated-technique sweep that `Novelty Search.md` §4.0 warned against. Six of nine underperform the plain M2 baseline; all are on the easier 70/30 split. They are documented here as honest negative/near-baseline results, not headline contributions.

### M30 — M2+M3 gated feature-fusion ensemble  ·  🔴 withdrawn (unverifiable)
- **What it does:** Fuses 768-d M2 (CNN) + 1280-d M3 (MobileNetV2) frozen features via a Gated Adaptive Fusion head (only the ~2.1M GAF head trained). Selected Novelty Item #3.
- **Why it was created:** The two backbones learn complementary representations; gated fusion learns which to trust per sample.
- **Actual outcome:** Reported macro **0.8213** — **withdrawn**. Invalid on three counts: legacy macro metric, 70/30 split, and **no committed `confusion_matrix_raw`** (unverifiable). The "+9.86% over M2" claim crosses both a metric and a split boundary.
- **Key findings:** The would-be headline number can't be trusted — the reason the whole reporting protocol was tightened. The idea may hold up near ~0.65 once re-run honestly.
- **Limitations:** Sound-event task only (not the OWMTL thesis); requires re-export.
- **Relationship:** Consumes M2/M3; re-run built as M30_v2. A McNemar/bootstrap significance test vs M2 exists (`Barshon's/Statistics/`, p=8.35e-17, ΔICBHI 95% CI [+0.033,+0.059]) — but that test used the withdrawn numbers, so it must be re-run after M30_v2.
- **Reproduction:** `Barshon's/M30/`.

### M30_v2 — Gated fusion, honest re-export  ·  🟡 built, not run
- **What it does:** M30 re-run on the official 60/40 split, committing `confusion_matrix_raw` + `icbhi_score_official`, with checkpoint-loading safety (`load_backbone_or_die()`), and an explicit **admission test** (must beat M2-alone and M3-alone on the same test cycles, per §4.0).
- **Why it was created:** To rescue or honestly kill M30.
- **Actual outcome:** Generated + tested (53/53 tests incl. 3 failure injections); **not yet executed on real data.** Verdict will land in `best_metrics.admission_test` — a **FAILS** verdict means drop the item (don't retry).
- **Relationship:** Corrected M30. **Reproduction:** `Asif's/M30_v2/` (needs M2+M3 checkpoints, ~20–30 min on T4).

### M31 — GradNorm multi-task learning  ·  ✅ real, below baseline
- **What it does / Why:** Shared M2 backbone + dual heads with GradNorm adaptive loss weighting, to fix task dominance (COPD/Normal dominate gradients).
- **Actual outcome:** Official **0.5535** (Se 0.5456, Sp 0.5614), 70/30 — **below M2's 0.6138** on an easier split.
- **Key findings:** Dynamic weighting did not help here; also used as the joint-MTL backbone for M15 v5 (which collapsed to 0.4073).
- **Relationship:** Novelty-sweep item; parent of M15 v5. **Reproduction:** `Barshon's/M31/`.

### M32 — Demographic fusion  ·  ✅ real, below baseline
- **What it does / Why:** Fuses age/sex/BMI/smoking metadata with M2 features via concat+MLP — clinically-motivated co-variates.
- **Actual outcome:** Official **0.4733** (Se 0.5721, Sp 0.3745), 70/30 — well below baseline; largest inflation vs macro (+0.18).
- **Key findings:** Demographic fusion hurt substantially on this setup.
- **Relationship:** Novelty-sweep item. **Reproduction:** `Barshon's/M32/m32-new (1).ipynb`.

### M33 / M33_v2 — Temporal transformer  ·  🔴 collapsed → ✅ fixed but below baseline
- **What it does / Why:** A temporal transformer aggregating a patient's respiratory-cycle sequence instead of mean-pooling — to capture inter-cycle dynamics.
- **Actual outcome:** **M33: official 0.3330 with Sp = 0.0000** — never classifies a Normal cycle correctly (total majority-class collapse; the macro metric hid it as 0.5506). **M33_v2: official 0.5832** (Se 0.6190, Sp 0.5473) — fixed the collapse but still below the M2 baseline.
- **Key findings:** The original result says nothing about temporal modelling (it's a broken run); v2 shows temporal aggregation is viable but not helpful here.
- **Limitations:** ~13M params; v1 audit-critical (`normal_detection_collapse`).
- **Relationship:** v2 is the debugged v1. **Reproduction:** `Barshon's/M33/`, `Barshon's/M33_v2/`.

### M34 — Curriculum learning  ·  ✅ real, below baseline
- **What it does / Why:** Root-pacing curriculum (acoustically easy→hard) instead of random shuffling, for better convergence.
- **Actual outcome:** Official **0.5754** (70/30) — **below** M2's 0.6138 on an easier split (earlier "approaching baseline" reading was a macro-metric artifact).
- **Key findings:** Curriculum pacing lost to the baseline. (A separate SNR-pacing curriculum is M21.)
- **Relationship:** Novelty-sweep item. **Reproduction:** `Barshon's/M34/`.

### M35 / M35_v2 — Physics-informed loss  ·  ✅ real (best verified) / 🔴 v2 broken
- **What it does / Why:** Adds a physics loss enforcing acoustic priors — Wiener spectral flatness (tonal wheezes) + Peak-to-Average Power Ratio (transient crackles) — to learn biologically-plausible features.
- **Actual outcome:** **M35: official 0.6864** (Se 0.6980, Sp 0.6747), 70/30 — **the best verified score in the project.** **M35_v2: official 0.6719 but best epoch 1/30** (audit `best_epoch_is_first` — training regressed from the first update; a broken re-run).
- **Key findings:** Acoustic priors genuinely help (M35) — but on the easier split, so not directly comparable to the 0.60–0.65 literature until re-run on 60/40. This is the asset the entire new direction is built on (physics-derived concepts). Its physics priors also *hurt* pediatric generalization (Gap7) — a double-edged, publishable finding.
- **Limitations:** 70/30 split; v2 broken.
- **Relationship:** Novelty-sweep item that became the seed of the new concept-bottleneck direction. **Reproduction:** `Barshon's/M35/`, `Barshon's/M35_v2/`.

### M36 — Multistage distillation  ·  🔴 collapsed
- **What it does / Why:** Staged Teacher→Assistant→Student KD to a tiny ~4.9K-param student for extreme edge deployment.
- **Actual outcome:** Official **0.5052 with Se = 0.0932** — detects only 9% of abnormal events (majority-class collapse; the macro 0.5137 was propped by Sp 0.9171).
- **Key findings:** A 4.9K-param student that collapsed to Normal is not a compression result. Broken, not merely compressed too far.
- **Relationship:** Compression family (M16/M18). **Reproduction:** `Barshon's/M36/`.

### M37 / M37_v2 — Audio LoRA (PEFT)  ·  ✅ real / ⚠️ v2 not established
- **What it does / Why:** Low-Rank Adaptation training only 0.23% of parameters — parameter-efficient fine-tuning to avoid overfitting on 920 recordings.
- **Actual outcome:** **M37: official 0.6753** (Se 0.7517, Sp 0.5989), 70/30 — the best efficiency-to-performance ratio in the project. **M37_v2:** applies LoRA to **AST** (per the notebook title) rather than the CNN — **⚠️ results not established in the reviewed sources; needs review.**
- **Key findings:** PEFT is highly effective for clinical audio; supports the foundation-model-adaptation arm (G5) of the new plan.
- **Limitations:** 70/30 split; v2 undocumented here.
- **Relationship:** Novelty-sweep item; the LoRA machinery is reused for FM probing in the new plan. **Reproduction:** `Barshon's/M37/`, `Barshon's/M37_v2/`.

---

# G. Augmentation, curriculum-SNR, and reporting utilities

### M24-CB — Class-balancing augmentation  ·  ✅ real
- **What it does / Why:** Targeted SpecAugment on the small known classes (Healthy n=26, URTI n=14) to correct imbalance on the disease head.
- **Actual outcome:** Re-run on real ICBHI audio (earlier synthetic version resolved); applied to Healthy/URTI.
- **Relationship:** Augmentation counterpart in Chunk D. **Reproduction:** `Barshon's/M24/M24_Class_Balancing_Augmentation.ipynb`.

### M21 — Acoustic-difficulty curriculum (SNR pacing)  ·  🔴 implausible
- **What it does / Why:** A `CurriculumSampler` ordering training by SNR (easy→hard) on M2.
- **Actual outcome:** Reports metrics of exactly **1.0** → audit `perfect_metrics_implausible` — almost certainly a train/test leak. Needs redoing if picked up.
- **Relationship:** Curriculum sibling of M34; low priority. **Reproduction:** `Barshon's/M21/`.

### M28 — Master experiment merge  ·  🟡 tooling
- **What it does / Why:** Merges every audit-clean `results_M*.json` into one consistent appendix + generates benchmark figures. **Hard rule:** only merge audit-real models (a synthetic row is worse than a missing one).
- **Actual outcome:** LaTeX table + 4 benchmark figures generated; must be re-run after schema fixes (M7/M11/M14/M16/M18/M19/M20) and the M30_v2 re-run.
- **Relationship:** Consumes all results. **Reproduction:** `Barshon's/M28/` (two notebooks: `M28_Master_Experiment_Merge.ipynb`, `m28-nb.ipynb`).

---

## Experimental progression — the story in one paragraph

Backbones settled cleanly (M1→M2/M3/M4→**M12 picks M2**, with M22 SpecAugment the best honest score at 0.6495 official). The disease head took several broken tries to become real (M13 OLD/UPDATED/v1–v3 synthetic → **v4 real**, F1 0.6061). The **core mechanism failed** on real data (M15 v6 0.5747 < M29 Energy 0.6466, and joint-MTL v5 collapsed to 0.4073) — and the significance analysis showed everything is statistically tied to chance at n=19. Trust experiments produced an honest paradox (M14: 95% coverage, 0% detection) rather than a guarantee. The M31–M37 novelty sweep mostly underperformed the M2 baseline (six of nine below it; M33/M36 collapsed outright), except **M35 physics loss (0.6864)** and **M37 LoRA (0.6753)** — the two that seeded the current direction. The audit + metric correction then withdrew the M30 headline and reframed the whole project at the published level (~0.60–0.65), which is why the direction pivoted from "novel detector" to the physics-grounded concept-bottleneck / faithfulness plan documented in `OWMTL_Merged_Decision_Roadmap.md`.

## Open documentation gaps (need a notebook review — not invented here)
- **M23** (Farhana, AST+SpecAugment) — no committed results/score found.
- **M38** (Asif, large-N open-set) — purpose/method/results not in the reviewed sources.
- **M37_v2** (LoRA on AST) — results not established.
- **M7** (Sami, deep ensemble) — checkpoints exist but not yet evaluated as an OOD detector; schema incomplete.

*Record compiled 2026-08-14 from the project's authoritative markdown + audit artifacts and the on-disk notebook inventory. Numbers trace to `ICBHI_SCORE_AUDIT.md`, `PROJECT_AUDIT.md`, `SIGNIFICANCE_REPORT.md`, and `Research_Progress_Report.md`. Items marked ⚠️ were not determinable from those sources and need a direct notebook review.*
