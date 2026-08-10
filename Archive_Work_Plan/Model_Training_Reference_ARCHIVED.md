# Model Training Reference — OWMTL Project

## Sequence
### Stage 1 (Weeks 1–2) — Nothing else can start without this

1. M1 — Provisional CNN backbone [A]
Fast, rough, just needs to work. This is the only true blocker in the whole project — everyone else is waiting on a checkpoint to build against.

### Stage 2 (Weeks 3–5) — Everyone builds in parallel off M1

Run all four of these at the same time, different members, no cross-blocking:

2. M4 — AST backbone, real training [A] (main candidate; M2/M3 optional side-comparisons if time allows)
3. M6 — OpenMax/Weibull baseline [B] (built directly on M1, doesn't need to wait for anything else)
4. M5 — Disease head, Stage 0 [B] (provisional, gets B's pipeline logic debugged — this is scaffolding, not a reported result)
5. M7 — Deep ensemble [D] (built on M1)

### Stage 3 (Week 6) — The backbone gate

6. M12 — Backbone selection [A] → requires Stage 2's M4 (+ M2/M3 if run)
7. M13 — Disease head re-fit on final backbone [B] → requires M12
8. M11 — Calibrators [D] → requires M7 + M13 (can start right after M13 lands)

Nothing past this point can start until M12 and M13 are done — this is the real chokepoint in the schedule, so protect it.

### Stage 4 (Weeks 7–9) — The core mechanism

9. M15 — Cross-task consistency scorer, Stage 0→1 [B] → requires M13
This is the paper. Everything downstream exists to test and support this result.

### Stage 5 (Weeks 10–12) — Fan-out once M15 exists 

Run in parallel:

10. M17 — OWL Stage 2 + forgetting curve [B] → requires M15
11. M18 — Compression sweep [C] → requires M17 (can start pipeline setup earlier, against M13)
12. M19 — OOD eval, Coswara/SPRSound [C] → requires M17
13. M20 — Conformal coverage across stages [D] → requires M11 + M17 + M19

### Stage 6 (Weeks 13–15) — One augmentation pass per member

Each member re-runs their final model with their assigned augmentation — not the full sweep:

14. M23 — AST + SpecAugment [A] → requires M4/M12
15. M24 — Disease head + class-balancing augmentation [B] → requires M15/M17
16. M25 or M26 — Compression or teacher + domain-robustness augmentation [C] → requires M18 or M17 (pick one, not both)
17. M27 — Calibration re-check on augmented backbone [D] → requires M20 + M23

### Stage 7 (Weeks 16–17) — Reconciliation

18. M28 — Merge everyone's tables into one Model Zoo appendix — no new training, just bringing it together.


# Significance
## Tier 1 — Must-do (the paper doesn't exist without these)

These are the critical path. Skip any one and the core novelty claim breaks.

##	Model	Why it's non-negotiable
- M1	Provisional CNN backbone	Not a reported result, but without it Member B is blocked for weeks. Cheapest model on the list — don't skip the process, just don't over-invest in it.
- M4	AST backbone	You need one strong encoder. AST is the most literature-favored choice (Tri-MTL precedent) — if time is short, this can stand in as the backbone, and M2/M3 become optional comparisons rather than required ones.
- M12	Final backbone selection	The decision itself, documented — this is Member A's whole defense answer.
- M13	Disease head on final backbone	The real (non-provisional) disease-diagnosis numbers.
- M6	OpenMax/Weibull baseline	Without this, you can't claim your mechanism beats the lab's own prior method — that comparison is what makes the novelty defensible, not just "we tried something new."
-  M15	Cross-task consistency scorer	This is the paper. Everything else supports or stress-tests this one result.
-  M17	OWL Stage 2 + forgetting curve	Without the staged protocol actually running end-to-end, "staged OWL" is just a section heading, not a result.
- M19	OOD eval (Coswara, SPRSound)	Your only large-N evidence. ICBHI's held-out classes (n=19) are too small to carry the paper alone — this is what makes the statistics defensible.

That's 8 models. If your team is genuinely squeezed, this is the minimum viable paper.

# Tier 2 — Strongly recommended (turns "acceptable" into "Q1-competitive")

Not optional in spirit — reviewers will ask for these — but not fatal if trimmed under real time pressure.

- M2, M3 — CNN + MobileNet backbones, so Member A has an actual ablation table instead of one architecture. Reviewers expect a comparison, not a single choice presented as obviously correct.
-  M18 — compression sweep (can shrink to 3 cluster-count points instead of 5, but skipping it entirely drops Member C's "deployable" claim to nothing).
- If Member D is on the team: M7 (ensemble) + M11 (calibrators) + M20 (conformal coverage) — pick these three as D's minimum viable trust pillar. This is enough to answer "can a clinician trust the flag?"
- One augmentation run per member, not the full sweep — do it once on each member's final model (M23 for A instead of M21+M22+M23; M24 for B; one of M25/M26 for C; M27 for D). This still satisfies the supervisor's mandate — augmented-vs-clean comparison exists for every workstream — at a third of the compute.

# Tier 3 — Cut first if squeezed

Real, defensible additions, but the first things to drop under deadline pressure without damaging the paper's core claim:

- M8, M9, M10 (MC-Dropout, SNGP, Evidential) — pick one alternative uncertainty signal to compare against the ensemble, not all three. Three redundant uncertainty baselines is diminishing returns.
- Full augmentation sweep across all three backbones (M21+M22 on top of M23) — one architecture's augmentation story is enough to make the point.
- M25 + M26 together (augmentation × compression combined ablation) — genuinely a stretch finding, do one or the other, not both.
-  M16 — this isn't really a separate model, it folds into M18 anyway; don't budget separate time for it.

**How to use this file:** this is the execution runbook. It lists **every model any member trains**, in the order they should be run (respecting the one real dependency — Member A's backbone handoff to Member B), with architecture, data, augmentation, and the exact outputs required by the supervisor's reporting protocol (§0.1 of the Individual Work Plan v2 / §8.5 of the Proposal v2). Models tagged with the same phase number can be trained **in parallel** by different members; a model tagged "requires M-#" cannot start until that model's winning/final checkpoint exists.

Every model entry below, without exception, must produce these four things before being marked done:
1. **Metrics** — accuracy, precision, recall, F1 (macro + per-class), confusion matrix (raw + normalized); for open-set outputs also unknown-detection precision/recall + AUROC/AUPR.
2. **Efficiency** — model size (MB), parameter count (total + trainable), training time (per-epoch + total, with GPU noted).
3. **Curves** — train/val loss vs. epoch, train/val accuracy (or macro-F1) vs. epoch.
4. **Augmented variant** — the same three items above, re-run with the assigned augmentation technique (flagged explicitly below where it's a separate run; some entries *are* the augmented run).

---

## Quick index (28 model-runs across 4 members)

| # | Owner | Phase / Week | Model | Requires |
|---|---|---|---|---|
| M1 | A | 1–2 | Provisional CNN backbone | — |
| M2 | A | 3–5 | CNN baseline (final) | — |
| M3 | A | 3–5 | MobileNet/DenseNet (lightweight) | — |
| M4 | A | 3–5 | AST (Audio Spectrogram Transformer) | — |
| M5 | B | 3–5 | Disease-diagnosis head (Stage 0, provisional backbone) | M1 |
| M6 | B | 3–5 | OpenMax + Weibull baseline | M1 |
| M7 | D | 3–5 | Deep ensemble (N≈5) | M1 |
| M8 | D | 3–5 | MC-Dropout variant | M1 |
| M9 | D | 3–5 | SNGP variant | M1 |
| M10 | D | 3–5 | Evidential deep-learning head | M1 |
| M11 | D | 3–5 | Post-hoc calibrators (temp/vector/focal) | M5, M7–M10 |
| M12 | A | 6 | **Backbone selection (final)** | M2, M3, M4 |
| M13 | B | 6 | Disease head + consistency scorer, re-fit on final backbone | M12, M5 |
| M14 | D | 6 | Conformal wrapper v1 (provisional score) | M11, M13 |
| M15 | B | 7–9 | Cross-task consistency scorer, full OWL Stage 0→1 | M13 |
| M16 | C | 7–9 | Distillation pipeline / initial student model | M13 |
| M17 | B | 10–12 | OWL Stage 2 + forgetting curve | M15 |
| M18 | C | 10–12 | Compressed student model, cluster-count sweep | M16 |
| M19 | C | 10–12 | OOD evaluation runs (Coswara, SPRSound) | M17 |
| M20 | D | 10–12 | Conformal coverage across OWL stages + calibration-under-shift | M14, M17, M19 |
| M21 | A | 13–15 | CNN baseline — **SpecAugment** | M2 |
| M22 | A | 13–15 | MobileNet/DenseNet — **SpecAugment** | M3 |
| M23 | A | 13–15 | AST — **SpecAugment** | M4 |
| M24 | B | 13–15 | Disease head — **class-balancing augmentation** | M13 |
| M25 | C | 13–15 | Compressed model — **domain-robustness augmentation** | M18 |
| M26 | C | 13–15 | Teacher (final B model) — **domain-robustness augmentation** | M17 |
| M27 | D | 13–15 | Calibration/coverage re-run on augmented backbone | M20, M21–M23 |
| M28 | A/B/C/D | 16–17 | Model Zoo & Reporting Appendix reconciliation (no new training) | all above |

---

## Phase 1 (Weeks 1–2) — Foundational, runs in parallel, no blocking

### M1 — [Member A] Provisional CNN backbone
- **Purpose:** unblock Member B immediately; this is the fast, "good enough" checkpoint the whole disease-diagnosis pipeline is built and debugged against before the real architecture search finishes.
- **Architecture:** 2D CNN over log-mel spectrograms (comparable to the MTL-MobileNet baseline family — simple conv stack, no heavy tuning yet).
- **Input representation:** log-mel spectrogram, 8 s windows @ 16 kHz (standard ICBHI preprocessing), 128-dim Fbank features, 25/10 ms window/overlap.
- **Data:** ICBHI 2017, full corpus, cycle-level, standard train/test split; sound-event labels (Normal/Crackle/Wheeze/Both).
- **Loss function:** Inverse-frequency class-weighted `CrossEntropyLoss` (to counter ICBHI class imbalance).
- **Augmentation:** none (this is the fast/dirty baseline; augmentation ablation comes later, M21).
- **Training budget:** intentionally short — get a working checkpoint in days, not weeks.
- **Required outputs:** full §0.1 metric suite, but treat this as a throwaway/reference checkpoint — the real reported numbers come from M2 (CNN baseline, tuned).
- **Hand off:** checkpoint delivered to Member B (M5, M6) and Member D (M7–M10) by end of Week 2.

*(Members C and D also do non-model infrastructure work this phase — Coswara/SPRSound acquisition and taxonomy mapping for C; calibration/UQ toolkit scaffolding and carving the held-out calibration partition for D. Neither produces a trained model yet, so nothing else is listed for Phase 1.)*

---

## Phase 2 (Weeks 3–5) — Parallel model zoo, no cross-member blocking except on M1

### M2 — [Member A] CNN baseline (tuned, final)
- **Purpose:** the properly-tuned version of M1 — this is the number that goes in the paper's architecture ablation table.
- **Architecture:** 2D CNN over log-mel spectrograms, same family as M1 but with a real hyperparameter sweep (depth, filter counts, dropout, learning-rate schedule).
- **Input representation:** log-mel (primary); MFCC and raw-waveform variants are a secondary representation sweep run on whichever architecture wins the backbone race, not required for all three architectures.
- **Data:** ICBHI 2017, full corpus, cycle-level, k-fold cross-validation (large-N task, ordinary CV is statistically fine here).
- **Loss function:** Inverse-frequency class-weighted `CrossEntropyLoss` (to counter ICBHI class imbalance and keep loss formulation constant across the backbone ablation group).
- **Augmentation:** none — clean run. (Augmented counterpart is M21.)
- **Required outputs:** full §0.1 metric suite; this is one cell of the architecture × representation ablation grid.

### M3 — [Member A] MobileNet/DenseNet (lightweight)
- **Purpose:** efficiency comparison point in the backbone race; also directly relevant to Member C's later compression work (a naturally small model is a useful reference point against C's compressed AST).
- **Architecture:** MobileNet or DenseNet variant, pretrained on ImageNet if available, fine-tuned on ICBHI log-mel spectrograms.
- **Data/representation:** same as M2.
- **Loss function:** Same inverse-frequency class-weighted `CrossEntropyLoss` as M2 and M4.
- **Augmentation:** none. (Augmented counterpart is M22.)
- **Required outputs:** full §0.1 metric suite — parameter count and model size are especially important here since this model's whole point is efficiency.

### M4 — [Member A] AST (Audio Spectrogram Transformer)
- **Purpose:** transformer-based backbone, the most likely "winner" given the Tri-MTL literature precedent; also the model most other ICBHI-AST papers use SpecAugment with, so it's the cleanest augmentation-ablation comparison point.
- **Architecture:** AST, ImageNet+AudioSet-pretrained checkpoint, fine-tuned on ICBHI. Standard AST preprocessing: 128-dim Fbank, mean/std normalization (−4.27 / 4.57, per prior AST-on-ICBHI work).
- **Data/representation:** Same 16 kHz sampling and 8.0 s duration as M2, but with two documented preprocessing deviations to preserve AudioSet pretraining and Kaldi conventions: (1) full-band mel filterbank (`20–8000 Hz`) instead of the 50–2000 Hz sub-band; (2) `n_fft = 512` instead of 1024.
- **Loss function:** Inverse-frequency class-weighted `CrossEntropyLoss` (held constant across M2, M3, and M4 to prevent experimental confounds in the M12 backbone selection).
- **Augmentation:** none. (Augmented counterpart is M23.)
- **Required outputs:** full §0.1 metric suite; training time will be notably longer than M2/M3 — record per-epoch time explicitly, it's a real deployability data point.

### M5 — [Member B] Disease-diagnosis head, Stage 0 (known-only), on provisional backbone
- **Requires:** M1.
- **Purpose:** get the disease head architecture and patient-level aggregation logic working end-to-end before the final backbone exists — this is deliberately "throwaway-adjacent" like M1, re-run properly at M13.
- **Architecture:** classification head on top of M1's frozen (or lightly fine-tuned) features, with patient-level aggregation of cycle-level features (mean/attention-pool over a patient's cycles).
- **Data:** ICBHI 2017, patient/diagnosis-level. Known classes only: COPD (64), Healthy (26), URTI (14) = 104 patients. Patient-independent split.
- **Augmentation:** none yet.
- **Required outputs:** full §0.1 metric suite on known-class classification only (no unknown-detection yet — that starts at M15).

### M6 — [Member B] OpenMax + Weibull baseline
- **Requires:** M1.
- **Purpose:** the same-lab prior-method baseline (Karim, Mahmud & Khan 2024) that the paper's actual novel mechanism (M15) must beat — build and validate this early so it's a stable, trustworthy comparison point, not a rushed afterthought.
- **Method:** compute the Mean Activation Vector (MAV) per known class on M5's features, fit a Weibull distribution to the tail distances, recalibrate softmax probabilities (OpenMax) to flag unknowns.
- **Data:** same as M5, plus the pooled unknown group (Bronchiectasis 7 + Pneumonia 6 + Bronchiolitis 6 = 19) for evaluation only, never for fitting.
- **Augmentation:** none (baseline reimplementation should match the original method's setup as closely as possible for a fair comparison).
- **Required outputs:** full §0.1 metric suite including unknown-detection precision/recall and AUROC/AUPR — this is a direct, citable comparison target for M15.

### M7 — [Member D] Deep ensemble (N≈5)
- **Requires:** M1.
- **Purpose:** gold-standard epistemic-uncertainty baseline and an ensemble-disagreement rejection signal, benchmarked later (M20) against Member B's cross-task disagreement score.
- **Architecture:** 5 independently-seeded copies of the shared backbone (M1) + dual heads (sound-event + disease), different random init and data order per seed.
- **Data:** ICBHI, using B's known/unknown split, plus D's own held-out calibration partition (carved in Phase 1, never used for training).
- **Augmentation:** none yet (augmented re-run happens as part of M27's broader check, not a standalone ensemble re-train).
- **Required outputs:** full §0.1 metric suite **per ensemble member**, plus the ensemble-disagreement score's own unknown-detection precision/recall/AUROC — report all 5 members' size/params/training-time individually and summed (total training cost matters for deployability commentary).

### M8 — [Member D] MC-Dropout variant
- **Requires:** M1.
- **Purpose:** cheap, single-model alternative to the ensemble for epistemic uncertainty.
- **Architecture:** same backbone/heads as M1, dropout retained active at inference time (multiple stochastic forward passes, typically 20–50, to estimate predictive variance).
- **Data:** same as M7.
- **Augmentation:** none yet.
- **Required outputs:** full §0.1 metric suite; report the number of inference-time forward passes used and its effect on latency (relevant to Member C's deployability framing too).

### M9 — [Member D] SNGP (Spectral-Normalized Neural Gaussian Process) variant
- **Requires:** M1.
- **Purpose:** single-model, single-forward-pass alternative to ensembles/MC-dropout — efficient epistemic uncertainty, shown strong in prior audio-calibration work.
- **Architecture:** spectral normalization applied to the backbone's residual layers + a Gaussian Process output layer replacing the final softmax.
- **Data:** same as M7.
- **Augmentation:** none yet.
- **Required outputs:** full §0.1 metric suite; this model's efficiency numbers (size/params/latency) are its main selling point vs. M7 — report the comparison explicitly.

### M10 — [Member D] Evidential deep-learning head
- **Requires:** M1.
- **Purpose:** orthogonal rejection signal — a Dirichlet-parameterized output that gives uncertainty directly from a single forward pass, no ensembling or sampling needed.
- **Architecture:** disease head replaced with a Dirichlet-evidential output layer (predicts concentration parameters instead of softmax probabilities), trained with the evidential loss (e.g., type-II maximum likelihood + KL regularizer).
- **Data:** same as M7.
- **Augmentation:** none yet.
- **Required outputs:** full §0.1 metric suite plus the evidential uncertainty's unknown-detection AUROC/AUPR.

### M11 — [Member D] Post-hoc calibrators (temperature / vector / focal-retrained)
- **Requires:** M5 (disease head logits) and M7–M10 (uncertainty signals to calibrate).
- **Purpose:** fix over-confidence in the raw softmax and cross-task disagreement signals before anything downstream (conformal wrapper, M14) relies on them.
- **Method:** three variants, each fit **only** on D's held-out calibration partition (never train or test data):
 1. Temperature scaling (single scalar).
 2. Vector/Platt scaling (per-class affine transform).
 3. Focal-loss-retrained variant (retrain final layer with focal loss to reduce overconfidence at the source, not just post-hoc).
- **Augmentation:** none yet.
- **Required outputs:** reliability diagrams + Expected Calibration Error (ECE) for each of the three variants, pre- and post-calibration; standard §0.1 items apply but ECE/reliability diagrams are the primary metric here, not accuracy (calibration ≠ accuracy).

---

## Phase 3 (Week 6) — The one real handoff

### M12 — [Member A] Final backbone selection
- **Requires:** M2, M3, M4 (full results from all three).
- **Purpose:** pick the winning architecture (likely AST-based per the Tri-MTL literature precedent, but decided on the actual ablation numbers, not assumed) and freeze it as the shared encoder everyone else builds on.
- **Decision inputs:** accuracy/F1 vs. compute/parameter-count tradeoff from M2–M4's full §0.1 tables — this is a deliberate, documented efficiency-vs-accuracy tradeoff decision, not just "highest accuracy wins," since Member C's compression story depends on starting from a reasonable size. Note: All candidates (M2, M3, M4) must be trained using the same inverse-frequency class-weighted CE loss so the architectural comparison is free of loss-formulation confounds.
- **Deliverable:** one frozen checkpoint + a short written justification (feeds directly into Member A's "why this backbone" defense talking point).

### M13 — [Member B] Disease head + consistency scorer, re-fit on final backbone
- **Requires:** M12, M5.
- **Purpose:** swap the provisional-backbone disease head (M5) onto the real, final backbone (M12) — a re-run, not a redesign, since the architecture was already debugged in Phase 2.
- **Architecture/data/augmentation:** identical to M5, just with M12's backbone instead of M1's.
- **Required outputs:** full §0.1 metric suite; this becomes the "real" disease-diagnosis numbers reported in the paper (M5's numbers were provisional/debugging only).

### M14 — [Member D] Conformal wrapper v1 (provisional disagreement score)
- **Requires:** M11, M13.
- **Purpose:** first pass at the distribution-free false-unknown-flag guarantee, built against M13's (now-final-backbone) disagreement score, even though Member B's full consistency scorer (M15) isn't trained yet — the point is to validate the conformal machinery itself early.
- **Method:** inductive/split conformal prediction over the calibrated disagreement score (from M11), targeting a false-flag rate ≤ α on D's held-out calibration partition.
- **Required outputs:** empirical vs. target α coverage table; prediction-set-size distribution. Re-run properly at M20 once B's full staged mechanism exists.

---

## Phase 4 (Weeks 7–9)

### M15 — [Member B] Cross-task consistency scorer, full OWL Stage 0→1
- **Requires:** M13.
- **Purpose:** the paper's headline novelty mechanism — quantify disagreement between the sound-event head's implied diagnosis and the disease head's actual prediction, then threshold it for unknown-detection.
- **Method:** Stage 0 (known-only training, already done at M13) → Stage 1 (cross-task threshold calibration on the pooled 19-patient unknown group).
- **Data:** ICBHI known (104) + pooled unknown (19), patient-independent, LOPO evaluation (group too small for k-fold).
- **Augmentation:** none yet. (Augmented counterpart is M24.)
- **Required outputs:** full §0.1 metric suite plus unknown-detection precision/recall/AUROC/AUPR; **direct comparison against M6 (OpenMax/Weibull)** on the same pooled-unknown set — this comparison table is the paper's headline result.

### M16 — [Member C] Distillation pipeline / initial student model
- **Requires:** M13 (teacher must exist in some usable form; full teacher is M17, but pipeline-building can start against M13's checkpoint).
- **Purpose:** stand up the cluster-quantized knowledge-distillation (CQKD-style, adapted from Khan & Rafat 2025) pipeline and get a first working student model — not yet the ablation-quality sweep (that's M18).
- **Method:** cluster-quantization-based distillation from the teacher (M13/M17) to a compressed student covering backbone + disease head + consistency scorer.
- **Required outputs:** a single working compressed checkpoint + basic §0.1 metrics as a sanity check before the full sweep.

---

## Phase 5 (Weeks 10–12)

### M17 — [Member B] OWL Stage 2 + forgetting-curve measurement
- **Requires:** M15.
- **Purpose:** incrementally incorporate a subset of previously-flagged "unknowns" (simulating a clinician confirming new cases) and measure catastrophic forgetting of Stage-0 performance.
- **Data:** same as M15, plus the incremental-incorporation subset defined in the OWL protocol.
- **Augmentation:** none yet. (Augmented counterpart is M26, run on this exact model.)
- **Required outputs:** full §0.1 metric suite per stage (0/1/2), forgetting-curve plot (Stage-0 performance retention across Stage 1→2), with and without CQKD-style regularization once M18 exists.

### M18 — [Member C] Compressed student model, cluster-count/quantization sweep
- **Requires:** M16, M17 (final teacher).
- **Purpose:** the actual ablation-quality compression study — sweep cluster-count / quantization granularity vs. accuracy retention, following the CIFAR-10/100-style ablation methodology from the original CQKD paper, adapted to this audio pipeline.
- **Method:** train multiple student variants at different compression levels (e.g., 3–5 cluster-count settings), distilled from M17's teacher.
- **Required outputs:** full §0.1 metric suite **per compression level**, plus **inference latency (ms/sample, batch size 1)** at every point — required in addition to the standard items since deployability is this workstream's headline claim. Compression-ratio vs. accuracy-retention curve is the primary figure.

### M19 — [Member C] OOD evaluation runs (Coswara, SPRSound)
- **Requires:** M17 (final open-world model, trained only on ICBHI).
- **Purpose:** the paper's largest-N evaluation — does the unknown-detection mechanism transfer to genuinely unseen populations?
- **Method:** inference-only (no fine-tuning) evaluation of M17's model on Coswara (crowdsourced, thousands of participants) and SPRSound (2,683 records, 292 pediatric participants), handling each dataset's different label taxonomy and recording format.
- **Required outputs:** full §0.1 metric suite per dataset; this is inference-only so training-time/epoch reporting doesn't apply, but model size/params (inherited from M17) and inference latency do.

### M20 — [Member D] Conformal coverage across OWL stages + calibration-under-shift
- **Requires:** M14, M17, M19.
- **Purpose:** the proper, final version of M14 — now run against B's actual staged mechanism (M17) and tested for whether calibration/coverage survive domain shift (Coswara/SPRSound, via M19).
- **Method:** conformal wrapper re-fit on M17's disagreement score; coverage validated across Stage 0→1→2; calibration (ECE, reliability diagrams) re-measured in-distribution and under M19's shift conditions.
- **Required outputs:** conformal coverage report (empirical vs. target α per stage), prediction-set-size and risk–coverage curves, calibration-under-shift table. This is the paper's headline trust figure.

---

## Phase 6 (Weeks 13–15) — Mandatory augmentation ablation (every member repeats their core model with augmentation)

### M21 — [Member A] CNN baseline — SpecAugment
- **Requires:** M2 (clean counterpart for comparison).
- **Method:** identical to M2, with SpecAugment (time + frequency masking on the log-mel spectrogram) applied to training data only.
- **Required outputs:** full §0.1 metric suite; report as the "augmented" row directly against M2 in the architecture × representation × augmentation grid.

### M22 — [Member A] MobileNet/DenseNet — SpecAugment
- **Requires:** M3.
- **Method:** identical to M3, with SpecAugment applied.
- **Required outputs:** same as M21, paired against M3.

### M23 — [Member A] AST — SpecAugment
- **Requires:** M4.
- **Method:** identical to M4, with SpecAugment applied — this is the most literature-standard augmented configuration (AST + SpecAugment is common practice in prior ICBHI work), so it doubles as a sanity check that the pipeline matches known results.
- **Required outputs:** same as M21, paired against M4.

### M24 — [Member B] Disease head — class-balancing augmentation
- **Requires:** M15 (clean counterpart).
- **Method:** noise injection, pitch-shift, and time-stretch (or RepAugment as a candidate alternative) applied to cycle-level audio before patient-level aggregation, targeting the smaller known classes (Healthy n=26, URTI n=14).
- **Required outputs:** full §0.1 metric suite on **both** known-class accuracy **and** unknown-detection metrics — check explicitly whether balancing known classes helps or hurts the cross-task disagreement signal; report either outcome as a genuine finding.

### M25 — [Member C] Compressed model — domain-robustness augmentation
- **Requires:** M18 (clean counterpart).
- **Method:** mild simulated channel/device noise added to ICBHI training data (train split only), then re-run the compression sweep to see whether it narrows the accuracy gap on Coswara/SPRSound.
- **Required outputs:** full §0.1 metric suite per compression level, plus OOD transfer numbers on Coswara/SPRSound (re-running M19-style evaluation on this augmented+compressed model).

### M26 — [Member C] Teacher (final B model) — domain-robustness augmentation
- **Requires:** M17.
- **Method:** same augmentation as M25, applied to the uncompressed teacher, so the augmentation effect can be isolated from the compression effect (teacher-augmented vs. student-augmented, both vs. their clean counterparts).
- **Required outputs:** full §0.1 metric suite + OOD transfer numbers, same schema as M19.

### M27 — [Member D] Calibration/coverage re-run on augmented backbone
- **Requires:** M20 (clean counterpart), M21–M23 (whichever augmented backbone A recommends as final).
- **Method:** re-fit calibrators and conformal wrapper using the augmented backbone/disagreement score; re-measure ECE and coverage, in-distribution and under Coswara/SPRSound shift.
- **Required outputs:** reliability diagrams + coverage numbers, augmented vs. non-augmented, for at least the two strongest rejection signals identified in M20's comparison (disagreement score + best uncertainty signal from M7–M10).

---

## Phase 7 (Weeks 16–17) — Reconciliation (no new training)

### M28 — Model Zoo & Reporting Appendix reconciliation
- **Requires:** all of the above.
- **Purpose:** merge all 27 models' §0.1 tables into one consistent schema (metrics / confusion matrix / size / params / training time / augmented-vs-clean) — a merge of already-logged data, not a re-formatting exercise, **if** the shared logging convention (fixed columns, consistent file naming) was followed from Week 1 as recommended in the Individual Work Plan v2.
- **Deliverable:** one appendix document (all four members' tables) + four main-text summary tables (one per member) for the manuscript draft.

---

## Notes on reading this file alongside the other two

- This file is the **execution reference** — what to actually run, in what order, with what config.
- `Individual_Work_Plan_v2.md` is the **ownership/grading reference** — who owns which workstream, CRediT roles, individual defense talking points.
- `Project_Proposal_OWMTL_Respiratory_Diagnosis_v2.md` is the **narrative/novelty reference** — why this project is publishable, related work positioning, and the full §8.5 protocol this file operationalizes.
- If the Phase-0 pilot (per the proposal's timeline) shows a weak signal and the plan gets revised with Dr. Khan, this file's model list should be revisited alongside it — some entries (especially the Phase 6 augmentation sweep and Phase 5 OOD/compression runs) are only worth the compute if the core mechanism (M15) is already showing a real effect.
