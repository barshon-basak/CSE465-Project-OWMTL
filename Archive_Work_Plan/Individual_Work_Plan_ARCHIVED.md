# Individual Work Plan — Group 5
## Open-World Multi-Task Learning for Respiratory Disease Diagnosis

**Purpose:** split one coherent Q1-target paper into individually substantial, independently trainable and gradable workstreams — each with its own model(s), dataset slice, ablations, and results section — while all of them integrate into a single shared architecture and manuscript. Three **core** workstreams (Members A, B, C) form the paper's backbone; a **proposed fourth** workstream (Member D) is available as a self-contained extension if the team has a fourth member.

---

## ✅ Progress Checklist

> **Legend:** ☑ = Done · ☐ = Not started · 🔄 = In progress (notebook exists, results incomplete)

---

### Member A — Backbone Engineering & Sound-Event Classification Lead

#### Models & Training
- [x] **M1** — Provisional CNN backbone *(results JSON + checkpoint + training curves delivered)*
- [ ] **M2** — CNN baseline (tuned, final) *(hyperparameter sweep for paper ablation table)*
- [ ] **M3** — MobileNet/DenseNet lightweight backbone *(efficiency comparison)*
- [x] **M4** — AST (Audio Spectrogram Transformer) backbone *(results JSON + training curves delivered)*
- [x] **M12** — Backbone selection (final decision) *(selection documented: M1 CNN chosen over M4 AST)*
- [ ] **M21** — CNN baseline + SpecAugment *(augmentation ablation)*
- [ ] **M22** — MobileNet/DenseNet + SpecAugment *(augmentation ablation)*
- [ ] **M23** — AST + SpecAugment *(augmentation ablation)*

#### Deliverables
- [x] Working, documented preprocessing pipeline (ICBHI 2017)
- [x] Provisional backbone checkpoint handed off (unblocked Member B)
- [x] Final backbone choice documented with justification
- [ ] Full sound-event ablation table: architecture × input representation × augmentation
- [ ] Per-class confusion matrices for all architectures
- [ ] Per-architecture compute/parameter-count comparison table
- [ ] Individual results section written for manuscript

---

### Member B — Open-World Mechanism & Disease-Diagnosis Lead

#### Models & Training
- [x] **M5** — Disease-diagnosis head, Stage 0 (provisional backbone) *(scaffolding — folded into M13)*
- [x] **M6** — OpenMax + Weibull baseline *(notebook exists, results JSON delivered)*
- [x] **M13** — Disease head + consistency scorer, re-fit on final backbone *(results JSON + checkpoint delivered)*
- [x] **M15** — Cross-task consistency scorer, full OWL Stage 0→1 *(the core mechanism)*
- [ ] **M17** — OWL Stage 2 + forgetting curve
- [ ] **M24** — Disease head + class-balancing augmentation

#### Deliverables
- [x] Working disease-diagnosis head on final backbone
- [x] Cross-task consistency scorer implementation
- [x] OpenMax/Weibull baseline completed with results
- [ ] Direct quantitative comparison: cross-task consistency vs. OpenMax/Weibull vs. Cho & Lee (2025)
- [ ] Forgetting-curve results across OWL stages
- [ ] Individual results section written for manuscript (headline novelty table)

---

### Member C — Cross-Dataset Generalization & Compression Lead

#### Models & Training
- [ ] **M16** — Distillation pipeline / initial student model
- [ ] **M18** — Compressed student model, cluster-count sweep
- [ ] **M19** — OOD evaluation runs (Coswara, SPRSound)
- [ ] **M25** — Compressed model + domain-robustness augmentation
- [ ] **M26** — Teacher (final B model) + domain-robustness augmentation

#### Deliverables
- [ ] Coswara dataset acquired and taxonomy-mapped
- [ ] SPRSound dataset acquired and taxonomy-mapped
- [ ] OOD evaluation harness built
- [ ] Full OOD generalization results table (Coswara, SPRSound)
- [ ] Compression ratio vs. accuracy retention curve
- [ ] Forgetting-curve comparison: compressed vs. uncompressed across OWL stages
- [ ] Individual results section written for manuscript

---

### Member D *(proposed)* — Trust, Calibration & Explainability Lead

#### Models & Training
- [ ] **M7** — Deep ensemble (N≈5)
- [ ] **M8** — MC-Dropout variant
- [ ] **M9** — SNGP variant
- [ ] **M10** — Evidential deep-learning head
- [ ] **M11** — Post-hoc calibrators (temperature/vector/focal scaling)
- [ ] **M14** — Conformal wrapper v1 (provisional score)
- [ ] **M20** — Conformal coverage across OWL stages + calibration-under-shift
- [ ] **M27** — Calibration/coverage re-run on augmented backbone

#### Deliverables
- [ ] Patient-independent held-out calibration partition carved
- [ ] Calibration table + reliability diagrams for all rejection signals
- [ ] Conformal coverage report (empirical vs. target α, across OWL stages)
- [ ] Rejection-signal comparison: uncertainty vs. cross-task disagreement
- [ ] Disagreement-attribution figure panel + faithfulness scores
- [ ] Individual results section written for manuscript

---

### Shared Responsibilities (All Members)

- [ ] Joint manuscript sections: Abstract, Introduction, Related Work, Discussion
- [ ] Joint statistical-protocol compliance (LOPO, patient-independent splits)
- [ ] Phase-0 pilot review with Dr. Khan
- [ ] CQKD / compression design review with Dr. Khan
- [ ] **M28** — Model Zoo & Reporting Appendix reconciliation
- [ ] Internal review with Dr. Khan (Weeks 19–20)
- [ ] Final submission

---

## 0. How the Split Works (read this first)

The supervisor's instruction — *individual training, individual significant work* — maps naturally onto the project's own structure, because the paper already has three technically distinct pillars, with a fourth that extends them cleanly:

1. **Shared audio backbone + large-sample sound-event classification** — engineering-heavy, architecture-focused (Member A).
2. **Open-world / open-set mechanism + disease-diagnosis task** — the paper's core novelty, methodology-focused (Member B).
3. **Cross-dataset generalization + model compression** — systems / transfer-learning-focused (Member C).
4. **Trust, calibration, and explainability of the open-world decision** — statistics / trustworthy-AI-focused (Member D, *proposed*).

These are genuinely different skill sets and genuinely separable experiments: no member does a subset of another's work, and no member's results depend on faking or idly waiting on another's unfinished code (with the one deliberate handoff point noted in §6). Each member trains their **own models**, on **their own dataset configuration**, and owns **their own ablation table and results section** in the final paper. All workstreams feed one shared paper narrative and one shared final model.

This also maps cleanly onto the standard **CRediT authorship-contribution categories** (Conceptualization, Methodology, Software, Investigation, Formal Analysis, Data Curation, Validation, Visualization, Writing) that most Q1 journals now require as an explicit contributor statement — so each member's individual role can be documented formally and defensibly at submission, which doubles as evidence for individual capstone grading.

### Workstreams at a glance

| Member | Owns | Core question answered |
|---|---|---|
| **A** | Backbone + sound-event classification | *What representation?* |
| **B** | Cross-task consistency + staged OWL | *Does the unknown-detection mechanism work?* |
| **C** | Cross-dataset OOD + CQKD compression | *Does it generalize and can it be deployed?* |
| **D** *(proposed)* | Calibration, uncertainty, guarantees, explainability | *Can a clinician trust the abstain/flag decision, and why did it fire?* |

---

## 1. Member A — Backbone Engineering & Sound-Event Classification Lead

### Role in one sentence
Owns the shared audio representation (the thing everyone else builds on top of) and the large-sample sound-event classification task — the part of the paper with the most data, the most room for rigorous ablation, and the lowest statistical risk.

### Dataset
- **ICBHI 2017** — full corpus, cycle-level: 6,898 respiratory cycles from 920 recordings, 126 patients.
- Sole responsibility for the **preprocessing pipeline**: resampling to a consistent rate, cycle segmentation from the annotation files, noise handling, and standardizing recording length (commonly 8 s windows at 16 kHz, following prior ICBHI work).

### Task
**4-way sound-event classification** — Normal / Crackle / Wheeze / Both — evaluated with standard k-fold cross-validation (large-N here, so ordinary CV is statistically fine; no LOPO required for this specific task).

### Models to train (this member's own model zoo)
Benchmark at least three backbone families head-to-head, since backbone choice is this member's primary ablation axis and directly determines what Member B builds the disease head on top of:
1. **CNN baseline** — 2D CNN over log-mel spectrograms (comparable to the MTL-MobileNet baseline in the literature).
2. **Lightweight architecture** — MobileNet or DenseNet variant (efficiency comparison point, also relevant to Member C's later compression work).
3. **Transformer-based** — Audio Spectrogram Transformer (AST), following the Tri-MTL literature, pretrained and fine-tuned on ICBHI.

### Deliverables
- A working, documented preprocessing pipeline (shared codebase asset — everyone else imports this).
- Full sound-event ablation table: architecture × input representation (raw waveform vs. log-mel vs. MFCC) × augmentation strategy.
- A chosen "winning" backbone, handed off to Member B as the shared encoder for the disease-diagnosis head.
- Individual results section: sound-event performance, per-class confusion matrices, per-architecture compute/parameter-count comparison.

### Why this is significant individual work
A full architecture search and representation-learning study in its own right — publishable as a standalone contribution to the ablation section, and the piece of the pipeline every other member's code literally imports. Nobody else touches this dataset slice or this task.

### Suggested CRediT roles
Software (primary), Methodology (backbone design), Formal Analysis (architecture comparison), Data Curation (ICBHI pipeline), Visualization (confusion matrices, spectrogram figures).

---

## 2. Member B — Open-World Mechanism & Disease-Diagnosis Lead

### Role in one sentence
Owns the paper's actual novelty claim: the cross-task consistency mechanism that detects unknown diseases, the staged open-world learning (OWL) protocol, and the benchmark comparison against Dr. Khan's own prior open-set method.

### Dataset
- **ICBHI 2017** — patient/diagnosis-level, using Member A's finalized backbone as a frozen or fine-tuned feature extractor (the one deliberate handoff point in the whole plan — see §6 for how to sequence it so nobody is blocked).
- Known classes: COPD (64), Healthy (26), URTI (14) = 104 patients.
- Held-out/unknown classes: Bronchiectasis (7), Pneumonia (6), Bronchiolitis (6) = 19 patients, **pooled into a single "unknown" evaluation group** — not split into three separate n=6 groups (see the statistical rationale in the research guideline).

### Task
**Coarse open-world disease diagnosis** — known-class diagnosis vs. pooled-unknown detection, using cross-task disagreement between the sound-event head (Member A's task) and a new disease-diagnosis head this member builds.

### Models to train (this member's own model zoo)
1. **Disease-diagnosis head** — trained on top of Member A's backbone, patient-level aggregation of cycle-level features.
2. **Cross-task consistency scorer** — the core novel mechanism: quantifies disagreement between the sound-event head's implied diagnosis and the disease head's actual prediction.
3. **OpenMax + Weibull baseline** — a direct reimplementation of Dr. Khan's prior method from the mosquito paper: compute the Mean Activation Vector (MAV) per known class, fit a Weibull distribution to the tail distances, and use the OpenMax-recalibrated probability to flag unknowns. A same-lab prior-method baseline to beat — a clean, citable comparison.
4. **Staged OWL protocol** — Stage 0 (known-only training) → Stage 1 (cross-task threshold calibration + unknown detection) → Stage 2 (incremental incorporation of confirmed "new" cases + forgetting-curve measurement).

### Deliverables
- Working disease-diagnosis head + cross-task consistency scorer.
- Direct quantitative comparison: cross-task consistency vs. OpenMax/Weibull vs. Cho & Lee (2025)'s prototype-distance approach, on the same pooled-unknown evaluation set.
- Forgetting-curve results across OWL stages.
- Individual results section: the paper's headline novelty table.

### Why this is significant individual work
This is the mechanism the entire paper's novelty rests on; it requires reimplementing a full competing baseline (OpenMax/Weibull) from scratch; and it is evaluated on a completely different task formulation (patient-level, open-world) than Member A's cycle-level closed-set task. No overlap in what is being measured.

### Suggested CRediT roles
Conceptualization (core mechanism), Methodology (OWL protocol design), Software (consistency scorer, OpenMax baseline), Investigation (staged OWL experiments), Formal Analysis (novelty benchmarking).

---

## 3. Member C — Cross-Dataset Generalization & Compression Lead

### Role in one sentence
Owns the paper's "real-world scale" evidence (large-N out-of-distribution testing on populations the model has never seen) and the efficiency/deployability contribution (cluster-quantized compression of the full pipeline).

### Datasets
- **Coswara** — crowdsourced, thousands of participants, used purely as an out-of-distribution stress test (no fine-tuning — the point is to throw a genuinely unseen population/recording setup at the model trained only on ICBHI).
- **SPRSound** — 2,683 records, 9,089 respiratory sound events, 292 pediatric participants (a second, independent domain shift — different age group, different recording protocol).

### Task
1. **Cross-dataset OOD generalization** — take Member B's finished open-world model (trained only on ICBHI) and evaluate how well its unknown-detection mechanism transfers to Coswara and SPRSound populations it has never seen. This is the largest-N evaluation in the entire paper and carries the statistical weight the small ICBHI held-out classes cannot.
2. **Cluster-quantized compression** — apply a cluster-quantization-based distillation approach (adapting Khan & Rafat, 2025 — correctly attributed as external prior work, not in-house) to compress the full backbone + disease head + consistency scorer into a deployable model, and measure whether quantization additionally regularizes drift across Member B's OWL stages (i.e., reduces forgetting).

### Models to train (this member's own model zoo)
1. **OOD evaluation pipeline** — no new training per se, but a full evaluation harness across Coswara and SPRSound, handling their different label taxonomies and recording formats relative to ICBHI.
2. **Compressed (student) model** — cluster-quantized version of the full pipeline, trained via distillation from Member B's full (teacher) model.
3. **Compression ablation** — sweep cluster-count / quantization granularity vs. accuracy retention, following the CIFAR-10/CIFAR-100-style ablation methodology in the original CQKD paper, adapted to this audio pipeline.

### Deliverables
- Full OOD generalization results table (Coswara, SPRSound) — the paper's primary "this generalizes" evidence.
- Compression ratio vs. accuracy retention curve.
- Forgetting-curve comparison: compressed vs. uncompressed model across Member B's OWL stages.
- Individual results section: generalization + deployability, the two things reviewers ask about model practicality.

### Why this is significant individual work
A completely separate technical skill set (transfer/domain-shift evaluation + model compression) applied to completely different data than either core member touches, producing the paper's two most reviewer-facing claims (does it generalize, can it actually be deployed) as a self-contained contribution.

### Suggested CRediT roles
Software (compression pipeline), Investigation (OOD experiments), Data Curation (Coswara/SPRSound integration), Formal Analysis (compression ablation), Validation (cross-dataset generalization claims).

---

## 4. Member D — Trust, Calibration & Explainability Lead *(proposed fourth workstream)*

> Include this workstream only if the team has a fourth member. It is a complete, self-contained pillar — but the paper stands on Members A–C alone, so Member D is framed as an extension rather than a dependency.

### Role in one sentence
Owns whether the open-world decision can be *trusted*: calibrating the confidence of the unknown-flag, wrapping it in a conformal layer that gives a formal false-flag guarantee, quantifying uncertainty under domain shift, and producing clinician-facing explanations of **why** each "unknown" was flagged.

### Dataset configuration (own slice — no collision with A/B/C)
- **ICBHI 2017** — reuses Member B's known/unknown split, but carves an additional **patient-independent held-out calibration set** from the *known* classes, used *only* for conformal calibration and temperature fitting, never for training. This clean calibration partition is a data-handling contribution most ICBHI papers omit.
- **Coswara + SPRSound** — shared with Member C but for a different measurement: C measures *accuracy* transfer, D measures *calibration* transfer (does confidence stay honest off-distribution — calibration is known to collapse under shift, so this is a distinct, non-redundant experiment on the same data).

### Task
1. **Calibration study** of the open-world confidence signal (known-class softmax + cross-task disagreement score) — measure and *fix* over-confidence.
2. **Conformal abstention layer** over Member B's detector — a distribution-free guarantee that the false-unknown-flag rate ≤ α, holding across OWL stages and under Coswara/SPRSound shift.
3. **Uncertainty-aware rejection** — benchmark epistemic-uncertainty-based unknown detection (ensemble / MC-dropout / evidential / SNGP) as an *alternative* rejection signal against Member B's disagreement score and OpenMax baseline.
4. **Explainability of the rejection** — disagreement-attribution maps showing which time–frequency regions of the spectrogram drove the two heads apart and triggered the "unknown" flag.

### Models to train (this member's own model zoo)
1. **Deep ensemble** of the shared backbone + dual heads (N ≈ 5 independently-seeded models) — the gold-standard uncertainty baseline and an ensemble-disagreement rejection signal.
2. **MC-Dropout Bayesian variant** — dropout-at-inference for cheap epistemic uncertainty.
3. **SNGP variant** — spectral-normalized neural Gaussian process; a single-model alternative to ensembles, shown strong/efficient in audio-calibration work.
4. **Evidential deep-learning head** — a Dirichlet-parameterized disease head that outputs uncertainty directly (an orthogonal rejection signal).
5. **Post-hoc calibrators** — temperature scaling, vector/Platt scaling, and a focal-loss-retrained variant, each fit on the held-out calibration partition.
6. **Conformal wrapper** — inductive/split conformal plus a conformal-risk-control variant; not "trained" but a full statistical layer with its own parameterization and coverage validation.
7. **Explainer suite** — Grad-CAM / attention-rollout / SHAP configured for the *disagreement* objective, plus a faithfulness harness (deletion/insertion AUC, pointing game against crackle/wheeze annotations).

### Deliverables
- **Calibration table + reliability diagrams** for every rejection signal (softmax, disagreement, ensemble, MC-dropout, SNGP, evidential), pre- and post-calibration, in-distribution and under shift.
- **Conformal coverage report** — empirical vs. target α across OWL Stage 0→1→2, with prediction-set-size and selective-risk (risk–coverage) curves. *The headline trust figure.*
- **Rejection-signal comparison** — does uncertainty-based rejection beat, complement, or lose to Member B's cross-task disagreement? A clean, citable ablation that de-risks B's mechanism either way.
- **Disagreement-attribution figure panel** + quantitative faithfulness scores.
- Individual results section: *Calibration, Guarantees, and Explainability of the Open-World Decision.*

### Why this is significant individual work
It trains a distinct model zoo (ensembles, Bayesian, SNGP, evidential — none of which A/B/C build), is evaluated on an entirely different axis (calibration error, coverage, faithfulness — never accuracy/F1), and converts the project's most attackable design choice — a hand-set disagreement threshold on the tiny n=19 unknown pool — into a formal, distribution-free guarantee. It is defensible standalone as *a trustworthy-AI layer for open-world respiratory screening.*

### Why this is genuine whitespace (not re-tread)
- **Plain explainability on lung sound is crowded** — Grad-CAM/LIME/SHAP on ICBHI is already published (including on ICBHI + Coswara). Member D therefore explains the *rejection decision specifically* (why the two heads disagreed), an angle no prior work takes because none has a cross-task rejection mechanism to explain.
- **Audio-classifier calibration foundations exist but not for ICBHI/respiratory open-set** — benchmarks of MC-dropout, deep ensembles, focal loss, and SNGP are on environmental sound and music, not respiratory audio.
- **Conformal open-set abstention is emerging but unclaimed in medical audio** — recent work formalizes conformal reject-options and open-set conformal p-values, but none is applied to respiratory sound or to a cross-task-disagreement detector.

### Suggested CRediT roles
Methodology (calibration + conformal protocol), Software (ensemble/Bayesian/SNGP/evidential + conformal wrapper + XAI harness), Validation (coverage guarantees, calibration-under-shift), Formal Analysis (risk–coverage, faithfulness), Visualization (reliability diagrams, disagreement-attribution maps).

### Alternatives considered for a fourth role
| Candidate | Verdict |
|---|---|
| **Trust / calibration / conformal / XAI (chosen)** | ✅ Non-overlapping, own model zoo, fixes the project's biggest statistical weakness, strong venue fit, verified whitespace. |
| Self-supervised backbone pretraining | ❌ Creates an *upstream* dependency on Member A — blocks the team; violates the "no idle waiting" principle. |
| Pure explainability (Grad-CAM/LIME/SHAP) | ❌ Already crowded on ICBHI+Coswara; not standalone unless tied to the rejection decision (which the chosen role does). |
| Noise-robustness / audio-enhancement only | ⚠️ Partially overlaps Member C's "does it survive shift" framing — folded into Member D as an optional stretch instead. |
| A brand-new disease/dataset + new classifier | ⚠️ Adds breadth but dilutes the focused open-world narrative — keep only as a scoped optional task. |

### Optional stretch (only if scope allows)
The role above is already complete. If the team also wants Member D to touch fresh data, the cleanest non-overlapping extension is a **noise-robustness study** (add stethoscope/ambient/heartbeat noise and test whether calibration and conformal coverage *survive* degradation) or treating **COVID-19 status on Coswara** as a second open-world stress task to test whether the conformal guarantee transfers to a new disease domain. Lock the core role first.

---

## 5. Shared Responsibilities (all members)

- **Joint manuscript sections** — Abstract, Introduction, Related Work, and Discussion are co-written; no single member owns the "story" alone, since the novelty claim depends on the pillars fitting together.
- **Joint statistical-protocol compliance** — every member independently applies the same rules from the evaluation guideline: LOPO instead of k-fold for small groups, no bootstrap CIs under n=15, patient-independent splits everywhere. Cross-check each other's evaluation code before merging results.
- **Joint faculty checkpoints** — everyone attends the Phase-0 pilot review and the CQKD/compression design review with Dr. Khan, since these are the two points where the whole team's direction could shift.

---

## 6. Sequencing and the One Real Dependency

Almost everything here runs in parallel from day one — Member C can build the Coswara/SPRSound data pipelines and label-taxonomy mapping immediately, and Member D can build the calibration/uncertainty toolkit against a provisional backbone, both independent of the others. The **one unavoidable handoff** is:

> Member B's disease-diagnosis head needs Member A's finalized backbone choice.

To avoid Member B sitting idle, Member A delivers a **provisional backbone** (the CNN baseline, fastest to get working) in the first 2–3 weeks, so Member B can build and debug the disease head and consistency scorer against it immediately, then swap in the final winning architecture (likely AST-based) once the full backbone ablation concludes. Nobody blocks anybody for more than the first few weeks. Member D's toolkit is model-agnostic, so swapping in B's final model is a re-run, not a rebuild — the same pattern.

### Combined timeline

| Week | Member A | Member B | Member C | Member D *(proposed)* |
|---|---|---|---|---|
| 1–2 | ICBHI preprocessing; provisional CNN | Disease-head skeleton | Coswara/SPRSound acquisition; taxonomy mapping | Calibration/UQ toolkit on provisional backbone; carve held-out calibration partition |
| 3–5 | Backbone ablation (CNN/MobileNet/AST) | OpenMax/Weibull baseline | OOD evaluation harness design | Train ensemble + MC-dropout + SNGP + evidential; temperature/focal calibration |
| 6 | **Final backbone handoff** | Swap to final backbone; consistency scorer | Compression pipeline design | Conformal wrapper v1 on provisional disagreement score; reliability diagrams |
| 7–9 | Sound-event ablation finalization | OWL Stage 0→1; consistency vs. OpenMax | Distillation/compression implementation | Conformal coverage across OWL stages; rejection-signal comparison |
| 10–12 | Integration support; debug shared code | OWL Stage 2; forgetting-curve | OOD runs (Coswara, SPRSound) | Calibration-under-shift; disagreement-attribution XAI + faithfulness |
| 13–15 | Joint ablation cross-checks | Full novelty benchmarking table | Compression ablation; forgetting-with-compression | Final trust results; risk–coverage curves; XAI figure panel |
| 16–18 | Joint manuscript writing | Joint manuscript writing | Joint manuscript writing | Joint manuscript (owns trust/calibration/XAI subsections) |
| 19–20 | Internal review with Dr. Khan | Internal review with Dr. Khan | Internal review with Dr. Khan | Internal review with Dr. Khan |
| 21 | Submission | Submission | Submission | Submission |

---

## 7. Individual Grading / Defense Talking Points

If the program requires each member to individually defend their contribution, each should be able to answer, on their own:

- **Member A** — "Why this backbone over the alternatives, and what does the ablation show about the accuracy/compute tradeoff?"
- **Member B** — "Why does cross-task disagreement outperform (or not) the lab's own prior OpenMax/Weibull method, and what does the forgetting curve tell us about staged learning?"
- **Member C** — "How much does performance degrade on genuinely unseen populations, and what is the compression/accuracy tradeoff for real-world deployment?"
- **Member D** *(if included)* — "When the model flags a patient as an unknown condition, how often is that flag wrong, is that rate provably bounded, does confidence stay honest under an unseen population, and which part of the breathing sound drove the decision?"

Each answer draws on a dataset, a model, and a results table that only that member produced — satisfying the individual-significant-work requirement directly.

---

## 8. Final Notes

- All workstreams share one paper, one codebase (with clear module boundaries: `backbone/`, `owl_mechanism/`, `generalization_compression/`, and — if Member D is included — `trust_calibration/`), and one final architecture, while each member's individual Investigation, Software, and Formal Analysis contributions remain cleanly separable and independently defensible.
- Set up the shared repository with these folders from week 1, plus a shared `data/` folder that Member A's preprocessing pipeline populates and everyone reads from — this avoids multiple people writing multiple ICBHI loaders.
- Revisit this plan with Dr. Khan after the Phase-0 pilot (per the research guideline). If the pilot signal is weak, the split may need adjusting before members commit months of individual work to it.
