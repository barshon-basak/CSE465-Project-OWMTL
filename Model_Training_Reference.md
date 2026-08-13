# Model Training Reference — OWMTL Project

> **Restructured 2026-08-05.** This file used to organize work into "Stages" owned by lettered
> members (A/B/C/D) with a target of ~28 model runs. That framing is retired: Dr. Khan's 2026
> guidance is explicit that the project is not graded on model count, and the audit
> (`Asif's/audit/PROJECT_AUDIT.md`) found that treating headcount as the goal produced a real cost —
> most of what was reported as "done" under the old plan turned out to be synthetic-data
> scaffolding, not results. The historical role-based version is archived at
> `Archive_Work_Plan/Model_Training_Reference_ARCHIVED.md`.
>
> This file is now organized into **chunks of work**, not people. A chunk describes what needs to be
> true, not who's responsible for making it true — pick up whichever chunk is next and unblocked.
> **`Novelty Search.md` is now equally load-bearing as this file** — check it before starting any
> new model, not just this reference.

---

## 0. Where the project actually stands (verify this before trusting anything below)

Run `python3 "Asif's/audit/audit_project.py"` for the current, authoritative answer — it's ~1 second,
no GPU, no dataset. As of the last run:

| Status | Models | What that means |
|---|---|---|
| ✅ **Real, verified** | M1, M2, M3, M4, M6, M11, M12, M13, M15, M17, M24-CB, M29 | Trained/evaluated on actual ICBHI audio, schema-compliant, audit-clean |
| 🔴 **Synthetic — not results** | M18, M19, M20, M21 | Dataset classes fabricate `torch.randn`/`np.random` tensors instead of loading audio. Every metric from these describes noise. |
| ⚪ **Not started** | M5, M7–M10, M14, M16, M22–M23, M25–M27 | — |

**The single most important consequence:** the core cross-task consistency mechanism and Open-World Learning pipeline (M13 v4, M15 v4, M17 v2) have now been fully rebuilt on real audio and verified audit-clean!

---

## 1. Chunks

Chunks are ordered by dependency, not by week or by person. A chunk "requires" another chunk when it
literally cannot start without that chunk's checkpoint/output — not because of a team schedule.

### Chunk A — Shared Foundations
**Status: ✅ done.** ICBHI preprocessing pipeline, `Model_Training_Protocol.md`'s schema and rules,
the audit tooling (`Asif's/audit/`). Everything else imports this.

### Chunk B — Sound-Event Backbone
**Status: ✅ done and decided.** M1 → M2/M3/M4 → M12 (selection: M2). See §2.1–§2.5 below for the
individual model specs. **Optional stretch only** (§2.6, M21–M23 SpecAugment variants) — not
required, since backbone selection is already settled and augmentation ablations don't change it.

### Chunk C — Open-Set Baselines
**Status: ✅ done.** M6 (OpenMax/Weibull, real, negative) and M29 (trivial post-hoc OOD scores, real,
sets the actual bar at AUROC 0.6466). See §2.7–§2.8.

### Chunk D — Disease Head + Cross-Task Mechanism
**Status: ✅ done.** M13 v4 (Prototypical Disease Head), M15 v4 (Cross-Task Consistency Scorer), and M17 v2 (Stage 2 OWL & Forgetting Curve) are **all real, verified, and audit-clean**. See §2.9–§2.11.

### Chunk E — Novelty Layer
**Status: scope decided, not started.** Co-equal in priority with Chunk D, not an afterthought.

**The set is fixed at 2–3 items and is not open for opportunistic additions** — see
`Novelty Search.md` §4.0 for the full reasoning. Short version: with one dataset and 19 unknown
patients, nine simultaneous changes can't be ablated apart, so a nine-technique paper reads as a
buzzword list and loses more credibility than it gains.

| Selected | Answers | Note |
|---|---|---|
| Meta-learning / prototypical disease head | Attack 5 (URTI n=14) | **Not separate from Chunk D — it *is* how M13 should be rebuilt.** |
| Conformal calibration of the disagreement score | Attack 6 (no formal guarantee) | Post-hoc wrapper on a real M15; see §2.18 (M14) |
| M2+M3 feature-fusion ensemble *(optional)* | Dr. Khan's list item #9 | Buildable today; drop it if it doesn't beat both backbones alone |

Everything else in `Novelty Search.md` §4 is **deferred by decision**. If a selected item fails,
swap a deferred one in — don't accumulate.

### Chunk F — Generalization & Compression (optional / stretch)
**Status: 🔴 broken (M18 constant-accuracy across a 62× sweep; M19 synthetic).** **Requires:** Chunk D
to be real first — there's no point compressing or stress-testing a mechanism that's never been
evaluated. Explicitly optional: skip if time runs out, since it's not what's graded on anymore. See
§2.12–§2.14.

### Chunk G — Trust, Calibration & Uncertainty (optional / stretch)
**Status: ⚪ mostly not started.** **Requires:** Chunk D. Also optional — several of its components
(ensemble, MC-Dropout, evidential, conformal calibration) overlap directly with Chunk E's novelty
items, so check `Novelty Search.md` before building these separately. See §2.15–§2.20.

### Chunk H — Reporting & Reconciliation
**Status: not started.** Merge every audit-clean `results_M*.json` into one appendix. No new
training. Only include models the audit marks real — do not merge synthetic runs into the paper's
tables. See §2.21.

---

## 2. Model specifications

Same model IDs as before (M1, M2, ... M29) — the numbering is load-bearing across the codebase
(folder names, JSON schemas, the audit tool) and hasn't changed. Only the ownership/phase framing
has been removed; the technical specs below are otherwise as originally written.

### 2.1 — M1: Provisional CNN backbone
- **Chunk:** B. **Requires:** nothing. **Status:** ✅ done, real.
- **Purpose:** the fast, "good enough" checkpoint the disease-diagnosis pipeline is built and debugged against before the real architecture search finishes.
- **Architecture:** 2D CNN over log-mel spectrograms (simple conv stack, no heavy tuning).
- **Input representation:** log-mel spectrogram, 8 s windows @ 16 kHz, 128-dim Fbank features, 25/10 ms window/overlap.
- **Data:** ICBHI 2017, full corpus, cycle-level, standard train/test split; sound-event labels (Normal/Crackle/Wheeze/Both).
- **Loss function:** Inverse-frequency class-weighted `CrossEntropyLoss`.
- **Augmentation:** none.
- **Required outputs:** full §3 metric suite. Treat this as a throwaway/reference checkpoint — the real reported numbers come from M2.

### 2.2 — M2: CNN baseline (tuned, final)
- **Chunk:** B. **Requires:** nothing. **Status:** ✅ done, real, **the M12-selected backbone.**
- **Purpose:** the properly-tuned version of M1 — the number that goes in the architecture ablation table.
- **Architecture:** 2D CNN over log-mel spectrograms, same family as M1 but with a real hyperparameter sweep (depth, filter counts, dropout, learning-rate schedule).
- **Data:** ICBHI 2017, full corpus, cycle-level, k-fold cross-validation.
- **Loss function:** Inverse-frequency class-weighted `CrossEntropyLoss` (kept constant across M2/M3/M4 to prevent confounds in the M12 selection).
- **Augmentation:** none.
- **Result:** **official ICBHI 0.6138** (legacy macro metric: ~~0.7227~~), accuracy 0.6138, F1 0.5238 — beats every other real candidate on both metrics. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`.

### 2.3 — M3: MobileNet/DenseNet (lightweight)
- **Chunk:** B. **Requires:** nothing. **Status:** ✅ done, real.
- **Purpose:** efficiency comparison point in the backbone race; also a useful reference point for Chunk F's compression work.
- **Architecture:** MobileNet or DenseNet variant, pretrained on ImageNet if available, fine-tuned on ICBHI log-mel spectrograms.
- **Loss function:** same as M2/M4.
- **Augmentation:** none.
- **Result:** **official ICBHI 0.5895** (legacy macro: ~~0.6984~~) — real, second-best real backbone candidate. Its SpecAugment variant M22 reaches **0.6495**, the best score in the project on the official 60/40 split.

### 2.4 — M4: AST (Audio Spectrogram Transformer)
- **Chunk:** B. **Requires:** nothing. **Status:** ✅ done, real.
- **Purpose:** transformer-based backbone comparison point.
- **Architecture:** AST, ImageNet+AudioSet-pretrained checkpoint, fine-tuned on ICBHI. Standard AST preprocessing: 128-dim Fbank, mean/std normalization (−4.27 / 4.57).
- **Data/representation:** two documented preprocessing deviations from §2 to preserve AudioSet pretraining: (1) full-band mel filterbank (20–8000 Hz) instead of 50–2000 Hz; (2) `n_fft=512` instead of 1024.
- **Loss function:** same as M2/M3.
- **Augmentation:** none.
- **Result:** legacy macro ICBHI 0.6359 — real, but **last** of the three real backbone candidates on every metric; also 24× larger and 30× slower than M2. Lost the M12 selection. ⚠️ **M4 has no committed `results_M4.json`** — its numbers are transcribed from notebook output into M12's `decision.comparison_table`, so the official metric cannot be recomputed for it. Commit the results JSON if M4 is to appear in the paper.

### 2.5 — M12: Final backbone selection
- **Chunk:** B. **Requires:** M2, M3, M4 (full results from all three). **Status:** ✅ done, real, verified.
- **Purpose:** pick the winning architecture on actual ablation numbers and freeze it as the shared encoder everything else builds on.
- **Decision inputs:** accuracy/F1 vs. compute/parameter-count tradeoff from M2–M4's full §3 tables — documented efficiency-vs-accuracy tradeoff, not just "highest accuracy wins."
- **Result:** **M2 selected.** Checkpoint verification confirmed the frozen `best_model.pth` reproduces the reported metrics exactly. Full decision rule, robustness check (what if M1 had been eligible?), and justification in `Asif's/M12/`.
- **Deliverable:** one frozen checkpoint + a generated written justification.

### 2.6 — M21/M22/M23: Backbone + SpecAugment (optional stretch)
- **Chunk:** B (optional). **Requires:** M2/M3/M4 respectively. **Status:** ⚪ M21 exists but reports metrics of exactly 1.0 (audit: `perfect_metrics_implausible` — almost certainly a train/test leak, needs redoing if picked up).
- **Method:** identical to the clean counterpart, with SpecAugment (time + frequency masking) applied to training data only.
- **Priority:** low. The backbone decision (M12) is already settled with real data; an augmentation ablation on it is a nice-to-have for the paper's completeness, not something that changes any conclusion. Don't spend time here before Chunk D is real.

### 2.7 — M6: OpenMax + Weibull baseline
- **Chunk:** C. **Requires:** M1 or M2 (the backbone with a features extractor). **Status:** ✅ done, real.
- **Purpose:** the same-lab prior-method baseline that the core mechanism (M15) needs to beat — the comparison that makes the novelty claim defensible.
- **Method:** compute the Mean Activation Vector (MAV) per known class, fit a Weibull distribution to the tail distances, recalibrate softmax probabilities (OpenMax) to flag unknowns.
- **Data:** known classes (COPD 64, Healthy 26, URTI 14 = 104 patients) + pooled unknown (Bronchiectasis 7 + Pneumonia 6 + Bronchiolitis 6 = 19), unknown group evaluation-only, never fitted.
- **Result:** real audio, 72 train / 32 known-test / 19 unknown-test patients. `AUROC 0.4516`, `unknown_recall 0.0255` — **a genuine negative result**, below chance. This is a real, citable finding, not a bug: OpenMax barely detects any true unknowns on this data.

### 2.8 — M29: Open-set baseline suite on the M12 backbone
- **Chunk:** C. **Requires:** M12 (frozen backbone checkpoint). **Status:** ✅ done, real. *(New model ID, added 2026-08. Not in the original 28-model list — see `Asif's/M29/README.md` for the full spec.)*
- **Purpose:** the trivial-baseline ablation Reviewer #2 would demand (`Novelty Search.md` §2, Attack 1) — MSP, entropy, energy, and Mahalanobis distance, computed on frozen M12 embeddings, no training.
- **Data:** exact real ICBHI known/unknown split — 104 known / 19 unknown patients, patient-level evaluation, unknown group never fitted.
- **Result:** best baseline (Energy) reaches **AUROC 0.6466**, comfortably beating M6's 0.4516 with zero training. **This is now the real bar any cross-task mechanism (M15) has to clear.**
- **Ablation group:** `rejection_method`, alongside M6 and M15.

### 2.9 — M13: Disease-diagnosis head, re-fit on final backbone
- **Chunk:** D. **Requires:** M12. **Status:** 🔴 broken. Two versions exist: "OLD" shows single-class collapse (predicts one class for everything, frozen validation curve, best epoch 1); "UPDATED" fixes the collapse but best epoch is still 1 of 50 (never really trained past initialization).
- **Purpose:** the real (non-provisional) disease-diagnosis numbers — patient-level classification of COPD/Healthy/URTI on top of the M12 backbone.
- **Architecture:** classification head on top of M12's frozen (or lightly fine-tuned) features, with patient-level aggregation of cycle-level features (mean/attention-pool over a patient's cycles — see `Novelty Search.md` §4.9 for a proposed temporal-transformer upgrade to this aggregation step).
- **Data:** ICBHI 2017, patient/diagnosis-level, known classes only (104 patients), patient-independent split.
- **What needs to happen:** re-run on real data with a working training loop (something in the "UPDATED" version still prevents it from training past epoch 1 — diagnose before re-running, don't just increase epochs). Consider building this as a meta-learning/prototypical head instead of a plain classifier — `Novelty Search.md` §4.3 argues this is both cheaper and better-suited to the n=14 URTI problem than the current architecture.

### 2.10 — M15: Cross-task consistency scorer, full OWL Stage 0→1
- **Chunk:** D. **Requires:** M13 (real). **Status:** 🔴 **not evaluated — this is the actual bottleneck.**
- **Purpose:** the paper's headline mechanism — quantify disagreement between the sound-event head's implied diagnosis and the disease head's actual prediction, then threshold it for unknown-detection.
- **What's currently there:** a notebook whose `ICBHI_OWL_Dataset` class returns `torch.randn(1, n_mels, 801) * 0.5` — no audio is ever loaded, despite the name. It also has a fallback that silently proceeds with a randomly-initialized model if the upstream checkpoint is missing. The reported AUROC (0.6183) and the "beats M6 by 36.9%" claim are both artifacts of this — the notebook compares against M6's 0.4516, which is itself below chance, so the ratio is meaningless even setting the synthetic-data problem aside.
- **Method (unchanged from original spec):** Stage 0 (known-only training, from M13) → Stage 1 (cross-task threshold calibration on the pooled 19-patient unknown group).
- **Data:** ICBHI known (104) + pooled unknown (19), patient-independent, **patient-level** LOPO evaluation (the existing broken run used 518/101 *cycles*, which both inflates N and weakens patient-independence on the tiny unknown group — fix this when rebuilding).
- **Required outputs:** full §3 metric suite plus unknown-detection precision/recall/AUROC/AUPR; direct comparison against **both** M6 (0.4516) and M29 (0.6466) — 0.6466 is the number that actually has to be cleared.
- **This is the single highest-priority item in the entire document.** Nothing downstream (M17–M20, any augmentation of the mechanism) is worth building until this runs on real data.

### 2.11 — M17: OWL Stage 2 + forgetting-curve measurement
- **Chunk:** D. **Requires:** M15 (real). **Status:** 🔴 synthetic, blocked on M15.
- **Purpose:** incrementally incorporate a subset of previously-flagged "unknowns" and measure catastrophic forgetting of Stage-0 performance.
- **Data:** same as M15, plus the incremental-incorporation subset.
- **Required outputs:** full §3 metric suite per stage (0/1/2), forgetting-curve plot.
- **Note:** `Novelty Search.md` §4.5 (multistage distillation) is directly relevant here — the CQKD-forgetting link is independently flagged as the paper's weakest theoretical claim (Attack 2, `Novelty Search.md` §2); consider a staged/teacher-assistant distillation as the actual forgetting-prevention mechanism instead of asserting CQKD does it.

### 2.12 — M16: Distillation pipeline / initial student model
- **Chunk:** F (optional). **Requires:** M17 (real). **Status:** 🟡 notebook generated (`Barshon's/M16/M16_Knowledge_Distillation.ipynb`), needs Kaggle/Colab run. Replaces legacy scaffolding with real audio Teacher-Student Knowledge Distillation ($T=3.0, \alpha=0.7$, 9.0× parameter reduction).
- **Purpose:** stand up the knowledge distillation pipeline from M17 teacher to lightweight student model.

### 2.13 — M18: Structured Pruning & Quantization Sweep
- **Chunk:** F (optional). **Requires:** M17 (real). **Status:** 🟡 notebook generated (`Barshon's/M18/M18_CQKD_Compression_Sweep.ipynb`), needs Kaggle/Colab run. Replaces legacy constant sweep with real audio structured L1 pruning (0%, 20%, 40%, 60%) and dynamic INT8 quantization.
- **Method:** sweep pruning ratio & quantization granularity vs. accuracy retention, model size (MB), and latency (ms).
- **Required outputs:** full §3 metric suite per compression level, plus inference latency and `compression_sweep_results.csv`.

### 2.14 — M19: OOD evaluation runs (Coswara, SPRSound)
- **Chunk:** F (optional). **Requires:** M17 (real). **Status:** ✅ **Done (real & audit-clean)**. Real audio evaluation completed across ICBHI Unknown (AUROC 0.5543), Coswara OOD (AUROC 0.4881), and SPRSound pediatric dataset ($N=1000$, AUPR 0.6265, Overall OOD AUPR 0.6366).
- **Purpose:** the largest-N evaluation in the project — does unknown-detection transfer to genuinely unseen populations?
- **Method:** inference-only evaluation of M17 Stage-2 prototypical model on Coswara and SPRSound.



### 2.15 — M7: Deep ensemble (N≈5)
- **Chunk:** G (optional). **Requires:** M1 or M2. **Status:** ⚪ not started.
- **Purpose:** gold-standard epistemic-uncertainty baseline. Overlaps with `Novelty Search.md`'s ensemble-fusion item (§4.4) — check there before building this as a separate thing.

### 2.16 — M8/M9/M10: MC-Dropout / SNGP / Evidential variants
- **Chunk:** G (optional). **Requires:** M1 or M2. **Status:** ⚪ not started.
- **Priority:** pick at most one of these three as an alternative uncertainty signal, not all three — diminishing returns, and evidential deep learning specifically overlaps with the disease-head redesign in Chunk E.

### 2.17 — M11: Post-hoc calibrators (temperature/vector/focal)
- **Chunk:** G (optional). **Requires:** M13 (real) and at least one of M7–M10. **Status:** ✅ done, real (`Barshon's/M11/m11-post-hoc-calibrators-temperature-vector.ipynb`). Re-run on real ICBHI audio log-mel spectrograms; Temperature, Vector, and Focal loss calibrators evaluated.

### 2.18 — M14: Conformal wrapper v2
- **Chunk:** G (optional). **Requires:** M2, M13 (real), M15 disagreement scores. **Status:** ✅ done, real (`Barshon's/M14/v2/results_M14.json`). Multi-score conformal calibration wrapper (Selected Novelty Item #2, answers Reviewer Attack #6). Evaluates 8 score formulations on real ICBHI audio; auto-selects multiplicative score (`dist*ent`, cal AUROC 0.5842) and achieves 95.45% empirical coverage at 95% nominal level.

### 2.19 — M20: Conformal coverage across OWL stages
- **Chunk:** G (optional). **Requires:** M14, M17, M19 (real). **Status:** 🔴 synthetic.

### 2.20 — M24/M25/M26/M27: Augmentation variants
- **Chunk:** D/F/G (optional, matches whichever mechanism they augment). **Requires:** the real (non-augmented) counterpart. **Status:** ⚪/🔴 depending on the base model. **Priority:** lowest in the document — augmentation ablations on top of mechanisms that haven't been verified on real data yet aren't worth building.

### 2.20 — M24/M25/M26/M27: Augmentation variants
- **Chunk:** D/F/G (optional, matches whichever mechanism they augment). **Requires:** the real (non-augmented) counterpart. **Status:** ✅ M24-CB done, real (`Barshon's/M24/M24_Class_Balancing_Augmentation.ipynb`). Re-run on real ICBHI audio with targeted SpecAugment on Healthy & URTI classes.
- **Purpose:** merge all audit-clean `results_M*.json` files into one consistent appendix.
- **Hard rule:** only merge models the audit marks real. A synthetic-data result in the paper's tables is worse than a missing row.

### 2.22 — M30: M2 + M3 Gated Feature-Fusion Ensemble
- **Chunk:** E (Novelty Layer). **Requires:** M2 and M3 (real checkpoints). **Status:** 🔴 **result withdrawn — needs re-export.** Fuses 768-dim M2 (CNN) and 1280-dim M3 (MobileNetV2) frozen features via a Gated Adaptive Fusion (GAF) head. **Selected Novelty Item #3** — see `Novelty Search.md` §4.4.
  - The reported ICBHI 0.8213 is invalid on three counts: it is the **legacy macro metric**, on a **`patient_independent_70_30` split** (M2/M3 use the official 60/40), and `results_M30.json` **commits no `confusion_matrix_raw`**, so no score can be recomputed from it at all. The "+9.86% over M2" claim crosses both a metric and a split boundary.
  - **To restore:** re-run on the official 60/40 split and export `confusion_matrix_raw` + `icbhi_score_official`. Both input checkpoints (M2, M3) are real and already in the repo, so this is cheap. For calibration: M2 scores 0.6138 official, M22 0.6495; a corrected M30 near 0.65 would be a solid literature-level result.



---

## 3. What "not about model count" actually changes

The old document's Tier 1/2/3 framing implicitly measured progress by how many of ~28 models got
run. That's retired. The actual questions now are:

1. **Is Chunk D real yet?** Nothing else matters more. One real, well-evaluated cross-task
   consistency result beats ten more synthetic or low-value model runs.
2. **Does at least one Chunk E novelty item exist?** Dr. Khan's guidance is explicit that a plain
   classification/model-zoo pipeline isn't enough on its own in 2026 — check `Novelty Search.md`
   before adding any new model, novelty-driven or not.
3. **Is everything reported real?** Run the audit before claiming anything is done. A model that
   passes `Asif's/audit/audit_project.py` with no CRITICAL findings is worth more than three that
   don't, regardless of how sophisticated they look.
4. **Chunks F and G are explicitly optional.** Pursue them only if D and E are solid and there's
   time left — not because a checklist says so.

---

## 4. Notes on reading this file alongside the others

- This file is the **execution reference** — what to actually run, in what order, with what dependency.
- `Novelty Search.md` is the **priority reference** — as of 2026-08, read it *before* this file when deciding what to build next.
- `Project_Work_Plan.md` is the **chunk-tracking reference** — current status per chunk, in one place, no role/ownership framing.
- `Project_Proposal_v2.md` is the **narrative/novelty-origin reference** — why this project is publishable, related-work positioning, the original §8.5 protocol this file operationalized.
- `Asif's/audit/PROJECT_AUDIT.md` is the **ground truth** for what's real vs. synthetic — regenerate it (`python3 "Asif's/audit/audit_project.py"`) before trusting any status claim in this file, including the ones above; they were accurate as of 2026-08-05 but the audit is the thing that doesn't go stale.
