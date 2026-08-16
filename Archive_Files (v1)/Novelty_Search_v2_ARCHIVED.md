# OWMTL Novelty Discovery Report
## "Cluster-Aware Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis"

**Prepared for:** Group 5 / BB — Backbone Engineering  
**Target venue:** Biomedical Signal Processing and Control (Q1, IF ~4.9)  
**Analysis date:** July 2026  
**Scope:** Phases 1–10 per research prompt; 50+ candidate ideas; 20 combined directions; 10 final recommendations

> **Status: ARCHIVED 2026-08-05. Superseded by the consolidated root-level `Novelty Search.md`.** This file's
> literature search and 50+ candidate ideas remain valid — v3 builds on top of them rather than
> replacing them. v3 adds: reconciliation against Dr. Khan's supervisor-suggested technique list, an
> honest read on which of his suggestions fit this project's audio-only data, and re-prioritization
> given the project's current real state (M2/M3/M12/M29 completed; 9 downstream models found running
> on synthetic data via `Asif's/audit/`). Read v3 first; come back here for the underlying literature.

---

## Phase 1 — Deep Understanding of the Proposal

### Core Hypothesis
A dual-head MTL model (sound-event head + disease-diagnosis head) sharing a single audio encoder can detect diseases **unseen at training time** by measuring the **cross-task disagreement** between the two heads' predictions — because a genuine unknown disease will produce an anomalous mismatch between the predicted sound-event profile and the predicted disease-class profile.

### Current Novelty Stack
| Component | Novelty Level | Status |
|---|---|---|
| MTL (sound-event + disease) on ICBHI | Low — crowded | Baseline only |
| Cross-task disagreement as OSR signal | Medium — mechanism exists elsewhere (Huo et al. 2025) | Domain-specific combination is new |
| Staged OWL protocol (Stage 0 → 1 → 2) | Medium | Novel in respiratory domain |
| Coarse pooled-unknown evaluation design | Medium — statistically motivated | Genuinely novel reframing |
| Cross-dataset OOD stress test (ICBHI → Coswara/SPRSound) | Medium | Established practice, solid execution |
| CQKD compression + OWL regularization | High — claims dual contribution (edge + forgetting mitigation) | Interesting but lightly supported |

### Strongest Contribution
The **statistically defensible evaluation redesign** (coarse pooled unknown + large-N OOD) is more novel and rigorous than the raw mechanism — most papers in this space have weak experimental designs. The proposal's methodological self-awareness about small-N problems is a genuine strength that reviewers will notice.

### Weakest Contribution
The **CQKD-OWL link** (claim that cluster quantization reduces catastrophic forgetting across OWL stages) is asserted but not theoretically motivated. This is the paper's most fragile novelty claim and will draw the hardest Reviewer #2 attacks.

### Key Assumptions
1. Cross-task disagreement is a reliable unknown-detection signal on ICBHI's disease classes (unverified until Phase 0 pilot).
2. COPD/Healthy/URTI form a sufficiently large known-class pool (n=104 patients) to train a stable disease head.
3. Coswara and SPRSound constitute genuine "unknown disease" population — not just domain-shifted versions of known COPD/URTI.
4. CQKD's cluster assignment mechanism imposes structure that happens to resist OWL-stage drift.

### Limitations
- No formal uncertainty quantification on the unknown-detection threshold.
- Catastrophic forgetting is measured but not formally bounded.
- The disagreement score has no coverage guarantee (unlike conformal methods).
- No patient metadata (demographics, symptoms) used, despite ICBHI containing some.
- Binary "known/unknown" OWL evaluation may miss fine-grained novelty characterization.

---

## Phase 2 — Reviewer #2 Attack Mode

Here is what the harshest plausible review would say:

### Attack 1: "The cross-task disagreement is a trivial proxy"
> "The 'novelty' here reduces to thresholding the distance between two softmax vectors. This is a soft version of what OpenMax already does, and Huo et al. (2025) demonstrated the same principle in a non-medical domain. The MTL wrapper does not change the fundamental approach. The paper does not show that this signal is *better* than simpler baselines (max-softmax score, Mahalanobis distance, or even entropy of a single head). Without that comparison, the cross-task disagreement claim is unfounded."

**Mitigation needed:** ablation against energy-based OOD score, Mahalanobis distance from training embeddings, and post-hoc entropy — not just OpenMax.

### Attack 2: "The CQKD–forgetting link has no theoretical grounding"
> "The authors claim that cluster-quantized distillation regularizes forgetting across OWL stages. This is stated but not motivated. CQKD was designed for inference-time compression, not continual learning regularization. Without a formal argument (e.g., showing that cluster assignments act as an EWC-style penalty on the embedding space) or an ablation showing forgetting is actually reduced by CQKD relative to standard KD, this claim is unsupported speculation."

**Mitigation needed:** A formal or empirical argument connecting quantization granularity to representation stability.

### Attack 3: "The OOD datasets do not guarantee unknown diseases"
> "Coswara contains COVID-19 recordings; SPRSound contains paediatric conditions. Some of these may overlap with URTI (which includes COVID-like presentations) or COPD (which shares wheeze patterns with paediatric bronchiolitis). The paper needs to demonstrate that the OOD datasets are genuinely disease-disjoint from the known classes, not just population-disjoint. Without this, the 'unknown disease detection' claim is confounded by domain shift."

**Mitigation needed:** Careful characterization of what diseases appear in Coswara/SPRSound vs ICBHI known classes.

### Attack 4: "The staged OWL protocol is not formally defined"
> "What does 'Stage 2: incrementally incorporate a subset of flagged unknowns' mean operationally? How many patients? What criteria determine clinician confirmation? Without a concrete protocol, the continual learning claim cannot be replicated. The paper should define a specific algorithm: how many patients per increment, what model update procedure (fine-tune, EWC, replay?), and what constitutes 'forgetting' in this context."

**Mitigation needed:** Algorithmic pseudocode for the incremental update step.

### Attack 5: "Statistical power is still insufficient for the disease head"
> "The known-class pool of n=104 patients (COPD ~76, Healthy ~26, URTI ~14) means URTI has 14 patients. Even with LOPO, the variance in URTI performance will be enormous. A model that performs well on COPD and Healthy can still 'pass' evaluation by collapsing URTI. The paper needs to report per-class results explicitly and discuss how the model handles URTI's minority status."

**Mitigation needed:** URTI-specific ablation, or explicit acknowledgment and per-class tables.

### Attack 6: "The open-world claim lacks a formal definition of 'novelty detection performance'"
> "AUROC for unknown vs. known is reported, but AUROC is threshold-free. For a real clinical deployment, a specific operating point must be selected. The paper needs to justify the chosen threshold and discuss precision-recall at that operating point, not just AUROC."

**Mitigation needed:** Precision-recall curves at clinical operating points; consider conformal coverage as a formal guarantee.

### Attack 7: "Why not just use a foundation model?"
> "OPERA (Zhang et al., NeurIPS 2024) pretrains respiratory audio encoders on 136K samples. RespLLM (Zhang et al., ICML 2025) combines audio and text. If these public checkpoints exist, why is a from-scratch 2D CNN backbone the primary model? The paper needs to either (a) show it outperforms OPERA-based fine-tuning, or (b) explicitly scope itself as an architecture-agnostic framework compatible with any encoder."

**Mitigation needed:** Either a comparison against OPERA-fine-tuned backbone, or explicit framing that the framework is encoder-agnostic.

---

## Phase 3 — Literature Search (2024–2026)

### Key papers found that are directly relevant to extending this proposal:

**Conformal Prediction for OOD/Medical AI:**
- "Exploring the Link Between OOD Detection and Conformal Prediction" (ICLR 2025, OpenReview) — formalizes using OOD scores as non-conformity scores, enabling coverage-guaranteed prediction sets.
- "Robust Conformal Prediction for Infrequent Classes" (OpenReview 2024) — directly addresses the small-class problem in clinical AI; coverage guarantees even for rare disease classes.
- "Improving Trustworthiness of AI Disease Severity Rating with Ordinal Conformal Prediction Sets" (Springer 2025) — applied CP to medical classification.

**Test-Time Adaptation for Medical/Respiratory Audio:**
- "Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification" (arXiv 2604.12647, 2026) — introduces difficulty-aware compute allocation at inference for respiratory audio embeddings; directly applicable to the OOD stress tests.
- "DHAuDS: A Dynamic and Heterogeneous Audio Benchmark for Test-Time Adaptation" (arXiv 2024) — benchmarks TTA for audio specifically.

**Respiratory Foundation Models:**
- **OPERA** (Zhang et al., NeurIPS 2024 Datasets & Benchmarks) — first open-source respiratory acoustic foundation model, pretrained on ~136K samples, benchmarked on 19 downstream tasks. **Critical gap: none of OPERA's 19 tasks involve open-world or OSR evaluation.**
- **RespLLM** (Zhang et al., ICML 2025) — multimodal LLM unifying audio + clinical text; outperforms baselines by 4.6% on trained tasks, 7.9% on unseen datasets.
- **RespiraMFM** (arXiv 2606.09966, 2026) — contrastive audio-language alignment for respiratory disease identification; introduces cross-modal alignment beyond simple concatenation.
- **StethoLM** (arXiv 2603.00355, 2026) — audio language model for cardiopulmonary analysis; supports zero-shot classification via report generation.
- **Adaptive Test-Time Scaling** (arXiv 2604.12647, 2026) — applies test-time compute scaling to respiratory audio zero-shot classification using TRIAGE (difficulty-gated escalation pipeline).

**Evidential Deep Learning:**
- "D-EDL: Differential Evidential Deep Learning for Robust Medical OOD Detection" (*Medical Image Analysis*, 2025) — addresses the over-penalty problem in standard EDL; introduces a "Ruling Out Module" inspired by clinical differential diagnosis. **This is strikingly analogous to the cross-task disagreement mechanism in this proposal.**
- "Continual Evidential Deep Learning for OOD Detection" (arXiv 2309.02995) — combines EDL with continual learning, directly relevant to the OWL staging.

**Cross-Domain Adaptation for Respiratory Audio:**
- Kim et al., "Stethoscope-Guided Supervised Contrastive Learning for Cross-Domain Adaptation on Respiratory Sound Classification" (ICASSP 2024) — treats different stethoscope types as separate domains; relevant to Coswara/SPRSound device shift.
- "Mitigating Stethoscope-Induced Shortcuts in Respiratory Sound Classification under Federated Domain Generalization with Causality-Inspired Interventions" (arXiv 2605.29862, 2026) — causal intervention to remove device-induced shortcuts; federated + causal in one paper.

**Open-World Continual Learning:**
- Kim et al., "Open-World Continual Learning: Unifying Novelty Detection and Continual Learning" (*Artificial Intelligence*, 2024) — formal framework; exactly the gap the proposal targets.
- "Open Issues in Open World Learning" (*AI Magazine*, 2025, Cruz et al.) — survey that distinguishes OOD detection from OWL characterization and adaptation; confirms no respiratory audio OWL paper has done the full cycle.

**Multimodal/Graph for Medical:**
- "Graph Attention Network-Based Multimodal Approach for Lung Diseases Classification" (*Scientific Reports*, 2026) — integrates CXR + clinical text with GAT; lung-disease-specific.

**State Space Models for Audio:**
- Audio Mamba (Erol et al., *IEEE Signal Processing Letters*, 2024) — bidirectional SSM for audio classification; linear complexity vs. AST's quadratic. Zero application to respiratory open-world problems.
- RawBMamba (arXiv 2406.06086, 2024) — SSM for audio deepfake detection (an open-set problem!); transferable framing.

**Federated Learning for Respiratory:**
- "An Enhanced Privacy-Preserving Federated Few-Shot Learning Framework for Respiratory Disease Diagnosis" (arXiv 2507.08050, 2026) — federated + few-shot; directly applicable to multi-site deployment.
- FedMLAC (arXiv 2506.10207, 2026) — mutual learning for heterogeneous federated audio classification.

---

## Phase 4 — Cross-Domain Novelty Mining

Ideas proven in other fields but **never applied to respiratory open-world diagnosis**:

### From Autonomous Driving
**Evidential scene uncertainty** — autonomous driving uses EDL's Dirichlet confidence to reject out-of-distribution road scenes in real time. This has never been applied to respiratory cycle-level classification. The parallel: a crackle detected at an unusual phase of the cycle is like an unexpected road object — the model should express Dirichlet-based epistemic uncertainty rather than a confident wrong class.

### From Industrial Anomaly Detection
**Memory bank + nearest-neighbor anomaly scoring (PatchCore-style)** — stores training-distribution embeddings, scores test inputs by their distance to the nearest training exemplar. Applied to respiratory: build a "cycle-level memory bank" of normal and known-disease embedding patches; at inference, flag cycles that exceed a distance threshold as potential unknowns. This requires no architectural change — it's a post-hoc wrapper on the encoder.

### From Protein Learning / Biology
**Causal disentanglement** — in drug discovery, causal models separate "disease mechanism" factors from "confound" factors (patient age, recording device). Applied here: disentangle the shared encoder into a "disease-causal" subspace and a "device/recording-confound" subspace. The cross-task disagreement signal would then be computed only on the causal subspace, making OOD detection robust to device shift — which is the exact confound between ICBHI (4 device types) and Coswara/SPRSound.

### From Continual Learning in NLP
**Prototype replay with cluster-preserved loss** — instead of replaying raw data (privacy concern), replay learnable class prototypes that maintain cluster structure. For Stage 2 OWL updates (incorporating confirmed unknowns), prototype replay would prevent COPD/Healthy/URTI forgetting without storing patient audio — directly addressing both privacy and catastrophic forgetting.

### From Robotics
**Active learning for novelty characterization** — in robotics, an agent selects which unknown observations to request human labels for, maximizing information gain. In the clinical context: the model could query the clinician for confirmation only on cycles where cross-task disagreement exceeds a threshold AND the Mahalanobis distance from any known class exceeds another threshold (conjunctive active learning). This directly addresses the "clinician confirmation" step in Stage 2 of the OWL protocol.

### From Remote Sensing
**Domain-invariant representation via causal intervention** — remote sensing uses style transfer and causal interventions to make models invariant to sensor type (Sentinel-2 vs. Landsat). Applied: use adversarial causal intervention to make the shared encoder's representation invariant to recording device — so that ICBHI's device confound does not corrupt the disease-diagnosis head. This would strengthen the Coswara/SPRSound OOD results.

### From Finance (Anomaly Detection in Time Series)
**Temporal energy-based models** — energy-based models assign low energy to in-distribution sequences and high energy to anomalies. For respiratory cycle sequences: the energy of a patient's respiratory cycle sequence (across multiple cycles in a recording) could serve as an unknown-disease detector more robust than single-cycle classification.

### From Speech Recognition
**Test-time entropy minimization (TENT)** — TTA method that adapts batch-norm parameters at test time by minimizing prediction entropy. Applied to the Coswara/SPRSound shift: adapting batch-norm statistics on unlabeled Coswara recordings before running the disease head would narrow the domain gap without any labeled target data — a practical addition to the OOD stress test that measures "how much TTA helps."

### From LLMs
**Conformal calibration with class-conditional coverage** — LLMs use conformal prediction to give calibrated confidence sets. Applied here: for each known disease class, compute a class-conditional conformal threshold. A test patient is flagged as unknown if they fall outside *all* known-class conformal sets simultaneously. This gives a **distribution-free guarantee** on false-negative rate for unknown detection — something no existing respiratory OSR paper provides.

### From Wildlife Monitoring (already partially relevant via Huo et al.)
**Nearest-class mean in hyperbolic space** — NCM in Euclidean space has been extended to hyperbolic space for hierarchical class structures (disease taxonomies are hierarchical: COPD → obstructive → chronic). A hyperbolic NCM could naturally encode the hierarchy "unknown is far from all known in hyperbolic distance" more expressively than Euclidean NCM.

---

## Phase 5 — First-Principles Analysis (2026 Perspective)

If redesigning this project from scratch in 2026, knowing what we know:

**Q: What is the fundamental problem?**
A: At deployment time, a respiratory screening tool encounters patients with diseases outside its training vocabulary. The model needs to (1) not make a wrong confident prediction, (2) signal uncertainty in a clinically meaningful way, (3) possibly characterize what the unknown looks like, and (4) eventually learn the new class.

**Q: What is the best signal for "unknown"?**
A: Multiple orthogonal signals are more reliable than one. The proposal uses one signal (cross-task disagreement). A 2026 design would use:
- Signal 1: Cross-task disagreement (existing)
- Signal 2: Conformal non-conformity score (formal coverage guarantee)
- Signal 3: Nearest-neighbor distance in embedding space (memory-bank style)
- Signal 4: Evidential uncertainty (Dirichlet total evidence — low evidence = unknown)
These four signals can be combined via a simple **score fusion rule** (e.g., max, product, or learnable weight), and the combination can be evaluated with a calibration-error metric.

**Q: What is the fundamental bottleneck?**
A: The disease head is trained on n=14 URTI patients. No matter how sophisticated the OWL mechanism, a disease head that barely learned URTI cannot reliably produce a "disagreement signal" for URTI-adjacent unknowns. The solution: **self-supervised pretraining of the shared encoder on unlabeled ICBHI cycles** (of which there are 6,898), before the supervised MTL fine-tuning step. This gives the encoder a better acoustic representation of the respiratory domain before it must learn to separate 3 disease classes from 104 patients.

**Q: What is missing from the evaluation?**
A: A **clinical utility metric**. AUROC is an ML metric. A clinician cares about: "If I set the model to flag X% of recordings as 'unknown,' what fraction of truly unknown diseases does it catch, and what is my false alarm rate on known diseases?" This is a **precision-recall-at-operating-point** analysis, not an AUROC analysis. This framing directly addresses the "what would this look like in practice" gap.

**Q: What is the weakest link in the reproducibility chain?**
A: The unknown-detection threshold calibration. Different calibration set sizes will give different thresholds. The paper should use **cross-conformal prediction** (a leave-one-out variant) to calibrate the threshold, so that the calibration is sample-size-robust and the coverage guarantee holds even for small validation sets.

---

## Phase 6 — Research Gap Discovery

### Methodology gaps
- No formal uncertainty quantification on the rejection threshold.
- No multi-signal fusion: one rejection signal (disagreement) vs. ensemble of signals.
- No self-supervised encoder pretraining before supervised MTL.

### Training strategy gaps
- No warm-start from OPERA foundation model weights (publicly available since NeurIPS 2024).
- No curriculum learning in OWL stages (easy known → hard known → unknown).

### Loss design gaps
- Cross-task disagreement loss is used for detection but not as a **training regularizer** — it could be added as an auxiliary loss to *increase* disagreement for unknown-like samples synthetically generated via mixup.
- No **hyperspherical uniformity loss** to prevent the embedding space from collapsing known classes near each other (which would make unknown detection harder).

### Representation learning gaps
- No disentanglement of disease-causal vs. device-confound features.
- No contrastive learning between disease classes at training time (would make the embedding space cleaner for distance-based OOD detection).

### Evaluation protocol gaps
- No **clinical operating point analysis** (sensitivity at fixed specificity).
- No **temporal evaluation** (does the model's unknown-detection improve across OWL stages? is there a stage where performance plateaus?).
- No **ablation of the calibration set size** (what happens to conformal coverage if calibration has only 10 patients vs. 30?).

### Clinical workflow gaps
- No **explanation of why the model flagged something unknown** (clinicians need a reason, not just a flag).
- No **active query strategy** for Stage 2 (which flagged unknowns should be sent for expert review first?).

### Uncertainty/calibration gaps
- No Expected Calibration Error (ECE) reported for the known-class disease head.
- No calibration under dataset shift (does ECE change between ICBHI test and Coswara?).

### Edge deployment gaps
- Inference latency on a Raspberry Pi 4 or similar ARM device (the claimed "edge deployability" needs actual edge hardware numbers).

---

## Phase 7 — Candidate Components Assessment

| Component | Gap it fills | Compatible with current pipeline? | Complexity | Priority |
|---|---|---|---|---|
| Conformal Prediction on unknown-detection | Formal coverage guarantee on rejection | Yes — post-hoc calibration | Low | **Highest** |
| Evidential Deep Learning (EDL) on disease head | Epistemic uncertainty for unknowns | Yes — replace softmax with Dirichlet output | Low–Medium | **High** |
| OPERA encoder fine-tuning | Better base representation | Yes — drop-in encoder swap | Low | **High** |
| Nearest-neighbor memory bank (PatchCore-style) | Complementary OOD signal | Yes — post-hoc | Low | **High** |
| Prototype replay for Stage 2 | Anti-forgetting without raw data storage | Yes — replaces or augments CQKD | Medium | **High** |
| Self-supervised pretraining on unlabeled ICBHI cycles | Better encoder before supervised MTL | Yes — training stage addition | Medium | High |
| Stethoscope-invariant domain adaptation | Removes device confound in OOD tests | Yes — contrastive auxiliary loss | Medium | High |
| Causal disentanglement of device vs. disease features | Principled domain invariance | Medium — requires architecture change | High | Medium |
| Test-Time Adaptation (TENT) on Coswara/SPRSound | Narrows domain gap without labels | Yes — inference-only addition | Low | Medium |
| Audio Mamba encoder as backbone alternative | Linear complexity vs. AST quadratic | Yes — ablation candidate | Medium | Medium |
| Patient-similarity graph (GNN) over ICBHI patients | Structural patient context | Partial — needs metadata | High | Low–Medium |
| Hyperbolic NCM for hierarchical disease taxonomy | Better distance metric for hierarchy | Medium | High | Low |
| Federated learning simulation | Privacy-preserving multi-site framing | No — out of scope | Very High | Out of scope |

---

## Phase 8 — 50+ Candidate Novelty Ideas

Below are 50 ideas evaluated individually. Each has a **novelty score (N)**, **publication potential (P)**, and **risk level (R)**.

---

### Cluster A — Uncertainty & Calibration

**Idea 1: Conformal Unknown-Detection with Distribution-Free Coverage Guarantee**
- *Description:* Use the cross-task disagreement score as a non-conformity score in an inductive conformal prediction framework. Calibrate the threshold on held-out known-class patients. Guarantee that flagging rate among known-class test patients is ≤ α (e.g., 5%).
- *Why scientifically interesting:* Converts a heuristic threshold into a **statistically guaranteed operating point** — the first in respiratory OSR.
- *Why different from current proposal:* Current proposal thresholds disagreement empirically; CP gives a formal coverage guarantee independent of the underlying model.
- *Why reviewers care:* Trustworthy AI is a major 2024–2026 editorial focus; a formal guarantee is publishable on its own.
- *Similar work:* CP applied to medical imaging (MICCAI 2023, Springer 2025), but not to respiratory OSR.
- *Implementation complexity:* Very low — requires a calibration set and a threshold computation.
- **N: 8/10 | P: 9/10 | R: Low**

**Idea 2: Evidential Deep Learning (EDL) on Disease Head**
- *Description:* Replace the softmax output of the disease-diagnosis head with a Dirichlet prior (EDL). Low total evidence → high epistemic uncertainty → unknown flag. Combine with cross-task disagreement as a two-signal detector.
- *Why interesting:* Dirichlet uncertainty directly models "I have no evidence for any class" — semantically closer to "unknown disease" than a high-entropy softmax.
- *Why different:* Current proposal uses softmax-based disagreement; EDL provides a principled Bayesian analog.
- *Similar work:* D-EDL (Medical Image Analysis, 2025) for medical OOD — not in audio/respiratory domain.
- **N: 7/10 | P: 8/10 | R: Low**

**Idea 3: Expected Calibration Error Under Dataset Shift**
- *Description:* Measure ECE of the known-class disease head on ICBHI test set AND on Coswara/SPRSound. Report calibration degradation as a function of domain shift severity.
- *Why interesting:* Calibration under shift is a key trustworthiness metric increasingly demanded by reviewers and regulators.
- *Different from current:* Current proposal does not report calibration at all.
- **N: 5/10 | P: 7/10 | R: Very Low** (easy add-on)

**Idea 4: Class-Conditional Conformal Coverage**
- *Description:* Instead of a single conformal threshold, compute class-conditional conformal thresholds: one per known disease class. A test sample is "unknown" if it falls outside ALL class-conditional prediction sets.
- *Why interesting:* Handles class imbalance gracefully — URTI's threshold is calibrated on 14 patients separately from COPD's 76.
- **N: 8/10 | P: 8/10 | R: Low**

**Idea 5: Conformal AUROC Correction**
- *Description:* Use the conformal AUROC correction (ICLR 2025 work) to report statistically valid AUROC bounds given the finite calibration set size — directly answers the small-N reviewer concern.
- **N: 6/10 | P: 7/10 | R: Very Low**

---

### Cluster B — Encoder & Representation

**Idea 6: OPERA Foundation Model as Encoder**
- *Description:* Replace the from-scratch 2D CNN/AST backbone with OPERA (NeurIPS 2024) fine-tuned on ICBHI known classes. Evaluate whether a domain-pretrained encoder makes the cross-task disagreement signal sharper.
- *Why interesting:* OPERA pretrained on 136K respiratory samples — orders of magnitude more acoustic experience than ICBHI alone.
- *Different from current:* Current proposal trains from scratch or uses ImageNet-pretrained AST.
- *Critical:* None of OPERA's 19 benchmark tasks include OWL/OSR — this proposal would be the first to evaluate OPERA under open-world conditions.
- **N: 7/10 | P: 9/10 | R: Low**

**Idea 7: Self-Supervised Masked Cycle Modeling Before MTL**
- *Description:* Before the supervised MTL phase, pre-train the encoder using masked spectrogram modeling (MAE-style) on all 6,898 unlabeled cycles. Then fine-tune with the MTL objective.
- *Why interesting:* Self-supervised pretraining is standard practice in NLP/vision but has been applied to ICBHI only in the OPERA work (which uses a different architecture). Doing this within the OWMTL framework demonstrates that OWL benefits from SSL pretraining.
- **N: 6/10 | P: 7/10 | R: Medium** (training cost)

**Idea 8: Audio Mamba as Backbone (Linear Complexity)**
- *Description:* Replace the 2D CNN or AST backbone with Audio Mamba (Erol et al., IEEE Signal Processing Letters, 2024) — a bidirectional SSM with linear vs. quadratic complexity. Benchmark against CNN and AST in the ablation.
- *Why interesting:* SSMs are the 2024–2026 alternative to transformers for audio; including one as an ablation option future-proofs the paper.
- **N: 5/10 | P: 6/10 | R: Medium**

**Idea 9: Stethoscope-Invariant Contrastive Encoder**
- *Description:* Add an auxiliary contrastive loss (following Kim et al., ICASSP 2024) that pulls same-disease recordings from different stethoscopes together and pushes different-disease recordings apart, regardless of device. This is run during the MTL training phase.
- *Why interesting:* Directly addresses the device-confound that corrupts Coswara/SPRSound OOD evaluation.
- **N: 6/10 | P: 7/10 | R: Medium**

**Idea 10: Hyperspherical Uniformity Regularizer on Disease Embedding**
- *Description:* Add a uniformity loss (Wang & Isola, 2020) to the disease head's embedding space to prevent known-class embeddings from clustering too tightly (which leaves no "room" for unknown diseases to be detected as outliers).
- **N: 6/10 | P: 6/10 | R: Low**

**Idea 11: Cycle-Level Contrastive Pre-Training (SimCLR-style)**
- *Description:* Apply time-shift and pitch-shift augmentations to cycles and use contrastive learning to pre-train the encoder before MTL, making the representation invariant to non-pathological acoustic variation.
- **N: 5/10 | P: 6/10 | R: Low**

---

### Cluster C — OWL Architecture & Training

**Idea 12: Prototype Replay for Anti-Forgetting in Stage 2**
- *Description:* Instead of (or in addition to) CQKD regularization, maintain learnable class prototypes for each known disease class. During Stage 2 updates, replay loss against prototypes ensures Stage-0 performance is preserved without storing raw audio (privacy-safe).
- *Why interesting:* Prototype replay is theoretically grounded in the continual learning literature; it directly replaces the speculative CQKD–forgetting link with an established method.
- *Different from current:* Current proposal uses CQKD as the forgetting-prevention mechanism; this is more principled and independently validated.
- **N: 7/10 | P: 8/10 | R: Low–Medium**

**Idea 13: Disagreement-Driven Synthetic Unknown Augmentation**
- *Description:* During training, generate "pseudo-unknown" training samples by interpolating between known disease class embeddings at ratios that produce maximal cross-task disagreement. Train the model to flag these high-disagreement interpolations as "unknown," creating a richer training signal for the rejection mechanism.
- *Why interesting:* This converts the passive disagreement signal into an active training target — the model explicitly learns what disagreement looks like.
- **N: 8/10 | P: 8/10 | R: Medium**

**Idea 14: Multi-Stage Disagreement Calibration (not just binary)**
- *Description:* Instead of a single unknown/known threshold, calibrate a 3-level system: (1) known confident, (2) known but uncertain, (3) likely unknown. This creates a clinically meaningful triage output — similar to radiological screening where "likely negative," "indeterminate," and "likely positive" are all reported.
- **N: 7/10 | P: 8/10 | R: Low**

**Idea 15: Active Query Strategy for Stage 2 (Information-Gain Based)**
- *Description:* At Stage 2, when flagged unknowns are candidates for clinician confirmation, use an information-gain criterion (e.g., mutual information between model prediction and potential new class label) to prioritize which flagged samples to query. This formalizes the "clinician confirmation" step.
- *Why interesting:* Active learning for OWL is identified as an open challenge in the Cruz et al. (2025) AI Magazine survey; this proposal would be among the first to implement it for clinical audio.
- **N: 8/10 | P: 8/10 | R: Medium**

**Idea 16: Open-World Novelty Characterization (not just detection)**
- *Description:* After flagging an unknown, apply k-means clustering to the unknown embeddings to estimate how many distinct unknown disease patterns have been encountered. Report this as a "novelty discovery" metric alongside detection metrics.
- *Why interesting:* Cruz et al. (2025) note that most OOD/OWL work stops at detection and never characterizes what the unknown is. Novel diseases are not monolithic — pneumonia vs. lung cancer produce different unknown profiles.
- **N: 8/10 | P: 8/10 | R: Medium**

**Idea 17: Gradient-Based Feature Attribution for Unknown Flags (Explainability)**
- *Description:* When the model flags an unknown, use GradCAM or integrated gradients to identify which time-frequency region of the spectrogram drove the cross-task disagreement. Report this as a "why unknown" explanation.
- *Why interesting:* Clinical trust requires explanations, not just flags. No respiratory OSR paper provides explanations.
- **N: 7/10 | P: 8/10 | R: Low**

**Idea 18: Cycle-Level vs. Patient-Level Disagreement**
- *Description:* Cross-task disagreement can be computed at (a) the individual respiratory cycle level, (b) aggregated across all cycles for one patient. Ablate both. Hypothesis: patient-level aggregation is more stable; cycle-level detects abnormal individual cycles.
- **N: 6/10 | P: 7/10 | R: Very Low**

**Idea 19: Disease-Symptom Knowledge Graph as Prior**
- *Description:* Encode the known clinical relationships between respiratory sounds and diseases (e.g., "COPD → bilateral crackles + wheeze") as a knowledge graph. Use this graph to define the cross-task disagreement: disagreement = distance from the expected graph edge between predicted disease and predicted sound event.
- *Why interesting:* Makes the disagreement signal clinically interpretable and grounded in medical ontology.
- **N: 8/10 | P: 8/10 | R: Medium–High**

**Idea 20: Energy-Based Unknown Detection**
- *Description:* Use energy scoring (Liu et al., NeurIPS 2020) on the disease head's logits as an alternative rejection signal: high-energy samples are in-distribution; low-energy are unknown. Ablate against cross-task disagreement.
- **N: 6/10 | P: 7/10 | R: Very Low**

---

### Cluster D — Evaluation & Protocol

**Idea 21: Clinical Operating Point Analysis**
- *Description:* Report sensitivity at fixed specificity (e.g., sensitivity at 95% specificity) for unknown detection, in addition to AUROC. This maps directly to clinical screening standards (where false-positive rates are regulated).
- **N: 5/10 | P: 8/10 | R: Very Low**

**Idea 22: Temporal Performance Degradation Curve Across OWL Stages**
- *Description:* Plot known-class accuracy as a function of OWL stage (Stage 0 → 1 → 2 → ...) to produce a forgetting curve. Compare CQKD-regularized vs. prototype-replay vs. standard fine-tune.
- **N: 6/10 | P: 7/10 | R: Low**

**Idea 23: Cross-Dataset Calibration Transfer**
- *Description:* Calibrate conformal thresholds on ICBHI; evaluate whether coverage holds on Coswara/SPRSound without recalibration. Report coverage gap as a "calibration portability" metric.
- **N: 7/10 | P: 8/10 | R: Low**

**Idea 24: Novelty Discovery Benchmark (How Many Unknown Types?)**
- *Description:* Evaluate how many unknown disease clusters the model can identify after Stage 1, given that 19 "unknown" patients actually span multiple conditions (Pneumonia, Bronchiolitis, Bronchiectasis, Asthma, etc.). NMI and ARI metrics for cluster quality.
- **N: 7/10 | P: 7/10 | R: Medium**

**Idea 25: Augmentation Effect on Unknown Detection (not just known-class accuracy)**
- *Description:* The proposal already ablates augmentation on known-class accuracy (§8.5.4). Extend this to unknown detection: does augmenting training data change the cross-task disagreement signal for unknowns? This is a non-obvious experiment with a publishable answer either way.
- **N: 6/10 | P: 7/10 | R: Very Low**

---

### Cluster E — Cross-Domain Transfer Ideas

**Idea 26: Memory Bank OOD Scoring (PatchCore-style)**
- *Description:* At training completion, cache a coreset of cycle-level embeddings from each known class. At inference, score a test cycle by its distance to the nearest cached embedding. High distance = unknown. Combine with cross-task disagreement.
- *From:* Industrial anomaly detection (PatchCore, CVPR 2022).
- **N: 7/10 | P: 7/10 | R: Low**

**Idea 27: Test-Time Adaptation via Entropy Minimization (TENT)**
- *Description:* At inference on Coswara/SPRSound, adapt batch-norm parameters by minimizing prediction entropy over unlabeled test batches. Evaluate how much TTA closes the domain gap.
- *From:* Domain adaptation literature (Wang et al., ICLR 2021).
- **N: 6/10 | P: 7/10 | R: Low**

**Idea 28: Causal Feature Disentanglement (Device vs. Disease)**
- *Description:* Decompose the encoder's feature map into a disease-causal component (invariant across devices) and a device-style component (device-specific). OWL detection runs only on the disease-causal component. Implements the "causal intervention" approach seen in federated respiratory domain generalization work (arXiv 2605.29862, 2026).
- *From:* Causal representation learning; remote sensing domain adaptation.
- **N: 9/10 | P: 9/10 | R: High** (implementation complexity)

**Idea 29: Score Fusion of Multiple OOD Signals**
- *Description:* Fuse cross-task disagreement + energy score + Mahalanobis distance from training embeddings via a learnable weighted combination. Train the fusion weights on the held-out calibration set.
- *From:* Ensemble OOD detection literature.
- **N: 7/10 | P: 7/10 | R: Low**

**Idea 30: Neuro-Symbolic Constraint on Disease-Sound Co-occurrence**
- *Description:* Encode clinical co-occurrence constraints as differentiable logical rules (e.g., "if disease = COPD then Prob(wheeze) + Prob(crackle) > 0.7") and add a constraint satisfaction loss. Violation of constraints increases the unknown score.
- *From:* Neuro-symbolic AI literature.
- **N: 9/10 | P: 8/10 | R: High**

**Idea 31: LLM-Assisted Unknown Disease Characterization**
- *Description:* When an unknown is detected, feed the predicted sound-event profile to an LLM (GPT-4 or a medical LLM) and ask it to generate a differential diagnosis of respiratory diseases consistent with that acoustic profile. Present to clinician.
- *From:* Agentic AI for healthcare; RespLLM (ICML 2025).
- **N: 8/10 | P: 7/10 | R: Medium** (scope risk)

**Idea 32: Demographic/Symptom Metadata as OWL Signal**
- *Description:* ICBHI contains patient age and gender metadata. RespLLM-style models also use symptom text. Integrate age/gender as a soft prior: an elderly patient flagged as "URTI" by the disease head but showing COPD-like sounds is a higher-priority unknown than a young patient with the same disagreement.
- **N: 7/10 | P: 7/10 | R: Medium**

**Idea 33: Hyperbolic Embedding for Hierarchical Disease Taxonomy**
- *Description:* Map disease embeddings into Poincaré disk space, where the hierarchy (respiratory disease → obstructive → COPD) is naturally encoded by depth. Unknown diseases would be distant from all known classes in hyperbolic distance.
- *From:* Hyperbolic representation learning (Nickel & Kiela, NeurIPS 2017); applied to medical ontologies.
- **N: 8/10 | P: 7/10 | R: High**

**Idea 34: Multi-Cycle Temporal Anomaly Score**
- *Description:* Instead of classifying individual cycles, model the sequence of cycle-level OOD scores across an entire recording as a time series. A recording with increasing OOD scores over time (progressive disease?) is more suspicious than one with random fluctuations.
- **N: 7/10 | P: 7/10 | R: Medium**

**Idea 35: Federated Simulation with Device-Partitioned ICBHI**
- *Description:* Treat ICBHI's 4 device types as 4 "sites" in a simulated federated setting. Train a federated OWMTL model and evaluate whether it generalizes to Coswara as an unseen site.
- **N: 7/10 | P: 7/10 | R: High** (out of scope for student timeline)

---

### Cluster F — Miscellaneous / Long-Tail Ideas

**Idea 36:** Dirichlet Process Mixture Model on disease embeddings to infer the number of latent disease clusters. **N: 7 | P: 6 | R: High**

**Idea 37:** Wasserstein distance between known-class embedding distribution and test-batch distribution as OWL signal. **N: 6 | P: 6 | R: Medium**

**Idea 38:** Mixup-OOD: generate pseudo-unknown samples by interpolating between known disease classes at class boundaries; train a detector on these. **N: 7 | P: 7 | R: Low**

**Idea 39:** Sound-event head as a "disease prior" — use the sound-event prediction to condition the disease head's prior (Bayesian updating). **N: 7 | P: 7 | R: Medium**

**Idea 40:** Gradient norm as uncertainty proxy: high gradient norms at inference time indicate OOD. **N: 5 | P: 5 | R: Low**

**Idea 41:** Zero-shot respiratory disease detection using CLAP + clinical text prompts as a baseline for the OWL task. **N: 6 | P: 7 | R: Low**

**Idea 42:** Conformal risk control (Bates et al., 2022) applied to the false-negative rate of unknown detection — formally bound the rate at which genuinely unknown patients are missed. **N: 8 | P: 8 | R: Low**

**Idea 43:** Feature squeezing: detect OOD by comparing predictions on original vs. smoothed spectrogram; high disagreement between original and smoothed predictions = OOD. **N: 6 | P: 6 | R: Low**

**Idea 44:** Temperature scaling calibration of the disease head + AUROC under calibration. **N: 5 | P: 7 | R: Very Low**

**Idea 45:** Cross-task disagreement as a **patient-level triage score** rather than a classification threshold — output a percentile rank of "how unusual is this patient relative to the training cohort." **N: 7 | P: 7 | R: Low**

**Idea 46:** Few-shot new-class integration at Stage 2: given only 3–5 confirmed unknown patients, few-shot update the disease head to classify the new disease. Report how many shots are needed before the new class is reliably distinguished from "unknown." **N: 8 | P: 8 | R: Medium**

**Idea 47:** Open-set contrastive learning: add an "anti-self-supervised" contrastive objective that maximizes distance between known-class embeddings and "virtual unknown" embeddings generated by encoder perturbation. **N: 7 | P: 7 | R: Medium**

**Idea 48:** Spectral normalization on the disease head (SNGP-style) to produce distance-aware predictions without full Bayesian inference. **N: 7 | P: 7 | R: Medium**

**Idea 49:** Conformal novelty detection: use a conformal anomaly detector (no class labels needed) as a pre-filter before the MTL model — if the cycle is conformal-anomalous, it goes to the human immediately. **N: 7 | P: 7 | R: Low**

**Idea 50:** Model ensemble uncertainty: train 3 MTL models with different random seeds and use prediction variance as an additional OOD signal. **N: 5 | P: 6 | R: Low**

**Idea 51:** Inductive conformal prediction applied to the OWL *stage transition* decision: formally guarantee that fewer than α% of known patients get falsely escalated to Stage 2. **N: 8 | P: 8 | R: Low**

**Idea 52:** Phonocardiogram-inspired segmentation: use cycle segmentation quality as an additional OWL signal — poorly segmentable cycles (unusual respiratory patterns) are flagged as potentially unknown. **N: 6 | P: 6 | R: Medium**

---

## Phase 9 — 20 Combined Novelty Directions

Combinations that **reinforce each other scientifically**:

---

**Direction 1 (★★★ Highest): Conformal Open-World MTL**
Cross-task disagreement ⊕ Conformal Prediction ⊕ Class-conditional coverage ⊕ Calibration under shift

*Why stronger together:* The disagreement gives the non-conformity score; CP converts it into a guaranteed coverage operating point; class-conditional coverage handles the URTI minority class; calibration under shift tests whether the guarantee transfers to Coswara. Each component answers a distinct reviewer objection. The paper's narrative becomes: "We don't just detect unknowns — we *guarantee* our detection rate."

*Integration:* Requires zero architectural change. Post-training calibration step (3–5 lines of code). New §5.4 subsection. New ablation: coverage with ICBHI calibration vs. coverage with Coswara recalibration.

---

**Direction 2 (★★★): Evidential MTL + Disagreement Fusion**
EDL on disease head ⊕ Cross-task disagreement ⊕ Score fusion ⊕ Conformal threshold

*Why stronger together:* EDL's Dirichlet uncertainty and cross-task disagreement are orthogonal signals — EDL captures "I don't have evidence for any class," disagreement captures "my two heads contradict." Their conjunction is more robust than either alone, and their union covers more unknown failure modes. Score fusion (learned on calibration set) is the bridge.

*Integration:* Replace the disease head's final softmax with a Dirichlet output layer. Add EDL loss term (reverse-KL). Fuse evidential uncertainty and disagreement score with a 2-parameter learnable combiner. Evaluate on calibration set.

---

**Direction 3 (★★★): OPERA-Initialized OWMTL with OWL Benchmark**
OPERA encoder ⊕ OWMTL pipeline ⊕ 19-task OWL evaluation

*Why stronger together:* OPERA's 19 benchmark tasks do not include any OWL/OSR evaluation. This proposal would be the **first paper to evaluate a respiratory acoustic foundation model in an open-world setting**. That framing alone is publishable at NeurIPS or ICML (upgrade from BSPC target), or it makes a BSPC paper significantly more competitive. The OPERA encoder provides a stronger acoustic representation; the OWMTL framework adds OWL capability; the result is a foundation-model-level contribution for respiratory AI.

*Integration:* Load OPERA's pretrained encoder weights → fine-tune on ICBHI with MTL objective → evaluate OWL. Compare against from-scratch training. Requires OPERA checkpoint (publicly available at NeurIPS 2024).

---

**Direction 4 (★★★): Multi-Signal OOD Ensemble + Conformal Guarantee**
Cross-task disagreement ⊕ Mahalanobis distance ⊕ Energy score ⊕ Conformal AUROC correction

*Why stronger together:* Three orthogonal OOD signals cover different failure modes; conformal AUROC correction gives statistically valid performance bounds on the small calibration set; the combination directly addresses Attack 1 from Phase 2 (disagreement being a trivial proxy).

*Integration:* Each signal is a post-hoc score on the trained model. Fusion weights learned on calibration set. New ablation table: each signal alone vs. fusion. Conformal AUROC reported alongside standard AUROC.

---

**Direction 5 (★★): Prototype-Replay + Prototype-Based Unknown Characterization**
Prototype replay (Stage 2 anti-forgetting) ⊕ Novelty clustering of unknowns ⊕ Few-shot class addition

*Why stronger together:* Prototypes serve triple duty: (1) replay known classes to prevent forgetting, (2) cluster unknown embeddings to characterize novelty, (3) become the seed for few-shot new-class integration when a clinician confirms a new disease. This is a unified "prototype lifecycle" narrative.

*Integration:* Maintain a prototype set per known class. At Stage 2, unknown embeddings are clustered; high-confidence clusters become new prototype candidates. Few-shot update adds new prototypes.

---

**Direction 6 (★★): Causal Device Disentanglement + OWL**
Causal intervention to isolate device features ⊕ Disease-causal OWL detection ⊕ Cross-dataset evaluation

*Why stronger together:* Device confound is the primary confounder in Coswara/SPRSound OOD tests. Causal disentanglement removes it, making the OOD stress test a purer test of disease-OOD generalization. The paper can then claim: "We detect unknown diseases, not unknown devices."

*Risk:* This requires an architectural addition (dual-path encoder with adversarial disentanglement). High complexity for a student timeline.

---

**Direction 7 (★★): Disagreement-Aware Explainability + Clinical Trust**
Cross-task disagreement ⊕ GradCAM on disagreement ⊕ Clinical operating point ⊕ Multi-level triage

*Why stronger together:* When the model flags an unknown, GradCAM shows which spectrogram region caused the disagreement; clinical operating point specifies the threshold used; multi-level triage (confident known / uncertain known / likely unknown) maps to clinical workflow. The contribution is a **clinically deployable screening protocol**, not just a model.

---

**Direction 8 (★★): Self-Supervised Pretraining + MTL + Conformal OWL**
MAE on ICBHI cycles (self-supervised) ⊕ MTL fine-tuning ⊕ Conformal threshold

*Why stronger together:* Self-supervised pretraining gives the encoder more acoustic experience before the small-N supervised stage; this should sharpen the disagreement signal; conformal calibration then guarantees the detection threshold. Three-stage training pipeline with a clean story arc: pretrain → fine-tune → calibrate.

---

**Direction 9 (★★): Knowledge Graph Prior + Neuro-Symbolic Disagreement**
Clinical disease-sound co-occurrence graph ⊕ Graph-constrained cross-task score ⊕ Interpretable rejection

*Why stronger together:* The disagreement score becomes interpretable: "the model predicted COPD but detected only normal sounds, violating the COPD → {crackle, wheeze} constraint in the clinical knowledge graph." Reviewers in biomedical AI increasingly reward clinically grounded explanations.

---

**Direction 10 (★★): Test-Time Adaptation + Conformal Recalibration on OOD**
TENT adaptation on Coswara ⊕ Conformal recalibration after adaptation ⊕ Coverage comparison

*Why stronger together:* TTA narrows the domain gap; conformal recalibration measures whether coverage was maintained through the adaptation. Together they answer: "Not only does our model adapt to a new environment, but its coverage guarantee transfers."

---

**Direction 11 (★): Temporal Multi-Cycle OWL with LOPO**
Cycle-level disagreement sequence ⊕ Patient-level aggregation ⊕ LOPO evaluation

*Why stronger together:* Aggregating cycle-level signals at the patient level (e.g., max, mean, or a 1D CNN over the cycle sequence) addresses the multiple-cycle-per-patient structure of ICBHI that single-cycle models ignore. LOPO evaluation is natural at the patient level.

---

**Direction 12 (★): Conformal + Prototype + Few-Shot Integration**
Conformal unknown detection ⊕ Prototype clustering of unknowns ⊕ Few-shot update with coverage

*Why stronger together:* Detection (conformal) → characterization (prototype clustering) → integration (few-shot update) → re-detection with updated conformal threshold. This is the complete OWL lifecycle.

---

**Direction 13 (★): Foundation Model Fine-Tuning with Stethoscope Invariance**
OPERA encoder ⊕ Stethoscope-invariant contrastive fine-tuning ⊕ Cross-dataset OWL

*Why stronger together:* OPERA provides acoustic generalization; stethoscope-invariant contrastive tuning removes device bias; together they maximize cross-dataset OWL robustness.

---

**Direction 14 (★): Mixture-of-Disagreement-Experts**
Multiple cross-task disagreement functions (per-class, global, per-head-layer) ⊕ Mixture-of-experts gating

*Why stronger together:* Different diseases may produce disagreement in different ways (URTI produces subtle disagreement vs. COPD produces strong disagreement). A MoE router that selects the most appropriate disagreement function per patient could outperform a single disagreement score.

---

**Direction 15 (★): SNGP Distance-Awareness + Conformal Calibration**
Spectral-normalized GP on disease head ⊕ Distance-aware predictions ⊕ Conformal calibration

*Why stronger together:* SNGP produces GP-style uncertainty without full Bayesian inference; conformal calibration wraps the uncertainty into a coverage guarantee. Together they are more computationally tractable than full Bayesian inference while providing stronger guarantees than point estimates.

---

**Direction 16 (★): Conformal Risk Control on Unknown False-Negative Rate**
Conformal risk control ⊕ Unknown detection ⊕ Clinical safety specification

*Why stronger together:* The key clinical risk in OWL is false negatives (unknowns missed, classified as a known disease). Conformal risk control (Bates et al., 2022) can formally bound this false-negative rate at a user-specified level. This is a stronger clinical safety claim than any AUROC number.

---

**Direction 17 (★): Audio Mamba Backbone + OWMTL**
Audio Mamba (linear-complexity SSM) ⊕ MTL heads ⊕ OWL disagreement ⊕ Edge deployment

*Why stronger together:* Audio Mamba's linear complexity is directly relevant to edge deployment; it replaces the CNN/AST in the ablation; its lower memory footprint complements the CQKD compression step. The combined contribution: "linear-complexity backbone + quantized distillation = maximal edge efficiency."

---

**Direction 18 (★): Zero-Shot OWL via CLAP Baseline Comparison**
CLAP zero-shot classification on unknown classes ⊕ OWMTL detection ⊕ Comparison

*Why stronger together:* CLAP can classify respiratory sounds zero-shot by matching audio to text prompts like "cough with crackle consistent with pneumonia." Evaluating CLAP zero-shot performance on the unknown held-out diseases provides a strong baseline that the OWMTL must beat — and if it does, the paper shows that the structured OWL approach outperforms zero-shot audio-language models.

---

**Direction 19 (★): Synthetic Unknown Generation via Diffusion/VAE + OWL Training**
VAE-generated synthetic unknowns ⊕ MTL training with synthetic unknown class ⊕ OWL evaluation on real unknowns

*Why stronger together:* Synthetic unknowns (generated by a VAE or spectrogram diffusion model) can populate a virtual "unknown" training class, making the rejection mechanism more robust at inference. The model is trained to reject its own synthetic unknowns, then evaluated on real held-out diseases.

---

**Direction 20 (★): Patient-Similarity Graph + MTL + OWL**
Patient-similarity GNN on clinical metadata ⊕ Graph-aggregated disease head ⊕ Unknown detection via graph anomaly

*Why stronger together:* A patient-similarity graph (connecting patients with similar age, symptom, and cycle-count profiles) provides structural context that the audio-only model cannot access. Unknown patients would be graph-anomalous (no similar known-class neighbors). Combined with the cross-task disagreement, this provides a multi-modal OWL signal.

---

## Phase 10 — Final Top-10 Recommendations

These 10 directions are selected for:
✅ Compatible with current pipeline (no full rebuild)
✅ Feasible in a student research timeline
✅ Significantly increases scientific contribution
✅ Not already common in respiratory sound analysis
✅ Q1-journal quality

---

### Recommendation 1 (Must-Do): Conformal Open-World Unknown Detection
**Direction 1 expanded**

**What it is:** After training the MTL model, use the cross-task disagreement score as a non-conformity score in an inductive conformal prediction framework. Calibrate on held-out known-class patients. This gives a formal, distribution-free guarantee on the false-positive rate of unknown detection.

**Why the supervisor's novelty concern is answered:** This transforms an empirically-tuned threshold into a mathematically-proven guarantee — a fundamentally different scientific contribution. No respiratory audio paper has done this. The CP framework is actively published in IEEE TMI, Medical Image Analysis, and BSPC target journals.

**How to integrate:**
1. After Stage 1 OWL training, hold out a calibration split of known-class patients (not used in training).
2. Compute cross-task disagreement scores on calibration patients.
3. Set conformal threshold at the (1-α)(1 + 1/n_cal) quantile of calibration scores.
4. At test time, flag patients whose disagreement exceeds the threshold as "unknown."
5. Coverage guarantee: among known-class test patients, fewer than α will be falsely flagged.
6. Extend to class-conditional thresholds for COPD, Healthy, URTI separately.
7. Report: conformal AUROC (with finite-sample correction), standard AUROC, and calibration portability from ICBHI to Coswara.

**Novelty claim addition:** "This is the first respiratory disease screening framework to provide distribution-free unknown-detection guarantees via conformal prediction."

**Implementation effort:** 1–2 weeks (pure post-processing, no training change).

---

### Recommendation 2 (Must-Do): Evidential Disease Head (Replace Softmax with Dirichlet)
**Direction 2 expanded**

**What it is:** Replace the softmax output of the disease-diagnosis head with an EDL Dirichlet output. The model outputs α parameters (evidence per class) instead of probabilities. Total evidence = Σαᵢ; uncertainty = K/(K + Σαᵢ). Low total evidence = likely unknown.

**Why the supervisor's novelty concern is answered:** EDL produces a principled Bayesian characterization of "unknown" — the model says "I have no evidence for any class" rather than "I'm confused between classes." This is architecturally novel in respiratory audio diagnosis, and paired with cross-task disagreement as a second signal, directly answers Attack 1.

**How to integrate:**
1. Replace the disease head's final softmax layer with a ReLU + softplus activation outputting αᵢ ≥ 1 for each class.
2. Replace cross-entropy loss with the EDL reverse-KL loss (Sensoy et al., NeurIPS 2018).
3. Combine evidential uncertainty and cross-task disagreement via score fusion (weighted sum, weights learned on calibration set).
4. Ablation: disagreement-only vs. evidential-only vs. fused vs. conformal-fused.

**Novelty claim addition:** "We are the first to apply evidential deep learning to open-world multi-task respiratory diagnosis, combining epistemic uncertainty with cross-task acoustic-clinical disagreement."

**Implementation effort:** 2–3 weeks (loss function change + calibration).

---

### Recommendation 3 (Must-Do): OPERA Encoder as Backbone Comparison
**Direction 3 expanded**

**What it is:** Load the publicly available OPERA-CT or OPERA-GT checkpoint (NeurIPS 2024, Zhang et al.) and use it as the shared encoder backbone. Fine-tune end-to-end with the MTL objective. Compare to the from-scratch CNN/AST baseline in the ablation.

**Why this is high-impact:** OPERA is the first open-source respiratory acoustic foundation model. This proposal would produce the **first OWL/OSR evaluation of a respiratory foundation model** — a contribution that the OPERA authors themselves have not done and that is directly publishable as a novel finding.

**How to integrate:**
1. Download OPERA weights from official repository (NeurIPS 2024 supplementary).
2. Freeze or partially unfreeze encoder layers during MTL fine-tuning (ablate both).
3. Compare cross-task disagreement signal quality (AUROC) with OPERA vs. from-scratch encoder.
4. Report: does a foundation model encoder produce a sharper disagreement signal for unknown detection?

**Novelty claim addition:** "We evaluate OPERA's respiratory acoustic representations in an open-world disease classification setting for the first time, demonstrating [X]% improvement in unknown detection AUROC over task-trained encoders."

**Implementation effort:** 1–2 weeks (checkpoint loading + ablation).

---

### Recommendation 4 (High Priority): Multi-Signal OOD Fusion with Ablation
**Direction 4 expanded**

**What it is:** Compute three independent OOD scores on the trained model: (1) cross-task disagreement, (2) Mahalanobis distance from training-class centroids in embedding space, (3) energy score from disease head logits. Fuse with a learnable 3-weight softmax combiner.

**Why the supervisor's novelty concern is answered:** Directly addresses Attack 1 ("the disagreement is trivial"). By ablating all three signals, the paper proves empirically that cross-task disagreement adds something beyond standard OOD baselines — or discovers that a simpler signal works just as well (which is also a publishable finding).

**How to integrate:**
1. Post-training: compute training-class centroid embeddings for Mahalanobis.
2. Post-training: compute energy scores from disease logits.
3. Post-training: compute disagreement scores.
4. Learn fusion weights on calibration set (3 parameters, not overfitting).
5. Ablation table: each signal alone, pairwise fusions, all three.

**Implementation effort:** 1–2 weeks.

---

### Recommendation 5 (High Priority): Prototype Replay for Stage 2 Anti-Forgetting
**Direction 5 expanded**

**What it is:** Instead of (or in addition to) CQKD as the forgetting prevention mechanism, maintain learnable class prototypes for COPD/Healthy/URTI. During Stage 2 updates, replay the prototype-based reconstruction loss for known classes, preventing catastrophic forgetting without storing raw audio.

**Why the supervisor's novelty concern is answered:** Prototype replay is theoretically grounded and independently validated. It directly replaces the weakest novelty claim (CQKD–forgetting link) with an established method, while freeing the CQKD contribution to stand on its own as a compression mechanism.

**How to integrate:**
1. After Stage 0 training, extract per-class prototype embeddings (centroid of training embeddings per class).
2. During Stage 2 fine-tuning on new known patients, add a prototype replay loss: pull current embeddings toward stored prototypes for all known classes.
3. Compare forgetting curves: CQKD alone vs. prototype replay alone vs. both combined.

**Implementation effort:** 2–3 weeks.

---

### Recommendation 6 (High Priority): Clinical Operating Point + Multi-Level Triage Report
**Direction 7 partial**

**What it is:** Report unknown detection results at a clinically meaningful operating point: sensitivity at 95% specificity (instead of only AUROC). Additionally, define a 3-level triage: (1) known-confident, (2) known-uncertain (conformal set contains >1 class), (3) likely unknown (outside all conformal sets). Map these to clinical actions.

**Why the supervisor's novelty concern is answered:** AUROC is an ML metric. "Sensitivity at 95% specificity" is a clinical standard (analogous to cancer screening protocols). The 3-level triage maps directly to how a clinician would use the tool. This is a non-trivial framing contribution.

**How to integrate:**
1. Compute precision-recall curve for unknown detection.
2. Report sensitivity at 95% specificity in main results table.
3. Define 3-level triage using conformal prediction sets (size 1 = confident; size >1 = uncertain; rejected = unknown).
4. Report triage distribution on ICBHI test set and on Coswara/SPRSound.

**Implementation effort:** <1 week (reporting change).

---

### Recommendation 7 (High Priority): GradCAM Explanation of Unknown Flags
**Direction 7 partial**

**What it is:** When the model flags a patient as "unknown," use GradCAM or integrated gradients on the disease head's penultimate layer to produce a spectrogram heatmap showing *which time-frequency region* drove the high-disagreement output. Display one example per unknown disease category.

**Why this matters:** No respiratory OSR paper currently explains its rejection decisions. Clinical trust requires a "why," not just a flag. This is cheap to implement and high-impact in a biomedical journal.

**How to integrate:**
1. After flagging an unknown patient, select the cycle with highest disagreement score.
2. Apply GradCAM with respect to the disease head output.
3. Overlay the heatmap on the mel-spectrogram.
4. Show example heatmaps for each true-unknown category (Pneumonia, Bronchiolitis, Bronchiectasis) in a Figure.

**Implementation effort:** 2–3 days.

---

### Recommendation 8 (Medium Priority): Self-Supervised Pretraining on Unlabeled ICBHI Cycles
**Direction 8 expanded**

**What it is:** Before the supervised MTL training phase, pretrain the shared encoder using masked spectrogram modeling (MAE-style) on all 6,898 ICBHI cycles (ignoring labels). This gives the encoder far more acoustic experience before it must discriminate 3 disease classes from 104 patients.

**Why this is novel:** Self-supervised respiratory audio pretraining has only been done at scale (OPERA), not within-dataset before a small-N supervised fine-tune. The research question — "does within-dataset SSL pretraining improve OWL detection even when foundation models are available?" — is novel and has a clean yes/no answer.

**How to integrate:**
1. Implement a simple masked autoencoder (mask 75% of mel-spectrogram patches, reconstruct).
2. Pretrain on all 6,898 ICBHI cycles (patient-independent split maintained).
3. MTL fine-tune with pre-trained encoder weights (frozen vs. full fine-tune ablation).
4. Compare OWL disagreement signal quality vs. ImageNet-initialized vs. OPERA-initialized.

**Implementation effort:** 3–4 weeks (significant but feasible; GPU training on Kaggle).

---

### Recommendation 9 (Medium Priority): Conformal Risk Control on False-Negative Rate
**Direction 16 expanded**

**What it is:** Apply conformal risk control (Bates et al., 2022) — a generalization of CP — to formally bound the rate at which truly unknown patients are *missed* (classified as known). This is the clinically critical error: falsely confident diagnosis of a truly unknown disease is dangerous.

**Why the supervisor's novelty concern is answered:** This provides a formal safety guarantee that complements the coverage guarantee of Recommendation 1. Together: "We guarantee we won't falsely flag known patients (Rec. 1) AND we formally bound how often we miss unknown patients (Rec. 9)." No respiratory AI paper has done either; having both is a major trust contribution.

**How to integrate:**
1. Use the conformal risk control framework (Bates et al., 2022) with risk function = indicator(unknown patient missed).
2. Set target risk level δ (e.g., "we miss at most 10% of unknown patients").
3. Calibrate using the held-out unknown patients (the pooled 19 held-out ICBHI patients).
4. Report: what disagreement threshold achieves δ = 0.10 false-negative rate with coverage guarantee?

**Implementation effort:** 1 week.

---

### Recommendation 10 (Medium Priority): Stethoscope-Invariant Contrastive Auxiliary Loss
**Direction 13 partial**

**What it is:** During MTL training, add a contrastive auxiliary loss that pulls together embeddings of the same-disease class across different ICBHI recording devices, and pushes apart embeddings from different disease classes (following Kim et al., ICASSP 2024 — but now integrated into the OWMTL training objective).

**Why this matters for OWL:** The primary confounder in the Coswara/SPRSound OOD tests is recording device/modality, not disease. A device-invariant encoder makes the cross-task disagreement signal a purer disease-OOD signal. The experiment directly answers: "How much of the cross-dataset OWL gap is due to device shift vs. disease shift?"

**How to integrate:**
1. Add supervised contrastive loss across ICBHI's 4 device types (metadata available in ICBHI).
2. Total loss = α × MTL loss + β × stethoscope-invariant contrastive loss.
3. Ablate: OWMTL without contrastive loss vs. with it on ICBHI test, then on Coswara/SPRSound.

**Implementation effort:** 2–3 weeks.

---

## Summary: What to Tell Your Supervisor

The proposal currently has **one central novelty claim** (cross-task disagreement as an OWL detector) and **one secondary claim** (CQKD as a forgetting regularizer). The secondary claim is weak. A stronger paper has:

1. **A formal guarantee on the detection threshold** (Rec. 1 + Rec. 9 — conformal prediction) → converts heuristic to theorem
2. **A principled uncertainty model on the disease head** (Rec. 2 — EDL) → makes the detector architecturally novel, not just post-hoc
3. **A foundation model comparison** (Rec. 3 — OPERA) → positions the work at the 2026 frontier
4. **Multi-signal ablation** (Rec. 4) → proves the cross-task disagreement is non-trivial
5. **A grounded forgetting-prevention mechanism** (Rec. 5 — prototype replay) → replaces the speculative CQKD–forgetting link
6. **Clinical deployment framing** (Rec. 6 + Rec. 7) → makes reviewers in biomedical journals happy

**Minimum viable upgrade** (2–3 weeks additional work, highest ROI):
- Recommendation 1: Conformal calibration of the disagreement threshold ← single most impactful addition
- Recommendation 6: Clinical operating point reporting ← reframing only
- Recommendation 4: Multi-signal ablation table ← 3 extra post-hoc scores
- Recommendation 7: GradCAM explanation figure ← 2–3 days

**If timeline allows** (4–6 additional weeks):
- Recommendation 2: EDL disease head
- Recommendation 3: OPERA encoder comparison
- Recommendation 5: Prototype replay

These six additions transform the paper from "a cross-task disagreement detector with an empirical threshold" into:

> **"A formally calibrated, evidentially grounded, foundation-model-informed open-world multi-task learning framework for trustworthy respiratory disease screening — the first respiratory AI system to provide distribution-free unknown-detection guarantees."**

That is a significantly novel claim.

---

*End of report. All literature claims are grounded in 2024–2026 papers; implementation recommendations are feasible within a 24-week capstone timeline. Literature recency verified as of July 2026.*
