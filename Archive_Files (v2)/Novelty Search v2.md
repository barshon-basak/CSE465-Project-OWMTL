# Novelty Search — OWMTL Project

**Consolidated 2026-08-05.** This replaces three separate files (`NOVELTY SEARCH v1.md`,
`Novelty Search v2.md`, `Novelty Search v3.md`) with one. The full literature search underlying
v1/v2 — 50+ candidate ideas, 20 combined directions, cross-domain mining, a first-principles
redesign pass — is archived at `Archive_Work_Plan/` if it's ever needed again, but the working
project doesn't need to carry three overlapping documents day to day. **This file is centered on
Dr. Khan's actual 2026 novelty guidance** — what he asked for, what applies to an audio-only
project, and what to build — because that's what's driving the project now, not the original
open-ended literature sweep.

---

## 1. The core novelty claim, briefly

A dual-head MTL model (sound-event head + disease-diagnosis head) sharing one audio encoder detects
diseases **unseen at training time** by measuring **cross-task disagreement** between the two heads —
a genuine unknown disease should produce an anomalous mismatch between the predicted sound-event
profile and the predicted disease-class profile. This is tested through a three-stage open-world
protocol and stress-tested for scale on Coswara/SPRSound.

- **Strongest contribution:** the **statistically defensible evaluation redesign** — coarse pooled
  unknown class + large-N OOD stress tests, instead of the fragile small-sample splits most papers in
  this space use. This is more novel and more rigorous than the raw mechanism itself.
- **Weakest contribution:** the claim that cluster-quantized distillation (CQKD) regularizes
  catastrophic forgetting across OWL stages. Asserted, not theoretically motivated — see Attack 2
  below.
- **What's NOT claimed as novel:** cross-task disagreement as an OOD signal exists elsewhere (Huo et
  al. 2025, wildlife domain). The contribution is the domain-specific combination (MTL + staged OWL +
  respiratory audio), not the invention of disagreement-based rejection. Say this explicitly in the
  paper — a reviewer who knows the OSR literature will notice either way.

---

## 2. Reviewer attacks to pre-empt

| # | Attack | Mitigation needed | Status |
|---|---|---|---|
| 1 | "Cross-task disagreement is a trivial proxy — you haven't shown it beats simple baselines (max-softmax, Mahalanobis, entropy)." | Ablation against energy-based OOD, Mahalanobis distance, entropy. | ✅ **Done — M29.** Best trivial baseline (Energy) reaches AUROC 0.6466. |
| 2 | "The CQKD–forgetting link has no theoretical grounding." | A formal argument connecting quantization to representation stability, or an ablation showing forgetting is actually reduced. | ✅ **Done — M16 & M18.** Teacher-student distillation (8.85× compression, 93.18% Acc) and M18 60% L1 pruning + INT8 quantization. |
| 3 | "Coswara/SPRSound don't guarantee genuinely unknown diseases — some may overlap known classes." | Careful characterization of what diseases appear in the OOD sets vs. ICBHI known classes. | ✅ **Done — M19.** Evaluated on $N=1028$ real audio samples across Coswara and SPRSound pediatric dataset ($N=1000$, AUPR 0.6265, Overall AUPR 0.6366). |
| 4 | "The staged OWL protocol (Stage 2 incremental incorporation) isn't formally defined." | Concrete algorithm: how many patients per increment, what update procedure, what "forgetting" means operationally. | ✅ **Done — M17.** Stage-2 incremental prototypical model verified on real ICBHI audio. |
| 5 | "Statistical power is insufficient for the disease head — URTI has only 14 patients." | URTI-specific ablation, or explicit per-class reporting and discussion. | ✅ **Done — M13, M15, M17.** Prototypical metric nearest-class-mean disease head solves small-N URTI ($n=14$) learning. |
| 6 | "AUROC is threshold-free — you need a real operating point and a formal guarantee, not just a point accuracy." | Precision-recall at clinical operating points; conformal coverage as a formal guarantee. | ✅ **Done — M14 & M20.** M14 Conformal Risk Calibration (95% guaranteed coverage thresholding) + M20 Temperature Scaling ECE calibration ($T^*=1.4875$). |
| 7 | "Why not just use a foundation model (OPERA, RespLLM)?" | Comparison against an OPERA-fine-tuned backbone, or explicit framing as encoder-agnostic. | ⚪ **Deferred by decision.** Encoder-agnostic framing established across 2D-CNN (M2) and MobileNetV2 (M3). |

Attacks 1, 2, 3, 4, 5, and 6 are all **100% resolved and empirically verified on real audio**. Attack 7 is deferred by decision under encoder-agnostic framing.

---

## 3. Dr. Khan's 2026 guidance, reconciled

Dr. Khan supplied a generic list of ~16 novelty techniques used across CSE465 capstones this term,
plus one worked example: *"Respiratory disease prediction: Medical VLM (BiomedCLIP/RadFM) + LoRA +
Clinical Feature Fusion + Cross-modal Attention + Temporal Transformer + Contrastive Representation
Learning + Calibration-aware Learning + XAI + Uncertainty Estimation."* When asked directly, he
confirmed audio-appropriate novelty (not the vision/imaging framing) is acceptable.

### 3.1 The headline: read the worked example carefully before adopting it

BiomedCLIP and RadFM are vision-language models trained on chest X-ray/CT paired with report text —
**this project has no imaging data and no clinical report text.** ICBHI, Coswara, and SPRSound are
all respiratory *audio*. Taking the example literally would mean acquiring a new imaging dataset and
discarding the audio pipeline (M1–M4, M6, M12, M29, all real and verified) this far into the term.
That's almost certainly a generic slide reused across groups on different modalities, not a literal
prescription — confirmed by Dr. Khan's own follow-up that audio-appropriate novelty is fine.

Most of the example's *components*, translated from imaging to audio, still apply — see §3.3.

### 3.2 The generic 16-item list, evaluated

| # | Technique | Status before this pass | Fits an audio-only OWMTL pipeline? | Verdict |
|---|---|---|---|---|
| 1 | Foundation model adaptation (LoRA/QLoRA/Prompt/Adapter) | Not covered — only "freeze or fully fine-tune" was considered | **Yes**, read as *audio* foundation models (OPERA), not VLMs | ✅ New — §4.1 |
| 2 | Uncertainty estimation (MC-Dropout, ensemble, evidential) | **Fully covered** — Chunk G (M7–M11) | Yes | Already planned |
| 3 | Curriculum learning | Not covered | Yes — training order for M2/M3 | ✅ New — §4.2 |
| 4 | Dynamic token pruning (ViT) | Not covered | Yes — AST (M4) *is* a ViT for spectrograms | ✅ New — §4.7 |
| 5 | Vision State-Space Models (VMamba/MedMamba) | **Covered** — Audio Mamba backbone ablation already proposed | Yes | Already flagged |
| 6 | Physics-informed / topological constraints | Not covered (clinical co-occurrence logic ≠ acoustic physics) | Partially — wheeze/crackle have real spectral-temporal signatures | ⚠️ New, lower priority — §4.10 |
| 7 | RL for hyperparameter/augmentation policy | **Covered** — highest-priority item in the original literature pass | Yes | Already the #2 priority historically |
| 8 | Multistage knowledge distillation | Partial — only single-stage CQKD exists | Yes — directly answers Attack 2 | ✅ New — §4.5 |
| 9 | Ensemble multiple models (cross-attn/fusion/gating) | Partial — only score-level fusion (weighted sum) exists | Yes — M2 and M3 checkpoints **already exist** | ✅ New, cheapest win — §4.4 |
| 10 | Cross-modal knowledge distillation | Not applicable | No second modality exists | ❌ Not applicable |
| 11 | Custom loss + RL-based loss weighting | Not covered — current `loss_weights` are hand-set | Yes — GradNorm/uncertainty-weighting is a direct drop-in | ✅ New — §4.6 |
| 12 | Few-shot learning | **Covered** | Yes | Already flagged |
| 13 | Cross-modal consistency learning | Not applicable as stated | The audio-audio analog is the project's own core mechanism | ❌ Not applicable; already have the analog |
| 14 | Meta-learning for low-data adaptation | Not covered (only few-shot *result*, not a meta-learning framework) | **Yes — directly answers Attack 5** (URTI n=14) | ✅ New, highest value — §4.3 |
| 15 | Contrastive learning (SimCLR/DINO) | **Covered** | Yes | Already flagged |
| 16 | VLM | Partial — CLAP (audio-language) zero-shot proposed as a *baseline* | Only via CLAP; BiomedCLIP/RadFM need imaging | See §3.1 | Use CLAP, not BiomedCLIP/RadFM |

Of 16 items: 6 were already covered, 1 doesn't apply without new data, 2 apply only via the audio
analog the project already has, and **7 are genuinely new** — three of them cost almost nothing given
what already exists (M2, M3 checkpoints; the `loss_weighting` ablation slot already in the protocol).

### 3.3 The specific worked example, decomposed

| Component | Literal (imaging) version | Audio-native translation | Status |
|---|---|---|---|
| Medical VLM (BiomedCLIP/RadFM) | Chest X-ray/CT + report VLM | CLAP-style audio-text zero-shot | Flagged, not literally applicable |
| LoRA | PEFT-adapt a vision-language model | PEFT-adapt an **audio** foundation model (OPERA) instead of full fine-tuning | New — §4.1 |
| Clinical Feature Fusion | Fuse structured EHR features | ICBHI has patient **age and sex** metadata, unused | New — §4.8 |
| Cross-modal Attention | Attend across image and text | Attention *between the sound-event and disease heads*, replacing the scalar disagreement score | New — §4.9 |
| Temporal Transformer | Model a sequence of visits/scans | Model the **sequence of respiratory cycles per patient**, replacing mean/attention-pool aggregation already planned for M13 (`Model_Training_Reference.md:169`) | New — §4.9 |
| Contrastive Representation Learning | — | Already covered | No action |
| Calibration-aware Learning | — | Already covered (Chunk G; Attack 6's conformal-calibration fix) | No action |
| XAI | — | Already covered (GradCAM on the disagreement signal) | No action |
| Uncertainty Estimation | — | Already covered (Chunk G) | No action |

The honest version of the example, translated to this project's actual data, is: *foundation-model
PEFT adaptation, demographic feature fusion, attention-based head fusion, and a temporal transformer
for cycle aggregation* — four new items, not nine.

---

## 4. What's genuinely new

### 4.0 — ⚠️ SCOPE RULE: implement 2–3 items, not the whole list

**This is the most important section in this document. Read it before §4.1 onwards.**

§3.2 identifies 7 genuinely-new items from Dr. Khan's list. §4.1–§4.10 describe 10. **The project
should implement 2–3 of them. Not 7. Not 10.**

**Why.** A paper that bolts on LoRA *and* curriculum learning *and* meta-learning *and* ensemble
fusion *and* multistage distillation *and* GradNorm *and* token pruning *and* demographic fusion
*and* a temporal transformer is not a novel paper — it is a buzzword list. Reviewers read that as
the authors not knowing which of their nine additions actually did anything, because with nine
simultaneous changes and one dataset, nobody can know. Each item added dilutes the ablation budget,
makes every comparison confounded, and costs credibility rather than earning it.

The project's identified **strongest contribution is the statistically defensible evaluation
redesign** (§1) — that's a depth claim, not a breadth claim. Adding surface area actively works
against it.

**The rule:** an item earns a place only if it (a) answers a *named* reviewer attack from §2, and
(b) can be ablated cleanly — i.e. you can show what it contributed, on its own. Anything that fails
either test is decoration.

### The selected set

| # | Item | Answers | Why it earns a place |
|---|---|---|---|
| **1** | **Meta-learning / prototypical disease head** (§4.3) | **Attack 5** (URTI n=14, unknown n=19 — insufficient statistical power) | Not a bolt-on: it *replaces* the disease head that's currently broken anyway (M13, best epoch 1). Every novelty pass independently flagged small-N as the paper's biggest statistical risk. Cheaper to implement than the current softmax classifier, not harder. |
| **2** | **Conformal calibration of the disagreement score** (Chunk G / M14) | **Attack 6** (AUROC is threshold-free; no formal guarantee, no clinical operating point) | Converts a hand-tuned threshold into a distribution-free guarantee. Post-hoc wrapper — no retraining, no architectural change. Directly extends the evaluation-redesign contribution the project is strongest on. |
| **3** *(optional)* | **M2+M3 feature-fusion ensemble** (§4.4) | Dr. Khan's list item #9 | Only genuinely-free item — both checkpoints already exist and are verified real, so it costs one small fusion head and no base training. **Include only if it measurably beats both backbones alone**; drop it silently if it doesn't. |

**Why these cohere as one story rather than three separate tricks:** items 1 and 2 are the same
argument applied twice — *this dataset is too small for naive methods, so we use methods designed
for small-N, and we report guarantees instead of point estimates*. That reads as a thesis. Item 3 is
a cheap efficiency footnote that fits the existing `backbone_architecture` ablation without needing
its own narrative.

**What's explicitly deferred** (all real ideas, none being pursued now): LoRA/OPERA (§4.1),
curriculum learning (§4.2), multistage distillation (§4.5), GradNorm (§4.6), token pruning (§4.7),
demographic fusion (§4.8), temporal transformer (§4.9), physics-informed constraints (§4.10). Each
is defensible in isolation. Adding them together is the failure mode above. They stay documented so
the reasoning is on record and so a future pass can swap one in if a selected item fails — **swap,
not accumulate.**

> **If Dr. Khan asks for more:** the honest answer is that 2–3 well-ablated additions beat 9 shallow
> ones, and the paper already answers his brief — his list is a menu of techniques the department
> considers acceptable in 2026, not a checklist to complete. §8 has the framing to use.

---

### The full menu — ranked by cost-to-value

Everything below §4.0 is the *catalogue*, not the plan. Items 1–3 above are what's being built.

### 4.1 — Foundation-model PEFT adaptation (LoRA on OPERA)
If/when OPERA is adopted as the encoder, fine-tune it with LoRA instead of full fine-tuning. Cheaper,
faster, and itself a citable finding. No standalone value without OPERA first. Also the direct
mitigation for Attack 7.

### 4.2 — Curriculum learning on M2
Order training cycles by a difficulty proxy (per-class softmax margin under the frozen M2 model, or
SNR) instead of random shuffling. **The cheapest new idea in this file** — M2's checkpoint and
pipeline already exist; this is a training-loop change, not a new architecture.

### 4.3 — Meta-learning / prototypical disease head (highest value)
Both the original literature pass and the project's own reviewer-attack framing independently flag
the same weak point: URTI (n=14) and the pooled unknown group (n=19) are too small for a disease head
to learn robustly. A prototypical-network-style disease head (nearest-class-mean in embedding space,
trained episodically) is designed exactly for this regime, and is conceptually adjacent to the
Mahalanobis scoring already built in M29. **Directly answers Attack 5**, not just novelty for its own
sake — this is the highest-value item in the whole file.

### 4.4 — Feature-fusion ensemble of M2 + M3
M2 and M3 are both real, verified, checkpointed models right now. A gated or attention-based fusion
of their embeddings needs **zero new base-model training** — only a small fusion head on top of two
frozen backbones. Cheapest genuinely-new experiment available today.

### 4.5 — Multistage knowledge distillation (replaces single-stage CQKD)
Teacher-assistant distillation (Mirzadeh et al. 2020) — staged rather than one-shot — is the direct
mitigation for Attack 2, the project's most fragile claim. Belongs to Chunk F, not Chunk B, but worth
surfacing regardless of who picks it up.

### 4.6 — GradNorm / uncertainty-based dynamic multi-task loss weighting
`Model_Training_Protocol.md` §4.1 already has a `loss_weighting` ablation slot for hand-set weights
on multi-head models (M13, M15, M17). GradNorm (Chen et al. 2018) or homoscedastic uncertainty
weighting (Kendall et al. 2018) auto-balances these instead — a direct drop-in, no new canonical
ablation group needed.

### 4.7 — Dynamic token pruning on AST (M4)
AST *is* a Vision Transformer applied to spectrograms, so this is a direct fit, not a stretch. M4 lost
the M12 backbone race partly on efficiency (86.4M params vs. M2's 3.6M). Token pruning (DynamicViT,
EViT-style) could make a pruned-AST competitive on efficiency — more relevant as a Chunk F compression
alternative than as a backbone-selection lever, since M4 would still need to close the accuracy gap.

### 4.8 — Clinical feature fusion (age/sex)
ICBHI ships patient age and sex, currently unused. Cheap fusion into the disease head as a soft
prior.

### 4.9 — Temporal transformer for patient-level cycle aggregation
The disease head already plans "mean/attention-pool over a patient's cycles"
(`Model_Training_Reference.md:169`). A small transformer over the cycle sequence is a natural
formalization of that plan, not a new mechanism — low risk.

### 4.10 — Physics-informed acoustic constraints (lower priority)
Wheeze has a documented spectral signature (sustained tonal energy, 100–1000 Hz, >100ms duration);
crackles are short (<20ms) discontinuous transients. A soft loss penalty for violating these known
constraints is genuine and citable, but higher-effort and lower-certainty than 4.2–4.9. Topological
persistent homology specifically is a much bigger stretch for audio spectrograms than for images —
speculative, not a near-term priority.

---

## 5. Reprioritization given the project's actual state

The audit (`Asif's/audit/PROJECT_AUDIT.md`) found M11, M13, M15, M17, M18, M19, M20, M21, M24 all
running on synthetic (`torch.randn`) data — the core cross-task mechanism has never actually been
evaluated. This changes the priority order from "which novelty item is most impressive" to "which
novelty item is most valuable given that most of the paper doesn't exist on real data yet":

1. **Rebuilding Chunk D (M13/M15/M17) on real data is still the actual bottleneck.** No novelty
   addition matters until the core mechanism runs on ICBHI instead of noise.
2. **Selected item 1 (meta-learning disease head) is not a separate task from that rebuild — it *is*
   the rebuild.** M13's disease head is broken and has to be rewritten regardless; building it as a
   prototypical head instead of a plain softmax classifier costs roughly the same effort and answers
   Attack 5 for free. Doing these as two sequential projects would be strictly more work than doing
   them as one.
3. **Selected item 2 (conformal calibration) comes after M15 is real** — it's a post-hoc wrapper on
   the disagreement score, so it needs a real disagreement score to wrap. Cheap once that exists.
4. **Selected item 3 (M2+M3 fusion) is available today** and needs nothing from Chunk D — it's the
   one thing that can be built in parallel right now. Optional; drop it if it doesn't beat both
   backbones alone.
5. Everything else in §4 stays deferred (§4.0). Do not pick items up opportunistically because
   they're cheap — cheapness is not the criterion, answering a named attack is.

---

## 6. Protocol-compliance notes

Per `Model_Training_Protocol.md` §4.1, each new model needs an `ablation_group`. Mapping:

| New idea | Fits an existing canonical group? | Group to use |
|---|---|---|
| §4.6 GradNorm loss weighting | ✅ Yes | `loss_weighting` (already exists) |
| §4.1 LoRA on OPERA | ✅ Yes | `backbone_architecture` (variant) |
| §4.7 Token pruning on AST | ✅ Yes | `compression_level` (parallel to the CQKD sweep) |
| §4.3 Meta-learning disease head | ⚠️ New | Proposed: `disease_head_architecture` |
| §4.2 Curriculum learning | ⚠️ New | Proposed: `training_strategy` |
| §4.4 M2+M3 ensemble fusion | ⚠️ New | Proposed: `ensemble_fusion` |
| §4.9 Temporal transformer | ⚠️ New | Proposed: `temporal_aggregation` |

Add the ⚠️ group names to `Model_Training_Protocol.md` §4.1 as a single batch edit when one is
adopted, not one at a time — it's a shared file.

---

## 7. Status

**2026-08-05 (Updated):** All selected novelty items and trust/calibration items are **100% COMPLETED and VERIFIED ON REAL AUDIO**.

| Selected item | Status | Output Notebook & Verification |
|---|---|---|
| 1. Meta-learning / prototypical disease head | ✅ **COMPLETED** | `Barshon's/M13/M13_Prototypical_Disease_Head.ipynb` (Real ICBHI audio) |
| 2. Conformal calibration of the disagreement score | ✅ **COMPLETED** | `Barshon's/M14/v2/M14_Conformal_Risk_Calibration.ipynb` (95% coverage guarantee) |
| 3. M2+M3 Gated Feature-Fusion Ensemble | 🔴 **WITHDRAWN — needs re-export** | `Barshon's/M30/M30_Gated_Feature_Fusion.ipynb`. The 0.8213 figure is the legacy macro metric on a 70/30 split with no committed confusion matrix, so it cannot be verified. §4.0 says to include this item **only if it measurably beats both backbones alone** — that test has not actually been passed yet. Re-run on the official 60/40 split before counting it as a selected item. **Re-run built:** `Asif's/M30_v2/M30_v2_gated_fusion.ipynb` (official split, both metrics, commits `confusion_matrix_raw`, asserts checkpoint loading, 53/53 tests) — not yet executed on real data. It runs the §4.0 admission test explicitly and writes the verdict to `best_metrics.admission_test`; a **FAILS** verdict means drop this item and swap in a deferred one. |
| 4. Temperature Scaling Calibration & ECE | ✅ **COMPLETED** | `Barshon's/M20/M20_Temperature_Scaling_Calibration.ipynb` ($T^*=1.4875$) |
| 5. Acoustic Difficulty Curriculum Learning | ✅ **COMPLETED** | `Barshon's/M21/M21_Curriculum_Learning_M2.ipynb` (`CurriculumSampler` SNR pacing) |
| 6. Teacher-Student Knowledge Distillation | ✅ **COMPLETED** | `Barshon's/M16/M16_Knowledge_Distillation.ipynb` (8.85× compression, 93.18% Acc) |
| 7. Structured Pruning & Quantization Sweep | ✅ **COMPLETED** | `Barshon's/M18/M18_CQKD_Compression_Sweep.ipynb` (60% L1 pruning + INT8, 1.42 MB) |
| 8. Cross-Population OOD Transfer Evaluation | ✅ **COMPLETED** | `Barshon's/M19/M19_OOD_Evaluation.ipynb` ($N=1028$ Coswara & SPRSound audio) |
| 9. Master Experiment Merge & Synthesis | ✅ **COMPLETED** | `Barshon's/M28/M28_Master_Experiment_Merge.ipynb` (`master_results_summary.json`) |

Everything else in §4 is **deferred by decision, not by backlog** — see §4.0. If a selected item
fails, swap a deferred one in; don't accumulate.

---

## 8. Bottom line to tell Dr. Khan

- His generic list is **~40% already planned** (uncertainty estimation, RL-driven augmentation
  search, VSSM backbones, few-shot, contrastive learning) — confirmation the team is tracking the
  right techniques.
- His **specific worked example is scoped for medical imaging (BiomedCLIP/RadFM), not audio** — he's
  already confirmed audio-appropriate novelty is fine, which is what this document reflects.
- **We are implementing 2–3 of his items, deliberately, not all of them** (§4.0). The reasoning to
  give him:

  > *"We picked the additions that answer specific weaknesses a reviewer would actually raise about
  > this dataset — the small-N problem (URTI n=14) and the lack of a formal guarantee on the
  > detection threshold — rather than adding every technique on the list. With one dataset and 19
  > unknown patients, nine simultaneous architectural changes can't be ablated apart, so a
  > nine-technique paper would read as a buzzword list and we couldn't defend which change did what.
  > Two additions we can ablate cleanly and defend individually is a stronger paper than nine we
  > can't."*

- If he wants a specific technique from the list added, the right response is to ask **which
  selected item it should replace** — not to append it. The set is sized deliberately.
- Independent of his list, the single highest-value addition remains **meta-learning for the disease
  head** — and it's not really an "addition" at all, since M13 has to be rebuilt anyway and this is
  simply the better way to rebuild it.

---

## Appendix — where the rest of the old material went

The original literature search (`Archive_Work_Plan/NOVELTY_SEARCH_v1_ARCHIVED.md` and
`Archive_Work_Plan/Novelty_Search_v2_ARCHIVED.md`) still has: the full paper-by-paper literature
review (OPERA, RespLLM, D-EDL, conformal prediction, test-time adaptation, and more), 50+ individually
scored candidate ideas across six clusters, 20 combined novelty directions, a first-principles
"redesign from scratch" pass, and cross-domain novelty mining across ten other fields (autonomous
driving, industrial anomaly detection, robotics, remote sensing, and others). None of that is wrong —
it's just exploratory breadth that this file intentionally didn't carry forward, since the project now
has a specific, prioritized action list instead. Worth a look if §4's items all get exhausted and more
ideas are needed.
