# New Research Directions — Generate → Prove → Disprove → Select

**Prepared:** 2026-08-11
**Context:** The original OWMTL novelty (cross-task disagreement) failed; Direction 1 (device/disease disentangled open-world) was rejected after revalidation because 2025–2026 papers closed it. This document runs a fresh, disciplined search for a *new* direction under the same rule: **creative in generation, ruthless in validation.**
**Honesty labels used throughout:** **[Verified gap]** = multiple recent papers confirm the problem is open; **[Plausible]** = no counter-paper found but absence of evidence ≠ evidence of absence; **[Uncertain]** = depends on an experiment we haven't run or a claim I couldn't fully confirm.

---

## SUMMARY VERDICT (up front)

- **7 candidates generated; 5 rejected on evidence; 2 survived; 1 selected.**
- **Selected direction:** **Acoustic Concept-Bottleneck Diagnosis (ACBD)** — reframe the sound-event→disease pipeline as an *interpretable-by-design* concept bottleneck where clinically-defined adventitious sounds (crackle / wheeze / both / normal) are the **only** channel to the diagnosis, then use that low-dimensional clinical concept space for **(a)** clinician-in-the-loop *concept intervention*, **(b)** measuring *concept leakage* (does the model cheat the bottleneck?), and **(c)** *concept-space* unknown-disease detection.
- **Why it wins:** it is the only survivor that is simultaneously (i) not yet done for respiratory auscultation **[Plausible novelty, evidence in §3]**, (ii) built almost entirely from assets you already have (your sound-event head *is* the concept predictor), (iii) a *mechanism/interpretability* contribution rather than an accuracy race, and (iv) a strong fit for your target journal class (BSPC/CBMS reward clinical interpretability). It also gives the open-world goal a genuinely different, defensible mechanism than the two that already failed.
- **Residual risk (stated honestly):** CBM is an established framework and CBM-for-health-audio now exists for *voice* (2026); the paper must be carried by a specific research question (leakage / device-robustness / intervention), not by "we applied CBM to lungs." Details in §5–§6.

---

## Step 1 — Candidate ideas generated (creative phase)

| # | Direction | Research problem | Existing limitation | Proposed core idea | Reuse | Difficulty |
|---|---|---|---|---|---|---|
| C1 | **Acoustic Concept-Bottleneck Diagnosis (ACBD)** | Respiratory DL is a black box; clinicians can't inspect or correct the reasoning | Interpretability today = post-hoc Grad-CAM/attention, which don't let a clinician *intervene* | Predict disease *only* from an interpretable acoustic-concept layer (crackle/wheeze/…); enable concept intervention, measure leakage, detect novelty in concept space | Sound-event head (=concepts), disease head, M29 OOD suite, M14 conformal, device labels | Medium |
| C2 | **FM reliability at the clinical operating point for rare respiratory diseases** | Respiratory FMs (OPERA/HeAR/M2D) report AUROC, not sensitivity/calibration at a usable threshold, on tiny disease sets | Benchmarks hide failure at deployment operating point | Systematic operating-point + calibration + small-N reliability audit of FMs | M20 calibration, M14 conformal, M29 | Medium-High (needs FM compute) |
| C3 | **Test-time adaptation for respiratory device shift** | Model degrades on unseen stethoscopes at deployment | Source data unavailable at test time | Source-free TTA to adapt to new device online | M2/M30, device labels | Medium |
| C4 | **Distill a respiratory FM to an edge stethoscope model** | FMs too big for embedded auscultation | On-device deployment gap | KD + quantization of OPERA/HeAR to a tiny student | M16/M18 compression | Medium |
| C5 | **ICBHI benchmark hygiene / reproducibility** | Inconsistent pipelines make cross-paper comparison meaningless | No standardized processing/split reporting | A reproducible re-benchmark + protocol | All models | Low-Medium |
| C6 | **Learning-to-defer for respiratory screening** | When should the model hand off to a clinician? | Abstention treated as unstructured reject | L2D with human-AI complementarity for lung sounds | M13, M14 | Medium |
| C7 | **Concept leakage / shortcut audit of respiratory MTL** | Does sound→disease MTL actually use the sounds, or cheat via the encoder? | Unquantified for respiratory audio | Information-theoretic leakage measurement | MTL heads | Low-Medium (folds into C1) |

---

## Step 2 & 3 — Prove and DISPROVE each candidate (validation phase)

### C3 — TTA for respiratory device shift → **REJECTED (already done, 2026)**
- **TRIAGE / Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification** (arXiv:2604.12647, 2026): adaptive inference-time computation across 9 tasks / 5 respiratory datasets for robustness in medical audio.
- **DHAuDS** (arXiv:2511.18421): a dedicated TTA benchmark spanning bioacoustic audio.
- **Causal FedDG** (arXiv:2605.29862) + **Stethoscope-guided contrastive** (ICASSP 2024, arXiv:2312.09603) already cover device adaptation.
- **Reviewer kill:** "TTA for medical/respiratory audio under shift is an active, occupied lane; what's the delta over TRIAGE?" No convincing answer. **[Verified — reject.]**

### C4 — Distill a respiratory FM for edge → **REJECTED (crowded + incremental)**
- **HeAR** (Google, 2024), **PulmoVec** (HeAR-based, arXiv:2603.15688, 2026), **BCoughBench** (wearable FM benchmarking, arXiv:2606.25116), plus generic edge-KD everywhere (CVPR EDGE workshops 2025/26).
- **Reviewer kill:** "KD+quantization of an audio model is standard practice; combining known compression with a known FM is incremental." **[Verified — reject.]**

### C5 — ICBHI benchmark hygiene → **REJECTED as primary (low novelty ceiling, wrong author profile)**
- ICBHI already has an official 60/40 patient-independent split; the 2025 *Electronics* review (14(14):2794) already catalogs the reproducibility gaps. A benchmarking/resource paper is legitimate but (a) is a *survey-adjacent* contribution with a low novelty ceiling, and (b) reviewers weight such "authority" papers by the group's standing — risky for a student capstone. **[Verified — reject as primary; keep the rigor as a supporting section.]**

### C6 — Learning-to-defer for respiratory → **REJECTED (crowded + overlaps rejected work)**
- Clinical L2D / selective prediction is a large, active field (arXiv:2508.07617 clinical case study; 2508.06997 conformal expert selection; 2605.08024 dual-head deferral in glaucoma). The respiratory-audio instantiation would be "apply L2D to lungs," and the abstention framing overlaps the already-rejected Direction 1. **[Verified — reject.]**

### C2 — FM operating-point reliability → **SURVIVES but DOWNGRADED (real gap, partially occupied, resource-heavy)**
- **Gap is real and explicitly stated by the field:** BCoughBench (arXiv:2606.25116, 2026) itself says the three FM families "report AU-ROC without sensitivity or calibration metrics, concealing failure at the operating point," and that COPD/asthma/pneumonia have "small test set sizes that may limit statistical reliability." **[Verified gap.]**
- **But partially occupied:** BCoughBench is already doing FM reliability benchmarking (under wearable-sensor conditions), and SpeechDx (arXiv:2606.17339) is a clinical-speech benchmark. A general FM-reliability paper now competes with well-resourced groups and needs FM compute you may not have.
- **Verdict:** viable *second* choice, but the crowding + compute cost make it weaker than C1. Best used as a *component* (operating-point + small-N reliability is exactly the rigor C1 needs anyway).

### C7 — Concept leakage / shortcut audit → **SURVIVES, folds into C1**
- Concept leakage is an active *general* CBM topic (arXiv:2504.09459 information-theoretic leakage; 2504.14094 leakage & interpretability; 2506.04877 ICLR 2026) — **but not studied for respiratory auscultation.** It is a genuine research *question*, not just an application, which is what C1 needs. **[Plausible for respiratory — keep, fold into C1 as a core research question.]**

### C1 — Acoustic Concept-Bottleneck Diagnosis → **SURVIVES (lead candidate)** — full disproof below

I attacked C1 from four angles:

1. **"CBM is old (Koh et al. 2020) — you're just applying it."**
   - True that CBM is established, and CBM-for-audio now exists: **Voice Concept Bottleneck for interpretable health assessment** (arXiv:2607.16967, 2026, depression/dysarthria from *voice*), and **Concept Complement Bottleneck for medical *imaging*** (arXiv:2410.15446, 2024). → **Mitigation:** the contribution is *not* "a CBM exists for lungs." It is the specific respiratory-auscultation instantiation **plus** a research question (leakage / device-robustness / intervention value) that these papers don't address, in a modality (adventitious lung sounds with clinically-codified concepts crackle/wheeze) neither touches. **[Plausible novelty, honestly caveated.]**

2. **"Sound→disease is just MTL, which is done (Suma 2025, Tri-MTL)."**
   - Crucial distinction: MTL predicts sound *and* disease **in parallel** from shared features. A CBM predicts disease **only from the concept predictions** (a true bottleneck), which is what enables *intervention* and makes *leakage* measurable. The failed OWMTL and the MTL papers are architecturally different (parallel heads, no bottleneck). → **This is a defensible architectural/scientific distinction, not cosmetic. [Verified distinction.]**

3. **"Concept-space OOD is already done."**
   - Concept-based OOD exists but is **VLM/LLM-driven, vision, zero-shot** (COOD, CMA 2025, EOE 2024) — none uses *clinically-defined acoustic concepts* for medical-audio novelty detection. → **[Plausible novelty for the respiratory-audio concept-space detector.]**

4. **"Interpretable respiratory models already exist."**
   - Yes — Grad-CAM, attention, and **prototype-based interpretability** (arXiv:2110.03536, 2021). But none is a *concept bottleneck* with *human-interveneable* clinical concepts. Post-hoc saliency ≠ interpretable-by-design with intervention. → **[Verified distinction.]**

**C1 survives** as **[Plausible novelty]** provided it is framed around a research question, not an application. No single paper kills it; the risk is *incrementality perception*, which §6 is designed to defuse.

---

## Step 4 — Publishability scoring (survivors)

| Idea | Novelty | Sci. significance | Literature gap | Feasibility (T4 + your time) | Impl. risk | Q1 potential |
|---|---|---|---|---|---|---|
| **C1 ACBD** (with C7 leakage + C2 rigor folded in) | Medium-High **[Plausible]** | High (interpretability + safety) | Strong (interpretable-by-design + concept-space novelty unclaimed for lungs) | High (reuses your heads) | Low-Medium | **High** |
| C2 FM reliability (standalone) | Medium **[Verified gap, partly occupied]** | High | Real but filling | Medium (needs FM compute) | Medium | Medium |

Everything else scored below the reject line in Step 2/3.

---

## Step 5 — Top surviving directions

### Survivor A (SELECTED) — Acoustic Concept-Bottleneck Diagnosis (ACBD)
- **Why it survived:** no paper does interpretable-by-design *concept-bottleneck* diagnosis for lung-sound auscultation with clinician *intervention*; concept *leakage* is unstudied for respiratory audio; concept-*space* novelty detection is unclaimed for clinical acoustic concepts. The three closest works are in different modalities (voice CBM 2607.16967; medical-imaging CBM 2410.15446) or are post-hoc (prototypes 2110.03536).
- **Evidence for the gap:** §2/§3 citations above.
- **Why meaningful:** clinicians distrust black boxes; a bottleneck that *forces* the diagnosis through named acoustic findings, lets a clinician correct a mis-heard crackle and see the diagnosis update, and quantifies whether the model actually listens — is a safety and trust contribution, not an accuracy trick.
- **What could still invalidate it:** (i) a 2026 preprint doing respiratory CBM appears before submission (re-check near submission); (ii) enforcing the bottleneck could collapse accuracy so far it's clinically moot — but *quantifying that tradeoff is itself the result*; (iii) reviewers demanding a richer concept set than 4 classes (mitigate by adding finer concepts — fine/coarse crackle, rhonchi, and demographic priors you already have).
- **Reuse:** your **sound-event head becomes the concept predictor** (near-zero new training); disease head becomes the concept→label predictor; **M29** OOD baselines get re-run *in concept space*; **M14** conformal wraps the concept-space score; **device labels** test concept-space robustness; **Coswara/SPRSound** as external shift; your McNemar/bootstrap code for stats.
- **New work:** a genuine bottleneck (disease predicted only from concept logits), an intervention API, an information-theoretic leakage measure (per arXiv:2504.09459), and the concept-space novelty detector.

### Survivor B (backup) — FM reliability at the clinical operating point for rare respiratory diseases
- **Why it survived:** the field admits the gap (BCoughBench's own words). **Why only backup:** partially occupied and compute-heavy; better absorbed as the *rigor layer* of Survivor A (report operating-point sensitivity + calibration + small-N reliability for every claim).

---

## Step 6 — SELECTED DIRECTION, specified

### Research problem
Deep respiratory diagnosis models are black boxes; clinicians cannot see *why* a disease was predicted, cannot *correct* the reasoning, and have no principled, interpretable signal when the model faces a disease it was never trained on.

### Research gap (Existing → Limitation → Unresolved → Ours)
- **Existing:** (1) MTL predicts sound events and disease *in parallel* (Suma 2025; Tri-MTL). (2) Interpretability for lung sound is *post-hoc* (Grad-CAM/attention/prototypes, arXiv:2110.03536). (3) CBMs are interpretable-by-design but demonstrated for imaging (2410.15446) and *voice* (2607.16967), not lung-sound auscultation. (4) Concept leakage is studied generically (2504.09459) but never for respiratory audio. (5) Concept-space OOD is vision/VLM only (COOD, CMA).
- **Limitation:** parallel MTL gives no intervention handle; post-hoc saliency can't be corrected by a clinician and can be unfaithful; imaging/voice CBMs don't transfer the *acoustic-adventitious-sound* concept vocabulary; nobody knows whether respiratory models actually use the clinical sounds or cheat; nobody detects novel disease in an interpretable clinical concept space.
- **Unresolved:** Can lung-sound diagnosis be routed *entirely* through clinically-named acoustic concepts without unacceptable accuracy loss, does that make it correctable and honest, and is the clinical concept space a *safer, more device-robust* place to flag unknown disease than the opaque embedding?
- **Ours:** ACBD — an interpretable-by-design acoustic concept bottleneck with intervention, leakage measurement, and concept-space novelty detection.

### Novel contribution (precise — avoids "novel framework")
1. The first **concept-bottleneck** formulation of lung-sound disease diagnosis in which the diagnosis is a function *only* of clinically-defined adventitious-sound concepts, with a working **clinician concept-intervention** mechanism. **[Plausible novelty]**
2. The first **information-theoretic concept-leakage measurement** for respiratory audio — quantifying how much a "sound→disease" model actually relies on the sounds vs. bypasses them. **[Plausible novelty]**
3. A **concept-space unknown-disease detector** and a head-to-head test of the hypothesis that flagging novelty in the low-dimensional clinical concept space is *more device-robust* than embedding-space OOD (your prior M29 suite). **[Uncertain — this is a hypothesis to be tested, and a nuanced/negative result is still publishable.]**

### Proposed method
- **Backbone → concept layer:** reuse your encoder; the sound-event head outputs concept activations for {normal, crackle, wheeze, both} (optionally extended with fine/coarse crackle, rhonchi if labels/annotation allow, plus age/sex as auxiliary concepts you already have).
- **Bottleneck:** disease predicted **only** from concept activations (independent/sequential CBM variants), vs. a "leaky joint" MTL control.
- **Intervention:** at test time, overwrite a concept value (simulating a clinician) and re-predict; measure diagnosis change.
- **Leakage:** information-theoretic estimate (mutual information between residual encoder info and label given concepts), following arXiv:2504.09459.
- **Concept-space novelty:** apply MSP/Energy/Mahalanobis (your M29) **in concept space**; compare against the same detectors in embedding space, stratified by device (leave-one-device-out) and on Coswara/SPRSound.

### Implementation steps
1. Convert sound-event head to a calibrated concept predictor; freeze/finetune encoder.
2. Train three variants: independent-CBM, sequential-CBM, joint (leaky) control.
3. Report accuracy/ICBHI-score tradeoff vs. the opaque baseline (the interpretability *cost*).
4. Build intervention API; run intervention experiments (single- and multi-concept correction).
5. Implement leakage metric; report per-variant leakage.
6. Re-run M29 detectors in concept space; device-stratified + external-set evaluation with bootstrap CIs.

### Experiments
- Diagnostic accuracy: opaque MTL vs. CBM variants (patient-independent, official split).
- Interpretability cost curve (bottleneck width vs. accuracy).
- Intervention: Δaccuracy per corrected concept (human-AI complementarity).
- Leakage: per-variant, with/without bottleneck regularization.
- Novelty: concept-space vs. embedding-space OOD, device-stratified + Coswara/SPRSound, with CIs.

### Ablations
- Bottleneck type {independent, sequential, joint} · concept-set size {4, extended} · with/without demographic concepts · concept-space detector {MSP, Energy, Mahalanobis} · with/without leakage regularization · leave-one-device-out vs. random split.

### Expected contribution & what results would convince a reviewer
- **Convincing = all of:** (i) CBM matches opaque accuracy within a small, quantified margin (interpretability is nearly free) *or* an honest, useful tradeoff curve; (ii) concept intervention measurably improves diagnosis (complementarity shown, not asserted); (iii) leakage is quantified and shown to be controllable; (iv) concept-space novelty detection is competitive with, and more device-robust than, embedding OOD — **or**, if not, a clear characterization of *when* it is (a scientific finding either way). Every number reported with CIs and at a clinical operating point (folding in Survivor B's rigor).

---

## Step 7 — Final decision

**Proceed with Survivor A (ACBD) as the new research direction**, absorbing C7 (leakage) as a core research question and Survivor B (operating-point/calibration rigor) as the evaluation standard.

This is a **[Plausible novelty]**, not a **[Verified novelty]** — the honest status, because CBM is an established framework and CBM-for-health-audio exists in adjacent modalities. It is nonetheless the strongest available option: it reuses your assets more than any other candidate, it is a mechanism/interpretability contribution (not an accuracy race a big lab wins), it gives the open-world goal a *third, genuinely different* mechanism after two failures, and it fits your target-journal class. The two directions that scored near it were either occupied (TTA, edge-KD) or filling fast and compute-heavy (FM benchmarking).

**Mandatory pre-implementation step (the lesson from M15 and Direction 1):** before writing a line of code, run one more targeted kill-search — `"concept bottleneck" respiratory OR "lung sound" OR auscultation 2026` and `interpretable-by-design respiratory disease concept intervention` — to confirm no 2026 preprint has taken it. If clear, implement. If not, the leakage and intervention *questions* likely still survive even if a bare respiratory-CBM appears, so the direction can be narrowed rather than abandoned.

---

## Appendix — Papers consulted (2024–2026), by role

**Occupies / kills a candidate**
- TRIAGE — *Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification*, arXiv:2604.12647 (2026) — kills C3.
- *DHAuDS: Dynamic & Heterogeneous Audio TTA Benchmark*, arXiv:2511.18421 — kills C3.
- HeAR / *PulmoVec*, arXiv:2603.15688 (2026); *BCoughBench*, arXiv:2606.25116 (2026) — crowd C4, downgrade C2.
- *SpeechDx: Multi-Task Benchmark for Clinical Speech AI*, arXiv:2606.17339 — crowds C2.
- Clinical L2D: arXiv:2508.07617, 2508.06997, 2605.08024 — kills C6.
- *Advances and Challenges … ICBHI2017 Review*, Electronics 14(14):2794 — pre-empts C5.

**Nearest neighbors to the selected direction (must be cited & differentiated)**
- *An Audio Language Model-Based Voice Concept Bottleneck Framework for Interpretable Health Assessment*, arXiv:2607.16967 (2026) — CBM for **voice**, not lung sounds.
- *Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis*, arXiv:2410.15446 (2024) — CBM for **imaging**.
- *Prototype Learning for Interpretable Respiratory Sound Analysis*, arXiv:2110.03536 (2021) — **post-hoc/prototype**, not a bottleneck.
- *Measuring Leakage in Concept-Based Methods: An Information-Theoretic Approach*, arXiv:2504.09459 (2025); *Leakage and Interpretability in Concept-Based Models*, arXiv:2504.14094 — leakage method, **general** domain.
- Concept-based OOD: COOD (2024), CMA (2025), EOE (2024) — **vision/VLM**, not clinical audio.
- Foundation models: OPERA (NeurIPS 2024, arXiv:2406.16148); M2D+Resp (Interspeech 2025); RespiraMFM (arXiv:2606.09966, 2026) — context/baselines.

**Supporting rigor**
- BCoughBench (arXiv:2606.25116) — the field's own admission that FMs hide operating-point/calibration failure and have small-N reliability issues (motivates the evaluation standard).
- PC-MCL, arXiv:2601.17080 (2026) — patient-consistent multi-cycle learning / label-bias correction (relevant to patient-level aggregation).

*Search pass 2026-08. Do not treat [Plausible]/[Uncertain] labels as [Verified]. Re-run the two kill-queries in Step 7 immediately before implementation; this subfield is producing directly-relevant preprints monthly.*
