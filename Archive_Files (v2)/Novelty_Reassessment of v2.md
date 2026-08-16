# OWMTL — Critical Novelty Reassessment

**Prepared:** 2026-08-11
**Scope:** Independent, reviewer-style reassessment of the OWMTL project's novelty, based on the four project files (Proposal v2, Novelty Search, Progress Report, Work Plan) and a fresh literature search (2024–2026).
**Stance:** Deliberately critical and conservative. The goal is not to defend the original idea but to find a contribution that survives a Q1 reviewer.

---

> # 🔴 SUPERSEDED — historical record only
>
> **Do not act on this document's recommendation.** Two things have invalidated it since it was
> written on 2026-08-11:
>
> **1. Its selected direction was rejected two days later.** `New_Directions_Search.md` states that
> Direction 1 (device/disease disentangled open-world benchmark) "was rejected after revalidation
> because 2025–2026 papers closed it," and selects **ACBD (Acoustic Concept-Bottleneck Diagnosis)**
> instead. `New_Directions_Search.md` is the current direction document.
>
> **2. It contains a factual error that its core proposal depended on.** §4.3 builds the benchmark
> on *"ICBHI's 7 stethoscopes as a controlled covariate axis."* **ICBHI has 4 recording devices**
> (AKGC417L, LittC2SE, Litt3200, Meditron). The 7 is the number of chest **locations** (Tc, Al, Ar,
> Pl, Pr, Ll, Lr). The error originated in `Research_Progress_Report.md` (now corrected) and
> propagated here.
>
> Device stratification also has an unverified prerequisite that still applies to ACBD's experiment
> 6: if patients don't span multiple devices, leave-one-device-out is just
> leave-those-patients-out and no device claim is possible. Run
> `Asif's/audit/check_device_structure.py` before planning any device-based experiment.
>
> **3. Its performance figures are the legacy macro metric.** §1.2 cites "M30 fusion 0.8213, M35
> physics loss 0.7839, M37 LoRA 0.7969." On the official ICBHI 2017 metric these are **M30
> unverifiable** (no committed confusion matrix), **M35 0.6864**, **M37 0.6753** — and all three are
> on a non-official 70/30 split. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`.
>
> Its §1.2 conclusion nevertheless gets *stronger*, not weaker, with corrected numbers: the results
> are not merely "incremental" in a saturated space, they sit **at** the published ICBHI level
> (~0.60–0.65 official) rather than above it.
>
> **What remains valid and worth reading:** §1.1 (the cross-task mechanism is Zamir et al. 2020's
> Consistency Energy — genuinely not novel), §1.3 (foundation models have overtaken the backbone
> story), §1.5 (the M14 conformal paradox), and the Appendix citation list.

---

## 0. TL;DR (read this first)

1. **The original core novelty is dead on arrival — and your own results already prove it.** "Cross-task disagreement between a sound-event head and a disease head detects unseen diseases" fails empirically (M15 AUROC 0.5747 vs. a two-line Energy baseline at 0.6466) *and* fails on prior art (the mechanism is a published, named OOD signal — "Consistency Energy," Zamir et al. 2020 — and open-set/open-world respiratory recognition already exists). Trying to "fix M15" is the wrong instinct: even if you raised it to 0.70, you would have a worse version of an already-published idea, measured on n=19 patients where the confidence interval (~±0.12) swallows the entire effect.

2. **Your strongest *positive* results (M30 fusion 0.8213, M35 physics loss 0.7839, M37 LoRA 0.7969) are all incremental closed-set sound-event classification.** That literature is saturated (135+ ICBHI papers; feature-fusion and attention-fusion nets already report 0.82–0.86). None of these is a Q1 contribution on its own, and none of them is "open-world" — publishing them would silently abandon the paper's actual thesis.

3. **The one genuinely defensible gap is hiding inside your biggest liability.** Your OOD evaluation (Coswara/SPRSound as "unknowns") *conflates* two different kinds of shift — a *new disease* (semantic novelty) and a *new device/population* (covariate shift). No respiratory-audio paper has separated these for the *detection* task, and ICBHI's 7-stethoscope structure gives you a uniquely clean, controlled way to do it. The reframe: **stop claiming a novel detector; instead show that the field's open-world respiratory detectors are confounded — they flag the microphone, not the disease — and build the controlled benchmark plus a covariate-invariant detector that fixes it.** This turns a failed result into a rigorous, clinically meaningful, evaluation-first contribution, which is exactly the axis your own planning docs already identify as your strongest.

The rest of this document justifies each of these claims against the literature, lists the verified gaps, develops and ranks candidate directions, and specifies the selected direction end-to-end.

---

## 1. Diagnosis — Why the current novelty is insufficient

### 1.1 The core mechanism is neither novel nor working

**Claimed novelty:** inter-head disagreement between a sound-event head and a disease head as an unknown-disease signal, under a staged open-world protocol.

Two independent problems, either of which is fatal on its own:

**(a) It is not novel.** The proposal itself already concedes (Proposal §3.3, Novelty Search §1) that "disagreement between two signals as an unknown detector" is a general OSR mechanism used elsewhere (Huo et al. 2025, wildlife). The literature is worse for you than that admission suggests:

- **Cross-task consistency as an OOD/confidence signal is a *named, canonical* method.** Zamir et al., *Robust Learning Through Cross-Task Consistency* (CVPR 2020, arXiv:2006.04096), explicitly propose "Consistency Energy" — the disagreement across tasks — as "an unsupervised confidence metric as well as for detection of out-of-distribution inputs." Your central idea is a domain transfer of a 6-year-old, highly-cited method. A reviewer who knows the OOD literature will cite this in the first paragraph of their review.
- **Open-set / open-world recognition for respiratory sound already exists.** Cho & Lee (2025, *CMC*) do open-set semi-supervised respiratory classification. "Towards Open World Sound Event Detection" (2025, arXiv:2605.03934) takes open-world audio a step further. Open-set *medical diagnosis* with prototypes — essentially your M13 route — was just published in a top venue: *Openness-aware multi-prototype learning for open set medical diagnosis*, **Medical Image Analysis 2025** (S1361841525004098), and it already handles the exact failure mode you hit (single-prototype under-representation, open-space risk).
- **The staged "OWL protocol" (M17) is also not new.** Class-incremental / continual learning for respiratory sound with anti-forgetting (generative replay) is published (e.g., *Privacy-preserving synthetic respiratory sounds for class-incremental learning*, and multiple 2024 continual-audio methods). Your replay-based Stage-2 is a standard baseline, not a contribution.

**(b) It does not work, and cannot be rescued at this sample size.** Your own report is admirably honest here:

- M15 (best formulation) AUROC = **0.5747**, *below* the post-hoc Energy baseline **0.6466** and barely above chance.
- You already diagnosed *why* it can't be fixed by architecture (Progress Report §8, M15 analysis): joint MTL forces the heads to *agree* on unknowns (AUROC collapses to 0.4073); sequential MTL makes the disease head a linear echo of the sound head (no divergent representation). The mechanism has no headroom on this data.
- Even the *comparison itself* is statistically void: with n=19 unknown patients the 95% CI on AUROC is ≈ ±0.12 (Progress Report Gap 8). The gap between M15 and Energy is **not significant**, so "improving" M15 to 0.66 would prove nothing.

**Conclusion:** the core novelty is a known method, applied to a task where similar methods already exist, that empirically loses to a trivial baseline, measured on a sample too small to support any claim either way. This is a four-way reject risk, not a one-way one.

### 1.2 The strong results are in the wrong (saturated) part of the space

| Your best models | Task | Why it is *not* a Q1 novelty |
|---|---|---|
| M30 gated fusion, ICBHI **0.8213** | Closed-set sound-event | Feature/attention fusion on ICBHI is crowded and already at 0.82–0.86: ADFF-Net (Technologies 2025), Attention Feature Fusion via Knowledge Propagation (2024, "new SOTA, ~1% gain"), MFITN-E2NetGA (2025), multi-view CNN+AST fusion (2025). M30 lands at the *lower* end of this pack and adds no new mechanism. |
| M35 physics-informed loss, **0.7839** | Closed-set sound-event | Spectral/tonal priors for crackle/wheeze are well known (crackles = short broadband transients, wheezes = narrowband tonal); frequency-selection and attention papers already exploit exactly these properties (arXiv:2507.20052, 2025). A soft loss penalty is a nice touch, not a headline. |
| M37 Audio LoRA, **0.7969** | Closed-set sound-event | PEFT on audio backbones is standard practice by 2025; and you're LoRA-adapting a *from-scratch small CNN*, not an audio foundation model — which is the version that would actually be interesting (see §1.3). |

Each is a solid engineering result. **None is open-world, and none is novel enough to anchor a Q1 paper.** Publishing them means quietly becoming "yet another ICBHI closed-set classification paper," which is the single most saturated niche in the field (135+ papers on one database, per the 2025 *Electronics* review).

### 1.3 The proposal has already been overtaken by foundation models

The biggest strategic problem: **the ground moved.** The proposal's backbone story (CNN vs. AST) and the "encoder-agnostic" dodge for Attack 7 are now outdated.

- **OPERA** (*Towards Open Respiratory Acoustic Foundation Models*, **NeurIPS 2024**, arXiv:2406.16148): pretrained on ~136K respiratory samples / 400+ hours, a 19-task benchmark, and explicitly demonstrates **generalization to unseen datasets and new respiratory audio modalities.**
- **M2D+Resp** (Interspeech 2025) pushes the OPERA benchmark average from 0.733 to **0.814**.

You "deferred Attack 7 by decision." A Q1 reviewer will not let you. When a domain foundation model exists, is open-source, and *already claims the generalization property you're studying*, "we used a 3.6M-param CNN and declared ourselves encoder-agnostic" reads as avoiding the strongest baseline. Any credible 2026 respiratory paper must either use or explicitly out-reason OPERA/M2D.

### 1.4 The evaluation design undermines its own headline claim

The proposal leans on Coswara + SPRSound as "large-N OOD stress tests" for unknown-disease detection. This is the paper's intended rigor differentiator — and it is actually the **deepest methodological flaw**:

- **Coswara** is smartphone-recorded COVID cough/breath; **SPRSound** is pediatric, different device, different physiology. When your detector flags these as "unknown," you *cannot tell* whether it detected a **new disease** (semantic novelty — the thing you claim) or simply a **new microphone / a child's chest** (covariate shift — a nuisance). Your own MMD analysis (Gap 7) confirms these datasets differ enormously in low-level acoustics.
- This is not a minor caveat; it invalidates the central measurement. A reviewer asks: "How do you know your open-world detector isn't just a device detector?" — and under the current design you have no answer.

Note this is *also* the seed of the fix (§4): the confound is real, unaddressed in respiratory audio, and controllable with ICBHI's device labels.

### 1.5 Secondary weaknesses a reviewer will raise

- **Conformal paradox (M14):** 95.45% coverage with **0% unknown detection** is not a "distribution-free guarantee" selling point — it is a demonstration that conformal wrapping a near-random score yields a clinically useless operating point. Worse, recent work shows conformal prediction is fragile exactly here: *Pitfalls of Conformal Predictions for Medical Image Classification* (2025, arXiv:2506.18162) and results that **selective conformal issuance breaks validity**. As written, M14 is a cautionary tale, not a contribution.
- **Data-integrity history:** the audit found M11/M13/M15/M17/M18–M24 previously ran on `torch.randn`. Even though re-run, this will make you (rightly) conservative; every claim needs an audit trail.
- **Buzzword risk:** M31–M37 add GradNorm, demographics, temporal transformer, curriculum, physics loss, multistage KD, LoRA. Your own Novelty Search §4.0 correctly warns this reads as a technique list. Most of these underperform the baseline (M31 0.6377, M32 0.6547, M33 0.5506, M36 0.5137) and should be *cut* from the narrative, not showcased.

---

## 2. Verified research gaps (each checked against recent work)

For each gap: the gap, the evidence it is real, and the check that it is *not already solved* in respiratory audio.

### Gap A — Open-world respiratory detection conflates semantic novelty with covariate shift *(STRONGEST)*
**The gap:** Respiratory "unknown-disease" / OOD detectors are evaluated on datasets that differ in device, population, and modality simultaneously, so a high AUROC cannot be attributed to disease novelty. No respiratory-audio work isolates *semantic* shift (new disease, same acquisition) from *covariate* shift (new device/population, same disease).
**Real?** Yes — this is exactly the "full-spectrum OOD" distinction (Yang et al. 2022, arXiv:2204.05306) that vision has formalized. Medical *imaging* just adopted it: MICCAI 2025 builds a full-spectrum semantic-vs-covariate OOD benchmark for medical VLMs (papers.miccai.org/miccai-2025/0222). Adjacent audio (DCASE machine-sound ASD) actively disentangles domain vs. semantic novelty (arXiv:2501.01604; *Subject Information Extraction for Novelty Detection with Domain Shifts*, 2025, arXiv:2504.21247).
**Solved in respiratory audio?** **No.** Device-shift work in respiratory audio is all about *classifier* robustness/generalization (stethoscope-guided contrastive learning, ICASSP 2024, arXiv:2312.09603; causality-inspired FedDG, arXiv:2605.29862) — none studies whether the *novelty detector* is confounded by device, and none uses ICBHI's 7-device structure as a controlled covariate axis for detection. **This intersection is open.**

### Gap B — No foundation-model-based open-world evaluation for respiratory sound
**The gap:** OPERA/M2D report closed-set task performance and generic "unseen-dataset" generalization, but nobody has characterized whether a *respiratory foundation model's* embedding supports **disease-novelty rejection** (as opposed to covariate robustness), or how its calibration behaves under shift.
**Real?** Yes — OPERA (NeurIPS 2024) and M2D+Resp (Interspeech 2025) exist and are open, but their benchmarks are closed-set/regression tasks; open-set/novelty is not in the 19-task suite.
**Solved?** Not for open-world respiratory. Closely tied to Gap A (a foundation-model embedding is the natural place to test covariate-invariance of the detector).

### Gap C — Calibration/failure-detection of respiratory models under real device shift
**The gap:** How honest is a respiratory model's uncertainty when the *device* changes but the disease does not? Recalibration-under-shift is noted as necessary but unquantified for auscultation.
**Real?** Yes — general finding that foundation-derived medical models need local recalibration under shift; drift-adaptive respiratory frameworks exist for COVID cough (JMIR 2025) but not for auscultation device shift.
**Solved?** Partially in cough/COVID; **not** for ICBHI-style multi-device auscultation. Strong *supporting* contribution to A, weaker as a standalone.

### Gap D — Statistically honest small-sample open-set evaluation protocol
**The gap:** Respiratory open-set papers report single-point AUROC on tiny held-out disease sets (here n=19) with no CIs, making results irreproducible/uncomparable.
**Real?** Yes, and you have already lived it (±0.12 CI).
**Solved?** No standard protocol exists for respiratory OOD. This is a *methodology* contribution that pairs naturally with A (and is cheap given you already ran McNemar/bootstrap for M30).

### Gaps judged *not* worth pursuing (already solved / crowded)
- Cross-task disagreement detector (Zamir 2020; your failed M15). **Closed.**
- Prototype-based open-set medical diagnosis (Medical Image Analysis 2025). **Closed.**
- Feature-fusion closed-set ICBHI (ADFF-Net, MFITN, 2024–25). **Saturated.**
- Class-incremental respiratory replay (multiple). **Crowded.**
- Physics-informed spectral priors for crackle/wheeze (frequency-selection/attention papers). **Established; incremental only.**

---

## 3. Candidate novelty directions (developed, then ranked in §4)

Each candidate lists: what's novel · gap addressed · why it matters · reuse · new work · proof experiments · risks.

### Direction 1 — **Disentangled Open-World Respiratory Benchmark + Covariate-Invariant Detector** *(the "confound" paper)*
- **Novel:** First respiratory-audio study to *separate* semantic novelty (unseen disease) from covariate shift (unseen device/population) for the **detection** task, using ICBHI's 7 stethoscopes as a controlled covariate axis; plus a detector trained to be invariant to device while sensitive to disease.
- **Gap:** A (primary), D (bundled), C and B (supporting).
- **Why it matters:** Clinically, a screening tool that "abstains" every time the stethoscope changes is useless; one that abstains on genuinely new pathology is the goal. Nobody has shown the field's detectors confuse the two. It is a *depth/rigor* claim — the axis your own docs say is your strongest — and a great fit for BSPC-class venues that reward clinically-grounded evaluation over algorithmic flash.
- **Reuse (high):** M2/M30 backbones, M29 baseline suite (MSP/Energy/Mahalanobis/Entropy), Coswara/SPRSound loaders, M13 prototypical head, M14 conformal wrapper, M7 ensemble, ICBHI device metadata already in the pipeline, your McNemar/bootstrap code.
- **New work (moderate):** (i) construct the controlled splits — leave-one-device-out (covariate-only), held-out-disease same-device (semantic-only), and both; (ii) add a device-adversarial / gradient-reversal branch (or feature-whitening) to make the detection score covariate-invariant; (iii) a "semantic-vs-covariate response" metric per detector.
- **Experiments:** show existing detectors' AUROC is driven by device (drops sharply when device is controlled); show your invariant detector keeps semantic AUROC while suppressing covariate false-flags; verify on Coswara/SPRSound as *far*-covariate stress.
- **Risks:** device-only ICBHI splits also shrink N (mitigate with leave-one-device-out + reporting protocol from Gap D); adversarial training instability (mitigate with simpler feature-standardization ablation as fallback).

### Direction 2 — **Foundation-model open-world audit (OPERA/M2D) for respiratory novelty**
- **Novel:** First to test whether a respiratory *foundation-model* embedding supports disease-novelty rejection and stays calibrated under device shift; PEFT (your M37 LoRA, redone on OPERA) as the adaptation lever.
- **Gap:** B (primary), C.
- **Why it matters:** Directly answers the Attack-7 threat; repositions you on the 2026 frontier.
- **Reuse (moderate):** M37 LoRA machinery, M29 baselines, calibration code (M20).
- **New work (high):** integrate OPERA/M2D weights and preprocessing (different mel config, licensing), re-run the full suite.
- **Experiments:** OPERA vs. your CNN on closed-set + on the disentangled splits from Direction 1.
- **Risks:** compute/integration cost; if OPERA simply wins everywhere the paper becomes "foundation models are good," which is not itself novel unless framed via Direction 1's disentanglement.

### Direction 3 — **Clinically-calibrated abstention with an honest operating point** *(fix the M14 paradox into a contribution)*
- **Novel:** Replace the coverage/detection paradox with a *cost-sensitive selective-classification* framework that reports a real clinical operating point (e.g., target unknown-recall at bounded false-abstention), and characterize *when* conformal is and isn't valid for respiratory OOD (citing 2025 pitfalls work).
- **Gap:** D, C.
- **Why it matters:** Turns a negative result into a usable protocol.
- **Reuse (high):** M14/M20 exactly; M29 scores.
- **New work (low–moderate):** cost curves, selective-risk analysis, honest validity discussion.
- **Risks:** on its own this is a "methods note," not a full Q1 paper — best as a *component* of Direction 1, not standalone.

### Direction 4 — **Physics-guided, device-robust representation** (extend M35)
- **Novel:** Combine spectral physics priors (M35) with device-invariance so priors don't overfit adult lungs (your own SPRSound finding: M35 *hurt* pediatric generalization).
- **Gap:** partial C; weak novelty.
- **Why it matters:** clinically sensible but incremental; physics priors already established.
- **Reuse (high):** M35.
- **Risks:** reviewers see spectral priors as known; low ceiling. **Not recommended as primary.**

---

## 4. Selected direction (ranked, then specified end-to-end)

### 4.1 Ranking

Scored 1–5 on each axis (higher = better); weighted toward Novelty, Significance, Feasibility, Publication, Reuse.

| Direction | Novelty | Sci. significance | Feasibility (time left) | Publication fit | Reuse of existing work | Verdict |
|---|---|---|---|---|---|---|
| **1 — Disentangled open-world benchmark + invariant detector** | 4 | 5 | 4 | 5 (BSPC/CMPB) | 5 | **Selected** |
| 2 — Foundation-model open-world audit | 4 | 4 | 2 (integration cost) | 4 | 3 | Strong add-on to #1, risky alone |
| 3 — Calibrated abstention protocol | 3 | 4 | 5 | 3 (methods note) | 5 | Fold into #1 |
| 4 — Physics + device-robust | 2 | 3 | 4 | 3 | 5 | Drop / ablation only |

**Selected: Direction 1**, with Direction 3 folded in as the operating-point section and Direction 2 as an optional strengthening backbone (OPERA as one encoder in the invariance study, using your M37 LoRA skills). Rationale: it is the only candidate that is simultaneously (a) genuinely unoccupied in respiratory audio, (b) reuses almost everything you built, (c) converts your two biggest liabilities (the failed detector and the confounded OOD sets) into the contribution, and (d) is a *depth* claim — the exact axis your planning docs identified as your comparative advantage.

### 4.2 Disproving the selected idea (adversarial prior-art check)

Before committing, I searched specifically to kill it:

- **Full-spectrum / semantic-vs-covariate OOD** is done in **vision** (Yang 2022) and **medical imaging** (MICCAI 2025). → *Threat: the general idea isn't new.* **Mitigation:** frame honestly as a domain instantiation, not invention — exactly as you already do for the disagreement mechanism.
- **Domain-vs-semantic disentanglement for novelty** is done in **machine-sound ASD** (DCASE; arXiv:2501.01604; 2504.21247). → *Threat: adjacent audio exists.* **Mitigation:** those are non-clinical machine sounds with abundant data and no disease semantics; the clinical framing, the ICBHI *device-controlled* testbed, and the "detectors flag the stethoscope, not the disease" demonstration are new.
- **Respiratory device-shift** work exists (ICASSP 2024; FedDG 2026) but targets **classifier** generalization, never the **detector's** confound. → *Confirms the specific intersection is open.*

**Result:** the idea survives as a *domain-specific, evaluation-first* contribution, provided you (i) cite the vision/ASD/imaging precedents up front and (ii) center the claim on the *respiratory-clinical confound demonstration + controlled benchmark + invariant detector*, not on "inventing disentangled OOD." This is the same defensible-framing discipline your Novelty Search §1 already applies to the disagreement mechanism. It does **not** clear as an algorithmic-novelty paper for NeurIPS/ICLR — but it clears cleanly for a Q1 biomedical-signal venue.

### 4.3 Full specification

**Research gap →** Respiratory open-world/OOD detectors are evaluated on datasets that mix new diseases with new devices and populations, so reported novelty-detection performance is uninterpretable and clinically misleading; no work isolates semantic from covariate shift for the *detection* task in auscultation.

**Novel idea →** A **Disentangled Open-World Respiratory (DOWR) evaluation** that uses ICBHI's 7 stethoscopes as a controlled covariate axis to separately measure a detector's response to (i) unseen disease at fixed device (semantic), (ii) unseen device at fixed disease (covariate), and (iii) both; plus a **covariate-invariant novelty detector** that is trained/adjusted to fire on semantic novelty while staying quiet under device shift. Core empirical claim: *existing respiratory OOD detectors largely flag the microphone, not the disease; DOWR reveals it and the invariant detector corrects it.*

**Proposed method →**
1. **Controlled splits (the benchmark).**
   - *Semantic-only:* train on known diseases from a fixed device subset; test known (same device) vs. held-out disease (same device). Uses ICBHI held-out classes (Pneumonia/Bronchiectasis/Bronchiolitis).
   - *Covariate-only:* leave-one-device-out — train on 6 stethoscopes, test *known* diseases recorded on the 7th. A good detector should **not** flag these.
   - *Combined + far-covariate:* Coswara/SPRSound as extreme covariate (plus genuine new disease), reported separately so the two effects are never conflated.
2. **Disentanglement metric.** For each detector, report semantic-AUROC and covariate-false-flag-rate; define a **Disentanglement Score** = semantic-AUROC − covariate-FPR (or an equivalent ratio). Rank all M29 baselines + M15 + ensemble (M7) + foundation embedding by it.
3. **Invariant detector.** Add a device-adversarial branch (gradient reversal) or feature-standardization/whitening on the backbone used for scoring, so the OOD score's variance across devices (at fixed disease) is minimized. Fallback if adversarial training is unstable: post-hoc per-device score normalization (cheap, no retraining).
4. **Honest operating point (folded Direction 3).** Report cost-sensitive selective classification with a real clinical target (e.g., unknown-recall ≥ X at bounded abstention), and a frank validity analysis of conformal wrapping given the 2025 pitfalls literature — replacing the M14 paradox.

**Implementation changes (mostly reuse) →**
- *Reuse as-is:* M2/M30 backbones and features; M29 post-hoc suite; M7 ensemble scores; M13 prototypical head (as one detector); M14/M20 calibration; Coswara/SPRSound pipeline; McNemar/bootstrap code.
- *Repurpose:* M15's scoring harness becomes one of several *scored* detectors, not "the contribution."
- *New (small):* device-label plumbing into eval (ICBHI provides device per file); leave-one-device-out split generator; gradient-reversal head or whitening layer (~tens of lines); Disentanglement-Score computation; cost curves.
- *Optional strengthening:* one OPERA/M2D embedding as an additional scored backbone (reuse M37 LoRA), answering Attack 7.

**Experiments →**
1. Rank every detector (MSP, Entropy, Energy, Mahalanobis, ensemble, prototype-distance, cross-task) by semantic-AUROC **and** covariate-FPR under the controlled splits — the headline table.
2. **Confound demonstration:** show that top detectors' apparent OOD performance on Coswara/SPRSound is largely covariate-driven (high covariate-FPR on leave-one-device-out with *known* diseases).
3. **Invariant detector:** show it preserves semantic-AUROC while cutting covariate-FPR vs. the same detector without invariance.
4. **Operating point:** cost curves + selective risk at a stated clinical target; conformal validity discussion.
5. Statistics throughout: bootstrap CIs, Wilcoxon across devices/folds, per Gap-D protocol.

**Ablations →**
- Detector family × invariance {none, whitening, adversarial}.
- Backbone {M2, M30, optional OPERA} × invariance.
- Device granularity: leave-one-device-out vs. device-group holdout.
- Semantic difficulty: near (one held-out disease) vs. far (Coswara).
- With/without conformal wrapper (does it help or just cap detection?).
- Prototype vs. energy vs. ensemble as the base score under invariance.

**Expected contribution →**
1. **DOWR** — the first controlled, device-disentangled open-world evaluation protocol for respiratory audio (a reusable benchmark = citations).
2. An empirical finding of real clinical import: **existing respiratory OOD detectors are confounded by device shift** — they abstain on the wrong thing.
3. A simple **covariate-invariant detector** that measurably improves the disentanglement, with an honest clinical operating point.
4. A statistically-sound evaluation template for small-N respiratory open-set work.

This is a coherent single thesis ("*measure and fix the device/disease confound in open-world auscultation*"), not a technique list — and every piece is defensible individually.

### 4.4 What to cut and what to keep

- **Keep, reframed:** M2/M30 (backbones for the detector study, not the headline), M29 (the baseline zoo you now *rank*), M13 (one detector + small-N story), M7 (ensemble detector), M14/M20 (the honest-operating-point section), Coswara/SPRSound (far-covariate, clearly labeled).
- **Demote to ablation/appendix:** M35 physics loss (nice, include the SPRSound "priors don't transfer to pediatric" finding — it *supports* the covariate story), M37 LoRA (fold into optional OPERA arm).
- **Cut from the narrative:** M31 GradNorm, M32 demographics, M33 temporal transformer, M36 multistage distillation (all under-baseline; keep only as a one-line "we also tried, didn't help" to preempt reviewers).
- **Stop doing:** trying to "fix M15" as the paper's core; adding more techniques from Dr. Khan's list.

---

## 5. Honest bottom line for the team and Dr. Khan

The project's real problem was never that the models were bad — several are fine. It was that the **novelty was pointed at a saturated, already-solved target** (a known disagreement mechanism, on an already-open-set task, measured on n=19). The fastest route to a defensible Q1 story is not to strengthen that target but to **pivot to the confound your own OOD design accidentally exposed**: in respiratory auscultation, nobody has separated "new disease" from "new device," and your data (7 stethoscopes + two far-covariate datasets) is unusually well-suited to doing exactly that. It reuses almost everything you have, it is a *rigor* contribution rather than a *gadget* contribution (which fits BSPC and plays to your strength), and it survives an adversarial prior-art check as a domain-specific, evaluation-first paper — provided you cite the vision/imaging/ASD precedents openly and don't overclaim algorithmic invention.

If Dr. Khan wants one sentence: *"We stopped trying to invent a new unknown-disease detector and instead showed that the field's existing detectors can't tell a new disease from a new stethoscope — then built the controlled benchmark and the invariant detector that fix it."*

---

## Appendix — Key sources consulted (2024–2026)

- Zamir et al., *Robust Learning Through Cross-Task Consistency* (Consistency Energy as OOD signal), CVPR 2020 — arXiv:2006.04096.
- OPERA, *Towards Open Respiratory Acoustic Foundation Models*, NeurIPS 2024 — arXiv:2406.16148; benchmark site opera-benchmark.github.io.
- Niizumi et al., *Towards Pre-training an Effective Respiratory Audio Foundation Model* (M2D+Resp, avg 0.814), Interspeech 2025.
- Cho & Lee, *Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning*, CMC 2025.
- *Openness-aware multi-prototype learning for open set medical diagnosis*, Medical Image Analysis 2025 — S1361841525004098.
- Yang et al., *Full-Spectrum Out-of-Distribution Detection*, 2022 — arXiv:2204.05306.
- *Delving into OOD Detection with Medical Vision-Language Models* (full-spectrum semantic/covariate benchmark), MICCAI 2025 — papers.miccai.org/miccai-2025/0222.
- *Disentangling Hierarchical Features for Anomalous Sound Detection Under Domain Shift*, 2025 — arXiv:2501.01604.
- *Subject Information Extraction for Novelty Detection with Domain Shifts*, 2025 — arXiv:2504.21247.
- *Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification*, ICASSP 2024 — arXiv:2312.09603.
- *Mitigating Stethoscope-Induced Shortcuts under Federated Domain Generalization with Causality-Inspired Interventions*, 2026 — arXiv:2605.29862.
- *Towards Open World Sound Event Detection*, 2025 — arXiv:2605.03934.
- ADFF-Net, *Attention-Based Dual-Stream Feature Fusion Network for Respiratory Sound Classification*, Technologies 2025 — doi:10.3390/technologies14010012.
- *Improving Deep Learning-based Respiratory Sound Analysis with Frequency Selection and Attention*, 2025 — arXiv:2507.20052.
- *Pitfalls of Conformal Predictions for Medical Image Classification*, 2025 — arXiv:2506.18162.
- *Advances and Challenges in Respiratory Sound Analysis: ICBHI2017 Review*, Electronics 2025 — 14(14):2794.
- Class-incremental respiratory sound via generative replay — S2352648321000519.

*Citations reflect a 2026-08 search pass; verify exact author lists / volumes before submission, and re-run the novelty check close to submission since this area moves fast.*
