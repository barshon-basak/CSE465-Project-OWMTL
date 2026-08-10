# Open-World Multi-Task Learning for Respiratory Disease Diagnosis
## Research Guideline: Statistical Validity & Design Rationale

> **Trimmed 2026-08-05.** This document originally duplicated the proposal's evaluation protocol,
> ablation plan, and timeline. Those sections were removed and now point to `Project_Proposal_v2.md`
> instead, since maintaining two copies meant they could drift apart. The pre-trim version is at
> `Archive_Work_Plan/Research_Guideline_FULL_ARCHIVED.md`.
>
> **What this file is for now:** the *why* behind the project's design decisions — specifically the
> statistical-validity argument (§3) that motivated the whole task redesign. That argument is this
> document's unique contribution and the project's identified strongest claim; it isn't stated at
> this depth anywhere else. `Project_Proposal_v2.md` is the manuscript basis; this is the reasoning
> underneath it.

---

## 1. Executive Summary

The project — *Open-World Multi-Task Learning (OWMTL) for Respiratory Disease Diagnosis via Cross-Task Consistency* — has genuine, verifiable whitespace in the literature: no existing work combines (a) a shared-backbone multi-task model (sound-event + disease diagnosis), (b) cross-task disagreement as an unknown-disease detector, and (c) a staged open-world learning (OWL) protocol for respiratory audio. This was confirmed through direct literature search; the closest paper (an open-set semi-supervised respiratory classifier, *Computers, Materials & Continua*, 2025) uses a completely different mechanism (prototype/contrastive distance thresholding, single-task).

The held-out "unknown" disease classes in ICBHI 2017 (Pneumonia n=6, Bronchiolitis n=6, Bronchiectasis n=7 patients) are too small to support fine-grained, per-class quantitative ablation — a concern independently corroborated by a *Patterns* (Cell Press) paper that was forced to abandon multi-class ICBHI diagnosis for the same reason (their smallest class had a single patient). The proposal's own mitigation plan (cross-dataset augmentation via the Fraiwan/KAUH dataset) does not fix this — KAUH's Pneumonia count (n=5) is smaller than ICBHI's, not larger.

**This document sets out the redesigned plan that fixes the statistical problem while preserving the novelty claim, adds a defensible methodological contribution tied to Dr. Khan's CQKD lineage, and targets realistic Q1 venues.**

---

## 2. Literature Positioning (Why This Is Still Worth Doing)

### 2.1 What's crowded (avoid re-doing this)
| Sub-area | Status | Evidence |
|---|---|---|
| MTL for lung-sound + disease classification | **Saturated** | Tri-MTL (2025), MTL-MobileNet (SN Computer Science, 2024), PC-MCL (2026) all do closed-set MTL on ICBHI |
| ICBHI as a benchmark generally | **Very saturated** | Systematic review found **135 technical publications** using ICBHI 2017 |
| Generic open-set "two-head disagreement" mechanism | **Not novel as a mechanism** | Used in hyperspectral domain adaptation (SoDa2), dense outlier detection (2021), wildlife OSR (2025) |

### 2.2 What's genuinely open (the whitespace)
| Gap | Status | Evidence |
|---|---|---|
| Open-set recognition specifically for ICBHI respiratory sounds | **One paper exists** (CMC 2025), different mechanism (prototype+contrastive distance, single-task, not MTL) | Confirmed via search |
| MTL cross-task consistency **as an OWL/unknown-disease detector** | **No prior work found** combining MTL + cross-task disagreement + staged OWL protocol for respiratory audio | Confirmed via search |
| CQKD (Dr. Khan's 2025 methodology) applied to open-world respiratory diagnosis | **Unclaimed** | Not previously published; ties novelty to lab lineage rather than a borrowed generic trick |

**Conclusion:** the *application + staged protocol* is legitimately novel; the *underlying disagreement mechanism* is not. The guideline below routes the paper's core contribution through CQKD specifically to avoid the "known trick, new domain" critique.

---

## 3. The Core Problem to Solve First: Statistical Validity

### 3.1 The numbers, restated
- ICBHI 2017: 920 recordings, 126 patients. Known classes: COPD (64), Healthy (26), URTI (14) = 104 patients. Held-out: Bronchiectasis (7), Pneumonia (6), Bronchiolitis (6) = 19 patients.
- Fraiwan/KAUH (proposed augmentation source): Pneumonia = **5 subjects** — smaller than ICBHI's, not a fix.
- HF Lung V2 (cited in slide as an "unknown class" source): has **no disease-diagnosis labels at all**, and V2 is **not publicly released**. Cannot be used as claimed.

### 3.2 Why n≈6 breaks "comprehensive ablation"
- Bootstrap confidence intervals are documented to be systematically too narrow (optimistic) at n=5, and remain unreliable even at n=20.
- Standard k-fold CV assumes enough samples per fold to be representative; at n=6 per class, folds become degenerate.
- A directly comparable published study (*Patterns*, Cell Press) using the same dataset was forced to collapse an 8-class diagnosis task into binary normal/abnormal because "some classes have only one patient" (asthma).

**Design rule going forward: never report a headline quantitative metric (AUROC, F1, accuracy ± CI) computed over fewer than ~15–20 independent patients.** Anything smaller becomes a qualitative case study, explicitly labeled as such.

---

## 4. Redesigned Task Formulation

### 4.1 Primary task (statistically defensible, this is what gets ablated and reported with CIs)
- **Sound-event classification** (Normal/Crackle/Wheeze/Both): 6,898 cycles — plenty of samples, keep this as a fully quantitative, fully ablated task.
- **Coarse open-world disease task**: known-class disease diagnosis (COPD/Healthy/URTI, n=104) **vs. a single pooled "unknown" class** (all 19 held-out patients pooled together, not split into 3 separate n=6 groups). This raises your smallest evaluation group from n=6 to n=19 and is statistically far more defensible.
- **Out-of-distribution (OOD) generalization test** (the real "scale" story — see §5): train entirely on ICBHI known classes, evaluate open-world rejection against an **externally sourced, much larger population** (Coswara, thousands of participants) as a genuinely unseen distribution. This is a stronger open-world claim than a same-dataset held-out split, and gives you real statistical power.

### 4.2 Secondary task (qualitative, clearly labeled, not headline results)
- Per-disease breakdown (Pneumonia n=6, Bronchiolitis n=6, Bronchiectasis n=7) reported as **individual patient-level case studies** with a small results table (which patients were correctly flagged, which weren't) — not as a metric with a confidence interval. Reviewers respond well to honesty about this; they penalize false precision.

---

## 5. Dataset Strategy

| Dataset | Role | Why |
|---|---|---|
| **ICBHI 2017** | Primary training set — known classes + pooled held-out set | Standard, well-annotated, patient-level and cycle-level labels |
| **Coswara** (IISc Bangalore) | **Large-N external OOD stress test** | Crowdsourced, thousands of participants, different recording modality/population — replaces the Fraiwan plan, which does not increase effective sample size |
| **SPRSound** (Shanghai Jiao Tong) | Secondary OOD stress test | Pediatric population, different device, different label granularity — provides a second independent domain shift |
| ~~Fraiwan/KAUH~~ | **Dropped as an augmentation source** | Pneumonia n=5, does not increase statistical power over ICBHI alone |
| ~~HF Lung V2~~ | **Dropped as an "unknown class" source** | No disease labels; V2 not publicly available |

**Rationale for the pivot:** instead of trying to patch ICBHI's tiny held-out classes with an equally tiny external dataset, treat cross-dataset transfer itself as the open-world test. "Never seen this population, device, or recording protocol before" is a *stronger* open-world claim than "never seen this ICBHI sub-class before," and it comes with real sample size.

---

## 6. Methodology and Novelty Anchor (CQKD Integration)

To avoid the "generic OSR trick applied to a new domain" critique, the methodological contribution should be routed through Dr. Khan's Cluster-Quantized Knowledge Distillation (CQKD, 2025) rather than a plain two-head disagreement score. Two concrete integration points to discuss with Dr. Khan:

1. **CQKD-compressed cross-task consistency signal**: instead of raw feature-space disagreement between the sound-event head and the disease head, distill both heads through a cluster-quantized teacher-student setup so the consistency signal itself is compact and edge-deployable — ties the open-world contribution to a deployability story (relevant given ICBHI's original clinical/screening motivation).
2. **CQKD-regularized OWL staging**: use cluster-quantization to constrain how much the shared backbone can drift between OWL stages, directly targeting the forgetting-curve metric already in your proposal's "Final Outputs" list, and giving you a mechanism-level (not just outcome-level) ablation to run.

This gives you an actual methods contribution to defend in the paper's related-work section: *"unlike prior cross-task disagreement methods [cite SoDa2, dense outlier detection, wildlife OSR], we ground the consistency signal in cluster-quantized distillation, which additionally yields X% model compression and Y improvement in forgetting resistance."*

---

## 7–8. Evaluation Protocol & Ablation Plan → see the proposal

These sections previously duplicated `Project_Proposal_v2.md` §7 (evaluation protocol) and §8
(ablation plan) almost verbatim. They now live only there, so there's one authoritative copy.

The rules that matter most, restated in one line each because they're referenced constantly:
**LOPO not k-fold for groups under ~30 patients; no bootstrap CIs under n=15; patient-independent
splits always; non-parametric tests for small-group comparisons.** These are also enforced
operationally in `Model_Training_Protocol.md` §1 and checked automatically by
`Asif's/audit/audit_project.py`.

---

## 10. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| No disagreement signal found in pilot | Medium | Phase 0 checkpoint catches this in week 2, not month 3 |
| CQKD adaptation doesn't transfer cleanly to this setting | Medium | Early design review with Dr. Khan (Phase 2) before full build |
| Reviewers still object to small-N held-out groups despite redesign | Low-Medium | Framing as qualitative case studies + large-N OOD tests as the real headline result should pre-empt this |
| Coswara label mismatch with ICBHI taxonomy complicates OOD framing | Medium | Treat Coswara purely as a domain-shift/"unknown" stress test, not as requiring taxonomy alignment |
| Timeline slips past semester deadline | Medium | Phases 0–1 are hard gates; if either fails, pivot decision should be made before Phase 3 begins |

---

*This document synthesizes literature search conducted across ICBHI-related publications, open-set/open-world recognition literature (vision and audio), dataset documentation (ICBHI, Fraiwan/KAUH, HF Lung V1/V2, Coswara, SPRSound), and journal scope/indexing data (Biomedical Signal Processing and Control, Computers in Biology and Medicine, IEEE JBHI) as of July 2026.*
