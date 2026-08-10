<h1 align="center">Open-World Multi-Task Learning for Respiratory Disease Diagnosis</h1>
<h3 align="center">Cluster-Aware OWMTL via Cross-Task Consistency on Respiratory Sound</h3>

<p align="center">
  <em>CSE465 — Machine Learning · Group 5 Capstone · North South University</em><br>
  <strong>Target venue:</strong> <em>Biomedical Signal Processing and Control</em> (Elsevier · Q1 · IF 4.9)
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20%26%20planning-yellow">
  <img alt="Task" src="https://img.shields.io/badge/domain-respiratory%20audio-blue">
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-ICBHI%20%7C%20Coswara%20%7C%20SPRSound-green">
  <img alt="License" src="https://img.shields.io/badge/license-academic-lightgrey">
</p>

---

## 📌 Overview

Automated respiratory-sound classification on the **ICBHI 2017** benchmark is a saturated field (135+ technical publications), and nearly all of it — including multi-task learning (MTL) approaches — assumes a **closed world**: every disease seen at test time was seen at training time. That assumption is clinically unrealistic; a deployed screening tool *will* meet conditions it was never trained on, and a model that confidently misclassifies an unknown disease as a known one is more dangerous than one that abstains.

This project proposes an **Open-World Multi-Task Learning (OWMTL)** framework that answers a single question:

> **Can a multi-task model detect, at inference time, that a patient's condition does not match any disease class it was trained on — using nothing but the disagreement between its own sound-event and disease-diagnosis predictions?**

The core idea is **cross-task consistency**: a shared audio backbone feeds two heads — a *sound-event* head (normal / crackle / wheeze / both) and a *disease-diagnosis* head. When the two heads' implied diagnoses disagree strongly, the input is flagged as a likely **unknown** disease. This is evaluated under a **staged open-world learning (OWL)** protocol and stress-tested for scale on external datasets (**Coswara**, **SPRSound**).

> ⚠️ **Repository status:** This is currently a **research & planning repository** — it holds the proposal, methodology guideline, work-split plan, and curated literature. The experimental pipeline is in early setup. See the [Roadmap](#-roadmap) for what lands next.

---

## 🎯 Novelty & Research Gap

| Claim | Status |
|---|---|
| MTL for ICBHI sound + disease classification | Established, **crowded** |
| Open-set recognition for ICBHI | **One** paper exists (different mechanism) |
| MTL **cross-task disagreement** as an open-world unknown-disease detector, under a staged OWL protocol, on respiratory audio | **No prior work found** ← *our contribution* |
| Cluster-quantized distillation for an edge-deployable open-world respiratory model | **Unclaimed** ← *second contribution* |

**Precise novelty statement:** *A multi-task architecture that uses cross-task disagreement between jointly-trained sound-event and disease-diagnosis heads, under a staged open-world learning protocol, to detect diseases unseen during training — evaluated with a statistically sound coarse-open-world / large-N-OOD design rather than fragile small-sample splits, and extended with a cluster-quantized distillation mechanism for edge deployability.*

We are explicit about what is **not** novel: two-signal "disagreement" as an open-set mechanism exists in other domains. The contribution is the **domain-specific combination** (MTL + staged OWL + respiratory audio) plus the **compression/deployability** extension — not the invention of disagreement-based rejection.

---

## 🧠 Method at a Glance

```
                    ┌─────────────────────────────┐
   raw lung-sound   │   Shared Audio Backbone      │
   ──────────────►  │   (CNN / MobileNet / AST —   │
   log-mel / MFCC   │    benchmarked in ablation)  │
                    └───────────┬─────────────────┘
                                │  shared representation
                 ┌──────────────┴───────────────┐
                 ▼                               ▼
      ┌────────────────────┐          ┌────────────────────────┐
      │  Sound-Event Head  │          │  Disease-Diagnosis Head │
      │  Normal / Crackle  │          │  COPD / Healthy / URTI  │
      │  / Wheeze / Both   │          │  (known classes)        │
      └─────────┬──────────┘          └───────────┬────────────┘
                │        implied vs. actual        │
                └──────────────┬───────────────────┘
                               ▼
                 ┌───────────────────────────────┐
                 │  Cross-Task Consistency Score  │
                 │  high disagreement ⇒ "UNKNOWN" │
                 │  (thresholded & calibrated per │
                 │   OWL stage)                   │
                 └───────────────────────────────┘
```

**Staged Open-World Learning (OWL) protocol**
- **Stage 0** — train on known classes only (COPD, Healthy, URTI).
- **Stage 1** — apply cross-task consistency thresholding; evaluate unknown-detection on the pooled held-out set and on Coswara / SPRSound.
- **Stage 2** — incrementally incorporate confirmed "new" cases and measure **catastrophic forgetting**, with and without cluster-quantized regularization.

---

## 📊 Datasets

| Dataset | Role | Key facts |
|---|---|---|
| **ICBHI 2017** | Primary train/eval | 920 recordings, 126 subjects, 6,898 cycles (normal/crackle/wheeze/both). Known: COPD (64), Healthy (26), URTI (14) = 104 patients. Held-out "unknown": Bronchiectasis (7), Pneumonia (6), Bronchiolitis (6) = 19 patients, **pooled**. |
| **Coswara** | Large-N OOD stress test | Crowdsourced breathing/cough/voice, ~2,635 participants — genuinely unseen population & recording modality. |
| **SPRSound** | Secondary OOD (pediatric shift) | 2,683 records, 9,089 events, 292 pediatric participants — different age group, device, and label granularity. |

> **Datasets are NOT included in this repository** (licensing + size). Obtain them from their official sources and place them under `data/` (see [Getting the Data](#-getting-the-data)). ICBHI's tiny per-disease classes cannot carry a headline metric — the evaluation is deliberately redesigned around a **coarse pooled open-world task** + **large-N cross-dataset OOD tests**, with per-disease results reported as **qualitative case studies only**.

---

## 🗂️ Repository Structure

```
CSE465-Project-OWMTL/
├── README.md                                  ← you are here
├── requirements.txt                           ← Python dependencies (skeleton)
├── .gitignore                                 ← ignores data/, .claude/, Python artifacts
├── Project_Proposal_v2.md                     ← full proposal (abstract → references)
├── Research_Guideline.md                      ← methodology, statistical-validity redesign, risk register
├── Project_Work_Plan.md                       ← chunk-based work plan (no fixed roles — see below)
├── Novelty Search.md                          ← ★ start here — active novelty priorities (2026)
├── Model_Training_Protocol.md                 ← results-JSON schema, metrics, checkpoint rules
├── Model_Training_Reference.md                ← what each model is, current real/synthetic status
├── Archive_Work_Plan/                         ← retired role-based planning docs, kept for history
├── Papers/
│   ├── Literature_Review_Curated_Papers.md    ← 15 curated, 2024+, reputable-venue references
│   ├── More Possible Papers.md                ← supplementary supporting papers
│   ├── [Paper 1] Multi-task Learning for Lung Sound and Lung Disease Classification.pdf
│   └── [Paper 2] Enhancing Respiratory Sound Classification Based on Open-Set.pdf
└── .claude/                                   ← local tooling config (ignored)

# Planned (added as the pipeline lands — see Roadmap):
# ├── data/            raw + preprocessed datasets (git-ignored)
# ├── backbone/        Chunk B — shared encoder + sound-event task
# ├── owl_mechanism/   Chunk D — disease head + cross-task consistency + OWL staging
# ├── generalization_compression/  Chunk F — Coswara/SPRSound OOD + CQKD compression
# ├── trust_calibration/  Chunk G — calibration + conformal + XAI
# ├── configs/         experiment configs
# └── notebooks/       exploratory analysis
```

---

## 📚 Key Documents (start here)

| Document | Read it for |
|---|---|
| **[Novelty Search](Novelty%20Search.md)** | ★ **Read this first.** The 2026 novelty priorities — what Dr. Khan's guidance actually requires, and §4.0's scope rule: **2–3 selected items, deliberately not the whole list.** |
| **[Project Proposal](Project_Proposal_v2.md)** | The full scientific case: abstract, related work, novelty table, methodology, ablation plan, references. |
| **[Research Guideline](Research_Guideline.md)** | *Why the task was redesigned* — the statistical-validity argument (why n≈6 classes can't carry ablation), dataset strategy, evaluation protocol, risk register. |
| **[Project Work Plan](Project_Work_Plan.md)** | Chunk-based work breakdown (no fixed roles) — current status per chunk and what to pick up next. |
| **[Model Training Reference](Model_Training_Reference.md)** | What each model is, what's real vs. synthetic (verified by the audit), and its dependency chunk. |
| **[Curated Literature](Papers/Literature_Review_Curated_Papers.md)** | 15 hand-picked 2024+ references from reputable venues, each mapped to the pillar it supports. |

---

## 🧩 Chunks of Work

The project used to split into lettered member roles (A/B/C/D). That's retired as of 2026-08 — the
audit found the role split had a real cost (most of what was tracked as "done" under one role turned
out to be synthetic-data scaffolding, not results), and the supervisor's 2026 guidance means progress
is no longer measured by model count or role coverage anyway. Work is now organized into
**independently-trainable chunks** — anyone can pick up any unblocked chunk.

| Chunk | Covers | Core question | Status |
|---|---|---|---|
| **B — Backbone & Sound-Event** | Shared audio encoder + 4-way cycle-level classification; architecture search (CNN / MobileNet / AST). | *What representation?* | ✅ Done, real (M2 selected) |
| **C — Open-Set Baselines** | OpenMax/Weibull + trivial post-hoc OOD scores on the frozen backbone. | *What's the real detection floor?* | ✅ Done, real (AUROC 0.6466) |
| **D — Core Mechanism** | Disease head, cross-task consistency scorer, staged OWL. | *Does unknown-detection actually work?* | 🔴 **Broken/synthetic — the current bottleneck** |
| **E — Novelty Layer** | 2–3 *selected* items (`Novelty Search.md` §4.0) — deliberately not the full supervisor list. | *Is there a contribution beyond the base mechanism?* | ⚪ Scope decided, not started |
| **F — Generalization & Compression** | Coswara/SPRSound OOD + cluster-quantized distillation. | *Does it generalize & deploy?* | 🔴 Broken/synthetic, optional |
| **G — Trust & Calibration** | Calibration, conformal-guaranteed abstention, explainability. | *Can the flag be trusted?* | ⚪ Mostly not started, optional |

See **[Project_Work_Plan.md](Project_Work_Plan.md)** for the full breakdown, current status per
chunk, and what to pick up next. The one fixed dependency is unchanged from the old plan: Chunk D
needs Chunk B's backbone checkpoint before it can start.

---

## 🧪 Evaluation Protocol (non-negotiable rules)

These guardrails exist because ICBHI's held-out disease classes are tiny; violating them produces false precision that reviewers penalize.

- **Leave-One-Patient-Out (LOPO)** for any group smaller than ~30 patients; k-fold only for the large-N sound-event task.
- **No bootstrap confidence intervals for n < 15** groups — raw per-patient outcome tables instead.
- **Patient-independent splits everywhere** — no patient's cycles appear in both train and test.
- **Non-parametric significance testing** (e.g., Wilcoxon signed-rank across patients) for small-group comparisons.
- Per-disease breakdowns (Pneumonia/Bronchiolitis/Bronchiectasis) are **qualitative case studies**, explicitly labeled, never headline metrics.

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10+**
- PyTorch (CUDA build recommended for training)
- `torchaudio`, `librosa`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`

> A pinned `requirements.txt` / `environment.yml` will be committed alongside the first pipeline code.

### Clone
```bash
git clone https://github.com/barshon-basak/CSE465-Project-OWMTL.git
cd CSE465-Project-OWMTL
```

### Getting the Data
Datasets are **not** redistributed here. Download from the official sources and place under `data/`:

| Dataset | Source |
|---|---|
| ICBHI 2017 | [BHI Challenge / Rocha et al., *Physiological Measurement* 2019](https://bhichallenge.med.auth.gr/) |
| Coswara | [iiscleap/Coswara-Data (GitHub)](https://github.com/iiscleap/Coswara-Data) |
| SPRSound | [SJTU-YONGFU-RESEARCH-GRP/SPRSound (GitHub)](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) |

Expected layout:
```
data/
├── icbhi/      # official ICBHI 2017 audio + annotation files
├── coswara/    # extracted Coswara recordings + metadata
└── sprsound/   # SPRSound records + JSON annotations
```

> Training/preprocessing commands will be documented here once the corresponding modules are committed.

---

## 🗺️ Roadmap

Aligned with the phased plan in the research guideline (~5–6 months, gated by a Phase-0 go/no-go).

- [x] Literature review & gap analysis
- [x] Task redesign for statistical validity (coarse pooled open-world + large-N OOD)
- [x] Proposal, methodology guideline, and work-split plan
- [x] Curated 2024+ reference list
- [ ] **Phase 0 — Pilot:** confirm cross-task disagreement signal exists on held-out classes *(go/no-go gate)*
- [ ] ICBHI preprocessing pipeline (shared codebase asset)
- [ ] Backbone ablation (CNN / MobileNet / AST) + sound-event results
- [ ] Disease head + cross-task consistency scorer + OpenMax/Weibull baseline
- [ ] Staged OWL protocol (Stage 0 → 1 → 2) + forgetting curves
- [ ] Coswara / SPRSound OOD generalization runs
- [ ] Cluster-quantized distillation + compression ablation
- [ ] Full ablation suite, statistical validation, per-disease case studies
- [ ] Manuscript drafting → internal review → submission

---

## 📖 Citing / Related Work

Core positioning references (full annotated list in **[Papers/Literature_Review_Curated_Papers.md](Papers/Literature_Review_Curated_Papers.md)**):

- Rocha, B.M., et al. (2019). *An open access database for the evaluation of respiratory sound classification algorithms.* **Physiological Measurement**, 40, 035001.
- Suma, K.V., et al. (2025). *Multi-task Learning for Lung Sound and Lung Disease Classification.* **SN Computer Science**, 6:51.
- Cho, W., Lee, S. (2025). *Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning.* **Computers, Materials & Continua**, 84(2), 2847–2863.
- Karim, A.A.J., Mahmud, M.Z., Khan, R. (2024). *Advanced vision transformers and open-set learning for robust mosquito classification.* **PLOS Computational Biology**, 20(12), e1012654.

A BibTeX entry for this project will be added once a preprint/manuscript is available.

---

## 📝 License & Academic Integrity

This repository contains coursework for **CSE465** at North South University. Code (once added) is intended for **academic and research use**. Third-party datasets remain under their **original licenses** and are **not** redistributed here. If you build on this work, please cite the underlying dataset and method papers listed above.

---

<p align="center"><sub>Group 5 · CSE465 · Supervised research toward a Q1 submission. Planning docs are living documents — revisit after the Phase-0 pilot.</sub></p>
