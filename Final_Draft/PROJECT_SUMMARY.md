# PROJECT_SUMMARY.md — OWMTL / ICBHI 2017, single source of truth

**Compiled:** 2026-08-31 · **Course:** CSE465, North South University · **Supervisor:** Dr. Riasat Khan
**Team:** Asif, Barshon Basak, Farhana, Sami

**How this file was built.** Every number below was read from a committed `results_M*.json`, a
committed analysis JSON, or recomputed here directly from the ICBHI annotation files. Where a
document in the repository disagrees with a committed result file, **the result file wins and the
disagreement is recorded** in §11. Nothing is copied from the earlier draft paper. Quantities that
do not exist are written `NOT RUN`, never estimated.

**Verification note.** The dataset counts in §5 were recomputed from
`~/Desktop/ICBHI_final_database/*.txt` plus `Asif's/ICBHI_challenge_train_test.txt` during the
writing of this summary, and reproduce the committed confusion-matrix supports exactly
(2,756 = 1579/649/385/143 and 2,636 = 1560/617/373/86). That is a genuine end-to-end check of the
split loader, not a restatement of it.

---

## 1. Project overview

**What it is.** A four-class respiratory sound-event classification project on the ICBHI 2017
Respiratory Sound Database (Normal / Crackle / Wheeze / Both, one label per annotated breathing
cycle), which turned into an **evaluation audit** of that benchmark and of our own pipeline.

**The problem it solves.** ICBHI 2017 is the field's default benchmark — a 2025 review counts well
over a hundred systems built on it — and almost every one is judged by a single headline score. We
found, by auditing 39 of our own committed experiments, that **five distinct measurement faults**
can move that score by up to 0.1475 without any change to the model. Two of the five are properties
of the published benchmark itself, not of us.

**Why it matters.** A benchmark whose reported numbers are not reproducible from committed evidence
cannot support the comparisons the literature makes on it. The corrections are cheap, checkable
from one confusion matrix and one stated partition, and we release the tooling that detects them.

**Main contribution (one sentence).** A priced, five-fault evaluation audit of ICBHI 2017 with
corrected baselines for a CNN and three vision transformers, a complete ablation and
interpretability study of the best model, and a pre-registered concept-validity gate that failed
twice and whose interpretation we then publicly retract on the evidence of a supervised probe.

**What this project is NOT.** It is not a new architecture, not a state-of-the-art claim, and not an
assertion that ICBHI's labels are wrong. The best model sits at the 2021 CNN level of the
literature and 6–10 points below the 2023–2026 frontier, and the paper says so plainly.

---

## 2. Research motivation and objectives

### The gap

| # | Weakness in the literature | Evidence |
|---|---|---|
| 1 | Reported scores rarely state the **partition** and the **metric definition**, which are the two facts needed to interpret them | Nguyen & Pernkopf (TBME 2022) say so directly; RespireNet moves **12.3 points** between the official split and 5-fold CV with the *same model* |
| 2 | The official partition is **described incorrectly** — multiple papers assert it has no patient overlap. It has. | `Asif's/audit/official_split.py`; patients 156 and 218 appear on both sides. **No published work flagging this was found.** |
| 3 | Confusion matrices are **rarely committed**, so no score can be independently recomputed | our own M15 and M30 are examples of the failure mode |
| 4 | Interpretability on this corpus is **post hoc and unverified** — heat maps are attributed to acoustic biomarkers without measuring whether the attribution lands there, and the reference standard's own reliability is never characterised | Sci. Rep. 2025 CNN-RNN fusion and similar |

### Questions actually asked (and where each is answered)

| Q | Question | Answer | Where |
|---|---|---|---|
| Q1 | How much of an ICBHI score is the measurement rather than the model? | Up to **0.1475** on one model | `tab:protocol` |
| Q2 | Do transformers beat a lightweight CNN on this corpus? | **No** — 12–38× the parameters, worse score | `tab:main` |
| Q3 | Does SpecAugment help? | **+0.0402** on the CNN, **nothing or negative** on all three transformers | `tab:augmentation` |
| Q4 | Which model components are established by the data? | **One** — fine-tuning the backbone | `tab:ablation` + `tab:unit` |
| Q5 | Can hand-built physics concepts reproduce ICBHI's crackle/wheeze labels? | **No.** Pre-registered gate failed twice | `tab:gate` |
| Q6 | Is that because the labels are unlearnable? | **No — retracted.** A frozen AudioSet probe reaches 0.71 / 0.76 | `tab:ceiling` |
| Q7 | Does the model attend to the acoustics the labels name? | **No** — it attends *less* to the wheeze band when a wheeze is present | `tab:xaiquant` |
| Q8 | Can the system detect an unseen disease? | **Not shown.** Every patient-level interval crosses chance at n = 19 | `tab:openset` |

### Pre-registered hypotheses that could fail, and did

This is the methodological spine of the project and the reason its negative results are worth
anything.

| Pre-registration | Threshold fixed before | Outcome |
|---|---|---|
| Gate G2: concept AUROC ≥ 0.65 **and** CI excludes chance | before Run 1 | **FAILED twice.** Test split read exactly twice; no third revision attempted |
| N11 supervised ceiling: ≥0.80 = "no ceiling" / 0.65–0.80 = "partial" / <0.65 = "ceiling confirmed" | before running | **"NO CEILING" branch fired** → we retracted our own ceiling claim |
| N8 pediatric physics: directional predictions on four named concepts | before reading any pediatric audio | **MIXED**, and we corrected our own test design (see §9) |
| M30 admission test: fusion must beat both its inputs on the same cycles | before running | **Never executed** → item dropped, not retried |

---

## 3. Project evolution

### Phase 1 — Original plan (v1/v2 proposal, ~July 2026)

**"Cluster-Aware Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis."**
Two heads on a shared CNN backbone — sound-event and disease — with their **cross-task
disagreement** used as an unseen-disease detector, plus a staged open-world learning protocol and a
cluster-quantized distillation extension for edge deployment. Target: *Biomedical Signal Processing
and Control* (Q1).

*Source:* `Archive_Files (v2)/Project_Proposal_v2.md`.

### Phase 2 — The mechanism failed (Aug 2026)

M15 (cross-task consistency) reached AUROC **0.5747** while a two-line post-hoc **energy score
reached 0.6466** (M29), and the joint multi-task variant collapsed to 0.4073. The significance
analysis then showed the comparison was moot: at **n = 19 unknown patients** every interval spans
chance (energy vs. disagreement, Δ = +0.072, p = 0.525).

*Source:* `Barshon's/M15/v6/`, `Asif's/M29/`, `Asif's/Statistics/SIGNIFICANCE_REPORT.md`.

### Phase 3 — The audit that changed everything (2026-08-13 → 08-16)

An automated audit of every results file found three classes of fault in our own work:
a non-standard metric, a silently substituted split, and a published split that is not
patient-independent. It also flagged several runs built on synthetic placeholder data. **The
project's own numbers were not measuring what their labels said.**

*Source:* `Asif's/audit/PROJECT_AUDIT.md`, `ICBHI_SCORE_AUDIT.md`, `official_split.py`.

### Phase 4 — Pivot to a physics-derived concept bottleneck, and Gate G2 (2026-08-14 → 08-16)

A gated roadmap (G0–G7) proposed replacing the failed mechanism with an interpretable-by-design
concept bottleneck built from DSP-derived acoustic concepts. **Gate G2** asked whether those
concepts reproduce ICBHI's labels at AUROC ≥ 0.65. It was run twice, with extractors revised
against five diagnosed failure modes between runs, and **failed both times**.

*Decision:* `DECISION_2026-08-16_PIVOT.md` — the bottleneck is retired as the headline; the
contribution becomes the corrected evaluation protocol plus the label-reliability finding.

### Phase 5 — The N-series and the retraction (2026-08-29 → 08-30)

Eleven scripted experiments (`Novelty Experiment/N1`–`N11`) closed out the supervisor's suggested
novelty directions on real data with controls and confidence intervals. Two of them corrected
claims the project had already made. The decisive one was **N11**:

> A logistic probe on a **frozen AudioSet embedding — a network that has never heard a respiratory
> corpus — reaches AUROC 0.7115 (crackle) and 0.7621 (wheeze)** on the same 2,636 test cycles Gate
> G2 failed on.

**The label-reliability *ceiling* argument was therefore retracted.** Our extractors were weak; the
reference standard was not the binding constraint.

*Decision:* `DECISION_2026-08-30_CONSOLIDATION.md`.

### Phase 6 — Course requirements and the fifth fault (2026-08-30)

M40–M42 added three pretrained transformers on the corrected split; M44 added interpretability;
M45/M46 added the twelve-row ablation; M48 added the protocol ablation, the seed band, the
cumulative ladder and the matched machine-vs-physician comparison. M48's Tier A run surfaced a
**fifth fault nobody had priced: checkpoint selection criterion, worth 0.075–0.102**.

### Why each change was made — the short version

| Change | Trigger | Would we do it again? |
|---|---|---|
| OWMTL mechanism → evaluation audit | The mechanism tied a two-line baseline, and the audit showed the comparison numbers were unreliable anyway | Yes. The audit produced a defensible contribution; more mechanism-chasing would not have |
| Concept bottleneck → retired as headline | Pre-registered gate failed twice | Yes — and honouring the two-read budget is what makes the null worth reporting |
| Ceiling claim → retracted | N11's pre-registered branch fired | Yes. Conceding it is what the gate discipline was for |
| Second clinician → dropped | N11 bounds the reference standard directly and more tightly than a second rater's κ would | Yes, but single-rater stays a stated limitation |
| M38 large-N open-set → dropped | n = 19 is structural; enlarging via a pediatric corpus confounds novelty with population shift | Yes |
| M4 (AST) → dropped, replaced by M43 | No committed results file existed; its numbers were prose only | Yes. **M43 then did not complete — this is the project's largest open gap** |

---

## 4. Final methodology

### The pipeline, component by component

| # | Component | Purpose | Input → Output | Setting as run |
|---|---|---|---|---|
| 1 | **Audited split loader** | Guarantee the partition is what it claims | split file + audio dir → train/test recording lists | `official_split.py`; **raises** if the file is absent (never falls back); reassigns leaking patients 156, 218 to train; device-suffix-safe filename join |
| 2 | Cycle segmentation | Define the unit of classification | recording + annotation → cycles | annotated start/end times; **patients** are the unit of splitting, **cycles** the unit of classification |
| 3 | Duration standardisation | Fixed-size network input | variable cycle → 8.0 s | **cyclic tiling** (`np.tile`), centre-truncate if longer. 99.8% of cycles are shorter than 8 s |
| 4 | Log-mel front end | Time–frequency representation | 8 s waveform → 128 × 801 | n_fft 1024, hop 160 (10 ms), win 400 (25 ms), 128 mel, 50–2000 Hz, `power_to_db(ref=max)` |
| 5 | Scaling | Numerical conditioning | dB map → [0,1] | per-spectrogram min–max |
| 6 | Channel adaptation | Use the pretrained stem as trained | 1 ch → 3 ch | replicate + ImageNet mean/std |
| 7 | Label encoding | Task definition | (crackle, wheeze) flags → {0,1,2,3} | Normal / Crackle / Wheeze / Both |
| 8 | Imbalance handling | Stop majority collapse | class counts → loss weights | inverse-frequency CE, weights normalised to mean 1 |
| 9 | Augmentation | Regularisation | train tensor → masked tensor | SpecAugment, **train split only**, 2 freq masks ≤24 bins, 2 time masks ≤80 frames, resampled each epoch |
| 10 | Backbone | Representation | 3 × 128 × 801 → features | MobileNetV2 / ViT-B/16 / Swin-T / DeiT-S, all ImageNet-1k |
| 11 | Head | Classification | features → 4 logits | GAP + dropout 0.3 + Linear |
| 12 | **Official scoring module** | Make every score checkable | predictions → score + evidence | raw 4×4 matrix committed; official `(Se+Sp)/2` **and** the macro variant, both labelled; patient-level bootstrap CI, B = 1000; paired McNemar / bootstrap |

**Training.** Adam, lr 5e-4, cosine schedule, weight decay 1e-4, grad clip 5.0, batch 16, 40 epochs,
dropout 0.3, seed 42, mixed precision. Transformers: AdamW, lr 1e-4 backbone / 1e-3 head, otherwise
identical.

**Hardware.** Kaggle Tesla T4 (16 GB) for the CNN runs and eight ablation rows; local NVIDIA RTX
4050 Laptop (6 GB) for the transformers and four ablation rows; Google Colab T4 for the two earliest
backbones.

### Three preprocessing stages that were specified but never implemented

The written protocol listed an explicit band-pass filter, spectral-gating denoising, and per-cycle
amplitude normalisation. **None of the three was in the training pipeline.** Rather than switch them
on and silently change the baseline every other measurement is referenced to, they were implemented
as **additive** ablation rows (P1, P2, P3). Only one helps.

**This is the honest resolution of a specification-versus-implementation gap and it is stated in the
paper.** A canonical module `Asif's/owmtl/features.py` exists, but **nothing imports it** and its
defaults disagree with the committed runs; it is marked a reference implementation only. Do not
"reconcile" it by changing the runs — that would invalidate every committed number.

---

## 5. Datasets

### ICBHI 2017 (primary)

920 recordings, 126 patients, 4 stethoscope models, 7 chest locations, 5.5 hours, 6,898 annotated
cycles. Cycle duration: median **2.54 s**, mean 2.70 s, range 0.20–16.16 s; **99.8% shorter than the
8 s input**.

**Indexed corpus after the device-suffix defect: 919 recordings, 6,887 cycles.**

| Partition | Overlap policy | Train rec. | Train cyc. | Test rec. | Test cyc. | Test pat. | Used for |
|---|---|---|---|---|---|---|---|
| Published verbatim | none | 538 | 4,131 | 381 | 2,756 | 49 | literature comparability |
| **Corrected (ours)** | reassign to train | 550 | **4,251** | 369 | **2,636** | **47** | **all new results** |
| Patient-disjoint alt. | drop from train | 525 | 4,014 | 381 | 2,756 | 49 | backbone re-decision |
| Identifier fallback | *a fault* | — | 6,406 | — | 492 | 11 | audited, never reported |

**Class distribution (recomputed from the annotation files, matches every committed matrix):**

| Partition / subset | Normal | Crackle | Wheeze | Both | Total |
|---|---|---|---|---|---|
| Whole corpus, indexed | 3,636 | 1,859 | 886 | 506 | 6,887 |
| Published train | 2,057 | 1,210 | 501 | 363 | 4,131 |
| Published test | 1,579 | 649 | 385 | 143 | 2,756 |
| **Corrected train** | 2,076 | 1,242 | 513 | 420 | 4,251 |
| **Corrected test** | **1,560** | **617** | **373** | **86** | **2,636** |
| Fallback test | 255 | 164 | 38 | 35 | 492 |

**Majority-class baselines:** accuracy 0.5918 (corrected test), 0.5729 (published test). Our best
model's accuracy is **0.5880 — below the corrected-partition majority prior.** Accuracy is the wrong
headline for this task and the paper says so.

**Augmentation and counts.** SpecAugment is applied *in place* at load time and resampled each
epoch, so the augmented training total **equals** the clean total (4,251). Reporting an inflated
"augmented total" would misdescribe the method. The one oversampling augmentation (M24 class
balancing) runs on the disease head on a different partition.

### Device–diagnosis confound (a dataset finding)

| Device | Recordings | Diagnoses (patients) |
|---|---|---|
| AKGC417L | 646 | COPD (32) **only** |
| Meditron | 127 | Healthy (26), URTI (14), Bronchiectasis (7), COPD (8), Bronchiolitis (6), LRTI (2), Pneumonia (1) |
| LittC2SE | 87 | COPD (16), Pneumonia (6), Asthma (1) |
| Litt3200 | 60 | COPD (11) **only** |

**Six diagnoses are 100% device-confounded** (Healthy, URTI, Bronchiectasis, Bronchiolitis, Asthma,
LRTI) and only **4 of 126 patients** span more than one device. Leave-one-device-out robustness is
therefore not available on this corpus. *Source:* `owmtl_concept_engine/.../device_structure_report.json`.

### Secondary corpora

| Corpus | Use | Size as used |
|---|---|---|
| SPRSound (BioCAS 2022) | pediatric covariate-shift analysis (N8) | 1,949 records → **6,656 annotated events, 243 children, ages 0.2–16.2 y (median 4.4)**; 177 poor-quality records dropped |
| Coswara | earlier cross-dataset OOD (M19) | **unusable — n = 2 in the OOD arm.** Do not cite its AUROC |
| AudioSet | pretraining of the frozen AST probe (N1, N11) | via `MIT/ast-finetuned-audioset-10-10-0.4593`, 86.5 M params |
| Clinician listening study | second reference standard | 132 clips, 1 physician, blind, 6 calibration exemplars, 12 hidden duplicates; **116 answered** |

---

## 6. Results — the verified record

> **Ranking rule:** scores are comparable **only within a partition**. Every table below carries a
> partition column for that reason.

### 6.1 Main results — corrected patient-independent partition (2,636 cycles, 47 patients)

| ID | Backbone | Aug. | **ICBHI** | 95% CI | Se | Sp | Acc | F1ₘ | Params | Best ep. |
|---|---|---|---|---|---|---|---|---|---|---|
| **M22-v2** | MobileNetV2 | yes | **0.5602** | [0.508, 0.614] | 0.4089 | 0.7115 | 0.5880 | 0.4141 | 2.23 M | 38/40 |
| M41 | Swin-T | no | 0.5304 | [0.472, 0.588] | 0.3429 | 0.7179 | 0.5649 | 0.3820 | 27.5 M | 18/40 |
| M41-a | Swin-T | yes | 0.5291 | [0.477, 0.581] | 0.3569 | 0.7013 | 0.5607 | 0.4058 | 27.5 M | 31/40 |
| M3-v2 | MobileNetV2 | no | 0.5200 | [0.463, 0.575] | 0.4266 | 0.6135 | 0.5372 | 0.3985 | 2.23 M | 27/40 |
| M42 | DeiT-S | no | 0.5149 | [0.466, 0.566] | 0.2946 | 0.7353 | 0.5554 | 0.3422 | 21.7 M | 2/40 |
| M40-a | ViT-B/16 | yes | 0.5000 | [0.500, 0.500] | **0.0000** | 1.0000 | 0.5918 | 0.1859 | 85.8 M | 1/40 |
| M40 | ViT-B/16 | no | 0.4994 | [0.498, 0.500] | **0.0000** | 0.9987 | 0.5910 | 0.1857 | 85.8 M | 1/40 |
| M42-a | DeiT-S | yes | 0.4981 | [0.445, 0.549] | 0.2974 | 0.6987 | 0.5349 | 0.3571 | 21.7 M | 27/40 |
| **M43** | AST | — | **`NOT RUN`** | — | — | — | — | — | — | — |

**Best-model per-class (M22-v2):**

| Class | Support | Prec. | Recall | Spec. | F1 |
|---|---|---|---|---|---|
| Normal | 1,560 | 0.6986 | 0.7115 | 0.5548 | 0.7050 |
| Crackle | 617 | 0.4501 | 0.5186 | 0.8063 | 0.4819 |
| Wheeze | 373 | 0.4911 | 0.2949 | 0.9496 | 0.3685 |
| Both | 86 | 0.0893 | 0.1163 | 0.9600 | 0.1010 |

Raw matrix: `[[1110,333,81,36],[269,320,8,20],[178,39,110,46],[32,19,25,10]]`.

### 6.2 Literature-comparable partition (2,756 cycles, 49 patients)

| Run | Backbone | ICBHI | Se | Sp | Note |
|---|---|---|---|---|---|
| **M22-v2-official** | MobileNetV2 + SpecAug | **0.5641** [0.512, 0.614] | 0.4562 | 0.6719 | **the comparability row — not the best model** |
| M3 | MobileNetV3-Small | 0.5132 | 0.1980 | 0.8284 | patient-disjoint variant |
| M2 | 2D-CNN scratch (4 blk) | 0.4720 | 0.2005 | 0.7435 | patient-disjoint variant |

M22-v2-official ranks **9th of 22** published systems, immediately above RespireNet (0.5620, EMBC
2021) and 6–10 points below the 2023–2026 frontier. **Our sensitivity (0.456) is level with 2024
SOTA (BTS 0.457) and above Patch-Mix (0.431) and RespireNet (0.401); the entire deficit is
specificity** (0.672 vs. a field spanning 0.722–0.821).

### 6.3 The five faults, priced

| Fault | What it is | Worth | Evidence |
|---|---|---|---|
| 1. Metric | `(recall_macro + specificity_macro)/2` reported as the challenge score | **mean +0.0948**, max +0.2176, min +0.0025 over 20 verifiable runs | `ICBHI_SCORE_AUDIT.md` |
| 2. Split fallback | `pid ≤ 111 → test` gave 11 patients / 492 cycles (7.1%) while recording itself as official 60/40 | **+0.0893** (E2 vs E0); **+0.0854** on the clean same-model pair | `M48_tier_E_table.json` |
| 3. Patient leak | published split assigns *recordings*; patients 156, 218 on both sides | **+0.0039 — inside the CI. Report as a validity fault, NOT as inflation** | `results_M22_v2_official.json` |
| 4. Filename join | `226_1b1_Pl_sc_Meditron` in the split file vs `..._LittC2SE` on disk | **1 recording / 11 cycles silently dropped** (train side only) | `official_split.py --audio-dir` |
| 5. **Checkpoint selection** | selecting on loss rather than on the challenge score | **0.075–0.102 across 3 seeds** | `M48_tier_A_table.json` |
| **1 + 2 combined** | the unaudited pipeline as it would have been published | **0.7077 vs 0.5602 = +0.1475** | `M48_tier_E_table.json` |

**Fault 3 also has a non-score consequence:** the repair removes 120 cycles, of which **57 are
"Both" — 39.9% of that class.** Two patients supply 40% of the official test set's rarest class, and
they are the same two who leak into training.

### 6.4 Extension models — 70/30 partition, NOT comparable to §6.1

| ID | Mechanism | ICBHI | Se | Sp | F1ₘ | Verdict |
|---|---|---|---|---|---|---|
| M35 | Physics-informed loss | 0.6864 | 0.6980 | 0.6747 | 0.6420 | best extension |
| M37 | LoRA (0.23% params) | 0.6753 | 0.7517 | 0.5989 | 0.6599 | best efficiency |
| M35-v2 | physics loss, repeat | 0.6719 | 0.7401 | 0.6036 | 0.6491 | **broken — best epoch 1/30** |
| M33-v2 | temporal transformer | 0.5832 | 0.6190 | 0.5473 | 0.4948 | below baseline |
| M34 | curriculum pacing | 0.5754 | 0.6340 | 0.5168 | 0.5268 | below baseline |
| M31 | GradNorm MTL | 0.5535 | 0.5456 | 0.5614 | 0.4351 | below baseline |
| M36 | multistage distillation | 0.5052 | **0.0932** | 0.9171 | 0.2155 | **collapsed** |
| M32 | demographic fusion | 0.4733 | 0.5721 | 0.3745 | 0.4378 | below baseline |
| M33 | temporal transformer v1 | 0.3330 | 0.6660 | **0.0000** | 0.2393 | **collapsed** |

**Six of eight underperformed a plain backbone on an easier partition; two collapsed; one of the two
that worked did not reproduce.** Accumulating mechanisms did not accumulate performance.

### 6.5 Concept gate, ceiling and human agreement

**Gate G2 (9 concepts, corrected partition):**

| Run | Concept | AUROC | 95% CI | AUPRC | Prev. | Verdict |
|---|---|---|---|---|---|---|
| 1 | crackle_score | 0.5506 | [0.526, 0.575] | — | 0.2667 | FAIL |
| 1 | wheeze_score | 0.5729 | [0.546, 0.600] | — | 0.1741 | FAIL |
| 2 | crackle_score | 0.5580 | [0.533, 0.583] | 0.3079 | 0.2667 | FAIL |
| 2 | wheeze_score | 0.5340 | [0.505, 0.563] | 0.1903 | 0.1741 | FAIL |

> Run 1 **passed** under an earlier CI-only rule at AUROC 0.55. Adding the ≥0.65 floor turned the
> same evidence into a failure. **A gate that cannot fail is not a gate** — report both verdicts.

**A separate 14-concept re-implementation** gives crackle 0.5556 [0.529, 0.581] and wheeze 0.5818
[0.552, 0.610], full-vector 0.6561 / 0.5847. **This is a different implementation, not a re-export
— never mix the two pairs in one table.**

**N11 supervised ceiling (same 2,636 test cycles):**

| Representation | crackle | wheeze |
|---|---|---|
| our gate, single scalar | 0.5580 | 0.5729 |
| our 14 concepts as a vector | **0.6608** [0.601, 0.713] | 0.5815 [0.492, 0.675] |
| **AST frozen (never saw ICBHI)** | **0.7115** [0.638, 0.771] | **0.7621** [0.702, 0.815] |
| our own CNN encoder | 0.7451 [0.694, 0.795] | 0.8673 [0.806, 0.918] |

**Clinician study (N9), 1 physician, blind, 116/132 answered:**

| | crackle | wheeze |
|---|---|---|
| κ vs ICBHI | **0.035** [−0.157, 0.210] | **0.266** [0.036, 0.444] |
| raw agreement | 0.5278 | 0.7182 |
| sensitivity vs ICBHI | **0.2308** | 0.2973 |
| specificity vs ICBHI | 0.8036 | 0.9315 |
| confusion TP/FP/FN/TN | 12/11/40/45 | 11/5/26/68 |
| **intra-rater κ** (8 hidden duplicates) | **0.714** | **0.600** |
| good-quality-only sensitivity check | κ 0.023 | κ 0.255 |

**The decisive contrast:** the rater agrees with *themselves* at κ 0.71/0.60 and with ICBHI at
κ 0.04/0.27. The disagreement is not rater noise. Our single rater's crackle sensitivity **0.2308**
independently replicates Tzeng et al.'s seven-physician **0.2323**.

**Machine vs physician on the SAME 132 clips (M48 Tier C):**

| | crackle (n=108) | wheeze (n=110) |
|---|---|---|
| probe vs ICBHI | 0.6786 [0.578, 0.778] | 0.7519 [0.636, 0.857] |
| probe vs clinician | 0.5097 [0.354, 0.665] | 0.5828 [0.412, 0.746] |
| **paired difference** | +0.169 [−0.025, +0.361], p=0.087 | **+0.169 [+0.002, +0.344], p=0.046** |

**Wheeze: significant. Crackle: same effect size, not significant at n=108 — report as undecided.**
⚠ The 132 clips are not a random sample (≥0.9 s, longest-first, abnormal oversampled), so the
*absolute* AUROC is not comparable to N11 and cannot overturn the gate failure. **The contrast is
the result; the level is not.**

### 6.6 Concept probing, leakage and the bottleneck

| Representation | concepts above chance | mean probe R² |
|---|---|---|
| **AST frozen (AudioSet only)** | **13 / 14** | **0.4102** |
| our CNN encoder | 12 / 14 | 0.4289 |
| AST + LoRA (0.341% params, 5 epochs) | 13 / 14 | 0.3195 |
| random-projection control | **0 / 14** | −0.0054 |

**LoRA degrades 13/14 concepts, mean ΔR² −0.0907, worst `rhonchi_presence` −0.219** — which N5
independently identifies as the *only* concept carrying disease information (+0.0293 bits, 6× the
next). Two experiments, different data, different estimators, same concept.

**Leakage (N5):** I(y;f|c) = **0.2025 bits** [0.110, 0.329], 12.77% of base, permutation null
−0.0086. Independently regenerated from the checkpoint it reproduces the committed 0.2209 to within
0.02 bits — a genuine reproducibility check that passed.

**Bottleneck non-replication (N6), 5 seeds, 43 test patients:**

| Mode | Acc | F1ₘ | 95% CI | seed sd | single-seed F1 |
|---|---|---|---|---|---|
| independent (strict) | 0.5814 | **0.5254** | [0.333, 0.700] | 0.059 | 0.4542 |
| leaky | 0.6512 | 0.4802 | [0.385, 0.554] | 0.021 | 0.5457 |
| sequential | 0.6744 | 0.4526 | [0.358, 0.540] | 0.032 | 0.4441 |
| opaque | 0.5814 | **0.4378** | [0.332, 0.521] | 0.019 | 0.5973 |
| shuffled control | 0.4884 | 0.2963 | [0.197, 0.393] | — | never run |

**The monotone ordering INVERTS across seeds. McNemar independent vs opaque p = 1.0000.** The
committed "interpretability cost of 0.2326 accuracy / 0.1432 macro-F1" was a **single-seed
artefact and is withdrawn.** Real vs shuffled concepts: accuracy +0.0930 (p=0.389, null); macro-F1
**+0.2291 [0.004, 0.441], p=0.045** — report both, quoting either alone misleads.

**Intervention (N7):** restoring true concepts *degrades* the diagnosis (−0.047 strict, −0.023
leaky); sensitivity ordering is identical to random ordering; in leaky mode per-concept sensitivity
collapses **20×** — direct evidence of bottleneck bypass. With the physician's 84 real corrections
(30 on test patients), accuracy moves 0.5349 → 0.3953.

### 6.7 Open-set and calibration

| Method | Space | AUROC | 95% CI |
|---|---|---|---|
| Energy (post hoc) | 768-d embedding | 0.6466 | [0.492, 0.802] |
| Energy | concept+embedding | 0.6414 | [0.496, 0.779] |
| Energy | 14-d concept | 0.6270 | — |
| Deep ensemble disagreement | 5 members | 0.6566 | *different partition* |
| Entropy / MSP / Mahalanobis | embedding | 0.5789 / 0.5025 / 0.5376 | — |
| **Cross-task disagreement** (the original thesis) | two heads | **0.5747** | — |
| Conformal wrapper | on the above | 0.4809 | below chance |
| OpenMax + Weibull | cycle level | 0.4516 | [0.42, 0.48] — significantly **worse** than chance |

**Every patient-level interval crosses 0.5.** Paired: energy vs disagreement Δ +0.072 p 0.525;
energy vs conformal Δ +0.166 p 0.170. The one significant comparison (energy vs OpenMax, Δ +0.195,
p 0.015) is confounded by unit of analysis.

**Conformal paradox, stated correctly:** 95.45% coverage at 95% nominal with **0% unknown
detection**. Coverage is guaranteed by construction so reporting it is circular; the informative
number is the empty-set rate on unknowns — **0.0, identical to knowns**, with unknowns getting
*larger* sets (1.84 vs 1.70). The wrapper detects nothing.

**Calibration (N4):** fitted temperature **4.89** (severe overconfidence); ECE 0.4049 → 0.1771; MCE
0.743 → 0.483; Brier 0.808 → 0.569; NLL 2.093 → 0.885; **accuracy invariant at 0.5116** (as a
monotone transform must be — this check caught a non-like-for-like comparison in M11). AURC 0.3424.
Selective accuracy at 100/90/80/70/50% coverage: 0.512 / 0.579 / **0.618** / 0.600 / 0.619.
**At Se = 0.90 the system refers 74.4% of all patients** at Sp 0.5714. That is the honest operating
point.

### 6.8 Interpretability (M44), 400 test cycles

| Measure | Value | Reading |
|---|---|---|
| Band pointing, wheeze present | 0.7140 (n=75) | — |
| Band pointing, wheeze absent | 0.7674 (n=325) | — |
| **Difference** | **−0.0534** [−0.079, −0.026], p=0.001 | **attends LESS to the wheeze band when a wheeze is present** |
| uniform baseline | 0.5547 | both groups are above it, so it does concentrate there *in general* |
| Tiling consistency | mean r **0.098**, median 0.180, 32.1% above 0.5 (n=361) | attribution differs across identical repeated audio → partly keyed on position |
| Attribution on padded material, wrap model | **0.5938** (uniform share 0.6792) | **59% of the evidence is on audio the tiling manufactured** |
| same, zero-padded model | 0.1013 | correctly ignores silence |

**Resolution caveat:** Grad-CAM is natively 4 × 26 cells (~500 Hz × 0.31 s) bilinearly upsampled.
Direction of both results is trustworthy; the spatial precision implied by the figure is not.
**The pointing game is not implementable** — ICBHI annotates whether, not where, and the crop *is*
the window, so the hit rate is 100% by construction.

### 6.9 Statistical evidence

**Seed band (3 runs, identical config):** 0.5540 / 0.5681 / 0.5657 → mean 0.5626, **sd 0.0075, range
0.0141**. M22-v2's 0.5602 sits inside it — also a cross-implementation reproducibility check.

**Unit of analysis:** the same 10 ablation rows tested twice — **7 of 10 significant per cycle, 1 of
10 per patient, 6 rows flip verdict.** The effective sample size is **47 patients, not 2,636
cycles.**

**Reading both together:** the paired test says this test set is too small to certify most deltas;
the seed band says run-to-run noise is too small to explain them. Most effects are probably real
and this test set cannot establish them. Report both.

---

## 7. Ablations

### 7.1 Leave-one-out (M45/M46), 12 rows, corrected partition

| Row | Change | ICBHI | Δ | 95% CI | Se | Sp | F1ₘ | Patient-level verdict |
|---|---|---|---|---|---|---|---|---|
| A0 | full model (reference) | 0.5602 | — | [0.508, 0.614] | 0.4089 | 0.7115 | 0.4141 | — |
| A1 | − SpecAugment | 0.5200 | −0.0402 | [0.463, 0.575] | 0.4266 | 0.6135 | 0.3985 | not shown to differ |
| A2 | − ImageNet init | 0.4999 | −0.0603 | [0.438, 0.551] | 0.3736 | 0.6263 | 0.3784 | not shown to differ |
| A3 | − class weighting | 0.5497 | −0.0105 | [0.502, 0.593] | 0.4275 | 0.6718 | 0.4294 | not shown to differ |
| **A4** | **frozen backbone** | **0.4595** | **−0.1007** | [0.400, 0.521] | 0.4349 | 0.4840 | 0.3374 | **DIFFERS** |
| A5 | 64 mels | 0.5427 | −0.0175 | [0.492, 0.587] | 0.4238 | 0.6615 | 0.4043 | not shown to differ |
| A6 | 4 s cycles | 0.5224 | −0.0378 | [0.463, 0.574] | 0.4954 | 0.5494 | 0.4058 | not shown to differ |
| P4 | zero-pad instead of tile | 0.5291 | −0.0311 | [0.478, 0.581] | 0.4684 | 0.5897 | 0.3841 | not shown to differ |
| P5 | − min–max scaling | 0.5539 | −0.0063 | [0.501, 0.604] | 0.4136 | 0.6942 | 0.4128 | not shown to differ |
| P1 | **+** band-pass | 0.5513 | −0.0089 | [0.492, 0.601] | 0.4442 | 0.6583 | 0.4077 | not shown to differ |
| P2 | **+** denoising | 0.5248 | −0.0354 | [0.458, 0.588] | 0.4572 | 0.5923 | 0.3983 | not shown to differ |
| **P3** | **+** amplitude norm. | **0.5764** | **+0.0162** | [0.514, 0.632] | 0.3996 | 0.7532 | 0.4217 | not shown to differ |

P1–P3 **add** a stage; their sign reads the other way. **Only A4 is established.**

### 7.2 Cumulative ladder (M48 Tier B) — PARTIAL

| Rung | Config | Order A | Order B |
|---|---|---|---|
| S0 | random init, frozen, plain CE | **`NOT RUN`** | **`NOT RUN`** |
| S1 | + ImageNet pretraining | **`NOT RUN`** | **`NOT RUN`** |
| S2 | + full fine-tuning | **`NOT RUN`** | **`NOT RUN`** |
| S3 | + SpecAug (A) / + class wt. (B) | 0.5497 | 0.5200 |
| S4 | + class wt. (A) / + SpecAug (B) | 0.5602 (+0.0105) | 0.5602 (+0.0402) |
| S5 | + amplitude normalisation | **0.5764** (+0.0162) | **0.5764** (+0.0162) |

**The ordering effect is measured, not caveated:** SpecAugment is credited +0.0402 in one ordering
and class weighting +0.0105 in the other. They interact.

### 7.3 Selection criterion and feature-extractor null (M48 Tier A)

| Run | by official score | by min loss | Δ | epochs apart |
|---|---|---|---|---|
| seed 42 | 0.5540 | 0.4788 | **+0.0752** | 15 |
| seed 1 | 0.5681 | 0.4666 | **+0.1015** | 19 |
| seed 2 | 0.5657 | 0.4665 | **+0.0992** | 17 |
| A24 (random+frozen) | 0.4979 | 0.4979 | 0.0000 | 0 |

**Negative-transfer decomposition:**

| Config | ICBHI |
|---|---|
| A0 pretrained + fine-tuned | 0.5602 |
| A2 random init + fine-tuned | 0.4999 |
| A24 random init + **frozen** | 0.4979 |
| A4 **pretrained** + frozen | **0.4595** |

**Frozen ImageNet features score 0.038 BELOW frozen random features — 2.7× the seed range.**
Frozen ImageNet representations are actively worse than random projections on log-mel spectrograms;
the transfer only pays once the backbone can move. This decomposes A4's −0.1007 into "pretraining is
worthless frozen" + "adaptation is what the 0.10 buys" — neither M45 row could produce it alone.

### 7.4 Protocol ablation (M48 Tier E)

Already tabulated in §6.3.

---

## 8. Key findings — what the experiments actually prove

1. **Measurement can dominate modelling.** The full audit is worth 0.1475 on one model, larger than
   any architectural or preprocessing choice measured anywhere in this project. *(Tier E)*
2. **A 2.23 M-parameter CNN beats three vision transformers of 21.7–85.8 M** under one shared
   recipe on this corpus. ViT-B/16 collapses to the majority class in both runs. *(§6.1)*
3. **The transformer era on ICBHI is an audio-pretraining era.** Five convergent measurements, four
   ours. *(§6.1, §7.3, §6.5, plus the published frozen-probe result of 0.5938)*
4. **SpecAugment is backbone-specific and is an operating-point shift**, not a uniform gain:
   +0.0402 on the CNN with Sp +0.098 and **Se −0.018**; ≤0 on all three transformers. *(§6)*
5. **Only one model component is established by this test set** — fine-tuning the backbone, worth
   0.1007, almost entirely in specificity. *(§7.1)*
6. **Frozen ImageNet features are worse than random projections on log-mel input.** *(§7.3)*
7. **Checkpoint selection is a first-class pipeline component**, worth 0.075–0.102 — more than
   pretraining or augmentation. *(§7.3)*
8. **Our physics-derived concept extractors were weak, and the labels were not the constraint.** A
   frozen AudioSet probe reaches 0.71/0.76 where they reached 0.56/0.57. **Ceiling claim retracted.**
9. **Machines reproduce ICBHI's cycle labels better than a trained listener agrees with them** —
   significantly so for wheeze on matched clips (+0.169, p = 0.046). *(§6.5)*
10. **The model hears the sounds; the sounds do not predict the disease labels.** A general-audio
    model encodes 13/14 concepts, yet 1 of 14 carries the bottleneck, 5 are dead, 1 is harmful.
11. **Task adaptation destroys the clinical encoding**, worst on the one diagnostic concept.
12. **The best model does not localise adventitious sounds.** It attends *less* to the wheeze band
    when a wheeze is present, and 59% of its evidence sits on audio the padding manufactured.
13. **Unit of analysis changes six of ten ablation verdicts.** n = 47 patients, not 2,636 cycles.
14. **Accumulating mechanisms did not accumulate performance:** 6 of 8 extensions lost to the plain
    backbone on an easier partition.
15. **Two of our own published-internally claims did not survive replication and were withdrawn**
    (the interpretability cost; the pediatric mechanism verdict).

---

## 9. Limitations and unfinished work

### Ranked by how much they would change the paper

| # | Item | Status | Impact if fixed |
|---|---|---|---|
| 1 | **Checkpoint selection reads the test partition** | Known, priced, stated | **Highest.** The corrected-partition absolute numbers are optimistic. Within-table comparisons, seed band and ablation deltas are unaffected (equally optimistic everywhere) |
| 2 | **M43 (AST) never completed** | `NOT RUN`; notebook written, needs 16 GB | The only audio-pretrained backbone — the one experiment that would test §8.3 directly rather than by convergent inference |
| 3 | **Cumulative ladder rungs S0–S2** | `NOT RUN`, ~55 min GPU | Completes Table 7.2; would price pretraining and fine-tuning cumulatively |
| 4 | **M35 (best extension) only exists at 70/30** | `NOT RUN` on corrected split | The single number that could change the top of the results table |
| 5 | Split effect measured for **one model only** | Known | M2 and M3 changed architecture between partitions, so their −0.1418 / −0.0763 are **confounded and are not quoted as split effects** |
| 6 | Ablation reference not byte-identical to best model | Known, bounded by seed band | 10 of 12 rows come from a separate harness; params differ by ~34 k |
| 7 | Single clinician | Structural | No inter-rater κ computable. Mitigated by intra-rater κ 0.71/0.60 and the exact replication of the published n = 7 sensitivity |
| 8 | Listening pack not a random sample | Known | Absolute AUROCs on the 132 clips are not corpus-comparable; only the paired contrast is |
| 9 | N5 permutation null at n_perm = 5 | Known | The null *mean* is solid; do not quote a null *interval* until n_perm ≥ 20 |
| 10 | N1 LoRA at one rank / 5 epochs | Known | Shows adaptation degrades concepts, not that degradation scales with adaptation strength |
| 11 | One corpus (adult side) | Structural | Scope of the claim is ICBHI and no further |

### Failed / invalidated, and what was done about each

| Run | Problem | Disposition |
|---|---|---|
| M4 (AST) | no committed results file; numbers prose only | **dropped from every table** |
| M15 v6 | reports a score with no confusion matrix | **excluded as unverifiable** |
| M30 | 0.8213 with no matrix, wrong metric, wrong split | **withdrawn** |
| M16, M18, M20, M21 | recording-level, no train/test separation — fitted and scored on the same rows | **excluded**; M21 also shows Se = 0.0186 |
| M33 v1 | Sp = 0.0000 (never classifies a Normal cycle) | reported as a collapse |
| M36 | Se = 0.0932 | reported as a collapse |
| M35-v2 | best epoch 1 of 30 | reported beside M35 |
| M6, M14, M19 | AUROC at or below chance (0.4516 / 0.4809 / 0.3287) | reported as negative results |
| M19 Coswara arm | n = 2 | **unusable; never cite its AUROC** |
| M23 | no committed results, no audit entry | superseded by M41 |
| M38 | never run; n = 19 is structural | **dropped** |
| N6 interpretability cost | single-seed artefact, sign inverts | **withdrawn** |
| N8 mechanism verdict | 4-prediction sign test floors at p = 0.0625 | **test redesigned**, verdict now MIXED |

### Explicitly not authorised (standing stop-list)

No new mechanism, no bottleneck head, no intervention API, **no third G2 extractor revision** (the
test split has been read twice; that budget is spent), no tuning a transformer until it wins, and
**no number in the report that is absent from a committed results file with a raw confusion matrix.**

---

## 10. The final research story

> **Problem.** ICBHI 2017 is the field's default respiratory-sound benchmark, with well over a
> hundred systems built on it, and each is judged by a single headline score.
>
> **Gap.** That score is reported without the two facts needed to interpret it — the partition and
> the metric definition — the partition is described incorrectly in the literature, confusion
> matrices are rarely committed, and interpretability claims are never validated against a
> characterised reference standard.
>
> **Idea.** Audit our own pipeline first. If a team auditing itself this hard made all of these
> mistakes, the argument about the wider literature lands harder unstated.
>
> **Method.** A split loader that raises rather than falls back; a scoring module that emits the
> official metric together with the raw matrix; patient-level intervals and paired tests on every
> comparison; one shared recipe across four pretrained backbones; and a concept-validity gate
> pre-registered with a threshold that could fail.
>
> **Experiments.** 39 committed runs re-audited and 20 rescored; four backbones × clean/augmented on
> the corrected partition; a 12-row leave-one-out ablation, a cumulative ladder, a 3-seed noise
> floor, a 5-row protocol ablation; gradient and occlusion attribution with two quantitative
> measures; a two-run pre-registered concept gate; a supervised ceiling probe; a blinded clinician
> listening study; and a matched machine-versus-physician comparison on the same clips.
>
> **Results.** Five faults, priced: 0.0948 mean metric inflation, 0.0893 split fallback, 0.0039
> patient leak, 1 recording silently dropped, and 0.075–0.102 for checkpoint selection — 0.1475
> combined. A 2.23 M CNN beats three transformers of 21.7–85.8 M. Only fine-tuning is established
> by the data. The best model attends *less* to the wheeze band when a wheeze is present. The
> concept gate failed twice, and a frozen AudioSet probe then reached 0.71/0.76 on the same cycles,
> forcing us to retract our own reading of that failure.
>
> **Contribution.** A measurement, not a mechanism: corrected baselines, a priced fault taxonomy, a
> pre-registered negative result honoured to the letter, two self-retractions, and released tooling
> that detects every fault described.

---

## 11. Evidence and source mapping

### Every headline number → its source file

| Claim | Number | Source |
|---|---|---|
| Best model | 0.5602 [0.508, 0.614] | `Asif's/M22_v2/Results/results_M22_v2.json` |
| Literature-comparable row | 0.5641 | `Asif's/M22_v2/Results/results_M22_v2_official.json` |
| Clean control | 0.5200 | `Asif's/M3_v2/Results/results_M3_v2.json` |
| Transformers | 0.4994–0.5304 | `M40_M43_transformers/Results/results_M4{0,1,2}{,_aug}.json` |
| Metric inflation | mean 0.0948, max 0.2176 | `Asif's/audit/ICBHI_SCORE_AUDIT.{md,json}` |
| Protocol ablation | E0–E5, E1+E2 = 0.7077 | `M48_core_pipeline_ablation/M48_tier_E_table.json` |
| Seed band | sd 0.0075, range 0.0141 | `M48_core_pipeline_ablation/M48_tier_A_table.json` |
| Selection criterion | 0.075–0.102 | same file, `A7_selection_criterion` |
| Negative transfer | A24 = 0.4979 | `M48_core_pipeline_ablation/results_M48_A24.json` |
| Ablation, 12 rows | 0.4595–0.5764 | `Asif's/M45/M45_ablation_table.json` |
| Paired tests | 1 of 10 at patient level | `Asif's/M45/M45_paired_tests.json` |
| Cumulative ladder | 3 of 6 rungs | `M48_core_pipeline_ablation/M48_cumulative_table.json` |
| Gate G2 | run 1 / run 2 | `Asif's/M39/M39_handoff/` and `Asif's/M39/2nd_run_handoff/results_M39.json` |
| 14-concept validation | 0.5556 / 0.5818 | `owmtl_concept_engine/notebooks/01_*/concept_validation_report.json` |
| Supervised ceiling | 0.7115 / 0.7621 | `Novelty Experiment/results/N11_supervised_ceiling.json` |
| Clinician reliability | κ 0.035 / 0.266 | `Novelty Experiment/results/N9_clinician_reliability.json` |
| Machine vs physician, same clips | +0.169, p 0.046 | `M48_core_pipeline_ablation/results_M48_C4.json` |
| Concept probing | 13/14, ΔR² −0.0907 | `Novelty Experiment/results/N1_fm_concept_probing.json` |
| Leakage | 0.2025 bits | `Novelty Experiment/results/N5_leakage_audit.json` |
| Bottleneck non-replication | ordering inverts | `Novelty Experiment/results/N6_physics_bottleneck.json` |
| Intervention | −0.047, 20× bypass | `Novelty Experiment/results/N7_clinician_intervention.json` |
| Pediatric shift | ρ −0.259 etc. | `Novelty Experiment/results/N8_pediatric_fragility.json` |
| Open-set suite | Energy 0.6466 | `Asif's/M29/results_M29.json`, `N2_concept_space_osr.json` |
| Calibration | T = 4.89, 74.4% referral | `Novelty Experiment/results/N4_honest_operating_point.json` |
| Significance tests | p = 0.525 etc. | `Asif's/Statistics/significance_results.json` |
| XAI | −0.0534, r = 0.098, 59.4% | `Asif's/M44/results_M44.json`, `M44_padding_comparison.json` |
| Device confound | 6 diagnoses, 4 patients | `owmtl_concept_engine/notebooks/01_*/device_structure_report.json` |
| Master merge | 39 rows | `Barshon's/M28/results_M28.json` |
| Literature numbers | 21 published systems | `Papers/RESULT_COMPARISON.md` (each with its own source column) |
| Human benchmarks | 0.4777, κ < 0.40 | `Papers/HUMAN_BENCHMARKS.md` |
| Class distribution | all four partitions | **recomputed 2026-08-31** from `~/Desktop/ICBHI_final_database/*.txt` + `Asif's/ICBHI_challenge_train_test.txt` |

### Internal conflicts found while writing this summary, and how each was resolved

| # | Conflict | Resolution |
|---|---|---|
| 1 | `RTK_requirements.md` §6 says "M22 (0.6495) is the current best"; its own §1 table and the consolidation record say M22-v2 (0.5602) | **0.5602.** §6 predates the corrected re-run and was never updated. 0.6495 is a fallback-partition number |
| 2 | Consolidation record lists M3 corrected at 0.5135; the committed JSON says **0.5132** | **0.5132** (`results_M3.json`). 0.5135 is M12_v2's independent re-inference, which differs by a few confusion-matrix cells |
| 3 | Pivot record says "3/126 patients span devices"; the committed report says **4** | **4** (`device_structure_report.json`) — patients 112, 158, 218, 226 |
| 4 | Metric inflation quoted as "+0.11 mean over 14 models" in older documents | **+0.0948 over 20 runs** — the later, larger audit |
| 5 | Consolidation lists M21 (0.4997) as canonical; M28 **excludes** it for having no train/test separation | **Excluded.** The exclusion is the more careful, later judgement, and M21's Se = 0.0186 confirms it |
| 6 | `Model_Training_Reference.md` gives M2 = 0.6138 / M3 = 0.5895 as headline | Those are **fallback-partition** numbers. Corrected: M2 0.4720, M3 0.5132 |
| 7 | "M2 loses 0.1418 / M3 loses 0.0763 to the split" | **Confounded.** The corrected M2 is a 4-block/width-32 network at lr 1e-3 (0.42 M params) while the fallback M2 is 5-block/width-48 at lr 5e-4 (3.63 M); M3 changed from MobileNetV2 to MobileNetV3-Small. **Only the M22 chain (−0.0854, −0.0039) is a one-variable split measurement** |
| 8 | Corpus is 6,898 cycles; our pipeline indexes 6,887 | Explained: `226_1b1_Pl_sc_Meditron` (split file) vs `..._LittC2SE` (disk) — **11 cycles, 6 Normal + 5 Crackle**, train side only. Verified by reading the annotation file |
| 9 | M45 rows report 2,263,160 params; M22-v2 reports 2,228,996 for the same nominal model | **Unresolved, small (34,164).** The two implementations differ slightly; the seed band bounds the consequence. Flagged in the paper's limitations |
| 10 | Human-benchmark paper cited project-wide as "Aviles-Solis et al. 2016" | **Wrong first author. Cite Melbye et al. 2016**, BMJ Open Respir. Res. 3(1):e000136 — verified against the OpenAlex record |
| 11 | README says corrected numbers "sit at the published ICBHI level (~0.60–0.65)" | **Outdated.** Corrected best is 0.5602 / 0.5641, which is the 2021 CNN level, 6–10 points below the frontier |
| 12 | Two different G2 number pairs exist (9-concept gate vs 14-concept engine) | **Never mix them.** The 9-concept M39 runs are the pre-registered gate and are what the paper cites; the 14-concept validation is reported separately and labelled as a different implementation |

---

## 12. Paper guidance

### Emphasise

- **The priced fault table (Tier E) and the E1+E2 = 0.7077 row.** It is the single number stating
  what the whole audit was worth, and it says our unaudited pipeline would have led the comparison
  table.
- **Conceding that fault 3 costs nothing.** An audit that confirms all of its own hypotheses is less
  believable than one that does not.
- **The Se/Sp decomposition against the literature.** Our sensitivity is level with 2024 SOTA; the
  entire deficit is specificity. Under the composite alone this is invisible — it is the concrete
  payoff of the paper's own reporting rule.
- **The retraction.** Naming in advance the outcome that would refute you, and then reporting it, is
  worth more than a result that never risked anything.
- **The seed band and the patient-level test together.** They point in opposite directions and the
  pair is more informative than either.
- **The two self-withdrawals** (interpretability cost; pediatric mechanism verdict).
- **The failure catalogue.** Nine of twelve run failures were caught by an automated check, and
  three of those checks exist because the failure actually happened here.

### Avoid

- ❌ **Never report 0.6495 or 0.7077 as a result.** They are audit exhibits.
- ❌ Never rank across partitions. Every table needs a partition column.
- ❌ Never write "X beats Y" for any open-set AUROC — every patient-level CI crosses chance at n = 19.
- ❌ Never quote M4, M15, M30, M16, M18, M20, M21 or M23 as results.
- ❌ Never present the patient leak as a source of score inflation (it is +0.0039).
- ❌ Never claim clinical interpretability from the attribution maps.
- ❌ Never claim the pediatric mechanism (5/7, p = 0.227) — report the confirming concepts individually.
- ❌ Never compare the 132-clip absolute AUROCs with the 2,636-cycle ones.
- ❌ Never present a cumulative-ladder rung that has not been run.
- ❌ Never cite "Aviles-Solis 2016" — it is Melbye.
- ❌ Never claim the M22-v2-official row (0.5641) as the best model; it is not patient-independent.

### The one sentence to lead with

> Our unaudited pipeline reported 0.7077 on this benchmark, which would have placed first in the
> comparison table; audited, the same model reports 0.5602, and the difference is entirely
> measurement.

---

## 13. File map — where to look for what

| Need | Path |
|---|---|
| The paper | `Final_Draft/main.tex`, `Final_Draft/references.bib` |
| Figures + how to make the missing one | `Final_Draft/FIGURES_SPEC.md`, `Final_Draft/figures/` |
| Conflicts with the earlier draft | `Final_Draft/CONFLICTS_AND_RESOLUTIONS.md` |
| Direction changes | `DECISION_2026-08-16_PIVOT.md`, `DECISION_2026-08-30_CONSOLIDATION.md` |
| Course checklist status | `RTK_requirements.md` |
| Paper format rules | `paper_requirement.md` |
| Per-model history | `Model_Training_Reference.md` |
| Schema / metric / split rules | `Model_Training_Protocol.md` §1–§5 |
| Audit tooling | `Asif's/audit/` (`audit_project.py`, `icbhi_score_audit.py`, `official_split.py`, `check_device_structure.py`) |
| Statistics tooling | `Asif's/Statistics/owmtl_scores.py` |
| Best model | `Asif's/M22_v2/` |
| Ablation | `Asif's/M45/`, `M48_core_pipeline_ablation/` |
| Interpretability | `Asif's/M44/` |
| Transformers | `M40_M43_transformers/` |
| Concept gate | `Asif's/M39/`, `owmtl_concept_engine/` |
| N1–N11 | `Novelty Experiment/` (+ `NOVELTY_STATUS.md`) |
| Literature positioning | `Papers/RESULT_COMPARISON.md`, `Papers/HUMAN_BENCHMARKS.md` |
| Clinician study | `CLINICIAN_LABELING_PACK.md`, `CLINICIAN_RESULTS_v1.md`, `Novelty Experiment/release_annotations/` |
| Superseded runs (kept, never cited) | `Archive_Files (v1)`–`(v4)` |

**Released assets.** `Novelty Experiment/release_annotations/ICBHI_clinician_annotations_v1.csv` —
132 cycles with clinician labels joined to the ICBHI reference, plus a README documenting
provenance, schema, the `ambiguous` deviation, measured reliability and four limitations. To our
knowledge these are the first fine-grained (fine-vs-coarse crackle) reference labels released for
this corpus.
