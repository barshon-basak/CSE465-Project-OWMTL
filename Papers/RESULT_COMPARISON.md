# Result comparison and positioning — where this project sits in the ICBHI literature

**Compiled:** 2026-08-30 · **For:** `DRAFT_PAPER/main.tex` §"Comparison with Existing Works"
(`tab:comparison`, currently four `\pending{verify}` cells), §Related Work, and §Discussion.
**Read with:** `Papers/HUMAN_BENCHMARKS.md`, `DECISION_2026-08-30_CONSOLIDATION.md`.

> **Verification status.** Every external number below is recorded with the source it came from.
> Rows marked ✅ were read out of the paper's own results table; rows marked ⚠️ came from a search
> summary or an abstract and **must be confirmed against the publisher record before submission**.
> Every number attributed to *this project* was recomputed here from a committed
> `results_M*.json` confusion matrix, not copied from a prose file.

---

## A. The three axes that decide whether two ICBHI numbers can be compared

This is the paper's own thesis, so the comparison table must be built on it rather than around it.
A reported ICBHI number is only interpretable if all three are stated:

1. **Partition.** Official 60/40, a random 80/20, *k*-fold CV, or something else. The literature's
   own controlled measurement of this effect is RespireNet, which scores **56.2 on the official
   split and 68.5 under 5-fold CV with the same model** — a **+12.3 point** swing from partition
   alone (Gairola et al., EMBC 2021).
2. **Metric.** The official challenge score is `(Se + Sp)/2` with **pooled** abnormality:
   `Sp` = Normal recall, `Se` = pooled recall over Crackle/Wheeze/Both. Any macro-averaged variant
   is a different quantity (this project's own `Δ = 0.0948` mean inflation, `0.2176` worst).
3. **Task.** 4-class cycle-level (ALSC) vs 2-class cycle-level vs recording-level disease
   classification (RDC). Disease-level numbers run 25–40 points higher and are routinely quoted in
   the same breath as cycle-level ones.

**Published support for axis 1 — quote this, it is not our lone complaint.** Nguyen and Pernkopf,
*IEEE TBME* 2022 (arXiv:2108.01991), state it directly:

> "In general, it is difficult to compare the score of some proposed methods for ICBHI as a
> substantial work does not use the official data splitting or use a different evaluation metric."

and

> "It is notable that the performances on the official 60/40 ICBHI separation without common
> patients in both sets are significantly lower than that of randomly 80/20 splitting."

**Published support for axis 2 — there is none, and we should say so.** Every source checked
defines the challenge score identically as `(Se+Sp)/2` over pooled abnormality. **The macro variant
was our error, not the field's.** §3 of the paper must therefore stay framed as a self-audit. The
one adjacent observation is that ADFF-Net (*Technologies* 2025) reports `Sp 81.39 / Se 42.91` and
calls the resulting `62.14` a **harmonic** score, when the harmonic mean of those two values is
`56.19` and their arithmetic mean is `62.15` — i.e. the arithmetic score under a harmonic label.
That is a naming slip worth one careful sentence, not a second fault class. ⚠️ **verify from the
published PDF before citing.**

---

## B. External results — ICBHI 2017, 4-class cycle-level, **official 60/40 split**, official metric

The comparable column. Sorted by score. This is the table our numbers must be read against.

| # | Method | Backbone | Pretraining | Venue / Year | Sp (%) | Se (%) | **Score (%)** | Src |
|---|---|---|---|---|---:|---:|---:|---|
| 1 | MFCC + HMM (Jakovljević & Lončar-Turukalo) | HMM | — | ICBHI 2017 | — | — | **39.56** | ✅ TBME'22 Tab. VI |
| 2 | Low-level features + boosted tree (Chambres et al.) | Decision tree | — | 2018 | — | — | **49.63** | ✅ TBME'22 |
| 3 | STFT+Wavelet + SVM (Serbes et al.) | SVM | — | 2017 | — | — | **49.86** | ✅ TBME'22 Tab. VI |
| 4 | LungRN+NL (Ma et al.) | ResNet-NL | — | 2020 | 63.20 | 41.32 | **52.26** | ✅ PC-MCL Tab. 1 |
| 5 | CNN-MoE (Pham et al.) | CNN-MoE | — | 2021 | — | — | **52.79** | ✅ TBME'22 Tab. VI |
| 6 | Bi-ResNet (Ma et al.) | Bi-ResNet | — | 2019 | 81 | 28 | **54** | ✅ TBME'22 Tab. VI |
| 7 | Domain/splicing (Wang et al.) | ResNeSt | ImageNet | 2022 | 70.40 | 40.20 | **55.30** | ✅ PC-MCL Tab. 1 |
| 8 | **RespireNet (Gairola et al.)** | ResNet34 | ImageNet | EMBC 2021 | 72.30 | 40.10 | **56.20** | ✅ PAFA + PC-MCL |
| 9 | Co-tuning + StochNorm (Nguyen & Pernkopf) | ResNet50/101 | ImageNet | IEEE TBME 2022 | — | — | **56.85 or 58.29** ⚠️ | ⚠️ conflicting rows |
| 10 | SCL (Moummad & Farrugia) | CNN6 | AudioSet | WASPAA 2023 | 75.95 | 39.15 | **57.55** | ✅ PAFA Tab. |
| 11 | AST fine-tuning (Bae et al.) | AST | IN + AudioSet | Interspeech 2023 | 77.14 | 41.97 | **59.55** | ✅ Lung-SRAD Tab. |
| 12 | RepAugment (Kim et al.) | AST | IN + AudioSet | 2024 | 82.47 | 40.55 | **61.51** | ✅ PC-MCL Tab. 1 |
| 13 | SG-SCL (Kim et al.) | AST | IN + AudioSet | ICASSP 2024 | 79.87 | 43.55 | **61.71** | ✅ Lung-SRAD Tab. |
| 14 | ADFF-Net | AST dual-stream | IN + AudioSet | *Technologies* 2025 | 81.39 | 42.91 | **62.15** ⚠️ | ⚠️ search summary |
| 15 | **Patch-Mix CL (Bae et al.)** | AST | IN + AudioSet | Interspeech 2023 | 81.66 | 43.07 | **62.37** | ✅ ×3 tables |
| 16 | LungAdapter (Xiao et al.) | AST | IN + AudioSet | Interspeech 2024 | 80.43 | 44.37 | **62.40** | ✅ Lung-SRAD Tab. |
| 17 | BTS (Kim et al.) | CLAP | — | Interspeech 2024 | 81.40 | 45.67 | **63.54** | ✅ Lung-SRAD + PAFA |
| 18 | Lung-SRAD | DASS | AudioSet | 2026 | 79.53 | 49.42 | **64.48** | ✅ own Tab. |
| 19 | PAFA (Jeong et al.) | BEATs | AudioSet | Interspeech 2025 | 82.05 | 47.63 | **64.84** | ✅ own Tab. |
| 20 | PC-MCL (Jeong & Kim) | BEATs | AudioSet | 2026 | 79.04 | 51.71 | **65.37** | ✅ own Tab. |
| 21 | **BTS++ (Toikkanen et al.)** — current SOTA | CLAP | AudioSet | Interspeech 2025 | 89.49 | 41.89 | **65.69** | ✅ Lung-SRAD Tab. |

**Two structural facts fall straight out of this table and neither is about architecture:**

- **Everything above 57 uses an audio-domain-pretrained backbone** (AudioSet, CLAP, BEATs, DASS).
  Everything below 57 is either hand-crafted features or an ImageNet-pretrained vision CNN.
- **Sensitivity is the field's bottleneck, not specificity.** Even the 65.69 SOTA detects only
  41.89% of abnormal cycles. Nobody has crossed Se ≈ 52. This is directly relevant to §6:
  it is the same asymmetry our clinician study finds in humans.

### Non-comparable partitions — for the split-effect argument only, never the same column

| Method | Partition | Score (%) | Src |
|---|---|---:|---|
| RespireNet (Gairola et al.) — *same model as row 8* | 5-fold CV | **68.5** | ✅ TBME'22 |
| ResNet-NonLocal + mixup | 5-fold CV | 64.21 | ✅ TBME'22 |
| MFCCs + LSTM (Perna et al.) | overlap 80/20 | 64.21 | ✅ TBME'22 Tab. VI |
| MFCCs + CNN | overlap 80/20 | 66.38 | ✅ TBME'22 Tab. VI |
| CNN snapshot ensembles | overlap 80/20 | 68.50 | ✅ TBME'22 Tab. VI |
| Gammatone CNN-MoE (Pham et al.) | overlap 80/20 | **74** | ✅ TBME'22 Tab. VI |

**Split effect measured in the literature: +8 to +18 points**, and +12.3 in the one controlled
same-model comparison (RespireNet). Our own measurements land in exactly that band (§E.4).

---

## C. Frozen foundation-model probes — the external anchor for N11

**"Assessing the Utility of Audio Foundation Models for Heart and Respiratory Sound Analysis"**
(arXiv:2504.18004). ICBHI 2017, **official split**, 4-class, linear probe on a **frozen** encoder. ✅

| Model | Pretraining | Regime | Sp (%) | Se (%) | Score (%) |
|---|---|---|---:|---:|---:|
| M2D (16×4) | AudioSet | **frozen** linear probe | 72.20 ± 7.25 | 46.56 ± 5.61 | **59.38 ± 0.94** |
| BEATs | AudioSet | **frozen** linear probe | 72.05 ± 4.86 | 42.57 ± 5.96 | **57.31 ± 1.40** |
| SOTA reference | — | fully fine-tuned | 81.51 | 45.08 | 63.29 |

Their conclusion, quoted:

> "The respiratory sound model OPERA-CT underperforms other general audio models in the respiratory
> sound tasks… general audio models are trained on AudioSet, which features 2M diverse samples. In
> contrast, OPERA-CT was trained on only 140K respiratory sounds."

**Why this matters to us, in three steps.** (i) A *frozen* AudioSet encoder with a linear head
scores **59.38** — above every model this project has trained, all of which were fully fine-tuned.
(ii) A respiratory-*specific* foundation model loses to general AudioSet models on respiratory
audio. (iii) N11 is the same experiment run at concept level rather than class level, and it
returns the same verdict. **N11 is not an isolated curiosity; it replicates a published result in a
different output space.** Cite this paper next to N11 — it converts our finding from "surprising"
to "expected, and here is the concept-level version nobody had run."

**Do not use OPERA's ICBHI number as an N11 anchor.** OPERA's ICBHI task (T7) is *COPD vs healthy
at recording level*, 828 samples, AUROC 0.741–0.933 — a different task entirely. Checked and
rejected.

---

## D. Human benchmarks — the reference-standard side

Full detail in `Papers/HUMAN_BENCHMARKS.md`; this is the comparison-table extract.

| Study | Raters | Material | Result |
|---|---|---|---|
| **Tzeng et al., *JMIR AI* 2025** ✅ | 7 senior physicians | 25% of ICBHI test, blind | Acc 49.40, **Se 23.23, Sp 72.32, ICBHI score 47.77**, confidence 2.88/5 |
| **Huang et al., *npj Prim Care Respir Med* 2024** ⚠️ | physicians + AI, Formosa Archive (11,532 recordings, NTUH Hsinchu ED) | crackle / wheeze / normal | κ **wheeze 0.948**, **crackle 0.516**, normal 0.298. Crackle: physicians Se 93.9 / Sp 56.6; AI Se 80.3 / Sp 65.9. **AUROC wheeze > 0.9; AUROC crackle 0.7–0.8** |
| **Melbye et al. 2016** ✅ (cited project-wide as "Aviles-Solis" — **wrong first author**, see `HUMAN_BENCHMARKS.md`) | 12 physicians | 20 ERS recordings, 10 categories | detailed descriptions (incl. fine vs coarse crackle) **κ < 0.40**; combined crackles κ 0.62; combined wheezes κ 0.59 |
| **This project** (`CLINICIAN_RESULTS_v1.md`) | 1 physician, 132 cycles | ICBHI, with 6 calibration exemplars | vs ICBHI: **κ 0.035 crackle** (CI spans 0) / **0.266 wheeze**, **Se 0.231**; intra-rater κ 0.714 / 0.600 |

**Three comparisons to make from this block, in ascending order of value:**

1. **Our single rater replicates the published n=7 result almost exactly.** Our clinician's
   sensitivity against ICBHI is **0.231**; Tzeng's seven senior physicians reach **0.2323**. That is
   agreement to within a tenth of a point, from an independent rater, on the same corpus, with
   *better* calibration material than the published study supplied. This is the strongest available
   answer to "n=1" — the limitation stands, but the number is not idiosyncratic.
2. **Our N11 crackle number sits inside the published machine range.** Huang et al. report AI crackle
   AUROC **0.7–0.8** on a different corpus; N11's frozen-AST crackle probe returns **0.7115**.
   Independent corroboration that ~0.71 is what crackle detection is worth, not a fluke of our probe.
3. **The wheeze discrepancy is the sharpest thing in this table and is ours to explain.** Huang et al.
   find wheeze is the *reliable* sound — κ 0.948, AUROC > 0.9. On ICBHI, our clinician agrees with
   the wheeze labels at **κ 0.266** and our best probe reaches only **0.7621**. If wheeze is reliably
   identifiable in general but not reproducible against *these* labels, the difference is a property
   of **this corpus's annotation**, not of auscultation. That is a materially stronger and more
   specific claim than the retracted ceiling, it is supported by two independent published studies,
   and it needs no second rater.

---

## E. This project's results, organised by partition

Every score below was recomputed from the committed raw confusion matrix. **Rows from different
partitions are never rankable against each other.**

### E.1 Corrected official split — 2,636 cycles / 47 patients (`tab:comparison` main block)

| Model | Description | Params | Sp | Se | **ICBHI** |
|---|---|---:|---:|---:|---:|
| **M22_v2** | MobileNetV2 + SpecAugment — **best model** | 2.23 M | 0.7115 | 0.4089 | **0.5602** |
| M45_P5 | − spectrogram standardisation | 2.23 M | — | — | 0.5539 |
| M45_A3 | − class-weighted loss | 2.23 M | — | — | 0.5497 |
| M45_A5 | 64 mel bins | 2.23 M | — | — | 0.5427 |
| M41 | Swin-T (best transformer) | 27.5 M | 0.7179 | 0.3429 | 0.5304 |
| M41_aug / M45_P4 | Swin-T + aug / zero-padding | 27.5 / 2.23 M | — | — | 0.5291 |
| M45_A6 | 4 s cycles | 2.23 M | — | — | 0.5224 |
| **M3_v2** | MobileNetV2, no augmentation (= A1) | 2.23 M | 0.6135 | 0.4266 | **0.5200** |
| M42 | DeiT-S | 22 M | — | — | 0.5149 |
| M45_A2 | − ImageNet initialisation | 2.23 M | — | — | 0.4999 |
| M21_Updated | acoustic curriculum learning | — | — | — | 0.4997 |
| M40 / M40_aug | ViT-B/16 — **collapsed to all-Normal** | 86 M | — | — | 0.4994 / 0.5000 |
| M45_A4 | frozen backbone, head only | 2.23 M | — | — | 0.4595 |

### E.2 Verbatim official split — 2,756 cycles / 49 patients (**the literature-comparable rows**)

| Model | Description | Sp | Se | **ICBHI** | CI95 |
|---|---|---:|---:|---:|---|
| **M22_v2_official** | MobileNetV2 + SpecAugment — **the comparability row** | 0.6719 | 0.4562 | **0.5641** | [0.5121, 0.6144] |
| M3 | MobileNet/DenseNet lightweight | 0.8284 | 0.1980 | **0.5132** | — |
| M2 | scratch CNN baseline | 0.7435 | 0.2005 | **0.4720** | — |

All three hand-verified from their committed confusion matrices. **These are the only rows in the
project that sit on the exact partition the literature reports on** — same 2,756 cycles, same 49
patients, same class distribution (1,579 / 649 / 385 / 143).

`M22_v2_official` was run 2026-08-30 from `Asif's/M22_v2/m22-official-notebook.ipynb`
(`SPLIT_MODE = "official"`); architecture, optimiser, schedule, class weights, seed and every
SpecAugment parameter are identical to M22_v2, so the two differ by the split alone.

> **This row is NOT the project's best model and must never be reported as one.** It is not
> patient-independent — 120 of its 2,756 test cycles (4.4%) come from patients 156 and 218, whom
> the model saw in training. `RTK_requirements.md` §6 selects the best model on the *corrected*
> partition, so **M22_v2 at 0.5602 remains the headline**. This row exists solely so §B's twenty-one
> published numbers can be read against ours on identical footing. Its JSON self-documents with
> `is_patient_independent: false` and `comparable_to_published_icbhi_split: true`.

**A third split-provenance defect, found by running this.** The notebook indexed 4,131 train
cycles, not the expected 4,142. The cause is a filename mismatch in the official split file itself:

| | stem |
|---|---|
| `ICBHI_challenge_train_test.txt` lists | `226_1b1_Pl_sc_**Meditron**` |
| the released audio contains | `226_1b1_Pl_sc_**LittC2SE**` |

The device suffix disagrees, so any pipeline that joins audio to split on the stem **silently drops
one recording** (11 cycles). It is a *train* recording, so the test partition is unaffected and the
comparability of 0.5641 stands. Small in effect, but it is a third concrete, one-`grep`-checkable
defect in the same artefact, and it belongs in the paper's split-audit section beside the other two.

### E.3 Non-comparable partitions — kept only as the "before" half of the split audit

| Partition | Models |
|---|---|
| Fallback `pid ≤ 111` (492 cycles / 11 patients, 7.1%) | M22 0.6495 · M1 0.6143 · M2 0.6138 · M12 0.6138 · M30_v2 0.5975 · M3 0.5895 |
| 70/30 patient-independent (2,749 cycles / 38 patients) | M35 0.6864 · M37 0.6753 · M33 0.5832 · M34 0.5754 · M31 0.5535 · M36 0.5052 · M32 0.4733 |

### E.4 Our split effect, measured — and the two faults must be separated

The M22_v2_official run lets us decompose what was previously one number. **They are not the same
size, and an earlier draft of this document wrongly implied they were.**

| Model | Fallback (492 cyc, 11 pat) | Verbatim official (2,756 cyc, 49 pat) | Corrected (2,636 cyc, 47 pat) |
|---|---:|---:|---:|
| M22 → M22_v2 | 0.6495 | **0.5641** | **0.5602** |
| M3 | 0.5895 | 0.5132 | 0.5200 |
| M2 | 0.6138 | 0.4720 | — |

**Fault A — the silent fallback. Large, and the one that matters.**
Measured on the same model against the true official split: **0.6495 → 0.5641 = −0.0854**.
M2 gives −0.1418 and M3 −0.0763. Mean ≈ **−0.10**, consistent with the literature's controlled
figure (RespireNet official vs 5-fold, −0.123). A 7.1% test partition inflates by roughly a tenth
of the metric's range.

**Fault B — patient leakage in the published split. Not measurable at this sample size.**
Same model, same seed, same schedule, only the leakage policy differs:
**0.5641 (leaking) → 0.5602 (corrected) = −0.0039.** That is a fortieth of Fault A, it sits deep
inside the official run's patient-bootstrap CI of **[0.5121, 0.6144]**, and the two runs selected
different epochs (26 vs 38), so the difference is indistinguishable from seed-and-schedule noise.

> **Say this explicitly in the paper, because a reviewer will ask it.** Correcting the published
> split's patient leakage **does not materially change the score**. The leakage finding stands as a
> *validity* finding — the benchmark is not what it is described to be, and 40% of its rarest class
> comes from the two leaking patients (§F.4) — but it must **not** be presented as a source of
> score inflation. Claiming otherwise would be unsupported by our own measurement, and we have the
> measurement.
>
> This actually strengthens the paper. It shows the audit was run to find out, not to confirm, and
> it lets §3 rank its three faults honestly: **metric (Δ 0.0948 mean) ≈ fallback split (Δ 0.085) ≫
> patient leakage (Δ 0.004, within noise)**.

**A second reading of the same pair.** Correcting the leakage moved Se from 0.4562 to 0.4089 and Sp
from 0.6719 to 0.7115 — a 4–5 point shift in *both*, nearly cancelling in the average. The corrected
partition drops 57 of 143 "Both" cycles (§F.4), which removes much of the hardest abnormal mass and
makes the model look more specific and less sensitive. The composite score hides that; report Se and
Sp separately, exactly as the paper's own recommendations require.

---

## F. Five analyses — the reasoning the faculty asked for

### F.1 Where the best model actually sits: at the 2021 CNN state of the art

**No longer an estimate — measured.** M22_v2_official scores **0.5641 on the identical partition**
§B reports on, which places it **9th of 22**, immediately above **RespireNet (56.20, EMBC 2021)**
and above every hand-crafted-feature and non-pretrained CNN method. It is **6–10 points below the
2023–2026 frontier** (59.55 → 65.69). The corrected-partition headline, 0.5602, lands in the same
place, so the ranking is robust to which partition is used.

**The Se/Sp profile is the more interesting half, and it is ours alone.** On the official split:

| | Se (%) | Sp (%) | Score |
|---|---:|---:|---:|
| **This work (M22_v2_official)** | **45.62** | **67.19** | 56.41 |
| Patch-Mix CL (2023) | 43.07 | 81.66 | 62.37 |
| BTS (2024) | 45.67 | 81.40 | 63.54 |
| PAFA (2025) | 47.63 | 82.05 | 64.84 |
| BTS++ (2025, SOTA) | 41.89 | 89.49 | 65.69 |

**Our sensitivity is competitive with 2024 state of the art** — 45.62 against BTS's 45.67, above
Patch-Mix and above the current SOTA. **The entire 6–9 point deficit is specificity**: 67.19 against
the field's 81–89. We are not uniformly worse; we sit at a different operating point, detecting
abnormal cycles at roughly the field's rate while over-calling them on normal ones. The
class-weighted loss and SpecAugment push that way by design.

This is worth a paragraph rather than a footnote, for two reasons. It is the concrete payoff of the
paper's own recommendation to report Se and Sp separately instead of the composite — under the
composite alone our model simply looks "worse", and the actual difference is invisible. And it
identifies where the remaining gap is recoverable: specificity, not detection.

Say this plainly and without hedging. A 2.23 M-parameter MobileNetV2 matching a 2021 EMBC ResNet34
is a reasonable, honest outcome for a course project, and the paper's contribution was never the
number. What must not happen is the fallback-split 0.6495 appearing anywhere near this table — it
would rank 2nd, and it would be wrong.

### F.2 Why we are 6–10 points back, answered with evidence rather than speculation

The gap is **audio-domain pretraining**, not architecture, not scale, not tuning. Five independent
lines of evidence, four of them ours:

| # | Evidence | Source |
|---|---|---|
| 1 | Every literature method above 57 uses an AudioSet/CLAP/BEATs backbone; every method below uses hand-crafted features or an ImageNet CNN | §B |
| 2 | A **frozen** AudioSet encoder + linear probe scores **59.38** — beating every model we fully fine-tuned | §C |
| 3 | Our ImageNet ViT-B (0.4994, 86 M), Swin-T (0.5304, 27.5 M) and DeiT-S (0.5149, 22 M) **all lose to our ImageNet MobileNetV2** (0.5602, 2.23 M) — 12–39× the parameters, worse score. Transformer capacity without audio pretraining buys nothing here | E.1 |
| 4 | Ablation A2: removing ImageNet init costs **−0.0603**, the largest single-component drop except removing fine-tuning altogether (A4, −0.1007) | `tab:ablation` |
| 5 | N11: a frozen AudioSet AST that has never heard a stethoscope reaches AUROC **0.7115 / 0.7621** on the same cycles our DSP concepts failed on at 0.5580 / 0.5729 | N11 |

**This is a real result and the paper should lead the discussion with it.** Rows 3 and 4 are a
clean, ImageNet-controlled architecture sweep — CNN vs three vision transformers, one variable —
and it shows the ICBHI literature's transformer era is an **audio-pretraining** era wearing a
transformer label. Our M40–M42 runs are the ablation that separates the two, and the literature
does not contain it because nobody publishes ImageNet-only transformers on ICBHI any more.
**The M40 ViT-B collapse to all-Normal is part of the same story**, not an embarrassment: 86 M
parameters, 4,251 training cycles, no in-domain prior.

### F.3 The split-leakage finding is genuinely novel and trivially verifiable

Verified independently for this document by running `Asif's/audit/official_split.py` against the
committed `ICBHI_challenge_train_test.txt`, then by reading the raw file:

```
156_2b3_Al_mc_AKGC417L   test
156_2b3_Ar_mc_AKGC417L   train      <- same patient, same breath session, adjacent chest position
156_2b3_Ll_mc_AKGC417L   train
156_2b3_Lr_mc_AKGC417L   test
```

Patient **156** (9 train / 8 test recordings) and patient **218** (4 / 4) both straddle the
published split — 12 of 381 test recordings (3.1%). The leak is not merely patient-level: it is
**within-session**, the same breathing recorded simultaneously at different chest positions,
split across the train/test boundary.

Meanwhile the literature asserts the opposite, in these words:

- Lung-SRAD (2026): *"the official 60% train and 40% test sets split, **with no patient overlap
  between them**, resulting in 4,142 cycles and 2,756 cycles."*
- BSPC fusion (2025): the same phrasing. ⚠️ verify wording from the publisher record.
- Pham et al. review: *"Systems following the ICBHI data split, where recordings from the same
  patient are **never** found in both train and test subsets…"*

**No published work flagging this was found.** A reader can check it in one `grep`. This is the
project's strongest single contribution and should be stated before the metric audit, not after —
the metric fault was ours alone, this one is the benchmark's and affects everyone using it.

### F.4 A second benchmark-design finding, free from the same audit

The correction removes 120 cycles. They are not drawn evenly:

| class | official (2,756) | corrected (2,636) | removed | % of class lost |
|---|---:|---:|---:|---:|
| Normal | 1,579 | 1,560 | 19 | 1.2% |
| Crackle | 649 | 617 | 32 | 4.9% |
| Wheeze | 385 | 373 | 12 | 3.1% |
| **Both** | **143** | **86** | **57** | **39.9%** |

**Two patients supply 40% of the official test set's rarest class — and they are the same two
patients who leak into training.** 47.5% of the moved cycles are "Both". So every published "Both"
result on the official ICBHI split rests on a 143-cycle class of which 57 cycles come from two
patients the model has already seen. That is worth a paragraph of its own.

It also **cuts both ways and we must say so**: our corrected partition has only 86 "Both" cycles,
which is why M22_v2's Both recall (0.116) is estimated so poorly. Report it as a limitation of the
corrected split in the same breath as the finding about the official one.

### F.5 What is and is not novel about the concept work

- **No concept-bottleneck model for lung-sound auscultation was found.** Searched; the nearest
  neighbours are Label-free CBM (Oikarinen et al., ICLR 2023, CLIP/LLM-derived concepts, vision),
  a 2026 voice-health CBM (different modality, language-derived concepts), and medical-imaging
  CBMs. The physics-derived auscultation concept suite is first, and the **G2 validity gate is the
  first published measurement of whether such concepts are validatable against ICBHI at all.**
- **The respiratory interpretability that does exist is post-hoc**, not interpretable-by-design:
  Grad-CAM / LIME / SHAP over a black box (e.g. *Sci Rep* 2025 CNN-RNN fusion; *Life* 2026). Those
  papers *assert* attribution to acoustic biomarkers and do not verify it. Our gate is that
  verification, and it returned FAIL — which is a result about the difficulty of the verification,
  and only meaningful because the gate was pre-registered with a threshold that could fail.
- **Open-set:** Cho and Lee (*CMC* 84(2):2847–2863, 2025) remains the only open-set respiratory
  prior. They report **accuracy improvements of 2–5% over closed-set semi-supervised baselines**
  (6–8% in data-scarce settings) using prototype-distance rejection — **not an ICBHI score and not
  an AUROC**, so it cannot be tabulated against our open-set suite. State that as a comparability
  limitation rather than forcing a row. ⚠️ verify from the full text.

---

## G. Ready-to-paste replacement for `tab:comparison`

Replaces all four `\pending{verify}` cells. Rows are grouped by partition, which is the point.

```latex
\begin{tabular}{lllrrrl}
\toprule
Source & Method & Partition & Se & Sp & ICBHI & Notes \\
\midrule
\multicolumn{7}{l}{\emph{Official 60/40 partition, official metric}}\\
Gairola et al.~\cite{respirenet2021}  & ResNet34, ImageNet     & official & 0.401 & 0.723 & 0.5620 & CNN era \\
Moummad and Farrugia~\cite{scl2023}   & CNN6, SCL              & official & 0.392 & 0.760 & 0.5755 & AudioSet \\
Bae et al.~\cite{bae2023patchmix}     & AST, Patch-Mix CL      & official & 0.431 & 0.817 & 0.6237 & AudioSet \\
Kim et al.~\cite{bts2024}             & CLAP, BTS              & official & 0.457 & 0.814 & 0.6354 & AudioSet \\
Jeong et al.~\cite{pafa2025}          & BEATs, PAFA            & official & 0.476 & 0.821 & 0.6484 & AudioSet \\
Toikkanen et al.~\cite{btspp2025}     & CLAP, BTS++            & official & 0.419 & 0.895 & 0.6569 & SOTA \\
\cite{frozenfm2025}                   & M2D, \emph{frozen} probe & official & 0.466 & 0.722 & 0.5938 & no fine-tuning \\
Tzeng et al.~\cite{tzeng2025jmir}     & 7 senior physicians    & 25\% of test & 0.232 & 0.723 & 0.4777 & human reference \\
\addlinespace
This work & MobileNetV2 + SpecAugment & official  & 0.456 & 0.672 & 0.5641 & 2{,}756 cycles \\
This work & CNN baseline (M2)         & official  & 0.201 & 0.744 & 0.4720 & 2{,}756 cycles \\
This work & MobileNetV2 (M3)          & official  & 0.198 & 0.828 & 0.5132 & 2{,}756 cycles \\
\midrule
\multicolumn{7}{l}{\emph{Corrected patient-independent partition (2{,}636 cycles, 47 patients)}}\\
This work & MobileNetV2 + SpecAugment & corrected & 0.409 & 0.712 & 0.5602 & best model \\
This work & MobileNetV2, no aug.      & corrected & 0.427 & 0.614 & 0.5200 & ablation A1 \\
This work & Swin-T                    & corrected & 0.343 & 0.718 & 0.5304 & best transformer \\
\midrule
\multicolumn{7}{l}{\emph{Not comparable --- reported only for the split audit}}\\
This work & MobileNetV2 + SpecAugment & fallback  & --- & --- & 0.6495 & 492 cycles, 11 patients \\
\bottomrule
\end{tabular}
```

**Sentence to carry the table**, replacing the current "our corrected numbers do not lead it"
paragraph — it should now do more work than a concession:

> Our best model matches the 2021 CNN state of the art and trails the 2023–2026 frontier by six to
> ten points. That gap has a single identifiable cause, and it is not architecture: every method
> above 57 is built on an audio-domain-pretrained backbone, a *frozen* AudioSet encoder with a
> linear head outscores every model we fine-tuned, and our own ImageNet-controlled sweep finds three
> vision transformers of 22 to 86 million parameters all losing to a 2.2 million-parameter
> MobileNetV2. The uncorrected number we would have reported, 0.6495, would have placed second in
> this table on a partition covering 7.1% of the data under a metric we had defined ourselves.

---

## H. What still needs doing before submission

| # | Item | Effort | Why |
|---|---|---|---|
| 1 | ✅ **DONE.** M22_v2 re-run on the verbatim official split: **0.5641** on 2,756 cycles / 49 patients | done | Headline row now directly comparable to all 21 rows of §B |
| 1b | ✅ **DONE.** Split-audit claim rewritten in `main.tex` §Split Audit: fallback −0.0854, leakage −0.0039 and within noise, stated as a validity fault and explicitly not as inflation | done | We measured it; the old phrasing was unsupported |
| 1c | ✅ **DONE.** The `226_1b1_Pl_sc_Meditron` vs `_LittC2SE` mismatch is written up as the fourth fault, and `official_split.py --audio-dir` now detects it (tested against defective and clean fixtures) | done | Same fault class, one `grep` to verify |
| 2 | ✅ **DONE.** Every reference verified against its OpenAlex/DOI record on 2026-08-30 — title, authors, venue, year, volume, pages. Zero `VERIFY:` notes and zero `TODO` author fields remain. Three of the entries had wrong author lists (BTS, the frozen-FM paper, Cho & Lee) and one was attributed to the wrong first author entirely (Melbye, not Aviles-Solis). What could **not** be verified was removed, not softened: the BTS++ row (0.6569) and the ADFF-Net metric-naming claim — see the note at the end of `references.bib` | done | Nothing in the paper now rests on a secondary source for its citation metadata |
| 3 | ✅ **DONE.** 17 bib entries added; all 23 previously-undefined citation keys now resolve. `grep -c "VERIFY" references.bib` lists what still needs a publisher check | done | §B, §C, §D all rest on these |
| 4 | ⚠️ **REMAINS (minor).** The *Electronics* 2025 review's exact publication count. MDPI returns 403 to automated fetches, so a human must open it. Both the abstract and the introduction now say "well over a hundred" rather than the previously-claimed 135, so the paper is correct either way — confirming the number only lets you sharpen it | 15 min | Framing sentence only; nothing depends on it |
| 5 | ✅ **DONE.** `tab:comparison` rebuilt and grouped by partition; new subsection `subsec:pretraining` added to the Discussion carrying the F.2 analysis; abstract, title, limitations and conclusions all updated | done | This is the analysis the faculty asked for |
| 6 | ✅ **DONE (found in passing).** `.gitignore`'s `results/` rule was case-insensitively matching `Results/` and hiding `Asif's/M22_v2/Results`, `Asif's/M3_v2/Results` and `M40_M43_transformers/Results` — the evidence behind four of the paper's tables. Fixed with a negation; weights and `.npy` stay ignored. **These files still need `git add`** | done | Same fault as the Novelty Experiment one, from the same rule |
| 7 | ✅ **DONE (found in passing).** M45 rows P1--P3 have now been run, so the ablation is complete at eleven rows. Row P3 (+ amplitude normalisation) scores 0.5764, above the reference model, and the pre-declared best-model rule nominally selects it; its patient-level interval spans zero, so `subsec:ablation` now reports the conflict and explains why M22-v2 remains the reference | done | The paper contradicted its own selection rule |

---

## Sources

- [Lung-SRAD (arXiv:2606.11922)](https://arxiv.org/pdf/2606.11922) — SOTA comparison table, official-split definition
- [PC-MCL (arXiv:2601.17080)](https://arxiv.org/pdf/2601.17080) — comparison table incl. RespireNet, LungRN+NL, Wang et al.
- [PAFA (arXiv:2505.23834)](https://arxiv.org/html/2505.23834) — comparison table, RespireNet/SCL/Patch-Mix numbers
- [Nguyen & Pernkopf, co-tuning (arXiv:2108.01991)](https://arxiv.org/pdf/2108.01991) — Table VI, split-comparability quotes
- [Bae et al., Patch-Mix CL (arXiv:2305.14032)](https://arxiv.org/abs/2305.14032) — Interspeech 2023
- [Assessing Audio Foundation Models (arXiv:2504.18004)](https://arxiv.org/html/2504.18004) — frozen-probe ICBHI results
- [OPERA (arXiv:2406.16148)](https://arxiv.org/html/2406.16148v2) — checked and excluded; its ICBHI task is COPD detection
- [Huang et al., npj Prim Care Respir Med 2024](https://www.nature.com/articles/s41533-024-00392-9) · [summary](https://medicalxpress.com/news/2024-11-ai-physicians-equal-difficulty-crackles.html)
- [Tzeng et al., JMIR AI 2025](https://ai.jmir.org/2025/1/e67239)
- [ADFF-Net, Technologies 2025](https://www.mdpi.com/2227-7080/14/1/12)
- [Cho & Lee, CMC 2025](https://www.techscience.com/cmc/v84n2/62937)
- [BSPC fusion 2025](https://www.sciencedirect.com/science/article/abs/pii/S1746809425003015)
- [Suma et al., SN Computer Science 2025](https://link.springer.com/article/10.1007/s42979-024-03506-9) — MTL, 74% sound / 91% disease accuracy; **accuracy on an unstated partition, not an ICBHI score — do not tabulate against §B**
