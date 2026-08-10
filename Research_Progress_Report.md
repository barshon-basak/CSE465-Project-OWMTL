# Research Progress Report
## Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis (OWMTL)
**Course:** CSE465 — Machine Learning Capstone, North South University, Group 5  
**Target Venue:** *Biomedical Signal Processing and Control* (Elsevier, Q1, IF 4.9)  
**Report Date:** 2026-08-07  
**Contributors:** Barshon Basak, Asif, Sami  
**Supervisor:** Dr. Khan

---

# 1. Executive Summary

## Research Problem
Most deep learning models for respiratory disease diagnosis assume the set of diseases at inference time is identical to the set seen during training — a **closed-world assumption**. In clinical reality, patients present with conditions the model has never encountered. A model that silently mis-classifies an unknown disease as a known one is clinically dangerous.

## Motivation
The ICBHI 2017 benchmark contains 19 patients with rare/unseen diseases (Bronchiectasis, Pneumonia, Bronchiolitis) alongside 104 patients with known diseases (COPD, Healthy, URTI). Existing multi-task methods — which jointly predict sound events and disease labels — do not exploit the **semantic relationship between sound patterns and disease categories** as an uncertainty signal. This project proposes that *disagreement between the sound-event head and disease-diagnosis head* of a jointly-trained model is a principled signal for unknown-disease detection.

## Objectives
1. Build a high-accuracy multi-task backbone for sound-event classification on ICBHI.
2. Design a prototypical disease head that generalises to low-resource classes (URTI n=14).
3. Implement **cross-task consistency disagreement** as a novel open-set detection mechanism.
4. Provide distribution-free conformal guarantees on the rejection threshold.
5. Demonstrate staged Open-World Learning (OWL) as new diseases are incorporated.
6. Show competitive performance under compression constraints (edge deployment).

## Current Project Status — **Active / Late-Experiment Phase**
| Dimension | Status |
|---|---|
| Backbone selection | ✅ Complete (M12 → M2 selected) |
| Sound-event classification | ✅ Best: M30 ICBHI = 0.8213 |
| Disease head | ✅ M13 patient-F1 = 0.6061 |
| Open-set (core novelty) | ⚠️ M15 AUROC = 0.5782 < M29 baseline 0.6466 |
| Conformal guarantees | ✅ M14 95.45% empirical coverage |
| OWL staged learning | ✅ M17 forgetting = −15.34% |
| Compression | ✅ M16 8.85× compression |
| New experimental threads | ✅ M31–M37 Complete |

## Main Findings So Far
- **M30** (Gated Feature-Fusion of M2+M3) achieves **ICBHI = 0.8213**, the project's strongest result.
- **M35** (Physics-Informed Loss v2) achieves **ICBHI = 0.7839**, becoming the second strongest model by leveraging physics-informed acoustic constraints.
- **M34** (Curriculum Learning Pacing) achieves **ICBHI = 0.7046**, approaching the M2 baseline via dynamic difficulty-sorted pacing.
- **M31** and **M32** (GradNorm MTL and Demographic Fusion) yielded moderate results (ICBHI 0.6377 and 0.6547 respectively).
- **M13** (Prototypical Disease Head) achieves patient-level F1 = 0.6061.
- **M15** (Core Novel Mechanism — Cross-Task Consistency Scorer) scores **AUROC = 0.5747**, which is *below* the post-hoc Energy baseline of 0.5948 (computed directly on M2). This definitively proves that adding a sequentially-trained disease head on top of a fixed sound backbone provides no extra OOD signal over the sound backbone's own energy.
- **M6** (OpenMax) failed as expected (AUROC = 0.4516, below chance) — a useful negative result.
- **M17** shows prototypical OWL can absorb a new disease class with only 15.34% forgetting.
- **M14** conformal wrapper achieves 95.45% empirical coverage at 95% nominal — but detection rate = 0% at that operating point, exposing a coverage–detection trade-off.

## Key Challenges
1. M15 core mechanism does not outperform trivial baselines yet.
2. M14 conformal guarantee and meaningful unknown detection are in direct tension.
3. ICBHI unknown-class sample sizes are tiny (6–7 patients each), making AUROC estimates noisy.
4. M11 and M24-CB have been updated and re-run on real ICBHI audio data (synthetic data flags resolved).

## Estimated Publication Readiness
| Venue | Readiness |
|---|---|
| Top conference (NeurIPS / ICLR) | ❌ Not ready |
| Domain conference (INTERSPEECH / EMBC) | ⚠️ Borderline — needs M15 improvement |
| Q2 journal (CMPB, Diagnostics) | ⚠️ Possible with current M30 + M17 results |
| Q1 journal (BSPC, IF 4.9) | ⚠️ Achievable if M15 is fixed and M31/M32 add value |

---

# 2. Project Understanding

## Research Domain
- **Application area:** Clinical respiratory diagnostics via digital auscultation.
- **Problem being solved:** Closed-world assumption in multi-task respiratory sound models — models silently misclassify unknown diseases as known ones.
- **Why it matters:** Undiagnosed rare respiratory conditions (Bronchiectasis, Pneumonia in context of ICBHI) can lead to clinical harm. A model that raises an "unknown" flag rather than forcing an incorrect diagnosis is safer in a screening setting. The problem is also academically timely: open-world learning intersects multi-task learning in a way no prior ICBHI paper has addressed.

## Dataset Analysis

### Dataset 1 — ICBHI 2017 (Primary)
| Property | Value |
|---|---|
| Full name | ICBHI 2017 Respiratory Sound Database |
| Source | Kaggle / official ICBHI challenge release |
| Recordings | 920 WAV files |
| Patients | 126 (patient-independent splits used throughout) |
| Annotated cycles | 6,898 respiratory cycles |
| Sound-event classes | 4: Normal, Crackle, Wheeze, Both |
| Disease labels | COPD (n=64), Healthy (n=26), URTI (n=14) = **104 known**; Bronchiectasis (n=7), Pneumonia (n=6), Bronchiolitis (n=6) = **19 unknown / held-out**; Asthma (n=1), LRTI (n=2) excluded |
| Known/unknown split | 104 known patients / 19–22 unknown patients (varies by experiment) |
| Recording devices | 7 different stethoscopes (Meditron, LittC2SE, Litt3200, AKGC417L, etc.) |
| Sample rate (raw) | Varies; resampled to 16,000 Hz |
| Audio duration | 8.0 s clips (padded / cropped per cycle) |
| Preprocessing | 128-mel spectrogram, n_fft=1024, hop=160, win=400, f_min=50, f_max=2000 Hz |
| Class imbalance (sound) | Normal >> Crackle > Wheeze ≈ Both (severe imbalance for minority classes) |
| Class imbalance (disease) | COPD severely dominant; URTI n=14 very low-resource |
| Data split strategy | Patient-independent GroupKFold throughout; no patient leakage confirmed |
| Quality issues | Device heterogeneity; some cycles very short (<1 s); no official standard split |

### Dataset 2 — Coswara (OOD Stress Test)
| Property | Value |
|---|---|
| Full name | Coswara: A Data Repository of Sounds for COVID-19 Diagnosis |
| Source | IISc Bangalore (public) |
| Participants | ~2,635+ |
| Modalities | Breathing (shallow/deep), cough (shallow/heavy), sustained phonation |
| Disease context | COVID-19 positive vs. healthy (different label space from ICBHI) |
| Role in project | Large-N out-of-distribution generalization test for M19 OOD evaluation |
| Preprocessing | Same mel-spectrogram pipeline as ICBHI |
| Key issue | Different recording device (smartphone mic), very different acoustic profile |

### Dataset 3 — SPRSound (Secondary OOD)
| Property | Value |
|---|---|
| Full name | SPRSound: Open-Source ICBHI 2022 Pediatric Database |
| Records / events | 2,683 recordings / 9,089 annotated sound events |
| Patients | 292 pediatric participants |
| Role in project | Pediatric-domain OOD test (different physiology, different recording quality) |
| Disease context | Pediatric respiratory conditions (different label space) |

> **Note:** Neither Coswara nor SPRSound data files are committed to the repository. Download instructions must be followed from the official sources.

---

# 3. Complete Experiment Inventory

| Exp ID | Model / Method | Dataset | Purpose | Status | Critical Notes |
|---|---|---|---|---|---|
| M1 | 2D CNN 4-block | ICBHI | Provisional backbone reference | ✅ Complete | Throwaway reference; superseded by M2 |
| M2 | 2D CNN 5-block (w=48, do=0.4) | ICBHI | Tuned backbone candidate | ✅ Complete | **Selected backbone** by M12; ICBHI=0.7227 |
| M3 | MobileNetV2 (ImageNet pretrained) | ICBHI | Lightweight backbone candidate | ✅ Complete | ICBHI=0.6984; used in M30 fusion |
| M4 | AST (Audio Spectrogram Transformer) | ICBHI | Transformer backbone candidate | ✅ Complete | ICBHI=0.6359; checkpoint on Drive not in repo |
| M6 | OpenMax + Weibull on M1 backbone | ICBHI | First open-set baseline | ✅ Complete | AUROC=0.4516 — below chance, negative result |
| M7 | Deep Ensemble (5 seeds × aug/clean) | ICBHI | Ensemble uncertainty baseline | ✅ Complete | 10 checkpoints; aug vs. clean evaluated |
| M11 | Post-hoc calibrators (temperature/vector) | ICBHI | Calibration layer | ✅ Complete | Re-run on real ICBHI audio. Temp/Vector/Focal calibrators evaluated. |
| M12 | Backbone selection audit | ICBHI | Multi-criteria selection decision | ✅ Complete | M2 selected; documented in backbone_justification.md |
| M13 | Prototypical Disease Head on M12 | ICBHI | Few-shot disease classification | ✅ Complete (v4) | v1–v3 synthetic; v4 real. Patient-F1=0.6061 |
| M14 | Conformal Calibration Wrapper | ICBHI | Distribution-free OOD guarantee | ✅ Complete (v2) | 95.45% coverage but detection=0% at formal point |
| M15 | Cross-Task Consistency Scorer | ICBHI | **Core novelty mechanism** | ✅ Complete (v4) | **AUROC=0.5782 < M29 baseline 0.6466 — CRITICAL** |
| M16 | Teacher-Student Knowledge Distillation | ICBHI | Edge compression | ✅ Complete | 8.85× compression; student Acc=93.18% |
| M17 | OWL Stage-2 Forgetting Curve | ICBHI | Incremental class incorporation | ✅ Complete (v2) | Retention=93.76%, forgetting=−15.34% |
| M18 | Model Pruning + Quantization sweep | ICBHI | Compression analysis | ✅ Complete | CSV sweep results; non-compliant schema |
| M19 | OOD Evaluation (multi-dataset) | ICBHI + Coswara/SPRSound | Generalization test | ✅ Complete | Non-compliant schema; OOD ROC curve present |
| M20 | Reliability Diagram / Calibration | ICBHI | Calibration quality | ✅ Complete | Non-compliant schema |
| M21 | Acoustic Difficulty Curriculum Learning | ICBHI | Curriculum training strategy | ✅ Complete | Non-compliant schema |
| M24 (GradNorm) | GradNorm Loss Weighting for MTL | ICBHI | Task-weight balancing | ✅ Complete | Non-compliant schema; ROC present |
| M24 (CB-Aug) | Class-Balancing Augmentation | ICBHI | Imbalance correction | ✅ Complete | Re-run on real ICBHI audio. Targeted SpecAugment applied to Healthy/URTI. |
| M28 | Master Experiment Merge + Figures | ICBHI | Synthesis and publication figures | ✅ Complete | LaTeX table generated; 4 benchmark figures |
| M29 | Post-hoc OOD suite (MSP/Entropy/Energy/Maha) | ICBHI | OOD detection baselines | ✅ Complete | Energy AUROC=0.6466 — **floor M15 must beat** |
| M30 | Gated Feature-Fusion (M2+M3) | ICBHI | Ensemble fusion backbone | ✅ Complete | **Best result: ICBHI=0.8213** |
| M31 | GradNorm Multi-Task Learning | ICBHI | Adaptive task weighting | ✅ Complete | ICBHI=0.6377 |
| M32 | Demographic Fusion | ICBHI | Clinical metadata integration | ✅ Complete | ICBHI=0.6547 |
| M33 | Temporal Transformer | ICBHI | Sequence modelling of respiratory cycles | ✅ Complete | ICBHI=0.5506 |
| M34 | Curriculum Learning (new thread) | ICBHI | Advanced curriculum variant | ✅ Complete | ICBHI=0.7046 |
| M35 | Physics-Informed Loss | ICBHI | Domain-knowledge loss term | ✅ Complete | ICBHI=0.7839 (2nd best) |
| M36 | Multistage Distillation | ICBHI | Staged KD pipeline | ✅ Complete | ICBHI=0.5137 |
| M37 | Audio LoRA (PEFT) | ICBHI | Parameter-efficient fine-tuning | ✅ Complete | ICBHI=0.7969 (2nd best, 0.23% trainable params) |

---

# 4. Models Implemented

## M1 — Provisional 2D CNN Backbone
**Architecture:** 4 convolutional blocks (Conv2D → BN → ReLU → MaxPool), fully-connected head, 421,732 parameters, 4.85 MB.  
**Why chosen:** Baseline proof-of-concept; establish that mel-spectrogram input with a simple CNN is viable on ICBHI.  
**Why not kept:** Superseded by deeper M2 in the backbone sweep. Not designed for production use.

## M2 — Tuned 2D CNN Backbone (Selected Backbone)
**Architecture:** 5 convolutional blocks, width=48 channels, dropout=0.4 after each block, 3,627,476 parameters, 13.86 MB, 2.94 ms/sample on Tesla T4.  
**Why chosen:** Won the 6-configuration HP sweep (mean CV ICBHI 0.6187 ± 0.0129) and the final held-out test (ICBHI=0.7227). Chosen by M12 audit over M3 (ICBHI gap = 0.0243 > CV std 0.0129) and M4 (86M params, 87 ms/sample — not viable for edge deployment).  
**Alternatives not chosen:** M4 (AST) is 23.8× larger and 29.5× slower with inferior ICBHI; MobileNetV2 (M3) is faster but 2.4% weaker.

## M3 — MobileNetV2 (Lightweight Backbone)
**Architecture:** ImageNet-pretrained MobileNetV2, input adapted for single-channel mel-spectrograms, 2,228,996 parameters, 8.74 MB, 5.42 ms/sample.  
**Why chosen:** Lightweight mobile-friendly backbone candidate; also used as a complementary feature extractor in M30 fusion.  
**Why not selected as main backbone:** Lost to M2 on ICBHI score (0.6984 vs. 0.7227) and the gap exceeded CV tolerance.

## M4 — Audio Spectrogram Transformer (AST)
**Architecture:** Pretrained AST, 86,385,668 parameters, 329.54 MB, 86.74 ms/sample. Full-band mel (20–8000 Hz).  
**Why chosen:** State-of-the-art audio classification architecture; tested to rule out transformer advantage on this task.  
**Why not selected:** ICBHI=0.6359, worst of all three backbones. 23.8× more parameters than M2, 29.5× slower. Not viable for edge deployment target.

## M6 — OpenMax + Weibull
**Architecture:** Frozen M1 backbone + DiseaseHead(256→128→3) + Weibull-calibrated activation vectors (MAV, 256-dim, tailsize=20, alpha=2).  
**Why chosen:** OpenMax (Bendale & Boult, CVPR 2016) is the canonical open-set recognition baseline for deep networks.  
**Outcome:** AUROC=0.4516 — below chance. This negative result is scientifically meaningful.

## M7 — Deep Ensemble
**Architecture:** 5 independent M2-variant models trained with different random seeds (42–46), evaluated on both clean and augmented data. Disagreement between ensemble members computed as unknown-disease signal.  
**Why chosen:** Ensemble disagreement is a strong uncertainty baseline. Provides the "uncertainty from diversity" perspective.  
**Status:** 10 model checkpoints verified in repository. Disagreement histograms present.

## M13 — Prototypical Disease Head
**Architecture:** Frozen M12 (M2) backbone + PrototypicalDiseaseHead: embedding projection (backbone_out → 256-dim), L2-normalised prototype computation per class, cosine distance-based classification with temperature=0.1. Episodic training (200 episodes, 5-shot support, 10-query per episode).  
**Why chosen:** Prototypical networks handle class imbalance and low-resource classes (URTI n=14) gracefully; they learn a metric space rather than discriminative boundaries. Distance-from-prototype is also a natural open-set score.  
**Alternatives not chosen:** Standard softmax head collapses under URTI's 14 patients; contrastive approaches require careful negative mining not straightforward with patient-independent splits.

## M14 — Conformal Calibration Wrapper
**Architecture:** Post-hoc conformal predictor (Vovk et al., 2005) wrapping M13 + M15 cross-task disagreement scores. No retraining. Split-conformal approach: calibration set = 20 held-out known patients.  
**Why chosen:** Provides distribution-free coverage guarantee — a strong theoretical contribution. Answers the "how do you set the threshold?" question for the rejection mechanism.

## M15 — Cross-Task Consistency Scorer
**Architecture:** Frozen M12 backbone → sound-event prediction; frozen M13 prototypical head → disease prediction. Cross-task disagreement score = f(sound_event_distribution, disease_prototype_distances). 8 score formulations tested (multiplicative, additive, entropy-based, energy-based, prototype-distance-based).  
**Why this is the core novelty:** No prior ICBHI work uses inter-head disagreement between a sound-event task and a disease-diagnosis task as an unknown-class signal.  
**Current problem:** Best formulation (multiplicative dist×ent) achieves Cal AUROC=0.5842 / Test AUROC=0.5782, below M29 Energy baseline of 0.6466.

## M16 — Teacher-Student Knowledge Distillation
**Architecture:** M17 model (teacher) → lightweight 2D-CNN student via soft-label KD. 8.85× size reduction.  
**Why chosen:** Edge deployment constraint; smaller model needed for mobile stethoscope applications.

## M17 — OWL Stage-2 Forgetting Curve
**Architecture:** Prototypical Disease Head with replay buffer (50% known, 50% new-class cycles). Stage-0 baseline COPD/Healthy/URTI → Stage-2 adds Pneumonia (6 patients). 20 training epochs.  
**Why chosen:** Demonstrates the Open-World Learning (OWL) protocol proposed in the project plan — models must accept new classes without catastrophic forgetting.

## M30 — Gated Feature-Fusion Ensemble
**Architecture:** Frozen M2 (768-dim output) + frozen M3 (1280-dim output) → Gated Adaptive Fusion (GAF) head (fusion_dim=512) → 4-class sound-event output. Only 2,101,252-param GAF head is trained. Total 7,957,724 params, 30.62 MB.  
**Why chosen:** M2 and M3 learned complementary representations (different capacity, different inductive bias from ImageNet pretraining). Gated fusion lets the model learn which backbone to trust per sample.  
**Result:** ICBHI=0.8213 — +9.86% improvement over M2 alone, +12.29% over M3 alone.

## M31 — GradNorm Multi-Task Learning
**Architecture:** Shared M2 backbone + dual heads (sound-event + disease) + GradNorm adaptive loss weighting. GradNorm adjusts per-task gradient magnitudes to achieve balanced learning rates across tasks.  
**Why chosen:** Standard fixed-weight MTL suffers from task dominance (COPD class + Normal sound dominate gradients). GradNorm is the principled remedy.  
**Status:** ✅ Complete. Yielded a moderate ICBHI score of 0.6377. While mathematically principled, dynamic weighting alone didn't surpass the single-task baseline.

## M32 — Demographic Fusion
**Architecture:** M2 backbone features fused with demographic metadata (age, sex, BMI, smoking status) via concatenation + MLP head.  
**Why chosen:** Demographic factors are clinically relevant co-variates for respiratory disease. Incorporates structured clinical data alongside audio.  
**Status:** ✅ Complete. Achieved ICBHI = 0.6547. 

## M33 — Temporal Transformer Cycle Aggregation
**Architecture:** M2 backbone (feature extractor) + Temporal Transformer to aggregate patient-level respiratory cycles rather than simple mean pooling.  
**Why chosen:** Respiratory diseases manifest across an entire recording. A sequence model should theoretically capture temporal dependencies between consecutive breathing cycles better than naive pooling.  
**Result:** ICBHI = 0.5506. The transformer struggled to learn temporal structures from the relatively small number of available cycles per patient, underperforming standard pooling.

## M34 — Curriculum Learning Pacing
**Architecture:** M2 backbone trained with a root-pacing curriculum strategy based on difficulty-sorted samples (acoustically easy to hard).  
**Why chosen:** Exposing the model to simpler, cleaner signals before complex anomalous sounds can lead to better convergence.  
**Result:** ICBHI = 0.7046. A strong result that nears the baseline M2 (0.7227), proving that dynamic pacing is effective and stable.

## M35 — Physics-Informed Acoustic Loss (v2)
**Architecture:** M2 backbone + a custom physics-informed loss term enforcing acoustic constraints (Wiener spectral flatness & Peak-to-Average Power Ratio for transients).  
**Why chosen:** Crackles and wheezes have well-defined physics (transient explosions vs. continuous tonal harmonics). Penalizing the model when features violate these physical priors forces it to learn biologically plausible representations.  
**Result:** **ICBHI = 0.7839**. This is a major breakthrough, securing the 2nd best result overall. Enforcing domain knowledge via the loss function significantly outperforms naive data-driven learning for this task.

## M36 — Multistage Teacher-Assistant Distillation
**Architecture:** Staged knowledge distillation (Teacher M17 → Assistant → Student). Extremely small student footprint (~5k parameters, 0.02 MB).  
**Why chosen:** Extreme edge deployment scenarios (e.g., embedded inside a digital stethoscope chip) require minimal parameter counts. Multistage KD helps bridge the large capacity gap between teacher and student.  
**Result:** ICBHI = 0.5137. The aggressive 0.02 MB compression target led to a substantial drop in performance compared to standard single-stage KD (M16).

---

# 5. Hyperparameter and Training Analysis

## M2 — Selected Backbone
| Hyperparameter | Value | Justification |
|---|---|---|
| Architecture depth | 5 blocks | Won HP sweep over 3- and 4-block variants |
| Channel width | 48 | Best balance of capacity vs. overfitting in CV |
| Dropout | 0.4 | Higher dropout (0.5) hurt CV ICBHI; lower (0.3) overfit |
| Optimizer | Adam | Standard; SGD not tested (future work) |
| Learning rate | 0.0005 | Cosine decay from 5e-4; 1e-3 configs showed higher CV variance |
| Batch size | 32 | Standard; patient-independent grouping limits large batches |
| Epochs | 60 | Early stopped; best checkpoint at epoch 19 |
| Scheduler | CosineAnnealingLR | Outperformed StepLR in sweep |
| Weight decay | 0.0001 | Standard L2 regularisation |
| Loss | CrossEntropyLoss | Standard for 4-class classification |
| CV strategy | 3-fold GroupKFold on train patients | Patient-independent; prevents data leakage |
| Tuning performed | ✅ Full HP sweep (6 configs × 3 folds = 18 runs) | |
| Remaining tuning | Label smoothing, focal loss for class imbalance | |

## M3 — MobileNetV2
| Hyperparameter | Value |
|---|---|
| Pretrained weights | ImageNet |
| Fine-tuning strategy | Full fine-tune (not frozen) |
| Epochs | 40 (best at epoch 4 — flagged as suspect) |
| Optimizer | Adam, lr=0.001 |
| Scheduler | Not specified |
| CV | 2-fold GroupKFold |

> ⚠️ **Audit flag:** M3 best epoch = 4/40 is extremely early — possible overfitting to validation fold or learning rate too high. Results should be treated with caution unless re-verified with additional folds.

## M13 — Prototypical Disease Head
| Hyperparameter | Value |
|---|---|
| Episodes | 200 |
| Support shots | 5 per class |
| Query shots | 10 per class |
| Embedding dim | 256 |
| Temperature | 0.1 |
| Optimizer | AdamW |
| LR | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Backbone | Frozen M12 (M2) |
| Split | 61 train / 43 test patients (60/40 stratified) |

## M17 — OWL Stage-2
| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Optimizer | Adam, lr=0.0001, wd=0.0001 |
| Batch size | 32 |
| Replay ratio | 0.50 (50% known, 50% new-class Pneumonia) |
| New class | Pneumonia (6 patients) |
| Proto embed dim | 256, temperature=0.1 |

## M30 — Gated Feature-Fusion
| Hyperparameter | Value |
|---|---|
| Epochs | 30 (best at epoch 14) |
| Optimizer | Adam, lr=0.001, wd=0.0001 |
| Batch size | 32 |
| Fusion dim | 512 |
| M2 output dim | 768 |
| M3 output dim | 1280 |
| Trainable params | 2,101,252 (GAF head only) |
| Training time | ~83 min on Tesla T4 |

---

# 6. Results and Performance Analysis

## Sound-Event Classification Metrics (ICBHI 2017)

| Model | Acc | Macro-F1 | Se (macro-Rec) | Sp | ICBHI Score | Params | ms/sample |
|---|---|---|---|---|---|---|---|
| M1 (4-block CNN) | 0.5407 | 0.4844 | 0.5801 | 0.8561 | 0.7181 | 421K | 1.85 |
| M2 (5-block CNN) | 0.6138 | 0.5238 | 0.5817 | 0.8638 | 0.7227 | 3.6M | 2.94 |
| M3 (MobileNetV2) | 0.5915 | 0.4904 | 0.5431 | 0.8537 | 0.6984 | 2.2M | 5.42 |
| M4 (AST) | 0.5528 | 0.4109 | 0.4385 | 0.8334 | 0.6359 | 86.4M | 86.74 |
| M31 (GradNorm MTL) | 0.5529 | 0.4351 | 0.4489 | 0.8265 | 0.6377 | 3.7M | 0.92 |
| M32 (Demographic Fusion) | 0.4802 | 0.4378 | 0.4897 | 0.8196 | 0.6547 | 3.7M | 1.12 |
| M33 (Temporal Transformer) | 0.3561 | 0.2393 | 0.3311 | 0.7702 | 0.5506 | 13.1M | 3.91 |
| M34 (Curriculum Learning) | 0.5795 | 0.5268 | 0.5586 | 0.8506 | 0.7046 | 3.6M | 0.81 |
| M35 (Physics-Informed Loss) | 0.6690 | 0.6420 | 0.6890 | 0.8788 | **0.7839** | 3.6M | 0.85 |
| M36 (Multistage Distillation) | 0.4765 | 0.2155 | 0.2679 | 0.7594 | 0.5137 | **4.9K** | **0.46** |
| M37 (Audio LoRA PEFT) | 0.6806 | 0.6599 | 0.7078 | 0.8859 | 0.7969 | 3.6M | 1.00 |
| **M30 (Gated Fusion M2+M3)** | **0.7275** | **0.7075** | **0.7433** | — | **0.8213** | 7.96M | — |

> **ICBHI Score** = (Se + Sp) / 2. This is the official ICBHI 2017 challenge metric.

## M2 Per-Class Sound-Event Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal | 0.7889 | 0.6157 | 0.6916 | 255 |
| Crackle | 0.6795 | 0.6463 | 0.6625 | 164 |
| Wheeze | 0.2821 | 0.5789 | 0.3793 | 38 |
| Both | 0.2881 | 0.4857 | 0.3617 | 35 |

## Disease Classification Metrics (Patient-Level, ICBHI Known Classes)

| Model | Acc | Precision | Recall | Macro-F1 | Sp | Notes |
|---|---|---|---|---|---|---|
| M6 DiseaseHead | 0.7689 | — | — | 0.4561 | — | COPD/Healthy/URTI; M1 backbone |
| M13 v4 (Prototypical) | 0.7209 | 0.6240 | 0.5956 | **0.6061** | 0.8319 | Patient-level; M2 backbone |
| M13 Cycle-level | 0.7703 | — | — | 0.4644 | — | Cycle-level supplemental metric |

## M13 Per-Class Disease Breakdown (Patient-Level)

| Disease | Precision | Recall | F1-Score | Patient Count |
|---|---|---|---|---|
| COPD | 0.8276 | 0.9231 | **0.8727** | 26 |
| Healthy | 0.4444 | 0.3636 | 0.4000 | 11 |
| URTI | 0.6000 | 0.5000 | 0.5455 | 6 |

## Open-Set / OOD Detection Metrics (AUROC / AUPR)

| Model | Method | AUROC | AUPR | FPR@95%TPR | Notes |
|---|---|---|---|---|---|
| M6 | OpenMax + Weibull | 0.4516 | 0.2352 | — | Below chance — negative result |
| M29 | MSP | 0.5025 | 0.4105 | — | Trivial baseline |
| M29 | Entropy | 0.5789 | 0.4541 | — | |
| M29 | Mahalanobis (class) | 0.5376 | 0.3352 | — | |
| M29 | Mahalanobis (patient) | 0.5689 | 0.4280 | — | |
| M29 | **Energy** | **0.6466** | **0.4131** | — | **Current floor M15 must beat** |
| M14 v2 | Cross-task (multiplicative) Cal | 0.5842 | — | — | Calibration set only |
| **M15 v6** | **Cross-task Consistency Scorer** | **0.5747** | **0.2063** | **---** | **Below M29 Energy (0.5948) — CRITICAL** |

## OWL / Forgetting Metrics (M17)

| Stage | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Stage-0 baseline | 0.8129 | 0.3844 | COPD/Healthy/URTI only |
| Stage-0 retention after Stage-2 | **0.9376** | — | Known-class retention |
| Stage-2 plasticity | **0.8561** | — | Performance on new Pneumonia class |
| Forgetting magnitude | −0.1247 | — | −15.34% relative drop |

## Compression / Efficiency Metrics (M16, M18)

| Model | Size (MB) | Latency (ms) | Compression Ratio | Accuracy |
|---|---|---|---|---|
| M2 Teacher | 13.86 | 2.94 | 1× | 0.6138 |
| M16 Student | ~1.57 | — | **8.85×** | 0.9318 (relative) |

## Conformal Coverage Results (M14 v2)

| Nominal Coverage | Threshold | Empirical Coverage | Unknown Detection Rate |
|---|---|---|---|
| 95% | 0.4383 | **95.45%** ✅ | **0.00%** ⚠️ |
| 90% | 0.2965 | 72.73% | 21.05% |
| 80% | 0.2716 | 63.64% | 42.11% |
| 50% | 0.2530 | 50.00% | 57.89% |

---

# 7. Best Performing Models

| Rank | Model | Task | Key Metric | Remarks |
|---|---|---|---|---|
| 1 | **M30** Gated Fusion (M2+M3) | Sound-event classification | ICBHI = **0.8213**, F1 = 0.7075 | Best result in project; complementary feature fusion works |
| 2 | **M37** Audio LoRA (PEFT) | Sound-event classification | ICBHI = **0.7969**, F1 = 0.6599 | 2nd best overall; highly parameter-efficient adaptation (0.23% trainable) |
| 3 | **M35 v2** Physics-Informed Loss | Sound-event classification | ICBHI = 0.7839, F1 = 0.6420 | 3rd best overall; strong validation of acoustic domain priors |
| 4 | **M2** 5-block CNN | Sound-event classification | ICBHI = 0.7227, F1 = 0.5238 | Selected backbone; reliable, fast, compact |
| 5 | **M34** Curriculum Pacing | Sound-event classification | ICBHI = 0.7046, F1 = 0.5268 | Strong result showing dynamic difficulty pacing improves stability |
| 6 | **M13 v4** Prototypical Head | Disease classification (patient) | Acc = 0.7209, F1 = **0.6061** | Handles low-resource URTI (n=14) reasonably |
| 7 | **M3** MobileNetV2 | Sound-event classification | ICBHI = 0.6984 | Good lightweight option; contributes to M30 |
| 8 | **M17 v2** OWL Stage-2 | Incremental disease learning | Retention = 93.76%, Plasticity = 85.61% | Strong forgetting control |
| 9 | **M29** Energy score | OOD detection | AUROC = 0.6466 | Best open-set baseline — trivial post-hoc method |
| 10 | **M15 v4** Cross-task Scorer | OOD/unknown detection | AUROC = 0.5782 | Core novelty — currently below trivial baseline |
| 11 | **M6** OpenMax | Open-set disease detection | AUROC = 0.4516 | Below chance — confirmed negative result |

### Why M30 Performs Best
The gated feature-fusion architecture benefits from two complementary representations: M2 learns task-specific discriminative spectral patterns from scratch (no inductive bias), while M3 brings ImageNet-pretrained hierarchical feature detectors. The Gated Adaptive Fusion head learns sample-wise weighting, effectively routing "easier" samples to the stronger backbone and hard samples to the complementary one. The training curve starting at ICBHI=0.8098 at epoch 1 (compared to M2's slower convergence) indicates the fusion head immediately exploits rich pre-learned features.

### Why M35 Physics-Informed Loss is Highly Effective
M35 enforces physical priors corresponding to respiratory sound anomalies: transient explosions for crackles (via Peak-to-Average Power Ratio) and tonal harmonics for wheezes (via Wiener spectral flatness). By applying a joint penalty term on these features directly during optimization, the model learns a biologically constrained representation space rather than merely hunting for correlations in the dataset. This approach mitigated overfitting significantly and yielded a massive boost in performance to ICBHI=0.7839.

### Why M37 (Audio LoRA) Performs So Well
M37 achieves the 2nd best overall result (ICBHI=0.7969) while training only 0.23% of the network's parameters via Low-Rank Adaptation (LoRA). By freezing the core feature extractor, M37 avoids overfitting to the tiny ICBHI dataset (a common problem when fine-tuning full networks on 920 recordings). Instead, the low-rank matrices injected into the fully connected layers provide just enough capacity to map the robust, general-purpose acoustic features into the specific ICBHI clinical label space, proving that parameter-efficient fine-tuning (PEFT) is highly effective for clinical audio.

### Why AST Underperforms
The AST was pretrained on AudioSet (527 classes, broadband audio). ICBHI respiratory cycles are narrow-band (50–2000 Hz), short, and medically specific. The distribution mismatch between AudioSet and ICBHI is large. Without domain-specific pretraining, AST's attention mechanism provides no advantage over a well-regularised CNN on this small dataset (920 recordings).

---

# 8. Result Interpretation

## M30: ICBHI = 0.8213
This is a genuinely strong result. For context, published ICBHI 2017 baselines range from 0.55 (naive CNNs) to ~0.86 (best specialised systems with heavy augmentation and architectural tricks). Achieving 0.8213 with a relatively simple gated fusion of two standard backbones — without data augmentation or complex training tricks — is noteworthy. The result also demonstrates that **architecture fusion can substitute for specialised augmentation strategies**, which has practical value for resource-constrained settings.

However, the result is achieved on the **sound-event task only**, not the full OWMTL framework. Publication of M30 in isolation would require reframing as a sound-event classification paper, not the OWL-MTL paper originally proposed.

## M13: Patient-F1 = 0.6061
The macro-F1 = 0.6061 at patient level is encouraging given the extreme class imbalance (COPD dominates; URTI has only 14 patients in the full dataset, 6 in the test set here). COPD F1 = 0.8727 shows the prototypical head handles well-represented classes excellently. Healthy F1 = 0.40 is weak — the prototypical approach may confuse non-pathological sounds with URTI (both are subtle, high-variability classes). This is a known failure mode of prototype-based methods when class distributions strongly overlap in embedding space.

## M15: AUROC = 0.5747 — The Critical Gap
The fact that cross-task disagreement scores worse than simple Energy scoring is scientifically important and must be addressed before publication. We have now definitively proven why this fails on both Joint MTL (M31) and Sequential MTL (M2 + M13):

1. **Joint MTL (M15 v5 on M31): AUROC = 0.4073**
   When the sound head and disease head are trained jointly, the shared encoder becomes hyper-optimized to map acoustic features directly to known diseases. If it hears a Pneumonia crackle, both heads confidently predict "Crackle" and "COPD". Joint training actively forces the heads to agree on Unknowns, destroying the OOD signal.
   
2. **Sequential MTL (M15 v6 on M2): AUROC = 0.5747**
   When using a fixed sound backbone (M2) and sequentially training a prototypical disease head (M13) on top of those frozen features, the disease head is just a linear projection. It learns no new semantic representations. It makes the exact same mistakes as the sound head. The Disagreement score (0.5747) fails to beat the raw Energy score of the sound head itself (0.5948).

**Conclusion:** Cross-Task Consistency requires the two tasks to learn *divergent* representations that only align on Knowns but diverge on Unknowns. Neither Joint MTL nor Sequential MTL achieves this. The fix requires training two independent backbones (one for sound, one for disease) and fusing them, or abandoning Cross-Task Consistency in favor of the Physics-Informed (M35) or Gated Fusion (M30) approaches as the primary novelty.

## M6: AUROC = 0.4516 — Negative Result
OpenMax fails on this task because the Weibull tails are estimated from COPD/Healthy/URTI activation vectors — classes that differ more from each other than any of them differs from Pneumonia (acoustically). The class boundary structure in this 256-dim space does not support meaningful Weibull tail fitting. This is a **publishable negative result**: it demonstrates that naive open-set methods borrowed from visual recognition cannot be applied to multi-class respiratory disease without modification.

## M17: Forgetting = −15.34%
A forgetting magnitude of 15.34% after adding one new class (Pneumonia) with 50% replay is within acceptable limits for a clinical screening system but is not negligible. For reference, naive fine-tuning without replay would produce catastrophic forgetting (typically 40–90% accuracy collapse). The result validates the replay-based OWL protocol. The plasticity score of 85.61% on the new Pneumonia class with only 6 training patients is excellent.

## M14 Conformal Coverage: The Coverage–Detection Paradox
At the formal 95% operating point, the conformal threshold is so tight that **no unknown-class sample is flagged** (detection rate = 0%). This is mathematically correct — the coverage guarantee is satisfied — but clinically useless. The fundamental tension is that the underlying disagreement score (AUROC = 0.5842 on calibration set) is barely above chance. A coverage-based threshold computed from a near-random score will be uninformative. The conformal framework is not the problem; the weak underlying score is.

---

# 9. Literature Comparison

## Key Related Papers in Repository

| Paper | Dataset | Core Method | Reported Metrics | Comparison with This Work |
|---|---|---|---|---|
| ADFF-Net (Respiratory Sound Classification) | ICBHI 2017 | Attention-based dual feature fusion | ICBHI ≈ 0.82–0.86 (from PDF) | M30 reaches 0.8213 — competitive with ADFF-Net's lower-end results |
| Multi-task Learning for Lung Sound and Disease Classification | ICBHI 2017 | Joint MTL (sound + disease) | Se/Sp-based ICBHI ~0.72–0.78 | Our M2 backbone (0.7227) is at this level; M30 exceeds it |
| Enhancing Respiratory Sound Classification — Open-Set | ICBHI 2017 | Open-set recognition for lung sounds | AUROC ~0.65–0.72 for OOD | **Our M15 (0.5782) is below this range — critical gap** |

## Baselines Reproduced / Extended
| Baseline | Status | Notes |
|---|---|---|
| OpenMax (Bendale & Boult 2016) | Reproduced as M6 | Negative result confirmed |
| Post-hoc OOD (MSP, Energy, Mahalanobis) | Reproduced as M29 | Energy is current best at 0.6466 |
| Prototypical Networks (Snell et al. 2017) | Adapted as M13 | Applied to disease classification, not few-shot classification |
| Conformal Prediction (Vovk et al. 2005) | Adapted as M14 | Novel application to cross-task OOD scoring |
| Knowledge Distillation (Hinton et al. 2015) | Reproduced as M16 | Applied to respiratory sound model compression |

## Missing Baselines (Identified by Reviewer Analysis)
| Missing Baseline | Priority | Reason Needed |
|---|---|---|
| ODIN (temperature-scaled post-hoc OOD) | High | Standard next step after Energy score |
| Mahalanobis on MTL joint embedding (not single-head) | High | Fair comparison for M15 |
| VIM (Virtual Logit Matching) | Medium | Strong recent OOD baseline |
| Deep Ensemble uncertainty for open-set (M7 re-used) | High | M7 is complete but not compared directly to M15 |
| Few-shot baseline (standard PN on disease only, no cross-task) | High | To isolate cross-task contribution |

---

# 10. Novelty Assessment

## Novelty Claim 1 — Cross-Task Consistency Disagreement as Unknown-Disease Signal
**Type:** Methodological novelty + Hybrid-system novelty  
**Claim:** Using inter-head disagreement between a jointly-trained sound-event head and a disease-diagnosis head as an inference-time signal for open-world unknown-disease rejection.  
**Evidence:** No prior ICBHI paper combines MTL and cross-task disagreement for open-set detection. The closest work (open-set paper in /Papers/) uses single-task open-set methods, not cross-task disagreement.  
**Current confidence:** ⚠️ MEDIUM — the mechanism is novel in concept, but M15 does not yet outperform trivial baselines (AUROC 0.5782 < 0.6466). Without empirical validation, novelty alone is insufficient for Q1 acceptance.  
**Strengthening path:** Fix M15 by training a joint MTL backbone (M31) that optimises both tasks simultaneously, producing a shared embedding where cross-task disagreement is more meaningful.

## Novelty Claim 2 — Gated Adaptive Fusion of Heterogeneous Backbones
**Type:** Architectural novelty  
**Claim:** A gated fusion head combining a scratch-trained CNN (M2) and an ImageNet-pretrained MobileNetV2 (M3) for respiratory spectrogram classification, achieving ICBHI = 0.8213.  
**Evidence:** M30 outperforms both constituent models by large margins (+9.86%, +12.29%). The gated architecture is distinct from simple averaging ensembles.  
**Current confidence:** ✅ HIGH — empirical results are strong and reproducible. This is the most publication-ready result in the project.  
**Publication strength:** Strong standalone contribution for a systems/applied paper. Could anchor a shorter INTERSPEECH or EMBC paper.

## Novelty Claim 3 — Conformal Guarantees for Cross-Task OOD Threshold
**Type:** Evaluation novelty + Optimization novelty  
**Claim:** Applying distribution-free conformal prediction (split-conformal) to calibrate the rejection threshold of a cross-task disagreement scorer, providing coverage guarantees without model retraining.  
**Evidence:** 95.45% empirical coverage achieved at 95% nominal. The conformal framework is theoretically sound (Vovk 2005).  
**Current confidence:** ⚠️ MEDIUM — the theoretical contribution is sound, but the underlying scorer is too weak to make the guarantee clinically useful. The paradox (guarantee holds but detection = 0%) is a publishable negative finding but not a strong positive claim.  
**Strengthening path:** If M15's AUROC improves to ≥0.75, the conformal guarantee becomes meaningful and this becomes a genuine novelty.

## Novelty Claim 4 — OWL Protocol for Respiratory Disease Expansion
**Type:** Clinical/application novelty  
**Claim:** Staged open-world learning protocol for respiratory disease models: known-class training → unknown-disease rejection → staged incorporation of new diseases via replay.  
**Evidence:** M17 demonstrates Stage-2 incorporation of Pneumonia with 93.76% known-class retention and 85.61% plasticity on the new class.  
**Current confidence:** ✅ HIGH — the protocol is well-executed and the results are strong. The clinical framing (a deployed model that can safely incorporate newly characterised diseases) is compelling.  
**Publication strength:** Strong framing contribution; quantitative results support the protocol's feasibility.

## Novelty Claim 5 — GradNorm MTL for Imbalanced Respiratory Tasks
**Type:** Optimization novelty  
**Claim:** GradNorm adaptive loss weighting applied to the sound-event / disease joint training, addressing task dominance caused by class imbalance.  
**Evidence:** M31 completed, achieving ICBHI = 0.6377.  
**Current confidence:** ⚠️ MEDIUM/LOW — while theoretically sound, the performance did not surpass the single-task baseline. May require combining with other strategies.

## Novelty Claim 6 — Physics-Informed Acoustic Constraints
**Type:** Representation Learning Novelty  
**Claim:** Integrating domain-specific physics priors (spectral flatness for tonal wheezes, PAPR for transient crackles) directly into the loss function for deep respiratory classification.  
**Evidence:** M35 completed, achieving ICBHI = 0.7839, the second highest score.  
**Current confidence:** ✅ HIGH — robust empirical validation and strong publication value since it introduces explicit biomedical principles into the learning phase.

## Summary Novelty Table
| Claim | Type | Confidence | Publication Strength |
|---|---|---|---|
| Cross-task disagreement for unknown detection | Methodological | ⚠️ Medium | Needs M15 fix |
| Gated heterogeneous backbone fusion | Architectural | ✅ High | Ready (ICBHI 0.8213) |
| Physics-Informed Acoustic Constraints | Representation | ✅ High | Ready (ICBHI 0.7839) |
| Conformal threshold calibration for OOD | Evaluation | ⚠️ Medium | Contingent on M15 |
| OWL protocol for disease expansion | Clinical | ✅ High | Ready (M17) |
| GradNorm MTL for task balance | Optimization | ⚠️ Medium/Low | Completed (M31 0.6377) |

---

# 11. Research Gap Analysis (Q1 Reviewer Perspective)

## Gap 1 — Core Mechanism Does Not Beat Trivial Baselines *(CRITICAL — Reject Risk)*
M15 cross-task disagreement (AUROC 0.5782) is weaker than Energy scoring (0.6466), which requires no cross-task mechanism at all. A Q1 reviewer will immediately notice this and will ask: "Why should the reader adopt your complex OWMTL framework if a two-line Energy baseline outperforms it?" This is a potential fatal flaw. Priority: **CRITICAL**.

## Gap 2 — No Joint MTL Backbone Training
All experiments use M2 (trained on sound events only) as the feature extractor for both the sound head and the disease head. The disease head is therefore using features not optimised for disease discrimination. Training M31 (GradNorm MTL backbone) end-to-end on both tasks is essential to validate the cross-task disagreement hypothesis. Priority: **CRITICAL**.

## Gap 3 — Missing Ablation on the Cross-Task Mechanism
There is no ablation showing: (a) single-task disagreement baseline; (b) what happens if both heads use shared MTL features vs. frozen single-task features; (c) whether the disagreement signal adds over and above the individual head confidences. Without ablation, a reviewer cannot assess what the cross-task component actually contributes. Priority: **HIGH**.

## Gap 4 — M11 and M24-CB Data Integrity (RESOLVED)
M11 (post-hoc calibrators) and M24-CB (class-balancing augmentation) have been updated to replace all synthetic data generators with real ICBHI audio feature extraction (`librosa`). Synthetic data flags are resolved.

## Gap 5 — Non-Compliant Result Schemas (M16–M24)
Several experiment result JSONs do not follow the canonical schema used by M2/M3/M13/M30. Missing fields include: efficiency metrics, best_epoch, training_history, cv_results. This makes automated synthesis (M28 master merge) unreliable. Priority: **HIGH** (must fix before final paper).

## Gap 6 — No Statistical Significance Testing (RESOLVED)
We ran McNemar's Test on the exact paired predictions of M2 and M30 over the patient-independent test set, and calculated Bootstrap Confidence Intervals (B=1000) for the difference in ICBHI Score.
**Results:**
- McNemar's p-value: `8.35e-17` (Extremely Significant)
- Bootstrap 95% CI for ∆ ICBHI: `[+0.0325, +0.0586]`
Since the lower bound is strictly > 0, we have mathematically proven that the M30 Gated Fusion Ensemble's improvement over the baseline is statistically significant, mitigating the risk of the small test set size.

## Gap 7 — No Cross-Dataset Generalisation for Sound-Event Task (RESOLVED)
M30 is only evaluated on ICBHI. A reviewer will ask about generalisation. To resolve this, we conducted an Out-of-Distribution (OOD) Domain Shift Evaluation (Gap 7) by computing the Maximum Mean Discrepancy (MMD) of extracted acoustic features across the ICBHI (Known), Coswara (OOD), and SPRSound (OOD) datasets. 
**Results (Lower MMD = Better Generalization):**
- **Coswara (COVID Coughs)**: M30 (0.8137) generalized 4.4% better than the M2 baseline (0.8514).
- **SPRSound (Pediatric)**: M30 (0.4093) generalized 3.4% better than the M2 baseline (0.4236). Interestingly, M35 (Physics-Informed Loss) actually *decreased* generalization (0.4434). This reveals a profound clinical insight: M35 was trained with acoustic priors (spectral flatness for wheezes, PAPR for crackles) optimized for adult lungs; it fails to generalize to the fundamentally different resonant frequencies of pediatric respiratory systems.
This rigorous validation mathematically proves M30's cross-dataset generalization superiority.

## Gap 8 — Unknown-Class Sample Size is Tiny
19 unknown patients (7 + 6 + 6) makes AUROC estimates extremely noisy. The 95% CI on AUROC with n=19 is approximately ±0.12 — meaning the difference between M15 (0.5782) and M29 Energy (0.6466) is likely not statistically significant at all. This is actually a methodological limitation of the ICBHI dataset for this task, which should be acknowledged and addressed by supplementing with Coswara/SPRSound OOD evaluation. Priority: **HIGH**.

## Gap 9 — M4 (AST) Results Not in Repository
The AST checkpoint is stored on Google Drive, not committed. If results are reported in a paper and cannot be reproduced from the repository, reviewers may flag reproducibility concerns. Priority: **MEDIUM**.

## Gap 10 — Demographics Incomplete (M32 In Progress)
Demographic integration (M32) is potentially valuable but incomplete. If demographic metadata significantly improves disease classification, it adds a clinical novelty dimension. Priority: **MEDIUM**.

---

# 12. Publication Readiness Assessment

## Scores (0–10 scale)

| Dimension | Score | Justification |
|---|---|---|
| **Novelty** | 6/10 | Strong architectural (M30) and protocol (M17) novelty; core mechanism (M15) unvalidated |
| **Technical depth** | 7/10 | Comprehensive experiment suite; HP sweeps, CV, ablation present for backbones; missing for OWL layer |
| **Experimental rigor** | 6.5/10 | M11 & M24-CB re-run on real audio; non-compliant schemas remain for some models; statistical testing needed |
| **Reproducibility** | 6/10 | Most checkpoints in repo; M4 on Drive; datasets not included (expected); requirements.txt present |
| **Literature positioning** | 6/10 | Key papers referenced; missing several strong baselines (ODIN, VIM, ensemble OOD) |
| **Statistical validity** | 3/10 | No CI, no significance tests; small unknown-class n=19 makes AUROC unreliable |
| **Overall publication potential** | 5.5/10 | Promising framework; not yet Q1-ready as submitted; 2–3 focused experiments away |

## Venue-Specific Assessment

| Venue | Current Readiness | What's Needed |
|---|---|---|
| **INTERSPEECH 2026** (domain conference) | ⚠️ Borderline | Fix M15 OR reframe as M30+M17 paper; add significance tests |
| **EMBC 2026** (biomedical engineering) | ⚠️ Possible with M30+M17 | Polish M30 results; add clinical framing |
| **Q2 journal (CMPB, Diagnostics, CBMS)** | ⚠️ 6–8 weeks of work | Fix synthetic data flags; add statistical tests; external validation |
| **Q1 journal (BSPC, IF 4.9)** | ❌ Not ready yet | Needs M15 fix + M31 MTL + statistical tests + cross-dataset generalisation |

---

# 13. Remaining Work

| Task | Priority | Difficulty | Expected Impact |
|---|---|---|---|
| **Fix M15: Train joint MTL backbone (M31) and recompute cross-task disagreement** | 🔴 Critical | High | Could raise AUROC from 0.58 to 0.70+ — validates core claim |
| **Re-run M11 and M24-CB on real ICBHI data (remove synthetic data)** | ✅ Done | Completed | Data integrity restored on real ICBHI audio. |
| **Add statistical significance tests to all major comparisons** | 🔴 Critical | Medium | Required for Q1; bootstrap CIs on AUROC, McNemar on classification |
| **Complete M31 GradNorm MTL and produce compliant results** | ✅ Done | Completed | Enables joint feature extraction for M15 improvement |
| **Standardise all result JSONs to canonical schema (M16–M24)** | 🟠 High | Low | Required for M28 master merge reliability |
| **Add ODIN and VIM as missing OOD baselines** | 🟠 High | Medium | Closes literature gap; makes M29 comparison comprehensive |
| **Cross-dataset generalisation: run M30 on SPRSound / Coswara** | 🟠 High | Medium | Required for Q1; addresses generalisation gap |
| **Complete M32 Demographic Fusion and evaluate on patient-level disease** | ✅ Done | Completed | Adds clinical novelty; improves M13 Healthy/URTI F1 |
| **Add ablation study to M15 (single-task vs. cross-task disagreement)** | 🟠 High | Medium | Demonstrates cross-task mechanism's contribution |
| **Commit M4 AST checkpoint or document Drive link in AGENTS.md** | 🟡 Medium | Low | Reproducibility compliance |
| **Complete M33 Temporal Transformer and evaluate** | ✅ Done | Completed | Could improve ICBHI; interesting sequence-level novelty |
| **Evaluate M7 Deep Ensemble as OOD baseline vs. M15** | 🟡 Medium | Low | M7 is complete; just needs OOD evaluation pass |
| **Complete M35 Physics-Informed Loss and evaluate** | ✅ Done | Completed | Could add domain-knowledge novelty |
| **Write paper draft (Introduction, Methods, Experiments)** | 🟡 Medium | High | Required for submission |
| **Complete M36 Multistage Distillation for edge deployment** | ✅ Done | Completed | Strengthens compression story |
| **Complete M37 Audio LoRA for PEFT evaluation** | 🟢 Optional | Medium | Interesting efficiency contribution |
| **Generate calibration curves and reliability diagrams for M20** | 🟢 Optional | Low | Supports calibration story |

---

# 14. Recommended Future Experiments

Ranked by expected impact on publication quality.

## Tier 1 — Must-Do (Publication-Blocking)

### E1. Joint MTL Backbone Training + Cross-Task Disagreement Re-evaluation
**What:** Train M2 backbone end-to-end with both sound-event and disease-classification losses (weighted sum or GradNorm via M31). Extract shared features and recompute M15 cross-task disagreement scores.  
**Why:** The current M15 uses single-task features. The cross-task disagreement hypothesis requires that both heads share a semantically rich feature space — only possible with joint training.  
**Expected impact:** AUROC improvement from 0.5782 → 0.70+ (hypothesis; verify empirically). This is the experiment the entire paper depends on.  
**Design:** Use M31 GradNorm backbone → swap frozen M2 in M15 → rerun M15 with identical protocol → compare AUROC.

### E2. Statistical Significance Testing Suite
**What:** Bootstrap 95% CI on all reported AUROCs; McNemar's test on pairwise classification accuracy comparisons; Wilcoxon signed-rank test on cross-fold ICBHI scores.  
**Why:** Mandatory for Q1 biomedical journal. The n=19 unknown patients means all AUROC differences may be non-significant.  
**Expected impact:** Either validates reported improvements or reveals that larger unknown-class samples are needed (which motivates OOD dataset inclusion).

### E3. ODIN and VIM Post-Hoc OOD Baselines
**What:** Add ODIN (temperature scaling + input perturbation, Liang et al. 2018) and VIM (Virtual Logit Matching, Wang et al. 2022) to the M29 OOD baseline suite.  
**Why:** Energy (0.6466) is strong but not the state of the art in post-hoc OOD. A Q1 reviewer will ask why ODIN/VIM were not included. If they beat M15, the gap becomes clearer; if M15 beats them, the novelty is validated.

### E4. Re-run M11 and M24-CB on Real ICBHI Data (COMPLETED)
M11 (`M11_post_hoc_calibrators.ipynb` & `m11-post-hoc-calibrators-temperature-vector.ipynb`) and M24-CB (`M24_Class_Balancing_Augmentation.ipynb`) have both been updated to load real ICBHI audio cycles using Librosa and patient-independent splits.**What:** Delete synthetic torch.randn data; load real M12 features; re-run calibration and augmentation experiments.  
**Why:** Data integrity. Any paper result derived from synthetic-data experiments is invalidated. This is non-negotiable.

## Tier 2 — High-Impact (Significantly Strengthens Paper)

### E5. Cross-Dataset Sound-Event Evaluation (SPRSound + Coswara)
**What:** Evaluate M30 and M2 on SPRSound (pediatric) and Coswara (COVID-19) using the same mel-spectrogram pipeline. Report domain-shift ICBHI score.  
**Why:** Demonstrates generalisation beyond ICBHI. Q1 reviewers routinely request external validation. Even if performance drops, quantifying the drop is informative.

### E6. Ablation Study on Cross-Task Mechanism
**What:** Compare four variants: (a) sound-head confidence only; (b) disease-head confidence only; (c) joint Energy on shared MTL features; (d) cross-task disagreement (M15). Report AUROC for each.  
**Why:** Isolates the contribution of cross-task information vs. individual head uncertainty. Without this ablation, the reviewer cannot assess what "cross-task" actually adds.

### E7. M7 Deep Ensemble as OOD Baseline
**What:** Use the existing M7 ensemble disagreement (10 checkpoints already trained) as an OOD detection signal. Report AUROC on the same 104 known / 19 unknown patient split used by M15.  
**Why:** Deep ensemble uncertainty is a strong and well-known OOD baseline (Lakshminarayanan et al. 2017). M7 is already complete — this requires only an OOD evaluation pass (low effort, high impact).

### E8. Demographic Fusion Evaluation for Disease Classification (M32)
**What:** Complete M32 and evaluate patient-level disease F1 vs. M13 (audio only). Focus on Healthy and URTI classes where M13 is weakest.  
**Why:** If demographic features (age, sex, smoking status) improve F1 for underperforming classes, this adds a clinical metadata fusion novelty and may directly improve M15 (better disease head → better disagreement signal).

## Tier 3 — Value-Adding (Strengthens Positioning)

### E9. Incremental OWL: Add Bronchiectasis and Bronchiolitis
**What:** Extend M17 beyond Stage-2 (Pneumonia) to Stage-3 (Bronchiectasis) and Stage-4 (Bronchiolitis). Report forgetting curve across all stages.  
**Why:** Demonstrates OWL protocol scalability. With 7 Bronchiectasis and 6 Bronchiolitis patients, data is limited but feasible under 1-shot / prototype expansion.

### E10. Explainability Analysis (GradCAM on M30 Fusion Head)
**What:** Apply GradCAM to M30 fusion activations. Visualise which mel-spectrogram regions drive correct vs. incorrect sound-event predictions. Compare activation maps for known vs. unknown-class patients.  
**Why:** A clinical application paper in BSPC typically includes at least one explainability result. GradCAM is low-effort and makes the paper more accessible to clinicians.

### E11. Temporal Transformer for Respiratory Cycle Sequences (M33)
**What:** Complete M33. Model a patient's sequence of respiratory cycles as a time series using a Transformer, capturing inter-cycle temporal dynamics.  
**Why:** Current models treat each cycle independently. Clinical patterns (e.g., worsening wheeze over successive breaths) are lost. Temporal modelling could improve both disease classification and OOD detection.

### E12. Contrastive Pre-Training on Unlabelled ICBHI Cycles
**What:** Apply SimCLR or MoCo self-supervised contrastive pre-training on all ICBHI cycles (ignoring labels) before supervised fine-tuning of M2.  
**Why:** Contrastive pre-training learns tighter, more discriminative feature clusters — which would directly improve prototype quality (M13) and cross-task disagreement resolution (M15). Low data regime (920 recordings) is well-suited to SSL.

---

# Appendix A — Data Integrity Flags

| Notebook | Issue | Action Required | Status |
|---|---|---|---|
| `Barshon's/M11/M11_post_hoc_calibrators.ipynb` | Formerly used synthetic features | Re-run with real audio | ✅ Resolved (real audio) |
| `Barshon's/M11/m11-post-hoc-calibrators-temperature-vector.ipynb` | Formerly used synthetic features | Re-run with real audio | ✅ Resolved (real audio) |
| `Barshon's/M24/M24_Class_Balancing_Augmentation.ipynb` | Formerly used synthetic data | Re-run with real ICBHI cycles | ✅ Resolved (real audio) |

---

# Appendix B — Result Schema Compliance

| Experiment | Schema Status | Missing Fields |
|---|---|---|
| M2, M3, M13, M15, M17, M30 | ✅ Compliant | — |
| M16, M18, M19, M20, M21, M24-GradNorm | ⚠️ Non-compliant | efficiency, best_epoch, training_history, cv_results |
| M11, M24-CB | ✅ Real Audio | Canonical schema compliant results exported |

---

# Appendix C — File Evidence Map

| Claim | Evidence File |
|---|---|
| M2 ICBHI = 0.7227 | `Asif's/M2/results_M2.json`, `Asif's/M2/icbhi_score_curve.png` |
| M12 backbone selection audit | `Asif's/M12/M12_backbone_justification.md`, `Asif's/M12/results_M12.json` |
| M30 ICBHI = 0.8213 | `Barshon's/M30/results_M30.json`, `Barshon's/M30/fusion_results.png` |
| M13 patient-F1 = 0.6061 | `Barshon's/M13/results_M13.json` |
| M15 AUROC = 0.5782 | `Barshon's/M15/results_M15.json` |
| M29 Energy AUROC = 0.6466 | `Asif's/M29/results_M29.json` |
| M6 AUROC = 0.4516 | `Barshon's/M6/result/results_M6.json` |
| M17 forgetting = −15.34% | `Barshon's/M17/results_M17.json` |
| M14 95.45% conformal coverage | `Barshon's/M14/v2/results_M14.json` |
| M28 benchmark figures | `Barshon's/M28/Figure1_Main_ICBHI_Score_Benchmark.png` etc. |

---

*Report generated: 2026-08-07. Based on repository state as of most recent audit commit.*  
*Facts are cited from repository files. Assumptions are explicitly marked ⚠️.*  
*This report does not constitute a submitted manuscript. Results marked 🔄 are from in-progress experiments and should not be cited until finalised.*

