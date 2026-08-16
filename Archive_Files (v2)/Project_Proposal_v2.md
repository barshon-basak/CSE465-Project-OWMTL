# Cluster-Aware Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis

**Target: Q1 journal publication**

**Revision note (v2):** This version adds a mandatory, project-wide **Standardized Model Training & Reporting Protocol** (§8.5) — full metric suites, efficiency reporting, learning curves, and a data-augmentation ablation applied identically to every model — per the supervisor's explicit instruction. The core narrative, novelty claim, dataset plan, and open-world mechanism (§§1–7) are unchanged from v1; this is a rigor/reporting upgrade, not a scope change, and it strengthens the paper's Q1 competitiveness by closing a gap reviewers routinely flag: incomplete efficiency reporting and missing augmentation ablations in ICBHI-based papers.

---

## 1. Abstract

Automated respiratory sound classification has been extensively studied on the ICBHI 2017 benchmark, but nearly all prior work — including multi-task learning (MTL) approaches that jointly classify sound events and diseases — operates under a **closed-world assumption**: every diagnosis seen at test time was present at training time. This is clinically unrealistic; a deployed screening tool will inevitably encounter conditions it was never trained on. This project proposes an **open-world multi-task learning (OWMTL) framework** that uses **cross-task consistency** between a sound-event head and a disease-diagnosis head to flag inputs likely to belong to an unseen disease, evaluated through a staged open-world learning (OWL) protocol. To avoid both (a) re-treading a saturated closed-set MTL literature and (b) the statistical fragility of ICBHI's smallest diagnostic classes, the evaluation is restructured around a coarse, pooled open-world task plus **large-sample, cross-dataset out-of-distribution (OOD) stress tests** (Coswara, SPRSound), with fine-grained per-disease results reported as qualitative case studies rather than headline statistics. The target venue is *Biomedical Signal Processing and Control* (Elsevier, Q1, IF 4.9).

---

## 2. Introduction and Motivation

Chronic and acute respiratory diseases remain a leading global cause of morbidity, and auscultation-based screening — amplified by digital stethoscopes — has become a major target for deep learning research. The ICBHI 2017 Respiratory Sound Database <cite index="85-1">was compiled to support a scientific challenge at the International Conference on Biomedical and Health Informatics, with the goal of developing algorithms able to characterize respiratory sound recordings from clinical and non-clinical environments, comprising 920 recordings from 126 subjects</cite>. It has since become the default benchmark for the field: a 2025 systematic review <cite index="87-1">identified 135 technical publications built on this single database</cite>, spanning signal processing, feature extraction, and classification methodology.

Within this literature, multi-task learning (MTL) — jointly predicting sound events (crackle/wheeze/normal) and disease diagnosis from a shared backbone — has emerged as a recurring theme, motivated by the idea that <cite index="76-1">task-sharing within a unified network enables the exchange of complementary information, acting as a regularizer that mitigates overfitting and improves model performance</cite>. However, **every MTL study identified in this review operates under closed-set assumptions**: the disease classes available at inference are identical to those seen at training. This is a meaningful gap, because a real screening deployment will encounter patients with conditions outside the training distribution, and a model that confidently misclassifies an unknown disease as a known one is more dangerous than one that abstains.

This project addresses that gap directly: **can a multi-task model detect, at inference time, that a patient's condition does not match any disease class it was trained on — using nothing but the disagreement between its own sound-event and disease-diagnosis predictions?**

---

## 3. Related Work and Positioning

### 3.1 Closed-set MTL on ICBHI (crowded — not the contribution)

- <cite index="74-1">Suma et al. (2025) propose a multitask learning approach integrating MTL with four architectures (2D CNN, ResNet50, MobileNet, DenseNet) to extract features from lung sound recordings, evaluated on ICBHI, with the MTL-MobileNet variant achieving 74% accuracy for lung sound analysis and 91% for lung disease classification</cite> (*SN Computer Science*, 2025).
- Tri-MTL (2025, arXiv:2505.06271) extends this line by incorporating an Audio Spectrogram Transformer and electronic-stethoscope metadata, <cite index="76-1">demonstrating that lung sound classification as an auxiliary task enhances disease diagnosis performance</cite>.
- **None of these papers address unseen/unknown disease detection.** They are cited here as closed-set MTL baselines the proposed backbone architecture will build on — not as competing open-world contributions.

### 3.2 Open-set respiratory recognition (closest direct prior art)

- <cite index="65-1">Cho & Lee (2025) explore an open-set semi-supervised learning setting for respiratory sound classification, where unlabeled data may contain additional unknown classes, addressed via a distance-based prototype network</cite> (*Computers, Materials & Continua*, 84(2), 2847–2863).
- This is the single closest paper to the proposed direction, and **must be cited and differentiated explicitly** in any submission. The differentiation is mechanistic: Cho & Lee use single-task prototype/distance-based rejection; this proposal uses **cross-task disagreement between two jointly-trained heads** as the rejection signal, combined with a **staged open-world learning protocol** (not a one-shot open/closed split) — a combination not present in their work or any other paper found during this search.

### 3.3 Cross-task / multi-head disagreement as an OSR mechanism (establishes the mechanism is not itself novel)

- <cite index="97-1">Huo et al. propose a post-processing open-set recognition method that measures agreement between a model's features and predicted logits, comparing a Nearest-Class-Mean-based probability distribution against softmax probabilities from the logit space</cite>, applied to wildlife camera-trap classification (SACAIR 2025 / Springer CCIS).

This confirms that "disagreement between two independent signals" as an unknown-detector is a **general OSR mechanism already applied in other domains**, not a novel algorithmic idea in itself. **The paper's contribution must therefore be framed around the domain-specific combination (MTL + staged OWL + respiratory audio) and the CQKD-style efficiency extension (Section 5.3), not around "inventing" disagreement-based rejection.** Reviewers familiar with the OSR literature will recognize the base mechanism; the proposal should pre-empt this by citing Huo et al. explicitly in the related-work section rather than hoping it goes unnoticed.

### 3.4 Why the smallest ICBHI classes cannot carry the evaluation

<cite index="100-1">Soni et al. (2022), working with the same ICBHI database, modified their task from multiclass diagnosis to binary normal-versus-abnormal classification specifically due to limitations in data independence between samples, noting that some classes have only one patient (asthma)</cite> (*Patterns*, Cell Press, 3(1), 100400). This is independent, peer-reviewed confirmation — from a different research group, for a different purpose — that ICBHI's fine-grained diagnostic classes cannot support standard statistical evaluation. This proposal's task redesign (Section 5.1) is a direct response to this documented limitation.

### 3.5 Dr. Khan's lab lineage (verified)

- Karim, Mahmud & Khan (2024), <cite index="59-1">"Advanced vision transformers and open-set learning for robust mosquito classification," introduces a framework integrating Transformer-based models with the ability to handle unseen classes at inference through open-set learning, employing the OpenMax technique and Weibull distribution</cite> (*PLOS Computational Biology*, 20(12): e1012654). **This is the verified methodological anchor from the supervising lab** and the natural citation to build institutional continuity from — the OpenMax/Weibull machinery here is a candidate second rejection mechanism to benchmark against cross-task consistency in ablations.
- Pavel, Islam, Babor, Mehadi & Khan (2024), "Non-small cell lung cancer detection through knowledge distillation approach with teaching assistant" (*PLOS ONE*, 19(11): e0306441), demonstrates the lab's applied expertise in knowledge-distillation pipelines for medical signal/image classification — relevant background for the CQKD-style efficiency extension, even though CQKD itself is external work.

---

## 4. Research Gap and Novelty Statement

| Claim | Status | Support |
|---|---|---|
| MTL for ICBHI sound+disease classification exists | Established, crowded | §3.1 |
| Open-set recognition for ICBHI exists | One paper (Cho & Lee 2025), different mechanism | §3.2 |
| MTL cross-task disagreement as an OWL detector, applied to respiratory audio | **No prior work found** | §3.2, §3.3 confirms mechanism exists elsewhere, not in this domain/combination |
| Fine-grained per-disease OWL evaluation on ICBHI held-out classes | Statistically unreliable, independently confirmed | §3.4 |

**Novelty claim (precise, defensible form):** *A multi-task architecture that uses cross-task disagreement between jointly-trained sound-event and disease-diagnosis heads, under a staged open-world learning protocol, to detect diseases unseen during training — evaluated with a statistically sound coarse-open-world/large-N-OOD design rather than fine-grained small-sample splits, and extended with a cluster-quantized distillation mechanism for edge deployability.*

This framing does **not** conflict with Cho & Lee (2025) (different mechanism), the MTL papers (closed-set only), or Huo et al. (2025) (different domain, no MTL, no staged OWL protocol).

---

## 5. Proposed Methodology

### 5.1 Task Formulation (redesigned for statistical validity)

- **Primary quantitative task A — Sound-event classification:** Normal / Crackle / Wheeze / Both, over the full 6,898-cycle ICBHI corpus. Full ablation depth here (large sample, low risk).
- **Primary quantitative task B — Coarse open-world diagnosis:** known classes (COPD, Healthy, URTI — 104 patients) vs. one pooled "unknown" class (all 19 held-out patients combined). Statistically defensible group sizes.
- **Primary quantitative task C — Cross-dataset OOD generalization:** train on ICBHI known classes only; evaluate unknown-flagging against Coswara (thousands of participants) and SPRSound (292 participants, paediatric) as genuinely unseen populations/recording conditions. This is the paper's main "scale" evidence.
- **Secondary qualitative task — Per-disease case studies:** Pneumonia (n=6), Bronchiolitis (n=6), Bronchiectasis (n=7) reported individually as patient-level tables, explicitly not accompanied by confidence intervals or significance tests.

### 5.2 Architecture

A shared audio encoder (CNN or transformer-based, to be benchmarked in ablation) feeds two task-specific heads:
1. **Sound-event head** — 4-way classification (normal/crackle/wheeze/both) at the cycle level.
2. **Disease-diagnosis head** — patient-level classification over known classes.

**Cross-task consistency score:** for a given input, compare the disease head's predicted class distribution against a mapping of the sound-event head's predictions to expected disease patterns (e.g., a COPD diagnosis should co-occur with a particular crackle/wheeze profile). High disagreement between the two heads' implied diagnoses is used as the unknown-class signal, thresholded and calibrated per OWL stage.

### 5.3 Efficiency Extension (methodological novelty anchor)

To avoid resting the paper's contribution on the disagreement mechanism alone (§3.3), integrate a **cluster-quantized distillation step** — correctly attributed to Khan & Rafat (2025, *Engineering Reports*) as the external technique being adapted, not claimed as in-house prior work — to compress the dual-head backbone for edge deployment, and evaluate whether cluster-quantization additionally regularizes drift across OWL stages (i.e., reduces forgetting). This gives the paper a second, orthogonal contribution: *a compressed, deployable open-world respiratory classifier*, not just an accuracy number.

### 5.4 Staged Open-World Learning (OWL) Protocol

1. **Stage 0:** Train on known classes only (COPD, Healthy, URTI).
2. **Stage 1:** Introduce cross-task consistency thresholding; evaluate unknown-detection on the pooled held-out set and on Coswara/SPRSound.
3. **Stage 2:** Incrementally incorporate a subset of previously-flagged "unknowns" (simulating a clinician confirming new cases) and re-evaluate for catastrophic forgetting of Stage-0 performance, with and without CQKD-style regularization.

---

## 6. Datasets

| Dataset | Role | Key facts |
|---|---|---|
| **ICBHI 2017** | Primary training/eval | <cite index="86-1">920 recordings from 126 participants across seven chest locations, using four devices, yielding 6,898 respiratory cycles annotated as normal, crackle, wheeze, or both</cite>. Cite the dataset paper: <cite index="82-1">Rocha, B. et al., "An open access database for the evaluation of respiratory sound classification algorithms," Physiological Measurement, 2019</cite>. |
| **Coswara** | Large-N OOD stress test | <cite index="7-1">Sharma et al., "Coswara: A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis," Interspeech 2020</cite> — crowdsourced, thousands of participants, structurally different recording modality from ICBHI. |
| **SPRSound** | Secondary OOD stress test (paediatric domain shift) | <cite index="107-1">2,683 records and 9,089 respiratory sound events from 292 participants</cite>, first open-access paediatric respiratory sound database — Zhang et al., *IEEE Transactions on Biomedical Circuits and Systems*, 16(5), 867–881, 2022. |
| ~~Fraiwan/KAUH~~ | **Dropped as an augmentation source** | KAUH's own Pneumonia subgroup is only ~5 subjects — smaller than ICBHI's, does not increase statistical power (Fraiwan et al., 2021, as cited in downstream benchmarking literature). |
| ~~HF Lung V2~~ | **Dropped as an "unknown class" source** | Confirmed to carry no disease-diagnosis labels and not to be publicly released; cannot serve the disease-rejection task. |

---

## 7. Evaluation Protocol

- **Leave-one-patient-out (LOPO)** for any group smaller than ~30 patients; k-fold only for the sound-event task, which has sufficient scale.
- **No bootstrap confidence intervals reported for n < 15** groups; raw per-patient outcome tables used instead for the qualitative per-disease case studies.
- **Patient-independent train/test splits enforced throughout** — no patient's cycles appear in both sets.
- **Non-parametric significance testing** (e.g., Wilcoxon signed-rank across patients) for comparisons involving small groups; standard parametric tests permitted for the sound-event task and the Coswara/SPRSound OOD tests given their larger sample sizes.

---

## 8. Ablation Plan

| Ablation | Task tier | Depth |
|---|---|---|
| Backbone choice (CNN vs. transformer encoder) | Sound-event (large-N) | Full |
| Loss weighting between heads | Sound-event / coarse OWL | Full |
| With/without cross-task consistency term | Coarse OWL | Full |
| CQKD-regularized vs. unregularized backbone across OWL stages | Coarse OWL, forgetting curve | Full |
| OWL stage count (1 vs. 2 vs. 3) | Coarse OWL | Full |
| Cross-task consistency vs. OpenMax/Weibull (Khan et al. 2024 mechanism) as the rejection method | Coarse OWL + Coswara/SPRSound | Full — this is a direct, citable comparison against the lab's own prior open-set method |
| Full model vs. ablated variants on Coswara/SPRSound | OOD generalization | Full — primary "scale" evidence |
| Per-disease qualitative breakdown (Pneumonia/Bronchiolitis/Bronchiectasis) | Small-N | Case study only, no CIs |
| **With vs. without data augmentation, per architecture** | All tiers | Stretch — see §8.5.4 |

---

## 8.5 Standardized Model Training & Reporting Protocol *(supervisor mandate — applies to every model)*

Every model trained anywhere in the project — regardless of which member trains it or which task it serves — is reported using the same four-part protocol below. This is what lets the four individually-owned workstreams recombine into one internally consistent results section instead of four incompatible reporting styles, and it closes the most common Q1-reviewer objection to ICBHI papers: partial metrics, no efficiency numbers, no learning curves, no augmentation ablation.

### 8.5.1 Required performance metrics (every model, every task)
For every classification model trained (sound-event, disease-diagnosis, OOD-transfer, compressed/student, ensemble/Bayesian/SNGP/evidential variants):
- **Accuracy**, **precision**, **recall**, and **F1-score** — macro-averaged (class-imbalance-robust, primary) and per-class (diagnostic detail), plus the ICBHI-standard **Sensitivity/Specificity/Score** where applicable for direct literature comparability.
- **Confusion matrix** — raw counts and row-normalized, for every model and every relevant train/test split (including per-OWL-stage matrices for the open-world mechanism, and pre-/post-compression matrices for the compression work).
- For open-set/coarse-OWL tasks specifically: known-class accuracy, unknown-detection recall/precision (i.e., how often a genuine unknown is correctly flagged vs. falsely flagged), and AUROC/AUPR for the unknown-vs-known separation score.

### 8.5.2 Required efficiency/complexity reporting (every model)
- **Model size** (on-disk, MB, and in the model's native precision, e.g., FP32/FP16).
- **Number of parameters** (total and trainable, in millions).
- **Training time** — per-epoch wall-clock time *and* total training time to convergence, both reported with the hardware used (GPU model, VRAM) so numbers are reproducible/comparable across members and against literature baselines.
- **Inference latency** (ms/sample, batch size 1) — required in addition to the above for the compression work, and recommended for every model since it strengthens the paper's "real-world deployability" narrative.

### 8.5.3 Required training curves (every model)
- **Training and validation loss vs. epoch**, plotted together on one figure per model.
- **Training and validation accuracy (or macro-F1, for imbalanced heads) vs. epoch**, plotted together on one figure per model.
- Curves must show early-stopping point (if used) and are stored alongside the checkpoint so overfitting/underfitting is auditable during the internal review with Dr. Khan, not just claimed in text.

### 8.5.4 Data-augmentation ablation *(stretch item — see note below)*

> **Scope note (2026-08-05):** this section originally mandated a full augmented-vs-clean sweep for
> every workstream. That's since been downgraded to a **stretch item**. The supervisor's 2026
> guidance shifted the grading axis away from run count toward novelty depth, and the project's own
> audit found the previous "cover every box" approach was producing scaffolding rather than results.
> Run an augmentation ablation only where the base model it augments is already real and verified.
> See `Model_Training_Reference.md` and `Novelty Search.md` §4.0.
Every member repeats §8.5.1–8.5.3 in full **with** augmentation applied to their training data, so each member reports a clean augmented-vs-non-augmented comparison on their own model zoo — this is a mandatory ablation row, not optional polish, since ICBHI's small and imbalanced classes are a recurring reviewer objection and a documented augmentation ablation directly answers it.

- **Sound-event backbone** (cycle-level spectrograms): SpecAugment (time/frequency masking), following standard AST-based ICBHI practice; reported as an additional axis in the architecture ablation grid.
- **Disease-diagnosis head** (patient-level, small known-class counts): class-balanced augmentation targeting the smaller known classes (Healthy n=26, URTI n=14) — noise injection, pitch-shift, time-stretch on cycle-level audio before patient-level aggregation — evaluated on whether it improves known-class balance *without* distorting the cross-task disagreement signal.
- **Cross-dataset OOD + compression**: augmentation as a **domain-robustness lever** rather than a class-balance one — mild simulated channel/device noise on ICBHI training data, to test whether it narrows the accuracy gap on Coswara/SPRSound.
- **Calibration/trust**: augmentation evaluated for its effect on *trust* metrics, not raw accuracy — does augmenting training change calibration error and conformal coverage, in-distribution and under shift?
- All augmented runs use a **held-out, patient-independent validation/calibration split untouched by augmentation** (augmentation is applied to training folds only), consistent with the existing patient-independent evaluation protocol (§7).

### 8.5.5 Where this shows up in the paper
A single consolidated **Model Zoo & Reporting Appendix** (all members' tables, one consistent schema: metrics / confusion matrix / size / params / training time / augmented vs. non-augmented) plus per-member main-text summary tables — giving reviewers both the full transparency they increasingly expect and a clean, skimmable main narrative.

---

## 9. Timeline

| Phase | Weeks | Deliverable |
|---|---|---|
| Pilot (signal check) | 1–2 | Confirm cross-task disagreement signal exists on held-out classes |
| Task/protocol finalization | 3 | Faculty sign-off on redesigned evaluation |
| CQKD adaptation design | 4–5 | Reviewed integration plan with Dr. Khan |
| Core pipeline build | 6–9 | Working MTL + cross-task consistency + CQKD pipeline |
| Coswara/SPRSound integration | 10–12 | OOD generalization results |
| Full ablation suite (incl. augmentation ablation, §8.5.4) | 13–15 | All planned ablations complete, incl. augmented-vs-non-augmented tables |
| Statistical validation & case studies | 16–17 | LOPO + non-parametric results, per-disease tables, reconciled Model Zoo & Reporting Appendix (§8.5.5) |
| Manuscript drafting | 18–21 | Full draft |
| Internal review & revision | 22–23 | Faculty-approved draft |
| Submission | 24 | Submitted to target venue |

---

## 10. References

1. Rocha, B.M., Filos, D., Mendes, L., et al. (2019). An open access database for the evaluation of respiratory sound classification algorithms. *Physiological Measurement*, 40, 035001.
2. Rocha, B.M., Filos, D., Mendes, L., et al. (2017). A Respiratory Sound Database for the Development of Automated Classification. *International Conference on Biomedical and Health Informatics (ICBHI)*.
3. Cho, W., Lee, S. (2025). Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning. *Computers, Materials & Continua*, 84(2), 2847–2863. https://doi.org/10.32604/cmc.2025.066373
4. Suma, K.V., Koppad, D., Kumar, P., et al. (2025). Multi-task Learning for Lung Sound and Lung Disease Classification. *SN Computer Science*, 6, 51. https://doi.org/10.1007/s42979-024-03506-9
5. [Authors]. (2025). Tri-MTL: A Triple Multitask Learning Approach for Respiratory Disease Diagnosis. arXiv:2505.06271.
6. Karim, A.A.J., Mahmud, M.Z., Khan, R. (2024). Advanced vision transformers and open-set learning for robust mosquito classification: A novel approach to entomological studies. *PLOS Computational Biology*, 20(12), e1012654. https://doi.org/10.1371/journal.pcbi.1012654
7. Pavel, M.A., Islam, R., Babor, S.B., Mehadi, R., Khan, R. (2024). Non-small cell lung cancer detection through knowledge distillation approach with teaching assistant. *PLOS ONE*, 19(11), e0306441. https://doi.org/10.1371/journal.pone.0306441
8. Khan, N.A., Rafat, A.M.S. (2025). Optimizing Deep Learning Models for Resource-Constrained Environments With Cluster-Quantized Knowledge Distillation. *Engineering Reports*, 7. [Note: not authored by Dr. Riasat Khan — verify intended reference before submission.]
9. Huo, J., Muthivhi, M., van Zyl, T.L., Gustafsson, F. (2025). Nearest-Class Mean and Logits Agreement for Wildlife Open-Set Recognition. *Southern African Conference for Artificial Intelligence Research (SACAIR)*, Springer CCIS vol. 2784, 316–329.
10. Soni, P.N., Shi, S., Sriram, P.R., Ng, A.Y., Rajpurkar, P. (2022). Contrastive learning of heart and lung sounds for label-efficient diagnosis. *Patterns*, 3(1), 100400.
11. Sharma, N., Krishnan, P., Kumar, R., et al. (2020). Coswara: A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis. *Interspeech 2020*, 4811–4815.
12. Zhang, Q., Zhang, J., Yuan, J., et al. (2022). SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database. *IEEE Transactions on Biomedical Circuits and Systems*, 16(5), 867–881.
13. Fraiwan, M., et al. (2021). [KAUH lung sound dataset — verify exact citation before submission.]
14. [Authors]. (2025). Advances and Challenges in Respiratory Sound Analysis: A Technique Review Based on the ICBHI2017 Database. *Electronics*, 14(14), 2794. https://doi.org/10.3390/electronics14142794
15. Park, D.S., Chan, W., Zhang, Y., et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*. [Standard masking-based augmentation for the sound-event backbone; widely used in AST-based ICBHI pipelines.]
16. Kim, J.-W., Toikkanen, M., Bae, S., Kim, M., Jung, H.-Y. (2024). RepAugment: Input-Agnostic Representation-Level Augmentation for Respiratory Sound Classification. arXiv:2405.02996. [Reports up to 7.14% accuracy improvement on minority ICBHI disease classes; candidate augmentation technique for small known-class balancing on the disease head.]
17. Bae, S., Kim, J.-W., et al. — noise injection / time-shift / stretch / pitch-shift as standard ICBHI augmentation baselines, as surveyed in multiple 2024–2026 ICBHI robustness studies. [Candidate for the domain-robustness augmentation lever; verify primary source before submission.]

**Items flagged `[verify before submission]`** are ones where full author lists or exact page/volume details could not be fully confirmed in this research pass — standard practice before any journal submission, but flagged explicitly here rather than presented as fully certain.

---

## 12. Pre-Submission Checklist

- [ ] Confirm with Dr. Khan which paper he intends as the "mother paper" — the mosquito open-set paper (verified his) or CQKD (not his) — and correct framing accordingly
- [ ] Complete Fraiwan et al. (2021) full citation
- [ ] Complete Tri-MTL author list from arXiv:2505.06271
- [ ] Phase 0 pilot completed and signal confirmed before full build
- [ ] All small-N results verified to be presented without CIs/significance claims
- [ ] Explicit differentiation paragraph from Cho & Lee (2025) present in the Related Work section of the final manuscript
- [ ] Verify novelty claim is still current at time of submission — a July 2026 literature check found no published open-world MTL + cross-task-disagreement + staged-OWL work on respiratory audio, and no conformal/distribution-free open-set guarantee applied to respiratory sound (§4); re-check shortly before submission since this is a fast-moving area
- [ ] Confirm reference #17 (augmentation baseline survey) with a verified primary citation before submission
- [ ] Model Zoo & Reporting Appendix tables (§8.5) reconciled into one consistent schema before manuscript drafting begins — **including only audit-verified real results** (`Asif's/audit/audit_project.py`), never synthetic-data runs
