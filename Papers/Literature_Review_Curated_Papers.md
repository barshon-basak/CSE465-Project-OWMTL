# Curated Literature — Group 5 OWMTL Respiratory Diagnosis
## 15 Papers for the Literature Review (2024–present, reputable venues only)

**Selection rules applied:**
- **Recency:** every entry is **2024 or newer** (dataset-origin papers older than 2024 — ICBHI 2019, Coswara 2023, SPRSound 2022 — are cited in the proposal's reference list but deliberately excluded here per the 2024+ constraint).
- **Reputation:** journals/conferences only — *Nature Machine Intelligence*, IJCV, *Artificial Intelligence*, *Biomedical Signal Processing and Control* (the target venue), PLOS, JMIR AI, *Scientific Reports*, AAAI *AI Magazine*, Springer *SN Computer Science*, Wiley, MDPI, Elsevier.
- **Relevance:** each paper is mapped to the pillar/member it supports (A = backbone/sound-event, B = open-world mechanism, C = OOD/compression, D = trust/calibration/XAI).
- **Coverage:** ~15 total, balanced across the four workstreams — not a single-topic dump.

> ⚠️ **Verify-before-submission note:** these were located via a July 2026 literature pass. Titles/authors/venues are recorded as returned by the search, but **exact page numbers, article IDs, and full author lists must be confirmed against the publisher record** before the manuscript's reference list is finalized — the same discipline the proposal already applies to its `[verify]`-flagged entries.

---

## A. Respiratory-Sound / ICBHI Core (Members A, B)

### 1. Multi-Task Learning for Lung Sound and Lung Disease Classification
- **Authors:** Suma, K.V., Koppad, D., Kumar, P., et al.
- **Venue / Year:** *SN Computer Science* (Springer Nature), 6:51, **2025**.
- **Link:** https://link.springer.com/article/10.1007/s42979-024-03506-9
- **Why it matters:** The **single most direct MTL anchor** — jointly classifies lung *sound* and lung *disease* on ICBHI across 2D-CNN/ResNet50/MobileNet/DenseNet (MTL-MobileNet: 74% sound / 91% disease). This is the closed-set MTL baseline Members A & B build the shared-backbone-plus-two-heads architecture on top of, and the paper the novelty must differentiate from (it is closed-set only).

### 2. ADFF-Net: An Attention-Based Dual-Stream Feature Fusion Network for Respiratory Sound Classification
- **Authors:** (see publisher record — MDPI *Technologies*)
- **Venue / Year:** *Technologies* (MDPI), 14(1):12, **2025/26**.
- **Link:** https://doi.org/10.3390/technologies14010012
- **Why it matters:** State-of-the-art **AST-based, attention-fusion** sound-event model (Mel-filterbank + Mel-spectrogram dual stream). The reference point for Member A's transformer-backbone ablation and the "attention over spectrograms" design axis.

### 3. Respiratory Sounds Classification by Fusing the Time-Domain and 2D Spectral Features
- **Venue / Year:** ***Biomedical Signal Processing and Control*** (Elsevier), **2025** — *this is the project's target journal.*
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1746809425003015
- **Why it matters:** Recent SOTA on the exact 4-class ICBHI sound-event task (time-domain + 2D time-frequency co-attention fusion, +1.00% ICBHI Score). Doubly useful: a Member-A benchmark **and** a scope/style template for the venue you are submitting to.

### 4. Advances and Challenges in Respiratory Sound Analysis: A Technique Review Based on the ICBHI 2017 Database
- **Venue / Year:** *Electronics* (MDPI), 14(14):2794, **2025**.
- **Link:** https://doi.org/10.3390/electronics14142794
- **Why it matters:** The **survey that frames the whole field** — documents the 135-publication saturation of ICBHI that the proposal's novelty argument leans on. Ideal opening citation for the Related Work "why this benchmark is crowded" paragraph.

---

## B. Open-Set / Open-World Recognition (Member B, framing)

### 5. Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning
- **Authors:** Cho, W., Lee, S.
- **Venue / Year:** *Computers, Materials & Continua*, 84(2):2847–2863, **2025**.
- **Link:** https://doi.org/10.32604/cmc.2025.066373
- **Why it matters:** **The closest direct prior art** — the only open-set-recognition paper on ICBHI respiratory audio. Uses prototype/distance-based rejection (single-task), so it is the paper Member B must explicitly differentiate from (cross-task disagreement + staged OWL, not prototype distance). Non-negotiable citation.

### 6. Challenges, Evaluation and Opportunities for Open-World Learning
- **Authors:** Kejriwal, M., Kildebeck, E., Steininger, R., Shrivastava, A.
- **Venue / Year:** ***Nature Machine Intelligence***, 6(6):580–588, **2024**.
- **Link:** https://www.nature.com/articles/s42256-024-00852-4
- **Why it matters:** Top-venue conceptual grounding for the **open-world learning (OWL) protocol** itself — defines detect/characterize/incrementally-learn novelty, which is exactly the Stage 0→1→2 structure of Member B's plan. Lends the staged-OWL framing institutional weight.

### 7. Open Issues in Open World Learning
- **Authors:** Cruz, S., Doctor, K., Funk, C., Scheirer, W.
- **Venue / Year:** *AI Magazine* (AAAI), 46(2), **2025**.
- **Link:** https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.70001
- **Why it matters:** Authored by **Scheirer** (originator of open-set recognition); a current, authoritative map of unresolved OWL problems. Positions the project against the recognized frontier and supports the "novel application, known mechanism" honesty the guideline insists on.

### 8. Generalized Out-of-Distribution Detection: A Survey
- **Authors:** Yang, J., Zhou, K., Li, Y., Liu, Z.
- **Venue / Year:** *International Journal of Computer Vision* (IJCV), 132, **2024**.
- **Link:** https://link.springer.com/article/10.1007/s11263-024-02117-4
- **Why it matters:** The **canonical taxonomy** unifying OOD detection / open-set / novelty / anomaly. Essential for precisely situating "cross-task disagreement as an unknown detector" (Member B) and the Coswara/SPRSound OOD framing (Member C) within an agreed vocabulary — reviewers expect this survey to be cited.

### 9. Open-World Continual Learning: Unifying Novelty Detection and Continual Learning
- **Authors:** Kim, G., Xiao, C., Konishi, T., Ke, Z., Liu, B.
- **Venue / Year:** ***Artificial Intelligence*** (Elsevier), 338:104237, **2025**.
- **Why it matters:** Directly underpins Member B's **Stage-2 incremental step and forgetting-curve** metric — formalizes the link between novelty detection and catastrophic forgetting that the OWL protocol measures. (Bing Liu's group; top-tier AI journal.)

---

## C. Dr. Khan's Lab Lineage + Compression / Distillation (Members B, C)

### 10. Advanced Vision Transformers and Open-Set Learning for Robust Mosquito Classification
- **Authors:** Karim, A.A.J., Mahmud, M.Z., Khan, R.
- **Venue / Year:** *PLOS Computational Biology*, 20(12):e1012654, **2024**.
- **Link:** https://doi.org/10.1371/journal.pcbi.1012654
- **Why it matters:** **The verified methodological anchor from the supervising lab** — its OpenMax/Weibull open-set machinery is the exact competing baseline Member B reimplements and benchmarks cross-task disagreement against. Establishes institutional continuity.

### 11. Non-Small Cell Lung Cancer Detection Through Knowledge Distillation Approach with Teaching Assistant
- **Authors:** Pavel, M.A., Islam, R., Babor, S.B., Mehadi, R., Khan, R.
- **Venue / Year:** *PLOS ONE*, 19(11):e0306441, **2024**.
- **Link:** https://doi.org/10.1371/journal.pone.0306441
- **Why it matters:** The lab's applied **knowledge-distillation** track record on medical signals — background/support for Member C's CQKD compression pillar and evidence the team can execute a distillation pipeline.

### 12. Optimizing Deep Learning Models for Resource-Constrained Environments with Cluster-Quantized Knowledge Distillation (CQKD)
- **Authors:** Khan, N.A., Rafat, A.M.S.
- **Venue / Year:** *Engineering Reports* (Wiley), 7, **2025**.
- **Link:** https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70187
- **Why it matters:** The **external CQKD technique Member C adapts** for edge-deployable compression and OWL-drift regularization. ⚠️ *Attribution flag (already in the proposal): confirm this is not authored by Dr. Riasat Khan before framing it as lab lineage — cite as adapted external work.*

---

## D. Trust, Calibration & Cross-Dataset Reality (Members C, D — supports the proposed 4th member)

### 13. Deep Learning-Driven Early Diagnosis of Respiratory Diseases using CNN-RNN Fusion on Lung Sound Data
- **Venue / Year:** ***Scientific Reports*** (Nature Portfolio), s41598-025-28832-7, **2025**.
- **Link:** https://www.nature.com/articles/s41598-025-28832-7
- **Why it matters:** Uses **both ICBHI and Coswara** (validating Member C's exact dataset pivot) *and* applies **Grad-CAM + LIME + SHAP** explainability. This is the paper Member D must differentiate from — it does *generic* XAI on closed-set labels; Member D instead explains the *open-world rejection decision*. Cite it to justify the dataset combo and to stake out the unclaimed XAI angle.

### 14. Improving the Robustness and Clinical Applicability of Automatic Respiratory Sound Classification Using Deep Learning-Based Audio Enhancement
- **Venue / Year:** *JMIR AI*, 2025:e67239, **2025**.
- **Link:** https://ai.jmir.org/2025/1/e67239
- **Why it matters:** Quantifies performance collapse under real **clinical noise** and recovery via enhancement (+21.88% ICBHI Score, P<.001). Anchors Member D's noise-robustness / calibration-under-degradation stretch task and the general "does it survive deployment conditions" argument.

### 15. Classification with Reject Option: Distribution-Free Error Guarantees via Conformal Prediction
- **Authors:** Szabadváry, J.H., et al.
- **Venue / Year:** *Machine Learning with Applications* (Elsevier), **2025** (arXiv:2506.21802).
- **Link:** https://www.sciencedirect.com/science/article/pii/S2666827025000477
- **Why it matters:** Provides the **distribution-free guarantee on the abstain/reject decision** that underpins the proposed 4th member's contribution — turning Member B's hand-tuned disagreement threshold into a formally bounded false-flag rate. The theoretical backbone of the "trustworthy open-world" angle.

---

## Coverage Map (at a glance)

| # | Paper (short) | Venue | Year | Serves |
|---|---|---|---|---|
| 1 | Suma — MTL lung sound+disease | SN Computer Science | 2025 | A, B |
| 2 | ADFF-Net | Technologies (MDPI) | 2025 | A |
| 3 | Time-domain + spectral fusion | Biomed. Signal Process. Control | 2025 | A (+ venue) |
| 4 | ICBHI technique review | Electronics (MDPI) | 2025 | Framing |
| 5 | Cho & Lee — open-set respiratory | Computers, Materials & Continua | 2025 | B |
| 6 | Kejriwal — OWL challenges | Nature Machine Intelligence | 2024 | B |
| 7 | Cruz/Scheirer — open issues OWL | AI Magazine (AAAI) | 2025 | B |
| 8 | Yang — Generalized OOD survey | IJCV | 2024 | B, C, D |
| 9 | Kim/Liu — open-world continual | Artificial Intelligence | 2025 | B |
| 10 | Karim & Khan — mosquito open-set | PLOS Comput. Biol. | 2024 | B (lab) |
| 11 | Pavel & Khan — KD lung cancer | PLOS ONE | 2024 | C (lab) |
| 12 | Khan & Rafat — CQKD | Engineering Reports (Wiley) | 2025 | C |
| 13 | CNN-RNN fusion (ICBHI+Coswara+XAI) | Scientific Reports | 2025 | C, D |
| 14 | Audio-enhancement robustness | JMIR AI | 2025 | D |
| 15 | Conformal reject option | Machine Learning w/ Applications | 2025 | D |

**Balance:** 4 respiratory-core · 5 open-world/OSR · 3 lab-lineage/compression · 3 trust/robustness/XAI — every member's workstream has ≥2 anchor papers, no single topic dominates.
