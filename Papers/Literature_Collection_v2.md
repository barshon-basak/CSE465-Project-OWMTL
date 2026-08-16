# Literature Collection v2 — Physics-Grounded Acoustic Concept Bottleneck + Faithfulness Audit

> **Rebuilt from scratch (2026-08-14) for the CURRENT direction.** The previous 15-paper list (`Papers/Literature_Review_Curated_Papers.md`) was for the retired cross-task-disagreement / OWMTL framing and is **discarded** — do not merge these. This collection is scoped to: a *physics-grounded, label-free acoustic concept bottleneck* for respiratory diagnosis, used to test *concept faithfulness/leakage*, *concept-space novelty detection*, *device/pediatric robustness*, and *foundation-model probing* ("do models actually listen to the clinical sounds?").
>
> **Rules applied:** 2023+ only (dataset-origin exceptions flagged in §H); relevance over coverage; peer-reviewed preferred, key preprints allowed; **code links only where a real repo was verified** — none invented; every paper carries a **novelty-validation tag**.
>
> **Novelty-validation tags:** 🟩 *supports novelty* · 🟦 *baseline* · 🟥 *competing/existing approach* · 🕳️ *reveals gap* · 📊 *result comparison* · 🧪 *dataset/evaluation*.
>
> **Verify-before-submission:** author lists and exact page/volume for several entries are recorded as returned by search; confirm against the publisher record before the manuscript reference list is finalized (the same discipline your project already applies).

---

## A. Core / directly related — concept-bottleneck & interpretable-by-design

### A1. Label-Free Concept Bottleneck Models 🟩🟦
- **Year:** 2023 · **Authors:** Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, Tsui-Wei Weng · **Venue:** ICLR 2023
- **Link:** https://arxiv.org/abs/2304.06129 · **Code:** https://github.com/Trustworthy-ML-Lab/Label-free-CBM
- **Method/contribution:** Builds a concept bottleneck **without hand-labeled concepts**, generating them from CLIP/LLMs — the canonical "label-free CBM."
- **Relevance:** The methodological basis for a *label-free* bottleneck — exactly the class your project sits in.
- **Supports which part:** Method foundation **and** the key differentiation: your concepts are **physics/DSP-derived and clinically-named**, not CLIP/LLM-derived — so this is what you cite to say "label-free CBM exists, but not clinically grounded for lung sound."
- **Key results for comparison:** Accuracy-vs-interpretability tradeoff methodology; concept-set construction.
- **Limitations/gap for you:** Concepts are semantic/visual, not signal-physics; no clinical grounding, no leakage-under-shift analysis.

### A2. Voice Concept Bottleneck for Interpretable Health Assessment 🟥🟩
- **Year:** 2026 · **Authors:** *(verify author list)* · **Venue:** arXiv (audio-language-model framework)
- **Link:** https://arxiv.org/abs/2607.16967 · **Code:** *(none found)*
- **Method/contribution:** A concept-bottleneck framework for **voice**-based health assessment via an audio-language model.
- **Relevance:** The **nearest neighbor** — a concept bottleneck for *health audio*, but a different modality (voice, not auscultation) and LLM-derived concepts.
- **Supports which part:** Novelty differentiation — proves the "audio CBM" idea is emerging but **not for lung-sound adventitious concepts**; you must cite and distinguish.
- **Limitations/gap:** Voice ≠ auscultation; concepts language-derived, not physics-measured; no device/pediatric robustness study.

### A3. Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis 🟥
- **Year:** 2024 · **Authors:** *(verify)* · **Venue:** arXiv (medical imaging)
- **Link:** https://arxiv.org/abs/2410.15446 · **Code:** *(none found)*
- **Method/contribution:** A CBM variant for medical **image** diagnosis that complements missing concepts.
- **Relevance:** Represents the crowded medical-CBM bandwagon your revalidation flagged.
- **Supports which part:** Differentiation — "medical CBMs exist for imaging; porting to lung sound is not the contribution; the physics-derived-concept + faithfulness question is."
- **Limitations/gap:** Imaging modality; annotated/complemented concepts, not signal-physics; no auscultation.

---

## B. Existing state-of-the-art — respiratory foundation models & top classifiers

### B1. OPERA: Towards Open Respiratory Acoustic Foundation Models 🟦📊
- **Year:** 2024 · **Authors:** Yuwei Zhang, Tong Xia, Jing Han, et al. *(verify full list)* · **Venue:** NeurIPS 2024 (Datasets & Benchmarks)
- **Link:** https://arxiv.org/abs/2406.16148 · **Code:** https://github.com/evelyn0414/OPERA · **Site:** https://opera-benchmark.github.io/
- **Method/contribution:** A respiratory acoustic **foundation model** pretrained on ~136K samples with a 19-task benchmark; demonstrates generalization to unseen datasets.
- **Relevance:** The FM you probe in Gate G5 ("do foundation models listen?") and the strongest baseline you must out-reason (answers your deferred "Attack 7").
- **Dataset/eval:** OPERA benchmark incl. ICBHI-adjacent tasks.
- **Key results for comparison:** Benchmark averages; unseen-dataset generalization claims.
- **Limitations/gap:** Benchmarks are closed-set/regression; **no concept-faithfulness or novelty-rejection probing** — your opening.

### B2. Towards Pre-training an Effective Respiratory Audio Foundation Model (M2D-Resp) 🟦📊🧪
- **Year:** 2025 · **Authors:** Daisuke Niizumi, Daiki Takeuchi, et al. *(verify)* · **Venue:** Interspeech 2025
- **Link:** https://arxiv.org/abs/2505.15307 · **Code:** https://github.com/nttcslab/m2d (see `app/icbhi_sprs`)
- **Method/contribution:** Masked-modeling-duo pretraining for respiratory audio; pushes the OPERA benchmark average to ~0.814.
- **Relevance:** Current SOTA FM **evaluated on ICBHI *and* SPRSound** — directly your datasets; a probing target and result-comparison anchor.
- **Dataset/eval:** ICBHI + SPRSound (pediatric) — the exact pair your covariate-shift study uses.
- **Key results:** Benchmark-average 0.814; ICBHI/SPRSound numbers to compare against.
- **Limitations/gap:** Closed-set performance focus; no faithfulness/leakage; no interpretable concept layer.

### B3. HeAR: Health Acoustic Representations 🟦
- **Year:** 2024 · **Authors:** Sebastien Baur, et al. (Google) *(verify)* · **Venue:** arXiv / Google Health AI
- **Link:** https://arxiv.org/abs/2403.02522 · **Code/model:** https://github.com/Google-Health/google-health (health_acoustic_representations) · https://huggingface.co/google/hear
- **Method/contribution:** A bioacoustic foundation model over health sounds (cough/breath/speech).
- **Relevance:** Second FM option for the probing arm; strengthens "does the FM encode clinical concepts?"
- **Limitations/gap:** General health-acoustic, not auscultation-specific; access via API/HF; no concept interpretability.

### B4. Respiratory Sounds Classification by Fusing Time-Domain and 2D Spectral Features 📊
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** *Biomedical Signal Processing and Control* (Elsevier)
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S1746809425003015 · **Code:** *(none found)*
- **Method/contribution:** Time-domain + 2D time-frequency co-attention fusion; recent SOTA on the 4-class ICBHI sound task.
- **Relevance:** Result-comparison anchor **and** a style/scope template for a likely target venue.
- **Key results:** ~+1.0% ICBHI Score over prior fusion nets — your closed-set comparison point.
- **Limitations/gap:** Closed-set, accuracy-focused; no interpretability/faithfulness.

### B5. ADFF-Net: Attention-Based Dual-Stream Feature Fusion Network 📊
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** *Technologies* (MDPI), 14(1):12
- **Link:** https://doi.org/10.3390/technologies14010012 · **Code:** *(none found)*
- **Method/contribution:** Attention dual-stream (mel-filterbank + mel-spectrogram) fusion for respiratory sound classification.
- **Relevance:** Represents the saturated closed-set fusion SOTA (~0.82–0.86 macro) your project must *not* try to beat on accuracy.
- **Supports which part:** Result comparison + the "accuracy is saturated, contribution must be rigor/interpretability" argument.
- **Limitations/gap:** Closed-set; no open-world/interpretability.

---

## C. Supporting the novelty claim — leakage/faithfulness & physics grounding

### C1. Measuring Leakage in Concept-Based Methods: An Information-Theoretic Approach 🟩
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2504.09459 · **Code:** *(none found)*
- **Method/contribution:** An information-theoretic estimator of **concept leakage** — how much a model bypasses the bottleneck.
- **Relevance:** The **core method for your leakage/faithfulness contribution** (I2) — the tool you apply, first, to respiratory audio.
- **Supports which part:** Directly enables the "does the model actually use the sounds?" measurement.
- **Limitations/gap:** General domain, not respiratory/clinical — your instantiation is the novelty.

### C2. Leakage and Interpretability in Concept-Based Models 🟩
- **Year:** 2025 · **Authors:** Enrico Parisini, et al. *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2504.14094 · **Code:** *(none found)*
- **Method/contribution:** Analyzes how leakage undermines the interpretability guarantee of CBMs.
- **Relevance:** Framing support for *why* faithfulness must be measured, not assumed.
- **Supports which part:** Motivates the whole faithfulness axis; pair with C1.
- **Limitations/gap:** Not clinical/audio.

### C3. Physics-Informed Interpretable Respiratory Inversion (Neural Operators) 🟩
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** *Scientific Reports* (Nature Portfolio)
- **Link:** https://www.nature.com/articles/s41598-026-40470-1 · **Code:** *(none found)*
- **Method/contribution:** Physics-informed neural operators for vocal-tract/respiratory **inversion** (a different task).
- **Relevance:** Establishes the credibility of **physics-grounded** respiratory-audio modeling — supports your "concepts from signal physics" premise.
- **Supports which part:** The physics-grounding half of the novelty (different task, so no conflict).
- **Limitations/gap:** Inversion, not diagnosis-via-concepts — the gap you fill.

### C4. Improving Respiratory Sound Analysis with Frequency Selection and Attention 🟩📊
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2507.20052 · **Code:** *(none found)*
- **Method/contribution:** Exploits the spectral signatures of crackles (broadband transients) and wheezes (narrowband tonal) via frequency selection + attention.
- **Relevance:** Empirical support that your **physics concepts** (PAPR for transients, spectral flatness/band for tonals) are genuinely discriminative.
- **Supports which part:** Grounds the concept extractors; also a closed-set result-comparison point.
- **Limitations/gap:** Uses spectral priors implicitly inside a black box — not an interpretable, interveneable concept layer.

---

## D. Baselines & comparison methods — open-set / OOD

### D1. Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning (Cho & Lee) 🟥🟦
- **Year:** 2025 · **Authors:** Wocheol Cho, Sangjun Lee · **Venue:** *Computers, Materials & Continua* 84(2):2847–2863
- **Link:** https://www.techscience.com/cmc/v84n2/62937 · **Code:** *(none found)*
- **Method/contribution:** Open-set respiratory recognition via a distance-based prototype network under semi-supervised learning.
- **Relevance:** The **single closest open-set respiratory prior** — must be cited and explicitly differentiated (they use single-task prototype rejection; you use physics-concept-space novelty + faithfulness).
- **Supports which part:** Baseline/competing method for your open-world evaluation section.
- **Key results:** Their open-set AUROC range is your comparison target.
- **Limitations/gap:** Single-task; no concept layer; no device/pediatric confound separation.

### D2. Generalized Out-of-Distribution Detection: A Survey 🟦🧪
- **Year:** 2024 · **Authors:** Jingkang Yang, Kaiyang Zhou, Yixuan Li, Ziwei Liu · **Venue:** *International Journal of Computer Vision* (IJCV) 132
- **Link:** https://link.springer.com/article/10.1007/s11263-024-02117-4 · **Code:** *(survey — n/a)*
- **Method/contribution:** The canonical taxonomy unifying OOD / open-set / novelty / anomaly detection, incl. semantic-vs-covariate shift.
- **Relevance:** The vocabulary and the baseline family (MSP/Energy/Mahalanobis) you run **in concept space** vs embedding space.
- **Supports which part:** Evaluation framing + baseline definitions; positions your concept-space novelty within an agreed taxonomy.
- **Limitations/gap:** Vision-centric; no clinical-acoustic instantiation.

---

## E. Dataset & evaluation-related

### E1. Advances and Challenges in Respiratory Sound Analysis: ICBHI 2017 Technique Review 🧪🕳️
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** *Electronics* (MDPI) 14(14):2794
- **Link:** https://www.mdpi.com/2079-9292/14/14/2794 · **Code:** *(n/a)*
- **Method/contribution:** Survey of ~135 ICBHI publications — preprocessing, methods, evaluation practices.
- **Relevance:** Your opening "why this benchmark is saturated" citation and a map of standard evaluation.
- **Supports which part:** State-of-the-art + gap framing (accuracy is saturated → rigor/interpretability contribution).
- **Limitations/gap:** Documents the saturation your project responds to.

### E2. BCoughBench: Benchmarking Respiratory Acoustic Foundation Models (reliability) 🕳️🧪
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2606.25116 · **Code:** *(verify)*
- **Method/contribution:** Benchmarks respiratory FMs under wearable conditions; explicitly notes FMs "report AU-ROC without sensitivity or calibration, concealing failure at the operating point," and small-N reliability issues.
- **Relevance:** The field's **own admission** of the operating-point/calibration gap your honest-evaluation contribution (I6) fills.
- **Supports which part:** Gap + evaluation-standard justification.
- **Limitations/gap:** Cough/wearable focus, not auscultation faithfulness.

> **Required dataset-origin citations (pre-2023 — cite anyway, outside the 2023+ rule):**
> ICBHI 2017 — Rocha et al., *Physiol. Meas.* 2019. · SPRSound (pediatric) — Zhang et al., *IEEE TBioCAS* 2022. · Coswara — Sharma et al., *Interspeech* 2020. These are unavoidable primary-dataset references; they are not "literature-review" papers and sit outside the 2023+ scope by necessity.

---

## F. Research-gap / limitation papers

### F1. Mitigating Stethoscope-Induced Shortcuts under Federated Domain Generalization (Causality-Inspired) 🕳️🟥
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2605.29862 · **Code:** *(verify)*
- **Method/contribution:** Leave-one-device-out on ICBHI/SPRSound; removes device shortcuts via gradient alignment + counterfactual augmentation.
- **Relevance:** Defines the **device-confound gap** and occupies the *classifier*-robustness lane — you must cite it and show your contribution is the *detector/faithfulness* angle, not classifier de-biasing.
- **Supports which part:** Gap boundary for your covariate-shift analysis (G4); differentiation.
- **Limitations/gap:** Targets classifier generalization, **not** whether the novelty detector or concept layer is confounded.

### F2. BTS-CARD: Counterfactual Adversarial Debiasing for Multimodal Respiratory Sound Classification 🕳️🟥
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** ICASSP 2026
- **Link:** https://arxiv.org/abs/2510.22263 · **Code:** *(verify)*
- **Method/contribution:** Removes device/metadata spurious correlations via counterfactual + adversarial debiasing → device-invariant representations.
- **Relevance:** The methodological half of "device invariance" is occupied — supports scoping your contribution away from it.
- **Supports which part:** Gap/differentiation (don't build a device-invariant *classifier*; audit the *concept layer*).
- **Limitations/gap:** Classifier-level invariance; no concept faithfulness.

### F3. A Device-Invariant Multi-Modal Learning Framework for Respiratory Disease Classification 🕳️🟥📊
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** *npj Digital Medicine* (s41746-026-02445-4)
- **Link:** https://www.nature.com/articles/s41746-026-02445-4 · **Code:** *(verify)*
- **Method/contribution:** Adversarial device-invariance + IRM-augmented loss for robustness across acquisition devices.
- **Relevance:** Top-tier evidence the device-invariance mechanism is published — this is why Direction 1 (device disentanglement) was rejected; keep the boundary clear.
- **Supports which part:** Gap boundary + result comparison for any device-robustness numbers.
- **Limitations/gap:** Classifier robustness, not detector-faithfulness or interpretable concepts.

### F4. Explainable Multi-Modal Deep Learning for Respiratory Audio under Domain Shift 🕳️
- **Year:** 2025/2026 · **Authors:** *(verify)* · **Venue:** *Life* (MDPI) 16(7):1108 · (companion preprint: arXiv:2512.00563)
- **Link:** https://www.mdpi.com/2075-1729/16/7/1108 · **Code:** *(none found)*
- **Method/contribution:** Multi-modal respiratory classification with **post-hoc XAI (Grad-CAM/IG/SHAP)** and domain-shift evaluation; claims attribution to acoustic biomarkers.
- **Relevance:** **The motivating gap for Path B** — it *asserts* the model attends to biomarkers (e.g., "300–1500 Hz") but never **verifies** it. Your faithfulness audit is exactly this verification.
- **Supports which part:** Your intro/gap paragraph ("papers claim biomarker attribution but don't verify — we do") + differentiation (post-hoc XAI vs interpretable-by-design bottleneck).
- **Limitations/gap:** Post-hoc, unverified attribution; closed-set; no leakage/intervention.

### F5. Pitfalls of Conformal Predictions for Medical Image Classification 🕳️🟩
- **Year:** 2025 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2506.18162 · **Code:** *(none found)*
- **Method/contribution:** Shows conformal prediction can be fragile/invalid for medical classification, esp. selective issuance.
- **Relevance:** Directly reframes your **M14 conformal paradox** (95% coverage, 0% detection) from a failure into a characterized, cited limitation.
- **Supports which part:** The honest-operating-point section (I6) — "when conformal is and isn't valid for respiratory OOD."
- **Limitations/gap:** Imaging; your respiratory instantiation is the contribution.

---

## G. Other supporting — foundation-model faithfulness / "does it listen" genre

### G1. HearSay: Do Audio LLMs Leak What They Hear? 🟩
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2601.03783 · **Code:** *(verify)*
- **Method/contribution:** Benchmarks whether audio LLMs actually use/leak the audio content.
- **Relevance:** Precedent for the **"does the model listen?"** framing in audio; shows the genre is timely (cite to establish currency, and to keep your framing sharply respiratory-specific).
- **Supports which part:** Faithfulness-audit framing (adjacent, non-clinical).
- **Limitations/gap:** Audio-LLM, not auscultation concepts.

### G2. StethoLM: Audio Language Model for Cardiopulmonary Analysis 🟥
- **Year:** 2026 · **Authors:** *(verify)* · **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2603.00355 · **Code:** https://github.com/yishani/StethoLM
- **Method/contribution:** An audio-language model spanning cardiopulmonary clinical tasks.
- **Relevance:** Occupies the **ALM-for-lung-sound** lane — cite so you don't overclaim; use CLAP/LLM-derived concepts only as a *contrast baseline*, not a headline.
- **Supports which part:** Differentiation (don't build an ALM; your novelty is the physics-concept faithfulness study).
- **Limitations/gap:** LLM/language-grounded, not physics-grounded concepts; no faithfulness/leakage measurement.

---

## Selection audit (why the list is this size, and what was deliberately excluded)

- **~22 papers, 7 purposes** — each maps to a specific need: method basis (A), SOTA/baseline you compare or probe (B), the tools that make the novelty (C), open-set baselines (D), datasets/evaluation (E), the gaps you exploit (F), and the timely framing genre (G).
- **Deliberately excluded** (relevant-sounding but not needed for *this* direction): generic ICBHI closed-set classifiers beyond one SOTA anchor; OpenMax/Energy *method* papers (pre-2023 and already implemented as your M6/M29 baselines); MTL-on-ICBHI papers (retired direction); vision-only CBM variants beyond one differentiation anchor; edge-compression/distillation papers (your M16/M18 are footnotes, not a contribution); test-time-adaptation papers (occupied lane, not your thesis). Your old 15-paper list's MTL/OWL/CQKD/lab-lineage entries are dropped entirely.
- **Code reality:** verified repos exist for OPERA, M2D, HeAR, Label-free CBM, StethoLM. All other entries have **no code link** (not invented). A few entries are marked *(verify)* where a repo may exist but wasn't confirmed.

*Compiled 2026-08-14 via a fresh 2023+ literature pass. Confirm author lists / exact page numbers against publisher records before the manuscript reference list is finalized, and re-run the search near submission — this subfield moves monthly.*
