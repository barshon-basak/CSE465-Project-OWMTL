> **Status: ARCHIVED 2026-08-05. Superseded by the consolidated root-level `Novelty Search.md`.** This file
> (July 2026) remains valid literature grounding — nothing in v3 contradicts it — but v3 reconciles
> it against Dr. Khan's supervisor-suggested technique list and the project's current real state
> (M2/M3/M12/M29 completed; 9 downstream models found running on synthetic data). Read v3 first.

## 1. Summary of the Existing Proposal

The project proposes an **Open-World Multi-Task Learning (OWMTL)** framework for respiratory disease diagnosis: a shared encoder feeds a sound-event head (crackle/wheeze/normal) and a disease-diagnosis head, and disagreement between what the two heads *imply* about diagnosis is used to flag patients with diseases the model never saw in training. This is tested through a three-stage open-world protocol (train on knowns → threshold on disagreement → incrementally absorb confirmed unknowns while checking for forgetting), evaluated on a coarse known-vs-pooled-unknown ICBHI split plus large-N cross-dataset stress tests on Coswara and SPRSound. A cluster-quantized distillation step (adapted from Khan & Rafat 2025) compresses the model and is tested as a possible regularizer against forgetting. v2 adds a mandatory standardized reporting protocol (full metrics, efficiency numbers, training curves, and an augmentation ablation) across all four team members' workstreams — a rigor upgrade, not a scope change.

This is an unusually well-scaffolded proposal — the related-work section already pre-empts the two most obvious reviewer objections (Cho & Lee 2025 as closest prior art; Huo et al. 2025 as proof the disagreement mechanism itself isn't new).

## 2. Current Novelty Already Present

Per your own §4 table, the claim rests on three legs:
- **Domain gap**: no MTL-on-ICBHI paper handles unseen diseases (all are closed-set).
- **Mechanism combination**: cross-task disagreement + staged OWL, applied to respiratory audio, is not present in Cho & Lee (2025) or anywhere else you found.
- **Statistical redesign**: coarse open-world task + large-N OOD stress tests, sidestepping ICBHI's small per-disease classes (independently justified by Soni et al. 2022).

This is a legitimate, defensible novelty claim — it's a domain-transfer-plus-combination contribution, not a new algorithm. That's a normal and publishable pattern, but it means the paper's strength depends heavily on execution depth (ablations, stress tests, honest framing) rather than algorithmic surprise. My research below mostly confirms this claim still holds as of July 2026, with a few gaps worth closing.

## 3. Literature Review and State-of-the-Art Analysis

**ICBHI/respiratory-specific (2023–2026):** I found no new open-set/unknown-disease paper on ICBHI beyond Cho & Lee (2025) — including checking a June 2026 systematic review of AI-based respiratory disease detection and a February 2026 multimodal respiratory diagnosis agent paper, neither of which touches open-world/unknown-class detection. A June 2026 systematic review covers audio and visual unimodal methods through multimodal integration for respiratory disease detection without any open-set framing. Your claim holds.

**Cross-task/multi-head disagreement as an OOD signal (general ML):** This is more established outside audio than your v2 draft implies. One 2022 paper builds a multi-task network with a rotation-prediction auxiliary head alongside semantic classification specifically because it observed a strong relationship between auxiliary-task accuracy and OOD detection accuracy. In medical imaging specifically, a CNN-based medical image segmentation paper shows multi-task learning improves confidence calibration and proposes a spectral-analysis method for OOD detection — MTL-for-OOD in a clinical setting already exists, just via a different mechanism and modality (segmentation, not audio classification). This doesn't break your novelty claim, but it means your related-work section should cite this alongside Huo et al. — right now it only cites a wildlife paper as evidence the mechanism generalizes, when a medical-imaging precedent is a stronger, more relevant citation for reviewers to recognize.

**Open-world learning generally:** A 2025 review of open world learning notes that meaningful progress has been made in the field's ability to detect, characterize, and incrementally learn novelty in dynamic environments, though novelty detection itself remains an open challenge. The field's energy is overwhelmingly in vision (object detection benchmarks like OWOD), which reinforces that your audio/medical application is genuinely underexplored territory rather than a crowded one.

**Conformal/distribution-free guarantees for open-set recognition:** This is the most interesting gap I found. A 2025 paper on conformal inference explicitly targets open-set and imbalanced classification, building on decades of open-set recognition work spanning audio, medical, and forensic domains — but it's a general statistical-ML paper with no audio or respiratory application. Your checklist already flags this ("no conformal/distribution-free open-set guarantee applied to respiratory sound") — that flag is correct and is a genuine, currently-open gap.

**Calibration for audio classifiers (relevant to your "ensemble/Bayesian/SNGP/evidential variants" ablation row):** Existing work investigates SNGP and ensemble calibration methods for deep audio classifiers, and a related benchmark found spectrally-normalized Gaussian processes achieve strong calibration with good computational efficiency compared to Monte Carlo dropout, deep ensembles, and focal loss — but on environmental sound and music, not respiratory/medical audio. This is a transferable technique, not a competing paper.

**Test-time adaptation for cross-dataset audio shift:** relevant to your Coswara/SPRSound stress tests. Recent work on test-time adaptation addresses distribution shift between training and test data by adapting a model at inference using only unlabeled test data — general TTA methods exist for exactly the kind of train-on-ICBHI/test-on-Coswara shift your Task C describes, and I found no application of TTA to respiratory OOD detection.

## 4. Research Gaps Identified

| Gap | Status | Confidence |
|---|---|---|
| Open-world/unknown-disease detection on ICBHI (beyond Cho & Lee) | Confirmed still open as of July 2026 | High |
| Distribution-free (conformal) open-set guarantee for respiratory audio | Confirmed open — general conformal open-set methods exist, unapplied here | High |
| Automated/learned augmentation policy search on ICBHI | Confirmed open — all ICBHI augmentation work uses hand-picked techniques | High |
| Test-time adaptation applied to the Coswara/SPRSound OOD stress test | Not found in respiratory literature | Medium (didn't exhaustively search bioacoustics TTA) |
| MTL+OOD precedent in medical imaging (different mechanism/modality) | Exists — strengthens related-work citation, doesn't threaten novelty | High |

## 5. Additional Novelty Ideas (Ranked by Impact)

**1. Add a conformal/distribution-free layer on top of the cross-task consistency score.** Instead of only thresholding the disagreement score, wrap it in a conformal calibration step that gives a statistically guaranteed unknown-detection error rate (not just a point accuracy). Conformal methods for open-set and imbalanced classification are an active 2025 research thread, and nothing applies this to respiratory audio. This directly answers Member D's calibration/trust angle already in your ablation table, converts "we measured AUROC" into "we guarantee coverage," and is a genuinely differentiating claim reviewers in a Q1 signal-processing venue would find credible. **Feasibility: high** — conformal calibration is a post-hoc wrapper, not a retrain.

**2. Automated augmentation policy search as a fifth "member" contribution, not just a manual ablation row.** Your §8.5.4 currently hand-assigns SpecAugment/noise-injection/pitch-shift per member. Replacing (or supplementing) this with a learned policy search — the audio-specific analog of AutoAugment — is both what your supervisor likely meant by "Auto Augmentation Library" (see §7) and a citable point of departure from every other ICBHI augmentation paper, all of which use fixed, hand-picked recipes. **Feasibility: high**, low compute if using a lightweight search like RandAugment-style random magnitude sampling rather than a full RL controller.

**3. Test-time adaptation on the Coswara/SPRSound stress tests.** Rather than a static train-once-evaluate-once OOD test, apply lightweight TTA (entropy minimization or batch-norm recalibration) at inference on Coswara/SPRSound before scoring cross-task disagreement, and report both with and without TTA. This turns your "scale" evidence into a second axis of contribution (how much does adaptation close the domain gap) without requiring labels on the target sets. **Feasibility: medium** — needs care that TTA doesn't distort the disagreement signal itself (worth an explicit ablation, similar to how you already re-check unknown-detection numbers post-augmentation for Member B).

**4. Reframe the medical-imaging MTL-for-OOD precedent as a fourth "closest prior art" citation.** Not a new experiment — a related-work fix. Citing the spectral-feature MTL-OOD medical imaging paper alongside Huo et al. pre-empts a reviewer who works in medical AI (more likely than one who works in wildlife camera traps) from flagging it as an oversight.

**5. Formal comparison of cross-task consistency vs. evidential deep learning as a third rejection mechanism.** You already compare against OpenMax/Weibull (Khan et al. 2024 lineage). Evidential deep learning has an established open-set-recognition track record and would let you report three rejection mechanisms rather than two, strengthening the "why does our specific mechanism matter" section. **Feasibility: medium** — adds one more ablation arm, reuses your existing pipeline.

## 6. Transferable Methods from Other Papers

- **SNGP-based calibration** (from general audio classifier calibration work) — drop-in replacement or complement for your softmax-based disagreement scoring, likely to improve the known-class confidence numbers Member D needs anyway.
- **Entropy-minimization TTA** — a few-line addition at inference time for the Coswara/SPRSound stress tests (idea #3 above).
- **Conformal calibration wrapper** — post-hoc, doesn't touch your training pipeline, bolts onto the existing cross-task consistency score (idea #1 above).
- **Bayesian-optimization-based composed augmentation policies**, as used for general audio classification — directly adaptable to Member A's spectrogram-level augmentation search instead of (or alongside) fixed SpecAugment.

None of these require a new research direction — all are adapters on your existing architecture, consistent with your supervisor's "technically feasible, not a new direction" framing.

## 7. Auto Augmentation Library Analysis

I couldn't identify a single named software package that unambiguously matches "Auto Augmentation Library" — it's possible Dr. Khan meant a specific codebase (worth a direct 30-second confirmation with him). But the corresponding *concept* in the literature is well-established and directly relevant: AutoAugment is one of the most popular automated data-augmentation approaches, formulating policy design as a discrete search problem over composed transformations, each with a probability and magnitude, with RandAugment and Fast AutoAugment as faster successors. Critically, a 2024 paper adapts this idea specifically to audio, proposing an Automated Audio Augmentation method that uses Bayesian optimization to search composed waveform- and spectrogram-level augmentation policies in a plug-and-play manner.

**Relevance to your project: high, and actionable now.** It slots directly into your existing, supervisor-mandated §8.5.4 augmentation ablation — instead of (or alongside) each member's hand-picked augmentation, run an automated policy search per member's model and report it as an additional row. No ICBHI paper has done this; every augmentation paper I found for this dataset (SpecAugment adaptations, VAE-based synthetic generation, audio-enhancement preprocessing) uses fixed, manually chosen techniques. This is a low-risk, high-credibility way to satisfy your supervisor's suggestion without inventing new methodology.

## 8. Recommended Final Novelty Combination

Layer three things on top of your existing core claim, in order of effort-to-payoff:
1. **Core (unchanged):** MTL cross-task disagreement + staged OWL + statistically-redesigned evaluation on respiratory audio.
2. **+ Conformal calibration** of the disagreement score, giving a distribution-free unknown-detection guarantee — this is your strongest genuinely-open gap.
3. **+ Automated augmentation policy search**, replacing the ad hoc per-member recipes in §8.5.4 — directly responds to your supervisor's stated interest and costs almost nothing given the ablation already exists.
4. **+ TTA on the OOD stress tests** as an optional fourth layer if time allows — nice-to-have, not load-bearing.

This keeps the paper's spine intact (nothing here contradicts §4's novelty statement) while adding two citable "we did something no one else did on this data" claims beyond the core mechanism.

## 9. Expected Research Contribution

With items 1–3 above: *"A multi-task open-world respiratory diagnosis framework that (a) detects unseen diseases via cross-task disagreement under a staged learning protocol, (b) gives a distribution-free statistical guarantee on that detection rather than a point accuracy, and (c) replaces hand-designed augmentation with an automatically searched policy — evaluated with statistically sound coarse/large-N design and a compressed, deployable variant."* That's a three-part contribution (detection mechanism + statistical guarantee + methodology-improving side contribution), which is a comfortably Q1-competitive shape for *Biomedical Signal Processing and Control*.

## 10. Risks of Overlap with Existing Work

- **Low risk:** the core MTL+OWL+respiratory combination — confirmed absent as of July 2026 across a fresh search.
- **Low-medium risk:** the disagreement-as-rejection-signal mechanism is general and well-known (confirmed further by the medical-imaging and self-supervised-learning precedents above) — your paper must keep framing this as "domain application of a known mechanism," never as "we invented disagreement-based rejection," exactly as your own §3.3 already warns.
- **Medium risk:** conformal open-set methods are a fast-moving area (multiple 2025–2026 papers appeared in my search) — re-verify closer to submission, same caution your v2 checklist already applies to the core claim.
- **Low risk:** automated augmentation for audio — the closest existing work is general audio classification (not ICBHI, not respiratory), so this is genuinely uncrowded.

## 11. Final Recommendations

1. Add the conformal-calibration extension and the automated-augmentation-search extension to §5 and §8.5 respectively — both are low-cost, high-credibility additions that close gaps your own checklist already half-flags.
2. Add the medical-imaging MTL-OOD precedent to §3.3 alongside Huo et al., to pre-empt a more likely reviewer objection.
3. Get a one-line confirmation from Dr. Khan on what he specifically meant by "Auto Augmentation Library" — if he has a specific tool in mind, use it; if not, the AutoAugment/RandAugment/Bayesian-audio-policy family above is a safe academic anchor.
4. Keep the existing statistical-redesign framing (§5.1, §7) — it's your best-supported and least risky differentiator and shouldn't be diluted by the new additions.
5. Re-run a narrow literature check on conformal open-set methods specifically in the 4–6 weeks before submission, given how active that sub-area is.Let me know if you'd like me to go deeper on any one section — particularly the conformal-calibration extension (§5, idea #1), since that's the strongest genuinely-open gap and would benefit from a closer look at implementation details before you bring it to Dr. Khan.