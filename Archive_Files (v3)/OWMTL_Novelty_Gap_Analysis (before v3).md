# OWMTL — High-Impact Novelty & Research-Gap Analysis

**Prepared for:** Barshon Basak (CSE465 Capstone, NSU, Group 5) · Supervisor: Dr. Khan
**Prepared:** 2026-08-14
**Method:** Reviewer-grade gap discovery + a fresh 2026 literature cross-check (searches run 2026-08). Every candidate is labeled **[Verified]** (multiple recent papers confirm), **[Plausible]** (no counter-paper found — absence of evidence ≠ evidence of absence), **[Uncertain]** (depends on an experiment not yet run), or **[Occupied]** (a recent paper already does it).
**Constraints locked with you:** best *honest* venue (not a fixed Q1 label); foundation-model compute is available; ~6–12 weeks to a draft; annotation is possible but undecided.

> **One-line answer up front.** Your accuracy story is capped — the corrected numbers (~0.60–0.65 official ICBHI) sit *at* the published level, so no accuracy claim will read as novel. The one genuinely open, defensible, feasible gap that reuses almost everything you built is: **a physics-grounded, label-free *acoustic concept* layer used not to win accuracy but to answer a question nobody has answered for lung sound — *do respiratory models (including foundation models) actually listen to the clinically-defined sounds, or do they cheat?* — and whether reasoning in that clinical concept space is more honest, better-calibrated, and more robust under device and pediatric shift than the opaque embedding.** This is an interpretability/rigor contribution, not an accuracy race, which is exactly the axis your own three validation passes kept pointing at. Details, ranking, and the Reviewer #2 attack in §§5–11.

---

## Step 1 — The project, in my own words (no novelty yet)

You set out to build an **open-world multi-task model** for respiratory auscultation on ICBHI 2017: one shared audio encoder feeding a **sound-event head** (normal / crackle / wheeze / both, at the breathing-cycle level) and a **disease head** (COPD / Healthy / URTI, at the patient level). The headline idea was that **disagreement between the two heads** would flag a patient whose disease was never seen in training (Bronchiectasis, Pneumonia, Bronchiolitis — the 19 held-out patients), giving a clinically safer "I don't know" instead of a confident wrong diagnosis. Around this you planned a staged open-world-learning protocol, conformal thresholding, cross-dataset OOD stress tests (Coswara, SPRSound), and an edge-compression story.

What actually happened, read honestly from your own progress report and audits:

- **The core mechanism failed twice, for a principled reason.** M15 cross-task disagreement reaches AUROC **0.5747**, *below* a two-line Energy baseline (**0.6466**) and barely above chance. You diagnosed *why*: joint MTL forces the two heads to agree on unknowns (AUROC collapses to 0.41), and sequential MTL makes the disease head a linear echo of the sound head — neither produces the *divergent* representations the mechanism needs. And at n=19 the 95% CI on AUROC is ≈ ±0.12, so the comparison is statistically void anyway.
- **Your strong numbers are all in the wrong (saturated) place.** M35 physics-loss (0.6864), M37 LoRA (0.6753), M30 fusion (withdrawn — no committed confusion matrix) are **closed-set sound-event** results on an easier 70/30 split. ICBHI closed-set classification has 135+ papers and sits at 0.60–0.86; none of these is open-world, and publishing them silently abandons your thesis.
- **A metric error inflated everything by ~0.11**, and two models (M33, M36) that looked "mediocre" were actually **collapsed** (Sp=0.00 and Se=0.09). Corrected, the project is honest and at-the-literature-level, not above it.
- **Three separate novelty passes each died on the same rock:** ICBHI is small (19 unknown patients), coarsely labeled (4 sound classes, no fine/coarse crackle, no timing/phase/pitch), COPD-dominated, and has only **4 devices** (the "7" that Direction 1 was built on is chest *locations*, not stethoscopes — a factual error that invalidated that whole direction).

So the real situation: the engineering is solid, the mechanism and the dataset are the ceiling, and the contribution has to come from **rigor, mechanism, or data — not accuracy.**

---

## Step 2 — Literature positioning

**What type of paper is this, honestly?** It is **not** an algorithmic-novelty paper for NeurIPS/ICLR/CVPR (no new learning algorithm that beats SOTA). It is best positioned as an **evaluation-first / interpretability / trustworthy-ML paper in a clinical-signal venue** — the class where a rigorous, clinically-grounded *question* answered carefully beats a gadget. That maps to BSPC, CMPB, *Diagnostics*, Frontiers, and to Interspeech/EMBC/CBMS on the conference side. This is also the axis ("depth, not breadth") your own `Novelty Search.md` §4.0 and `Novelty_Reassessment.md` independently identified as your comparative advantage.

**Closest research area:** interpretable & trustworthy deep learning for respiratory sound / auscultation (concept-based interpretability, calibration, OOD/open-set, faithfulness).

**Neighboring areas:** general OOD/open-set recognition (Yang et al. IJCV 2024 taxonomy; full-spectrum OOD); concept-bottleneck models (Koh 2020; label-free CBM Oikarinen 2023; medical-imaging & voice CBMs 2024–2026); domain generalization / device-shift robustness for auscultation; audio foundation models (OPERA, HeAR, M2D).

**Interdisciplinary donors worth mining:** clinical acoustics / respiratory physiology (the physics that *defines* crackles vs wheezes vs rhonchi); DSP (spectral flatness, PAPR, cepstral timing, phase detection); information theory (concept leakage as mutual information); conformal / distribution-free inference; causal inference (shortcut/confound reasoning). These are the fields your *concepts* and *rigor* can borrow from without acquiring new modalities.

---

## Step 3 — Reverse-engineering the field (what "everyone" does)

From the 2025 *Electronics* ICBHI review (135+ papers), the curated 15-paper list, and this pass, the field's default recipe is remarkably uniform:

- **Preprocessing:** resample to 4–16 kHz → log-mel spectrogram (64–128 mels) → fixed-length cycle clips → SpecAugment. (You do exactly this.)
- **Architecture:** a CNN (ResNet/MobileNet) or an AST/transformer backbone; recently, fine-tuning an audio foundation model (OPERA/HeAR/M2D). Fusion/attention variants for a few points. (You have M2/M3/M4/M30/M37.)
- **Compared against:** prior ICBHI-score numbers on the 4-class sound task and/or the coarse disease task; the "official 60/40" or an ad-hoc split.
- **Losses:** cross-entropy, occasionally class-balanced/focal; a few 2025 papers add spectral/physics-ish priors or contrastive terms. (Your M35 is in this family.)
- **Evaluation:** ICBHI Score = (Se+Sp)/2, accuracy, macro-F1; rarely calibration; rarely CIs; almost never operating-point sensitivity or device-stratified analysis.
- **Interpretability, when present:** post-hoc Grad-CAM / attention / SHAP / prototypes — *never* a concept bottleneck a clinician can intervene on.

**What your project currently repeats:** the entire preprocessing + backbone + closed-set-classification + ICBHI-score stack (M1–M4, M22, M30, M35, M37). Your *intended* differentiators — cross-task disagreement, staged OWL, conformal — are each individually a known mechanism ported to respiratory (Zamir 2020 Consistency Energy; open-world continual learning; split-conformal), which is why reviewers read them as incremental.

**What the field does NOT do (the openings):** (1) route the diagnosis *through* clinically-named acoustic concepts as a true bottleneck; (2) measure whether the model actually *uses* the sounds (leakage/faithfulness); (3) report reliability at a clinical operating point with CIs and device stratification; (4) separate "new disease" from "new microphone/child" when claiming novelty detection. These four are where your assets can land something real.

---

## Step 4 — Research-gap discovery (by type)

### Methodological gap
The whole pipeline is a *parallel* MTL predictor. A **bottleneck redesign** — disease predicted *only* from an interpretable acoustic-concept vector — is methodologically different (it enables intervention and makes leakage measurable) and is unclaimed for lung sound. **[Plausible]** The redesign that survives is *not* "CBM for lungs" (that reads as a modality port, and voice/imaging CBMs already exist) but **concepts computed from signal physics/DSP rather than from human labels or CLIP/LLMs.** No respiratory concept-bottleneck paper found in the 2026 pass. **[Plausible]**

### Algorithmic gap
An entirely different *learning strategy* is available but most are decoration here: the honest one is **treating concepts as the learning target and diagnosis as a thin function of them**, plus **information-theoretic leakage regularization** (mutual information between residual encoder info and label given concepts, à la arXiv:2504.09459). Leakage measurement for respiratory audio is unstudied. **[Plausible]**

### Representation gap
Better representations *for interpretability*, not accuracy: a **physics-derived concept representation** — fine vs coarse crackle (duration + center frequency), inspiratory vs expiratory timing/phase (breathing-phase detection is a solved sub-model, arXiv:1903.10251), wheeze dominant-frequency band, rhonchi, spectral flatness + PAPR (your M35). These are *measurable without new labels* and *clinically auditable*. Whether foundation-model embeddings even encode these concepts is an open probing question. **[Plausible]**

### Optimization gap
Loss functions / curriculum / dynamic weighting / meta-learning are mostly **decoration** for you — you already tried GradNorm (M31, below baseline), curriculum (M34, below baseline), physics loss (M35, helps but on the easy split). The *one* optimization idea that earns a place is a **leakage-penalized bottleneck loss** (novel here, and it directly serves the thesis). Everything else in this bucket is a technique-list trap your own §4.0 warns against. **[Mostly Occupied / decoration]**

### Knowledge gap (import from another field)
The highest-value imports: **clinical respiratory acoustics** (defines the concept vocabulary and predicts *which* physics fails under pediatric shift), **information theory** (leakage), **conformal/distribution-free inference** (honest operating point), **causal/shortcut reasoning** (device confound — but see below, largely occupied). RL / control theory / network science / neuro-symbolic are stretches that would read as bolt-ons. **[Selective — acoustics + info-theory are genuine, [Plausible]]**

### Evaluation gap *(your strongest axis)*
This is where the real openings are, all **[Verified]** as under-reported in respiratory audio:
- **Faithfulness / leakage:** does the model listen to the sounds or cheat? Never measured for lung sound.
- **Operating-point reliability:** the field admits FMs "report AU-ROC without sensitivity or calibration, concealing failure at the operating point" (BCoughBench 2026). You have M14/M20 to do this properly.
- **Calibration & conformal under shift** (your M14 gives 95% coverage but 0% detection — a *finding*, not a failure, if framed as "conformal-wrapping a near-random score is clinically useless").
- **Device-stratified + pediatric-shift stress** — your Gap-7 result that M35 physics priors *hurt* pediatric generalization is a genuinely publishable insight (see Step 5).
- **Statistically honest small-N protocol** — CIs, Wilcoxon, never a bare point AUROC at n=19.

### Deployment gap
Edge/compression/distillation/quantization is **[Occupied / crowded]** (HeAR→PulmoVec, generic edge-KD, your own M16/M18). Keep it as a one-figure footnote, not a contribution. Real-time/streaming inference on cycles is mildly interesting but not a headline.

---

## Step 5 — Cross-domain innovation (the honest version)

You asked me to mine techniques from other domains. Here is the honest scorecard — **most cross-domain ports would be rejected as decoration on this dataset**, and I'll say so plainly rather than manufacture novelty.

| Technique (donor domain) | Could it integrate? | Impl. difficulty | Research impact here | Publication novelty | Honest verdict |
|---|---|---|---|---|---|
| **Concept Bottleneck + physics-derived concepts** (interpretable ML) | Yes — your sound head *is* a concept predictor; add DSP concepts | Medium | **High** — enables intervention + leakage | **[Plausible] high** | **Lead candidate** |
| **Information-theoretic leakage** (info theory) | Yes — measures if model uses the sounds | Low–Med | High (a real question) | [Plausible] | **Core sub-contribution** |
| **Concept-space OOD / novelty** (OOD literature) | Yes — run M29 suite in concept space | Low | Medium–High | [Plausible] for clinical audio | **Fold in** |
| **Test-Time Adaptation** (domain shift) | Yes but | Medium | Low delta | **[Occupied]** TRIAGE 2026, DHAuDS | Drop |
| **Foundation-model probing/faithfulness** (repr. learning) | Yes — you have FM compute | Medium | **High** — do FMs listen? | **[Plausible]** unclaimed for respiratory | **Strong add-on** |
| **State-Space Models / Mamba backbone** | Yes | Medium | ~0 (a backbone swap) | Low — reads as "another backbone" | Decoration — skip |
| **Diffusion / EBM / Neural ODE / Hypernetworks** | Forced | High | Low, high risk | Low (port for its own sake) | Skip |
| **GNN / Knowledge Graph over concepts** (co-occurrence of findings) | Maybe — model concept→disease as a clinical graph | Med–High | Medium (nice but speculative) | [Uncertain] | Optional stretch |
| **Retrieval-Augmented / Memory** (prototypes at scale) | Weakly | High | Low on n≈900 | Low | Skip |
| **Neuro-symbolic** (encode clinical rules crackle→disease) | Yes — a symbolic concept→disease layer *is* mildly neuro-symbolic | Med | Medium | [Plausible] as framing | Optional framing |
| **Causal / shortcut de-confounding** (device) | Yes | Med–High | Medium | **[Occupied]** stethoscope-shortcut papers 2024–2026 | Cite, don't claim |
| **Contrastive / masked SSL pretrain** | Yes | Med | Low delta vs FM | Crowded | Skip (use FM instead) |
| **Curriculum / meta-learning / MoE / NAS / AutoML** | Yes | Varies | You tried several → below baseline | Low | Decoration — skip |

**The one genuinely fresh cross-domain synthesis** (not in your docs): **treat the pediatric-physics-failure as physics, not noise.** Your Gap-7 result — M35's adult-tuned acoustic priors (spectral flatness for wheeze, PAPR for crackle) *reduce* generalization to SPRSound because a child's smaller airways resonate at higher frequencies — is a specific, mechanistic, testable claim borrowed from **respiratory physiology / acoustic scaling**. Almost nobody frames physics-informed audio priors as *body-size-dependent and therefore covariate-fragile*. That is a real scientific finding you already have data for. **[Plausible novelty, and you have the result in hand.]**

---

## Step 6 — Novelty matrix

Originality and publication scored 1–10 (10 = top-venue-defining). "Pipeline change" = does it alter the training/inference pipeline vs. being post-hoc.

| # | Title | Description | Why novel (honest) | Effort | Orig. | Pub. | Risk | Perf. gain | New data? | Pipeline change? |
|---|---|---|---|---|---|---|---|---|---|---|
| I1 | **Physics-Grounded Acoustic Concept Bottleneck (PG-ACB)** | Diagnosis routed *only* through DSP-computed clinical concepts (fine/coarse crackle, phase, wheeze band, rhonchi, flatness, PAPR) | Concepts from **signal physics**, not labels/CLIP/LLM; no respiratory CBM exists | Medium | 7 | 7 | Med (accuracy may drop — but that tradeoff *is* the result) | Not the point | No (DSP only) | Yes |
| I2 | **Concept-leakage / faithfulness audit** | Info-theoretic measure of whether the model uses the sounds or bypasses them | Leakage unstudied for respiratory audio; "does it listen?" is a real question | Low–Med | 7 | 7 | Low | N/A | No | No (analysis) |
| I3 | **Do foundation models listen? FM concept-probing** | Probe OPERA/HeAR/M2D embeddings for clinical concepts; compare faithfulness vs your CNN | No faithfulness probing of respiratory FMs found; answers your deferred "Attack 7" | Medium | 8 | 8 | Med (FM integration) | N/A | No | No (probing) |
| I4 | **Concept-space novelty detection** | Run MSP/Energy/Mahalanobis in the low-dim clinical concept space vs embedding space | Concept-space OOD is vision/VLM only; unclaimed for clinical acoustic concepts | Low | 6 | 6 | Med (may not beat embedding OOD — negative is still a finding) | N/A | No | No |
| I5 | **Physics-priors-are-body-size-fragile** | Formalize + test that adult acoustic priors fail on pediatric shift | Specific mechanistic covariate-shift insight; you already have the SPRSound result | Low | 7 | 7 | Low | N/A | No | No |
| I6 | **Honest operating-point + conformal-validity study** | Cost-sensitive selective classification; characterize when conformal is valid for respiratory OOD | Field admits AUROC hides operating-point failure; you have M14/M20 | Low–Med | 6 | 6 | Low | N/A | No | No |
| I7 | **Fine-grained ICBHI concept annotations (dataset)** | Team labels fine/coarse crackle + phase on an ICBHI subset; release + CBM built on it | "First fine-grained acoustic-concept annotation for ICBHI" — dataset+method | High (annotation) | 9 | 9 | Med (effort) | Enables real CBM | **Yes** | Yes |
| I8 | **Clinician-intervention on concepts** | Overwrite a concept value, re-predict, measure diagnosis correction (human-AI complementarity) | Interveneability is the CBM selling point; unshown for lung sound | Medium | 6 | 6 | Med (needs a working bottleneck) | N/A | No | Partial |
| I9 | **Neuro-symbolic concept→disease layer** | Encode clinical crackle/wheeze→disease rules as a transparent reasoning layer | Adds auditable clinical logic; mild novelty | Med–High | 5 | 5 | Med | Low | No | Yes |
| I10 | **GNN over concept co-occurrence** | Model findings→disease as a clinical graph | Speculative; nice story, thin evidence at n≈900 | High | 4 | 4 | High | Low | No | Yes |

---

## Step 7 — Ranking by novelty tier

**Minor novelty (incremental — do NOT anchor a paper on these):** I9 neuro-symbolic layer (as *framing* only), I10 GNN, any backbone swap (Mamba/AST), TTA, edge-compression, curriculum/meta/GradNorm bolt-ons. Your M31/M34 already show several of these lose to baseline.

**Moderate novelty (solid components, not a standalone paper):** I4 concept-space novelty, I6 operating-point/conformal-validity, I8 intervention. Each strengthens a paper; none carries one alone.

**Strong novelty (paper-anchoring, feasible):** **I1** physics-grounded concept bottleneck, **I2** leakage/faithfulness, **I5** physics-priors-are-body-size-fragile. These are unclaimed for respiratory audio and reuse your assets.

**Conference/Q1-level novelty (highest ceiling):** **I3** "do foundation models actually listen?" (a genuinely current, unclaimed question — needs your FM compute) and **I7** the fine-grained annotation dataset+method (the single most Q1-defensible move, but costs annotation effort). I3 + I1 + I2 combined is a coherent top-tier story.

**High-risk / high-reward:** I7 (annotation cost, but converts your biggest weakness into a released asset + citations) and I3 (if a foundation model simply wins everywhere, the paper must be framed as the *faithfulness/robustness* study, not "FMs are good").

---

## Step 8 — Combinations (why the whole beats the parts)

1. **I1 + I2** — Physics-concept bottleneck **+** leakage measurement. The bottleneck gives you a place to measure leakage; leakage tells you whether the bottleneck is real or cosmetic. Neither is convincing alone; together they answer "is this interpretability faithful?"
2. **I1 + I2 + I3** *(the recommended spine)* — add foundation-model probing. Now the question scales: "do *state-of-the-art* respiratory models listen to clinical sounds, and does forcing them through a physics-concept bottleneck cost accuracy?" This absorbs your deferred Attack 7 and puts you on the 2026 frontier.
3. **I1 + I5** — physics-concept bottleneck **+** pediatric-physics-fragility. The concepts are physics; the finding is *when physics breaks*. A single narrative: physics-grounded concepts are auditable **and** they expose exactly where the physics assumption fails under body-size shift.
4. **I1 + I4 + I6** — bottleneck **+** concept-space novelty **+** honest operating point. Converts your dead M15/M14 into: "flag unknown disease in the clinical concept space, with a real operating point instead of a 0%-detection conformal paradox."
5. **I2 + I3** — leakage **+** FM probing = "faithfulness audit of respiratory foundation models." Standalone-viable even without the bottleneck.
6. **I5 + I6 + device stratification** — a pure **reliability-under-shift** paper (device + pediatric), CIs throughout. The evaluation-first fallback if the bottleneck accuracy collapses.
7. **I1 + I8** — bottleneck **+** clinician intervention = the "correctable diagnosis" demo clinicians care about.
8. **I7 + I1 + I8** — annotation **+** bottleneck **+** intervention = the **dataset+method Q1 package** (highest ceiling, highest cost).
9. **I3 + I6** — FM probing **+** operating-point reliability = directly the BCoughBench-shaped gap ("FMs hide operating-point failure"), now for auscultation specifically.
10. **I1 + I2 + I5 + I6** — the **complete "faithful, physics-grounded, honestly-evaluated respiratory diagnosis" paper**: concepts from physics, leakage measured, physics-fragility characterized, operating point honest. This is the strongest *feasible-without-annotation* single paper, and it reuses M35, M29, M14, M20, device labels, Coswara/SPRSound, and your stats code.

**Why combinations matter here specifically:** none of your individual mechanisms beats a trivial baseline (M15 < Energy; M14 = 0% detection). But *a coherent question* answered with several honest measurements does not need to beat a baseline to be publishable — it needs to be rigorous and true. Combination #10 (or #2 if you use FM compute) is a thesis; the parts are footnotes.

---

## Step 9 — Future research opportunities

- **Master's thesis:** the full **PG-ACB + faithfulness + FM-probing** program (#2/#10), with the intervention study and a small annotated subset.
- **PhD:** faithfulness and covariate-robustness of *medical audio foundation models* as a general agenda — physics-grounded probes across auscultation, cough, voice; formal links between concept leakage and shift-robustness.
- **Journal extension:** add the released **fine-grained ICBHI annotations** (I7) + a multi-dataset (ICBHI + SPRSound + Coswara) concept benchmark.
- **Open-source project:** a **DSP concept-extractor library** for respiratory audio (fine/coarse crackle, phase, wheeze band, rhonchi, flatness, PAPR) + a concept-space OOD toolkit — reusable, citable, low-glamour-high-utility.
- **Industry/deployment:** a **correctable, auditable** screening tool where a clinician overrides a mis-heard concept and sees the diagnosis update — the intervention story (I8) is the deployment pitch.

---

## Step 10 — Reviewer #2 (the adversarial pass)

**Weak contributions Reviewer #2 will name:**
- *"Concept bottlenecks are established (Koh 2020); voice and imaging medical CBMs already exist. This is a modality port."* → **Fix:** center the claim on **physics/DSP-derived concepts** (not labels/CLIP) + the **leakage/faithfulness question** + the **FM-probing** — none of which the voice/imaging CBMs address. State up front that CBM-the-framework is not your novelty.
- *"Your bottleneck is only 4 coarse concepts — clinically too weak to separate COPD from pneumonia."* → **Fix:** this is the real ICBHI hole. Either (a) expand the concept set with DSP-derived fine/coarse crackle + phase + rhonchi + pitch (label-free), or (b) annotate a subset (I7). Report the accuracy–interpretability tradeoff *as the finding*, don't hide it.
- *"Cross-task disagreement / M15 is your headline and it loses to Energy."* → **Fix:** demote M15 to one *scored* detector among many; the paper is no longer "our detector wins," it's "we measure faithfulness and reliability." A negative detector result inside a faithfulness paper is fine.
- *"n=19 unknowns — nothing is significant."* → **Fix:** never report a bare AUROC; CIs + Wilcoxon throughout (you have the code); use Coswara/SPRSound for scale and label the covariate confound honestly.
- *"You ignored foundation models (OPERA/HeAR/M2D)."* → **Fix:** I3 makes FMs the *object of study*, not an ignored baseline. This is the single most important reviewer-defense given a 2026 submission.
- *"Split/metric hygiene."* → **Fix:** everything on the official 60/40 split, official (Se+Sp)/2 metric, committed confusion matrices. Your audit already exposed this; make the corrected protocol a *virtue* you state explicitly.

**Missing experiments they'll demand:** ODIN/VIM OOD baselines; a leakage ablation (with/without bottleneck regularization); device-stratified evaluation (check `check_device_structure.py` first — with only 4 devices and possible device–diagnosis confounding, leave-one-device-out may be infeasible; if so, say so and lean on pediatric shift instead); intervention Δaccuracy per corrected concept; concept-space vs embedding-space OOD head-to-head.

**Likely reasons for rejection & the pre-emption:** (1) "incremental CBM port" → physics-concepts + faithfulness question; (2) "no significance at n=19" → CIs + external scale; (3) "accuracy not competitive" → reframe as interpretability/rigor, not accuracy; (4) "FMs ignored" → I3. If all four are pre-empted in the intro, this clears a clinical-signal venue.

---

## Step 11 — Final recommendation (best 5 directions, ranked)

I'm recommending **five**, ordered by what I'd actually do given your constraints (best honest venue, FM compute available, 6–12 weeks, annotation optional). The top pick is a *program* built from the strongest survivors; the rest are the fallbacks and add-ons in priority order.

### ① PRIMARY — "Does it listen?" A physics-grounded acoustic concept bottleneck + faithfulness audit (I1 + I2 + I5 + I6, optionally + I3)
- **Why it's best:** it is the only direction that is simultaneously (a) genuinely unclaimed for respiratory audio in the 2026 pass **[Plausible]**, (b) reuses M35 (physics), M29 (OOD suite, now run in concept space), M14/M20 (operating point), device labels, Coswara/SPRSound, and your stats code, (c) is a *mechanism/rigor* contribution that does **not** need to beat SOTA accuracy, and (d) turns your two biggest liabilities — the failed detector and the confounded OOD sets — into the object of study. It's the honest maximum of your existing work.
- **Implementation difficulty:** Medium. New pieces are bounded: DSP concept extractors (crackle type via duration+center-freq thresholds; breathing-phase detector — a known reproducible sub-model; wheeze band; rhonchi; you already compute flatness+PAPR), a strict bottleneck head, an info-theoretic leakage estimator, and re-running M29 in concept space.
- **Timeline:** ~6–9 weeks. Weeks 1–2 DSP concept extractors + validate against ICBHI cycle labels; 3–4 train independent/sequential/leaky-control bottlenecks + tradeoff curve; 5 leakage; 6 concept-space vs embedding OOD (device + pediatric), CIs; 7–9 write.
- **Publication potential:** **High** for a clinical-signal venue (BSPC/CMPB/Diagnostics/Interspeech/EMBC). Not an ICLR/NeurIPS algorithmic paper — and you shouldn't pretend it is.
- **Undergrad-feasible?** Yes — all CNN/DSP-scale, fits T4; the riskiest piece (reliable phase/crackle-type extraction) has published references.
- **Retrain from scratch?** No — reuses M2/M35 backbone; concept head + analyses are lightweight.
- **Scientific-contribution increase:** Large — moves you from "another ICBHI classifier at the literature level" to "the first faithfulness/physics-concept study of respiratory diagnosis."
- **Add I3 (FM probing) if** you commit the foundation-model compute: it raises the ceiling toward Q1 and closes the Attack-7 hole, at +2–3 weeks of integration risk.

### ② HIGHEST-CEILING (if you'll annotate) — Fine-grained acoustic-concept annotations for ICBHI + the bottleneck built on them (I7 + I1 + I8)
- **Why:** your own ACBD §7 and I n-dependently reach the same conclusion — **the dataset is the ceiling, so change the dataset.** "First fine-grained (fine/coarse crackle, inspiratory phase, wheeze pitch) acoustic-concept annotations for ICBHI, plus an interpretable CBM and clinician-intervention built on them" is a **dataset + method** contribution that is Q1-defensible in a way no method-only paper on ICBHI can be.
- **Difficulty / timeline:** Medium-High; the bottleneck is *annotation effort*, not code. Feasible in 6–12 weeks if the 3 of you annotate a bounded subset (even 150–300 cycles with a clinical reference) and use DSP pre-labeling to cut the load.
- **Pub potential:** **Highest** of all options. **Undergrad-feasible?** Yes, if scoped to a subset. **Retrain from scratch?** No.
- **Do this instead of ① if** the team can commit annotation time; **fold the subset into ①** if you can only annotate a little.

### ③ STANDALONE FALLBACK — Do respiratory foundation models actually listen? (I3 + I2 + I6)
- **Why:** even without the bottleneck, "a faithfulness + operating-point reliability audit of OPERA/HeAR/M2D for auscultation" is unclaimed **[Plausible]** and rides a gap the field admits (BCoughBench: "FMs report AUROC without sensitivity/calibration"). Needs your FM compute; strong, current, and defensible.
- **Difficulty:** Medium (FM integration). **Timeline:** 6–8 weeks. **Pub potential:** High. **Undergrad-feasible?** Yes with FM compute. **Retrain?** No (frozen FM + probes/LoRA).

### ④ PURE-RIGOR FALLBACK — Reliability under device & pediatric shift (I5 + I6 + device stratification)
- **Why:** the safest paper if bottleneck accuracy collapses. Headline = your real finding that **adult-tuned physics priors are body-size-fragile** (M35 hurts SPRSound), plus honest operating-point + conformal-validity analysis with CIs. Pure evaluation contribution, no new mechanism to fail.
- **Difficulty:** Low–Medium. **Timeline:** 4–6 weeks. **Pub potential:** Medium–High (domain venue). **Undergrad-feasible?** Very. **Retrain?** No. **Caveat:** run `check_device_structure.py` first — if patients don't span devices, drop the device axis and lean on pediatric shift.

### ⑤ COMPONENT / DEMO — Clinician concept-intervention (I8), folded into ① or ②
- **Why:** the "correctable diagnosis" demo is what clinicians and a BSPC audience find compelling, and it's the clearest human-AI-complementarity result. **Not a standalone paper** — it's the figure that sells ① or ②.
- **Difficulty:** Medium (needs a working bottleneck first). **Timeline:** +1–2 weeks on top of ①. **Pub value:** high as a component, low alone.

---

### What is NOT genuinely novel (so you don't waste weeks on it)
Stated plainly, per your instruction: **backbone swaps** (Mamba/AST/fusion), **TTA** (TRIAGE/DHAuDS 2026 own it), **edge KD/quantization/pruning** (crowded; your M16/M18 are fine as a footnote), **curriculum / GradNorm / meta-learning bolt-ons** (you already saw M31/M34 lose to baseline), **generic post-hoc XAI** (Grad-CAM/SHAP is done to death on ICBHI), the **device/disease disentanglement Direction 1** (stethoscope-shortcut papers 2024–2026 occupy it, and you only have 4 devices), and **cross-task disagreement as the headline** (Zamir 2020 + your own M15 failure). Each of these is either occupied or empirically dead for you. Reusing them as *baselines or footnotes* is fine; anchoring novelty on them is not.

---

## Step 12 — Dr. Khan's generalized novelty list: evaluation, additions & revalidation

You asked me to run his 16-item list through the project and add only what's *genuinely new*. Your own `Novelty Search.md` §3.2 already scored this list once; crucially, several items you **already built, and they lost** — so the strongest evidence here is empirical, not speculative. The table below is the honest verdict; then I add the handful of items that survive as real additions, each revalidated against a fresh 2026 search.

### 12.1 The 16 items, scored against *this* project

| # | Faculty item | Status in your project | Verdict for the reframed (faithfulness/concept) paper |
|---|---|---|---|
| 1 | Foundation-model adaptation (LoRA/QLoRA/Prompt/Adapter) | M37 LoRA done; = my **I3** | **Keep** — as FM-probing (I3). QLoRA/Adapter are cheap variants, not new novelty. Note: VLM framing is imaging — use *audio* FMs (OPERA/HeAR/M2D), not BiomedCLIP. |
| 2 | Uncertainty estimation (MC-Dropout / Deep Ensemble / Evidential) | M7 ensemble, M14 conformal, M20 temp-scaling done; **Evidential NOT built** | **Partial add** — evidential *per-concept* uncertainty is a genuine new component (§12.2a). |
| 3 | Curriculum learning | **M34 done — below baseline (0.5754 < 0.6138)** | **Drop** — you already tried it and it lost. Decoration. |
| 4 | Dynamic token pruning (ViT) | Only touches M4 AST, which lost the backbone race | **Drop** — efficiency footnote at best. |
| 5 | Vision State-Space Models (VMamba/MedMamba) | Not built | **Drop** — a backbone swap; reads as "another encoder," no faithfulness value. |
| 6 | Physics-informed **& topological** constraints | Physics = M35 (your core asset); **topological NOT built** | **Physics = the spine of the Primary direction.** Topological = genuine but high-risk add (§12.2d). |
| 7 | RL for HP / augmentation policy | Not built | **Drop** — AutoML flavor, compute-heavy, orthogonal to an interpretability thesis. |
| 8 | Multistage knowledge distillation | **M36 done — collapsed (Se=0.093)** | **Drop as novelty** — keep only as an edge footnote. |
| 9 | Ensemble (cross-attn / fusion / gating) | M30 (withdrawn), M7 done | **Fold in** — ensemble *in concept space* is a minor uncertainty/OOD arm, not a headline. |
| 10 | Cross-modal knowledge distillation | No 2nd modality | **Repurpose** — CLAP/LLM-derived concepts as a **contrast baseline** to physics concepts (§12.2b). |
| 11 | Custom loss + RL-based loss weighting | M35 custom loss done; GradNorm = **M31, below baseline** | **Keep** the custom-loss idea *only* as the **leakage-penalized bottleneck loss (I2)**; drop RL weighting. |
| 12 | Few-shot learning | M13 prototypical head done | **Keep** — already core; the bottleneck handles few-shot disease naturally. |
| 13 | Cross-modal consistency learning | No 2nd modality; audio-audio analog = your dead M15 | **Drop** — not applicable. |
| 14 | Meta-learning for low-data | M13 is prototypical (meta-adjacent) | **Covered** — no separate addition needed. |
| 15 | Contrastive learning (SimCLR/DINO) | Not built; **SSL contrastive for lung sound is occupied** (see §12.3) | **Optional/low-priority** — only *concept-supervised* contrastive is mildly new (§12.2c). |

**Headline conclusion:** the list does **not** change the Primary recommendation. Most items are either already-built-and-lost (curriculum, GradNorm, multistage KD), occupied (SSL contrastive, ALM), or decoration on this dataset (Mamba, token pruning, RL). Physics-informed constraints and FM adaptation are already the backbone of my Primary direction. What the list *does* usefully contribute is **three small ablation arms + one high-risk stretch** that strengthen the faithfulness paper — below.

### 12.2 What genuinely survives as an addition

**(a) Evidential / uncertainty-aware *concept* heads** *(from item 2)* — **[Plausible]**, revalidated. Instead of only calibrating post-hoc (M14/M20), predict **per-concept uncertainty** (evidential or MC-dropout) and route it into the honest operating point (I6) and abstention. Novel angle: uncertainty is reported *per clinical concept* ("the model is unsure whether this is a fine crackle"), which is far more auditable than a single scalar confidence. Evidential DL for respiratory sound returned no direct hit → open. Modest novelty; strong *fit* with the thesis. **Fold into I6.**

**(b) Physics-grounded vs. language-grounded concepts — a contrast arm** *(repurposing item 10)* — **[Verified as a sharpening, not a headline]**. Build a second concept vocabulary from **CLAP / audio-language supervision** (LLM-named concepts) and compare it head-to-head with your **physics/DSP-derived** concepts on faithfulness, leakage, and device/pediatric robustness. This directly operationalizes your central claim ("physics concepts are more clinically grounded and more shift-robust than learned/language concepts"). *Caveat from revalidation:* an audio-language model for cardiopulmonary sound already exists (**StethoLM**, arXiv:2603.00355, 2026), so do **not** propose building an ALM — use CLAP-style concepts only as the *baseline arm you beat or characterize*. **Add as an ablation axis to I1.**

**(c) Concept-supervised contrastive representation** *(from item 15)* — **[Occupied base, marginal twist]**. Plain SSL contrastive for lung sound is already published (§12.3), so this is *not* novel on its own. The only mildly-new framing is a **supervised-contrastive loss over concept labels** to tighten the concept space and improve concept-space OOD (I4). Low priority; include only if I4 needs a cleaner space.

**(d) Topological features of respiratory cycles** *(the "topological persistent homology" half of item 6)* — **[Plausible, genuinely unclaimed, HIGH RISK]**. Revalidation found TDA/persistent-homology applied to vowels, music, and audio fingerprinting, but **no application to respiratory/lung-sound classification**. So persistent-homology descriptors of a cycle's time-delay embedding (capturing the *topology* of crackle transient bursts vs. sustained wheeze tori) are an unclaimed micro-contribution. Honest risk: TDA on audio is finicky, effort is high, and the payoff is uncertain — this is a *high-risk/high-reward* optional feature, best as one extra "physics-adjacent concept generator," **not** the paper's spine. Your own `Novelty Search.md` §4.10 rightly called this "a much bigger stretch for audio than images" — I agree, but the search confirms it's at least *open*.

### 12.3 Revalidation test (fresh 2026 searches on the new candidates only)

| New candidate | Search result | Label |
|---|---|---|
| CLAP/LLM-derived concepts for lung sound | CLAP (Elizalde 2023) is general; **StethoLM (arXiv:2603.00355, 2026)** already builds an audio-language model for cardiopulmonary sound → the ALM lane is **occupied** | **[Occupied]** as a headline; **[OK]** only as a contrast baseline |
| Evidential DL for respiratory sound | No direct respiratory hit | **[Plausible]** open (as a component) |
| SSL / contrastive for lung sound | *Leveraging unlabeled data for lung sound classification through self-supervised contrastive learning* (ScienceDirect S1746809425009887) → **occupied** | **[Occupied]** — only concept-supervised twist is marginal |
| Persistent homology / TDA for lung sound | TDA found for vowels, music, audio ID — **none for respiratory** | **[Plausible]** genuinely unclaimed, but high-risk |
| Physics-informed concept bottleneck (re-confirm) | Still no respiratory concept-bottleneck paper | **[Plausible]** — Primary direction holds |

**Net effect on the report:** the Primary recommendation (§Step 11 ①) is **unchanged and reinforced**. Dr. Khan's list adds, at most: an **evidential per-concept uncertainty** arm (into I6), a **physics-vs-language concept contrast** arm (into I1), and an **optional high-risk topological concept generator**. Nothing on the list rises to a standalone contribution, and — importantly — three of his items are things you already ran and that already lost, which is itself the cleanest possible answer to "why not just add more techniques?": *you did, and the data ceiling ate them.* Use that sentence with him.

> **One line for Dr. Khan:** *"We mapped all 16 techniques onto the project. Six we already built (three of them came in below our own baseline — curriculum, GradNorm, multistage distillation), several are occupied by 2026 papers, and the rest are backbone swaps that wouldn't survive an ablation on one small dataset. The two that genuinely help — physics-informed constraints and foundation-model adaptation — are already the backbone of the direction we're proposing, which reframes the project around whether the model actually listens to the clinical sounds rather than chasing an accuracy number we can't move."*

---

## Step 13 — ACBD-modified vs. Primary ①: critical head-to-head (revalidated 2026-08-14)

**Provenance note first, for honesty:** the `ACBD_Revalidation v3` document was one of *your* uploaded project files (prepared 2026-08-11), not something generated in this analysis. That matters because it means two *independent* passes — yours and mine — converged on the **same physics-grounded, label-free acoustic concept bottleneck**. Convergence is mild evidence the idea is real. But it also means these are **not two independent paths**: they share ~80% of their machinery (physics/DSP concept extractors, the bottleneck head, leakage measurement, concept-space OOD, and reuse of M35/M29/M14/M20). Treating them as rival options is slightly artificial. The honest framing is: **one shared engine, two different places to point the headline.** Below I separate them precisely and revalidate each.

### 13.1 What each path actually is

- **Path A — ACBD-modified (the survivor in your v3 doc):** a *constructive/interpretability* thesis. Headline = *"we build an interpretable-by-design respiratory diagnosis routed only through clinically-named physics concepts, which a clinician can correct (intervention)."* Success = the bottleneck **diagnoses acceptably** and intervention **measurably helps**. T4-only; no foundation model required (your v3 §5 says so explicitly).
- **Path B — Primary ① (this report):** an *evaluative/critical* thesis using the **same bottleneck as an instrument**. Headline = *"do respiratory models — including foundation models — actually reason from the clinical sounds, and are physics-grounded concepts more faithful / less leaky / more shift-robust than learned or opaque ones?"* Success = the **measurements are rigorous and reveal something** (a faithfulness gap, leakage, or physics-fragility under pediatric/device shift). Uses the FM compute you now have (FM probing = I3).

The pivotal difference is **what has to be true for the paper to succeed**, and it's not cosmetic:

- Path A **needs the bottleneck to work**. Its central risk is exactly the §2.5 validity hole your own doc flags: force 3–6 diseases through a handful of COPD-correlated concepts and the disease head may collapse to "predict COPD," accuracy drops, and intervention effects go weak — in which case the *positive* thesis has failed and you're left with a null constructive result.
- Path B **does not need the bottleneck to win**. If forcing the bottleneck costs accuracy, or the model leaks, or physics concepts break on children — *those are the findings.* The validity hole that threatens Path A **becomes the result** in Path B. That is the single most important asymmetry between them.

### 13.2 Head-to-head

| Axis | Path A — ACBD-modified (constructive) | Path B — Primary ① (faithfulness audit) |
|---|---|---|
| **Core machinery** | Physics concept bottleneck + intervention + leakage + concept-space OOD | *Same*, plus FM probing + pediatric-physics-fragility + honest operating point |
| **Headline claim** | "An interpretable, correctable respiratory diagnosis" | "Do models listen? Are physics concepts more faithful/robust?" |
| **Must be true to succeed** | Bottleneck diagnoses well; intervention helps | Measurements are rigorous and reveal *something* |
| **Main failure mode** | Validity hole → accuracy collapse → null constructive result | "So what?" → findings come out boring/faithful → no punchline |
| **Novelty defense vs. "incremental CBM port"** | **Weaker** — reviewer can still say "CBM applied to lungs" | **Stronger** — it's an evaluation/analysis question, not a new SOTA method; FM-faithfulness for respiratory is unclaimed + timely |
| **Does the §2.5 validity hole hurt it?** | **Yes — it's the central threat** | **No — it's converted into a finding** |
| **Foundation models** | Not used (T4-only) | Used (rides the 2026 frontier; closes your deferred Attack 7) |
| **Compute / engineering risk** | Low | Medium (FM integration) |
| **Scope-creep risk** | Lower (tighter, one artifact) | **Higher** — an "audit" can sprawl into a bag of measurements without a sharp claim |
| **Venue ceiling (honest)** | Domain-conf / Q2 unless you annotate (your v3 §7) | Higher — evaluation-first + FM audit fits a strong venue better, *if* the result is sharp |
| **Best feasible with your assets** | Yes (cheapest) | Yes (you have FM compute now) |

### 13.3 Revalidation result (fresh 2026-08-14 searches)

- **Shared core still open:** "physics-derived acoustic concept bottleneck respiratory/lung sound" → no direct hit (only general acoustic-DL reviews). **[Plausible]** holds for *both* paths.
- **Path A specific risk unchanged:** the medical-CBM bandwagon (chest-X-ray causal CBM, voice CBM, imaging CBM) is still active, so the "incremental port" reviewer kill your v3 §2.2 identified is **still live** for the constructive framing. **[Verified — still the decisive risk for Path A.]**
- **Path B specific check:** concept-leakage / faithfulness for *respiratory auscultation* still unclaimed, **but** the general "do audio models leak / take shortcuts" genre is now visibly active in 2026 — *HearSay: Do Audio LLMs Leak What They Hear?* (arXiv:2601.03783), *Auditing Protocol-Level Shortcuts in Audio LLM Judges* (arXiv:2607.13477), plus the voice CBM (2607.16967). None is respiratory auscultation, so Path B is **[Plausible] open** — but the timeliness cuts both ways: the frame is fashionable (good for acceptance) yet slightly more crowded than six months ago (so it must be sharply respiratory-specific, not a generic "we audited a model"). **[Plausible, with a timeliness caveat.]**
- **Neither path escapes the §7 rock.** ICBHI's structural ceiling (n=19 unknown, 4 coarse labels, COPD-dominant, 4 devices) still binds *both*. Path B *softens* it (an audit needn't beat SOTA) but does not remove it; only annotation (Option 2) or a realistic-venue decision (Option 1) removes it — and that is true for A and B alike.

### 13.4 Advantages / disadvantages, stated plainly

**Path A — advantages:** cheapest (T4-only, no FM); tightest scope; always yields *something* shippable (an interpretable model + an intervention demo) even if modest; the intervention story is the most clinician-legible result you can produce. **Disadvantages:** weakest novelty defense (the "port" kill is live); its success *depends* on the bottleneck actually diagnosing well, which the validity hole directly threatens; ceiling is domain-conf/Q2 without annotation, by your own doc's honest verdict.

**Path B — advantages:** turns your biggest liability (the validity hole, the failed M15, the confounded OOD sets) into the *content*; sidesteps both reviewer kills that cap Path A (the "port" kill and the "accuracy isn't competitive" kill); uses the FM compute you now have and closes Attack 7; higher venue ceiling; rides a current 2026 theme. **Disadvantages:** higher variance — if the faithfulness findings are null/boring there's no headline; more engineering (FM integration); genuinely higher scope-creep risk (your own `Novelty Search.md` §4.0 warns precisely against sprawling measurement lists); the audit genre is fashionable enough that framing must be sharply respiratory-specific to stand out.

### 13.5 Likely outcomes (best / expected / worst)

- **Path A:** *Best* — with a small annotation effort, a genuinely Q1-defensible "first fine-grained concept CBM for ICBHI + intervention." *Expected* — a solid, honest domain-conference / Q2 interpretability paper. *Worst* — the bottleneck collapses toward COPD, accuracy drops, intervention is weak → a thin "we tried an interpretable model, it cost accuracy" note.
- **Path B:** *Best* — a sharp, current faithfulness/robustness audit ("respiratory models don't listen; physics concepts break on children; here's the honest operating point") at a strong venue. *Expected* — a rigorous evaluation paper with 2–3 real findings, defensible at BSPC-class or a good conference. *Worst* — a diffuse audit whose measurements don't cohere into one claim; reviewers say "interesting numbers, unclear contribution."

### 13.6 The honest recommendation on *which*

They are not either/or at the code level — **build the shared engine first** (physics concept extractors → bottleneck → leakage → concept-space OOD; ~weeks 1–4), because both paths need it. Then let the *data pick the headline*: if the bottleneck diagnoses acceptably and intervention helps, ship the **constructive Path A** story; if instead you find leakage / faithfulness gaps / physics-fragility, ship the **critical Path B** story. A single paper can even carry both — "a physics-grounded concept bottleneck *and* what auditing it (and the foundation models) reveals" — provided you keep **one** headline and demote the other to support, rather than presenting a measurement buffet.

**My weighted call, given your constraints (best honest venue, FM compute available, 6–12 weeks):** point the headline at **Path B**, keep Path A's intervention demo as the clinician-facing figure inside it, and treat annotation (Option 2) as the upgrade that lifts *either* framing to Q1. Path B is the better bet specifically because it is the only one of the two whose success does **not** depend on beating an accuracy ceiling you've already shown you can't move — and that, not novelty wording, is the real lesson of the §7 rock.

---

## Appendix — Evidence map (2026 cross-check)

**Confirms the openings (no counter-paper found → [Plausible]):**
- No respiratory-auscultation **concept-bottleneck diagnosis** paper (searches on CBM + lung sound/auscultation 2026).
- No **physics/DSP-derived (label-free) clinical concept** bottleneck for respiratory — "label-free CBM" exists only as CLIP/LLM-derived, vision (Oikarinen et al., *Label-Free Concept Bottleneck Models*, ICLR 2023, arXiv:2304.06129).
- No **concept-leakage / faithfulness** study for respiratory audio (leakage methods are general: arXiv:2504.09459).
- No **faithfulness/concept-probing audit of respiratory foundation models**.

**Occupies / kills candidates (→ do not anchor on these):**
- Test-time adaptation for respiratory audio: TRIAGE (arXiv:2604.12647, 2026), DHAuDS (arXiv:2511.18421).
- Stethoscope-shortcut / device confound for *classifiers*: *Mitigating Stethoscope-Induced Shortcuts … Causality-Inspired Interventions* (arXiv:2605.29862, 2026); *Stethoscope-guided Supervised Contrastive Learning* (ICASSP 2024, arXiv:2312.09603).
- Semantic-vs-covariate OOD is formalized in vision/medical-imaging: Yang et al. *Generalized OOD Detection* (IJCV 2024); *Delving into OOD Detection with Medical VLMs* (MICCAI 2025).
- Edge KD/compression: HeAR / PulmoVec (arXiv:2603.15688), generic edge-KD.
- Audio-language model for cardiopulmonary sound: **StethoLM** (arXiv:2603.00355, 2026) — occupies the ALM/LLM-for-lung-sound lane.
- Self-supervised contrastive for lung sound: *Leveraging unlabeled data for lung sound classification through self-supervised contrastive learning* (Biomed. Signal Process. Control, S1746809425009887).

**Genuinely unclaimed on fresh 2026 revalidation (→ [Plausible]):**
- Evidential / per-concept uncertainty for respiratory sound (no direct hit).
- Persistent-homology / topological data analysis for respiratory sound (TDA found only for vowels/music/audio-ID: arXiv:2310.06508).

**Context / baselines (cite, build on):**
- OPERA (*Towards Open Respiratory Acoustic Foundation Models*, NeurIPS 2024, arXiv:2406.16148); M2D+Resp (Interspeech 2025); HeAR (Google 2024) — the foundation models for I3.
- Voice CBM (arXiv:2607.16967, 2026); medical-imaging CBM (arXiv:2410.15446, 2024) — the *adjacent* CBMs to differentiate from.
- Prototype interpretability (arXiv:2110.03536) — post-hoc, to contrast with interpretable-by-design.
- Physics-informed respiratory *inversion* (Sci. Reports s41598-026-40470-1, 2026) — different task, supports the physics framing.
- BCoughBench (arXiv:2606.25116, 2026) — the field's own admission that FMs hide operating-point/calibration failure (motivates I3/I6).

*Cross-check pass 2026-08. Treat [Plausible]/[Uncertain] as unproven; re-run the concept-bottleneck + FM-faithfulness kill-searches immediately before you commit code and again near submission — this subfield produces directly-relevant preprints monthly.*
