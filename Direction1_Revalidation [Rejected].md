# Direction 1 — Independent Revalidation

**Prepared:** 2026-08-11
**Question asked:** Is Direction 1 ("disentangle semantic disease-novelty from covariate device/population shift for open-world respiratory detection; build a device-controlled benchmark + a covariate-invariant novelty detector") genuinely worth pursuing as a Q1 contribution — or was it recommended too quickly?
**Method:** Fresh, adversarial 2024–2026 literature search whose explicit goal was to *kill* the idea, not support it. No assumption that the prior recommendation was right.

---

## VERDICT (up front)

**C — Reject Direction 1 as the primary Q1 novelty.** (With a narrow, clearly sub‑Q1 fallback described in §6, offered only for honesty, not recommended.)

Direction 1 does **not** survive the disproof pass. Between mid‑2025 and mid‑2026 — largely *after* the window the first analysis drew on — the literature closed essentially every building block it depends on. What remains unclaimed is a modality port ("run an existing analysis on respiratory audio instead of ultrasound, using existing device‑invariance tools") which is exactly the kind of incremental, combination‑of‑existing‑techniques contribution you asked me to screen out. A skeptical Q1 reviewer has at least four recent papers to cite as prior art. I recommend not re‑tracking the project onto it.

The rest of this document is the evidence for that verdict, done the way you asked — skeptically and with multiple papers per claim, not assumptions.

---

## 1. What Direction 1 actually claims (re‑derived precisely)

**Origin.** The original OWMTL novelty (cross‑task disagreement as an unseen‑disease detector) failed empirically (M15 AUROC 0.5747 < Energy 0.6466) and on prior art (a published method). The first reassessment noticed that your Coswara/SPRSound "OOD" evaluation *conflates* two shifts and proposed converting that flaw into the contribution.

**Exact problem Direction 1 solves.** In open‑world respiratory screening, a model should abstain ("unknown") when it meets a **new disease** (semantic novelty) but *not* when it merely meets a **new stethoscope or a new population** (covariate shift). Direction 1 claims current respiratory detectors confuse the two.

**Gap it claims.** "No respiratory‑audio work separates semantic from covariate shift for the *detection* task; ICBHI's 7 devices give a controlled covariate axis nobody has exploited for detection."

**Assumptions that must all hold for it to be Q1‑novel:**
1. The *concept* (semantic‑vs‑covariate disentanglement for OOD **detection**) is not already established in a medical modality.
2. A *device‑invariant respiratory representation/detector* does not already exist.
3. The *leave‑one‑device‑out ICBHI testbed* has not already been built.
4. The clinical framing ("abstain on disease, not on device") is not already published for respiratory audio.

**Every one of these assumptions is now false or nearly so.** Details below.

---

## 2. Disproof — what the 2024–2026 literature already covers

### 2.1 Assumption 1 fails: the concept is published in a sibling medical modality
- **Influence of Classification Task and Distribution Shift Type on OOD Detection in Fetal Ultrasound** — arXiv:2509.18326, Springer/MICCAI‑track 2025. This paper *is* Direction 1's thesis: it studies how OOD‑detection performance depends on whether the OOD is an **image‑characteristic (covariate) shift** vs an **anatomical (semantic) shift**, across 8 uncertainty methods and 4 tasks, and concludes "superior OOD detection does not guarantee optimal abstained prediction" and that task/shift‑type must be aligned. That is precisely the insight Direction 1 offers — already demonstrated, in a medical modality, at a strong venue.
- **Full‑Spectrum OOD Detection** — Yang et al., arXiv:2204.05306; formalized in **OpenOOD v1.5** (semantic‑shift detection while staying robust to covariate shift; near‑/far‑OOD splits). The conceptual scaffolding Direction 1 would import is 3–4 years old and standardized.
- **Exploring Covariate and Concept Shift for Detection and Calibration of OOD** (KL‑based covariate vs concept scores) — same idea, general domain.

→ Direction 1's conceptual contribution reduces to **"port a known analysis from imaging to respiratory audio."** That is a modality transfer, not a new insight.

### 2.2 Assumption 2 fails: device‑invariant respiratory representations already exist (multiple, 2026)
- **A device‑invariant multi‑modal learning framework for respiratory disease classification** — *npj Digital Medicine* 2026 (s41746‑026‑02445‑4). Embeds an **adversarial branch to enforce device‑invariant features** plus an **IRM‑augmented loss** for robustness to non‑structural shifts. This is the exact mechanism Direction 1 proposed as its "invariant detector," in a top‑tier journal, months ago.
- **BTS‑CARD: Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial Debiasing** — arXiv:2510.22263, **ICASSP 2026**. Counterfactual + **adversarial debiasing to remove spurious correlations from acquisition device** (and age/sex), yielding metadata/device‑invariant representations robust across clinical sites.

→ The methodological half of Direction 1 (adversarial/gradient‑reversal device invariance on respiratory audio) is **occupied twice over**.

### 2.3 Assumption 3 fails: the leave‑one‑device‑out ICBHI testbed is built
- **Mitigating Stethoscope‑Induced Shortcuts in Respiratory Sound Classification under Federated Domain Generalization with Causality‑Inspired Interventions** — arXiv:2605.29862 (2026). Uses **leave‑one‑device‑out on ICBHI and SPRSound**, treats devices as the covariate axis, and builds device‑invariant representations via gradient alignment + counterfactual augmentation. This is the controlled testbed Direction 1 said "nobody has exploited."
- **Stethoscope‑guided Supervised Contrastive Learning** — ICASSP 2024, arXiv:2312.09603 — earlier cross‑device‑adaptation precedent.

→ The "novel controlled device axis" is already the standard evaluation for this sub‑problem.

### 2.4 Assumption 4 fails: open‑set respiratory + the clinical framing already exist
- **Enhancing Respiratory Sound Classification Based on Open‑Set Semi‑Supervised Learning** — Cho & Lee 2025 (CMC / ScienceDirect). Open‑set respiratory recognition where unlabeled data contains unknown classes; explicitly motivated by "new respiratory sound types not in training" and OOD degrading closed‑set models.
- **Open‑set lung sound recognition via conditional Gaussian capsule network + variational time–frequency reconstruction** — *Biomedical Signal Processing and Control* (S1746809423009035) — an *additional* open‑set respiratory method, in your exact target journal.
- Recording‑level respiratory work **under internal and domain‑shift evaluation** (Life 2025, 10.3390/life16071108) already frames "strong internal performance doesn't transfer under domain shift."

→ Open‑world/open‑set respiratory recognition, and the "robust under domain shift" framing, are established — including inside BSPC itself.

### 2.5 The adjacent‑audio precedents also stand (from the first pass, still valid)
- **Disentangling Hierarchical Features for Anomalous Sound Detection Under Domain Shift** (arXiv:2501.01604) and **Subject Information Extraction for Novelty Detection with Domain Shifts** (arXiv:2504.21247, 2025) — semantic‑vs‑nuisance novelty detection under domain shift, in machine/general audio. A reviewer can argue Direction 1 is "DCASE‑style domain‑robust novelty detection, applied to lungs."

---

## 3. What survives (and why it is not enough)

**The single unclaimed intersection:** a study that *specifically* runs semantic‑vs‑covariate **OOD‑detection** disentanglement (not classification robustness) on **respiratory audio**, using ICBHI device labels, and demonstrates that popular detectors respond to device rather than disease.

Why that sliver does not clear a Q1 bar:
1. **It is a port.** The analysis exists (fetal ultrasound, 2509.18326); the tools exist (npj 2026, ICASSP 2026, FedDG 2026); the datasets/splits exist (LODO ICBHI). Combining published analysis + published tools + existing splits on a new modality is the textbook definition of incremental.
2. **The "fix" is taken.** Any invariant‑detector you build will be compared to — and likely dominated by — the adversarial/IRM device‑invariance already published for respiratory audio. You'd be reproducing their method to score OOD.
3. **The finding may not even be surprising anymore.** "Detectors flag the device" is already the stated motivation of the 2026 device‑invariance papers; you'd be empirically confirming a premise the field already accepts, not overturning a belief.
4. **Statistical fragility remains.** ICBHI's semantic‑novelty side is still 19 patients; LODO further shrinks per‑cell N. The disentanglement table will have wide CIs on the semantic axis — the same n‑problem that sank M15.

Net: the survivor is a solid **workshop / domain‑conference** contribution (e.g., an ICASSP/EMBC/Interspeech short paper, or a DCASE‑style benchmark note), **not** a Q1 journal novelty. Presenting it to BSPC as a headline contribution would very likely draw an "incremental; see [npj 2026], [ICASSP 2026], [MICCAI 2025 ultrasound]" rejection.

---

## 4. What a skeptical Q1 reviewer writes (simulated)

> "The paper transfers the semantic‑vs‑covariate OOD‑detection analysis of Ref [fetal ultrasound, 2025] to respiratory audio. The proposed device‑invariant detector reuses adversarial/IRM invariance already established for this modality by [npj Dig Med 2026] and [ICASSP 2026], and the leave‑one‑device‑out ICBHI protocol by [arXiv:2605.29862]. Open‑set respiratory recognition is covered by [Cho & Lee 2025] and [capsule‑net, BSPC]. The semantic‑novelty evaluation rests on 19 held‑out patients, so the central comparison lacks statistical power. I do not see a contribution beyond a modality‑specific reproduction. Reject."

I cannot construct a convincing rebuttal to this review with the current assets. That is the decisive test, and Direction 1 fails it.

---

## 5. Feasibility note (secondary — feasibility was never the problem)

For completeness: Direction 1 *is* feasible with your assets (M2/M30, M29 baselines, M7, M13/M14, Coswara/SPRSound, ICBHI device labels, your McNemar/bootstrap code; new work ≈ LODO splitter + gradient‑reversal head, tens of lines). **But you explicitly said not to green‑light on feasibility.** It is buildable and not novel enough — the worst quadrant to spend the remaining term in.

---

## 6. Narrow fallback (offered for transparency, not recommended as the Q1 headline)

If — and only if — you decide a **domain‑conference** paper is an acceptable outcome, the survivor in §3 can be sharpened into something defensible at that lower bar by making the *diagnostic* claim the product rather than the fix:

- **Framing:** "A device‑controlled audit of open‑world respiratory detectors." Not "we built a better detector," but "we provide the first respiratory‑audio protocol that *separates* the two shifts and quantify how much of every popular detector's apparent unknown‑disease performance is actually device detection."
- **Delta vs the fetal‑ultrasound paper:** different modality *plus* a purpose‑built, reusable ICBHI device‑stratified OOD benchmark + a decision‑theoretic abstention rule (abstain‑on‑semantic, pass‑on‑covariate) with honest CIs — packaged as a benchmark/resource contribution (which is citeable even when incremental).
- **Still must:** cite [2509.18326], [npj 2026], [2510.22263], [2605.29862], [Cho & Lee 2025] up front and position as complementary, not competing.

This is a real option, but it is **B‑at‑a‑lower‑venue**, and it does not meet your stated bar of "realistic Q1 publication potential + genuine novelty." I am flagging it, not endorsing it.

---

## 7. Where the genuinely open white space now looks like it is (hypotheses, NOT validated)

You asked me not to hand you another "trust me, it's novel" idea. So these are **leads that must go through the same disproof pass before any commitment** — I am explicitly *not* certifying them:

1. **Foundation‑model‑native open‑world respiratory (the OPERA/M2D angle).** The 2026 device‑invariance papers are mostly built on task‑specific encoders; whether a *respiratory foundation embedding* (OPERA, NeurIPS 2024; M2D+Resp, Interspeech 2025) already dissolves the device/disease confound "for free," or still needs help, appears genuinely unstudied and is squarely on the current frontier. This also finally answers your long‑deferred Attack 7. Higher integration cost; needs its own kill‑search.
2. **The measurement problem itself.** Every open‑set respiratory paper reports single‑point AUROC on tiny disease sets with no CIs. A rigorous *statistical evaluation protocol* for small‑N respiratory open‑set detection (power analysis, when AUROC differences are meaningless, pre‑registered splits) is unglamorous but genuinely missing — and pairs with any of the above. Risk: reads as "methods note," not a full paper, on its own.
3. **A clinical axis no acoustic paper touches** (e.g., severity‑graded or longitudinal within‑patient novelty, if any accessible dataset supports it). Unknown feasibility with your current data — would need dataset scouting first.

None of these is blessed. Each needs a dedicated adversarial search exactly like this one before you re‑track the project. Do that *before* implementation, not after.

---

## 8. Bottom line

- **Direction 1 as framed:** reject (verdict **C**). Its concept, its method, its testbed, and its clinical framing were each published elsewhere in 2025–2026; what's left is an incremental modality port with a known statistical‑power weakness.
- **Do not** spend implementation time "fixing" it into Q1 shape — the ceiling is a domain‑conference benchmark note (§6), not a Q1 journal paper.
- **Next step:** pick one lead from §7, run the *same* disproof protocol on it (kill‑search first, build second), and only re‑track the project once a direction survives that test. That discipline — disprove before committing — is the actual lesson from both the original M15 failure and Direction 1.

---

## Appendix — Prior art that closes Direction 1 (2024–2026)

| # | Component of Direction 1 | Prior art that occupies it | Venue / year |
|---|---|---|---|
| 1 | Semantic‑vs‑covariate shift changes OOD **detection** (medical modality) | *Influence of Classification Task and Distribution Shift Type on OOD Detection in Fetal Ultrasound*, arXiv:2509.18326 | Springer/MICCAI‑track 2025 |
| 2 | Full‑spectrum OOD (semantic detection under covariate robustness) | Yang et al., *Full‑Spectrum OOD Detection*, arXiv:2204.05306; OpenOOD v1.5 | 2022 / 2023 |
| 3 | Device‑invariant respiratory representation (adversarial + IRM) | *A device‑invariant multi‑modal learning framework for respiratory disease classification*, npj Digital Medicine, s41746‑026‑02445‑4 | 2026 |
| 4 | Device/metadata‑invariant respiratory audio via adversarial+counterfactual debiasing | BTS‑CARD, arXiv:2510.22263 | ICASSP 2026 |
| 5 | Leave‑one‑device‑out ICBHI/SPRSound device‑shift testbed | *Mitigating Stethoscope‑Induced Shortcuts … Causality‑Inspired Interventions*, arXiv:2605.29862 | 2026 |
| 6 | Cross‑device adaptation for respiratory sound | *Stethoscope‑guided Supervised Contrastive Learning*, arXiv:2312.09603 | ICASSP 2024 |
| 7 | Open‑set respiratory unknown‑class recognition | Cho & Lee, *Open‑Set Semi‑Supervised Learning*, CMC 2025; *Conditional Gaussian Capsule Network*, BSPC (S1746809423009035) | 2025 / 2023 |
| 8 | Novelty detection under domain/style shift (adjacent audio) | *Disentangling Hierarchical Features for ASD under Domain Shift*, arXiv:2501.01604; *Subject Information Extraction for Novelty Detection with Domain Shifts*, arXiv:2504.21247 | 2025 |
| 9 | Cross‑task disagreement as OOD (original OWMTL core) | Zamir et al., *Cross‑Task Consistency* (Consistency Energy), arXiv:2006.04096 | CVPR 2020 |

*Search pass: 2026‑08. Verify exact author lists/venues before citing; re‑run the check near submission — this area is moving unusually fast (several key threats are dated 2026).*
