# OWMTL — Week-by-Week Build Sheet (label-free track)

**For:** Barshon Basak · NSU · Supervisor: Dr. Khan
**Prepared:** 2026-08-14
**Track:** Label-free (default). The **annotation lane** is marked ⟦ANNOT⟧ wherever it would slot in — switch it on only if you and Dr. Khan decide to (Gate G0).
**Reads against:** `OWMTL_Merged_Decision_Roadmap.md` (gates G0–G7) and `OWMTL_Novelty_Gap_Analysis.md` (ideas I1–I10).

> **How to read this.** Each week has: **Goal · Tasks · Reuses (existing model) · New code · Deliverable · Owner · Gate.** The one hard mid-project decision is **G3 in Week 5** — everything before it is shared engine that both Path A and Path B need, so you build without having to choose the headline yet.

---

## Standing rules (apply to *every* experiment, all weeks)

These fix the exact problems your own audit caught, and they are non-negotiable per the proposal's §8.5 protocol:

1. **Official split + official metric only.** Patient-independent ICBHI **60/40** split; ICBHI Score = (Se+Sp)/2 with the *pooled-abnormal* definition. No 70/30 numbers in any comparison.
2. **Commit the confusion matrix** (`confusion_matrix_raw`) for every model, every split. (This is what made M30 unverifiable — don't repeat it.)
3. **Every headline number gets a CI** (bootstrap B=1000) + a paired test (McNemar / Wilcoxon). You already have this code from M30.
4. **No synthetic data, ever.** Real ICBHI audio only (the M11/M13/M15 synthetic-data episode must not recur).
5. **One results schema** across all three of you, so the tables merge (the §8.5.5 Model Zoo appendix).

**Suggested owners** (from who built what): **Asif** = engine/backbone/stats (M2, M29, audit, statistics). **Barshon** = bottleneck/intervention/leakage/calibration (M13, M14, M20). **Sami** = OOD/shift/ensemble/foundation-model (M7). Adjust freely.

---

## Reuse map (what you already have → what it becomes)

| Existing asset | Role in the new plan |
|---|---|
| **M2** 5-block CNN (official 0.6138) | The frozen **encoder** under the concept layer |
| **M35** physics loss (PAPR + spectral flatness) | Reuse its **DSP code** as physics-concept generators |
| **M13** prototypical disease head | The **concept → disease** predictor (and few-shot for small classes) |
| **M29** post-hoc OOD suite (MSP/Energy/Mahalanobis) | Re-run **in concept space** vs embedding space (I4) |
| **M14 / M20** conformal + temperature scaling | The **honest operating point** section (I6) |
| **M7** deep ensemble (10 checkpoints) | One **scored detector / uncertainty** baseline |
| **device labels** (4 devices) + `check_device_structure.py` | Gate **G4** device-shift analysis |
| **Coswara / SPRSound** loaders | **External + pediatric** covariate shift (I5) |
| **M37** LoRA machinery | **Foundation-model probing** adapter (I3, Gate G5) |
| **McNemar / bootstrap** code | CIs everywhere (standing rule 3) |

**Genuinely new code to write (all CNN/DSP-scale, fits T4):** DSP concept extractors, the strict bottleneck head, the leaky-joint control, the info-theoretic leakage estimator, the intervention API, the concept-space OOD wrapper. That's the whole new surface — everything else is reuse.

---

## Week 0 — Pre-flight (2–3 days): Gates G0, G1 + hygiene

| | |
|---|---|
| **Goal** | Clear the two pre-code gates and lock the protocol before touching the engine. |
| **Tasks** | (a) **G0:** 15-min decision with Dr. Khan — annotate a subset or not? Default = **no** (label-free). (b) **G1:** re-run the kill-searches (`physics-derived acoustic concept bottleneck respiratory`, `faithfulness/leakage audit respiratory sound`, `foundation model concept probing respiratory`) — confirm still open. (c) Run `Asif's/audit/check_device_structure.py` now (cheap; result feeds G4). (d) Freeze the official-split + confusion-matrix + CI harness. |
| **Reuses** | audit scripts, statistics code |
| **New code** | none |
| **Deliverable** | a one-page "go memo": G0 answer, G1 search log with dates, device-structure result, protocol checklist |
| **Owner** | Asif (audit/searches), Barshon (protocol), all (G0 with Dr. Khan) |
| **Gate** | **G0** (annotate?), **G1** (still open?) — if G1 finds a twin, narrow per roadmap before proceeding |

⟦ANNOT⟧ *If G0 = yes:* start a labeling guide (fine/coarse crackle, inspiratory phase, wheeze pitch) this week; the labeling runs as a parallel lane through Week 4.

---

## Weeks 1–2 — Shared engine, part 1: DSP concept extractors → Gate G2

| | |
|---|---|
| **Goal** | Turn each respiratory cycle into a vector of **clinically-named** acoustic concepts computed from signal physics (no labels needed). |
| **Tasks** | Implement per-cycle extractors: **crackle** (transient/PAPR), **wheeze** (tonal + dominant-frequency band), **fine vs coarse crackle** (duration + center-frequency thresholds), **inspiratory/expiratory phase** (breathing-phase detector, ref arXiv:1903.10251), **rhonchi**, plus **spectral flatness + PAPR** (lift straight from M35). Then validate the extractors against ICBHI's coarse cycle labels (crackle/wheeze presence). |
| **Reuses** | **M35** DSP code (flatness/PAPR); M2 preprocessing pipeline |
| **New code** | the concept-extractor module + a validation script (extractor vs ICBHI label agreement) |
| **Deliverable** | per-cycle **concept vectors** for all 6,898 cycles + a validation table + visual QC on sample cycles |
| **Owner** | Asif (extractors), Barshon (validation/QC) |
| **Gate** | **G2** — do the extractors reproduce cycle labels at a sane rate? *Yes →* Week 3. *No →* shrink to the reliable subset (PAPR/flatness/wheeze-band) **or** pivot to the reliability-only paper (roadmap G2 fallback). |

⟦ANNOT⟧ *If labeling is on:* validate extractors against your **fine-grained** labels too (a much stronger check), and add the human labels as extra concepts.

---

## Week 3 — Shared engine, part 2: bottleneck + controls

| | |
|---|---|
| **Goal** | Build the strict bottleneck and the controls you'll compare it against. |
| **Tasks** | Freeze M2 encoder. Build: **(i) independent-CBM** (predict concepts, then disease from concepts only), **(ii) sequential-CBM**, **(iii) leaky-joint control** (disease sees encoder features too), **(iv) opaque baseline** (M2 → disease directly). Use M13's prototypical head as the concept→disease predictor. |
| **Reuses** | **M2** (frozen encoder), **M13** (concept→disease head) |
| **New code** | strict bottleneck head + leaky-joint control wiring |
| **Deliverable** | 4 trained models + the **accuracy–interpretability tradeoff** table, with **per-class disease F1** (watch for collapse toward COPD) |
| **Owner** | Barshon (bottleneck), Asif (baselines/training) |
| **Gate** | — (feeds G3) |

---

## Week 4 — Shared engine, part 3: leakage + intervention (the G3 inputs)

| | |
|---|---|
| **Goal** | Produce the two remaining numbers the decision hall needs. |
| **Tasks** | (a) Implement the **information-theoretic leakage** estimator (mutual information between residual encoder info and label given concepts, ref arXiv:2504.09459); compute per variant. (b) Build the **intervention API**: overwrite a concept value (simulate a clinician correcting a mis-heard crackle), re-predict, measure Δaccuracy per corrected concept. |
| **Reuses** | the Week-3 models |
| **New code** | leakage estimator + intervention API |
| **Deliverable** | **leakage table** (per variant) + **intervention Δaccuracy** table — together with Week 3's tradeoff, these are the three G3 inputs |
| **Owner** | Barshon (leakage + intervention) |
| **Gate** | — (feeds G3) |

---

## Week 5 — ★ Gate G3: THE DECISION HALL (headline A or B) ★

| | |
|---|---|
| **Goal** | Read the three numbers and let the **data** pick the headline. Do not pick by preference. |
| **Decide** | Review: tradeoff (did the bottleneck hold accuracy or collapse toward COPD?), leakage (low or heavy?), intervention (helps or weak?). |
| **If Path A** | bottleneck accurate + low leakage + intervention helps → **headline = "interpretable, clinician-correctable respiratory diagnosis."** Leakage/OOD/shift become supporting sections. |
| **If Path B** | collapse / heavy leakage / weak intervention → **headline = "physics-concept bottleneck shows models don't route through the clinical sounds — a faithfulness + robustness audit."** The weak intervention becomes evidence, not failure. |
| **If ambiguous** | default to **Path B** (lower risk — it doesn't need to win on accuracy). |
| **Reuses** | all Week 3–4 outputs |
| **Deliverable** | a **one-paragraph headline-decision memo**, signed off with Dr. Khan |
| **Owner** | all three + Dr. Khan |
| **Gate** | **G3** — the only A-vs-B fork. Downstream is identical either way. |

---

## Weeks 6–7 — Shared downstream evaluation: Gates G4, G5

| | |
|---|---|
| **Goal** | Run the evaluations that both headlines need, with CIs throughout. |
| **Tasks** | (a) **Concept-space OOD (I4):** run M29 detectors (MSP/Energy/Mahalanobis) **in concept space** vs embedding space; add M7 ensemble + M15 scorer as scored detectors (M15 is now just *one detector*, not the thesis). (b) **G4 device shift:** if `check_device_structure.py` passed → leave-one-device-out on *known* diseases (a good detector should NOT flag these); if not → drop device, use pediatric shift only. (c) **Pediatric-physics-fragility (I5):** test on SPRSound whether the physics concepts degrade on children (your Gap-7 finding, now a headline result). (d) **Honest operating point (I6):** M14 conformal + M20 calibration + cost-sensitive selective classification at a stated clinical target. (e) **G5 FM probing (I3):** LoRA-adapt an OPERA/M2D embedding (reuse M37), probe whether it encodes the concepts and stays faithful under shift. |
| **Reuses** | **M29, M7, M15, M14, M20, M37**, device labels, Coswara/SPRSound |
| **New code** | concept-space OOD wrapper; small per-device normalization; probing harness |
| **Deliverable** | the full results set: concept-space vs embedding OOD, device/pediatric shift, operating-point curves, FM-probing table — all with **bootstrap CIs + Wilcoxon** |
| **Owner** | Sami (OOD/shift/FM), Barshon (operating point), Asif (stats) |
| **Gate** | **G4** (device separable?), **G5** (FM worth it? — you have compute, so likely yes) |

⟦ANNOT⟧ *If labeling is on:* this is where the released annotations + the richer bottleneck lift the ceiling to Q1.

---

## Week 8 — Gate G6: sharpen + figures

| | |
|---|---|
| **Goal** | Make sure the findings cohere into **one** claim; build the figures. |
| **Tasks** | Consolidate everything into the Model-Zoo table (one schema). Test the contribution against the **Reviewer #2 attacks** (report §Step 10). If diffuse → cut to the single strongest finding (Path B: usually the physics-fragility or leakage result; Path A: the intervention demo). |
| **Reuses** | all results |
| **New code** | figure scripts (tradeoff curve, leakage, concept-space OOD, shift, operating point) |
| **Deliverable** | final figure set + a **one-sentence contribution statement** that survives Reviewer #2 |
| **Owner** | all three |
| **Gate** | **G6** — cohere into one claim? *No →* sharpen. *Yes →* write. |

---

## Week 9 — Write draft + Gate G7: venue

| | |
|---|---|
| **Goal** | A complete draft aimed at the honest venue. |
| **Tasks** | Draft Intro (cite the two "explainable multi-modal" papers as motivation — they *assert* biomarker attribution but never verify it), Methods, Experiments, Results around the chosen headline. Decide venue with Dr. Khan. |
| **Reuses** | everything |
| **Deliverable** | full first draft + venue decision |
| **Owner** | all three + Dr. Khan |
| **Gate** | **G7** — annotated + strong → Q1; sharp + label-free → strong domain venue / borderline Q1; modest → Q2 / domain conference |

---

## Timeline at a glance

| Week | Phase | Gate | Headline decided? |
|---|---|---|---|
| 0 | Pre-flight | G0, G1 | no |
| 1–2 | DSP concept extractors | **G2** | no |
| 3 | Bottleneck + controls | — | no |
| 4 | Leakage + intervention | — | no |
| **5** | **Decision hall** | **★ G3 ★** | **YES** |
| 6–7 | Downstream eval | G4, G5 | fixed |
| 8 | Sharpen + figures | G6 | fixed |
| 9 | Write + venue | G7 | fixed |

**Buffer:** you said 6–12 weeks; this is a 9-week core, leaving 1–3 weeks of slack for the two riskiest pieces (phase/crackle-type extraction in Weeks 1–2, and FM integration in Weeks 6–7) or for the annotation lane.

---

## The three things most likely to slow you down (plan for them now)

1. **Reliable phase / crackle-type extraction (Weeks 1–2)** — the one genuinely new engineering risk. Mitigation: start from the published breathing-phase detector (arXiv:1903.10251); if a concept is unreliable, drop it rather than shipping a noisy bottleneck (that's the G2 fallback).
2. **The G3 outcome is not yours to choose** — if the bottleneck collapses, that's Path B, not a failure. Don't spend Week 5 trying to "rescue" accuracy; that was the M15 mistake.
3. **FM integration (Weeks 6–7)** — different mel config + licensing. If it's eating time, G5 says drop it and cite OPERA as a baseline; the paper still stands.

---

*Built against the cross-validated roadmap (2026-08-14). Re-run the G1 kill-search once more right before submission.*
