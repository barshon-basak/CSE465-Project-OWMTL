# Paper outline — the reframed contribution

**Drafted:** 2026-08-16, immediately after the G2 pivot (`DECISION_2026-08-16_PIVOT.md`).
**Rule followed throughout:** every claim below is traceable to a result that exists *today*, or is
explicitly marked ⏳ pending. Nothing is written that we hope to be able to say.

---

## Working title

> **What can a concept detector actually be validated against? Corrected evaluation baselines and a
> reliability ceiling for adventitious-sound detection on ICBHI**

Alternative, if the clinician data lands strongly:
*"Label reliability bounds concept-level respiratory sound analysis: a corrected protocol, a
negative result, and a clinician-referenced ceiling"*

## The one-sentence claim

Concept-level validation on ICBHI is bounded by the reliability of its cycle labels; we quantify
that bound, show a clinically-grounded DSP concept suite fails against it under a pre-registered
gate, and supply the corrected evaluation protocol and tooling the field needs to measure either.

## Target venue

**Biomedical Signal Processing and Control** (or CBMS / CMPB). These reward clinically-grounded
evaluation rigor over algorithmic novelty, which is what this is. A pure negative result is a hard
sell; a **negative result + corrected baselines + released tooling** is a reproducibility
contribution, which is a recognised category.

---

## Structure

### 1. Introduction
- ICBHI 2017 is the field's default benchmark: 135+ papers on one corpus (Electronics 2025 review).
- Interpretable/concept-based methods are arriving in medical audio, and they require validating a
  detector against a labelled reference.
- **Gap:** nobody has asked what that reference standard can actually support. Papers report
  detector performance against ICBHI labels without characterising the labels' own reliability.
- **Contributions** (four, all evidenced):
  1. Three evaluation errors, found in our own pipeline, with corrected baselines.
  2. A pre-registered concept-validity gate, run twice, failing both times.
  3. A reliability ceiling from published human benchmarks + our own clinician study.
  4. Released tooling: canonical score format, paired DeLong/McNemar/bootstrap, corrected split
     loader, audit tool.

### 2. Related work
- ICBHI classification: saturated, ~0.60–0.65 official-metric SOTA.
- Concept bottlenecks: Koh et al. 2020; Label-free CBM (Oikarinen, ICLR) — **LLM/CLIP-derived, not
  physics-derived**; ARDS CBM (MLHC 2025) — EHR + notes, no audio. *Name our concepts
  "physics-derived", never "label-free", to avoid collision.*
- Interpretable respiratory audio: HISET is ensemble feature engineering — no bottleneck, no
  leakage measure, no intervention. Closest work, and it does none of the three.
- Human benchmarks: Tzeng et al. *JMIR AI* 2025; Aviles-Solis et al. 2016.
- **Position honestly:** we are not inventing concept bottlenecks or disentangled evaluation. The
  contribution is the respiratory-audio measurement, not the method category.

### 3. Three evaluation errors and their corrections ← **the section reviewers will cite**

| Error | What we found | Effect |
|---|---|---|
| **Non-standard metric** | reported "ICBHI score" was `(recall_macro + specificity_macro)/2` | **+0.11 mean inflation** across 14 models, worst **+0.22** |
| **Mislabelled split** | "official 60/40" was a `pid ≤ 111` fallback | **11 test patients, 7.1% of cycles** — not 40% |
| **Split not patient-independent** | official ICBHI file assigns *recordings* | patients **156, 218** appear on both sides |

Each gets: how it arises, how to detect it, corrected numbers, and the audit check that catches it.

The metric error also **masks pathologies**: one model reported 0.5137 while detecting 9% of
abnormal events; another reported 0.5506 with **specificity 0.0000**. Both look mediocre-but-working
under the macro metric.

> **Framing discipline:** present these as *errors we made and caught*, not as accusations about
> other groups. That is both honest and far more persuasive. The implicit argument — if a team
> auditing itself this hard made all three, the 135-paper literature likely contains them — lands
> harder unstated.

### 4. Concept extractors and the pre-registered gate
- Nine physics-derived concepts from CORSA-grounded DSP; two are validatable against ICBHI
  (crackle/wheeze presence), five are proxies, two descriptive. **The 2/5/2 split is itself a
  finding** about what this benchmark can validate.
- Gate: AUROC ≥ 0.65 **and** 95% DeLong CI excluding chance, on the test split.
- **Why the threshold matters:** an earlier CI-only rule passed at AUROC 0.55, because at n≈2,600
  even a trivial effect is significant. Report this — a gate that cannot fail is not a gate.
- Synthetic-signal validation: the extractors provably respond to constructed crackles/wheezes
  (43/43), which is what makes the real-audio failure interpretable rather than a code bug.

### 5. Results

| Concept | Run 1 | Run 2 | AUPRC | prevalence | Gate |
|---|---:|---:|---:|---:|---|
| `crackle_score` | 0.5506 | 0.5580 | 0.308 | 0.267 | FAIL |
| `wheeze_score` | 0.5729 | 0.5340 | 0.190 | 0.174 | FAIL |

- Run 2 used extractors revised against five diagnosed failure modes. Each fix worked mechanically
  and traded one failure mode for its opposite (saturation → under-firing; silence → firing on 99%
  of everything). **Report the diagnostics, not just the AUROCs** — they are what makes this a
  finding rather than a null.
- **Two runs, then stop.** State explicitly that the test split was read twice and no third
  revision was attempted. This is the paper's methodological spine.

### 6. The reliability ceiling
- 7 senior physicians, blind, clean ICBHI: **47.77% ICBHI score, 23.23% sensitivity**, confidence
  2.88/5 (Tzeng 2025).
- 12 physicians: **κ < 0.40** for detailed adventitious-sound descriptions; κ = 0.62 / 0.59 for
  combined crackle / wheeze (Aviles-Solis 2016).
- ⏳ **Our clinician study** — one physician, 120 blind clips + 12 hidden duplicates, *with*
  calibration exemplars the JMIR study did not report giving. Yields: intra-rater agreement,
  clinician-vs-ICBHI agreement, and the first fine/coarse crackle reference on ICBHI.
- **The argument:** `crackle_fine_ratio` targeted a distinction humans agree on at κ < 0.40. Its
  ceiling was set before any DSP was written. Generalise carefully — this bounds *concept-level
  validation against these labels*, not respiratory ML overall.
- ⏳ **Inter-rater (second clinician, ~30 shared clips) — in flight.** Interpretation
  **pre-registered 2026-08-18 before the labels existed**, three outcomes fixed in advance
  (`INTER_RATER_READINGS`): (A) raters agree with each other but not ICBHI → the labels are the
  outlier; (B) raters disagree with each other → the ceiling is the task's; (C) everyone agrees →
  **no ceiling, our extractors are simply weak, and we say so.** Writing (C) down in advance is
  what makes (A) or (B) worth believing. Report the pre-registration itself in the methods —
  reviewers of a reliability paper will weight it heavily.

### 7. Discussion
- What a corrected baseline table looks like ⏳ (needs M2/M3/M22 re-run on the corrected split).
- Recommendations: report the official metric with Se/Sp separately; state the split's provenance
  and patient-independence; commit confusion matrices; report CIs and paired tests; validate
  detectors against a characterised reference standard.
- **Limitations, stated plainly:** single clinician (vs 7 and 12 in the cited studies); one corpus;
  our extractors are one implementation and a better one may exist; we cannot separate "our
  detector is weak" from "0.65 is unreachable here" from this experiment alone.

### 8. Conclusion
Concept-level validation on ICBHI is bounded by label reliability in a way the literature does not
report. Here is the corrected protocol, a pre-registered negative result, and the tooling.

---

## Evidence ledger — what exists vs. what is needed

**Have today**
- ✅ **Clinician labels, first pass** (`CLINICIAN_RESULTS_v1.md`) — intra-rater 88%/92% PASS;
  crackle κ = +0.035 (CI includes 0) and 23.1% sensitivity vs ICBHI, replicating Tzeng's 23.23%;
  `wheeze_score` rises 0.534 → 0.660 when scored against the physician instead of ICBHI
- ✅ Two G2 runs with full diagnostics + gate discipline
- ✅ Metric audit: 14 models recomputed, +0.11 mean inflation (`ICBHI_SCORE_AUDIT.md`)
- ✅ Split audit: mislabel + non-patient-independence (`official_split.py`)
- ✅ Synthetic validation of extractors, 43/43
- ✅ Human benchmarks (`Papers/HUMAN_BENCHMARKS.md`)
- ✅ Tooling: `owmtl_scores.py`, `audit_project.py`, corrected split loader
- ✅ Statistics: CIs + pairwise tests over the open-set suite

**Needed**
1. ⏳ **Second clinician, ~30 shared clips** — in flight, pre-registered. The only genuinely new
   evidence still outstanding, and the one reviewers will ask for by name.
2. ⏳ **M2/M3/M22 re-run on the corrected official split** — §7's baseline table needs real numbers,
   and their current ones are on the 11-patient split. **Highest-priority compute.**
   Notebooks are now patched (`patch_official_metric.py`) to (a) compute and **select checkpoints
   on** the official ICBHI score rather than the macro variant, (b) report both so per-model
   inflation is quotable from one run, and (c) dump per-cycle probabilities keyed by
   `<wav_stem>#<start>-<end>` so M2/M3/M22 become **pairable with DeLong without retraining**.
   Note the old checkpoints are unusable regardless: they were selected on the macro metric *and*
   trained on the wrong split, so the metric fix adds no compute cost to a re-run that was
   already mandatory.
3. ⏳ **A transformer baseline — decided 2026-08-28, currently missing.** The corrected-baseline
   table is CNN-only (M2 / M3 / M22). The project's one transformer, **M4 (AST, pretrained)**, is
   unusable as evidence: `results_M4.json` lives in Barshon's Drive, not the repo, so it never
   entered the metric audit and has no recomputed official score. Reviewers of a 2026
   respiratory-audio paper will ask why every baseline is a CNN. Fix: run at least one audio
   transformer (AST / SSAST / AudioMAE / BEATs) on the corrected split with
   `patch_official_metric.py` applied, so it emits the same per-cycle score dump and becomes
   DeLong-comparable to the CNNs without retraining them.
4. ✅ Kappa analysis of the returned labels — done, with bootstrap CIs on κ *and* sensitivity,
   fully reproducible from the documented command (seed 20260818).
5. ⏳ Release packaging: corrected split file, score format, audit tool.

**Explicitly NOT needed** — and must not be started: bottleneck head, intervention API, concept
leakage, a fourth mechanism.

---

## What could still sink it, and the honest answer to each

| Risk | Response |
|---|---|
| "This is just a negative result" | It is a negative result **plus** corrected baselines **plus** tooling. Lead with the corrections (§3), not the failure. |
| "n=1 clinician" | Stated as a limitation; the published n=7 and n=12 studies carry the reliability argument. Ours adds the fine-grained reference, which neither has. |
| "Your extractors are just bad" | **Concede it directly.** That is why §6 exists — the ceiling argument does not depend on our extractors being good. |
| "One dataset" | True. ICBHI is the field's default benchmark and the errors are ICBHI-specific; scope the claim to it. |
| Reviewer wants a working method | The real risk. Mitigation: make §3 substantial enough to stand alone as a reproducibility contribution. |
