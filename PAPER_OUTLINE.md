# Paper outline — the reframed contribution

> ## 🔴 PARTLY SUPERSEDED — 2026-08-30, see `DECISION_2026-08-30_CONSOLIDATION.md`
>
> N11 (supervised ceiling) shows a probe on a **frozen AudioSet embedding — a network that never
> saw a respiratory corpus — reaches AUROC 0.71 (crackle) / 0.76 (wheeze)** on the same test cycles
> G2 failed on. The labels are learnable well above the 0.65 gate.
>
> **The label-reliability *ceiling* argument is retracted.** Our extractors were weak; the reference
> standard was not the binding constraint. The clinician κ numbers stand as a *finding* — machines
> reproduce these labels better than trained listeners agree with them — not as a bound.
>
> Everything else here (the three evaluation errors, the corrected baselines, the two-run gate and
> its failure, the released tooling) is unaffected.

---

**Drafted:** 2026-08-16, immediately after the G2 pivot (`DECISION_2026-08-16_PIVOT.md`).
**Rule followed throughout:** every claim below is traceable to a result that exists *today*, or is
explicitly marked ⏳ pending. Nothing is written that we hope to be able to say.

---

## Working title

> **What can a concept detector actually be validated against? Corrected evaluation baselines and a
> retracted ceiling for adventitious-sound detection on ICBHI**

Alternative, and the stronger framing after N11:
*"Machines reproduce ICBHI's cycle labels better than clinicians agree with them: a corrected
evaluation protocol, a failed concept gate, and the retraction of our own ceiling claim"*

## The one-sentence claim

ICBHI's cycle labels are learnable well above the threshold our pre-registered gate set — a probe on
a frozen AudioSet embedding that never saw a respiratory corpus reaches AUROC 0.71 / 0.76 on the
same test cycles our physics-derived detectors failed on at 0.56 / 0.57 — so we retract our own
label-reliability ceiling claim, report that our detectors were weak, and show instead that a
self-consistent clinician agrees with those same labels at κ = 0.035; we supply the corrected
evaluation protocol and tooling that makes each part checkable.

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
  3. A supervised ceiling estimate that **refutes our own earlier claim** (the labels are
     learnable), plus the human-agreement comparison it turns into the real finding.
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

### 6. What these labels actually support *(rewritten 2026-08-30 — the ceiling is retracted)*
- **N11, pre-registered, decides the question G2 could not.** A logistic probe fitted on the 79
  train patients, scored on the same 2,636 test cycles / 47 test patients:

  | space | crackle | wheeze |
  |---|---:|---:|
  | our gate, single score | 0.5580 | 0.5729 |
  | our 14 concepts as a vector | **0.6608** | 0.5815 |
  | AST frozen (never saw ICBHI) | **0.7115** | **0.7621** |
  | M2 encoder (trained on these labels) | **0.7451** | **0.8673** |

  The pre-registered **"no ceiling"** branch fired. **Retract the ceiling claim.** Our extractors
  were weak — say it plainly; that concession is what the gate discipline was for.
- **Second finding from the same table:** the gate thresholded one hand-built *scalar*. The same 14
  concepts *as a vector* clear 0.65 on crackle. A validity gate for a concept suite should be posed
  over the suite. Report this as a criticism of our own design.
- **What replaces the ceiling, and it is sharper:** 7 senior physicians reach **47.77%** ICBHI score
  and 23.23% sensitivity (Tzeng 2025); our own rater answers 116/132 clips, agrees with ICBHI at
  **κ 0.035** (crackle, CI spans 0) / **0.266** (wheeze) at Se 0.231 — within a point of Tzeng —
  while agreeing with *themselves* at κ 0.714 / 0.600 on the 8 answered hidden duplicates.
  **Machines reproduce this reference standard substantially better than trained listeners agree
  with it.**
- **Still bounded by human agreement, but only one concept:** `fine_crackle_ratio` — κ < 0.40 among
  12 physicians (Aviles-Solis 2016), R² ≈ 0 in every representation, constant on the clinician
  subset. Not estimable by any route here.
- **Scope:** not a claim that ICBHI's labels are wrong, and no longer a claim that reliability bounds
  detection.
- ❌ **Inter-rater (second clinician) — dropped 2026-08-30.** No second rater is available, and N11
  removed the need for one: it bounds the reference standard directly, which was the only job the
  second rater had. Note the shape of the outcome — the pre-registered `INTER_RATER_READINGS`
  branch (C), *"no ceiling, our extractors are simply weak, and we say so"*, is the branch that
  fired, just via a probe instead of a rater. **Report the pre-registration in the methods anyway**;
  a reliability paper that names the outcome that would refute it, and then reports that outcome,
  is worth more than one that never risked it. Single-rater goes to Limitations.

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
1. ✅ **Second clinician — dropped, not outstanding.** N11 replaces it (see §6). If a reviewer asks
   for inter-rater agreement, the answer is that the ceiling question it would have addressed is
   answered by a supervised probe on the full test set rather than by 30 shared clips.
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
