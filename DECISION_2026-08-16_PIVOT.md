# Decision record — 2026-08-16: Gate G2 failed, project pivots to the reliability branch

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

**Status:** DECIDED · **Gate:** G2 (`OWMTL_Decision_Roadmap (v3).md`) · **Branch taken:** *if not
satisfied (b)* — the reliability-only paper.

This is a decision record, not a proposal. It is written so that the reasoning survives, and so
nobody re-litigates it in three weeks without new evidence.

---

## The decision

**The physics-derived concept bottleneck is retired as the paper's headline.** The project's
contribution is now the **evaluation and reliability** work: a corrected measurement protocol for
ICBHI, and the finding that concept-level detection on this corpus is bounded by the reliability of
its labels.

The concept extractors are **not deleted**. They survive as the *evidence* for that finding — a
serious, clinically-grounded attempt whose failure is informative — not as a proposed method.

---

## Why: what the evidence actually said

Gate G2 asked whether the DSP concept extractors reproduce ICBHI's crackle/wheeze labels materially
better than chance. It was run twice, with the second run using extractors revised against five
diagnosed failure modes.

| Concept | Run 1 | Run 2 | Gate (AUROC ≥ 0.65 + CI excludes chance) |
|---|---:|---:|---|
| `crackle_score` | 0.5506 | 0.5580 | **FAIL** |
| `wheeze_score` | 0.5729 | 0.5340 | **FAIL** |

AUPRC sat barely above prevalence both times (0.308 vs 0.267; 0.190 vs 0.174).

The revisions worked *mechanically* — crackle rate fell from ~10/s to ~2.5/s, killing the
saturation; the wheeze high-pass ended the 47% silence — but each fix traded one failure mode for
its opposite. The wheeze detector went from silent on half of wheeze cycles to firing on 99% of
everything. `crackle_fine_ratio` never worked on real audio at all (85% zero).

**The test split has now been read twice. A third revision aimed at this gate would be fitting it,
and any resulting pass would be uninterpretable.** That is the discipline the gate existed to
enforce, and honouring it is the whole reason the result is worth anything.

## What this does *not* mean

It does not mean "our extractors are bad, therefore the idea is dead." Two things must be
distinguished, and the write-up must keep distinguishing them:

1. Our extractors are weak. **True, and demonstrated twice.**
2. The 0.65 target is reachable on this reference standard by *any* method. **Unknown, and this
   experiment cannot decide it.**

Seven senior physicians score **47.77%** ICBHI on this same corpus, missing ~77% of the cycles
ICBHI marks abnormal (Tzeng et al., *JMIR AI* 2025). Physician inter-observer agreement on
*detailed* adventitious-sound descriptions is **κ < 0.40** (Aviles-Solis et al., 2016). See
`Papers/HUMAN_BENCHMARKS.md`.

`crackle_fine_ratio` was targeting a distinction that physicians themselves agree on at κ < 0.40.
It had a low ceiling before a line of DSP was written.

---

## What the project's contribution now is

Not a mechanism. A **measurement**. Specifically, four things that are all already evidenced:

1. **A corrected ICBHI evaluation protocol.** Three errors found in our own work, each of which
   plausibly recurs in the wider literature:
   - the reported "ICBHI score" was a macro variant inflating results by ~0.11 (mean across 14 models, worst 0.22);
   - the "official 60/40 split" was actually an 11-patient / 7.1% split, mislabelled in four models;
   - the genuine official split is **not patient-independent** — patients 156 and 218 appear on both sides.
2. **A negative result with a gate that could fail, and did.** Two runs, pre-registered threshold,
   test split read once per run, no post-hoc tuning.
3. **Label reliability as the binding constraint**, evidenced by the human benchmarks plus our own
   clinician study — which ships calibration exemplars the published physician study did not.
4. **Reusable tooling**: paired DeLong / McNemar / bootstrap over a canonical raw-score format
   (`Asif's/Statistics/owmtl_scores.py`), an audit tool that catches all three error classes above,
   and the corrected split loader.

## Scope of the claim

Deliberately bounded. We are **not** claiming ICBHI's labels are wrong, that the benchmark should be
abandoned, or that concept bottlenecks cannot work for auscultation. We are claiming that
**concept-level validation against ICBHI cycle labels is bounded in a way the literature does not
currently report**, and supplying the corrected protocol and tooling to measure it.

---

## What continues, what stops

**Continues**
- The clinician listening study — already in flight, and the one piece of genuinely new evidence.
  Its value is now the fine-grained reliability question, not extractor validation.
- Re-running M2/M3/M22 on the corrected official split (needed for the corrected-baselines table).
- `Asif's/Statistics/` tooling and the audit.

**Stops**
- Tuning the concept extractors. Two runs is the budget; it is spent.
- The bottleneck head (M13 re-wire), intervention API, and concept-leakage measurement — all
  presupposed a working bottleneck. Gate G3 is not reached and will not be.
- Any new mechanism. Chasing a fourth idea with the time remaining is how this ends with no paper.

## Gates, updated

| Gate | Status |
|---|---|
| G0 annotation | ✅ **first pass returned 2026-08-18** — intra-rater PASS; see `CLINICIAN_RESULTS_v1.md` |
| G1 kill-search | ✅ clear (2026-08-15) |
| G2 extractor validity | ❌ **FAILED twice** → branch (b) |
| G3 A-vs-B headline | **not reached** — presupposed G2 |
| G4 device structure | ❌ failed early (3/126 patients span devices) |
| G5–G7 | superseded by the reframe |

---

## Honest assessment of where this lands

A rigorous negative result with corrected baselines and released tooling is **comfortably
publishable**, and is a genuinely useful contribution to a literature with 135+ papers on this one
corpus. Whether it clears a Q1 biomedical venue depends on execution completeness, not on finding
another mechanism: the realistic path is a benchmark/reproducibility contribution, executed fully,
with baselines and tooling released.

That is a smaller claim than the project started with. It is also the first claim it has made that
the evidence actually supports.
