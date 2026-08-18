# Supervisor briefing — project direction

**For:** the meeting with Dr. Khan · **Prepared:** 2026-08-18
**Backing documents:** `DECISION_2026-08-16_PIVOT.md`, `PAPER_OUTLINE.md`, `Papers/HUMAN_BENCHMARKS.md`

---

## The 30-second version

> We built physics-derived detectors for named lung sounds — crackle, wheeze, fine vs. coarse
> crackle — and tested whether they measure what their names claim. They don't: AUROC 0.56 against
> ICBHI's labels, twice, against a pre-registered threshold of 0.65.
>
> But the same corpus defeats humans. Seven senior physicians score 47.77% on it, and physician
> agreement on detailed sound descriptions is κ < 0.40. So the honest finding isn't "our detector
> failed" — it's **that this benchmark cannot support the concept-level claims the field is making
> on it**, and we can now show that with numbers.
>
> The paper is the corrected evaluation protocol, the negative result, and the reliability ceiling.
> The clinician's labels are the piece that makes it ours rather than a literature review.

---

## 1. Why we need the doctor's validation set

Our concept detectors scored **AUROC 0.5506 → 0.5580** (crackle) and **0.5729 → 0.5340** (wheeze)
against ICBHI's cycle labels, across two runs with two different parameterisations. There are
exactly two explanations:

| | Explanation | How we tell |
|---|---|---|
| **(a)** | Our extractors are weak | Compare against a *different* reference standard |
| **(b)** | ICBHI's labels are themselves unreliable | Same test — if a clinician also disagrees with ICBHI, it's the labels |

We cannot separate these from ICBHI alone. The clinician gives us an **independent reference on the
same clips**, so the comparison finally has two sides.

**Two things make it worth the physician's 75 minutes:**

1. **It is the only possible ground truth for 5 of our 9 concepts.** ICBHI labels crackle/wheeze
   *presence* and nothing finer. Fine vs. coarse crackle, wheeze pitch, and rhonchi have **no
   reference standard in the dataset at all** — so they are currently unfalsifiable. Only a
   clinician can change that.
2. **It measures the benchmark, not just us.** Comparing the clinician against ICBHI's own labels
   is a direct measurement of how reliable those labels are — a result about the corpus that 135+
   published papers depend on.

**Design choices worth mentioning:** the clinician labels **blind** (never sees our detector's
output — otherwise the "agreement" is partly agreement with our own suggestion), gets **six
calibration exemplars**, and 12 clips are **secretly duplicated** so we can measure whether they
agree with *themselves*. We check that first: if intra-rater agreement is poor, the task is too
hard from recordings, and that is itself the finding.

---

## 2. Where the project goes

Retired as the headline: the open-world cross-task mechanism (2026-08), the device/disease
disentanglement (failed feasibility), and the concept bottleneck as a *proposed method* (failed
Gate G2 twice).

**The contribution is now the evaluation and reliability work.** Four deliverables, all already
evidenced:

| # | Deliverable | Status |
|---|---|---|
| 1 | **Three evaluation errors + corrected baselines** — found by auditing our own pipeline | ✅ found; re-runs in flight |
| 2 | **A pre-registered gate that could fail, and did** — twice, with no post-hoc tuning | ✅ done |
| 3 | **Label reliability as the binding constraint** — published human benchmarks + our clinician study | ⏳ clinician in flight |
| 4 | **Released tooling** — paired DeLong/McNemar, corrected split loader, audit tool | ✅ built |

### The three errors (this is the part reviewers will cite)

Each was found in **our own work**, and each silently invalidated results:

| Error | What it was | Effect |
|---|---|---|
| Non-standard metric | reported "ICBHI score" was a macro variant, not the challenge metric | **+0.11 mean inflation** across 14 models, **+0.22** worst |
| Mislabelled split | "official 60/40" was a `patient_id ≤ 111` fallback | **11 test patients, 7.1% of cycles** — not 40% |
| Split not patient-independent | the official ICBHI file assigns *recordings* | patients **156** and **218** sit on both sides |

The metric error also **hid broken models**: one reported 0.5137 while detecting 9% of abnormal
events; another reported 0.5506 with **specificity 0.0000**. Both looked mediocre-but-working.

If a team auditing itself this hard made all three, they are unlikely to be unique to us.

---

## 3. What we will actually do next

1. **Re-run M2 / M3 / M22** on the corrected split → the corrected baseline table. *(In progress —
   notebooks are fixed and run unattended now.)*
2. **Clinician labels return** → intra-rater check first, then κ against ICBHI and against our
   detectors.
3. **Write up**, leading with the corrected protocol (§3 above), not with the failure.
4. **Release** the corrected split loader, score format, and audit tool.

**Explicitly not doing:** more extractor tuning (the gate's budget is spent), the bottleneck head,
intervention, concept leakage, or a fifth mechanism.

---

## 4. What's the novelty

Stated plainly, because this is the question that matters.

**It is not** a new architecture, a new mechanism, or a SOTA number. We should not pretend
otherwise.

**It is:**

1. **The first characterisation of the reliability ceiling for concept-level validation on ICBHI.**
   Papers report detector performance against these labels without ever asking what the labels can
   support. We quantify it.
2. **Three concrete, reproducible evaluation errors with corrected baselines** — a
   reproducibility contribution to a literature with 135+ papers on one corpus.
3. **A pre-registered gate methodology**, in a subfield where thresholds are almost always chosen
   after seeing results. We set 0.65 in advance, read the test split once per run, failed twice, and
   did not tune again. That discipline is itself unusual enough to be worth reporting.
4. **The first fine-grained (fine vs. coarse crackle) clinician reference on ICBHI clips** — if the
   physician's labels come back usable.

This is a **rigor contribution rather than a gadget contribution**. That is a recognised category —
reproducibility and benchmark papers are published and heavily cited — and it happens to be the one
axis where this project is genuinely strong.

---

## 5. Likely pushback, and honest answers

**"So you failed."**
> We failed to build a working detector. We succeeded at finding out *why*, and at showing the
> reference standard is a large part of it. A negative result with a pre-registered threshold and
> corrected baselines is publishable; an unexamined positive one isn't. We also caught three
> evaluation errors that were silently inflating our own numbers by up to 0.22.

**"Where is the novelty if you're not proposing a method?"**
> The novelty is the measurement. Nobody has asked what ICBHI's labels can support, and we can now
> answer it with our own experiment plus a clinician reference. §4 above.

**"Why abandon the concept bottleneck?"**
> The bottleneck needs concepts that measure what they claim. Ours reach AUROC 0.56 after two
> attempts. Building a diagnosis head on top of that would produce a result nobody could interpret.
> Gate G2 existed precisely to catch this before we spent weeks on the head.

**"You've changed direction four times."** *(This one is fair — concede it.)*
> Yes, and the first three changes were expensive because we searched the literature *after*
> building. The cross-task mechanism turned out to be Zamir et al. 2020; the device axis turned out
> to be infeasible on ICBHI. We now run the kill-search first — it takes 30 minutes. This pivot is
> different in kind: it's the first direction supported by evidence we already have rather than by
> hope.

**"Can this be Q1?"**
> Honestly: borderline, and it depends on finishing rather than on finding another mechanism. A
> negative result alone is a hard sell. A negative result *plus* corrected baselines *plus* released
> tooling is a reproducibility contribution, which BSPC-class venues do publish. That path is
> finishable in the time we have. Chasing a fifth mechanism is not.

**"Should you switch datasets?"**
> No. The errors are ICBHI-specific, and ICBHI being the field's default is exactly what makes them
> worth reporting. SPRSound is worth *one supporting table* if time allows — its annotation
> provenance is better documented (11 pediatric doctors) — but as a supporting section, not a new
> direction.

---

## 6. The one decision to ask him for

**Does he accept a rigor/reproducibility contribution as the deliverable, instead of a proposed
method?**

If yes: the plan above is finishable and we stop hunting for mechanisms.
If no: we need to talk about scope, because there is not enough runway to build, validate, and
write up a fifth idea — and the last four did not survive contact with the evidence.
