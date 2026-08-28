# Clinician labelling — first-pass results (v1)

**Received:** 2026-08-18 · **Rater:** 1 physician, blind, with 6 calibration exemplars
**Data:** `Asif's/labels v1 - labels.csv` + `Asif's/clip_key.csv`
**Reproduce:** `python3 "Asif's/engine/analyze_clinician_labels.py" --labels ... --key ... --concepts ...`

> **v1 — the physician has said he will review further over 2–3 days.** Every number here may move.
> Treat as a strong provisional read, not a final result.
>
> **Every figure in this document is printed by the command above.** κ and sensitivity intervals
> are 10,000-draw percentile bootstraps resampling *clips* (seed 20260818), so they are exactly
> reproducible rather than quoted from a one-off calculation.

---

## Headline

**The labels are internally reliable and profoundly disagree with ICBHI.** That combination is the
finding — it is not a sloppy rater, it is a rater who is consistent with himself and inconsistent
with the benchmark.

| Measure | Result |
|---|---|
| **Intra-rater agreement** (hidden duplicates) | **88%** exact, **92%** presence-only → **PASS** |
| **vs ICBHI, crackle** | κ = **+0.035** (95% CI −0.122 to +0.192) — *no agreement beyond chance* |
| **vs ICBHI, wheeze** | κ = **+0.266** (95% CI +0.085 to +0.445) — fair |
| **Crackle sensitivity vs ICBHI** | **23.1%** (95% CI 12.0–35.3%) |

**That 23.1% is a near-exact independent replication of Tzeng et al. (2025), where seven senior
physicians reached 23.23% on this same corpus** — with a different rater, a different clip sample,
and calibration exemplars their study did not report providing.

---

## 1. Quality gate: did he agree with himself? — **PASS**

12 clips were secretly duplicated. Four pairs lost one copy to `unusable`, leaving 8 comparable.

| | Agreement |
|---|---|
| crackles, exact | **7/8 (88%)** |
| wheeze, exact | **7/8 (88%)** |
| crackles, presence-only (all 12) | **11/12 (92%)** |

The pack's rule was *">3 of 12 self-disagreements → the task is too hard from recordings."* He had
**one**. The single crackle flip was `clip_039` (fine) vs `clip_069` (none); the single wheeze flip
was `clip_029` (none) vs `clip_087` (wheeze).

**This is the result that makes everything below worth reading.** Had it failed, the disagreement
with ICBHI would have been uninterpretable.

*Caveat: designed for 12 comparable pairs, got 8, because unusable clips were drawn into the
duplicate set. If a v2 pack is ever built, draw duplicates only from clips already judged usable.*

---

## 2. Clinician vs ICBHI — the core finding

Usable clips, definite calls only, excluded **per column** (n=108 crackle, 110 wheeze) — a
non-definite wheeze call does not disqualify that clip's crackle call.

| | Physician + | ICBHI + | Raw agreement | κ (95% CI) | Sensitivity (95% CI) |
|---|---:|---:|---:|---|---|
| **crackle** | 23 | 52 | 52.8% | **+0.035** (−0.122, +0.192) | **23.1%** (12.0, 35.3) |
| **wheeze** | 16 | 37 | 71.8% | **+0.266** (+0.085, +0.445) | 29.7% (15.4, 45.2) |

**Crackle κ's confidence interval includes zero.** Against a literature benchmark of
physician-vs-physician crackle κ ≈ 0.62 (Aviles-Solis 2016), agreement with ICBHI's crackle labels
is statistically indistinguishable from chance.

Wheeze fares better (κ = 0.266, CI excludes zero) — consistent with wheezes being easier for humans
than crackles (70–90% vs 55–75% detection rates in the literature).

He called adventitious sounds at **roughly half** ICBHI's rate: 21.3% crackle-positive where the
stratified pack contained ~45.8%, and 14.5% wheeze-positive where it contained ~33.3%.

**Sensitivity analysis:** treating his `ambiguous` calls as positive instead of excluding them
changes nothing material (crackle κ −0.003 [−0.166, +0.161], sensitivity 25.9% [14.8, 37.9]; wheeze κ +0.289
[+0.112, +0.460], sensitivity 35.0% [20.5, 50.0]).
The finding does not depend on that judgement call.

---

## 3. Our extractors, scored against the clinician instead of ICBHI

This is what G0 was *for*: a second reference standard, so "is our detector bad or are the labels
unreliable?" becomes answerable.

| Concept | vs ICBHI (M39) | **vs clinician** | 95% CI | Read |
|---|---:|---:|---|---|
| `crackle_score` | 0.5580 | **0.5143** | [0.396, 0.633] | chance against **both** → our detector's fault |
| `wheeze_score` | 0.5340 | **0.6599** | [0.503, 0.817] | **improves, and clears the 0.65 gate** |
| `rhonchi_score` | — | **0.7090** | [0.480, 0.938] | best point estimate, but n=10 positives |
| `crackle_fine_ratio` | — | 0.4803 | [0.217, 0.744] | **underpowered** — only 4 negatives |

**The answer differs by concept, which is the genuinely interesting part:**

- **Crackle: it's us.** ~0.51–0.56 against both reference standards. Changing the reference does not
  rescue it. Our crackle detector is weak, full stop.
- **Wheeze: it's substantially the labels.** The same detector scores 0.534 against ICBHI and
  **0.660 against a physician** — crossing the pre-registered 0.65 threshold. The detector did not
  change; only the reference standard did.

**Do not over-read the wheeze result.** n=16 positives, and the CI's lower bound is 0.503 — it
clears chance by a hair. It is a signal worth reporting with its CI attached, not a claim that the
wheeze detector works.

---

## 4. The `ambiguous` label — our instrument bug, not his deviation

> **Corrected 2026-08-18** after asking him. An earlier version of this document said he "invented"
> the category and read it as evidence our taxonomy was inadequate. That inference was wrong, and
> the real explanation is simpler and entirely our fault.

We shipped **six** calibration exemplars, one of them named `example_ambiguous.wav` — but the
answer options only ever offered `none / fine / coarse / both / unsure`. **We gave him a sixth
reference sound with no matching answer.** He did exactly the right thing: matched what he heard to
our exemplar and wrote the name we had used for it.

In his words, `ambiguous` means the clip **sounds like that reference exemplar — none of the named
classes fit it.**

Two consequences:

1. **It is not an uncertainty flag.** `unsure` ("I can't tell") was used once; `ambiguous`
   ("this is a real sound, but not one of your categories") eight times. Treating them as the same
   thing would be wrong. His confidence on `ambiguous` clips was 43% high — lower than `none`
   (89%) but far from absent, consistent with a *deliberate* call rather than a hedge.
2. **~7% of clips contained a sound fitting none of the four named categories** — and that is an
   *anchored* judgement, because he had an exemplar for it. That is a more defensible statement
   than the taxonomy inference it replaces, and it still points the same direction as
   Aviles-Solis et al.: named categories do not cleanly cover what is audible in this corpus.

**Lesson for any future pack:** every reference exemplar must have a matching answer option. Ours
did not, and only a direct question to the rater surfaced it.

His confidence tracks difficulty sensibly, which is further evidence the labels are trustworthy:

| Call | share rated "high confidence" |
|---|---|
| `none` | 89.4% |
| `fine` | 63.2% |
| `ambiguous` | 42.9% |

**He also marked 12.9% of clips `unusable`** — a real statement about ICBHI's audio quality, and
worth a line in the paper.

**`clip_025` — resolved.** He confirmed it is genuinely **unusable**; the stray `unsure`/`unsure`
labels are ignored. The analyser already excluded it (all `unusable` clips are dropped before any
agreement is computed), so no number in this document changes. The raw file is left unedited on
purpose — it is the rater's primary record.

---

## 5. What this changes

**For the paper:** this is now the empirical core of §6 (the reliability ceiling). We have our own
independent replication of the published sensitivity figure, with CIs, plus the crackle-vs-wheeze
asymmetry that no cited study reports.

**For the concepts:** `crackle_fine_ratio` cannot be validated — 19 fine vs **3 coarse** calls. That
is not a failure of the exercise; it is the κ < 0.40 literature result showing up in our own data.
It stays a proxy permanently, now with our own evidence rather than only a citation.

**For `CONCEPT_VALIDATION`:** no promotions yet. `wheeze_score` is the only candidate and its CI is
too fragile to act on before the physician's review lands.

---

## 6. Open items

1. **His 2–3 day review** — may change labels. Everything here re-runs in one command.
2. ~~Ask what `ambiguous` meant~~ — **answered**, see §4. It was our missing answer option.
3. ~~`clip_025`~~ — **answered**, confirmed unusable.
4. **Second rater — in progress.** Gives inter-rater κ, the one thing reviewers will ask for that
   we currently lack. `--rater2 <csv>` is built and tested against synthetic raters; it collapses
   hidden duplicates to their source clip so no recording is counted twice, and scores both raters
   against ICBHI on the *same* subset so the comparison is like-for-like.

   **The interpretation is pre-registered (2026-08-18, before those labels exist)** in
   `INTER_RATER_READINGS` in the analyser, and the script prints which reading fired:

   | | Condition | Meaning |
   |---|---|---|
   | **A** | κ_inter ≥ 0.40 **and** κ_inter − κ_ICBHI ≥ 0.20 | clinicians agree with each other, not with ICBHI → **the benchmark labels are the unreliable term**. Strongest form of our claim. |
   | **B** | κ_inter < 0.40 | clinicians do not agree with each other either → the ceiling belongs to the *task*, not to ICBHI's annotation. Argument survives, target changes. |
   | **C** | κ_inter ≥ 0.40 **and** κ_ICBHI ≥ 0.40 | no ceiling; the honest conclusion becomes *our extractors are weak*. **Costs us the central claim — written down in advance precisely so it gets reported if it happens.** |

   At n≈30 every interval will be wide; the script marks results below n=30 descriptive rather
   than confirmatory, and that caveat travels with the number into the paper.
5. Numbers here are **n=1 rater** and must be reported as such until rater 2 lands.
