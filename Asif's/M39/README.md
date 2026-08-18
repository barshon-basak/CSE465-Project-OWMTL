# M39 — Physics-Derived Acoustic Concept Extraction & Validation (Gate G2)

**Owner:** Asif · **Gate:** G2 · **Requires:** ICBHI audio + `Asif's/ICBHI_challenge_train_test.txt`
**Status:** notebook ready, tested 43/43. **First real run done 2026-08-16 — extractors then
revised; a re-run is needed.**
**No model, no training, no GPU** — pure DSP.

Computes 9 physics-derived acoustic concepts for every annotated ICBHI cycle and answers the G2
question: **do the extractors measure what their names claim?**

## What can and cannot be validated

ICBHI labels each cycle for crackle *presence* and wheeze *presence* — nothing finer.

| | Concepts | Validated against |
|---|---|---|
| **Validatable** (2) | `crackle_score`, `wheeze_score` | ICBHI labels — AUROC + DeLong CI |
| **Proxy** (5) | `crackle_fine_ratio`, `crackle_rate_hz`, `wheeze_pitch_hz`, `rhonchi_score`, `inspiratory_fraction` | **nothing** |
| **Descriptive** (2) | `spectral_flatness`, `temporal_papr` | n/a |

The proxies get a distributions section that makes **no accuracy claim**. Promoting any of them
requires the clinician exercise in `CLINICIAN_LABELING_PACK.md` (G0). The status is machine-readable
in `concept_extractors.py::CONCEPT_VALIDATION` and copied into the results JSON, so it travels with
the numbers rather than living only in prose.

**Never write "fine crackle" for `crackle_fine_ratio`.** Write "physics-derived proxy for
fine-crackle fraction". Naming a proxy after the thing it proxies is how the ICBHI-metric problem
started.

## The gate  *(tightened 2026-08-16)*

A concept passes if **AUROC ≥ 0.65 AND its 95% CI excludes chance**, on the test split.

> **Why both conditions.** The first version required only "CI excludes chance". At n=2,636 cycles
> that admits AUROC 0.52 — statistically significant, practically useless. The first real run duly
> reported **PASS** with crackle 0.5506 / wheeze 0.5729 while the crackle detector was firing on
> 100% of cycles at ~10 events/s in Normal and Crackle alike. A gate that cannot fail is not a gate.

**AUPRC is reported alongside**, against its own chance baseline (the positive prevalence, ~0.27
crackle / ~0.17 wheeze). AUROC is optimistic under class imbalance, so the two together are more
honest than either alone.

**Train is printed before test**, and the test block carries a one-look banner. Diagnose on train,
change the extractors, then read test once. Re-tuning after reading test is fitting the gate.

| Verdict | Meaning | Roadmap branch |
|---|---|---|
| **PASS** (2/2) | Build the bottleneck head. Full plan live. | G2 satisfied |
| **PARTIAL** (1/2) | Shrink to the reliable concept subset, thinner bottleneck. | G2 (a) |
| **FAIL** (0/2) | Pivot to the reliability-only paper. | G2 (b) |

All three are live branches. None is a dead end.

> **Do not tune the extractor constants to make a failing concept pass.** They are clinically
> motivated (CORSA) and were fixed before any ICBHI result was seen. Adjusting them against the test
> split is fitting the gate, not passing it — and they live in `concept_extractors.py`, so any
> change shows in the diff.

## Two things done differently from M35

**Raw waveform, not log-mels.** M35 computed flatness/PAPR from a normalised log-mel via
`exp(spec * 3.0)`. Fine for a soft penalty; not a defensible basis for a number the paper calls
"spectral flatness". And crackles are 5–15 ms while the shared mel hop is 10 ms — a mel frame barely
resolves one.

**True cycle duration, never the 8 s tiled clip.** M2/M3 tile short cycles to 8 s for the CNN. Doing
that here would repeat every crackle ~4× and make `crackle_rate_hz` fiction.

## Three bugs the synthetic tests caught before this ever ran

`Asif's/engine/test_concept_extractors.py` builds signals with *constructed* acoustic ground truth
and asserts the extractors recover it:

1. A pure 400 Hz tone scored **rhonchi 0.970** — the 60–300 Hz band still had a locally-prominent
   leakage bin. Fixed by requiring the in-band peak to be a real share of the frame's global peak.
2. The brick-wall FFT band-pass rang (Gibbs), producing a **phantom crackle ~10 ms after every real
   one** — a 50% inflation of `crackle_rate_hz`. Fixed with raised-cosine band edges.
3. Detection fired on **numerical residue at 1e-16** when the background was near-silent
   (median/MAD → 0). Fixed with an absolute energy floor — a crackle is an *explosive* sound, not
   merely a statistical outlier.

Running on real ICBHI audio would have surfaced none of these. It only shows the code executes,
never that a detector named "crackle" responds to crackles.

## Running it

Colab or local. ~10–15 min for 6,898 cycles; no GPU. Needs:

1. **ICBHI audio** — cell 0b downloads it (paste your Kaggle key).
2. **`ICBHI_challenge_train_test.txt`** — committed in the repo, upload to `/content/`.

The notebook **refuses to run without the split file** rather than falling back to a patient-id
rule. That silent fallback is what gave M2/M3/M12/M22 an 11-patient test set mislabelled as the
official 60/40 split.

Split is the official file with patients **156** and **218** reassigned to train (they appear on
both sides, violating Protocol §1). Cost: 12 of 381 test recordings.

## Outputs

```
results_M39.json           gate_g2 verdict + per-concept CI + validation status per concept
scores_M39.csv             CC1 raw per-cycle scores  (unit_type='cycle')
concepts_M39.csv           the full per-cycle concept table — input to the bottleneck head
concept_distributions.png  9 boxplots by ICBHI label group, colour-coded by validation status
```

`scores_M39.csv` is `unit_type='cycle'`, so `owmtl_scores.py` will **refuse** to pair it with
M29/M38's patient-level scores. That guard is deliberate — it is trap #6 from `CLAUDE.md` encoded in
the tooling.

## Testing

43/43 end-to-end, on a corpus built from **real stems** in the official split file with **synthesised
audio whose acoustics match the labels** — crackle-labelled cycles genuinely contain crackles. That
makes the run a real test of the whole chain (labels → audio → extractors → AUROC → verdict), not a
smoke test: a bug anywhere in it shows up as a wrong verdict.

Plus a **negative control**: identical labels, no acoustic signal.

| Corpus | crackle AUROC | wheeze AUROC | Verdict |
|---|---:|---:|---|
| Sounds planted | 0.998 | 1.000 | PASS |
| No signal | 0.477 | 0.500 | **FAIL** |

The gate correctly fails when there is nothing to detect. A gate that cannot fail is not a gate.

Also asserted: patient-independence (156/218 never in test), the 2/5/2 validation-status split, CC1
schema and `unit_type`, that the dump loads in the real `owmtl_scores.py` tooling, and that the
`passes` flag always agrees with its own CI.

## Regenerating

```
python3 gen_M39.py
```

The extractors are **spliced from `Asif's/engine/concept_extractors.py`**, not copied. Edit the
module and regenerate; never hand-edit the notebook. Cell 4 re-runs the core synthetic assertions
inside Colab, so a run proves the spliced copy behaves like the tested module rather than assuming
it.


---

## Run history

**Run 1 (2026-08-16) — reported PASS, but the extractors were broken.**

| Concept | AUROC | 95% CI |
|---|---:|---|
| `crackle_score` | 0.5506 | [0.5260, 0.5751] |
| `wheeze_score` | 0.5729 | [0.5462, 0.5996] |

Both "passed" the original CI-only rule purely because n=2,636 makes a trivial effect significant.
The diagnostics showed the extractors were not working on real auscultation at all:

* the crackle detector fired on **100% of cycles** — 9.70 events/s in Normal vs 10.19/s in Crackle;
* the wheeze detector was **silent on 47% of wheeze-labelled cycles**;
* `crackle_fine_ratio` was **zero on 80% of cycles** — the fine/coarse split collapsed entirely.

### Read against the human benchmark before concluding anything

Added 2026-08-16 after a literature check (`Papers/HUMAN_BENCHMARKS.md`). AUROC 0.55 looks like
failure in isolation. Against what humans achieve on this exact corpus it looks different:

| Reference | Result |
|---|---|
| **Our `crackle_score`** vs ICBHI labels | AUROC 0.5506 [0.5260, 0.5751] |
| **7 senior physicians** vs ICBHI labels, clean audio (Tzeng et al., *JMIR AI* 2025) | **ICBHI score 47.77%**, sensitivity **23.23%**, confidence 2.88/5 |
| **12 physicians**, detailed adventitious-sound descriptions (Aviles-Solis et al., 2016) | **κ < 0.40** (poor-to-fair) |
| Same study, combined categories | crackles κ = 0.62, wheezes κ = 0.59 |

Two things follow:

1. **The ceiling on this task is set substantially by label reliability, not only by detector
   quality.** Seven senior physicians miss ~77% of the cycles ICBHI marks abnormal. Our extractors
   are weak, but the reference standard is not a clean one.
2. **`crackle_fine_ratio` targets a distinction physicians agree on at κ < 0.40.** Keeping it a
   permanently-labelled *proxy* is now backed by published inter-observer data, not just caution.
   If the clinician's duplicate-clip check comes back weak on fine/coarse, that is the *expected*
   result.

**This is context, not an excuse.** A detector at 0.55 is still weak, the revised extractors still
need their re-run, and "humans are also bad at this" is a claim about the task, not evidence that
our method works. Report both numbers together; never the human benchmark alone as cover.

**Extractor revisions made in response** (all principled fixes to diagnosed failure modes, not
threshold tuning against the test split):

1. **Local adaptive baseline** — median/MAD over a sliding ~60 ms window instead of the whole
   cycle. Continuous breath sound was setting the cycle-wide reference, so every fluctuation in it
   read as an outlier.
2. **Width cap (30 ms)** — CORSA puts crackles at 5–15 ms. A slower excursion is breath-sound
   amplitude modulation.
3. **Local-ratio test** — the peak must exceed its local background by ≥2×. A crackle is
   *explosive*; on a steady signal the local MAD collapses and the z-score explodes on noise.
4. **Edge guard** — skip half a baseline window at each end, where the sliding median is
   edge-padded rather than measured and the cut itself reads as a transient.
5. **High-pass at 80 Hz before wheeze analysis** — real auscultation is dominated by sub-100 Hz
   energy (heart, muscle, handling), so the frame's global peak sat below the wheeze band and the
   `min_band_share` test rejected genuine wheezes.

Verified on a simulated breath-sound background, where the revised detector is now monotonic
(crackle_score 0.499 breath-only → 0.680 with 6 crackles → 0.840 with 15) rather than saturated.
**This has not yet been confirmed on real ICBHI audio** — that is what the re-run is for.

---

## Run 2 (2026-08-16) — **GATE G2: FAIL (0/2)**

Revised extractors, tightened gate (AUROC ≥ 0.65 **and** CI excluding chance).

| Concept | AUROC | 95% CI | AUPRC | chance (prevalence) | Verdict |
|---|---:|---|---:|---:|---|
| `crackle_score` | 0.5580 | [0.5329, 0.5831] | 0.3079 | 0.267 | **FAIL** |
| `wheeze_score` | 0.5340 | [0.5048, 0.5631] | 0.1903 | 0.174 | **FAIL** |

Movement from run 1 was negligible: crackle 0.5506 → 0.5580, wheeze 0.5729 → 0.5340. AUPRC sits
barely above prevalence in both cases (+0.04 and +0.016).

### The fixes worked mechanically, but traded one failure mode for another

| Diagnostic | Run 1 | Run 2 |
|---|---|---|
| `crackle_rate_hz`, Normal vs Crackle | 9.70 vs 10.19 /s | **2.54 vs 3.07 /s** |
| `wheeze_score` == 0 on wheeze-labelled cycles | 47% | **1%** |
| `wheeze_score` == 0 on normal cycles | 60% | **1%** |
| `rhonchi_score` == 0 | 60.4% | **13.9%** |
| `crackle_fine_ratio` == 0 | 79.8% | 84.9% |

* **Crackle saturation is genuinely fixed** — the local baseline, width cap and energy floor cut the
  rate from ~10/s to ~2.5–3/s. But discrimination barely improved (2.54 vs 3.07): the detector now
  fires *less*, not more *selectively*.
* **The wheeze high-pass over-corrected.** It went from silent on 47% of wheeze cycles to firing on
  99% of *everything*, normal cycles included. Same for rhonchi (60% silent → 86% firing). Fixing
  the under-detection introduced equivalent over-detection.
* **`crackle_fine_ratio` remains dead** (85% zero). It has never worked on real audio.

### Stop tuning here

The test split has now been read **twice**. A third parameter revision aimed at this gate would be
fitting it, and any subsequent "pass" would be uninterpretable. Per roadmap G2 *if not satisfied
(b)*: the honest branch is the **reliability-only paper**, which does not depend on a clean
bottleneck.

Note also that the 0.65 threshold may simply be unreachable against this reference standard by any
method — seven senior physicians score 47.77% on the same corpus (`Papers/HUMAN_BENCHMARKS.md`).
That does not rescue the extractors, but it does mean **"our concepts failed" and "these labels
cannot support a 0.65 target" are not distinguishable from this experiment alone**, and the write-up
must say so rather than claiming only the first.
