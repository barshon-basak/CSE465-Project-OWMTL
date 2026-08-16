# M39 — Physics-Derived Acoustic Concept Extraction & Validation (Gate G2)

**Owner:** Asif · **Gate:** G2 · **Requires:** ICBHI audio + `Asif's/ICBHI_challenge_train_test.txt`
**Status:** notebook ready, tested 43/43, **not yet run on real data**
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

## The gate

A concept passes if its **95% CI excludes chance** on the test split — materially better than a coin
flip, not merely above 0.5 by a point estimate.

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
