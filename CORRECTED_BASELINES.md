# Corrected baseline table — M2 / M3 (official ICBHI metric, corrected split)

**Generated 2026-08-29 directly from the committed `results_M*.json` files.** No notebook
re-run is required to reproduce this table — the numbers below were always correct in the
results files; only the notebooks' on-screen summary tables were displaying the wrong field.

Source files:
- `M2` → `Asif's/M2/m2_v4/results_M2.json`
- `M3` → `Asif's/M3/29 aug run/results_M3.json`

## The table

| Model | **Official ICBHI** | Se | Sp | Accuracy | Macro-F1 | Params | Size MB | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2 | **0.4720** | 0.2005 | 0.7435 | 0.5116 | 0.2884 | 421,732 | 1.62 | 19 |
| M3 | **0.5132** | 0.1980 | 0.8284 | 0.5591 | 0.3500 | 929,316 | 3.67 | 17 |

**M3 − M2 = +0.0412** on the official metric (M3 ahead).

Both runs share the identical corrected protocol, so this delta is a valid comparison:
- split: `official_icbhi_60_40_patient_disjoint` — 77 train / 49 test patients, official file verified (920 recs, 539/381)
- checkpoint selection on a patient-grouped validation slice carved from TRAIN only (12 val patients)
- `test_touched_times: 1` — the official test set was read exactly once, at final evaluation

## The inflated variant, for the paper's §3

| Model | Official | Macro (non-standard) | Inflation |
|---|---:|---:|---:|
| M2 | 0.4720 | 0.5480 | +0.0760 |
| M3 | 0.5132 | 0.5639 | +0.0507 |

Never report the macro column. It is the variant this project used before the metric audit;
it is not the ICBHI 2017 challenge metric and is not comparable to published work.

## Why both scores sit below 0.50–0.52

Published ICBHI SOTA on the official 60/40 split is ~0.60–0.65. Both models land under that,
and M2 lands below the 0.50 a trivial always-one-class model scores. The per-class recalls
show why — both models collapse toward Normal:

| Model | Normal | Crackle | Wheeze | Both |
|---|---:|---:|---:|---:|
| M2 | 0.744 | 0.074 | 0.486 | 0.007 |
| M3 | 0.828 | 0.217 | 0.184 | 0.147 |

This is the corrected picture, not a broken run: high Sp, very low Se. It is consistent with
the reliability-ceiling argument and should be reported plainly rather than softened.
