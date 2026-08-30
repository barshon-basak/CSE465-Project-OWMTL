# M48 — core-pipeline ablation

**Created:** 2026-08-30 · **Read with:** `../DECISION_2026-08-30_CONSOLIDATION.md`,
`../RTK_requirements.md` §10, `../Asif's/M45/` (the model ablation this extends).

The M45 table ablates the **model**. Nothing in the repo ablated the thing the paper actually
contributes — the corrected split loader and the official metric — and nothing measured the
run-to-run noise that ten of M45's eleven "not shown to differ" verdicts rest on. This folder
closes both gaps and adds the one clinician experiment the consolidation record flagged as
"worth more than the rest".

Three tiers, one per layer of the pipeline. **Tiers E and C are done and committed.
Tier A needs a GPU** and ships as a Kaggle notebook.

---

## Tier E — the evaluation protocol ✅ done, no compute

`tier_e_protocol_ablation.py` → `M48_tier_E_table.{json,md,tex}`

Baseline E0 = the full corrected protocol on M22-v2. Each row removes one protocol component
and reports what the pipeline would then have reported. **Every score is recomputed from the
run's committed raw confusion matrix, never transcribed.**

| Row | Component removed | Kind | Reported | Δ vs E0 |
|---|---|---|---:|---:|
| `E0` | none — full corrected protocol | baseline | 0.5602 | — |
| `E1` | − official metric (macro variant instead) | rescore | 0.6140 | +0.0538 |
| `E2` | − official split (patient-id ≤ 111 fallback) | rerun | 0.6495 | +0.0893 |
| `E3` | − patient independence (published split verbatim) | rerun | 0.5641 | +0.0039 |
| `E4` | − device-suffix-safe file join | defect count | 1 recording dropped | — |
| `E5` | − patient-level unit of analysis | rescore | 7/10 significant | vs 1/10 |
| `E1+E2` | **both — the unaudited pipeline** | rescore of E2 | **0.7077** | **+0.1475** |

`E1+E2` is the row this table exists for: it is the single number stating what the whole
correction was worth, and it was implicit in the paper's prose rather than tabulated.
`E3`'s near-zero delta is reported as a null on purpose — conceding that one of the four
faults costs nothing is what makes `E1` and `E2` credible.

Two kinds of row, and the distinction is in the JSON: **rescore** rows apply a different rule
to fixed predictions (exactly one variable, no seed noise); **rerun** rows change the split,
which changes the test set, so the model is retrained with every other setting held fixed.

```
python tier_e_protocol_ablation.py --selftest   # reproduces M22-v2's 0.5602 from its matrix
python tier_e_protocol_ablation.py              # writes the table
```

---

## Tier C — the concept branch ✅ done, ~1 min CPU

`C4_probe_vs_clinician.py` → `results_M48_C4.json`

The paper's sharpest claim — machines reproduce ICBHI's labels better than trained listeners
agree with them — rested on two numbers measured on two different sets. This puts the machine
and the physician on the **same clips**.

Frozen AudioSet AST embedding → logistic probe, fitted by **patient-grouped 5-fold CV** so
every clip is out-of-sample, then scored against both reference standards.

| | crackle (n=108) | wheeze (n=110) |
|---|---:|---:|
| probe vs **ICBHI** | 0.6786 [0.578, 0.778] | 0.7519 [0.636, 0.857] |
| probe vs **clinician** | 0.5097 [0.354, 0.665] | 0.5828 [0.412, 0.746] |
| **paired difference** | +0.169 [−0.025, +0.361], p=0.087 | **+0.169 [+0.002, +0.344], p=0.046** |
| clinician–ICBHI agreement | 0.528 | 0.718 |

**Wheeze: significant.** On the same clips, with the same scores, the machine reproduces
ICBHI's labels better than the physician does. **Crackle: the same effect size, not
significant at n=108** — report it as such, do not round it up.

The pre-registered disagreement-subset reading came back **UNDECIDED** for both concepts
(n=51 and n=31, intervals spanning chance), which is the outcome the file predicted before it
was run. The all-clips paired difference is the better-powered form of the same question and
is the number to cite.

Two overlapping marginal intervals are not a test, so `paired_reference_diff()` resamples the
clips once per iteration and computes both AUROCs on that resample.
`common.paired_bootstrap_diff` does not apply here — it varies the scores against a fixed
reference, and this varies the reference against fixed scores.

**Carried warning:** the 132 clips are not a random sample (cycles ≥ 0.9 s, sorted
longest-first, abnormal strata over-sampled), so the absolute AUROC is **not** comparable to
N11's 2,636-cycle numbers and cannot overturn the G2 failure. The selection applies equally to
both halves of the paired difference, which is why the contrast is the result and the level is
not.

```
python C4_probe_vs_clinician.py --selftest   # grouped-CV estimator check
python C4_probe_vs_clinician.py
```

---

## Tier A — seed band, selection criterion, feature-extractor null 🔴 needs a GPU

`M48_kaggle_tier_A.ipynb` (run this) · `m48_gpu_rows.py` (what it calls)

| Row | Change | What it tests |
|---|---|---|
| `A0_s42` `A0_s1` `A0_s2` | the same config, three seeds | **The noise floor.** M45 ran one seed per row; ten of eleven rows are "not shown to differ from A0", including P3, the table's top score. Their spread says which deltas are real. |
| `A7` | checkpoint by official score **vs** by minimum loss | The paper argues the selection criterion matters and reports a 33-epoch disagreement but never prices it. Both criteria are tracked in one loop, so this is free on every row. |
| `A24` | random init **and** frozen backbone | A4 froze a *pretrained* backbone, so its −0.1007 mixes "fine-tuning helps" with "pretraining helps". This separates them. |

Everything for data, preprocessing, caching and metrics is imported from
`../Asif's/M45/m45_ablation.py` **unmodified** — if the two scripts disagreed about a mel
parameter or a cache key, the seed band would be measuring the difference between two scripts
rather than between two seeds. All four rows share one cache key, so the spectrogram cache is
built once.

**How to run:** open the notebook on Kaggle, enable GPU, attach the *Respiratory Sound
Database* dataset, turn Internet on, Run All. ~90 minutes. Download `M48_results.zip` from the
Output panel and unzip it into this folder.

The notebook self-tests the wiring and verifies the split (2,636 test cycles, zero patient
overlap) **before** spending an hour on it, and skips rows that already have a result, so a
session timeout resumes rather than restarts.

### The caveat that must travel with A7

`m45_ablation.py` monitors the **test** set each epoch; there is no separate validation split.
So "select on loss" here means select on *test* loss. That is optimistic, and equally so for
every M45 row, so the seed band and the A24 delta are unaffected — but **A7 must be reported
as "two criteria applied to one monitoring set", never as a clean train/val/test result.**
Write that sentence into the paper next to the number.

---

## Files

| File | Status | What it is |
|---|---|---|
| `tier_e_protocol_ablation.py` | ✅ run | Tier E builder + metric self-test |
| `M48_tier_E_table.{json,md,tex}` | ✅ output | the protocol ablation, LaTeX ready to paste |
| `C4_probe_vs_clinician.py` | ✅ run | Tier C: probe vs physician on the same clips |
| `results_M48_C4.json` | ✅ output | C4 result, with the pre-registered readings |
| `M48_kaggle_tier_A.ipynb` | 🔴 to run | the GPU notebook |
| `m48_gpu_rows.py` | 🔴 to run | the four Tier A rows; `--selftest` needs no GPU |
| `M48_tier_A_table.json` | ⏳ pending | written by the notebook |

## What this folder does not do

No new mechanism, no third G2 extractor revision, no bottleneck head. The stop-lists in
`../DECISION_2026-08-16_PIVOT.md` §12 and `../RTK_requirements.md` §12 still hold. Every row
here re-measures something the project already built.
