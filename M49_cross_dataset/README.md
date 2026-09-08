# M49 — external validation on SPRSound and HF_Lung_V1

**Created:** 2026-09-08 · **Read with:** `../M48_core_pipeline_ablation/README.md`,
`../Asif's/M45/` (the run that produced the checkpoint), `../Model_Training_Protocol.md` §1.

The paper reports one corpus. Every number in it — the corrected split, the official metric,
the ablation ladder, the seed band — describes how the pipeline behaves on ICBHI 2017 and says
nothing about whether it behaves that way anywhere else. This folder runs the best checkpoint,
**unchanged**, over two corpora it has never seen and scores it with the same official metric.

Inference only. No fine-tuning, no threshold tuning, no target label used to fit anything in
the headline block.

---

## Which checkpoint is "the best model"

Both notebooks default to **`Asif's/M45/best_M45_P3.pth`** — MobileNetV2 + SpecAugment **+
per-cycle peak amplitude normalisation**, official ICBHI **0.5764**. Scanning every committed
`results_M*.json` on the corrected split, that is the highest score in the repository, and the
cumulative ladder calls the same configuration `S5` / FULL PIPELINE.

**This is not the run the paper currently calls best.** §"Performance of the Best Model" names
`M22_v2` at 0.5602, because P3 arrived later as an ADD row in the M45 ablation. The paper's own
stated selection rule — highest challenge score on the corrected partition among runs with a
committed confusion matrix — actually selects P3, so the two disagree and **that is a decision
to make deliberately, not to inherit from this folder's default.**

Either way the notebooks work unchanged: set `CKPT_NAME` in the checkpoint cell to
`best_model.pth` to evaluate M22_v2 instead. The gate adapts on its own — it holds each
checkpoint to the score stored inside *it* (0.5764 or 0.5602), not to a hard-coded constant.

## The gate

Before any external number is computed, both notebooks re-score the checkpoint on the 2,636
ICBHI test cycles and **assert it reproduces the score stored inside the checkpoint**
(`best_M45_P3.pth` → 0.5764; `M22_v2/Results/best_model.pth` → 0.5602). A mismatch aborts.

That check is the reason the external numbers can be read at all. `m49_xval.py` copies the
`Net` class out of `m45_ablation.run_row` — it is a closure and cannot be imported — and a
copied forward pass is exactly where a channel-expansion or ImageNet-normalisation drift would
hide. `load_state_dict(strict=True)` catches structural drift; the ICBHI gate catches the rest.
An external score produced by an unverified forward path is not a result.

Everything else — mel parameters, the official metric, `LABEL_OF`, the corrected-split loader —
is **imported** from `../Asif's/M45/m45_ablation.py`, not re-implemented, so a cross-dataset
delta cannot turn out to be the gap between two scripts.

---

## The two corpora and what shifts

| | SPRSound (BioCAS 2022) | HF_Lung_V1 |
|---|---|---|
| Source | Shanghai Children's Medical Center | Taiwan (datathon + 18 patients) |
| Population | **pediatric**, 1 month – 18 years | adult |
| Device | Yunting Model II | Littmann 3200 · HF_Type-1 |
| Native rate | 8 kHz | **4 kHz** — Nyquist sits *at* the model's mel `fmax` |
| Annotation | events + record label, JSON | breath phases + adventitious spans, `_label.txt` |
| Unit used | `event_annotation` segment | inhalation **+** following exhalation |
| Bootstrap unit | **patient** (filename field 1) | **recording session** — no patient ID exists |
| Obtained by | `git clone --sparse` from GitHub | 7-Zip parts from GitLab |

HF_Lung_V1's 4 kHz sampling is a genuine confound, not a footnote: the model's mel filterbank
runs to 2,000 Hz and 4 kHz audio has its Nyquist exactly there, so the top of the band arrives
attenuated by the source recorder's anti-aliasing. It is recorded in
`dataset_info.native_sample_rates_hz` and must travel with the number.

### Taxonomy mapping

Neither corpus uses ICBHI's four labels. Every mapping is in `m49_xval.py` and is copied into
each results JSON under `dataset_info.label_mapping`, because a cross-dataset score cannot be
checked without it.

| ICBHI class | SPRSound event | SPRSound record | HF_Lung_V1 |
|---|---|---|---|
| Normal | `Normal` | `Normal` | no adventitious span overlaps the cycle |
| Crackle | `Fine Crackle`, `Coarse Crackle` | `DAS` | a `D` span overlaps by ≥ 50 ms |
| Wheeze | `Wheeze`, `Rhonchi`, `Stridor` | `CAS` | `Wheeze` / `Stridor` / `Rhonchi` overlaps |
| Both | `Wheeze+Crackle` | `CAS & DAS` | both bits set |
| *excluded* | — | `Poor Quality` | — |

Rhonchi and stridor go to Wheeze because both source taxonomies define them as **continuous
adventitious sounds**, which is the class ICBHI calls Wheeze; HF_Lung_V1's own paper pools
W/S/R into CAS for the same reason.

**An unmapped label string raises**, with the offending vocabulary listed. It is never bucketed
into Normal — that would silently inflate specificity, which is half the reported metric.

### What the real corpora actually contain

Both indexers were run against the real data during development, and the counts are recorded
here so a later run that disagrees is visibly a change rather than a surprise.

**SPRSound BioCAS2022** — 9,089 events over 2,683 recordings and 288 patients, matching the
published totals exactly. Vocabulary is exactly the seven documented event types and the five
record types, with no unmapped strings. 187 records carry `Poor Quality` and an empty
`event_annotation`, so they drop out of both levels. Native rate 8 kHz throughout.

| | Normal | Crackle | Wheeze | Both |
|---|---:|---:|---:|---:|
| event level (n=9,089) | 6,887 | 1,233 | 935 | 34 |
| record level (n=2,496) | 1,785 | 347 | 233 | 131 |

**HF_Lung_V1 test half** — 7,092 cycles over 1,956 recordings and 1,242 recording groups.
Vocabulary is exactly `I, E, D, Wheeze, Stridor, Rhonchi`. Native rate 4 kHz throughout.
Label times are written as `HH:MM:SS.mmm`, not as plain seconds.

Two things the real data changed, and both are now reported rather than assumed:

1. **Most rows are not whole cycles.** There are 6,872 inhalation labels against 2,748
   exhalation labels in the test half, so only **35.6 %** of rows are complete `I+E` cycles;
   4,344 are inhalation-only and 220 exhalation-only. Discarding the unpaired phases would
   throw away nearly two thirds of the corpus and would not do so at random, so they are kept
   and counted in `index_stats.phase_composition`.
2. **The window has to tile much harder here.** Median row length is 1.03 s against ICBHI's
   2.42 s, so the 8 s input is filled by tiling about **7.8×** rather than ICBHI's ~3.3×.
   Every extra repetition adds a seam — the artefact M44 measured and the P4 row tested — so
   `dataset_info.segment_stats` carries the factor next to the score.

---

## What each notebook reports

Per run, following `Model_Training_Protocol.md` §3–§4:

- `icbhi_score_official` with a **grouped** bootstrap CI, and
  `icbhi_score_official_ci95_unit` naming what was resampled — `patient` for SPRSound,
  `recording_group` for HF_Lung_V1. Calling the latter patient-level would overstate what the
  resampling controls for.
- Se, Sp, accuracy, macro P/R/F1, per-class metrics, raw **and** row-normalised confusion
  matrix, and the true-vs-predicted label histogram — the histogram is what shows a collapse
  onto Normal at a glance.
- `binary_detection` — the same predictions rescored as detect-only. Sp is unchanged by
  construction and Se rises, so the gap is the cost of sub-typing, directly comparable to the
  in-domain version in the paper's error table.
- `transfer` — the in-domain score, the external score, the delta, and 0.50 as the
  always-Normal floor an external number has to clear to mean anything.
- `feature_probe` *(optional, `RUN_PROBE`)* — a frozen-backbone logistic probe under grouped
  5-fold CV. It **uses target labels**, so it is not a zero-shot number and lives in its own
  block. It answers what one zero-shot score cannot: whether the representation carries nothing
  about this corpus, or the representation is fine and only the ICBHI-fitted decision boundary
  fails to transfer.

A slice with no Normal rows or no abnormal rows makes the official score undefined. It is
written as `null`, never as `NaN` (not valid JSON) and never as a fabricated interval.

---

## Running it

**On Kaggle** — GPU T4, **Internet ON**, and Add Data:
`vbookshelf/respiratory-sound-database` *plus* `Asif's/M45/best_M45_P3.pth` uploaded as a
private dataset. Then Run All. Each notebook fetches its own corpus, self-tests, passes the
ICBHI gate, evaluates, and zips `M49_results.zip` into the Output panel. Unzip it here and
commit.

| Notebook | Corpus | Rough runtime |
|---|---|---|
| `M49_sprsound_kaggle.ipynb` | SPRSound BioCAS2022, event **and** record level | ~10 min fetch + ~5 min |
| `M49_hflung_kaggle.ipynb` | HF_Lung_V1, I+E cycles | ~10 min fetch + ~10 min |

Set `SMOKE = 200` in the environment cell for a two-minute wiring check; the results JSON then
carries `SMOKE RUN … NOT reportable` in `meta.notes`.

**Locally** (no GPU needed for the self-test):

```
python m49_xval.py --selftest
python m49_xval.py --dataset sprsound --root .../BioCAS2022  --ckpt ../Asif's/M45/best_M45_P3.pth
python m49_xval.py --dataset hflung   --root .../HF_Lung_V1  --ckpt ../Asif's/M45/best_M45_P3.pth
```

Add `--icbhi_audio <dir>` to run the gate locally too; without it the in-domain reference is
the checkpoint's stored score rather than a reproduction, and the notebook says so.

---

## The self-test

`--selftest` builds a synthetic SPRSound (wav + JSON) and a synthetic HF_Lung_V1
(wav + `_label.txt`) and pushes both through the real index builders, the real spectrogram
function and the real network. It exists because the two expensive failure modes here are
**silent**: a taxonomy mapping that buckets a class into the wrong bin, and a cycle builder
that produces the right *number* of rows with the wrong times. Both yield a plausible score
that is simply not the quantity claimed.

It checks that every label string lands in the right ICBHI class, that an inhalation and its
exhalation are paired into one cycle with the right start and end, that an adventitious span
*outside* a cycle does not label it, that two slices of one session share a bootstrap group,
that an unknown token raises, that the metric being applied reproduces 0.5764 from the
committed P3 confusion matrix, and that a one-class slice returns a missing interval rather
than crashing.

Both notebooks run it before touching real data and abort if it fails.

---

## Files

| File | What it is |
|---|---|
| `m49_xval.py` | the evaluation module — adapters, gate, metrics, self-test |
| `gen_M49_kaggle.py` | the generator; **regenerate, never hand-edit the .ipynb** |
| `M49_sprsound_kaggle.ipynb` | SPRSound notebook |
| `M49_hflung_kaggle.ipynb` | HF_Lung_V1 notebook |
| `results_M49_*.json` · `preds_M49_*.npy` · `confusion_M49_*.png` | outputs, once run |

## What this folder does not do

No fine-tuning on either corpus, no domain adaptation, no new mechanism. The stop-lists in
`../DECISION_2026-08-16_PIVOT.md` §12 and `../RTK_requirements.md` §12 still hold. This measures
an existing model somewhere else, which is the one thing the paper claims nothing about.
