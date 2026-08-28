# Novelty Experiment

Eight scripts, one per novelty direction Dr. Khan suggested. Each one runs an experiment, writes a
JSON result, and states in its own docstring what already existed in the repo and what was missing.

**Read `NOVELTY_STATUS.md` first** — it is the tracker: status, results and what is still blocked.

---

## Quick start

```bash
cd "Novelty Experiment"

python test_nx.py          # 9 self-checks on the statistics — run this before trusting a number
python run_all.py          # everything that runs CPU-only, no audio, no GPU
```

That produces `results/*.json`, `figures/*.png` and `results/RUN_SUMMARY.json`. Nothing needs to be
downloaded: all the CPU experiments read `concepts_all.npz`, which is already in the repo at
`../owmtl_concept_engine/owmtl_concept_engine/notebooks/01_concept_extraction_and_validation/`.

With the optional inputs:

```bash
python run_all.py --features /path/to/M2_features.npy \
                  --sprsound_dir /path/to/SPRSound/wavs \
                  --audio_dir /path/to/ICBHI/audio_and_txt_files
```

---

## The eight

| Script | Runs on | Needs |
|---|---|---|
| `N1_fm_concept_probing.py` | GPU | ICBHI audio + `transformers`, `torch`, `librosa` |
| `N2_concept_space_osr.py` | CPU | nothing (`--features` adds the embedding comparison) |
| `N3_prototypical_fewshot.py` | CPU | nothing (`--features` adds the M13 arm) |
| `N4_honest_operating_point.py` | CPU | nothing |
| `N5_leakage_audit.py` | CPU | `--features` for the full `I(y;f|c)` |
| `N6_physics_bottleneck.py` | CPU | `--features` for 3 of the 4 bottleneck modes |
| `N7_clinician_intervention.py` | CPU | nothing; `clinician_corrections.csv` upgrades it |
| `N8_pediatric_fragility.py` | CPU | `--sprsound_dir` for the actual comparison |

Every script takes `--help`.

---

## Layout

```
common.py        shared plumbing: data loading, patient aggregation, bootstrap CIs,
                 paired bootstrap, permutation nulls, risk-coverage, ECE, result writer
N1..N8_*.py      one experiment each
run_all.py       runs them, collects statuses, writes RUN_SUMMARY.json
test_nx.py       assert-based self-check of every non-trivial statistic
results/         JSON output, one file per experiment (+ raw score dumps for paired tests)
figures/         one PNG per experiment that produces a figure
NOVELTY_STATUS.md   the tracker — status, numbers, what is blocked, run log
```

`common.py` locates and reuses the existing `owmtl` package
(`../owmtl_concept_engine/owmtl_concept_engine/owmtl/`) rather than reimplementing it. The concept
extractors, the leakage estimator, the bottleneck definition, the split loader and the score dumper
all come from there. On Kaggle/Colab it also finds `/kaggle/input/.../owmtl-package`, so the same
scripts run unchanged in a notebook.

---

## The two files that unblock most of it

| File | Unblocks | How |
|---|---|---|
| `M2_features.npy` — 768-d frozen M2 embeddings, 6898 rows, aligned to the cycle index | N5 full, N6 all modes, N2/N3/N4 embedding arms | already exists as a Kaggle dataset; regenerate with `owmtl.m2_features.export_features(records, audio_dir, load_m2)`. Drop it in this folder and every script finds it automatically. |
| SPRSound `.wav` directory | N8 | `--sprsound_dir` (also reads `$SPRSOUND_DIR`) |

A missing input never produces a wrong number. Scripts either skip the affected arm with a printed
reason, or write `results/<id>_BLOCKED.json` saying exactly what they need.

---

## House rules these scripts enforce

These are not style preferences — they are the rules the 2026-08-16 audit was written to enforce, now
implemented in code rather than left to discipline:

- **Patient is the unit.** Cycle-level rows inflate n by ~60× and make every CI wrong.
  `common.patient_frame` aggregates first.
- **A CI that spans chance is not a result.** `common.honest_verdict` returns the literal string
  *"NOT SHOWN TO BEAT CHANCE — do not write 'beats'"*. At n = 19 unknown patients this fires on
  essentially everything, which is the point.
- **Comparisons are paired.** Two overlapping CIs are not a test.
  `common.paired_bootstrap_diff` resamples both scores together.
- **Controls are mandatory.** Shuffled concepts (N6, N7), random projections (N1), permutation nulls
  (N1, N5). A positive result without its control is not reported.
- **Thresholds are chosen on calibration data, never on test** (N4).
- **Undefined metrics are refused, not approximated.** N3 will not emit an "ICBHI score" for the
  3-class disease task, because (Se+Sp)/2 is defined over the 4-class sound-event task.
- **A claim bigger than the data is refused.** N3 skips k = 20 because URTI has only 10 train
  patients.

---

## Adding a ninth experiment

Copy the shape of `N2`: import `common as C`, load with `C.load_concepts()`, aggregate with
`C.patient_frame()`, wrap every headline number in `C.bootstrap_ci`, pass the CI through
`C.honest_verdict`, and finish with `C.save_result(EXP_ID, doc)`. Add a row to `EXPERIMENTS` in
`run_all.py` and a section to `NOVELTY_STATUS.md`. If it contains a branch or a loop worth trusting,
add one assert to `test_nx.py`.
