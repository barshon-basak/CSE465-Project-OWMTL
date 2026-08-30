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
`../owmtl_concept_engine/notebooks/01_concept_extraction_and_validation/`.

With the optional inputs:

```bash
python run_all.py --features /path/to/M2_features.npy \
                  --sprsound_dir /path/to/SPRSound/wavs \
                  --audio_dir /path/to/ICBHI/audio_and_txt_files
```

---

## The eight

All eight have run to completion. Each writes `results/<id>.json` and a figure.

| Script | Runs on | Needs | Status |
|---|---|---|---|
| `N1_fm_concept_probing.py` | GPU | ICBHI audio + `transformers`, `torch`, `librosa` | ✅ AST frozen + LoRA |
| `N2_concept_space_osr.py` | CPU | nothing (`--features` adds the embedding arm) | ✅ |
| `N3_prototypical_fewshot.py` | CPU | nothing (`--features` adds the M13 arm) | ✅ |
| `N4_honest_operating_point.py` | CPU | nothing | ✅ |
| `N5_leakage_audit.py` | CPU | `--features` for the full `I(y;f\|c)` | ✅ |
| `N6_physics_bottleneck.py` | CPU | `--features` for 3 of the 4 bottleneck modes | ✅ all 4 modes |
| `N7_clinician_intervention.py` | CPU | nothing; `clinician_corrections.csv` upgrades it | ✅ (simulated clinician) |
| `N8_pediatric_fragility.py` | CPU | `--sprsound_dir` for the actual comparison | ✅ both arms |
| `N9_clinician_reliability.py` | CPU | the **returned** clinician sheet | ✅ 116/132 answered |
| `N10_gate_vs_clinician.py` | CPU | N9's output | ✅ paired diff UNDECIDED |
| `N11_supervised_ceiling.py` | CPU | `embeddings/ast_frozen.npy` (optional arms degrade loudly) | ✅ **ceiling answered: labels ARE learnable** |

**N9 is what makes N7 real.** It scores the clinician listening study against ICBHI, measures
intra-rater reliability from the 12 repeated clips, and writes `clinician_corrections.csv` — the file
N7 already reads. Today `Asif's/ICBHI_Dataset_labeling/labels.csv` is the unfilled template (0 of 132
clips answered), so N9 reports BLOCKED and invents nothing. Once the sheet comes back:

```bash
python N9_clinician_reliability.py                                  # scores it, writes corrections
python N7_clinician_intervention.py --features M2_features.npy      # now says REAL, not SIMULATED
```

Every script takes `--help`.

### Reproducing the full set from scratch

```bash
# one-time inputs
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or /cpu without a GPU
pip install librosa transformers
python make_m2_features.py --audio_dir "C:/Users/Barshon/Desktop/ICBHI_final_database"
git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git ~/Desktop/SPRSound

# the foundation-model arm (GPU; ~11 min embed, ~60 min LoRA, ~15 min probe)
python N1_fm_concept_probing.py --stage embed --audio_dir "<ICBHI>"
python N1_fm_concept_probing.py --stage lora  --audio_dir "<ICBHI>" --batch_size 8
python N1_fm_concept_probing.py --stage probe --features M2_features.npy

# everything else
python run_all.py --features M2_features.npy \
                  --sprsound_dir "~/Desktop/SPRSound/BioCAS2022"
python run_all.py --summary-only     # rebuild RUN_SUMMARY.json without re-running
```

`--limit N` on `make_m2_features.py` and on N1's embed stage runs a one-minute wiring check that
writes nothing. Use it before committing to a long pass.

### Timings actually observed (RTX 4050 6 GB / this CPU)

| step | time |
|---|---|
| `make_m2_features.py` (6898 cycles) | 7m46s CPU |
| N1 embed, frozen AST | ~11 min GPU |
| N1 LoRA, 5 epochs | ~60 min (audio loading dominates, not the GPU) |
| N1 probe, 4 spaces × 14 concepts | ~15 min CPU |
| N5 with `--features`, 5 permutations | ~90 min CPU |
| N8 both arms, 6656 events | ~6 min CPU |
| N2 / N3 / N4 / N6 / N7 | < 2 min each |

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
(`../owmtl_concept_engine/owmtl/`) rather than reimplementing it. The concept
extractors, the leakage estimator, the bottleneck definition, the split loader and the score dumper
all come from there. On Kaggle/Colab it also finds `/kaggle/input/.../owmtl-package`, so the same
scripts run unchanged in a notebook.

---

## Generated inputs (not in git — regenerate with the commands above)

| File | Size | Produced by |
|---|---|---|
| `M2_features.npy` | 21.2 MB | `make_m2_features.py` from `../Asif's/M2/M2_best_model.pth` + ICBHI audio |
| `embeddings/ast_frozen.npy` | 21.2 MB | N1 `--stage embed` |
| `embeddings/ast_lora.npy` | 21.2 MB | N1 `--stage lora` |
| `embeddings/ast_lora_adapter.pt` | 1.2 MB | N1 `--stage lora` (294,912 LoRA params) |

`make_m2_features.py` reads the committed M2 checkpoint, **infers depth/base_width from the
checkpoint's own tensor shapes** rather than trusting a class default, and **refuses to write unless
its row order matches `concepts_all.npz`** — a misaligned feature matrix would corrupt N2–N7 at once
with nothing downstream able to detect it.

**Do not swap in `owmtl.m2_features.export_features` for this checkpoint.** Its `default_logmel`
applies a plain `power_to_db` with no normalisation, but M2 was trained on `power_to_db(ref=np.max)`
followed by per-sample min-max to [0, 1] with wrap-padding for short cycles. The embeddings would
load without error and be silently wrong.

If ICBHI's diagnosis table is missing from your audio mirror, both `make_m2_features.py` and N1
derive it from `concepts_all.npz` automatically — the label plays no part in computing an embedding.

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
