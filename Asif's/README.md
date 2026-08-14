# Member A — shared representation & AST backbone

**Asif Mahbub** · CSE465 Group 5 · OWMTL respiratory diagnosis

Owns the shared data contract (preprocessing, splits, metric definitions) and the
AST backbone. Barshon owns the CNN / MobileNet / ResNet comparison runs (M2, M3)
and his OpenMax baseline (M6); the backbone pillar is split, not transferred.

| Model | What | Status |
|---|---|---|
| **M4** | AST, clean | notebook ready |
| **M23** | AST + SpecAugment | same notebook, one flag |
| **M12** | Backbone selection *by open-world separability* | harness ready, needs M2/M3 |

---

## What's here

```
owmtl/                     the shared package — every member imports this
  icbhi.py                 metadata, cycle index, splits         (stdlib only)
  features.py              the 7 preprocessing stages + SpecAugment
  separability.py          open-world separability (the M12 criterion)
  reporting.py             metrics, results JSON, required plots
scripts/
  build_index.py           generates the two committed artifacts
  shortcut_probe.py        Phase-0 gate: is diagnosis predictable from metadata?
tests/
  test_icbhi.py            17 tests, no dependencies, runs in 0.3 s
M4-AST-soundevent.ipynb    the Colab notebook (M4 and M23)
splits/                    cycles_index_v1.csv + split_v1.json  (generate, then commit)
```

Run the tests any time: `python tests/test_icbhi.py`

---

## For everyone else in the group

Once `splits/` is committed, this replaces every hand-written ICBHI loader:

```python
import sys; sys.path.insert(0, "path/to/Asif's")
from owmtl import icbhi

split = icbhi.load_split("splits/split_v1.json")
index = icbhi.read_cycle_index("splits/cycles_index_v1.csv")

# Member A — sound-event partitions
train = icbhi.select_cycles(index, split, partition="train")
test  = icbhi.select_cycles(index, split, partition="test")

# Members B / C — disease-task patient groups
known_train  = icbhi.disease_patients(split, "known_train")     # 104 known, minus calib/test
unknown_eval = icbhi.disease_patients(split, "unknown_eval")    # the 19, evaluation only

# Member D — the never-trained calibration partition
calib = icbhi.disease_patients(split, "known_calib")
```

Preprocessing, so all four members' spectrograms are comparable:

```python
from owmtl import features
cfg  = features.AudioConfig()                              # 16 kHz · 8 s · 128 mel · 50–2000 Hz
wave = features.load_cycle(wav_path, row["start"], row["end"], cfg)   # stages 1–5
spec = features.log_mel(wave, cfg)                                    # stage 6
```

Metrics and the results JSON:

```python
from owmtl import reporting
m = reporting.compute_metrics(y_true, y_pred, icbhi.SOUND_EVENT_CLASSES)
out = reporting.finalise_run(payload, icbhi.SOUND_EVENT_CLASSES, "results/")
print(out["problems"])   # empty == protocol §10 checklist satisfied
```

---

## Three decisions baked into the split, and why

### 1. The 19 unseen-disease patients never appear in any training set

Under the official ICBHI split, most of the Bronchiectasis / Pneumonia /
Bronchiolitis patients land in the training half. The shared backbone would
therefore be trained on audio from the very patients later presented to it as
"unknown", and Member B's open-world result would be partly memorisation rather
than novelty detection.

`split_v1.json` marks all 19 as `held_out` for **every** task and **every**
member. `icbhi.validate()` fails loudly if one leaks back in, and
`test_validate_catches_an_injected_leak` proves that check actually fires.

Cost: the sound-event training set loses ~1,000 cycles. Worth it — the
alternative is a headline result a reviewer can dismiss in one sentence.

### 2. Two split schemes, reported separately

- `owmtl` — leak-free. Required for anything feeding the open-world claim.
- `official` — the ICBHI 60/40 challenge split, for comparability with published
  numbers.

The vbookshelf Kaggle mirror **omits** `ICBHI_challenge_train_test.txt`, so
`official` is unavailable until someone sources that file. Until then no number
we produce can be placed beside a published ICBHI result. Worth ten minutes of
someone's time to find it.

### 3. A calibration partition exists from day one

15% of known-disease training patients are carved into `known_calib` and never
trained on. Member D needs a clean partition for temperature scaling and
conformal calibration; carving it retroactively after training has started is
impossible without redoing everything.

---

## Corrections to the existing planning docs

These are load-bearing, not nitpicks.

**The M1 train/test split is wrong.** `Barshon's/M1-provisional-cnn-backbone.ipynb`
falls back to `split = "test" if pid <= 111 else "train"`. ICBHI patient IDs run
101–226, so that puts 11 patients in test — the notebook output confirms
**6,406 train / 492 test**, a 93/7 split, not the 60/40 the comment claims. The
test set holds 38 Wheeze and 35 Both cycles in total. Every metric from that run
is noise. Re-run against `split_v1.json`.

**The ICBHI Score formula in `Model_Training_Protocol.md` §3 is not the challenge
metric.** The protocol defines it as `(macro sensitivity + macro specificity) / 2`.
The ICBHI 2017 challenge metric pools the abnormal classes:

```
Sp    = correct Normal / all Normal
Se    = correct (Crackle + Wheeze + Both) / all abnormal
Score = (Sp + Se) / 2
```

A number computed the protocol's way cannot be compared to any published ICBHI
result. `reporting.compute_metrics` returns both — `icbhi_score` (official,
used for model selection) and `icbhi_score_macro` (the protocol's, kept so
existing runs stay readable). M1 selected its best epoch on the macro variant.

**`n_fft` disagrees.** Protocol §2 says 1024; M1 ran 512. That is a real
difference in spectral resolution and the two are not directly comparable. This
package uses 1024, per the protocol.

**AST cannot use the shared librosa log-mel front-end.** Its pretrained weights
were learned on Kaldi filterbank features with AudioSet normalisation statistics
(−4.268 / 4.569). Feeding it librosa log-mel would handicap it and corrupt the
backbone comparison. `features.ast_fbank` matches the pretraining front-end
exactly; runs record `frontend: "kaldi_fbank"` so the deviation is visible in the
ablation table rather than hidden.

---

## Phase-0 gate — run this before M13

```bash
python scripts/shortcut_probe.py --index splits/cycles_index_v1.csv \
                                 --split splits/split_v1.json
```

COPD is 64 of 126 patients and COPD recordings are not evenly distributed across
the four stethoscopes. If diagnosis is predictable from device and chest location
alone, then a disease head can reach high accuracy without learning pathology —
and cross-task disagreement would be detecting *device mismatch*, not unseen
disease. That would make the project's headline claim an artifact.

The probe uses no audio and runs in seconds. Its output belongs in the paper
either way: a null result pre-empts the most predictable reviewer objection to
any ICBHI diagnosis paper, and a positive result changes what Member B builds.

---

## The M12 contribution

Selecting a backbone by sound-event F1 is a bake-off on a benchmark with 135
publications on it. It also assumes, without checking, that closed-set accuracy
predicts open-world usefulness.

`separability.py` measures the thing we actually need from a shared encoder:
freeze a backbone trained on sound-event labels only, embed the known-disease and
the 19 unseen-disease patients, and measure how separable they are. No disease
head is trained; nothing is fitted on the unknown group.

The claim:

> The representation that maximises closed-set sound-event F1 is not the one that
> maximises open-world separability, so accuracy is the wrong backbone-selection
> criterion for OWMTL.

`separability.compare_backbones()` produces the table that supports or refutes it.
A null result is still a finding — nobody has checked whether the two criteria
coincide on ICBHI. Same three backbones, same compute, but the selection criterion
becomes ours instead of inherited.

To include M2 and M3 in the table, Barshon appends Section 9 of the notebook to
his runs; it needs only a `forward(..., return_embedding=True)` and about twenty
lines.

---

## Setup

```bash
# 1. Generate the shared artifacts (once, then commit both files)
python scripts/build_index.py --data-root <ICBHI root> --out splits/

# 2. Sanity-check the corpus before anyone trains on it
python scripts/shortcut_probe.py

# 3. Verify the contract still holds
python tests/test_icbhi.py
```

Then open `M4-AST-soundevent.ipynb` in Colab, run it end to end with
`USE_SPECAUGMENT = False` (→ M4), flip the flag, and run it again (→ M23).

`icbhi.py` needs nothing but the standard library. `features.py` needs
librosa + scipy, `separability.py` and `reporting.py` need scikit-learn,
matplotlib and scipy, and the AST front-end additionally needs torchaudio.
