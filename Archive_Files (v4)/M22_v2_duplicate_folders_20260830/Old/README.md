# M22_v2 — re-running the SpecAugment result on the corrected split

## Why

`Model_Training_Protocol.md` §1 records that M2 / M3 / M12 / M22 were labelled
`patient_independent_official_60_40` but actually executed a silent fallback
(`patient_id <= 111 -> test`), giving **11 test patients and 7.1% of cycles**. The JSON labels
were corrected in place afterwards — **but M22 was never re-run**.

So the project's most-cited result, *"SpecAugment gives +0.036 official ICBHI"* (0.6495 vs
0.6135), is measured on 11 patients and 492 cycles. `RTK_requirements.md` §6 still names M22 as
"the current best model" on the strength of that number, even though its own definition requires
the corrected split.

This notebook re-measures it.

## What is held constant

Everything except the split. Architecture (`mobilenet_v2`, ImageNet-pretrained), Adam @ lr 5e-4,
cosine schedule, weight decay 1e-4, grad clip 5.0, batch 16, 40 epochs, AMP, dropout 0.3,
inverse-frequency class weights normalised to mean 1.0, seed 42, the shared §2 preprocessing, and
all five SpecAugment parameters (2 freq masks ≤24 bins, 2 time masks ≤80 frames, fill 0.0) are
copied verbatim from the original M22 run.

| | original M22 | this run |
|---|---|---|
| split | `..._patient_id_fallback` | `official_60_40_patient_independent_corrected` |
| recordings | — | 551 train / 369 test (59.9 / 40.1) |
| test patients | 11 | **47** |
| overlap policy | `drop_from_train` | reassign patients 156, 218 to train (protocol §1) |

## How to run it — two passes

The notebook has one switch at the top of Cell 2:

```python
VARIANT = "augmented"   # -> M22_v2, SpecAugment ON
VARIANT = "clean"       # -> M3_v2,  SpecAugment OFF
```

1. **Attach two datasets on Kaggle**
   - `vbookshelf/respiratory-sound-database` (the ICBHI audio)
   - a dataset containing `ICBHI_challenge_train_test.txt` — upload the committed copy from
     `Asif's/ICBHI_challenge_train_test.txt`
2. **Pass 1:** set `VARIANT = "augmented"`, then **Save Version → Save & Run All (Commit)**.
   Interactive sessions time out; committed runs do not (§11.B).
3. **Pass 2:** set `VARIANT = "clean"`, run again.
4. **Paired comparison:** attach pass 1's output as a dataset to pass 2 (or vice versa) and run
   Cell 15. It finds both `preds_*.csv` dumps and computes McNemar + the bootstrap Δ.

Roughly 40 epochs × ~2–4 min on a T4 per pass, plus a one-off spectrogram cache build.

## Why both passes are required

Essential #10 demands a CI **and** a paired test on every headline comparison. The claim under
test is a *difference* — SpecAugment vs no SpecAugment. Running only the augmented pass gives an
absolute number with nothing to compare it against, and the old clean baseline (M3) was trained on
a different partition, so it is not a valid control.

Two passes on the identical split, identical seed, identical everything-else is what makes the Δ
mean something.

## The split file is mandatory

Cell 3 **raises** if `ICBHI_challenge_train_test.txt` is not found. It does not fall back to a
patient-id rule. That silent fallback is the exact bug this notebook exists to correct, and a
notebook that can quietly produce the wrong split is worse than one that stops.

Cell 4 then verifies the file really is the official split (920 recordings, 539/381, 126 patients,
patients 156 and 218 straddling it) and asserts the corrected counts (551/369, zero patient
overlap) before any training starts.

## Outputs

```
results_M22_v2.json / results_M3_v2.json     §4 schema, incl. confusion_matrix_raw,
                                             icbhi_score_official + patient-bootstrap CI
preds_M22_v2.csv    / preds_M3_v2.csv        per-cycle predictions (enables the paired test)
augmentation_effect_corrected_split.json     Δ, CI, McNemar, verdict
best_model.pth                               best checkpoint by icbhi_score_official
loss_curve.png accuracy_curve.png f1_curve.png confusion_matrix.png training_curves.png
```

## What to expect

**The absolute score will very likely fall.** M2 and M3 drop from ~0.61 to 0.4139 / 0.4490 when
moved onto a real patient-disjoint partition. A similar drop here is the expected result, not a
failure — it is precisely the measurement the paper is about.

**The number that matters is Δ, not the absolute.** If the Δ interval spans zero, the +0.036 claim
does not survive the corrected split and should be withdrawn — the same way the single-seed
bottleneck result was. Do not compare this run's absolute score against 0.6495: different split,
different test set, 11 patients vs 47.

## After the run

- Update `RTK_requirements.md` §6 — the "best model" designation currently rests on the number
  this run replaces.
- Re-run `python "Asif's/audit/audit_project.py"` and confirm both IDs are clean.
- Update the paper's augmentation subsection with the Δ and its CI.
