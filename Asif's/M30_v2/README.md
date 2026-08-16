# M30 v2 — Gated Feature-Fusion Ensemble (official ICBHI split)

**Owner:** Asif · **Chunk:** E (selected novelty item #3) · **Requires:** M2 + M3 checkpoints
**Status:** notebook ready, tested 53/53, **not yet run on real data**

Re-run of `Barshon's/M30` on the **official ICBHI 60/40 split**, reporting **both** ICBHI metrics and
committing the confusion matrix. Architecture, `fusion_dim`, dropout, learning rate and epoch budget
are unchanged from the original — only the split, the metric reporting, and the checkpoint-loading
safety differ.

## Why the original was withdrawn

`Barshon's/M30/results_M30.json` reports ICBHI 0.8213 and "+9.86% over M2". Three independent
defects, any one disqualifying:

**1. Wrong split.** The original's `build_icbhi_splits()` has this docstring:

```python
def build_icbhi_splits(data_root, cfg):
    """Load all ICBHI cycles and perform official patient-independent split."""
    ...
    np.random.shuffle(all_pids)
    n_train = int(len(all_pids) * 0.70)
```

It says *official split*; it does a **random 70/30 shuffle**. M2 and M3 — the backbones it fuses and
claims to beat — both use the official 60/40. The "+9.86%" compares across split boundaries. This is
the same class of defect as trap #7 in `Asif's/CLAUDE.md`: text that contradicts its own code.

**2. Wrong metric.** 0.8213 is the legacy macro variant, not the ICBHI 2017 challenge score.
See `Asif's/audit/ICBHI_SCORE_AUDIT.md`.

**3. Unverifiable.** No `confusion_matrix_raw` committed, so no score can be recomputed from the
file by anyone — including a reviewer who asks.

## Two things v2 adds

### Checkpoint loading is asserted, not hoped for

The original:

```python
try:
    ckpt2 = smart_load_checkpoint(CFG['m2_ckpt_path'], DEVICE)
    m2_backbone.load_state_dict(sd2, strict=False)
    print('✅ Loaded M2 Backbone weights')
except Exception as e:
    print(f'⚠️ M2 load note: {e}')       # <-- then continues anyway
```

If a checkpoint path were wrong, this prints a warning and trains a "fusion of M2 and M3" over **two
randomly-initialised backbones**. `strict=False` also silently tolerates a state_dict whose keys
don't match at all. That is the exact trap `Asif's/CLAUDE.md` records for M15.

v2's `load_backbone_or_die()` requires three things or raises:

1. the file loads;
2. **no backbone parameter is missing** from the state_dict;
3. the weights **measurably changed** — a fingerprint before/after must differ.

Check 3 catches the nastiest case: a load that "succeeds" under `strict=False` while changing
nothing. Both failure modes are covered by the test suite.

### It runs the test that admits this item

`Novelty Search.md` (moved -> `../../Archive_Files (v2)/Novelty Search v2.md`) §4.0 selects the fusion ensemble **"only if it measurably beats both backbones
alone; drop it silently if it doesn't."** *That test has never been run* — the original compared
against M2's number from a different split.

v2 evaluates **M2-alone and M3-alone on the same test cycles** as the fusion, before training, and
prints a verdict:

- **PASSES** — beats both by ≥0.01
- **MARGINAL** — beats both but within noise; report as "no meaningful difference"
- **FAILS** — per §4.0, drop the item and swap in a deferred one rather than adding alongside

The verdict is written to `best_metrics.admission_test` in the results JSON, so it can't be quietly
ignored. **A FAILS verdict is a legitimate result**, not a problem with the run: it would mean
heterogeneous feature fusion doesn't help here, which the original's cross-split comparison
concealed.

## Running it

Colab/Kaggle. Only the fusion head trains (≈2.1M of 8.0M params); both backbones are frozen.

**Runtime is dominated by a single feature-extraction pass, not by training.** Because the backbones
are frozen, a given cycle's 2048-d embedding is *identical at every epoch* — so embeddings are
extracted once (Cell 7) and the head trains on the cached vectors. Expect a few minutes of
extraction, then all 30 epochs in seconds.

> **Fixed 2026-08-14.** The first version recomputed librosa log-mels from disk every epoch: ~3.4
> min/epoch × 30 ≈ 100 minutes for bitwise-identical inputs. It also spammed Colab with thousands of
> harmless `AssertionError: can only test a child process` tracebacks — a known PyTorch/Colab
> interaction where a `DataLoader`'s worker-pool iterator is garbage-collected in the wrong process.
> Caching the embeddings removed both: the head trains on a `TensorDataset` with `num_workers=0`, so
> there are no worker processes to churn. Mathematically equivalent (frozen backbones in eval mode,
> no augmentation), just ~30× less work.

Needs three inputs:
1. ICBHI audio — cell 0b downloads it (paste your Kaggle key first)
2. `Asif's/M2/best_model.pth` → upload as `/content/M2_best_model.pth`
3. `Asif's/M3/best_model.pth` → upload as `/content/M3_best_model.pth`

The notebook fails fast with an explicit message if any is missing — it deliberately refuses to run
on randomly-initialised backbones, since that's the failure mode it exists to rule out.

## The split, and why M2/M3's numbers are not anchors

> **Corrected 2026-08-15.** `Asif's/ICBHI_challenge_train_test.txt` is now committed, and analysing
> it invalidated the calibration this section used to give.

The official file assigns **920 recordings — 539 train / 381 test (58.6 / 41.4)** across **126
patients**. The notebook asserts all four numbers on load and refuses to run if any differs.

**M2, M3 and M12 were never on this split.** Their `results_M2/M3/M12.json` all record
`split_method: "patient_independent_official_60_40"` while reporting `train_patients: 115,
test_patients: 11, test_samples: 492` — 115 + 11 = 126 patients and 6406 + 492 = 6898 cycles, i.e.
*every* patient and *every* cycle, split 91/9. That is the documented `pid <= 111` fallback: ICBHI
patient IDs run 101–226, so it selects exactly 11 patients. So **0.6138 / 0.5895 / 0.6495 are
11-patient numbers and are not comparable to published ICBHI work.** Cell 7 still prints them, but
labelled superseded, and a deviation from them is expected rather than alarming.

Until M2/M3 are retrained here, the only valid anchors are the ones this notebook measures itself:
M2-alone and M3-alone on the same ~2,750 test cycles. Published ICBHI SOTA is ≈ **0.60–0.65**; a
fusion in that band is a solid, honest result and **will not look like 0.8213.**

## The official split is not patient-disjoint

Patients **156** and **218** have recordings on *both* sides of the official file (9 train + 8 test,
and 4 train + 4 test). This collides with `Model_Training_Protocol.md` §1 requirement 1, which makes
patient-independence a hard requirement. `CFG["official_overlap_policy"]` decides:

| Policy | Effect | Result |
|---|---|---|
| **`drop_from_train`** *(default)* | Drop those two patients' 13 train recordings | 526 train / **381 test** recordings, 77 / **49** patients, **patient-disjoint** |
| `as_is` | Official split verbatim | 539 / 381, 2 patients leak, violates §1 |

The default keeps the official **test** set byte-identical, so the reported score stays directly
comparable to published ICBHI results, and pays for patient-independence out of the training set
instead — 13 of 539 recordings, 2.4%. The resulting 49 test patients matches the count usually
quoted for the official split. Whichever policy is active is recorded in `dataset_info`.

## Outputs

```
results_M30.json          <- ablation_group: ensemble_fusion, WITH confusion_matrix_raw
                             + icbhi_score_official + best_metrics.admission_test
best_model.pth            <- fusion head
loss_curve.png / accuracy_curve.png / icbhi_score_curve.png / confusion_matrix.png
```

The ICBHI curve plots **both metrics on the same axes** so the constant offset between them is
visible, with M2's official score as a horizontal reference line.

## Testing

53/53 on a synthetic ICBHI-shaped corpus, **using the real committed M2/M3 checkpoints**
(symlinked into the test tree so the notebook's own discovery globs resolve them, exercising
discovery rather than bypassing it).

> **Why real checkpoints, learned the hard way (2026-08-14).** The suite originally built fake
> checkpoints from its *own* copies of the backbone classes, so it could only ever prove
> self-consistency — never that the notebook's classes match the actual checkpoints. One bug got
> through: `M3_MobileNet` named its final layer `classifier`, but M3's real checkpoint uses `head`.
> The first Colab run died in cell 5 on exactly this. `load_backbone_or_die()` did its job and
> refused.
>
> That was **not** cosmetic. The M3-alone baseline is computed through M3's own classifier, so a
> randomly-initialised one would have depressed it and biased the admission test *in favour of*
> fusion — the very comparison this re-run exists to get right. Note the fingerprint check alone
> would have passed (the `features.*` trunk loaded fine); only the missing-parameter check caught it.
>
> Switching the test to the real checkpoints then surfaced a second, quieter mismatch: M3 stores its
> ImageNet normalisation as buffers named `in_mean`/`in_std`, while the notebook registered
> `mean`/`std`. Buffers aren't parameters, so nothing would have raised — the checkpoint's values
> would simply have been ignored in favour of the hardcoded defaults. Those defaults happen to be
> identical today (verified), so no result was wrong, but it would have broken silently the moment M3
> were retrained with different normalisation. Buffers are now named to match, and
> `load_backbone_or_die()` reports any buffer it could not load.

Beyond schema and plot checks, the suite asserts:

- `split_method` really is `patient_independent_official_60_40`, and the test patients match the
  split file (not a 70/30 shuffle)
- the official score **equals a recomputation from the committed confusion matrix**
- the legacy macro score is ≥ the official one (inflation direction holds)
- backbones are frozen (trainable ≪ total)
- the admission test's margin and `beats_both` flag are internally consistent

Plus three deliberate failure injections, all of which must be *refused*:

| Injection | Expected |
|---|---|
| state_dict with wrong keys | raises, naming the tensors that would have stayed random |
| missing checkpoint file | raises |
| no official split file | completes, but records `split_source: patient_id_fallback` and **does not** claim "official" in `split_method` |

Timing observed in the test (150 cycles): extraction 75 s, head training **0.6 s/epoch**.

That last one matters: a silent fallback to a different split is exactly the defect this re-run
exists to correct, so the fallback path is required to be honest about itself.

## Regenerating

```
python3 gen_M30_v2.py
```

Notebook is generated, never hand-edited.
