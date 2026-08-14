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

`Novelty Search.md` §4.0 selects the fusion ensemble **"only if it measurably beats both backbones
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

Colab/Kaggle, ~20–30 min on a T4. Only the fusion head trains (≈2.1M of 8.0M params); both backbones
are frozen.

Needs three inputs:
1. ICBHI audio — cell 0b downloads it (paste your Kaggle key first)
2. `Asif's/M2/best_model.pth` → upload as `/content/M2_best_model.pth`
3. `Asif's/M3/best_model.pth` → upload as `/content/M3_best_model.pth`

The notebook fails fast with an explicit message if any is missing — it deliberately refuses to run
on randomly-initialised backbones, since that's the failure mode it exists to rule out.

## Calibration — what a good result looks like

On the official split and official metric: **M2 = 0.6138**, **M3 = 0.5895**, **M22 = 0.6495**, and
published ICBHI SOTA ≈ **0.60–0.65**.

A fusion landing near 0.65 is a solid, honest, literature-level result. **It will not look like
0.8213, and it shouldn't.** Cell 7 also prints M2/M3 against their published values as a sanity
anchor — a large deviation there means the split or checkpoints differ from the published runs, and
nothing downstream is trustworthy.

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

53/53 on a synthetic ICBHI-shaped corpus with checkpoints saved from *this notebook's own* backbone
classes.

> **Known limitation, learned the hard way (2026-08-14).** Because the test checkpoints were
> generated from the notebook's own class definitions, the suite could only ever prove
> *self-consistency* — it could not catch a mismatch between those definitions and the **real** M2/M3
> checkpoints. One got through: `M3_MobileNet` named its final layer `classifier`, while M3's actual
> notebook (and checkpoint) uses `head`. The first Colab run died in cell 5 on exactly this.
> `load_backbone_or_die()` did its job and refused. Fixed by renaming to `head`.
>
> This was **not** cosmetic: cell 10 evaluates M3-alone through `forward()`, so a randomly-initialised
> classifier would have depressed the M3-alone baseline and biased the admission test *in favour of*
> fusion — the exact comparison this re-run exists to get right. Note also that the fingerprint check
> alone would have passed (the `features.*` trunk loaded fine); only the missing-parameter check
> caught it.
>
> The real M2/M3 checkpoints are committed in the repo and should be a test input.

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

That last one matters: a silent fallback to a different split is exactly the defect this re-run
exists to correct, so the fallback path is required to be honest about itself.

## Regenerating

```
python3 gen_M30_v2.py
```

Notebook is generated, never hand-edited.
