# M22 — MobileNetV2 + SpecAugment

**Owner:** Asif · **Requires:** M3 (clean counterpart) · **Status:** notebook ready, tested, not yet run on real data

M3 with SpecAugment applied to the training split and **nothing else changed** — same architecture
(MobileNetV2, ImageNet-pretrained), same winning hyperparameters, same preprocessing, same
patient-independent split, same seed. Single-variable discipline is the whole point: it's what makes
"augmented vs. clean" a valid comparison instead of two unrelated runs.

## Why this exists

Two reasons:

1. **Course requirement #5** — "apply data augmentation techniques for training samples and find the
   results similar to #3 and #4." M3 is the only pretrained model in the project, so this is its
   augmentation counterpart.
2. **M3 overfits early** — best validation was epoch 4 of 40. SpecAugment is the standard first
   response to that, so this is a genuine test, not a box-tick.

## Running it

Colab, T4, ~15–20 min — cheaper than M3 since there's no hyperparameter sweep (M3's winning config
is reused verbatim; re-sweeping would confound the augmentation comparison with a hyperparameter
change).

Needs the same three inputs as M3: ICBHI dataset, and reuses `/content/owmtl_spec_cache` if run in
the same session as M2/M3 (the un-augmented spectrogram cache is shared — augmentation happens
on-the-fly at read time, not baked into the cache).

## What it produces

```
results_M22.json          <- ablation_group: augmentation_effect, baseline_model_id: M3
loss_curve.png / accuracy_curve.png / f1_curve.png / confusion_matrix.png / icbhi_score_curve.png
specaugment_preview.png   <- clean vs. augmented vs. difference, one real training cycle
```

The final summary cell prints a **live comparison against M3** — accuracy/precision/recall/F1/ICBHI
delta, params (identical by construction), size, time/epoch, and which epoch was best in each run —
loaded directly from `Asif's/M3/results_M3.json` so it can't drift from the actual clean baseline.

## Design notes

**SpecAugment is applied on-the-fly, not cached.** Two frequency masks (≤24 of 128 mel bins) and two
time masks (≤80 of 801 frames) per training spectrogram, freshly sampled every epoch via torch's
seeded RNG. A cached augmentation would freeze one masked view per sample for the whole run — worse
than fresh masks, and it would also break cache compatibility with M2/M3.

**Train-augmented, test-clean is asserted at runtime**, not just intended — the notebook raises if
either loader has the wrong `augment` flag before training starts. Augmenting the test split would
silently invalidate the comparison.

**No re-sweep.** `run_hp_sweep` is hardcoded `False` and `fallback_config` is frozen to M3's actual
winner (`mobilenet_v2`, lr 5e-4, cosine). This is deliberate: with a sweep, a difference between M22
and M3 could be architecture-selection noise instead of an augmentation effect.

**Tested end-to-end** on a synthetic mini-ICBHI corpus, same discipline as M2/M3 — 40/40 checks
passed, including that `train_loader.dataset.augment is True` and
`test_loader.dataset.augment is False`. Caught one real bug in that process: the results file and
handoff bundle were initially still named after M3 (copy-paste from the source notebook) — fixed
before this was tested clean.

## Reading the outcome

The notebook states its interpretation before the numbers can be spun:

- **SpecAugment helps** (ICBHI and F1 both up) → report as the augmented row against M3.
- **SpecAugment hurts** → also a real, reportable finding — masking up to 24/128 mel bins may be too
  aggressive for 8s respiratory cycles, where the diagnostic event (a crackle) is often brief and
  narrowband. Worth trying a milder setting as a follow-up, not a reason to hide the result.
- **No meaningful difference** → report as "no significant effect," not spun either direction.

Also watch the **best-epoch shift**: augmentation delaying the best epoch (a later epoch than M3's 4)
is the expected regularization signature; an earlier or unchanged best epoch means it isn't acting as
one here.

## For the presentation video

If this is the augmentation half of a video already covering M3: same three visuals (backbone table,
confusion matrix, curves) plus `specaugment_preview.png`, and the comparison table the summary cell
prints. One line covers it: *"Same MobileNetV2, same hyperparameters — only difference is SpecAugment
on training. Here's what changed."* Then read the delta, honestly, whichever direction it goes.
