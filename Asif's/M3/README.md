# M3 — MobileNet / DenseNet Lightweight Backbone

**Owner:** Asif (Member A) · **Requires:** nothing · **Status:** notebook ready, not yet run

`M3_lightweight_backbone.ipynb` is the **efficiency comparison point** in the backbone race — the
lightweight row of the `backbone_architecture` ablation that `Model_Training_Reference.md:114`
requires for M12, and the natural-small-model reference Member C's compression work measures against.

M3 is not trying to beat M4. `Model_Training_Reference.md:168` states its purpose directly:
*"parameter count and model size are especially important here since this model's whole point is
efficiency."* The deliverable is the accuracy-vs-cost frontier, not a win.

## What M2's result changed about M3's question

M2 finished on 2026-07-30 and **won the backbone race outright** — ICBHI 0.7227 / F1 0.5238,
ahead of both M1 (0.7181 / 0.4844) and M4 (0.6359 / 0.4109). *(Those ICBHI figures are the legacy
macro metric; on the official ICBHI 2017 metric M2 scores 0.6138 and M3 0.5895 — the ranking is
unchanged. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`.)* Crucially it landed at **3.6 M params
/ 13.86 MB**, which is already lightweight.

That makes M3's most interesting comparison **M3 vs M2**, not M3 vs M4:

| | Params | Size |
|---|---|---|
| M2 (tuned CNN, from scratch) | 3.6 M | 13.9 MB |
| M3 `mobilenet_v3_large` | 3.0 M | 11.5 MB |
| M3 `mobilenet_v3_small` | 0.9 M | 3.7 MB |
| M4 (AST) | 86.4 M | 988 MB |

MobileNetV3-Large is within 1.2× of M2's size, so that pairing is close to **size-matched** — it
isolates the one thing M3 actually adds (ImageNet pretraining + a modern efficient architecture)
from raw capacity. M3-vs-M4 confounds size and accuracy at once; M3-vs-M2 does not. The notebook's
final cell computes both and flags when the sizes are within 2× so you can make that claim safely.

If MobileNetV3-Small (0.9 M / 3.7 MB, **3.8× smaller than M2**) holds up, that is the strongest
deployability row Member A can put in the paper.

## Running it on Colab

Identical to M2 — T4 runtime, upload, Run all, download the ZIP before closing the tab.

**Cache note.** Both notebooks use the same §2 preprocessing and share a cache at
`/content/owmtl_spec_cache`, but that lives on Colab's *local* disk, not Drive — so it only carries
over within a single session. Since M2 already finished in an earlier session, M3 will rebuild the
cache (~5–10 min). That's a one-time cost, not per-epoch.

**M3 needs internet** on first run to fetch the ImageNet weights (torchvision caches them
afterwards). If the download fails, the notebook prints a loud warning and falls back to random
init — check `config.pretrained_weights_loaded` in `results_M3.json`; if it's `false`, the run is
not an "ImageNet-pretrained" result and must be redone.

### Expected cost on a T4

| Stage | Time |
|---|---|
| Spectrogram cache (skipped if M2 already ran) | ~5–10 min |
| Architecture sweep (6 candidates × 2 folds × 10 epochs) | ~1.5–2.5 h |
| Final training run (40 epochs) | ~20–40 min |
| **Total** | **~2–3 hours** |

More expensive than M2 because DenseNet121 and MobileNetV3-Large are heavier per epoch than the
small CNN. That is also why `sweep_folds` defaults to **2** here rather than M2's 3 — architecture
families separate far more strongly than hyperparameters do, so 2 folds still ranks them clearly at
half the cost. Raise it to 3 in `CFG` if you have the hours.

**If you're short on GPU time,** either drop `densenet121` from `SWEEP_SPACE` in Cell 10 (it is the
single most expensive candidate and MobileNet is the more cited choice on ICBHI), or set
`run_hp_sweep: False` to train `fallback_config` (MobileNetV3-Large) directly. Dropping DenseNet is
the better trade — it costs you one row, not the architecture comparison itself.

## What it produces

Same file set as M2, in `.../OWMTL/M3/results/`:

```
results_M3.json          <- the M12 ablation-table row (§4 + §4.1 schema)
M3_hp_sweep.json         <- per-architecture, per-fold table with params + size
loss_curve.png
accuracy_curve.png
f1_curve.png
confusion_matrix.png
icbhi_score_curve.png
hp_sweep_comparison.png  <- left: score per candidate | right: accuracy-vs-size frontier
```

The right panel of `hp_sweep_comparison.png` is **M3's headline figure** — it is the plot that
justifies the efficiency claim.

## Design notes

**Channel adaptation.** ImageNet backbones need 3 channels; a log-mel has 1. The notebook repeats
the channel 3× and applies ImageNet mean/std, keeping the pretrained stem convolution intact.
Collapsing the stem to 1 channel instead would throw away what those filters learned. Recorded in
`ablation.known_deviations` since M1/M2 consume 1 channel directly — the underlying spectrogram is
bit-identical to M2's.

**No resizing to 224×224.** All four candidates are fully convolutional with adaptive pooling and
take the native 128×801 spectrogram (verified in Cell 6, which prints the table). This keeps §2
preprocessing identical across M2/M3/M4 instead of M3 alone being resampled.

**The sweep compares families, not hyperparameters.** M2's sweep varied depth/width/dropout because
M2 *is* one family. M3's whole purpose is a family comparison, so the budget goes to
MobileNetV3-Small / MobileNetV3-Large / MobileNetV2 / DenseNet121 — plus one frozen-features run
that quantifies what fine-tuning actually buys over using ImageNet as a fixed feature extractor.
A reviewer will ask; that row answers it with a number.

**Parameter counts are for the assembled model** (backbone `.features` + 4-class head), not the
1000-class ImageNet original — so they compare directly against M2's and M4's numbers. Expect
roughly 0.9 M (MNV3-Small) to 7.0 M (DenseNet121) against M4's 86.4 M.

## After the run

1. Commit `results_M3.json`, `M3_hp_sweep.json` and the PNGs here. `.pth` is gitignored — Drive link
   in `Best_model_pth_file Link.txt`, same convention as M4.
2. Send `results_M3.json` to Barshon. **With M2 and M3 both done, the `backbone_architecture` group
   finally has all four rows** (M1, M2, M3, M4), and M12's decision can be re-derived from data
   instead of re-asserted. The current `M12_backbone_justification.txt` compares only M1 and M4 and
   contradicts itself about which one won — worth flagging when you hand these over.

## Reading the result

Both outcomes are publishable:

- **M3 lands near M4** → strong deployability claim, and a solid reference point for Member C.
- **M3 lags M4 clearly** → that gap *is* the value AST's AudioSet pretraining adds, which is exactly
  what M12's "why this backbone" justification needs stated numerically.

The final cell computes this comparison for you, including a check on whether the gap to M4 is
smaller than the cross-validation spread (if it is, report "matches M4 within CV noise at Nx
smaller", not "slightly worse").

## Caveat worth carrying into the write-up

Same as M2: best-checkpoint selection uses the test split, matching M1/M2/M4, so the four-way
comparison is fair but no absolute number is a clean held-out estimate. The k-fold spread in
`M3_hp_sweep.json` is the honest one.
