# M2 — CNN Baseline (Tuned, Final)

**Owner:** Asif (Member A) · **Requires:** nothing · **Status:** notebook ready, not yet run

`M2_cnn_baseline_tuned.ipynb` is the tuned counterpart of Barshon's M1 — the CNN row of the
`backbone_architecture` ablation table that `Model_Training_Reference.md:114` requires for M12.

## Running it on Colab

1. **Runtime → Change runtime type → T4 GPU.** (The notebook warns and continues on CPU, but a
   full run would take days.)
2. Upload the notebook (File → Upload notebook) and **Run all**.
3. Cell 0 mounts Drive; outputs go to `/content/drive/MyDrive/OWMTL/M2/`, so a disconnect costs
   at most one epoch.
4. If the dataset isn't found, Cell 1 prints the exact `kaggle datasets download` commands. Run
   them in a new cell, then re-run Cell 1.
5. When it finishes, run the last cell and download `M2_handoff_bundle.zip` **before closing the
   tab**.

### Expected cost on a T4

| Stage | Time |
|---|---|
| Spectrogram cache build (once) | ~5–10 min |
| Hyperparameter sweep (6 configs × 3 folds × 12 epochs) | ~35–60 min |
| Final training run (60 epochs) | ~10–20 min |
| **Total** | **~1–1.5 hours** |

The cache lives at `/content/owmtl_spec_cache` and is **shared with M3** — identical §2 preprocessing
means an identical cache signature, so whichever notebook runs second skips extraction entirely.

Almost all of the speedup over M1's ~4.4 hours comes from caching the spectrograms to local disk
once instead of recomputing them with librosa every epoch. M2 has no augmentation, so the
spectrograms are identical across epochs and the cache is exact, not an approximation.

### If you're short on GPU hours

In Cell 1's `CFG`, set `"run_hp_sweep": False`. It trains straight from `fallback_config` (which is
M1's exact configuration) and skips the ~40-minute sweep. You still get a valid protocol-compliant
`results_M2.json` — you just lose the "tuned" claim that distinguishes M2 from M1, which is the
whole point of the model, so only do this as a fallback.

## What it produces

Written to `.../OWMTL/M2/results/`:

```
results_M2.json          <- the M12 ablation-table row (§4 + §4.1 schema)
M2_hp_sweep.json         <- per-config, per-fold sweep table
loss_curve.png
accuracy_curve.png
f1_curve.png
confusion_matrix.png     <- raw + row-normalized
icbhi_score_curve.png    <- primary-metric decomposition (Se vs Sp)
hp_sweep_comparison.png
```

plus `checkpoints/best_model.pth`.

## After the run

1. Commit `results_M2.json`, `M2_hp_sweep.json`, and the PNGs into this folder. `.pth` files are
   gitignored (`.gitignore:27`) — put the checkpoint on Drive and record the link in
   `Best_model_pth_file Link.txt`, the same convention Barshon used for M4.
2. Send `results_M2.json` to Barshon for M12. His current justification compares only M1 and M4;
   this supplies one of the two missing rows (M3 is the other).

## Design notes

**Two evaluation protocols, deliberately.** `Model_Training_Reference.md` §M2 asks for k-fold CV,
but M1 and M4 were both scored on the official 60/40 split. Reporting M2 only on k-fold would make
it non-comparable to them in the M12 table. So the notebook uses patient-grouped k-fold **on the
train patients only** to pick hyperparameters (no test-set leakage into selection), and reports the
headline number on the official 60/40 split. The fold-to-fold spread in `M2_hp_sweep.json` is the
honest generalization estimate.

**Preprocessing uses the §2 default `n_fft=1024`**, not M1's 512 or M4's 512 + 20–8000 Hz band.
Both of those are self-declared deviations in their own results files; M2 is a reported ablation row
so it uses the shared default. This is recorded in `ablation.known_deviations`.

**Protocol §1 is asserted, not assumed.** The notebook raises if any patient appears in both train
and test, and re-checks it inside every CV fold.

## Caveat worth carrying into the write-up

M2 selects its best checkpoint on the test split — the same thing M1 and M4 do, so the three-way
comparison is fair, but none of the three absolute numbers is a clean held-out estimate. Say so in
the paper rather than letting a reviewer find it.
