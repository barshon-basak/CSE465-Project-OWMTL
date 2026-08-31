# M41 — Swin-T Sound-Event Backbone (pre-trained transformer)

**Owner:** Farhana · **Supersedes:** M23 (AST + SpecAugment) · **Status:** ✅ run on real ICBHI,
corrected official 60/40 split · **For:** `RTK_requirements.md` req. 2 (pre-trained model per member)
+ req. 3 (≥ 4 transformer models)

## Result — negative transfer, not a broken run

| Run | Se | Sp | **icbhi_score_official** | acc | macro-F1 | params | size |
|---|---|---|---|---|---|---|---|
| M41 clean | 0.4033 | 0.5468 | **0.4751** | 0.4882 | 0.3669 | 27.5 M | 105 MB |
| M41 + SpecAugment | 0.3773 | 0.5583 | **0.4678** | 0.4844 | 0.3620 | 27.5 M | 105 MB |
| Δ (aug − clean) | −0.026 | +0.012 | **−0.007** | −0.004 | −0.005 | — | — |

*(macro `icbhi_score`: 0.5874 clean / 0.5804 aug — the inflated variant; lead with `official`.)*

**Swin-T (ImageNet-1k pre-trained) sits at chance on ICBHI.** Over 24 epochs the training loss
halves (1.35 → 0.51) while validation `icbhi_score_official` stays flat at ~0.49 with no trend —
Se and Sp just seesaw inversely as the decision threshold drifts. The model fits the training set
but learns **no generalisable respiratory-sound signal**. SpecAugment changes nothing (Δ ≈ 0), as
expected for a model that isn't learning.

**Contrast:** M4 (AST, *AudioSet*-pre-trained) reached Se 0.44 / macro-ICBHI 0.64 on the same data.

**Conclusion:** image-domain pre-training does not transfer to respiratory sound classification;
audio-domain pre-training (AST) is necessary. This is a *stronger* version of what
`RTK_requirements.md` §4 predicts ("expect the transformers to lose") — Swin fails entirely rather
than merely losing to the CNN. Per §12 ("no tuning a transformer until it beats M2 — report the
loss honestly") the model was **not** tuned to chase a number.

## What was done

- Backbone: `timm` `swin_tiny_patch4_window7_224`, ImageNet-1k, full fine-tune, backbone lr 2e-5 /
  head lr 8e-4, AdamW + cosine, drop_path 0.2, head dropout 0.3, label smoothing 0.1, class-weighted
  CE (sqrt-tempered), seed 42.
- Input: shared 11-stage pipeline (`owmtl.features`) — band-pass 50–2000 Hz, 8 s cyclic pad, peak
  norm, spectral-gating denoise, 128-mel log-mel, fit-only standardisation → resize to 224×224,
  3-channel + ImageNet norm.
- Split: `Asif's/audit/official_split.py` `mode="patient_independent"` (corrected official 60/40).
  Patient-level val split (15% of train patients) for early-stop / checkpoint selection; **test
  evaluated once.** One recording absent from the official split file
  (`226_1b1_Pl_sc_LittC2SE`) assigned to its patient's partition.
- Two runs from one notebook via `CFG["is_augmented"]`. Monitored `icbhi_score_official`.

## Files

```
gen_M41.py                     generator — edit this, re-run to rebuild the notebook
M41_SwinT_soundevent.ipynb     the notebook (Kaggle: attach ICBHI + m41_deps datasets, GPU, run all)
results_M41.json               clean run — full §3 metric suite + training_history + §4.1 ablation block
results_M41_aug.json           SpecAugment run
{loss,accuracy,f1,confusion_matrix}_curve_M41[_aug].png
```

`best_model.pth` (315 MB) is not committed (`.gitignore` `*.pth` / `*.pt`); on Drive if needed.

## For the master table (Sami, T2)

Row: `model=M41` / `type=transformer` / **`split=official_60_40_corrected`** / `augmented=false|true`
/ `icbhi_score_official=0.4751|0.4678` / `params=27.52M` / `size_MB=105.05` / `s_per_epoch≈30`.
Present alongside M40/M42/M43. Frame as the accuracy/compute trade-off + negative transfer, not SOTA.

## Honest limits

- Test = one split draw. No bootstrap CI computed (add before the manuscript if M41 is cited).
- 224×224 resize compresses the ~800-frame time axis ~3.5× — a sliding-window input (2 s chunks, no
  time compression) was **not** tried; it might lift M41 off the chance floor but per §12 that is
  out of scope unless the team decides otherwise.
