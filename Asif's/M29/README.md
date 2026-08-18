# M29 — Open-Set Baseline Suite on the M12 Backbone

**Owner:** Asif (Member A) · **Requires:** M2/M12 checkpoint only · **Status:** ✅ run on real ICBHI data, audit-verified

## Result

| Score | AUROC | AUPR | Unknown recall |
|---|---|---|---|
| **Energy (best)** | **0.6466** | 0.4131 | 0.1053 |
| Entropy | 0.5789 | 0.4541 | 0.1053 |
| Mahalanobis (patient) | 0.5689 | 0.4280 | 0.1053 |
| Mahalanobis (class) | 0.5376 | 0.3352 | 0.0000 |
| MSP | 0.5025 | 0.4105 | 0.1053 |
| *(reference)* M6 OpenMax | *0.4516* | *0.2352* | *0.0255* |

`dataset_info` confirms this ran on the real corpus: **104 known / 19 unknown patients**, an exact
match to `Model_Training_Reference.md:182,190`. 62 known-fit / 42 known-test patients, 6,860 total
cycles, all fitting done on known-fit only.

**The trivial Energy score beats M6's OpenMax baseline by a wide margin using zero training** —
just a forward pass through the frozen sound-event backbone. This is now the real bar: M15, once
rebuilt on real data, needs to clear **0.6466**, not 0.4516. See `Asif's/audit/README.md` for the
full read on what this means for the project.

Four standard OOD scores computed on the frozen M12 backbone and evaluated at **patient level** on
ICBHI's real known-vs-pooled-unknown split. No training — the backbone is frozen.

| Score | Definition |
|---|---|
| MSP | `1 − max softmax` (Hendrycks & Gimpel) |
| Entropy | normalised `H(softmax)` |
| Energy | `−logsumexp(logits)` (Liu et al.) |
| Mahalanobis (class) | min distance to a class-conditional Gaussian, shared covariance |
| Mahalanobis (patient) | distance to the known-patient embedding distribution |

## Why this, why now

[`Novelty Search.md`](../../Archive_Files%20(v2)/Novelty%20Search%20v2.md) *(archived)* §2 **Attack 1** is the harshest
plausible review of this project, and it names the exact mitigation:

> *"The paper does not show that this signal is better than simpler baselines (max-softmax score,
> Mahalanobis distance, or even entropy of a single head). Without that comparison, the cross-task
> disagreement claim is unfounded."*
>
> **Mitigation needed:** ablation against energy-based OOD score, Mahalanobis distance from training
> embeddings, and post-hoc entropy — not just OpenMax.

This notebook *is* that mitigation. It is also the **only unknown-detection experiment currently
runnable on real data** — the audit found M13/M15/M17/M19 all operate on synthetic tensors, so no
genuine cross-task result exists yet.

**M15's claim is not "beats OpenMax". It is "beats the best trivial post-hoc score on the same
backbone."** M29 defines that bar.

## Running it

Colab, T4, ~15–25 min (one forward pass over 6,898 cycles plus scoring — no training).

Needs three inputs, all auto-discovered:
1. ICBHI dataset (`kaggle datasets download -d vbookshelf/respiratory-sound-database`)
2. `patient_diagnosis.csv` — ships with that dataset
3. **`Asif's/M2/best_model.pth`** — upload to `/content/` or Drive. This is the M12-selected backbone.

If it shares a session with M2/M3 it reuses `/content/owmtl_spec_cache`.

## Design decisions worth defending

**The unknown group is never fitted on.** Scorers and thresholds are fit on known-*fit* patients
only; the 19 unknown patients are evaluation-exclusive, asserted at runtime, per
[`Model_Training_Reference.md:190`](../../Model_Training_Reference.md).

**Patient-level, not cycle-level.** Cycle scores are averaged per patient. This is the level
[`:266`](../../Model_Training_Reference.md) specifies — and the level M15 should have used but
didn't (it evaluated 518/101 *cycles*, inflating effective N).

**Preprocessing is restated verbatim from M2** rather than imported, because any drift silently
stops these being the M12 backbone's embeddings.

**Asthma (n=1) and LRTI (n=2) are excluded**, documented in `dataset_info.excluded_rationale`
rather than silently dropped. Too few to place in either group without distorting it.

**Architecture is read from the checkpoint's `model_config`**, so this tracks whatever M12 selected.
A missing-weights assertion catches a wrong or mismatched checkpoint — it fired during testing and
caught a genuine key mismatch.

## Reading the outcome

The notebook prints its own interpretation so the framing is fixed before the numbers are seen:

- **Best baseline ≳ 0.70** → M15 must beat *this*, not M6's 0.4516. Attack 1 becomes a live risk.
- **Best baseline 0.60–0.70** → that's the real floor for M15.
- **Nothing clears 0.60** → with n=19 unknowns the task may be *underpowered* rather than the
  methods inadequate — which points at the evaluation-redesign contribution `Novelty Search.md`
  §1 already calls the project's strongest, and makes a characterised negative the honest framing.

All three are publishable. An untested claim is not.

## Honest limits

- **n = 19 unknown patients.** Wide intervals; treat few-point differences as noise.
- **One split draw.** Repeat over seeds, or LOPO the unknown group, before the manuscript.
- **Post-hoc scores on a sound-event backbone** — a floor, not a ceiling. A weak result here does
  not by itself vindicate the cross-task mechanism.

## Note

**M29 is a new model ID not in `Model_Training_Reference.md`.** Add it to the Quick Index under the
`rejection_method` ablation group (M6 vs M29 vs M15) when you commit results.
