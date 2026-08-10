# Project Audit

Automated protocol-compliance and **result-validity** checker for every results file in the repo.

```bash
python3 "Asif's/audit/audit_project.py"          # writes PROJECT_AUDIT.md + .json
python3 "Asif's/audit/audit_project.py" --quiet  # one-line summary
```

Standard library only. No GPU, no dataset, runs in under a second. Exits `1` if any CRITICAL
finding exists, so it can be wired into CI or a pre-commit hook later.

## Why this exists

Schema compliance is not the same as validity. A run can produce a perfectly-formed
`results_M*.json` and still be reporting nothing — a constant-class predictor, a metric at chance,
or a leak that produces 1.0 across the board. The M28 merge would happily consume all three.

**Every check here was written against a failure that is actually in this repo**, not a
hypothetical one:

| Check | Catches | Found in |
|---|---|---|
| **`synthetic_data_not_real_dataset`** | **a Dataset that fabricates its input instead of loading audio** | **M11, M13, M15, M17, M18, M19, M20, M21, M24** |
| `model_may_be_randomly_initialised` | a fallback that proceeds with an untrained model | M15, M17, M18, M19 |
| `discrimination_at_or_below_chance` | AUROC ≤ 0.5 — a coin flip does as well | M6, M15, M19 |
| `comparison_against_subchance_baseline` | "beats baseline by N%" where the baseline is below chance | M15 |
| `perfect_metrics_implausible` | metrics of exactly 1.0 — leak or train-set eval | M21 |
| `metric_constant_across_sweep` | accuracy that doesn't move as capacity changes | M18 |
| `single_class_collapse` | one class at recall 1.0, the rest at 0.0 | M13 [OLD] |
| `frozen_validation_metric` | validation score identical every epoch | M13 [OLD] |
| `best_epoch_is_first` | best epoch is 1 — the model never trained | M13 (both) |
| `best_epoch_very_early` | best epoch in the first 15% of the budget | M3 |
| `no_better_than_majority_class` | accuracy at the class prior | — |
| `missing_open_set_metrics` | M6/M15/M17 with no AUROC/AUPR anywhere | — |
| `not_protocol_compliant` | `M*_metrics.json` files the M28 merge can't read | M15, M19, M21 |
| `schema_*` | §4 / §4.1 blocks the merge expects | several |
| `patient_independence_*` | protocol §1, the one non-negotiable requirement | several |
| `cites_subchance_reference_value` | a file citing another model's already-broken number as context (not a claim) | M29 |

## Current state

As of the last run: **30 CRITICAL, 8 WARNING, 7 INFO** across 34 files.

### The headline finding

**Every model downstream of M12 — except M6 — is running on synthetically generated data, not
ICBHI.** M11, M13, M15, M17, M18, M19, M20, M21 and M24 all build their inputs with
`torch.randn` / `np.random` inside the Dataset instead of loading audio. M15's class is even
*named* `ICBHI_OWL_Dataset` while returning `torch.randn(1, n_mels, 801) * 0.5`; M11's is named
`SyntheticICBHIDataset` outright.

That single fact explains every other anomaly at once:

| Symptom | Cause |
|---|---|
| M21 accuracy = 1.0 | synthetic data with the class label added directly into fixed frequency bands |
| M18 accuracy 0.28 flat across a 62× sweep | noise in, constant class out |
| M19 OOD AUROC 0.517 / 0.483 | noise |
| M15 AUROC 0.618 | noise + known/unknown label priors that happen to differ |

So the correct reading is **not** "the mechanism is at chance." It is **"the mechanism has never
been evaluated."** These are pipeline-scaffolding runs that were committed as results. The work is
earlier than the checklist claims, not failed — which is a far more recoverable position.

### What is real

| Status | Models |
|---|---|
| ✅ Real data, trustworthy | **M1, M2, M3, M4, M6, M12** |
| 🔴 Synthetic — not results | M11, M13, M15, M17, M18, M19, M20, M21, M24 |

**M6 is one real downstream result, and it is a genuine negative.** It loads real audio (19/19
cells executed, 72 train / 32 known-test / 19 unknown-test patients) and reports
`open_set.auroc = 0.4516` with `unknown_recall = 0.0255` — the OpenMax baseline detects 2.5% of
unknowns, worse than chance.

**M29 (Asif, `rejection_method` group) is the second real downstream result, and it changes the
picture.** Four trivial post-hoc OOD scores computed on the frozen M12 backbone, evaluated at
patient level on the real 104-known / 19-unknown split — exact match to
`Model_Training_Reference.md:182,190`, confirming this ran on the genuine corpus, not synthetic
data:

| Score | AUROC | AUPR | Unknown recall |
|---|---|---|---|
| **Energy (best)** | **0.6466** | 0.4131 | 0.1053 |
| Entropy | 0.5789 | 0.4541 | 0.1053 |
| Mahalanobis (patient) | 0.5689 | 0.4280 | 0.1053 |
| Mahalanobis (class) | 0.5376 | 0.3352 | 0.0000 |
| MSP | 0.5025 | 0.4105 | 0.1053 |
| *(reference)* M6 OpenMax | *0.4516* | *0.2352* | *0.0255* |

A trivial energy score — no disease head, no cross-task mechanism, just the sound-event backbone —
clears **AUROC 0.6466**, comfortably beating M6's 0.4516 and landing just short of the conventional
0.70 "acceptable discrimination" line. Two consequences:

1. **This is now the real bar for M15.** Once M15 is rebuilt on real data, "beats M6" is a low bar
   already cleared by a zero-training baseline. The paper's novelty claim needs to clear **0.6466**,
   not 0.4516 — this is exactly the mitigation `Novelty Search.md` §2 (Attack 1) demands.
2. **Unknown detection is not hopeless on this backbone.** With n=19 unknown patients the interval
   on 0.6466 is wide, but "everything is at chance" is no longer the working hypothesis — a signal
   exists in the embeddings; the open question is whether cross-task disagreement adds anything on
   top of it.

Read `PROJECT_AUDIT.md` for the full detail with evidence and file paths.

## Adding a check

Write a function with the signature `check_x(model_id, payload, source, findings)` that appends
`Finding(SEVERITY, model_id, "check_name", message, evidence, source)`, then add it to the tuple in
`audit_protocol_json`. Checks are individually exception-guarded — a broken check degrades to an
INFO finding rather than killing the run.

Keep messages explanatory rather than terse: this report is read by people deciding whether to
spend GPU hours, so each finding should say *why* the number can't support the claim, not just
that a threshold tripped.

## A note on scope

This tool audits **numbers in files**. It cannot tell you whether the underlying experiment was
designed correctly. Two things it will not catch, both of which apply here:

- **M15 evaluates 518 known / 101 unknown *cycles*.** `Model_Training_Reference.md:266` specifies
  patient-level LOPO over 104 known / 19 unknown *patients*. Cycle-level evaluation inflates the
  effective N and weakens patient-independence on the unknown group.
- **M18 trained on 300 samples and tested on 100**, far below the corpus size — a scale problem the
  constant-accuracy check flags only indirectly.
