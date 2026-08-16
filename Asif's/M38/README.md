# M38 — Large-N Open-Set Baseline Suite (near-OOD and far-OOD)

**Owner:** Asif · **Chunk:** C · **Requires:** M12's frozen backbone (`Asif's/M2/best_model.pth`)
**Status:** notebook ready, tested 62/62, **not yet run on real data**

M29's four post-hoc OOD scorers evaluated against **three** OOD pools instead of one, with a 95%
confidence interval on every AUROC. No training — the backbone stays frozen, exactly as in M29.

## Why this exists

`Novelty Search.md` (moved -> `../../Archive_Files (v2)/Novelty Search v2.md`) §1 names the project's strongest contribution as *"the statistically defensible
evaluation redesign — coarse pooled unknown class + **large-N OOD stress tests**."*

`Asif's/Statistics/SIGNIFICANCE_REPORT.md` showed that had never actually been delivered. At n=19
unknown patients, **every** open-set AUROC in the repo — M29's best included — has a 95% CI that
includes chance. The evaluation was still the fragile small-sample split the contribution claims to
replace.

M38 changes the sample size, not the method. Same scorers, same frozen backbone, same preprocessing,
same in-distribution pool. Only the OOD pool varies.

This is deliberately **not** a new novelty item. `Novelty Search.md` §4.0 fixes the set at 2–3 and
all three are built; §4.0's own rule is *"swap, not accumulate."* M38 adds depth to the selected
contribution rather than surface area to the technique list.

## The three regimes

| Regime | OOD pool | Modality | Role |
|---|---|---|---|
| **A — ICBHI unknown** | 19 patients | stethoscope, lung cycles | Reproduces M29. Sanity check. |
| **B — SPRSound** | large N | stethoscope, lung cycles (pediatric) | **Near-OOD. The result that matters.** |
| **C — Coswara** | large N | phone mic, voluntary cough | **Far-OOD control, not a finding.** |

The in-distribution side (ICBHI known-test patients) is held identical across all three, so the only
variable is what is being detected.

### Read regime C carefully

Coswara is heavy-cough audio recorded on phone microphones; ICBHI is stethoscope auscultation of
breathing cycles. A detector can separate those on **recording modality alone**, knowing nothing
about disease. A high Coswara AUROC is therefore *not* evidence for the open-world mechanism — it is
evidence the model can tell a phone from a stethoscope, which is exactly the confound
`Novelty Search.md` §2 Attack 3 warns about.

Its value is inverted: **if a detector cannot clear far-OOD, its near-OOD numbers aren't worth
interpreting.** A null on regime C would mean the scorers are broken, not that the task is hard.

## Running it

Colab or Kaggle, ~20–30 min, GPU optional (inference-only, so CPU works — just slower). Needs:

1. The three Kaggle datasets — cell 0b downloads all of them idempotently (paste your Kaggle key first).
2. **`Asif's/M2/best_model.pth` uploaded to `/content/` or Drive.** This is the one thing the
   download cell can't fetch, and the notebook fails fast with an explicit message if it's missing.

## What it produces

```
results_M38.json            <- ablation_group: ood_generalization, baseline_model_id: M38 -> M29
scores_M38.csv              <- raw per-sample scores  (see below — this matters)
auroc_forest_M38.png        <- every regime x score with its 95% CI against the chance line
roc_curves_M38.png          <- best scorer per regime
score_distributions_M38.png <- in-distribution vs each OOD pool
```

**`scores_M38.csv` is the quietly important output.** No model in this project has ever saved raw
per-sample scores, which is why `SIGNIFICANCE_REPORT.md` had to fall back on an analytic
Hanley-McNeil approximation instead of a proper **paired DeLong test**. Any future method that dumps
the same columns (`regime, pool, group_id, score_name, score, is_ood`) for the same `group_id`s can
be compared against M38 with the real test.

## Design notes

**Confidence intervals are computed inline**, per regime per scorer, not bolted on afterward. A bare
point estimate is what produced the situation the significance report had to document.

**Nothing is fitted on any OOD pool.** The Mahalanobis scorers are fit on ICBHI known-*fit* cycles
only; ICBHI unknowns, SPRSound and Coswara are all evaluation-only. Asserted in code.

**Evaluation level is tracked per regime, not assumed.** External sets have no cycle annotations, so
each file is one sample (first 8 s, tiled if shorter), then aggregated to participant level where the
directory layout or filename convention allows it. Where it doesn't, the regime reports
`evaluation_level: "file"` and says so loudly rather than silently mixing levels — that mix-up is
trap #6 in `Asif's/CLAUDE.md` and the project has already been burned by it once (M6 is cycle-level
while everything else is patient-level).

**Preprocessing is inherited verbatim from M2.** If it differed by one parameter these would not be
the M12 backbone's embeddings and the comparison to M29 would be mis-specified.

## Bug this corrects

`Barshon's/M19/results_M19.json` reports Coswara with `num_samples: 2`. Its configured path is:

```
.../coswara_data/kaggle_data/pGxub66GjDdAaJDd95hGHo3BcnJ3
```

That trailing segment is a **single participant's folder**, not the dataset root — Coswara's layout
is `kaggle_data/<PARTICIPANT_ID>/*.wav`. M19 evaluated one participant's two files.

**M19's Coswara AUROC (0.4881) should not be cited by anyone.** M38 resolves the parent directory
and globs recursively, and warns at runtime if fewer than 10 files turn up — the exact signature of
this bug.

Separately: M19's SPRSound result (AUROC 0.3226, n=1000) is *significantly below chance* under any
plausible in-distribution count — the CI never rises above [0.23, 0.41]. That's either a sign-flip
bug or a real domain-shift finding, and regime B is what will tell the difference, since it supplies
the trivial-baseline control M19 never had.

## Testing

62/62 checks on a synthetic mini-corpus covering all three datasets (ICBHI patients/cycles/annotations,
SPRSound flat `<patient>_<rest>.wav` layout, Coswara participant-folder tree) plus a real
M2-architecture checkpoint. Every notebook cell executes in order in one namespace; assertions cover
schema compliance, split hygiene, the identical-in-distribution-pool invariant, CI correctness,
participant-level grouping (12 participants not 36 files), the raw-score dump, plots, and the
interpretation guardrails in the JSON.

**A second run deliberately reproduces the M19 bug** — Coswara pointed at one participant folder —
and asserts the run still completes, the tiny `n_ood` is visible, and the other two regimes are
unaffected.

Numbers from the synthetic run are meaningless (random noise is trivially separable); the test
checks mechanics, not science.

## One audit-tool change came out of this

M38's far-OOD regime can legitimately produce `unknown_recall = 1.0` — at a threshold retaining 95%
of known samples, a cleanly separated pool gets caught in full. `audit_project.py`'s
`check_perfect_metrics` read that as a train/test leak and raised CRITICAL.

That reasoning is wrong for open-set *operating-point* statistics: recall at a chosen threshold is a
property of where the threshold sits and how separable the pools are, not evidence of learning. The
check now splits the two cases — closed-set `accuracy`/`f1` at 1.0 stays CRITICAL, and so does a
perfect **AUROC** (threshold-free, therefore still suspicious), while a perfect open-set
operating-point metric becomes a WARNING that asks the right question: *is the separation for the
right reason, or is it modality mismatch?*

Verified no regression: full-repo audit findings are unchanged at 15 CRITICAL / 62 WARNING / 35 INFO.

## Regenerating the notebook

```
python3 gen_M38.py
```

Per workstream convention the notebook is generated, never hand-edited.
