# Archive v4 — superseded model runs (2026-08-30)

Runs that were **replaced by a corrected re-run**, kept out of the working tree so the model
folders show only the current version. Nothing here is cited by the paper. Nothing here should
be cited by the paper.

**These are not deleted, deliberately.** They are the evidence of *what was wrong* — the
`supersedes_note` field in each new `results_M*.json` refers to the numbers in these files, and
`SYNTHETIC_DATA_REMEDIATION.md` cites them by value. If a supervisor or reviewer asks "how do you
know the corrected pipeline is right?", the answer is the before/after pair, and half of that pair
lives here.

---

## What is here and why it was superseded

| Archived | Replaced by | Why the old run was withdrawn |
|---|---|---|
| `M16_v1_superseded/` | `Barshon's/M16/M16_v2/` | Indexed **one row per `.wav`** (851 whole recordings, tiled to 8 s, one diagnosis label each) and had **no train/test split at all** — the student was distilled over those rows and scored on the same rows. `student_accuracy = 0.9318` is a training-set number on a COPD-dominated corpus. The teacher was also loaded with `strict=False` inside a `try/except` that only printed a note. |
| `M18_old_superseded/` | `Barshon's/M18/M18_updated/` | Same 851-recording indexing, no split, and `load_base_model` swallowed every checkpoint error with `except Exception: pass` — so the sweep could run over a **randomly initialised network**. The committed sweep is consistent with exactly that: 0.0411 accuracy at pruning 0.0, below the 0.25 four-class chance line, beside a "best" of 0.9318 reached only once pruning collapsed the model onto the majority class. |
| `M20_old_superseded/` | `Barshon's/M20/M20_updated/` | Temperature fitted **and** ECE measured on the same 851 rows. Also carried `else: all_logits = torch.randn(100, 3)`, so a run that loaded no audio would have calibrated 100 random logit vectors and exported the result. |
| `M21_old_superseded/` | `Barshon's/M21/M21_Updated/` | `break` after the **first** annotation line per recording, then labelled the whole 8 s clip with cycle 1's label (920 rows for a 6898-cycle corpus). Reported accuracy 0.575 was last-epoch *training* accuracy. |
| `M2_17aug_run_v2_superseded/` | `Asif's/M2/m2_v4/` | Three corrected-split M2 runs existed (0.4139 / 0.4671 / 0.4720). `m2_v4` is the one `M12_v2` nominated and evaluated, so it is canonical; the other two are earlier passes of the same pipeline. |
| `M2_29aug_run_v3_superseded/` | `Asif's/M2/m2_v4/` | As above. |
| `M3_17aug_run_superseded/` | `Asif's/M3_v2/` and `Asif's/M3/29 aug run/` | Earliest corrected-split M3 pass (0.4490), superseded by two later runs that agree with each other (0.5132 / 0.5200). |
| `M14_v1_superseded/` | `Barshon's/M14/v2/` | v1's AUROC 0.4522 is below chance and is audit-critical. v2 is the run the paper cites. `compute_significance.py` still reads v1 from here, deliberately — the v1/v2 pair is the audit trail. |
| `M15_v4_superseded/`, `M15_v5_superseded/` | `Barshon's/M15/v6/` | Three versions of the cross-task scorer; v6 is the one M28 selected and the only one evaluated at patient level. |
| `M30_v1_withdrawn/` | `Asif's/M30_v2/` | Reported macro 0.8213 with **no committed confusion matrix**, so the official score cannot be recomputed — withdrawn as unverifiable. M30_v2 re-ran it on the official split and scored 0.5975, below M2 alone. |
| `M13cbm_single_seed_superseded/` | `Novelty Experiment/results/N6_physics_bottleneck.json` | The four bottleneck variants at **one seed**, with empty `efficiency` and `training_history`. N6 re-ran all four modes at five seeds and found the committed monotone ordering **inverts** — the single-seed `interpretability_cost_acc = 0.2326` must not be cited. |
| `M19_loose_export/` | `Barshon's/M19/results_M19.json` | `M19_metrics.json` held 4 loose fields instead of the §4 schema. Two exports per model with different numbers is how the M28 benchmark table drifted; the §4 file is now the only export. |

## The common root cause

All four ran on **real audio** — none of them fabricated their input. What they got wrong was the
*evaluation*: recording-level instead of cycle-level, and fitting and reporting on the same rows.
Each replacement run uses the corrected patient-independent official split (551/369 recordings,
0% patient leakage, verified by assertion), fits on TRAIN, and scores on TEST only.

## What is NOT here

- `Asif's/M22/` — still live. Its fallback-split result (official 0.6495) is the paper's current
  best model and the *before* half of the split-audit comparison against `M22_v2` (0.5602).
- `Asif's/M2/m2_v4/`, `Asif's/M3_v2/`, `Asif's/M3/29 aug run/` — the canonical runs, decided in
  `DECISION_2026-08-30_CONSOLIDATION.md` §3. (`SYNTHETIC_DATA_REMEDIATION.md` was deleted in
  0f7b68b; the consolidation record replaces it.)

## Do not

- Cite any number in this folder.
- Merge these into M28 — it now skips any path containing `Archive_Files` outright. The `EXCLUDED`
  dict stays for models that are live but withdrawn; it cannot separate two runs of the same id.
- Delete this folder to save space; it is a few MB and it is the audit trail.
