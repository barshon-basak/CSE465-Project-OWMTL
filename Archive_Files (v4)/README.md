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
| `M19_loose_export/` | `Barshon's/M19/results_M19.json` | `M19_metrics.json` held 4 loose fields instead of the §4 schema. Two exports per model with different numbers is how the M28 benchmark table drifted; the §4 file is now the only export. |

## The common root cause

All four ran on **real audio** — none of them fabricated their input. What they got wrong was the
*evaluation*: recording-level instead of cycle-level, and fitting and reporting on the same rows.
Each replacement run uses the corrected patient-independent official split (551/369 recordings,
0% patient leakage, verified by assertion), fits on TRAIN, and scores on TEST only.

## What is NOT here

- `Asif's/M22/` — still live. Its fallback-split result (official 0.6495) is the paper's current
  best model and the *before* half of the split-audit comparison against `M22_v2` (0.5602).
- The M2 / M3 multi-run folders — the canonical-run decision is still open, see
  `SYNTHETIC_DATA_REMEDIATION.md` item 1.

## Do not

- Cite any number in this folder.
- Merge these into M28 — Section 3's `EXCLUDED` dict keeps them out by model id.
- Delete this folder to save space; it is a few MB and it is the audit trail.
