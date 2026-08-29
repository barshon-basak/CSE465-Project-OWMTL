# CLAUDE.md — Asif's workstream

Instructions for any Claude session working in this folder.

> **2026-08-05:** the project retired its old per-member (A/B/C/D) role split in favor of a
> chunk-based plan — see `Project_Work_Plan.md` and `Model_Training_Reference.md` §1. This file's
> earlier title referenced "Member A"; that framing is gone. Asif has been doing what was Chunk B
> (backbone) and Chunk C (open-set baselines) work, but nothing here is a fixed lettered assignment
> anymore — pick up whichever chunk is next and unblocked, per `Project_Work_Plan.md` §4.

---

## Standing instructions

**1. Always end with next steps.** Every response finishes with a short, concrete "What to do next"
— numbered, specific, and ordered by value. Say what *Asif* does (run this on Colab, send this to
Barshon, ask Dr. Khan about X) versus what Claude should do next session. Never end a response
without it.

**2. Always work from the core documents.** Before building or changing anything, read:
- [`Model_Training_Protocol.md`](../Model_Training_Protocol.md) — the §4 results-JSON schema, §3
  metric suite, §2 preprocessing constants, §11 checkpoint/persistence rules. Outputs must conform.
- [`Model_Training_Reference.md`](../Model_Training_Reference.md) — what each model is, what it
  requires, and which chunk (§1) it belongs to. §0 has the current real-vs-synthetic status —
  regenerate it by running the audit rather than trusting a cached read.
- [`Project_Work_Plan.md`](../Project_Work_Plan.md) — chunk status and what to pick up next, with no
  fixed role assignment.

Cite them by line when a decision depends on them (e.g. `Model_Training_Reference.md:169`). Line
numbers drift whenever these files are restructured — re-grep for the claim before trusting an old
citation, including ones in this file. The team has already been burned once by a justification
document that contradicted its own data — so ground claims in the files, not memory.

**3. Always keep looking for novelty — but depth over breadth.** Check work against
[`Novelty Search.md`](../Archive_Files%20(v2)/Novelty%20Search%20v2.md) *(archived — moved to `Archive_Files (v2)/`)* — the single active reference (the old v1/v2/v3 files
are consolidated into it; full literature search archived in `Archive_Work_Plan/`).

**The project deliberately implements only 2–3 novelty items, not Dr. Khan's whole list.** See
`Novelty Search.md` §4.0 for which ones and why. Implementing everything would produce a
buzzword-heavy paper that does nine things shallowly instead of three things well — reviewers
penalize that, and it's the single easiest way to lose a Q1 submission. **Never propose adding a
novelty item without saying what it displaces or why the current set is insufficient.** If asked to
"add more novelty," push back and ask what's wrong with the selected set first.

When a result lands, ask: does this strengthen a selected novelty claim, kill one, or open a new one?
Flag it explicitly. A negative result that is *defensible* is worth more to a Q1 submission than a
positive one that isn't. The project's strongest contribution is the **statistically defensible
evaluation redesign**, not the raw mechanism — keep that in view.

**4. Never commit. Asif commits.** Do not run `git commit`, `git push`, `git merge`, `git rebase`,
or anything else that writes to git history — not even when the work is obviously finished, not even
when asked to "wrap up" or "finalize." Stage nothing. When work is done, say so and list the changed
files; Asif reviews and commits. Read-only git (`git status`, `git diff`, `git log`, `git show`) is
fine and encouraged for understanding state.

---

## Where the project actually stands

Last full audit: run `python3 "Asif's/audit/audit_project.py"` (no GPU, no dataset, ~1 s).

| Status | Models |
|---|---|
| ✅ Real data, verified | **M1, M2, M3, M4, M6, M11, M12, M13, M15, M17, M19, M22_v2, M24, M29, M30_v2, M31–M37, M39** |
| 🟡 Real audio, provenance unproven + no train/test split | M16, M18, M20, M21 |
| 🔴 Synthetic data — not results | *(none)* |

**The synthetic-data finding is closed (2026-08-29).** The line that stood here — *"every model
downstream of M12 except M6 and M29 runs on `torch.randn` data"* — no longer holds. M11, M13, M15,
M17, M19 and M24 were re-run on real ICBHI audio and the audit now reports **zero
`synthetic_data_not_real_dataset` findings**. The core cross-task mechanism **has** now been
evaluated on real data, and it lost: M15 v6 scores 0.5747 against M29's zero-training Energy
baseline at 0.6466.

**What replaced it** is narrower, and tracked in `SYNTHETIC_DATA_REMEDIATION.md` at the repo root:
M28's hardcoded benchmark table, M15 v6's fabricated-label fallback and the silent all-zero
spectrogram fallback in 39 files are all fixed; M16/M18/M20/M21 remain open because their notebooks
carry no committed outputs and evaluate 851 whole recordings with no train/test split.

**M6 is real, and it's a genuine negative:** OpenMax `auroc = 0.4516`, `unknown_recall = 0.0255` on
real audio with real patient counts (72 train / 32 known-test / 19 unknown-test). Citable as-is.

**M29 is real, and it changes the target.** Four trivial post-hoc OOD scores (MSP, entropy, energy,
Mahalanobis) on the frozen M12 backbone, patient-level, on the exact real split
(`Model_Training_Reference.md:156` — 104 known / 19 unknown patients). Best baseline
(**Energy, AUROC 0.6466**) comfortably beats M6's 0.4516 using zero training. Consequence: once M15
is rebuilt on real data, "beats M6" is a bar a zero-training baseline already clears — the paper's
novelty claim has to beat **0.6466**, not 0.4516. Full numbers in `Asif's/M29/README.md`.

### Asif's own models (all real, all verified)

| Model | Metric | Value | Params | Size |
|---|---|---|---|---|
| M2 — tuned CNN (`2D_CNN_5Block_w48_do0.4`) | **official** ICBHI / Acc / F1 | **0.6138** / 0.6138 / 0.5238 | 3.6 M | 13.9 MB |
| M3 — MobileNetV2 | **official** ICBHI / Acc / F1 | **0.5895** / 0.5915 / 0.4904 | 2.2 M | 8.7 MB |
| M22 — M3 + SpecAugment | **official** ICBHI / Acc / F1 | **0.6495** / 0.6524 / 0.5253 | 2.2 M | 8.7 MB |
| M12 — backbone decision | selects **M2** | | | |
| M29 — open-set baselines on M12 | best AUROC (Energy) | **0.6466** | 3.6 M (frozen, M2's) | — |
| M30 v2 — M2+M3 gated fusion | **official** ICBHI | **0.5975** — ❌ loses to M2 alone (0.6098) | 8.0 M | — |

**⚠️ ICBHI metric:** always report `icbhi_score_official` (the ICBHI 2017 challenge metric), never
the project's legacy `icbhi_score` macro variant — the latter runs ~0.11 higher and is not
comparable to published work. Both are now in every results JSON; see
`Asif's/audit/ICBHI_SCORE_AUDIT.md` and `Model_Training_Protocol.md` §3. Published ICBHI SOTA on the
official split is ~0.60-0.65, so M2's 0.6138 sits at the literature level.

M12's caveat worth remembering: M2 beats M1 on ICBHI by only 0.0046, *inside* the 0.0129 CV
tolerance. The real evidence for M2 is accuracy (+0.0731) and macro-F1 (+0.0394). Don't lead with
the ICBHI number. (The M2-M3 margin, 0.0243, is identical under both metrics, so M12's decision is
unaffected by the metric correction.)

---

## Conventions established in this workstream

- **Notebooks are generated, not hand-edited.** Write a `gen_*.py` that emits the `.ipynb`, so
  regeneration is reproducible. Watch the nested-triple-quote trap: use `code(r'''…''')` wrappers
  with `"""docstrings"""` inside, never the reverse.
- **Test before shipping.** Build a synthetic mini-ICBHI corpus and execute every cell end-to-end on
  CPU before handing a notebook to Colab. M2 and M3 each passed 55/55 functional + 17/17 recovery
  checks this way, and both then ran on Colab with **zero code changes**.
- **Documents that state numbers are generated from those numbers.** `M12_backbone_justification.md`
  is written by the notebook from the loaded JSON, so it structurally cannot drift.
- **Spectrogram cache** at `/content/owmtl_spec_cache` (local disk, never Drive) — 8.7× speedup over
  recomputing librosa features each epoch. Safe only for un-augmented runs.
- **Checkpoints** are gitignored (`.gitignore:27`). Put the `.pth` on Drive and commit a
  `Best_model_pth_file Link.txt` next to the notebook — the convention Barshon used for M4.
- **Verify handoffs.** M12 re-loads the winning checkpoint and confirms it reproduces the reported
  metrics before declaring it frozen.
- **Mark cited numbers as citations.** When a results JSON reports another model's number for
  context (not as this model's own claim), name the field with the literal word `reference` (e.g.
  `reference_M6_openmax`). The audit tool keys on that word to distinguish "M29 cites M6's known
  problem" (INFO) from "M15 computes an uncritical ratio against a broken baseline" (CRITICAL) —
  see `_is_cited_reference_field` in `audit_project.py`. Without the naming convention the two look
  identical to the tool.

---

## Traps this project has already hit

Check for these in any new work, and keep the audit's checks in mind as a written record of them:

1. **Synthetic data committed as results** — closed 2026-08-29, but it took four rounds to find
   every form it took: a `torch.randn` Dataset, a hand-typed benchmark table, a `pid % 3`
   label fallback, and an `except: return np.zeros(...)` that turned an unreadable wav into a
   silent all-zero cycle. **A fallback that substitutes data is the same bug as a synthetic
   Dataset.** Make the missing input a hard failure. See `SYNTHETIC_DATA_REMEDIATION.md`.
2. **Metrics of exactly 1.0** — M21. Always ask what the evaluation split actually is.
3. **A metric constant across a sweep** — M18, flat 0.28 over a 62× parameter range.
4. **Ratios against sub-chance baselines** — M15's "beats M6 by 36.9%" where M6 = 0.4516.
5. **Best epoch = 1** — M13, twice. The model never trained.
6. **Cycle-level evaluation where the protocol says patient-level** — M15 used 518/101 cycles where
   `Model_Training_Reference.md:178` specifies patient-level LOPO over 104/19 patients.
7. **Justification text drifting from its own data** — the original M12 named M1 the winner in its
   header while arguing for AST in three of four sections.

---

## Useful paths

```
Asif's/M2/     tuned CNN — results_M2.json, best_model.pth, sweep table
Asif's/M3/     MobileNet/DenseNet — results_M3.json, sweep with efficiency frontier
Asif's/M12/    backbone decision — justification, audit trail, robustness check
Asif's/M29/    open-set baselines on the M12 backbone — the real bar for M15 (AUROC 0.6466)
Asif's/audit/  audit_project.py + PROJECT_AUDIT.md  (run this first in a new session)

../Project_Work_Plan.md           chunk status + what to pick up next (root level, no fixed roles)
../Archive_Work_Plan/              retired role-based planning docs, historical only
```

**M29 is now in `Model_Training_Reference.md`** (§2.8, `rejection_method` ablation group alongside
M6 and M15) — added when the reference was restructured into chunks on 2026-08-05.

**M21/M22/M23** (SpecAugment variants of M2/M3/M4) are optional-stretch items under Chunk B — the
backbone decision is already settled on real data, so an augmentation ablation on top of it doesn't
change any conclusion. Don't spend GPU hours there before Chunk D (the core mechanism) is real —
that's still the actual bottleneck, and it's no longer any one person's fixed responsibility to fix.
Given M29's result, the case for prioritizing a *real* M15 rebuild over any of M21–M23 is stronger
than ever: the paper's headline mechanism has never been evaluated on real data at all.
