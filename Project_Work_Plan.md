# Project Work Plan — OWMTL Project

**Replaces:** `Individual_Work_Plan.md` (archived at `Archive_Work_Plan/Individual_Work_Plan_ARCHIVED.md`).
**Why:** two things changed since the old plan was written. First, Dr. Khan's 2026 novelty guidance
is explicit — this isn't graded on model count or a basic classification pipeline, and novelty is
now the most important axis of the project, not an add-on. Second, the audit
(`Asif's/audit/PROJECT_AUDIT.md`) found that the old per-member split had produced a real cost: most
of what was tracked as "Member B done" turned out to be synthetic-data scaffolding, not results.
Organizing by lettered role stopped matching how the work was actually getting done.

**What replaces it:** a plan organized by **chunks of work** — what needs to be true, not who's
assigned to make it true. Anyone can pick up any unblocked chunk. Individual accountability (which
most programs still require) now attaches to *whichever chunk you actually did*, not to a role you
were assigned in week 1.

---

## 1. Read this first

**Priority order for anything you're about to build:**

1. `Novelty Search.md` — is there a novelty angle for this? As of 2026 this matters as much as
   whether the model runs at all.
2. `Model_Training_Reference.md` §0 — is the chunk you're about to touch already real, or does it
   need a rebuild first?
3. `Asif's/audit/audit_project.py` — run it. If your target model already exists but is flagged, fix
   *that* before adding anything new on top of it.
4. `Model_Training_Protocol.md` — the schema/metrics/checkpoint rules every model output must follow, whoever builds it.

If you only read one section of this document, read §3 (current chunk status) and §4 (what to pick
up next).

---

## 2. The two fixed ideas: novelty first, and depth over breadth

The old plan measured progress by how many of ~28 planned models existed. That's gone. What matters
now, in order:

1. **Is the core mechanism (Chunk D) real yet?** This is still the single most important open item —
   see §3.
2. **Does the project have 2–3 well-executed novelty contributions?** Not one, and importantly
   **not nine.** `Novelty Search.md` §4.0 fixes the set and explains the reasoning.
3. **Is everything claimed as "done" actually verified real?** A model the audit flags is not
   progress, no matter how sophisticated the notebook looks.

**Do not make this project buzzword-heavy.** Dr. Khan's novelty list is a menu of techniques the
department considers acceptable in 2026 — it is not a checklist to complete. Implementing all of
them would mean nine simultaneous changes on one dataset with 19 unknown patients, which cannot be
ablated apart; the paper would read as a technique list rather than a contribution, and no reviewer
could tell (nor could we) which change did anything. **Two or three additions we can ablate cleanly
and defend individually is a stronger paper than nine we cannot.** The project's strongest
contribution is a *depth* claim — the statistically defensible evaluation redesign — and piling on
surface area actively works against it.

Everything else — hitting a specific model count, covering every entry in the old 28-model list,
running every member's assigned augmentation, implementing every technique on the supervisor's
list — is explicitly **not** a goal.

---

## 3. Current chunk status

Full detail and per-model specs live in `Model_Training_Reference.md` §1–§2. Summary:

| Chunk | What it is | Status | Blocks |
|---|---|---|---|
| **A** — Shared foundations | Preprocessing, protocol, audit tooling | ✅ Done | — |
| **B** — Sound-event backbone | M1–M4, M12 (backbone = M2) | ✅ Done, real | Chunks D onward use M12's checkpoint |
| **C** — Open-set baselines | M6 (real negative), M29 (real, sets the bar at AUROC 0.6466) | ✅ Done, real | Chunk D's target metric |
| **D** — Disease head + core mechanism | M13, M15, M17 | ✅ **Done (M13 v4, M15 v4, M17 v2 all real & audit-clean)** | Chunks F, G |
| **E** — Novelty layer | now **2** selected items, not 3 | ⚠️ **Both surviving items are in trouble.** M13 prototypical head is real. M14 conformal is real but test AUROC 0.4809 (below chance) with 0% unknown detection at the 95% coverage point. ❌ M30 fusion **dropped** — it failed its own §4.0 admission test (0.5975 vs M2-alone 0.6098 on identical test cycles; `Asif's/M30_v2/`). Per §4.0 a deferred item should be **swapped in**, not added alongside. | — |
| **F** — Generalization & compression | M16, M18, M19 | ✅ **Done (M19 OOD transfer completed; M16 Knowledge Distillation & M18 Pruning/Quantization Sweep generated for real audio)** | Requires Chunk D real first |
| **G** — Trust & calibration | M7–M11, M14, M20 | ✅ **Done (M14 conformal calibration v2 & M20 temperature scaling ECE calibration generated)** | Requires Chunk D real first |
| **H** — Reporting | M28-equivalent merge | ✅ **Done (M28 Master Merge completed: master_results_summary.json, 4 publication figures, and LaTeX tables generated)** | Requires everything else to be real |

**The headline finding, in one sentence:** the paper's core novelty claim (cross-task consistency
detecting unknown diseases) has never actually been tested — the notebook that's supposed to test it
generates random noise instead of loading ICBHI audio. Fixing that is worth more than anything else
in this document.

---

## 4. What to pick up next

In priority order:

1. **Rebuild Chunk D on real data — and build the disease head as a prototypical/meta-learning head
   while you're in there.** These are one task, not two. M13 has to be rewritten regardless (it stops
   training at epoch 1 — diagnose that before assuming more epochs will help), and building it as a
   prototypical head costs roughly the same effort while answering Attack 5 (URTI n=14) for free.
   Doing them sequentially would be strictly more work. Then M15, evaluated at **patient level** (not
   cycle level — the broken version inflated N by evaluating cycles), against the real bar M29
   established (AUROC 0.6466), not M6's broken 0.4516.
2. **Then conformal calibration of the disagreement score** — selected novelty item 2. It's a
   post-hoc wrapper, so it needs a real M15 to wrap, but it's cheap once that exists and it answers
   Attack 6 (no formal guarantee on the threshold).
3. **In parallel and buildable today: the M2+M3 feature-fusion ensemble** (optional item 3). Both
   checkpoints are real and already exist, so it needs no new base training and nothing from Chunk D.
   Drop it silently if it doesn't measurably beat both backbones alone.

   ⚠️ **Do not pick up other novelty items just because they're cheap.** Curriculum learning, LoRA,
   GradNorm, token pruning and the rest are all documented in `Novelty Search.md` §4 and all
   deliberately **deferred** (§4.0). Cheapness is not the selection criterion — answering a named
   reviewer attack is. If a selected item fails, swap a deferred one in; don't accumulate.
4. **Chunks F and G stay parked** until D is real. Don't spend time compressing or stress-testing a
   mechanism that hasn't been shown to work.

---

## 5. Shared responsibilities (no longer tied to a role)

These don't belong to any one chunk and don't need a fixed assignment — whoever has bandwidth:

- **Manuscript sections** (Abstract, Introduction, Related Work, Discussion) — co-written, since the
  novelty claim depends on the pieces fitting together as one narrative.
- **Statistical-protocol compliance** — LOPO instead of k-fold for small groups, no bootstrap CIs
  under n=15, patient-independent splits everywhere (`Model_Training_Protocol.md` §1). Cross-check
  each other's evaluation code before merging results — this is exactly the kind of check that would
  have caught the synthetic-data problem earlier if it had been done systematically. The audit tool
  now does this automatically; run it on anything before calling it done.
- **Faculty checkpoints** — whoever is available attends reviews with Dr. Khan, since these are the
  points where the whole project's direction can shift (as it just did, twice: once on synthetic
  data, once on novelty priority).

---

## 6. Individual accountability, without pre-assigned roles

Most programs still require each person to be able to defend their own contribution. That doesn't
require the old A/B/C/D split — it just means: **whichever chunk you actually built, you should be
able to answer for it.** The old defense questions still apply, just re-anchored to chunks instead of
letters:

- **Whoever worked on Chunk B (backbone)** — "Why this backbone over the alternatives, and what does
  the ablation show about the accuracy/compute tradeoff?" (Answerable now — `Asif's/M12/`.)
- **Whoever works on Chunk D (core mechanism)** — "Why does cross-task disagreement outperform (or
  not) the trivial baselines in M29, and what does the forgetting curve tell us about staged
  learning?"
- **Whoever works on Chunk F (generalization/compression)** — "How much does performance degrade on
  genuinely unseen populations, and what is the compression/accuracy tradeoff?"
- **Whoever works on Chunk G (trust/calibration)** — "When the model flags a patient as unknown, how
  often is that flag wrong, and does confidence stay honest under an unseen population?"
- **Whoever works on Chunk E (novelty)** — "What does this add beyond the base mechanism, and how do
  you know it isn't just decoration?" (This is the one question the old plan didn't have to answer,
  because novelty wasn't the centerpiece before.)

Each answer should draw on a results table the audit has actually verified — not on a notebook that
looks complete.

---

## 7. Sequencing note

The old plan's "one real dependency" (backbone → disease head) still holds and is unchanged: Chunk D
needs Chunk B's checkpoint (M12/M2). Everything else genuinely can run in parallel. The difference
from the old plan isn't the dependency graph — it's that progress is no longer measured by covering
every box in a 28-model checklist, and that a real result on Chunk D matters more than any amount of
work on Chunks F/G while D is still broken.

---

## 8. Final notes

- Codebase organization is unchanged: everyone's individual folder (`Asif's/`, `Barshon's/`, etc.)
  still holds that person's notebooks and results — the folder-per-person structure is just storage,
  not a claim about who's allowed to touch which chunk.
- `Asif's/CLAUDE.md` has the standing instructions for AI-assisted work in that folder — it points to
  `Novelty Search.md` and this file already.
- Revisit this plan whenever the real-data status of Chunk D changes materially, or if Dr. Khan's
  novelty guidance shifts again — both have already forced one rewrite of the project's planning
  documents this term, so treat the plan as provisional, not fixed.
