# OWMTL — Respiratory Sound Analysis on ICBHI 2017

**Course:** CSE465, North South University · **Target:** Q1 biomedical signal-processing venue
**Status as of 2026-08-16:** pivoted to the reliability/evaluation contribution — see
[`DECISION_2026-08-16_PIVOT.md`](DECISION_2026-08-16_PIVOT.md).

> ### Read these three first
> 1. **[`DECISION_2026-08-16_PIVOT.md`](DECISION_2026-08-16_PIVOT.md)** — what the project is now,
>    and why. Gate G2 failed twice; the concept bottleneck is retired as the headline.
> 2. **[`PAPER_OUTLINE.md`](PAPER_OUTLINE.md)** — the paper being written, with an evidence ledger
>    separating what exists today from what is still pending.
> 3. **[`Model_Training_Protocol.md`](Model_Training_Protocol.md) §1** — the ten hard rules. The
>    metric and split rules are non-negotiable and were added because we broke them.

---

## Three things to know before touching any number

These were all found by auditing our own work, and each one silently invalidated results:

| | The trap | The rule now |
|---|---|---|
| 1 | The reported "ICBHI score" was `(recall_macro + specificity_macro)/2` — **inflated by ~0.11** on average, 0.22 at worst | Report **`icbhi_score_official`**. Regenerate with `Asif's/audit/icbhi_score_audit.py` |
| 2 | The "official 60/40 split" was a `pid ≤ 111` fallback — **11 test patients, 7.1% of cycles** | Use `Asif's/audit/official_split.py`. A notebook that can't find the split file must **raise**, never fall back |
| 3 | The real official split is **not patient-independent** — patients 156 and 218 sit on both sides | Same loader; default policy reassigns them to train |

Run `python3 "Asif's/audit/audit_project.py"` before trusting any status claim in any document.

---

## Current documents

| File | What it's for |
|---|---|
| `DECISION_2026-08-16_PIVOT.md` | **The current direction.** Decision record for the G2 pivot |
| `PAPER_OUTLINE.md` | **The paper.** Structure + evidence ledger + risk table |
| `Model_Training_Protocol.md` | Results-JSON schema, metric definitions, the §1 hard rules |
| `Model_Training_Reference.md` | Per-model specs (M1–M39) and status |
| `Project_Work_Plan.md` | Chunk status |
| `OWMTL_Decision_Roadmap (v3).md` | The G0–G7 gate roadmap. G2's branch (b) is the one we took |
| `OWMTL_Week0_Go_Memo.md` | G0/G1/G4 outcomes + protocol freeze |
| `CLINICIAN_LABELING_PACK.md` | The clinician listening study — protocol, delivery, analysis |
| `Papers/HUMAN_BENCHMARKS.md` | **Human performance reference points.** Read before interpreting any concept AUROC |

## Code

| Path | What |
|---|---|
| `Asif's/audit/` | `audit_project.py`, `icbhi_score_audit.py`, `official_split.py`, `check_device_structure.py` |
| `Asif's/Statistics/` | `owmtl_scores.py` — canonical score format + paired DeLong / McNemar / bootstrap |
| `Asif's/engine/` | Concept extractors + tests, clinician listening-pack builder, LaTeX handouts |
| `Asif's/M39/` | Gate G2 — concept extraction & validation (**FAILED**, twice; that is the result) |
| `Asif's/M2/ M3/ M12/ M22/ M29/ M30_v2/ M38/` | Backbones, backbone selection, augmentation, open-set baselines |

**Convention:** notebooks are *generated* by a `gen_*.py`, never hand-edited, and tested end-to-end
on a synthetic corpus before they touch Colab.

---

## Where the older documents went

The strategy documents turned over several times. Nothing was deleted; earlier versions are
archived, and citations to them elsewhere in the repo are annotated with their new location.

| Archive | Contains |
|---|---|
| `Archive_Files (v1)/` | Original proposal, protocol/reference, novelty search v1–v2 |
| `Archive_Files (v2)/` | `Novelty Search v2.md`, `Project_Proposal_v2.md`, `Research_Progress_Report till v2.md`, `Novelty_Reassessment of v2.md` |
| `Archive_Files (v3)/` | ACBD revalidation, novelty gap analysis, build plan, fix triage — all superseded by the 2026-08-16 pivot. Plus `scratch/` |

**Most-cited moved file:** `Novelty Search.md` → `Archive_Files (v2)/Novelty Search v2.md`. Its
§4.0 scope rule (2–3 novelty items, *swap don't accumulate*) still applies and is still worth
reading; the specific selected items are superseded.

---

## Where the project actually stands

Honest summary, because several documents in the history overstate it:

- **No working mechanism.** Cross-task disagreement tied a trivial baseline; the gated fusion
  ensemble failed its own admission test; the conformal wrapper sits below chance; the concept
  extractors failed Gate G2 twice.
- **Open-set detection is unresolved at n=19** — every AUROC's 95% CI includes chance.
- **Corrected classification numbers sit at the published ICBHI level** (~0.60–0.65 official),
  not above it.
- **What is real:** the corrected evaluation protocol, a pre-registered gate that actually failed,
  the human-benchmark reliability context, a clinician study in flight, and released tooling.

That is the contribution. It is smaller than the project started with, and it is the first claim
the evidence supports.
