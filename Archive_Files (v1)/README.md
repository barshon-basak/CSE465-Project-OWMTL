# Archive — Work Plan (pre-2026-08-05)

These are the role-based (Member A/B/C/D) versions of the project's planning documents, snapshotted
on 2026-08-05 before the team retired the per-member split in favor of a chunk-based plan.

## Why this exists

Dr. Khan's 2026 novelty guidance made two things clear: the project is no longer graded on model
count ("he doesn't expect anything like a basic classification project or a 10-model-run project"),
and role-based individual assignment isn't the organizing structure anymore. The audit
(`Asif's/audit/PROJECT_AUDIT.md`) also found that the role split had produced a real cost — most of
"Member B"'s deliverables turned out to be synthetic data, not real results — which made "who owns
which lettered role" a less useful frame than "what real, verified work exists and what chunk of
work is next."

## What's here

| File | What it was |
|---|---|
| `Individual_Work_Plan_ARCHIVED.md` | The full CRediT-role-based work split (Members A–D, timeline, defense talking points). Superseded by `Project_Work_Plan.md` at the repo root. |
| `Model_Training_Protocol_ARCHIVED.md` | The results-JSON schema and training protocol, as it stood with `member`/`member_name` fields and a per-member augmentation table. Superseded by the root `Model_Training_Protocol.md`, which keeps the same schema and rules but drops the role framing. |
| `Model_Training_Reference_ARCHIVED.md` | The 28-model, 4-member execution runbook organized into "Stages" with an "Owner" column. Superseded by the root `Model_Training_Reference.md`, restructured into chunks with no ownership column, and updated to reflect what the audit found real vs. synthetic. |

## Still valid, not archived

The three novelty files were later (same day) consolidated into a single root-level
`Novelty Search.md`, with the full literature search archived here as
`NOVELTY_SEARCH_v1_ARCHIVED.md` and `Novelty_Search_v2_ARCHIVED.md`. Those two still hold the
complete paper-by-paper review, 50+ scored candidate ideas, 20 combined directions, and the
cross-domain mining pass — none of it wrong, just more exploratory breadth than the working project
needs day to day. The active `Novelty Search.md` is the project's highest-priority document.

`Research_Guideline_FULL_ARCHIVED.md` is the pre-trim copy of the root `Research_Guideline.md`,
kept before its outdated timeline/checklist sections were removed.

## If you need something from these files

The technical content (§3 metric suite, §4 results JSON schema, §11 checkpoint safety protocol,
model architectures/purposes) carried over into the active documents largely unchanged — only the
role/ownership framing was removed. Check the active root-level files first; come here only for the
original role-based framing or historical line-number citations from work completed before
2026-08-05 (e.g., citations inside `Asif's/M2/README.md`, `Asif's/M12/`, `Asif's/M29/` that predate
this restructure may point to line numbers in these archived files, not the current ones).
