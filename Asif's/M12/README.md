# M12 — Final Backbone Selection

**Owner:** Asif (Member A) · **Requires:** M2, M3, M4 · **Status:** ✅ decided and verified

> **⚠️ Metric note (2026-08-13) — decision unchanged.** The ICBHI column below is the project's macro
> variant, not the ICBHI 2017 challenge metric (`Asif's/audit/ICBHI_SCORE_AUDIT.md`). Official
> scores: **M2 0.6138 > M3 0.5895**. The M2–M3 margin is **identical under both metrics (0.0243)**,
> so the selection, the CV-tolerance comparison, and the conclusion all stand as written.

## The decision

**M2 (tuned 2D CNN, `2D_CNN_5Block_w48_do0.4`) is the frozen shared backbone.**

| Model | Architecture | Acc | Macro-F1 | Se | Sp | ICBHI | Params | Size (MB) | ms/sample |
|---|---|---|---|---|---|---|---|---|---|
| M1 *(reference only)* | `2D_CNN_4Block` | 0.5407 | 0.4844 | 0.5801 | 0.8561 | 0.7181 | 421,732 | 4.85 | 1.85 |
| **M2 ← selected** | `2D_CNN_5Block_w48_do0.4` | **0.6138** | **0.5238** | **0.5817** | **0.8638** | **0.7227** | 3,627,476 | 13.86 | 2.94 |
| M3 | `mobilenet_v2` | 0.5915 | 0.4904 | 0.5431 | 0.8537 | 0.6984 | 2,228,996 | 8.74 | 5.42 |
| M4 | `AST_pretrained` | 0.5528 | 0.4109 | 0.4385 | 0.8334 | 0.6359 | 86,385,668 | 329.54 | 86.74 |

M2 wins on **every** metric while being 24× smaller and 30× faster than the AST.

## How to re-run it

```bash
cd /path/to/CSE465-Project-OWMTL
jupyter nbconvert --execute --to notebook --inplace "Asif's/M12/M12_backbone_selection.ipynb"
```

Unlike M2/M3 this needs **no GPU, no Colab, no dataset** — it is pure analysis over the committed
`results_M*.json` files and runs locally in seconds. It auto-discovers the repo root, so it works
from anywhere inside the tree. Re-run it any time a candidate's results change; every output
regenerates from the data.

## Why the decision is defensible

**The rule was fixed in code before the numbers were read** (Cell 5 defines it, Cell 6 applies it),
because `Model_Training_Reference.md:243` demands *"a deliberate, documented efficiency-vs-accuracy
tradeoff decision, not just 'highest accuracy wins'"*:

1. **Eligibility** — candidates are M2, M3, M4 (`:114`). M1 excluded (`:144` — throwaway
   checkpoint, explicitly superseded by M2).
2. **Rank** by ICBHI score.
3. **Separation** — the leader wins outright only if its margin beats its own cross-validation std.
4. **Efficiency tiebreak** — among tied candidates, prefer the more deployable one.
5. **Sanity gate** — the winner must not lose to another candidate on *both* accuracy and macro-F1.

**How it resolved:** M2 led M3 by 0.0243 ICBHI against a 0.0129 tolerance (M2's own patient-grouped
CV std), so step 3 settled it outright — no tiebreak needed. The sanity gate passed: M2 also has the
best accuracy and best macro-F1.

That step-3 threshold is a real measurement from `M2_hp_sweep.json`, not an assumed tolerance. Had
the margin been smaller, the efficiency tiebreak would have fired and M3 (2.2 M params / 8.7 MB)
would have won instead — the rule was genuinely capable of returning a different answer.

## The checkpoint was verified, not just cited

`Model_Training_Reference.md:244` makes the deliverable *"one frozen checkpoint + a short written
justification"*. Section 6 re-loads `Asif's/M2/best_model.pth` and confirms it is the file that
produced the table:

```
[x] best_score matches results JSON        (checkpoint 0.7227 vs reported 0.7227)
[x] model_state present                    (34 tensors)
[x] parameter count consistent             (state_dict 3,630,457 vs params 3,627,476,
                                            +2,981 = BatchNorm buffers, expected)
```

This is the check that catches the failure where the table says one thing and the file Member B
actually receives is something else.

## Deliverables

```
results_M12.json                 §4 schema + a `decision` block with the full audit
M12_backbone_justification.md    the written defense (§244) — generated from the data
decision_audit.json              every rule step, its inputs and its outcome
backbone_comparison.png          four-way metric comparison
efficiency_vs_performance.png    the accuracy-vs-cost tradeoff (the paper figure)
M12_handoff_bundle.zip           all of the above
```

The justification is **generated, never hand-written**. The previous M12 justification drifted from
its own data — it declared M1 the winner in the header while arguing for AST in three of its four
rationale sections, and quoted M1's parameter count (421,732 / 4.85 MB) as the AST's. Generating the
document from the loaded JSON removes that failure mode structurally.

## What this changes for the team

Every downstream model that consumes "the M12 backbone" should be built on **M2**:

- **Member B** — M13 (disease head) → M15 (cross-task consistency) → M17 (OWL Stage 2)
- **Member C** — M16 / M18 (CQKD distillation; M2's 3.6 M / 13.9 MB is a sensible teacher size)

The earlier M12 pass selected **M1**, decided when only two models existed. M2 improves on it by
+0.0046 ICBHI, **+0.0731 accuracy** and **+0.0394 macro-F1**. Anything already fit against M1 is
sitting on a strictly weaker encoder.

`Barshon's/M12/M12_backbone_justification.txt` states a different winner and should be superseded —
leaving both in the repo invites the wrong one being cited.

## Honest caveats for the write-up

- **Best-checkpoint selection used the test split** for all of M1–M4 alike. The comparison between
  them is fair, but no absolute number here is a clean held-out estimate. The k-fold spreads in
  `M2_hp_sweep.json` / `M3_hp_sweep.json` are the honest generalisation figures.
- **M4's row is transcribed, not loaded.** `results_M4.json` is not in the repo — it lives in
  Barshon's Drive, with only the link committed. The values come from the executed output of
  `Barshon's/M4/M4_ast_backbone.ipynb`. Ask him to commit the JSON, then re-run this notebook; it
  picks the file up automatically and discards the transcription.
- **M4's size is the state_dict estimate (329.54 MB), not the 988.85 MB its handoff cell printed** —
  that figure was the full checkpoint including optimiser state, which is not what §3 asks for.
  Using it would have overstated the AST's disadvantage roughly threefold.
- **The AST finishing last is a real finding, but check it got a fair shot.** M4 trained 40 epochs
  with its best checkpoint at epoch 10, which reads as early overfitting rather than undertraining.
  Worth a sentence in the paper either way, since it contradicts the Tri-MTL precedent the proposal
  expected to hold.
