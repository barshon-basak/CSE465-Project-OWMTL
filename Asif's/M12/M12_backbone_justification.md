# M12 — Backbone Selection Justification

**Decision:** `M2` — CNN Baseline (Tuned, Final) (`2D_CNN_5Block_w48_do0.4`) is selected as the final frozen shared backbone.

**Author:** Asif (Member A) &nbsp;|&nbsp; **Date:** 2026-08-03 &nbsp;|&nbsp; **Rule version:** 1.0

> Generated programmatically from the candidates' `results_M*.json` files by `M12_backbone_selection.ipynb`. Every number below is read from those files, so this document cannot drift from the data it describes.

> ### ⚠️ Metric note (added 2026-08-13) — **the decision is unchanged**
>
> The `ICBHI` column below is the project's macro variant, `(recall_macro + specificity_macro)/2`,
> not the ICBHI 2017 challenge metric. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`. On the official
> metric the candidates score **M2 0.6138 > M3 0.5895**, and M4 has no committed results JSON so it
> cannot be recomputed (its row here is transcribed from notebook output — see
> `decision.comparison_table.M4.source`).
>
> **The selection is unaffected.** The M2–M3 margin is **identical under both metrics (0.0243)**, so
> the ranking, the comparison against the 0.0129 CV tolerance, and the separation conclusion all
> hold exactly as written. No re-decision is required.
>
> This banner is hand-added rather than regenerated because the numbers below are correct *as the
> macro metric* — they are mislabelled, not wrong. When `M12_backbone_selection.ipynb` is next run,
> it should emit `icbhi_score_official` alongside and this note can be dropped.

---

## 1. Candidates

`Model_Training_Reference.md:114` requires M2, M3 and M4 as the inputs to this decision.

| Model | Architecture | Acc | Macro-F1 | Se | Sp | ICBHI | Params | Size (MB) | ms/sample |
|---|---|---|---|---|---|---|---|---|---|
| **M1** *(reference only)* | `2D_CNN_4Block` | 0.5407 | 0.4844 | 0.5801 | 0.8561 | 0.7181 | 421,732 | 4.85 | 1.85 |
| **M2** **← selected** | `2D_CNN_5Block_w48_do0.4` | 0.6138 | 0.5238 | 0.5817 | 0.8638 | 0.7227 | 3,627,476 | 13.86 | 2.94 |
| **M3** | `mobilenet_v2` | 0.5915 | 0.4904 | 0.5431 | 0.8537 | 0.6984 | 2,228,996 | 8.74 | 5.42 |
| **M4** | `AST_pretrained` | 0.5528 | 0.4109 | 0.4385 | 0.8334 | 0.6359 | 86,385,668 | 329.54 | 86.74 |

All rows share the same evaluation setup — ICBHI 2017, official patient-independent 60/40 split, cycle-level, 4-class sound event, inverse-frequency class-weighted `CrossEntropyLoss`, no augmentation, seed 42 — so the comparison is free of the loss-formulation confound `Model_Training_Reference.md:243` warns against.

### Why M1 is excluded

`Model_Training_Reference.md:144` is explicit that M1 is a throwaway: *"treat this as a throwaway/reference checkpoint — the real reported numbers come from M2 (CNN baseline, tuned)."* It is reported above for context and excluded from the decision. In any case M2 dominates it on every reported metric, so its exclusion does not change the outcome.

---

## 2. Decision rule

`Model_Training_Reference.md:243` requires *"a deliberate, documented efficiency-vs-accuracy tradeoff decision, not just 'highest accuracy wins'"*. The rule below was fixed in code before the numbers were read:

1. **Eligibility** — candidates are M2, M3, M4; M1 excluded.
2. **Rank** by ICBHI score, the field-standard primary metric.
3. **Separation** — the leader wins outright only if its margin over the runner-up exceeds the leader's own cross-validation standard deviation, measured across patient-grouped folds.
4. **Efficiency tiebreak** — among statistically tied candidates, prefer the more deployable one (params → size → latency).
5. **Sanity gate** — the winner must not lose to another candidate on *both* accuracy and macro-F1.

---

## 3. How the rule resolved

**Ranking by ICBHI:** M2 (0.7227) > M3 (0.6984) > M4 (0.6359)

**Separation test:** M2 leads M3 by **0.0243** ICBHI. The tolerance is M2's cross-validation std of **0.0129** (measured across patient-grouped folds in its own hyperparameter sweep).

Because 0.0243 > 0.0129, the lead is larger than the fold-to-fold noise of the architecture itself. **M2 is statistically separated and wins outright** — the efficiency tiebreak was not needed.

**Sanity gate:** PASSED — no other candidate beats M2 on both accuracy and macro-F1.

---

### Robustness: does the decision depend on excluding M1?

The same rule was re-run with M1 promoted to a full candidate. It would select **M1**, and that selection FAILS the sanity gate.

Promoting M1 would select M1, but that selection FAILS the sanity gate: ['M2', 'M3'] beat it on both accuracy and macro-F1. M2 leads M1 by only 0.0046 ICBHI -- inside the 0.0129 cross-validation tolerance -- so on the primary metric alone they are tied and the efficiency tiebreak prefers the smaller M1. This is almost certainly how the earlier M12 pass selected M1. The documentary exclusion and the numbers therefore agree: M2 is the defensible choice, and the real evidence for it is accuracy and macro-F1, not ICBHI score.

---

## 4. Efficiency and downstream fit

The selected backbone is **23.8× smaller** in parameters than the AST candidate (M4) (3,627,476 vs 86,385,668) and **29.5× faster** at inference (2.937 vs 86.74 ms/sample), while scoring **+0.0868** ICBHI against it.

At 3,627,476 parameters / 13.86 MB this sits in a sensible range for Member C's CQKD compression workstream (M16/M18): large enough to be a meaningful teacher, small enough that the compressed student remains a credible deployability claim. `Model_Training_Reference.md:243` names that downstream constraint as an explicit input to this decision.

---

## 5. Frozen checkpoint

The handoff checkpoint was re-loaded and checked against the reported table:

- ✅ best_score matches results JSON — checkpoint 0.7227 vs reported 0.7227
- ✅ model_state present — 34 tensors
- ✅ parameter count consistent with results JSON — checkpoint state_dict 3,630,457 vs reported params 3,627,476 (+2,981 = BatchNorm buffers, expected)

**Overall: all checks passed.**

Path: `/home/asif/Desktop/CSE465-Project-OWMTL/Asif's/M2/best_model.pth`

---

## 6. Consequences for downstream models

Every downstream model that consumes "the M12 backbone" — M13 (disease head), M15 (cross-task consistency), M17 (OWL Stage 2), and Member C's M16/M18 distillation — should be built on **M2**.

An earlier pass at M12 selected **M1**, made when M2 and M3 did not yet exist and only two models could be compared. M2 improves on M1 by **+0.0046** ICBHI, **+0.0731** accuracy and **+0.0394** macro-F1. Any downstream model already trained against M1 is therefore sitting on a strictly weaker encoder and should be re-fit against M2 before its numbers go in the paper.

---

## 7. Reproducibility

- `results_M12.json` — §4-schema record including the full `decision` block
- `decision_audit.json` — every rule step with its inputs and outcome
- `backbone_comparison.png`, `efficiency_vs_performance.png` — the figures above

Sources for each row:

- **M1**: loaded from /home/asif/Desktop/CSE465-Project-OWMTL/Barshon's/M1/results_M1.json
- **M2**: loaded from /home/asif/Desktop/CSE465-Project-OWMTL/Asif's/M2/results_M2.json
- **M3**: loaded from /home/asif/Desktop/CSE465-Project-OWMTL/Asif's/M3/results_M3.json
- **M4**: TRANSCRIBED from executed output of Barshon's/M4/M4_ast_backbone.ipynb (results_M4.json is not in the repo -- it lives in Barshon's Drive)
