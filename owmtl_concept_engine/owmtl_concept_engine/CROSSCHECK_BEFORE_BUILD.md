# Cross-Check — anything left to fix/implement before the Build Plan?

**Date:** 2026-08-14 · **Method:** reconciled every issue in `Model_Training_Reference.md` against the delivered
`owmtl_concept_engine/`, and reviewed the three notebooks previously marked ⚠️ (M38, M37_v2, M23) plus M2's
architecture. **Bottom line: nothing left blocks the build plan.** One deterministic gap (M2 feature hook) is now
closed; three ⚠️ items are resolved; and one downstream task turned out to be *already done*.

---

## 1. Resolutions from reviewing the ⚠️ notebooks

| Item | Previous status | **Resolved finding** | New disposition |
|---|---|---|---|
| **M38** (large-N open-set) | ⚠️ unknown | **A real, high-value asset.** Runs the M29 scorers on 3 regimes — ICBHI(19) / **SPRSound large-N near-OOD** / Coswara far-OOD control — computes **CIs inline**, **writes raw per-sample scores (`scores_M38.csv`)**, and **fixes the M19 Coswara `num_samples=2` bug** (globs the dataset root, not one participant). | **REUSE in Phase 2** (steps 06 + 08a). Not discard. |
| **M37_v2** (filename says "AST LoRA") | ⚠️ unknown | Actually **LoRA on the M2 *CNN*** (transfers 4/4 M2 FC tensors, 8,356 trainable = 0.2%, ICBHI 0.7969 macro / 0.6753 official) — i.e. ~a **re-run of M37**, *not* AST. | **Duplicate of M37.** LoRA machinery kept for G5 (FM probing). No new work. |
| **M23** (AST + SpecAugment, Farhana) | ⚠️ unknown | An **AST (M4-lineage) run**; captured outputs are M4's (peak ICBHI 0.6359). AST lost the backbone race and is **not** the new encoder. | **Footnote/discard** for the new plan. |

**Two knock-on effects on the triage:**
- **CC1 (raw score dumping) is already done for the open-set side** — M38 writes `scores_M38.csv`. My
  `owmtl/eval_utils.dump_scores` generalizes the same habit to the concept/disease side. Consistent, no conflict.
- **The "fix M19 counting bug" task (was Phase-2 step 08) is superseded by M38** — use M38's Coswara/SPRSound
  loading instead of re-fixing M19. M19 itself stays a footnote.

---

## 2. The one deterministic gap — now closed

**M2 feature hook for step 02.** Confirmed M2 (`Asif's/M2/M2_cnn_baseline_tuned.ipynb`) already exposes
`get_embedding(x)` (penultimate 768-dim GAP features). I inlined the exact architecture into
**`owmtl/m2_ready.py`** and verified it reproduces the documented spec (**3,627,476 params, 768-dim**). Step 02 is
now turn-key:

```python
from owmtl.m2_ready import ready_load_m2
from owmtl.m2_features import export_features
load_m2 = ready_load_m2("Asif's/M2/best_model.pth")     # only the checkpoint path is needed
export_features(records, AUDIO_DIR, load_m2, out_path="/kaggle/working/M2_features.npy")
```

---

## 3. Full reconciliation — every Reference issue vs the build plan

| Reference issue | Blocks build plan? | Status |
|---|---|---|
| **M35 on 70/30, v2 broken; needed as concept generators** | was blocking | ✅ **Rebuilt** as `owmtl/concept_extractors.py` (official split, validated in nb01). v2 discarded. |
| **M13 not a strict bottleneck; early versions broke at epoch 1** | was blocking | ✅ **Rebuilt** as `owmtl/bottleneck.py` (4 variants) + nb02. |
| **CC1 no raw scores saved anywhere** | was blocking | ✅ `eval_utils.dump_scores` + already present in M38. |
| **CC2 device structure unknown** | was blocking | ✅ `owmtl/device_check.py` + CLI. Run it to get the G4 verdict. |
| **M2 feature hook for downstream** | minor | ✅ `owmtl/m2_ready.py` (this round). |
| Leakage measurement (G3 input) | pre-G3 needed | ✅ `owmtl/leakage.py` + nb03. |
| Intervention (G3 input) | pre-G3 needed | ✅ `owmtl/intervention.py` + nb04. |
| M14 conformal paradox | downstream (I6) | ⏸️ Phase-2 step 07 — after G3. |
| M19 Coswara n=2 / below-chance AUROC | downstream | ✅ **superseded by M38**; M19 footnoted. |
| Gap7 pediatric-physics fragility | downstream (I5) | ⏸️ Phase-2 step 08a — hardened after G3; M38's SPRSound regime feeds it. |
| M7 ensemble schema/OOD eval | downstream | ⏸️ Phase-2 step 06 — a scored detector. |
| M20 / M11 calibration schema | downstream | ⏸️ Phase-2 step 07 — schema re-export. |
| M30/M30_v2 fusion | discard | ⚫ footnote; optional single M30_v2 run only if a stronger encoder is wanted. |
| M31/M32/M33/M34/M36 (below baseline / collapsed) | discard | ⚫ footnote; never cite collapses. |
| M16/M18 compression | discard | ⚫ deployment footnote. |
| M21 SNR-curriculum train/test leak | safety check | 🟡 **Optional 10-min check** the leak is isolated to M21. The new engine uses the official split via `icbhi_data`, so it cannot inherit M21's leak — low risk, but worth confirming if M21 ever wrote a shared split file. |
| M23 / M37_v2 / M38 undocumented | review | ✅ reviewed above. |

---

## 4. Verdict — is anything left before the build plan?

**No blocker remains.** The pre-G3 engine is complete and turn-key:

- **Covered by delivered, tested code:** CC2, CC1, M35→concepts, M13→bottleneck, leakage, intervention, **M2 loader**.
- **Already done elsewhere:** open-set score dumping + the M19 fix (M38).
- **Correctly deferred (post-G3 / result- or decision-dependent):** M14, Gap7, M7, M20/M11, device LODO, FM probing.
- **Correctly discarded (footnote):** M30–M36 sweep, M16/M18, M23, M19.

**Optional, cheap, before you run (not blocking):**
1. `scripts/check_device_structure.py --wav_dir <ICBHI>` — get the G4 verdict (decides step 08b later).
2. 10-min check that **M21**'s train/test leak never touched a shared split file (the new engine is immune, but confirm).
3. When you reach Phase 2, **reuse M38** for the SPRSound near-OOD / Coswara control rather than rebuilding.

**You are clear to start executing the build plan** in the `STEP_SEQUENCE.md` order:
`00c device check → 01 concepts+G2 → 02 export M2 feats (ready_load_m2) → nb02 tradeoff → nb03 leakage → nb04 intervention → read reports → Gate G3.`
Stop at G3 and bring the four reports back for the path-specific Phase-2 build.
