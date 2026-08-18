# OWMTL — Build-Plan Execution Sequence (what's done, what's now, what must WAIT)

This is the ordered checklist for executing `OWMTL_Build_Sheet.md` (moved -> `../../Archive_Files (v3)/OWMTL_Build_Plan (v3).md` (moved -> `../../Archive_Files (v3)/OWMTL_Build_Plan (v3).md`)). Every step is numbered so the run order is
unambiguous. Status legend:

- ✅ **DONE** — code delivered & unit-tested in `owmtl_concept_engine/`.
- ▶️ **RUN-NOW** — code exists; you just run it on real ICBHI (deterministic).
- 🟢 **BUILT-THIS-ROUND** — new deterministic code delivered now (pre-G3).
- ⏸️ **WAIT — RESULT** — cannot proceed until a *run result* is known (do not assume).
- ⏸️ **WAIT — DECISION** — cannot proceed until *you/Dr. Khan decide* (do not assume).

> **The one rule:** everything up to and including **step 05** is deterministic and produces the three inputs the
> Gate-G3 decision needs. **Stop at G3.** Do not implement anything past G3 until (a) the G2 validity result, the
> device-structure result, and the G3 Path-A/B decision are known. Those are marked ⏸️ below.

---

## Phase 0 — Pre-flight (Week 0)

| # | Step | Status | Notes |
|---|---|---|---|
| 00a | **G0 — annotate a concept subset?** | ⏸️ **WAIT — DECISION** | Your call with Dr. Khan. Default = label-free (no annotation). Changes nothing in steps 01–05; only lifts the *ceiling* later. |
| 00b | **G1 — pre-code kill-search** | ✅ DONE | Re-validated 2026-08-14 (concept-bottleneck + faithfulness still open). Re-run once more just before submission. |
| 00c | **CC2 — device-structure check** | ▶️ RUN-NOW → then ⏸️ **WAIT — RESULT** | Run `scripts/check_device_structure.py --wav_dir <ICBHI>`. Its verdict decides the **device axis (Gate G4)** later. Deterministic to run; the *result* gates step 08b. |
| 00d | **CC1 — score dumping + protocol harness** | ✅ DONE | `owmtl/eval_utils.py` (dump_scores, CIs, McNemar, §4 writer). Baked into every notebook. |

## Phase 1 — Build the engine (Weeks 1–4, all deterministic → produce the G3 inputs)

| # | Step | Status | Deliverable |
|---|---|---|---|
| 01 | **Concept extraction + Gate-G2 validation** | ▶️ RUN-NOW | `notebooks/01_concept_extraction_and_validation.ipynb` → `concepts_all.npz`, `concept_validation_report.json`. **G2 result gates step 02+** (see WAIT below). |
| 02 | **Export frozen M2 features** | 🟢 BUILT-THIS-ROUND | `owmtl/m2_features.py` (fill in `load_m2` with your M2 class). Deterministic dump from the already-frozen backbone → `M2_features.npy`. |
| 03 | **Concept bottleneck + controls (tradeoff)** | ▶️ RUN-NOW | `notebooks/02_concept_bottleneck_training.ipynb` → `results_M13cbm_*.json`, prints the accuracy–interpretability tradeoff (G3 input #1). |
| 04 | **Concept leakage / sufficiency** | 🟢 BUILT-THIS-ROUND | `owmtl/leakage.py` + `notebooks/03_concept_leakage_measurement.ipynb` → `leakage_report.json` (G3 input #2). |
| 05 | **Concept intervention (clinician demo)** | 🟢 BUILT-THIS-ROUND | `owmtl/intervention.py` + `notebooks/04_concept_intervention.ipynb` → `intervention_report.json` (G3 input #3). |

## ★ Gate G3 — Path A vs Path B  ·  ⏸️ WAIT — DECISION (needs the results of 03–05)

**Do not build past here yet.** Once steps 03–05 have run, you have the three inputs (tradeoff, leakage,
intervention). Read them and decide the headline **from the data**:
- bottleneck holds accuracy + low leakage + concept-driven intervention + no COPD-collapse → **Path A**;
- collapses / heavy leakage / weak intervention → **Path B** (the collapse *is* the finding).

Everything below is **shared** between A and B, but its *shape* depends on results above (which concepts survived
G2, whether the device axis exists), so it is intentionally **not** implemented yet.

## Phase 2 — Downstream evaluation (Weeks 6–7) — WAIT for the gates above

| # | Step | Status | Blocked on |
|---|---|---|---|
| 06 | Concept-space vs embedding OOD (I4) | ⏸️ WAIT — RESULT | needs the validated concept set (G2) + trained variants (03). Also uses the fixed M29 + score dumps. |
| 07 | Honest operating point / conformal validity (I6, re-frames M14) | ⏸️ WAIT — RESULT | needs a concept-space score to wrap. |
| 08a | Pediatric covariate-shift (SPRSound) + physics-fragility (I5, hardens Gap7) | ⏸️ WAIT — RESULT | deterministic once concepts exist, but interpret only after G3 framing. |
| 08b | Device leave-one-device-out | ⏸️ WAIT — RESULT | **blocked on CC2 verdict (00c)** — build only if device is separable. |
| 09 | Foundation-model probing (I3 / Gate G5) | ⏸️ WAIT — DECISION | your call whether to spend FM compute; mainly a Path-B enrichment. Reuses M37 LoRA. |

## Phase 3 — Synthesis (Weeks 8–9) — WAIT

| # | Step | Status | Blocked on |
|---|---|---|---|
| 10 | Sharpen to one claim + figures (G6) | ⏸️ WAIT — RESULT | needs all Phase-2 results. |
| 11 | Write draft + venue (G7) | ⏸️ WAIT — DECISION | needs the headline (G3) + result strength. |

---

## So: is anything left before the build plan? — Direct answer

**No blocker remains to *start* the build plan.** The deterministic pre-decision engine is now complete:
- Steps 00c/00d, 01, 02, 03, 04, 05 all have delivered, tested code.
- The only things standing between you and Gate G3 are **running** notebooks 01→02→03→04 on real ICBHI (plus the
  one-time `load_m2` fill-in for step 02).

**What I deliberately did NOT build (correctly waiting):**
- **G3 Path A/B** — a decision from the run results.
- **Steps 06–08a, 10** — shape depends on the G2 validity result and the G3 framing.
- **Step 08b (device LODO)** — depends on the CC2 device-structure verdict.
- **Step 09 (FM probing)** — depends on your G5 decision.
- **G0 annotation** — your decision with Dr. Khan.

When you've run 01–05 and made the G3 call, come back with the three reports (`concept_validation_report.json`,
the `results_M13cbm_*` tradeoff, `leakage_report.json`, `intervention_report.json`) and I'll implement the correct
Phase-2 steps for whichever path the data chose.
