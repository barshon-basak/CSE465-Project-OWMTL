# OWMTL — Pre-Build Fix Triage (aligned to the NEW direction)

**Prepared:** 2026-08-14 · **Basis:** `Model_Training_Reference.md` (logbook) + the audit/significance sources.
**Guiding principle (per your instruction):** judge every model by the **new direction** — physics-grounded, label-free acoustic **concept bottleneck** + **faithfulness/robustness audit** — *not* by its old purpose. Several models that were "must-fix" under the old cross-task/OWMTL plan are now **discard-and-footnote**, because fixing them would be effort spent on a thesis you've retired.

> **The most important reframe:** three of the biggest "issues" are no longer bugs to fix — under the new plan they are **findings to harden**: the M14 conformal paradox (→ your honest-operating-point result), M15 tying with chance (→ "detectors aren't shown to work at n=19"), and Gap7's pediatric-physics fragility (→ a headline robustness finding). Don't "fix" these; re-frame and stat-harden them.

---

## 0. How the new direction re-scores everything (the disposition map)

| Disposition | Meaning | Models |
|---|---|---|
| ✅ **Keep as-is** | Clean asset or citable result; no work | M1, M2, M12, M22, M6, M24-CB |
| 🟥 **Critical fix (mandatory before/at engine build)** | *Is* the engine; broken or not-yet-in-the-right-form | **M35 (+ discard v2), M13**, + two cross-cutting fixes (§3) |
| 🟧 **Fix-before-use (downstream; fix when that stage starts)** | Needed in Weeks 6–9, not before | M29, M19, Gap7, M14, M20, M11, M7, M28 |
| ⚫ **Discard / do-not-fix (old direction)** | Not part of the new thesis; footnote only | M15(as thesis), M17, M30/M30_v2, M31, M32, M33/v2, M34, M36, M16, M18, M21, M3/M4 (unless cited), M37-number |
| 🔍 **Review-to-classify** | Undocumented; 1–2 h each to place | M38, M37_v2, M23 |

The rest of this document details the 🟥 and 🟧 items (with your six fields each), lists the ⚫ discards compactly, and ends with a priority-ordered action list.

---

## 1. 🟥 Critical fixes — mandatory, because they ARE the new engine

### M35 (+ M35_v2) — physics loss → **physics concept extractors**
1. **What needs fixing:** M35 exists only as a *loss term* on the 70/30 split; M35_v2 is a broken re-run (best epoch 1). The new plan needs its DSP as **standalone per-cycle concept generators** (PAPR, spectral flatness — *plus* new clinically-named concepts: fine/coarse crackle, wheeze pitch band, inspiratory phase, rhonchi), computed and **validated against ICBHI cycle labels**, on the **official 60/40 split**.
2. **Why:** These concepts are the bottleneck's input — the entire novelty rests on them. On the 70/30 split M35 isn't comparable to anything; the broken v2 must not be cited.
3. **How:** Refactor the M35 DSP code into a `concept_extractors` module; add the four new extractors (crackle-type via duration+center-freq; wheeze band; breathing-phase per arXiv:1903.10251; rhonchi); re-run/re-validate on the official split; **discard M35_v2**.
4. **Fit to new direction:** This is Build-Sheet Weeks 1–2 and Gate G2 — the foundation both Path A and Path B stand on.
5. **Expected outcome:** A validated per-cycle concept vector for all 6,898 cycles + an extractor-vs-label agreement table (the G2 evidence).
6. **Mandatory?** **Mandatory, first.** Nothing in the new plan proceeds without it.

### M13 — prototypical head → **strict concept→disease bottleneck head**
1. **What needs fixing:** M13 v4 predicts disease from *encoder features*. The new plan needs it to predict disease **only from the concept vector** (a true bottleneck), with the independent/sequential/leaky-joint/opaque variants. Also: earlier versions never trained past epoch 1 — confirm v4 genuinely trains, and start **dumping per-patient scores** (see §3).
2. **Why:** Without the strict bottleneck there is no leakage to measure and no intervention to run — the two core contributions collapse. The Healthy-class weakness (F1 0.40) also needs watching as a G3 collapse signal.
3. **How:** Re-wire M13 to consume the concept vector as its only input; build the leaky-joint control and opaque baseline alongside; verify a real training curve (not epoch-1).
4. **Fit to new direction:** Build-Sheet Week 3; produces the accuracy–interpretability tradeoff that Gate G3 reads.
5. **Expected outcome:** Four trained variants + the per-class disease F1 table (COPD-collapse check).
6. **Mandatory?** **Mandatory** — it's the second engine piece, right after the concepts.

---

## 2. 🟧 Fix-before-use — needed in Weeks 6–9, fix when you reach that stage

### M29 — post-hoc OOD suite → run **in concept space**
- **What/Why:** Clean and reusable, but (a) must be re-run on the concept vector vs the embedding (the I4 comparison), and (b) **saves no raw per-patient scores**, which blocks a proper paired test. **How:** re-run in concept space; dump `{patient_id: score}`. **Fit:** the concept-space-vs-embedding novelty result (Weeks 6–7). **Expected:** ranked detectors in both spaces, with CIs. **Mandatory?** Postponable to the downstream-eval stage; the score-dumping habit (§3) starts now.

### M19 — cross-dataset OOD → **fix the counting bug, re-run as covariate-shift**
- **What/Why:** Its detection AUROCs are below chance and **Coswara has num_samples=2** — a data-loading bug that makes the number meaningless; the whole file is non-schema. Under the new plan Coswara/SPRSound are the *covariate-shift* stress (known disease, new device/population), not semantic-novelty detection. **How:** fix the Coswara loader (n=2 → the real count), re-run as a **covariate-shift** evaluation (does a detector wrongly fire on *known* disease under shift?), export to §4 schema, dump scores. **Fit:** Gate G4 + the physics-fragility story. **Expected:** interpretable covariate-FPR numbers with CIs; the junk AUROCs retired. **Mandatory?** Mandatory **before any cross-dataset claim**, but that's a Week-6–7 task, not a blocker to starting.

### Gap7 — pediatric-physics fragility → **harden, don't fix**
- **What/Why:** This is a *finding*, not a bug — but it's MMD-based and needs to sit on the official metric with CIs to be a headline (I5). Re-verify the numbers after M35 is re-run on the official split. **How:** recompute the shift analysis with the corrected M35 concepts; add bootstrap CIs; state it as "adult-tuned acoustic priors degrade on pediatric airways." **Fit:** headline robustness result for either path. **Expected:** a defensible, cited fragility claim. **Mandatory?** Postponable to Weeks 6–7; high value.

### M14 — conformal wrapper → **re-target + reframe as a finding**
- **What/Why:** Currently wraps the dead M15 score; schema-incomplete; test AUROC below chance. Under the new plan it must wrap the **concept-space** score and be presented as the **conformal-validity analysis** (cite *Pitfalls of Conformal Predictions*, arXiv:2506.18162), not a guarantee. **How:** re-point onto the concept-space detector; fix the §4 schema; write it up as "when conformal is/ isn't valid for respiratory OOD." **Fit:** the honest-operating-point section (I6). **Expected:** a characterized operating point + validity discussion. **Mandatory?** Postponable (downstream trust component).

### M20 / M11 — calibration → **schema re-export only**
- **What/Why:** Both are real but non-schema (M20: 6 loose fields; M11: missing blocks); M20's OWL-stage extension was synthetic. **How:** re-export to the §4 schema; drop the synthetic OWL extension. **Fit:** support the operating-point/trust section. **Expected:** merge-able calibration rows. **Mandatory?** Minor; postponable.

### M7 — deep ensemble → **finish schema + run the OOD pass**
- **What/Why:** 10 checkpoints exist but it was never evaluated as an OOD detector, and the JSON is incomplete. It's a cheap, high-value **scored detector** for the concept-space-vs-embedding comparison. **How:** complete the §4 schema; run the OOD evaluation on the 104/19 split and in concept space; dump scores. **Fit:** one baseline detector in I4. **Expected:** an ensemble-disagreement AUROC with CIs. **Mandatory?** Postponable; do it during the Weeks-6–7 detector sweep.

### M28 — master merge → **re-run last**
- **What/Why:** Tooling; must consume the *new* concept-bottleneck results and the fixed schemas. **How:** re-run after all upstream fixes; enforce "audit-clean only." **Fit:** the results appendix. **Expected:** one consistent results table. **Mandatory?** End-stage (Week 9).

---

## 3. Cross-cutting fixes (apply across many notebooks — these matter most for the new thesis)

### CC1 — **Start dumping raw per-patient/per-cycle scores everywhere** 🟥
- **What/Why:** The significance report is explicit: **no raw per-patient scores are saved anywhere in the repo**, so only a conservative independent-samples z-test was possible — a proper **paired DeLong test** is impossible. Your new thesis is *honest evaluation with CIs + paired tests* (Protocol Essential #10). Without raw scores you cannot deliver it.
- **How:** every notebook that produces a detection/score writes `{patient_id: score}` to a `.npy`/`.csv` next to `results_*.json`. One-line change per notebook.
- **Fit/Expected/Mandatory:** Foundational to Path A *and* Path B; enables real paired tests. **Mandatory, from the first engine notebook onward.**

### CC2 — **Run `check_device_structure.py` before any device claim** 🟥(cheap)
- **What/Why:** Gate G4 depends on whether patients span ICBHI's 4 devices, or device is confounded with diagnosis. Cheap (~seconds), decides whether leave-one-device-out is even possible.
- **How:** run the existing script; record the verdict.
- **Fit/Expected/Mandatory:** Decides the device axis of the robustness section. **Do it in Week 0** — cheap insurance against building an impossible experiment.

### CC3 — **Official split + committed confusion matrix + §4 schema, retroactively for reused assets**
- **What/Why:** Protocol Essentials #8–10 are now hard. Any reused asset that a paper number depends on must be on the official 60/40 split with a committed matrix. (Already true for M2/M12/M22/M29; **needs applying to M35's re-run** and the schema-light trust models.)
- **Mandatory?** Mandatory for anything cited; the schema fills are minor and can trail.

---

## 4. ⚫ Discard / do-not-fix — old-direction models (footnote only; do NOT spend build time here)

Each of these was tied to the retired thesis. Under the new plan the correct action is a one-line "we also tried X; it did not help / is out of scope," **not** a fix.

- **M15** — retired as the thesis. Keep v6 as *one baseline detector* only; the only minor touch worth doing is committing its confusion matrix **if** you cite it as a baseline. Do **not** invest in "fixing" it. Also resolve the 22-vs-19 patient discrepancy in the record (documentation, not a re-run).
- **M17 (OWL forgetting)** — staged open-world learning is old-framing; retire unless you keep an incremental-learning section (you likely won't). Footnote.
- **M30 / M30_v2 (gated fusion)** — a closed-set *accuracy* play, not the concept/faithfulness thesis. **Optional:** run M30_v2 once *only if* you want a stronger encoder than M2 for the bottleneck — but the plan already offers OPERA/M2D for that, so this is low value. Otherwise discard. Do not "rescue" M30.
- **M31 (GradNorm MTL)** — joint MTL is retired (and it was the backbone that made M15 v5 collapse). Discard/footnote.
- **M32 (demographic fusion)** — below baseline. Discard now; *optionally* revisit age/sex later as **auxiliary bottleneck concepts** (a different, cheaper use) — but not a fix of M32.
- **M33 / M33_v2 (temporal transformer)** — not in the thesis; never cite v1's collapse. Footnote v2 as "tried, below baseline."
- **M34 (curriculum)** — below baseline, out of scope. Footnote.
- **M36 (multistage distillation)** — collapsed (Se 0.09); never cite. Compression is a footnote at most. Discard.
- **M16 / M18 (compression)** — deployment footnote only; M18's schema/synthetic history isn't worth fixing unless you add a deployment section. Discard for now.
- **M21 (SNR curriculum)** — metrics=1.0 train/test leak; out of scope. Discard — **but** do a 10-minute check that the leak is isolated to M21 and didn't contaminate a shared split file (safety, not a fix).
- **M37 (number) / M3 / M4** — M37's LoRA *machinery* is kept for FM probing (G5), but the M37 *result* isn't needed — don't re-run it on the official split unless you cite it. M3/M4 matter only if M22/M30 are cited; M4's missing JSON is only worth fixing if M4 appears in the paper.

---

## 5. 🔍 Review-to-classify (1–2 h each, do in Week 0 to avoid duplicate work)

- **M38 (large-N open-set)** — potentially the most relevant of the three: if it already addresses the n=19 power problem, it may feed the honest-evaluation contribution. **Review first.**
- **M37_v2 (LoRA on AST)** — relevant to the FM-probing arm (G5); check what it actually did.
- **M23 (AST + SpecAugment, Farhana)** — low priority; document its status when convenient.

---

## 6. Final priority-ordered action list

**Phase 0 — Week 0 (cheap, unblock everything):**
1. **CC2** — run `check_device_structure.py` (decides the device axis). *[minutes]*
2. **CC1** — adopt raw per-patient score dumping as a standing rule (add to the engine notebook template). *[foundational]*
3. **Review M38 + M37_v2** (and M23 if time) — know what you already have for the FM/large-N arms. *[1–3 h]*
4. **M21 leak check** — confirm the train/test leak is isolated. *[10 min]*

**Phase 1 — engine build, Weeks 1–3 (the mandatory 🟥 fixes):**
5. **M35 → concept extractors**, re-validated on the **official split**; discard M35_v2. *(Gate G2)*
6. **M13 → strict concept-bottleneck head** + controls; confirm real training; dump scores. *(feeds Gate G3)*

**Phase 2 — downstream eval, Weeks 6–7 (the 🟧 fixes, when you reach them):**
7. **M29** in concept space + score dump.
8. **M19** counting-bug fix + covariate-shift re-run + schema.
9. **Gap7** hardened (official metric + CIs).
10. **M14** re-target + reframe; **M20/M11** schema re-export; **M7** OOD eval pass.

**Phase 3 — reporting, Weeks 8–9:**
11. **M28** re-run on the cleaned/new results (audit-clean only).

**No action (footnote only):** M15-as-thesis, M17, M30/M30_v2, M31, M32, M33/v2, M34, M36, M16, M18, M37-number, M3/M4.

---

## 7. When are the models "ready to proceed with the new plan"?

**You can start the new plan as soon as Phase 0 + Phase 1 are done** — i.e., after the device-structure check, the score-dumping rule, and the two engine fixes (**M35 concept extractors** and **M13 bottleneck head**). Everything in Phase 2 is needed only *when you reach the downstream-evaluation stage* (Weeks 6–7), not before — so it does not gate the start.

Concretely, the **gating set before you write Path-A/B code** is exactly four items: **CC2 (device check), CC1 (score dumping), M35 refactor+official-split, M13 re-wire.** The rest is either downstream (fix on arrival) or discard (never). That keeps you moving without dragging the retired thesis's baggage forward.

*This triage uses the new direction as the sole guiding principle. Anything tied only to the retired cross-task/OWMTL thesis is footnoted, not fixed — deliberately, to avoid spending your remaining weeks maintaining a direction you've already left.*
