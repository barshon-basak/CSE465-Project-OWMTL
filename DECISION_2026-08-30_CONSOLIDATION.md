# Decision record — 2026-08-30: consolidation, and the ceiling claim is retracted

**Status:** DECIDED · **Supersedes in part:** `DECISION_2026-08-16_PIVOT.md` §"What the project's
contribution now is" item 3, and `PAPER_OUTLINE.md` §6.
**Read with:** `RTK_requirements.md` (course checklist), `Novelty Experiment/NOVELTY_STATUS.md`.

Four things were decided and executed today: the duplicate-run problem, the repo structure, the
second-physician deadlock, and — the one that matters — what the G2 failure actually means.

---

## 1. The finding that changes the paper: N11

The project could not decide between two readings of its own negative result:

- **(a)** our DSP extractors are weak, or
- **(b)** AUROC 0.65 against ICBHI cycle labels is unreachable by any method.

`DECISION_2026-08-16_PIVOT.md` called (b) "unknown, and this experiment cannot decide it". N10 tried
to decide it with the clinician labels and returned UNDECIDED. **N11 decides it**, by asking a
strictly easier question: how well does a model that is *allowed to see the labels* do?

`Novelty Experiment/N11_supervised_ceiling.py` — logistic probe, fit on the 79 train patients,
scored on the same 2,636 test cycles / 47 test patients G2 was scored on. Zero patient overlap,
verified. Readings were pre-registered in the file before it was run.

| Target | our G2 score (best of 2 runs) | 14 DSP concepts as a vector | **AST frozen** (never saw ICBHI) | M2 encoder |
|---|---:|---:|---:|---:|
| crackle | 0.5580 | 0.6608 [0.601, 0.713] | **0.7115** [0.638, 0.771] | 0.7451 [0.694, 0.795] |
| wheeze | 0.5729 | 0.5815 [0.492, 0.675] | **0.7621** [0.702, 0.815] | 0.8673 [0.806, 0.918] |

**The pre-registered "NO CEILING" branch fired.** Three consequences, in order of how much they cost
us:

1. **Reading (b) is dead and the ceiling claim is retracted.** These labels support detection at
   0.71–0.87. The binding constraint was never the reference standard. The pivot document promised
   to concede this if shown — this is that moment, and conceding it is what the two-run gate
   discipline was for.
2. **The AST arm is clean.** AudioSet-pretrained, never exposed to a respiratory corpus or an ICBHI
   label, and it still clears the gate on both concepts. This cannot be waved away as circularity
   the way the M2 arm can.
3. **Part of the G2 failure was the gate's own design.** G2 thresholded *one hand-built scalar*. The
   *same 14 concepts* as a vector reach 0.6608 on crackle — above the threshold the scalar failed.
   So the concept vocabulary carried more than the scalar score exposed. Report this; it is an
   honest methodological point and reviewers will value it more than the null.

**What survives untouched:** the three evaluation errors and corrected baselines (§3 — still the
lead contribution); the two-run pre-registered gate and its failure, exactly as reported; N5
(concepts carry almost no disease information); N6 (the interpretability-cost claim does not
replicate); N7 (the bottleneck is bypassable); N1 (LoRA degrades 13/14 concepts). None of those
depended on the ceiling.

### The headline this opens, which is better than the one it closes

> A frozen AudioSet model that has never heard a respiratory corpus recovers ICBHI's wheeze labels
> at AUROC 0.76 and crackle at 0.71. On the same corpus, our physician agrees with those labels at
> κ = 0.035 (crackle) and 23.1% sensitivity, and seven senior physicians in the published study
> score 47.77%. **Machines reproduce these labels substantially better than trained listeners
> agree with them.**

That is a sharper, more defensible and more interesting claim than "concept validation is bounded",
it uses evidence already on disk, and it does not need a second physician.

**Scope discipline, unchanged:** this bounds nothing about respiratory ML overall, and it is not a
claim that ICBHI's labels are wrong. It is a statement about what these labels support.

---

## 2. The second-physician deadlock — resolved without one

There is no second physician and the paper stops waiting for one. It was needed for exactly one job:
bounding the reference standard. **N11 does that job better** — a supervised probe is a tighter and
more direct bound than a second rater's κ would have been.

What we ship on the human side instead:

- clinician-vs-ICBHI κ 0.035 (crackle) / 0.266 (wheeze), with patient-bootstrap CIs — a *finding*
  now, not a ceiling argument;
- **intra-rater** κ 0.714 / 0.600 from the 12 deliberately repeated clips, which is the reliability
  evidence a single rater *can* produce;
- the published n=7 (Tzeng 2025) and n=12 (Aviles-Solis 2016) studies for the population claim;
- `release_annotations/` — 132 fine-grained annotated cycles, the first on this corpus.

"Single rater, no inter-rater agreement" moves to Limitations and stays there. No further clinician
work is scheduled or blocking.

---

## 3. Canonical-run register

One run per model is canonical. Everything else is archived to `Archive_Files (v4)/` — kept, never
cited. Moved today: `M2_17aug_run_v2`, `M2_29aug_run_v3`, `M3_17aug_run`, `M14_v1`, `M15_v4`,
`M15_v5`, `M30_v1_withdrawn`, `M13cbm_single_seed`.

**Canonical, corrected official 60/40, patient-independent — the paper's table:**

| Model | What | Official ICBHI | Location |
|---|---|---:|---|
| **M22_v2** | MobileNetV2 + SpecAugment — **best model** | **0.5602** | `Asif's/M22_v2/` |
| M41 / M41_aug | Swin-T — best transformer | 0.5304 / 0.5291 | `M40_M43_transformers/Results/` |
| M42 / M42_aug | DeiT-S | 0.5149 / 0.4981 | same |
| M3_v2 | MobileNetV2, no augmentation (= ablation row A1) | 0.5200 | `Asif's/M3_v2/` |
| M3 | MobileNetV2, M12_v2's decision candidate | 0.5135 | `Asif's/M3/29 aug run/` |
| M21_Updated | curriculum, corrected | 0.4997 | `Barshon's/M21/M21_Updated/` |
| M40 / M40_aug | ViT-B/16 — **collapsed to all-Normal** | 0.4994 / 0.5000 | `M40_M43_transformers/Results/` |
| M2 | scratch CNN, M12_v2's decision candidate | 0.4720 | `Asif's/M2/m2_v4/` |
| M45 A0–A6, P4, P5 | ablation | 0.4595–0.5602 | `Asif's/M45/` |
| M16_v2, M18_updated, M20_updated | compression / calibration, corrected | — | `Barshon's/` |
| M43 | AST — **running on Kaggle** | pending | — |

**Legacy fallback-split numbers.** M2 0.6138, M3 0.5895, M22 0.6495, M1 0.6143, M12 0.6138,
M30_v2 0.5975. These are kept **only** as the "before" half of the split audit. They are not
comparable to the table above and must never appear in the same column.

**70/30 novelty runs** (M31–M37, M24) are not rankable against either. Every table row carries a
split column; M35's 0.6864 is a 70/30 number and is written as one.

**Two G2 number-pairs exist and must not be mixed:**

- **the pre-registered gate** — `Asif's/M39/M39_handoff` (run 1: crackle 0.5506, wheeze 0.5729) and
  `Asif's/M39/2nd_run_handoff` (run 2: 0.5580 / 0.5340). **9 concepts. This is the gate. Cite this.**
- the later **14-concept engine** validation, 0.5556 / 0.5818 (`owmtl_concept_engine/notebooks/01_*`).
  A different implementation, not a re-export. `N9` and `N10` had conflated the two; both are fixed.

**Bookkeeping corrections:** M37's real result (0.6753) lives in `Barshon's/M37_v2/`, not `M37/` —
the master index has them swapped. `M11`'s result file was named `results_M11 (3).json`; renamed.

---

## 4. Repo structure

`owmtl_concept_engine/owmtl_concept_engine/` is un-nested — contents moved up one level, the empty
`owmtl_critical_fixes/` shell and the duplicate README/STEP_SEQUENCE removed, and the path
candidates in `Novelty Experiment/common.py` updated to match. Verified by re-running N9/N10/N11.

`M28` now skips any path containing `Archive_Files` when it globs `results_M*.json`. Previously
archiving a run did not remove it from the merged table — id-based exclusion cannot separate two
runs of the same model. Archiving now actually means excluded.

---

## 5. Incomplete or missing runs — the full list

| # | Item | State | Decision |
|---|---|---|---|
| 1 | **M43 AST** | running on Kaggle | commit `results_M43.json` on return; closes requirement 3 and the M4 hole |
| 2 | **T1 metrics backfill** | 12 critical / 94 warning findings; 17 of 39 JSONs miss required fields | run `icbhi_score_audit.py --write`, then re-export what has a checkpoint, mark the rest "not recoverable". No GPU |
| 3 | **T2 master table** | last built 2026-08-29, predates M40–M45 | re-run M28 (archive filter now in place), with the split column |
| 4 | **M46 rows P1/P2/P3** | band-pass, denoising, amplitude normalisation **were never implemented** | implement the three stages, run the three ablation rows. This is the last real code gap in course requirement 1 |
| 5 | **M40 ViT collapse** | all-Normal in both runs | one re-run with warmup + lower backbone LR; if it collapses again, report it as the small-data ViT finding |
| 6 | **M35 on corrected split** | only exists at 70/30 | run it — it is the only number that would change the results table's top row |
| 7 | **M38 large-N open-set** | generator + notebook, never run | **dropped.** At n=19 unknown patients every CI spans chance; enlarging via SPRSound confounds novelty with pediatric shift. Say so once in Limitations |
| 8 | **M4 (AST, original)** | no committed JSON, numbers only transcribed | **dropped** from all tables; M43 replaces it |
| 9 | N5 permutation null | n_perm = 5 | raise to ≥20 (~2 h CPU) before quoting a null *interval* |
| 10 | N1 LoRA rank sweep | one rank | optional; tests whether concept destruction scales with adaptation strength |
| 11 | Paper `\pending{}` markers | ~24, **most stale** | M44, M45 (A2–A6, P4, P5), M40–M42 and the corrected re-runs all exist; paste the numbers in |

Also found and fixed today: `.gitignore`'s blanket `results/` rule was hiding
**`Novelty Experiment/results/` entirely** — every N1–N11 result JSON quoted in `NOVELTY_STATUS.md`
existed only on one laptop. All 28 files are now tracked; the large regenerable inputs
(`M2_features.npy`, `embeddings/`) stay ignored.

**The one optional run worth more than the rest** (not started, ~20 lines): score N11's AST probe on
the 132 clinician-labelled clips using N10's existing clip mapping. It would show directly whether
the machine agrees with ICBHI *on the cycles where the physician does not* — the exact evidence
§1's new headline rests on. Everything it needs is already on disk.

---

## 6. The path from here

Ordered by what unblocks what. Steps 1–3 need no GPU.

1. **Rewrite the claim** — §1's retraction into `PAPER_OUTLINE.md` §6 and `main.tex`. This is the
   fork in the road; everything downstream is phrasing that depends on it.
2. **T1 backfill + re-run the audit** — makes 30+ models countable for requirements 4 and 5, and
   clears the 12 criticals.
3. **Strip the stale `\pending{}` markers** — M44, M45, the transformers, the corrected re-runs.
4. **M43 lands** (in flight) → **re-declare the best model** under `RTK_requirements.md` §6's rule.
5. **M46 P1/P2/P3**, then fold M45 + M46 into one ablation section.
6. **M35 corrected** and the **M40 warmup re-run** — both cheap on Kaggle, both optional.
7. **T2 master table + figure pack** last, once the audit is clean.
8. Optional, high value: the AST-probe-vs-clinician comparison from §5.

## 7. What this does not authorise

The stop-list stands. N11 does **not** reopen the concept bottleneck: N5, N6 and N7 independently
show the concepts carry almost no disease information, that the interpretability-cost claim does not
replicate, and that the bottleneck is bypassable. No bottleneck head, no intervention API, no fourth
mechanism, and **no third G2 extractor revision** — N11 is a ceiling estimate with a different
estimator and a different question, and nothing in it feeds back into extractor design.
