# Novelty Experiment — status, results and progress tracker

**Owner of this file:** whoever last ran something here. Update the run log at the bottom.
**Created:** 2026-08-29 · **Source of the list:** Dr. Khan's eight suggested novelty directions.
**Read with:** `README.md` (how to run), `../DECISION_2026-08-16_PIVOT.md` (why several of these
already failed their gates), `../RTK_requirements.md` (the separate course checklist — do not merge).

---

## 0. Why this folder exists

The faculty gave eight directions. An audit of the repo (2026-08-29) found that **four already had
real committed results, two were half-built in the wrong form, and two were effectively untouched** —
but scattered across eight folders, four naming schemes and two abandoned plan documents, so nobody
could answer "did we try this?" without an hour of archaeology.

This folder answers it in one place. Each of the eight has:

- a script `N<k>_*.py` that runs the experiment and writes `results/N<k>_*.json`,
- a docstring at the top of that script stating **what already existed** and **what was missing**,
- a status row in §2 below.

**Important framing for the supervisor meeting:** several of these directions have *already been
tested and failed* on this corpus — that is documented, pre-registered, and is the project's actual
contribution after the 2026-08-16 pivot. The purpose of this folder is to show the attempts were
made **rigorously**, with controls and confidence intervals, not to manufacture wins. Every script
here refuses to state a result its confidence interval does not support.

### One conflict, stated plainly

`../RTK_requirements.md` §12 and `../DECISION_2026-08-16_PIVOT.md` both explicitly **forbid** this
work: *"no bottleneck head, no intervention API, no concept-leakage work, no fourth idea."* Items 5–7
were run anyway (commits of 2026-08-15 and 2026-08-26). **The faculty list supersedes that stop-list.**
Written down here so nobody re-litigates it in three weeks.

---

## 1. Scoreboard

| # | Faculty item | Before this folder | Now | Script |
|---|---|---|---|---|
| 1 | Foundation-Model Concept Probing (LoRA) | 🔴 no FM in the repo at all | 🟡 code complete, **needs GPU + audio** | `N1_fm_concept_probing.py` |
| 2 | Concept-Space Open-Set Recognition | 🟡 embedding space only | ✅ **run** — concept arm done | `N2_concept_space_osr.py` |
| 3 | Prototypical Few-Shot Disease Head | 🟡 one fixed k, no baseline | ✅ **run** — shot sweep done | `N3_prototypical_fewshot.py` |
| 4 | Calibration-Aware Honest Operating Point | 🟡 parts, never assembled | ✅ **run** | `N4_honest_operating_point.py` |
| 5 | Concept Leakage / Faithfulness Audit | ✅ done, no CI/null | 🟡 **run (partial)** — needs `M2_features.npy` | `N5_leakage_audit.py` |
| 6 | Physics-Derived Acoustic Concept Bottleneck | ✅ done, 1 seed, no control | 🟡 **run (partial)** — needs `M2_features.npy` | `N6_physics_bottleneck.py` |
| 7 | Clinician Concept Intervention | 🟡 mechanism only | ✅ **run** — curve added | `N7_clinician_intervention.py` |
| 8 | Pediatric Physics-Fragility | 🔴 12-line MMD stub | 🟡 **pre-registered** — needs SPRSound | `N8_pediatric_fragility.py` |

Legend: ✅ ran to completion here · 🟡 runs, blocked on one input · 🔴 nothing.

**Two files unblock five of the eight:**

| File | Unblocks | How to produce it |
|---|---|---|
| `M2_features.npy` (768-d, 6898 rows) | N5 full, N6 all 4 modes, N2/N3/N4 embedding arms | `owmtl.m2_features.export_features(records, audio_dir, load_m2)` — it exists as a Kaggle dataset, it is just not committed |
| SPRSound audio | N8 | `--sprsound_dir <dir of .wav>` |

Plus a GPU + the ICBHI audio for N1.

---

## 2. Item-by-item: what existed, what was missing, what we got

### 1. Foundation-Model Concept Probing (LoRA) — 🟡 code complete, not yet run

**What existed.** `Barshon's/M37_v2/` is titled *"Audio Spectrogram Transformer LoRA PEFT"* but the
code instantiates `M37_LoRA_CNN` wrapping the project's own scratch-trained `M2_CNN` — 3,635,700
total params, 8,356 trainable, LoRA on two FC layers. **There is no foundation model anywhere in the
repo**; a grep for OPERA / M2D / HeAR / CLAP / AudioMAE / BEATs / wav2vec / PANNs returns nothing.
M4 (AST) has no committed `results_M4.json`. Its reported 0.7969 is the inflated macro ICBHI variant
on the 70/30 split.

**What was missing.** Everything the faculty actually asked for: a real foundation model, and
*probing* — testing whether the FM embedding encodes the 14 physics concepts. `STEP_SEQUENCE.md`
step 09, never built.

**What `N1` does.** Frozen AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) → linear probe of each of
the 14 concepts (AUROC for binary, ridge R² for continuous), patient-grouped CV, bootstrap CI,
within-patient permutation null. Then LoRA-adapt AST to the 4-class task (hand-rolled adapter — no
`peft` dependency) and **re-probe**. `probe_lora − probe_frozen` is the headline: if task adaptation
destroys concept encoding, the model bought its accuracy by ignoring the clinical sounds. A random
projection control runs every time.

**Status.** Probe validated on synthetic data with a planted concept (`--dry-run` passes: recovers
the planted concept at R²=0.9996, rejects pure noise at −0.0072). Needs `--audio_dir` + GPU.

---

### 2. Concept-Space Open-Set Recognition — ✅ RUN

**What existed.** M29: MSP/entropy/energy/Mahalanobis on frozen M12 **embeddings**, Energy AUROC
0.6466, CI spanning chance at n=19. M38 (large-N) has a generator and a notebook but **was never
run** — no results JSON. The concept-space arm (`STEP_SEQUENCE.md` step 06) was never built.

**Result (104 known / 19 unknown patients, official 60/40, patient level):**

| detector | AUROC | 95% CI | unknown recall @95% known-TPR | verdict |
|---|---|---|---|---|
| energy | 0.6267 | [0.477, 0.766] | 0.053 | not shown to beat chance |
| entropy | 0.6218 | [0.469, 0.761] | 0.053 | not shown to beat chance |
| msp | 0.6120 | [0.463, 0.753] | 0.053 | not shown to beat chance |
| knn | 0.5251 | [0.365, 0.689] | 0.053 | not shown to beat chance |
| mahalanobis | 0.5202 | [0.366, 0.678] | 0.000 | not shown to beat chance |

**Reading.** The interpretable 14-d concept space performs *within noise of* the 768-d embedding
space (M29's 0.6466). At n=19 unknown patients no CI can exclude chance — that is a property of the
corpus, not of the method. **The publishable sentence is: "a 14-dimensional, clinically-named score
matches a 768-dimensional opaque one on this task, and neither is shown to beat chance at n=19."**
Adding `--features` runs the paired concept-vs-embedding bootstrap, which is the only defensible
form of the comparison.

---

### 3. Prototypical Few-Shot Disease Head — ✅ RUN

**What existed.** M13 v4: real, careful work — nearest-class-mean over frozen M2/M12 embeddings,
patient-level, accuracy 0.7209 / macro-F1 0.6061 (v3 scored URTI recall 0.026 on synthetic data).
But: one fixed k, no baseline, and it reports `icbhi_score` 0.7137 — a metric that is **undefined**
for a 3-class disease task (the official ICBHI score is (Se+Sp)/2 over the 4-class sound-event task).

**Result (concept space, 61 train / 43 test patients):**

| k (support/class) | proto macro-F1 | 95% CI | linear head, same support | paired verdict |
|---|---|---|---|---|
| 1 | 0.3091 | [0.153, 0.533] | 0.2932 | not shown to differ |
| 2 | 0.3116 | [0.165, 0.493] | 0.3153 | not shown to differ |
| 5 | 0.3027 | [0.143, 0.490] | 0.3587 | not shown to differ |
| 10 | 0.2615 | [0.169, 0.424] | 0.4237 | **linear wins** |
| 20 | *skipped* | — | — | k > rarest train class (10 patients) |
| all-train | 0.2309 | [0.133, 0.328] | — | — |

**Reading.** In concept space the prototypical head does not beat a plain linear head, and loses to
it by k=10. The classic few-shot advantage does not appear here. M13's 0.6061 was on M2 *embeddings*,
so it is not contradicted — but it was never tested against a linear baseline either. **Run with
`--features` to settle it.** The k=20 row is refused rather than fabricated: URTI has only 10 train
patients, so a "20-shot" claim would be inventing support data.

---

### 4. Calibration-Aware Honest Operating Point — ✅ RUN

**What existed.** M11 (calibration, ECE 0.446 → 0.0878) and M14 v2 (conformal, AUROC 0.4809, 0%
unknown detection at 95% coverage), on different splits, never combined. A grep for
`risk_coverage` / `selective` / `abstain` / `deferral` across the whole repo returns **nothing**.
`STEP_SEQUENCE.md` step 07, never built.

**Result (42 fit / 19 calibration / 43 test patients, concept space):**

| quantity | value |
|---|---|
| fitted temperature | **12.56** — the model is severely overconfident |
| ECE | 0.3207 → **0.1150** |
| accuracy | 0.4651 → 0.4651 (**invariant, as it must be**) |
| AURC | 0.5680 |
| selective accuracy @ coverage 100/90/80/70/50% | 0.465 / 0.474 / 0.441 / 0.467 / 0.381 |
| clinical point @ target Se 0.90 | test Se **0.9310**, Sp **0.2143**, referral rate **0.8837** |
| expected cost @ FN:FP = 10:1 | 0.7209 per patient |
| conformal empty-set rate, known vs unknown | **0.0 vs 0.0** |

**Three findings, all reportable:**

1. **The accuracy-invariance check catches a problem in M11.** Temperature scaling is monotone, so
   accuracy *cannot* change. M11 reports accuracy moving 0.3573 → 0.6595 under calibration — that
   comparison was not like-for-like and the ECE improvement there is not interpretable.
2. **Selective prediction does not work.** Accuracy does *not* rise as coverage falls (0.465 at 100%,
   0.381 at 50%). The model's confidence is uninformative about its own errors. No other model in
   the repo reports this, and it is a real result.
3. **The M14 conformal paradox, stated correctly.** Coverage is guaranteed by construction, so
   reporting it is circular. The informative number is the *empty-set rate on unknown patients* —
   it is 0.0, identical to known patients. The wrapper detects nothing. That is what M14 should have
   reported instead of 95.45% coverage.

At target sensitivity 0.90 the model refers **88% of all patients**. That is the honest operating
point, and it is not clinically usable — which is exactly the kind of statement the corrected
protocol exists to make possible.

---

### 5. Concept Leakage / Faithfulness Audit — 🟡 RUN (partial)

**What existed.** Genuinely done and careful: `owmtl/leakage.py` + notebook 03 →
`leakage = I(y;f|c) = 0.2209 bits` (13.93% of base), concepts-only accuracy 0.9092 vs
concepts+features 0.9212, patient-grouped CV over 6311 cycles, `estimator_valid: true`, verdict
*"MODERATE leakage"*. The estimator already fixed two failure modes (class-weighting destroying the
log-likelihood; a patient-level fit underpowered to the point of sign errors).

**What was missing.** No uncertainty on 0.2209. No null — a CV log-likelihood difference between a
14-d and a 782-d model has a *positive bias by construction* and nobody measured it. No per-concept
breakdown.

**Result — drop-one-concept importance (6311 cycles, 104 patients, patient-grouped CV):**

| concept | bits lost if removed |
|---|---|
| **rhonchi_presence** | **+0.0293** |
| wheeze_presence | +0.0049 |
| crackle_presence | +0.0036 |
| papr_db | +0.0033 |
| dominant_freq_hz | +0.0029 |
| inspiratory_energy_fraction | +0.0018 |
| spectral_flatness / coarse_crackle_ratio / low_high_freq_ratio | +0.0015 – 0.0016 |
| crackle_rate_hz, fine_crackle_ratio | +0.0004 – 0.0005 |
| wheeze_dominant_freq_hz, transient_timing_centroid | ≈ 0 |
| wheeze_duration_ratio | **−0.0039** (removing it *helps*) |

**Reading.** **One concept out of fourteen carries the bottleneck.** `rhonchi_presence` contributes
6× more than the next concept; five are dead (≤0.001 bits) and one is actively harmful. This is the
actionable form of the leakage result and it is new. It also explains N6 and N7: a bottleneck resting
on one concept cannot be meaningfully intervened on.

**Blocked half.** `I(y;f|c)` needs the frozen encoder features. `M2_features.npy` is not committed
(Kaggle dataset). With `--features` the script adds a patient-level bootstrap CI on the 0.2209 and a
permutation null that measures the estimator's own positive bias. **Until that runs, "MODERATE
leakage" is a point estimate with no error bar and should not be quoted as a finding.**

---

### 6. Physics-Derived Acoustic Concept Bottleneck — 🟡 RUN (partial)

**What existed.** Done — and it is the result that **failed its own pre-registered gate, twice**,
which is why `DECISION_2026-08-16_PIVOT.md` exists.

- G2 validation: `crackle_presence` AUROC 0.5556 [0.5292, 0.5811]; `wheeze_presence` 0.5818
  [0.5522, 0.6103]; full 14-vector 0.6561 / 0.5847. Gate was ≥0.65 with CI excluding chance → **FAIL**.
- Bottleneck head, official 60/40, one seed: opaque 0.5973 F1 → leaky 0.5457 → sequential 0.4441 →
  independent 0.4542. Monotone degradation as the bottleneck tightens.

**What was missing.** One seed. No random-concept control. No paired test. `config`, `efficiency` and
`training_history` all `{}` — the runs fail the project's own §4 audit.

**Result (5 seeds, 43 test patients, sklearn engine mirroring `owmtl/bottleneck.py`):**

| variant | accuracy | macro-F1 | 95% CI | seed sd |
|---|---|---|---|---|
| independent (real concepts) | 0.5814 | **0.5254** | [0.333, 0.700] | 0.059 |
| **independent (shuffled concepts)** | 0.4884 | **0.2963** | [0.197, 0.393] | — |
| sequential / leaky / opaque | *skipped* | — | — | needs `M2_features.npy` |

**The control, both metrics:**

| metric | real − shuffled | 95% CI | p | verdict |
|---|---|---|---|---|
| accuracy | +0.0930 | [−0.093, 0.279] | 0.389 | not shown to differ |
| macro-F1 | **+0.2291** | **[0.004, 0.441]** | **0.045** | **real concepts beat shuffled** |

**Reading.** This is the first evidence in the project that the physics concepts carry *something*
beyond their marginal distribution — but only on macro-F1, and only marginally (the CI lower bound is
0.004). Accuracy is dominated by COPD (64 of 104 patients) and shows nothing. **State both metrics.**
Quoting only the F1 row would be cherry-picking; quoting only accuracy would bury a real effect.

---

### 7. Clinician Concept Intervention — ✅ RUN

**What existed.** `owmtl/intervention.py` + notebook 04 → sensitivity ranking (top
`inspiratory_energy_fraction` 0.4376; `fine_crackle_ratio` **exactly 0.0** — a dead concept) and two
directed edits, one of which moves the model the wrong way (`wheeze_presence→COPD` Δp −0.0008). No
clinician anywhere: `CLINICIAN_LABELING_PACK.md` and `build_listening_pack.py` exist, **no returned
labels are committed**.

**What was missing.** The intervention *curve* — accuracy as a function of how many concepts the
clinician has corrected. That is the canonical evidence for a concept bottleneck. A ranking says the
model *reacts*; only the curve says the reaction is *correct*.

**Result (independent mode, 43 test patients, starting from "no findings recorded yet" = the train
population mean, restoring k true concept values):**

| ordering | k = 0 | k = 14 | gain |
|---|---|---|---|
| by sensitivity | 0.5814 | 0.5349 | **−0.0465** |
| random order | 0.5814 | 0.5349 | −0.0465 |
| shuffled-concept control | 0.5814 | 0.4419 | −0.1395 |

**Directed interventions:**

| assertion | Δp(COPD) | patients increased | direction |
|---|---|---|---|
| crackle_presence → COPD | +0.1639 | 88.4% | correct |
| wheeze_presence → COPD | −0.0193 | 32.6% | **wrong way** |
| rhonchi_presence → COPD | −0.2773 | 11.6% | **wrong way** |

**Reading.** **Correcting concepts does not improve the diagnosis — it slightly degrades it.** The
sensitivity ordering is indistinguishable from a random ordering. Two of three directed
interventions push the probability the wrong way. Combined with N5 (one concept carries everything)
this is coherent: a bottleneck resting on `rhonchi_presence` alone has nothing to intervene on.

**The clinician demo cannot be presented as a working mechanism.** It is a well-instrumented negative
result, which is what this project's contribution now is. A real-clinician hook is wired in: drop
`clinician_corrections.csv` (columns `patient,concept,value`) into this folder and the same curve
reruns on real corrections. Until then every number here is labelled **SIMULATED**.

---

### 8. Pediatric Physics-Fragility Analysis — 🟡 PRE-REGISTERED

**What existed.** `Barshon's/Gap7/results_Gap7_OOD.json` is **twelve lines**, MMD only: SPRSound M2
0.4236, M30 0.4093, **M35 0.4434** — the physics loss is worst on children. That single number is
the entire claimed headline. No CI, no schema fields, no downstream accuracy, no per-concept
analysis, and M35 is on the 70/30 split. Worse: M35's physics *loss* is a different object from the
14 physics *concepts*, so fragility has never been measured on the concepts at all. Note the device
axis (G4) already failed — 3/126 patients span devices — so pediatric shift is the **only** covariate
stress available.

**What `N8` adds, and why it is the strongest version of this item.** An MMD says "the distributions
differ" and cannot say why. The physics *can*: a child's airway is shorter and narrower, so its
resonant frequency is higher. That is a **directional prediction on named concepts, fixed in code
before any pediatric audio is read**:

| concept | predicted in children |
|---|---|
| `wheeze_dominant_freq_hz` | **higher** |
| `dominant_freq_hz` | **higher** |
| `low_high_freq_ratio` | **lower** |
| `rhonchi_presence` | **lower** (rhonchi are < 300 Hz) |

The other ten concepts are the control set. Tested one-sided per concept, plus a binomial sign test
over the four. If the signs come out as predicted, *"adult-tuned acoustic priors fail on pediatric
airways via frequency scaling"* becomes a **mechanistic claim** instead of a distance measurement —
and `OWMTL_Decision_Roadmap (v3).md` §183 already records this as a `[Plausible] open` novelty gap
with no prior claiming it.

**Status.** The adult reference distribution (mean/std/median for all 14 concepts over 6898 ICBHI
cycles) and the prediction table are written to `results/N8_pediatric_fragility.json`. Supplying
`--sprsound_dir` completes the test without changing a line — which is what makes this a genuine
pre-registration rather than a post-hoc story.

---

## 3. What the eight add up to

Five of the eight now have results, and **four of those five are negative**:

- concept-space OSR ties the embedding space, neither beats chance at n=19 (N2);
- the prototypical head does not beat a linear head (N3);
- confidence is uninformative, selective prediction does not work, conformal detects nothing (N4);
- one concept out of fourteen carries the bottleneck; five are dead (N5);
- correcting concepts does not improve the diagnosis, and two of three directed interventions push
  the wrong way (N7).

The one positive: **real physics concepts beat shuffled concepts on macro-F1** (+0.229, CI [0.004,
0.441], p = 0.045) — marginal, and only on one of two metrics (N6).

This is consistent with the 2026-08-16 pivot rather than a reason to reopen it. The contribution is
the *measurement*: concept-level methods on ICBHI are bounded by the reliability of its labels, and
here is the tooling that shows it, with controls, nulls and confidence intervals that the published
literature on this corpus largely does not report.

**For the supervisor:** all eight directions were implemented. Three await one input each
(`M2_features.npy`, SPRSound audio, a GPU) and every one of those is a data/compute dependency, not
a design gap. The scripts are written so that supplying the input finishes the analysis with no code
change.

---

## 4. Priority order for the remaining work

| # | Task | Effort | Unblocks | Why first |
|---|---|---|---|---|
| 1 | Commit `M2_features.npy` (768-d, 6898 rows) | minutes — it already exists on Kaggle | N5 full, N6 all modes, N2/N3/N4 embedding arms | one file, five experiments, no GPU |
| 2 | Re-run `run_all.py --features M2_features.npy` | ~10 min CPU | — | turns two PARTIALs into OK |
| 3 | N8 with SPRSound | ~1 h | the mechanistic headline | the only pre-registered mechanism in the project |
| 4 | N1 embed + lora + probe | ~40 min on a T4 | the only genuinely absent direction | needs GPU |
| 5 | Real clinician corrections CSV | depends on the listening study | N7 | converts SIMULATED → real |

---

## 5. Run log

| date | who | what ran | outcome |
|---|---|---|---|
| 2026-08-29 | audit | repo-wide audit of all eight directions | 4 done, 2 half, 2 untouched |
| 2026-08-29 | — | `test_nx.py` | 9/9 self-checks pass |
| 2026-08-29 | — | `run_all.py` (no extra inputs) | N2 ✅ N3 ✅ N4 ✅ N7 ✅ · N5 🟡 N6 🟡 · N8 pre-registered · N1 blocked |
| | | | |

> Append a row every time you run something. A number without a row here is a number nobody can
> reproduce.
