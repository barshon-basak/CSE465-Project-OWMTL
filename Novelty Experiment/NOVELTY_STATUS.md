# Novelty Experiment — status, results and progress tracker

**Owner of this file:** whoever last ran something here. Update the run log at the bottom.
**Created:** 2026-08-29 · **Last run:** 2026-08-29 (with `M2_features.npy`)
**Source of the list:** Dr. Khan's eight suggested novelty directions.
**Read with:** `README.md` (how to run), `../DECISION_2026-08-16_PIVOT.md` (why several of these
already failed their gates), `../RTK_requirements.md` (the separate course checklist — do not merge).

---

## 0. Why this folder exists

The faculty gave eight directions. An audit of the repo (2026-08-29) found that **four already had
real committed results, two were half-built in the wrong form, and two were effectively untouched** —
but scattered across eight folders, four naming schemes and two abandoned plan documents, so nobody
could answer "did we try this?" without an hour of archaeology.

This folder answers it in one place. Each of the eight has a script `N<k>_*.py`, a result in
`results/`, a docstring stating what already existed and what was missing, and a status row below.

**Framing for the supervisor meeting:** several of these directions have *already been tested and
failed* on this corpus — that is documented, pre-registered, and is the project's actual contribution
after the 2026-08-16 pivot. This folder shows the attempts were made **rigorously**, with controls
and confidence intervals. Every script refuses to state a result its CI does not support.

### One conflict, stated plainly

`../RTK_requirements.md` §12 and `../DECISION_2026-08-16_PIVOT.md` both explicitly **forbid** this
work: *"no bottleneck head, no intervention API, no concept-leakage work, no fourth idea."* Items 5–7
were run anyway (commits of 2026-08-15 and 2026-08-26). **The faculty list supersedes that stop-list.**
Written down here so nobody re-litigates it in three weeks.

---

## 1. Scoreboard

**All eight faculty items ran to completion on real data, and nothing is simulated any more.**
The clinician listening study came back on 2026-08-29 (`Asif's/labels v1 - labels.csv`, 116/132
clips answered), which closed the last data dependency and unlocked two further experiments the
project had never been able to run.

| # | Faculty item | Before this folder | Now | Script |
|---|---|---|---|---|
| 1 | Foundation-Model Concept Probing (LoRA) | 🔴 no FM in the repo at all | ✅ **complete** — AST frozen + LoRA + 2 controls | `N1_fm_concept_probing.py` |
| 2 | Concept-Space Open-Set Recognition | 🟡 embedding space only | ✅ **complete** — both spaces + paired test | `N2_concept_space_osr.py` |
| 3 | Prototypical Few-Shot Disease Head | 🟡 one fixed k, no baseline | ✅ **complete** — both spaces, shot sweep | `N3_prototypical_fewshot.py` |
| 4 | Calibration-Aware Honest Operating Point | 🟡 parts, never assembled | ✅ **complete** | `N4_honest_operating_point.py` |
| 5 | Concept Leakage / Faithfulness Audit | ✅ done, no CI/null | ✅ **complete** — CI + null + per-concept | `N5_leakage_audit.py` |
| 6 | Physics-Derived Acoustic Concept Bottleneck | ✅ done, 1 seed, no control | ✅ **complete** — all 4 modes, 5 seeds | `N6_physics_bottleneck.py` |
| 7 | Clinician Concept Intervention | 🟡 mechanism only | ✅ **complete** — 2 modes + curve, **REAL clinician** | `N7_clinician_intervention.py` |
| 8 | Pediatric Physics-Fragility | 🔴 12-line MMD stub | ✅ **complete** — 2 arms, pre-registered | `N8_pediatric_fragility.py` |
| 9 | *(support)* Clinician Label Reliability | 🔴 nothing | ✅ **complete** — real labels scored | `N9_clinician_reliability.py` |
| 11 | *(support)* **Supervised ceiling — is 0.65 reachable at all?** | 🔴 the open question | ✅ **complete — answers it: YES** | `N11_supervised_ceiling.py` |
| 10 | *(support)* Gate G2 vs an independent reference | 🔴 impossible before now | ✅ **complete** — the G2 ambiguity finally measured | `N10_gate_vs_clinician.py` |

### Inputs that were acquired to get here

| Input | How it was obtained | Unblocked |
|---|---|---|
| `M2_features.npy` (6898 × 768, 21.2 MB) | **generated** by `make_m2_features.py` from the committed M2 checkpoint + local ICBHI audio (7m46s CPU); row-alignment to `concepts_all.npz` verified | N5 full, N6 all modes, N1's M2 arm, embedding arms of N2/N3/N4/N7 |
| CUDA torch + `transformers` | installed (`torch 2.13.0+cu126`, RTX 4050 6 GB) | N1's foundation-model arm |
| `embeddings/ast_frozen.npy`, `ast_lora.npy` | AST embedding pass (~11 min each on the 4050) + 5-epoch LoRA fine-tune | N1 headline |
| SPRSound BioCAS2022 | shallow-cloned to `Desktop/SPRSound` (~718 MB) | N8 both arms |

### Synthetic-data audit (2026-08-29)

Every script was checked for synthetic data standing in for real data. **One substitution exists.**

| Usage | Where | Verdict |
|---|---|---|
| `random_control` probe arm | N1 | **Control**, not a stand-in — it is what shows the real probe is not a high-dimensional artefact. Keep. |
| `torch.randn(...) * 0.01` | N1 | LoRA A-matrix **initialisation**. Not data. |
| `rng.permutation` / `rng.choice` | N5, N6, N7, N8, `common.py` | Bootstrap resampling and permutation nulls **applied to real data**. Keep. |
| shuffled-concept controls | N6, N7 | Deliberate controls. Keep. |
| `--dry-run`, `--selftest`, `test_nx.py` | N1, N9, tests | Wiring checks that write **no result**. Keep. |
| **DSP values used as a clinician oracle** | **N7** | **The one real substitution.** Blocked on the returned listening sheet — see N9. |

`common.load_concepts()` raises rather than substituting random data if `concepts_all.npz` is absent,
so no experiment can silently fall back to synthetic input.

### ✅ The clinician study is in

`Asif's/labels v1 - labels.csv` — **116 of 132 clips answered**. The 16 blanks are exactly the 16
clips the rater marked `unusable`, so they are a deliberate answer rather than missing data. Two
schema deviations were handled explicitly rather than absorbed: `ambiguous` appears 12 times and is
treated as `unsure` (excluded from agreement, never coerced to a class), and audio quality is used
for a sensitivity analysis rather than a silent filter.

This closed the last simulated component (N7) and made two new experiments possible — N9 and N10.

> **Do not use `owmtl.m2_features.export_features` for the M2 checkpoint.** Its `default_logmel`
> applies a plain `power_to_db` with no normalisation, but M2 was trained on
> `power_to_db(ref=np.max)` followed by per-sample min-max to [0, 1] with wrap-padding. The
> embeddings would load without error and be silently wrong. `make_m2_features.py` reproduces M2's
> actual preprocessing and refuses to write unless its row order matches `concepts_all.npz`.

---

## 2. Item-by-item

### 1. Foundation-Model Concept Probing (LoRA) — ✅ COMPLETE (AST frozen + LoRA + 2 controls)

**What existed.** `Barshon's/M37_v2` is titled *"Audio Spectrogram Transformer LoRA PEFT"* but
instantiates `M37_LoRA_CNN` wrapping the project's own scratch-trained `M2_CNN` — 3,635,700 params,
8,356 trainable, LoRA on two FC layers. **No foundation model exists anywhere in the repo** (grep for
OPERA / M2D / HeAR / CLAP / AudioMAE / BEATs / wav2vec / PANNs returns nothing). M4 (AST) has no
committed results JSON. Concept *probing* had never been attempted at all.

**Result — linear probe of the 14 physics concepts from the frozen M2 encoder** (6898 cycles,
patient-grouped CV, ridge R², bootstrap CI, within-patient permutation null):

| concept | R² | 95% CI | | concept | R² | 95% CI |
|---|---|---|---|---|---|---|
| wheeze_duration_ratio | **0.777** | [0.766, 0.786] | | papr_db | 0.520 | [0.498, 0.540] |
| spectral_flatness | **0.759** | [0.717, 0.790] | | crackle_presence | 0.501 | [0.484, 0.516] |
| rhonchi_presence | **0.648** | [0.632, 0.664] | | coarse_crackle_ratio | 0.366 | [0.351, 0.383] |
| wheeze_presence | **0.605** | [0.586, 0.623] | | low_high_freq_ratio | 0.363 | [0.313, 0.406] |
| dominant_freq_hz | 0.548 | [0.479, 0.621] | | crackle_rate_hz | 0.356 | [0.328, 0.381] |
| wheeze_dominant_freq_hz | 0.289 | [0.247, 0.328] | | transient_timing_centroid | 0.276 | [0.258, 0.295] |
| inspiratory_energy_fraction | 0.003 | [−0.025, 0.030] | | fine_crackle_ratio | **−0.008** | [−0.117, 0.036] |

**Random-projection control: 0 of 14 above chance, mean R² = −0.0054.**

**Reading — this is the folder's strongest positive result, and it sharpens the whole project.**
The encoder **does** linearly encode 12 of 14 physics concepts, some at R² > 0.75, while the control
sits flat at zero. So the network is not ignoring the clinical acoustics — it represents them well.
Combined with N5 (the concepts carry almost no *disease* information) the picture becomes precise:

> The model hears the sounds. The sounds do not predict the ICBHI disease labels.

That is a much stronger and more defensible statement than "our extractors are weak", and it is
exactly the label-reliability argument the pivot document makes, now with direct evidence.

`fine_crackle_ratio` is unrecoverable (R² ≈ 0) — consistent with the pivot record that it was 85%
zero on real audio and was targeting a distinction physicians themselves agree on at κ < 0.40.

**FOUNDATION-MODEL ARM — RUN.** Frozen AST (`MIT/ast-finetuned-audioset-10-10-0.4593`, 86.5 M params)
embedded all 6,898 cycles on the RTX 4050 (~11 min), then LoRA (r=8, q_proj+v_proj on all 12 layers,
**294,912 trainable = 0.341%**) was fine-tuned on the 4-class ICBHI task for 5 epochs (loss
1.0003 → 0.5627), and the same probe was re-run on the adapted embedding.

| space | concepts above chance | mean probe R² |
|---|---|---|
| **AST frozen** | **13 / 14** | **0.4102** |
| **AST + LoRA** | 13 / 14 | **0.3195** |
| M2 CNN (trained from scratch on this task) | 12 / 14 | 0.4289 |
| random-projection control | **0 / 14** | −0.0054 |

**LoRA effect (adapted − frozen): 13 of 14 concepts DEGRADE, mean Δ R² = −0.0907.**

| most degraded | Δ R² | | least affected | Δ R² |
|---|---|---|---|---|
| **`rhonchi_presence`** | **−0.219** | | `wheeze_dominant_freq_hz` | −0.033 |
| `dominant_freq_hz` | −0.183 | | `inspiratory_energy_fraction` | −0.006 |
| `spectral_flatness` | −0.154 | | `fine_crackle_ratio` | +0.008 |
| `papr_db` | −0.142 | | | |
| `crackle_presence` | −0.128 | | | |

**Reading — this is the strongest single result in the folder.**

1. **A foundation model pre-trained on AudioSet already encodes the clinical acoustics** (13/14 above
   chance) without ever seeing a respiratory corpus. The random control confirms this is not a
   high-dimensional artefact.
2. **Fine-tuning on the ICBHI task systematically destroys that encoding.** Adapting only 0.34% of
   the parameters is enough to strip nearly a tenth of the recoverable concept variance across
   almost every concept. The model bought its task performance by *discarding* clinically meaningful
   structure.
3. **The concept it destroys most is `rhonchi_presence` (−0.219)** — which N5 independently
   identifies as the *only* concept carrying disease information (6× the next). Two experiments
   built on different data and different estimators converge on the same concept from opposite
   directions.

That is a faithfulness finding with a mechanism, and it is exactly the argument
`DECISION_2026-08-16_PIVOT.md` makes, now with direct evidence rather than inference.

`fine_crackle_ratio` is unrecoverable everywhere (R² ≈ 0) — consistent with the pivot record that it
was 85% zero on real audio and targeted a distinction physicians agree on at κ < 0.40.

---

### 2. Concept-Space Open-Set Recognition — ✅ COMPLETE

**What existed.** M29 on frozen M12 **embeddings**, Energy AUROC 0.6466, CI spanning chance at n=19.
M38 (large-N) has a generator and notebook but **was never run**. The concept-space arm was never built.

**Result** (104 known / 19 unknown patients, official 60/40, patient level, unknown group never fitted):

| space | msp | entropy | **energy** | mahalanobis | knn |
|---|---|---|---|---|---|
| concept 14-d | 0.612 | 0.622 | **0.627** | 0.520 | 0.525 |
| embedding 768-d | 0.555 | 0.551 | 0.603 | 0.486 | 0.514 |
| concat | 0.611 | 0.613 | **0.641** | 0.479 | 0.519 |

Best: concat / energy **0.6414** [0.496, 0.779]. Every CI spans chance.

**Paired bootstrap, concept − embedding** (same resample for both — the only valid comparison):

| detector | Δ AUROC | 95% CI | p |
|---|---|---|---|
| entropy | +0.071 | [−0.108, 0.240] | 0.405 |
| msp | +0.058 | [−0.122, 0.225] | 0.503 |
| mahalanobis | +0.034 | [−0.089, 0.171] | 0.625 |
| energy | +0.023 | [−0.164, 0.209] | 0.802 |
| knn | +0.011 | [−0.135, 0.162] | 0.895 |

**Reading.** **A 14-dimensional, clinically-named score matches a 768-dimensional opaque one** —
every paired difference is small and none is distinguishable from zero. Concept space is *numerically
ahead on all five detectors*, but that ordering is well inside noise and must not be reported as a
win. At n = 19 unknown patients no CI can exclude chance; that is a property of the corpus, not of
the method.

This is a genuinely publishable calibration: **you pay nothing in detection performance for full
interpretability of the open-set score**, and the corpus cannot support a stronger claim than that.

---

### 3. Prototypical Few-Shot Disease Head — ✅ COMPLETE

**What existed.** M13 v4 — real work: nearest-class-mean over frozen M2/M12 embeddings, patient-level,
accuracy 0.7209 / macro-F1 0.6061. But one fixed k, no baseline, and it reports `icbhi_score` 0.7137,
a metric **undefined** for a 3-class disease task (official ICBHI score is (Se+Sp)/2 over the 4-class
sound-event task).

**Result** (61 train / 43 test patients, 200 episodes per k, paired against a linear head on the
identical support set):

| k | concept proto | concept linear | **M2 proto** | **M2 linear** | verdict (M2) |
|---|---|---|---|---|---|
| 1 | 0.309 | 0.293 | 0.424 | 0.394 | not shown to differ |
| 2 | 0.312 | 0.315 | 0.441 | 0.417 | not shown to differ |
| 5 | 0.303 | 0.359 | 0.483 | 0.485 | not shown to differ |
| 10 | 0.262 | 0.424 | 0.505 | 0.525 | not shown to differ |
| 20 | *refused* | — | *refused* | — | k > rarest train class (10 patients) |
| all-train | 0.231 | 0.427 | **0.548** [0.486, 0.596] | 0.502 | — |

**Reading.** The M2-embedding arm **reproduces M13** (0.548 vs M13's 0.6061, overlapping at n=43) —
so M13's number is sound. But across every k, **prototypical is statistically indistinguishable from
a plain linear head fitted on the same support patients.** The classic few-shot advantage does not
appear on this task in either space. In concept space the linear head actually wins by k=10.

The k=20 row is **refused rather than fabricated**: URTI has only 10 training patients, so a
"20-shot" claim would be inventing support data.

---

### 4. Calibration-Aware Honest Operating Point — ✅ COMPLETE

**What existed.** M11 (calibration) and M14 v2 (conformal, AUROC 0.4809, 0% unknown detection at 95%
coverage) on different splits, never combined. A grep for `risk_coverage` / `selective` / `abstain` /
`deferral` across the whole repo returns **nothing**.

**Result** (M2 embedding space; 42 fit / 19 calibration / 43 test patients):

| quantity | value |
|---|---|
| fitted temperature | **4.89** — severely overconfident |
| ECE / MCE | 0.4049 → **0.1771** / 0.743 → 0.483 |
| Brier / NLL | 0.808 → 0.569 / 2.093 → 0.885 |
| accuracy | 0.5116 → 0.5116 (**invariant, as it must be**) |
| AURC | **0.3424** |
| selective accuracy @ coverage 100/90/80/70/50% | 0.512 / 0.579 / **0.618** / 0.600 / 0.619 |
| clinical point @ target Se 0.90 | test Se **0.8966** [0.759, 1.000], Sp **0.5714**, referral **0.744** |
| expected cost @ FN:FP = 10:1 | 0.8372 per patient |
| conformal empty-set rate, known vs unknown | **0.0 vs 0.0** (mean set size 1.70 vs 1.84) |

**Three findings, all reportable:**

1. **The accuracy-invariance check catches a problem in M11.** Temperature scaling is monotone, so
   accuracy *cannot* change. M11 reports 0.3573 → 0.6595 under calibration. That comparison was not
   like-for-like and its ECE improvement is not interpretable. Ours holds at 0.5116 exactly.
2. **Selective prediction works in embedding space but not in concept space.** Accuracy rises
   0.512 → 0.618 as coverage drops to 80% on M2 embeddings; in concept space it *fell* (0.465 →
   0.381). So the encoder's confidence is informative about its own errors and the concept model's
   is not — the first evidence in this project that abstention could buy anything.
3. **The M14 conformal paradox, stated correctly.** Coverage is guaranteed by construction, so
   reporting it is circular. The informative number is the *empty-set rate on unknown patients*: it
   is 0.0, identical to known patients, with unknown patients getting *larger* prediction sets
   (1.84 vs 1.70) rather than empty ones. The wrapper detects nothing. That is what M14 should have
   reported instead of 95.45% coverage.

At the sensitivity a clinician would demand, the model still refers **74% of all patients**. That is
the honest operating point.

---

### 5. Concept Leakage / Faithfulness Audit — ✅ COMPLETE

**What existed.** Genuinely done and careful: `leakage = I(y;f|c) = 0.2209 bits` (13.93% of base),
concepts-only accuracy 0.9092 vs concepts+features 0.9212, patient-grouped CV over 6311 cycles,
`estimator_valid: true`, verdict *"MODERATE leakage"*. Missing: no CI, no null (the estimator has a
positive bias by construction when going from 14 to 782 predictors, never measured), no per-concept
breakdown.

**Result — the headline, now with an interval and a null** (6,311 cycles, 104 patients,
patient-grouped CV, features regenerated independently by `make_m2_features.py`):

| quantity | value |
|---|---|
| leakage I(y;f\|c) | **0.2025 bits** (12.77% of base) — committed run said 0.2209 |
| patient-bootstrap CI | **[0.1096, 0.3292]** — **excludes zero** |
| permutation null (features shuffled across patients) | **−0.0086** [−0.0149, −0.0013], n=5 |
| verdict | **leakage exceeds the estimator's own bias** |
| accuracy: concepts / concepts+features / features | 0.9086 / 0.9303 / 0.9287 |
| `estimator_valid` | true |

Two things worth stating:

- **The independent regeneration reproduces the committed number to within 0.02 bits** (0.2025 vs
  0.2209), from a feature matrix rebuilt from the checkpoint with different code. That is a real
  reproducibility check, and it passed.
- **The permutation null is slightly NEGATIVE, not positive.** The worry that a 782-predictor model
  would beat a 14-predictor one by construction turns out to be unfounded for this estimator — the
  inner log-loss CV shrinks the extra dimensions toward the prior. So "MODERATE leakage" survives
  scrutiny: 0.2025 bits against a null of −0.009, with a CI excluding zero.

**Result — drop-one-concept importance** (same run):

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

**Reading. One concept out of fourteen carries the bottleneck.** `rhonchi_presence` contributes 6×
the next concept; five are dead (≤ 0.001 bits) and one is actively harmful. This is the actionable
form of the leakage result, it is new, and it explains N6 and N7: a bottleneck resting on one
concept has nothing to intervene on.

**Cost note.** The permutation null refits `LogisticRegressionCV` on ~780 dimensions per replicate
(5 outer folds × 10 Cs × 5 inner folds). The default was lowered from 20 to 5 — 20 runs for hours.
The null *mean* is well pinned at 5 replicates; the null *CI* is indicative only. Raise `--n_perm`
before quoting an interval on the null itself.

---

### 6. Physics-Derived Acoustic Concept Bottleneck — ✅ COMPLETE, and the committed result does not replicate

**What existed.** The result that **failed its pre-registered gate, twice** (why the pivot exists):
G2 validation `crackle_presence` AUROC 0.5556 [0.529, 0.581], `wheeze_presence` 0.5818 [0.552, 0.610]
against a ≥0.65 threshold. The bottleneck head, **one seed**, reported a clean monotone story:

> opaque 0.5973 → leaky 0.5457 → sequential 0.4441 → independent 0.4542 (macro-F1),
> "interpretability cost 0.2326 accuracy / 0.1432 macro-F1"

**Result (5 seeds, stratified bootstrap CIs, 43 test patients, all four modes):**

| mode | accuracy | macro-F1 | 95% CI | seed sd | committed F1 |
|---|---|---|---|---|---|
| **independent** (true bottleneck) | 0.5814 | **0.5254** | [0.333, 0.700] | 0.059 | 0.4542 |
| leaky | 0.6512 | 0.4802 | [0.385, 0.554] | 0.021 | 0.5457 |
| sequential | **0.6744** | 0.4526 | [0.358, 0.540] | 0.032 | 0.4441 |
| opaque (upper bound) | 0.5814 | **0.4378** | [0.332, 0.521] | 0.019 | 0.5973 |
| *shuffled-concept control* | 0.4884 | 0.2963 | [0.197, 0.393] | — | *never run* |

**⚠️ The monotone ordering inverts.** With five seeds instead of one, the strict bottleneck
(`independent`) has the **highest** macro-F1 and the unconstrained `opaque` model the **lowest** —
the opposite of the committed run. McNemar independent vs opaque: **p = 1.0000**. The claimed
"interpretability cost of 0.2326 accuracy / 0.1432 macro-F1" was a **single-seed artefact**, and the
corrected statement is that at n=43 there is **no measurable interpretability cost in either
direction**.

This is the most important correction in the folder. It does not rescue the bottleneck — nothing here
performs well — but the specific claim "forcing the diagnosis through concepts costs accuracy" is not
supported by the data once seed variance is accounted for, and it should not go into the paper.

**The control the committed run never had:**

| metric | real − shuffled concepts | 95% CI | p | verdict |
|---|---|---|---|---|
| accuracy | +0.0930 | [−0.093, 0.279] | 0.389 | not shown to differ |
| macro-F1 | **+0.2291** | **[0.004, 0.441]** | **0.045** | **real concepts beat shuffled** |

The physics concepts do carry something beyond their marginal distribution — but only on macro-F1 and
only marginally (CI lower bound 0.004). Accuracy is dominated by COPD (64 of 104 patients) and shows
nothing. **Report both metrics**; quoting either alone misleads.

---

### 7. Clinician Concept Intervention — ✅ COMPLETE (both modes, REAL clinician)

**What existed.** Sensitivity ranking (top `inspiratory_energy_fraction` 0.4376; `fine_crackle_ratio`
exactly 0.0) and two directed edits, one moving the model the wrong way — recorded but not flagged.
No clinician anywhere: the listening pack exists, **no returned labels are committed**. And no
intervention *curve*, which is the canonical evidence for a concept bottleneck.

**Result — intervention curve** (start from "no findings recorded yet" = train population mean,
restore k true concept values, 43 test patients):

| ordering | independent k=0 → k=14 | leaky k=0 → k=14 |
|---|---|---|
| by sensitivity | 0.5814 → 0.5349 (**−0.047**) | 0.6512 → 0.6279 (**−0.023**) |
| random order | 0.5814 → 0.5349 (−0.047) | 0.6512 → 0.6279 (−0.023) |
| shuffled-concept control | 0.5814 → 0.4419 (−0.140) | 0.6512 → 0.6512 (0.000) |

**Directed interventions** (clinician asserts a finding, effect on p(COPD)):

| assertion | independent | leaky |
|---|---|---|
| crackle_presence → COPD | +0.164 (88.4% ↑) ✓ | −0.010 (11.6% ↑) ✗ |
| wheeze_presence → COPD | −0.019 (32.6% ↑) ✗ | −0.000 (27.9% ↑) ✗ |
| rhonchi_presence → COPD | −0.277 (11.6% ↑) ✗ | +0.001 (30.2% ↑) ✓ |

**Reading.**

1. **Correcting concepts does not improve the diagnosis — it slightly degrades it**, in both modes.
   The sensitivity ordering is *identical* to a random ordering.
2. **The leaky mode bypasses the concept layer almost entirely.** Per-concept sensitivity collapses
   from ~0.70 (independent) to ~0.03 (leaky) — a 20× drop. Given the option, the model routes around
   the concepts. That is a direct, quantitative demonstration of bottleneck bypass and it is the
   cleanest evidence for the leakage argument in the whole project.
3. Five of six directed interventions across the two modes push the probability the **wrong way**.

**The clinician demo cannot be presented as a working mechanism.** It is a well-instrumented negative
result.

**Now run with REAL clinician corrections** (N9 writes `clinician_corrections.csv` automatically):
84 corrections across 42 patients, of which **30 landed on test patients**. Substituting the
clinician's own crackle/wheeze judgements for the DSP values moves accuracy **0.5349 → 0.3953**.

Read this carefully — it is *not* "the clinician is wrong". The concept space was built to a DSP
convention aligned with ICBHI's annotation, and N9 shows the clinician disagrees with that
convention at κ = 0.035 on crackles. Substituting a genuinely independent human judgement therefore
moves the model off the operating point it was fitted to. The drop is another measurement of the
same label-reliability gap, arriving from a third direction.

---

### 8. Pediatric Physics-Fragility Analysis — ✅ COMPLETE (pre-registered, both arms)

**What existed.** `Barshon's/Gap7/results_Gap7_OOD.json` is **twelve lines**, MMD only: SPRSound M2
0.4236, M30 0.4093, **M35 0.4434**. No CI, no schema, no downstream accuracy, no per-concept
analysis, and M35 is on the 70/30 split. Worse: M35's physics *loss* is a different object from the
14 physics *concepts*, so fragility was never measured on the concepts. The device axis (G4) already
failed — 3/126 patients span devices — so pediatric shift is the **only** covariate stress available.

**What N8 adds.** An MMD says the distributions differ and cannot say why. The physics can: a child's
airway is shorter and narrower, so resonant frequency is higher. That is a **directional prediction
on named concepts, fixed in code before any pediatric audio is read**:

| concept | predicted in children |
|---|---|
| `wheeze_dominant_freq_hz` | **higher** |
| `dominant_freq_hz` | **higher** |
| `low_high_freq_ratio` | **lower** |
| `rhonchi_presence` | **lower** (rhonchi are < 300 Hz) |

The other ten concepts are the control set. One-sided Mann-Whitney per concept, plus a binomial sign
test over the four. If the signs come out as predicted, *"adult-tuned acoustic priors fail on
pediatric airways via frequency scaling"* becomes a **mechanistic claim** instead of a distance
measurement — and `OWMTL_Decision_Roadmap (v3).md` §183 already logs this as a `[Plausible] open`
novelty gap with no prior claiming it.

**A SECOND, CONFOUND-FREE ARM.** The cross-corpus comparison is confounded — different corpus,
stethoscopes, protocol and annotators could each move a spectral concept. But SPRSound filenames
encode **age** (`<patient>_<age>_<sex>_<position>_<record>.wav`), which allows the same physics to be
tested *within* the pediatric cohort, where all of those are held fixed. If resonant frequency scales
inversely with airway size, frequency concepts must fall with age **inside** the cohort too. The
gradient predictions are derived in code from the cross-corpus ones so the two cannot drift apart.

**Data.** SPRSound BioCAS2022: 1,949 records → **6,656 annotated respiratory events, 243 children,
ages 0.2–16.2 y (median 4.4)**. 177 "Poor Quality" records dropped. Events are cut from the JSON
start/end times, matching ICBHI's raw unpadded annotated cycles — using arbitrary fixed windows would
compare one breath against several and measure segmentation rather than airways. n is close to
ICBHI's 6,898 cycles.

**Result — cross-corpus (adult ICBHI → pediatric SPRSound), Cohen's d:**

| concept | d | 95% CI | adult-vs-child AUROC | pre-registered |
|---|---|---|---|---|
| `dominant_freq_hz` | **+1.382** | [1.241, 1.558] | 0.930 | **higher ✓** |
| `low_high_freq_ratio` | **−0.512** | [−0.574, −0.472] | 0.126 | **lower ✓** |
| `rhonchi_presence` | **−3.870** | [−3.959, −3.784] | 0.015 | **lower ✓** |
| `wheeze_dominant_freq_hz` | −0.179 | [−0.207, −0.149] | 0.545 | higher ✗ |

**Result — within-cohort age gradient (the confound-free arm), Spearman ρ vs age, patient bootstrap:**

| concept | ρ | 95% CI | pre-registered |
|---|---|---|---|
| `dominant_freq_hz` | **−0.259** | [−0.348, −0.156] | **negative ✓** |
| `wheeze_dominant_freq_hz` | **−0.240** | [−0.310, −0.159] | **negative ✓** |
| `low_high_freq_ratio` | **+0.300** | [0.187, 0.399] | **positive ✓** |
| `rhonchi_presence` | −0.090 | [−0.152, −0.031] | positive ✗ |

Concept-space MMD² = **0.4805** [0.457, 0.502] (Gap7 reported an embedding MMD of 0.4236 with no CI).

**⚠ RESULT REVISED (prediction set extended from 4 to 8).** The original 4-prediction sign test
floors at p = 0.0625 and could never have supported the hypothesis whatever the data showed — a test
with no power to pass. Four more predictions were added, derived from the same two mechanisms
(frequency scaling; allometric respiratory rate), which lifts the floor to 0.0039. **With a test that
can actually pass, the mechanism does not reach significance.**

| arm | confirmed (CI excludes 0) | contradicted | sign test | verdict |
|---|---|---|---|---|
| within-cohort age gradient | 5/7 | 2 | 5/7, p = 0.227 | **MIXED** |
| cross-corpus | 5/8 | 3 | 5/8, p = 0.363 | **MIXED** |
| round-1 pre-registered only | 3/4 both arms | — | p = 0.3125 (floor 0.0625) | underpowered |

**What holds.** The three *frequency* concepts confirm consistently in the confound-free age arm —
`dominant_freq_hz` ρ = −0.259 [−0.348, −0.156], `wheeze_dominant_freq_hz` −0.240 [−0.310, −0.159],
`low_high_freq_ratio` +0.300 [0.187, 0.399]. That is the pattern inverse frequency scaling predicts,
with corpus, device, protocol and annotator all held fixed.

**What does not.** The crackle-morphology predictions fail: `fine_crackle_ratio` is degenerate and
not estimable, `coarse_crackle_ratio` contradicts cross-corpus, `wheeze_duration_ratio` contradicts
within-cohort, and `rhonchi_presence` contradicts in the age arm while confirming cross-corpus.

**Honest summary for the paper:** *the frequency concepts behave as the physics predicts and the
crackle-morphology concepts do not; a mechanism claim needs more named predictions or a second
pediatric corpus.* Registration rounds are tracked in code (`PREDICTIONS[concept] = (direction,
round, mechanism)`) and reported separately, so round-2 additions are never laundered as
pre-registered.

Concept-space MMD² = **0.4805** [0.457, 0.502] (Gap7 reported an embedding MMD of 0.4236, no CI).

---

### 9. Clinician Label Reliability — ✅ COMPLETE

**The listening study came back and it is the project's single most important measurement.**
One clinician, blind to ICBHI, 116/132 clips answered, 12 hidden duplicates.

| quantity | crackles | wheeze |
|---|---|---|
| vs ICBHI, Cohen's κ | **0.035** [−0.157, 0.210] *(slight)* | **0.266** [0.036, 0.444] *(fair)* |
| vs ICBHI, raw agreement | 0.528 [0.421, 0.639] | 0.718 [0.610, 0.832] |
| sensitivity vs ICBHI | **0.231** | 0.297 |
| specificity vs ICBHI | 0.804 | 0.932 |
| confusion (TP/FP/FN/TN) | 12 / 11 / 40 / 45 | 11 / 5 / 26 / 68 |
| **intra-rater κ (hidden duplicates)** | **0.714** *(substantial)* | **0.600** *(substantial)* |
| n | 108 answers, 42 patients | 110 answers, 42 patients |

Sensitivity analysis on `ok`-quality audio only: κ 0.023 / 0.255 — essentially unchanged, so audio
quality is not driving the result.

**The decisive contrast is the last row against the first.** The rater agrees with *themselves* at
κ = 0.71 / 0.60 but with ICBHI at κ = 0.04 / 0.27. The disagreement is therefore **not rater noise** —
it is a genuine discrepancy with the reference standard. That is the project's central claim, and it
is now measured on its own data rather than inherited from citations.

**An independent replication of the published benchmark.** Our clinician's crackle sensitivity
against ICBHI is **0.231**; Tzeng et al. report **0.2323** for seven senior physicians on this same
corpus. Our rater missed 40 of 52 ICBHI-positive crackle cycles — 77% — matching the published
"~77% of abnormal cycles missed" almost exactly. One rater, a different sample, an independently
built listening pack, and the same number.

**Released asset (Gate G0's ceiling lever).** `release_annotations/` now holds
`ICBHI_clinician_annotations_v1.csv` (132 cycles, clinician labels joined to the ICBHI reference)
plus a README documenting provenance, schema, the `ambiguous` deviation, measured reliability and
four limitations. To our knowledge these are the first fine-grained (fine-vs-coarse crackle)
reference labels released for ICBHI — which is exactly what the roadmap's G0 identified as the only
lever that lifts the venue ceiling.

**Limitations, stated plainly.** One rater, so no inter-rater agreement is computable. The sample is
not random — the pack excluded cycles under 0.9 s and sorted each stratum longest-first. Fine-vs-
coarse has only 19 fine and 3 coarse answers. These labels are an independent second opinion, not a
correction of ICBHI; where they disagree, neither is established as correct.

---

### 10. Gate G2 Re-Run Against an Independent Reference — ✅ COMPLETE

**The experiment the project could never run.** `DECISION_2026-08-16_PIVOT.md` recorded an
ambiguity it explicitly could not resolve: *"Our extractors are weak — TRUE, demonstrated twice"*
versus *"the 0.65 target is reachable on this reference standard by any method — UNKNOWN, and this
experiment cannot decide it."* Every validation to date used the same reference standard whose
reliability was in question. The clinician labels supply a second one.

Same extractors, same 132 clips, two reference standards. The clip→concept-row mapping is not
assumed but **verified**: `clip_key.csv` stores ICBHI's own labels, so a correct mapping must
reproduce them — all 132 did, and the script refuses to run otherwise.

| concept | vs ICBHI | vs clinician | paired difference |
|---|---|---|---|
| `crackle_presence` | 0.5251 [0.415, 0.630] | 0.5581 [0.417, 0.693] | **+0.033** [−0.137, 0.196] |
| `wheeze_presence` | 0.7360 [0.621, 0.843] | 0.8334 [0.723, 0.928] | **+0.097** [−0.045, 0.250] |

**Verdict: UNDECIDED.** Both differences point the same way — the extractors track the clinician
slightly better than they track ICBHI — but neither interval excludes zero. The pivot document's
ambiguity **stands as written**, now with a measurement behind it instead of an assumption.

> ⚠️ **The absolute AUROCs here must not be compared with G2's 0.5556 / 0.5818.** The listening pack
> is not a random sample: cycles under 0.9 s were dropped and each stratum sorted longest-first, so
> these clips are systematically easier. `wheeze_presence` clears 0.65 on this subset **and that does
> not overturn the G2 failure.** Only the paired difference — same score, same clips, two references,
> so selection bias cancels — is a valid comparison. The script prints this warning on every run and
> stores it in the JSON.

**First fine/coarse validation: not estimable.** `fine_crackle_ratio` is *constant* across all 22
fine-or-coarse clips (it is 99.7% zero corpus-wide), so no AUROC exists. Reporting it as "AUROC
0.500" would read as a precise finding of no signal when it is actually no measurement at all. This
is now the fourth independent line of evidence against that concept.

---

## 3. What the eight add up to

**Two clean positives, and they are the paper.**

> **1. Models hear the clinical sounds; the sounds do not predict the ICBHI disease labels.**
> A frozen AudioSet foundation model encodes 13/14 physics concepts (mean R² 0.41) having never seen
> a respiratory corpus, and the random control recovers nothing (N1). Yet those concepts support
> almost no diagnosis: one of fourteen carries the bottleneck, five are dead, one is harmful (N5);
> the bottleneck is not interventionable (N7); the prototypical head ties a linear one (N3).
>
> **2. Task adaptation destroys the encoding — and it destroys the diagnostic concept most.**
> LoRA on 0.34% of AST's parameters degrades 13/14 concepts (mean ΔR² −0.091), worst of all
> `rhonchi_presence` (−0.219) — independently identified by N5 as the *only* concept carrying disease
> information. Given the option to bypass the concept layer entirely, the bottleneck does: N7's
> leaky mode drops per-concept sensitivity 20×.

**A late fourth result overturns the framing of all of them (2026-08-30):**

> **N11 — the labels are learnable, so the ceiling argument is retracted.** A logistic probe on the
> frozen AudioSet AST embedding, fit on train patients and scored on the same 2,636 test cycles G2
> failed on, reaches **AUROC 0.7115 (crackle) / 0.7621 (wheeze)** — well above the 0.65 gate, from a
> network that never saw a respiratory corpus. The 14 DSP concepts *as a vector* reach 0.6608 on
> crackle, above the threshold the single G2 scalar failed. Reading (b) of the pivot document — "0.65
> may be unreachable here" — is dead. Our extractors were weak, and the paper now says so. What
> replaces the ceiling claim is sharper: **machines reproduce these labels far better than a trained
> physician agrees with them** (κ 0.035 crackle, Se 23.1%). See `../DECISION_2026-08-30_CONSOLIDATION.md`.

**A third, mechanistic result stands on its own:**

> **3. The frequency concepts shift with airway size as the physics predicts — but the mechanism is
> not established.** In a confound-free within-cohort age gradient across 243 children, all three
> frequency concepts move as predicted with intervals excluding zero. Once the prediction set is
> extended from 4 to 8 (giving the sign test power it previously lacked), the overall test does not
> reach significance: 5/7, p = 0.227. Report the confirming concepts individually; do not claim the
> mechanism (N8).

Supporting and corrective results:

- concept-space OSR **matches** embedding-space OSR — full interpretability at no measured cost —
  but neither beats chance at n=19 (N2);
- the model is severely overconfident (T=4.89); selective prediction works on embeddings but not on
  concepts; conformal detects nothing, empty-set rate 0.0 on known *and* unknown (N4);
- **the committed "interpretability cost" does not replicate across seeds and its sign inverts** (N6)
  — a claim that must not ship as written;
- real physics concepts do beat shuffled ones, but only on macro-F1 and marginally (+0.229, CI
  [0.004, 0.441], p = 0.045) (N6);
- leakage 0.2025 bits reproduces the committed 0.2209 from independently regenerated features, with
  a CI excluding zero and a null of −0.009 (N5).

This is consistent with the 2026-08-16 pivot rather than a reason to reopen it — but N1 and N8 are
genuinely new, positive, and mechanistic, which the pivot did not anticipate.

---

## 4. What is left

| # | Task | Effort | Why |
|---|---|---|---|
| 1 | Fold N6's non-replication into the paper | writing | a committed claim that does not survive seed variance must not ship |
| 2 | ~~Get a second clinician~~ | — | **dropped 2026-08-30.** N11 bounds the reference standard directly, which is the only job the second rater had. Single-rater is a stated limitation now, not a blocker |
| 3 | Raise N8's pre-registered predictions above 4 | design | with 4 predictions the binomial sign test floors at p = 0.0625 and can never reach significance; more named predictions would fix the power, not just the phrasing |
| 4 | N5 null at `--n_perm` ≥ 20 | ~2 h CPU | the null *mean* is solid at 5; only quote a null *interval* after this |
| 5 | Re-run N1's LoRA at other ranks / epochs | ~1 h each on the 4050 | tests whether concept destruction scales with adaptation strength — the obvious follow-up to the headline |

---

## 5. Run log

| date | who | what ran | outcome |
|---|---|---|---|
| 2026-08-29 | audit | repo-wide audit of all eight directions | 4 done, 2 half, 2 untouched |
| 2026-08-29 | — | `test_nx.py` | 9/9 self-checks pass |
| 2026-08-29 | — | `run_all.py` (no extra inputs) | N2 ✅ N3 ✅ N4 ✅ N7 ✅ · N5 🟡 N6 🟡 · N8 pre-registered · N1 blocked |
| 2026-08-29 | — | `make_m2_features.py --audio_dir <ICBHI>` | **M2_features.npy (6898×768, 21.2 MB) generated**; alignment to concepts_all.npz verified; 0 unreadable cycles; 7m46s CPU |
| 2026-08-29 | — | N1/N2/N3/N4 `--features` | N1: **12/14 concepts recoverable from M2, control 0/14**. N2: concept ≈ embedding, all paired diffs null. N3: proto ≈ linear at every k. N4: T=4.89, selective prediction works on embeddings |
| 2026-08-29 | — | N6 `--features` (all 4 modes) | **committed monotone ordering INVERTS**; McNemar p=1.0; shuffled control p=0.045 on F1 |
| 2026-08-29 | — | N7 `--features --mode leaky` + independent | curve flat/negative in both; **leaky sensitivity collapses 20×** → bypass demonstrated |
| 2026-08-29 | — | N5 `--features` (5 perms) | leakage **0.2025** bits CI[0.110, 0.329] excludes zero; null **−0.0086** → exceeds its own bias; reproduces committed 0.2209 |
| 2026-08-29 | — | installed CUDA torch 2.13.0+cu126 + transformers 5.16.1 | RTX 4050 (6 GB) live |
| 2026-08-29 | — | cloned SPRSound BioCAS2022 → `Desktop/SPRSound` | 1,949 records / 6,656 events / 243 children |
| 2026-08-29 | — | N8 full (both arms) | **3/4 confirmed on both arms**; all 3 frequency concepts confirmed in the confound-free age gradient |
| 2026-08-29 | — | N1 `--stage embed` / `lora` / `probe` | AST frozen **13/14** concepts (mean R² 0.410); **LoRA degrades 13/14**, mean ΔR² **−0.091**, worst `rhonchi_presence` −0.219 |
| 2026-08-29 | — | fixed `inject_lora` (transformers 5 renamed q/v) | first LoRA run injected **0 adapters** and silently trained a linear probe; now raises instead |
| 2026-08-29 | — | fixed N8 verdict criterion | 4-prediction binomial floors at p=0.0625 and hardcoded "NOT supported"; now judged per-concept |
| 2026-08-29 | — | **N8 predictions extended 4 → 8** (2 mechanisms, 2 registration rounds) | sign-test floor 0.0625 → 0.0039. With real power the mechanism is **MIXED, not supported**: 5/7 p=0.227 (age), 5/8 p=0.363 (cross-corpus) |
| 2026-08-29 | — | paper: added §probe (N1), §bottleneck (N6), §pediatric (N8) | abstract, limitations and conclusions updated; 4 new bib entries; LaTeX structure verified balanced |
| 2026-08-29 | — | **clinician sheet returned** (`Asif's/labels v1 - labels.csv`) | 116/132 answered; `ambiguous` ×12 handled as unsure; 16 blanks = 16 `unusable` |
| 2026-08-29 | — | N9 on real labels | vs ICBHI κ **0.035** / **0.266**; intra-rater κ **0.714** / **0.600**; Se 0.231 vs Tzeng's 0.2323 |
| 2026-08-29 | — | N9 release asset | `release_annotations/` — first fine-grained ICBHI concept labels + README (G0 ceiling lever) |
| 2026-08-29 | — | N7 re-run with real corrections | 30 test-patient corrections; accuracy 0.5349 → 0.3953. No simulated component remains |
| 2026-08-29 | — | **N10 — G2 vs an independent reference** | paired diff +0.033 / +0.097, neither excludes zero → **UNDECIDED**; pivot's ambiguity stands, now measured |
| 2026-08-29 | — | N10 selection-bias + degeneracy guards | pack is longest-first stratified → absolute AUROC not comparable to G2; `fine_crackle_ratio` constant → not estimable |
| 2026-08-29 | — | synthetic-data audit of all scripts | one substitution found: N7's clinician oracle. All other randomness is controls, bootstraps or nulls on real data |
| 2026-08-29 | — | built `N9_clinician_reliability.py`, `--selftest` | 90%-agreeing synthetic rater → κ 0.758; random rater → κ −0.160. Statistics validated |
| 2026-08-29 | — | N9 against the real sheet | **BLOCKED: 0/132 clips answered** — `labels.csv`/`.xlsx` are the blank template. No result invented |

| 2026-08-30 | — | repo consolidation | 8 superseded run folders archived; concept engine un-nested; M28 now skips `Archive_Files` |
| 2026-08-30 | — | **N11 built + `--selftest` + full run** | planted-signal 1.000 / noise 0.511 PASS. **AST frozen 0.7115 crackle / 0.7621 wheeze**, M2 0.7451 / 0.8673, 14-concept vector 0.6608 / 0.5815. Pre-registered **NO CEILING** branch fired |
| 2026-08-30 | — | N9 / N10 re-run after the ceiling retraction | conflated G2 pairs separated (9-concept M39 gate vs 14-concept engine); N9's ceiling reading rewritten |

> Append a row every time you run something. A number without a row here is a number nobody can
> reproduce.
