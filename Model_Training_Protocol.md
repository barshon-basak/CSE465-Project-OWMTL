# Model Training Protocol — OWMTL Project

> **What this is:** A short, practical guide so all training results are compatible and mergeable, whoever produces them. Follow the essentials below; everything else is up to you.
>
> **What this is NOT:** A rigid step-by-step script, and not a role assignment. You have freedom in how you structure your code, which libraries you use, and how you organize your workflow — as long as the outputs match. This document was restructured on 2026-08-05 to drop the old per-member (A/B/C/D) framing; the historical role-based version is in `Archive_Work_Plan/Model_Training_Protocol_ARCHIVED.md` if you need it.
>
> **Novelty comes first now — but only 2–3 items, deliberately.** Dr. Khan's 2026 guidance is explicit: the project is not graded on how many models get trained, and a basic classification pipeline isn't enough on its own. Before starting any new model run, check `Novelty Search.md` — it's the highest-priority document in this repo. A run that doesn't trace back to either (a) the core mechanism actually working on real data, or (b) one of the **selected** novelty items in `Novelty Search.md` §4.0, is probably not worth the compute.
>
> **Do not make this project buzzword-heavy.** The supervisor's novelty list is a menu, not a checklist. Implementing all of it would mean nine simultaneous changes on one dataset with 19 unknown patients — unablatable, undefendable, and read by reviewers as a technique list rather than a contribution. The selected set is fixed at 2–3 items (§4.0 of `Novelty Search.md`); if something new looks compelling, it **replaces** a selected item rather than joining it.

---

> ## 🧭 Direction update (2026-08-14) — read before starting any new run
>
> The project's **headline direction has pivoted** and this protocol has been aligned to it. The original core mechanism (**cross-task disagreement** as an unseen-disease detector) is **retired as the thesis** — it lost to a trivial Energy baseline (M15 AUROC 0.5747 < 0.6466) and is a published method (Zamir et al. 2020). Two later directions were also rejected on cross-check (Direction 1 device/disease disentanglement; ACBD-as-originally-specified). See `OWMTL_Project_Evolution.md` for the full trail.
>
> **The current direction is a shared engine with one data-driven fork:** a **physics-grounded, label-free acoustic concept bottleneck** (clinically-named concepts — fine/coarse crackle, wheeze pitch band, inspiratory phase, rhonchi, spectral flatness, PAPR — computed by DSP, *not* human-labeled), used either as (**Path A**) an interpretable, clinician-correctable diagnosis, or (**Path B**) a faithfulness/robustness audit ("do the models actually listen to the clinical sounds?"). Which headline is chosen is decided by *data* at a mid-project gate (**G3**), not up front.
>
> **Current source-of-truth documents** (read alongside `Novelty Search.md`):
> - `OWMTL_Merged_Decision_Roadmap.md` — the gated roadmap (G0–G7).
> - `OWMTL_Build_Sheet.md` — the week-by-week build plan and which existing model feeds each step.
> - `OWMTL_Novelty_Gap_Analysis.md` — the gap analysis + 2026 cross-check.
>
> **What this changes below:** three reporting rules are now **hard** (official split + official metric only, commit the confusion matrix for *every* model, CIs + a paired test on *every* headline comparison — §1). New model types (concept bottleneck, leakage, intervention, concept-space OOD, foundation-model probing) get required metrics (§3.5), ablation groups, and component flags (§4.1). The cross-task mechanism is **not deleted** — it survives as **one scored baseline detector**, not the contribution.

---

## 1. The Essentials (Must-Do)

These are the only hard requirements. Everything else in this doc is guidance.

1. **Patient-independent splits.** No patient's cycles in both train and test. This is a research validity requirement, not a style choice.

   > **⚠️ The official ICBHI split does not satisfy this.** `ICBHI_challenge_train_test.txt`
   > (committed at `Asif's/ICBHI_challenge_train_test.txt`) assigns **recordings**, not patients —
   > and patients **156** and **218** have recordings on *both* sides. So "use the official split"
   > and "be patient-independent" cannot both be satisfied verbatim.
   >
   > **Project policy:** use the official split with every recording of a leaking patient
   > reassigned to **train** (conservative — the test set then contains no patient seen during
   > training). Cost: 12 of 381 test recordings (3.1%). Result: **551 train / 369 test = 59.9/40.1**,
   > which is closer to a nominal 60/40 than the published split itself.
   >
   > Call it `official_60_40_patient_independent_corrected`, **never** plain "official 60/40" —
   > it is the official split *corrected*, and conflating the two is what this rule exists to stop.
   > `Asif's/audit/official_split.py` implements and audits both modes; run it standalone to see
   > the numbers.
   >
   > **Historical note:** M2/M3/M12/M22 were all labelled `patient_independent_official_60_40` but
   > actually ran a fallback rule (`patient_id <= 111 -> test`) giving 11 test patients and 7.1% of
   > cycles, because the split file was absent from the runtime and the fallback was silent. Their
   > labels are now corrected in-place. **Any notebook that cannot find the split file must raise,
   > not fall back.**
2. **Real data only.** Every `results_M*.json` must come from a run that actually loaded ICBHI audio (or Coswara/SPRSound for OOD runs) — never from a `torch.randn`/`np.random` placeholder Dataset. The audit tool (`Asif's/audit/audit_project.py`) checks for this automatically; a run it flags `synthetic_data_not_real_dataset` is not a result and cannot be reported.
3. **Save checkpoints every epoch** so Kaggle/Colab disconnects don't lose progress (see §11 for PyTorch 2.6+ checkpoint rules and Kaggle persistence protocol).
4. **Produce a structured results JSON** per model (schema in §4) — this is what makes the final merge work.
5. **Compute all metrics in §3** — accuracy, precision, recall, F1, confusion matrix, model size, params, training time.
6. **Generate the required plots** (§5) — loss curves, accuracy curves, confusion matrix.
7. **Use the shared preprocessing parameters** for audio (sample rate, mel bins, etc.) so results stay comparable across models.
8. **Official split + official metric are the only reportable comparison.** Every number that enters a comparison, table, or claim uses the **patient-independent official 60/40 split** and the **official ICBHI metric** (`icbhi_score_official`, §3). **Never compare across splits** — a 70/30 number and a 60/40 number are not on the same scale (a random 70/30 split is easier than the deliberately-hard official one), and neither is comparable to published work. Legacy 70/30 runs (M30–M37) must be re-run on the official split before they appear in any comparison. *(This is the rule that the M30 withdrawal and the metric-correction audit made non-negotiable.)*
9. **Commit `confusion_matrix_raw` for every model, every split** — not just sound-event models. Without it, no score can be independently verified by a teammate, the audit tool, or a reviewer. A results JSON with a reported score and no committed matrix is not a result. *(This is exactly why M30's 0.8213 is unverifiable and withdrawn.)*
10. **Every headline comparison carries a confidence interval + a paired test.** Report bootstrap 95% CIs (B=1000) on the primary metric / AUROC, and a paired significance test (McNemar on paired predictions for classification; Wilcoxon signed-rank across patients/folds; Hanley-McNeil or bootstrap for AUROC differences). With only 19 unknown patients, a bare point AUROC is meaningless — the CI (≈ ±0.12 at n=19) must be shown. Reuse the M30 McNemar/bootstrap code and `Asif's/Statistics/`.

That's it. The rest of this document explains *how* to do these things.

---

## 2. Shared Preprocessing Parameters

These audio parameters should stay consistent across models so spectrograms are comparable. If you have a strong reason to deviate for a specific model, document it in your results JSON's `ablation.known_deviations`.

```
Sample rate  : 16000 Hz
Duration     : 8.0 seconds
Mel bins     : 128
FFT size     : 1024
Hop length   : 160  (10 ms)
Window length: 400  (25 ms)
Freq range   : 50–2000 Hz
```

**Random seed:** Use `42` as the default for base experiments. If you need different seeds (e.g., ensemble members), just document what you used.

---

## 3. Required Metrics

Every model run must report these. Compute them on the **test/validation set** using the best checkpoint.

### Classification Metrics

| Metric | Scope |
|---|---|
| Accuracy | Overall |
| Precision | Macro + per-class |
| Recall (Sensitivity) | Macro + per-class |
| F1 Score | Macro + per-class |
| Confusion Matrix | Raw counts + normalized (row-wise) |

For sound-event models, also compute:
- **Specificity** (macro + per-class)
- **`icbhi_score`** = (Macro Sensitivity + Macro Specificity) / 2 — *project-internal metric, see the warning below*
- **`icbhi_score_official`** = (Se + Sp) / 2 — **the ICBHI 2017 challenge metric. This is the one that goes in the paper.**
  - `Se` = correctly classified **abnormal** events (Crackle + Wheeze + Both) / all abnormal events
  - `Sp` = correctly classified **Normal** events / all Normal events

> ### ⚠️ These two numbers are not interchangeable
>
> `icbhi_score` (macro form) is **not** the ICBHI 2017 challenge score and is **not comparable to
> published ICBHI results.** `specificity_macro` averages per-class specificity, and each class's
> specificity counts true negatives contributed by the other three classes — so rare classes
> (Wheeze n≈38, Both n≈35) score ~0.95 specificity almost regardless of whether the model detects
> them at all. That pulls the macro average up and inflates the score.
>
> Measured across this repo: **mean inflation +0.11, worst case +0.22**
> (`Asif's/audit/ICBHI_SCORE_AUDIT.md`). Published ICBHI SOTA on the official 60/40 split is
> roughly **0.60–0.65** — read our numbers against that using the *official* column only.
>
> The macro form also **hides model pathologies the official metric exposes.** M36 reports
> `icbhi_score` 0.5137 while detecting only 9% of abnormal events (Se = 0.0932); M33 reports 0.5506
> while never classifying a single Normal cycle correctly (Sp = 0.0000). In both cases the macro
> metric is propped up by the specificity term. Always look at Se and Sp separately before
> believing a score.
>
> **Report both, lead with the official one.** Recompute historical runs with
> `python3 "Asif's/audit/icbhi_score_audit.py"` (add `--write` to backfill the fields into
> existing results JSONs).

**Also required for sound-event models:** commit `confusion_matrix_raw`. Without it neither score
can be independently verified — by a teammate, by the audit tool, or by a reviewer who asks. Two
models currently report an ICBHI score with no committed matrix, including M30, the project's
highest headline number.

For open-set / unknown-detection models, also compute:
- Unknown-detection precision, recall, AUROC, AUPR
- **AUROC/AUPR with bootstrap 95% CIs** (mandatory at n=19 — see Essential #10)

### Concept-bottleneck / faithfulness / open-world metrics (new direction — §3.5)

For any run that is part of the concept-bottleneck engine (see the Direction-update block), also compute and record the metrics relevant to its role. Put these under a `concept_metrics` object in the results JSON (schema note in §4).

| Metric | For which run | Definition / how |
|---|---|---|
| **Concept accuracy** (per concept + macro) | concept extractors, CBM | how well each DSP-derived concept matches the ICBHI cycle label it maps to (crackle/wheeze presence), plus per-concept reliability |
| **Accuracy–interpretability tradeoff** | CBM variants vs opaque | disease metric of the strict bottleneck vs the opaque baseline, on the **same official split** — report the *gap*, not just the bottleneck number |
| **Per-class disease F1** | CBM, disease head | watch explicitly for collapse toward COPD (the §2.5 validity-hole failure mode) |
| **Concept leakage** | CBM variants | information-theoretic estimate — mutual information between residual encoder info and the label *given* the concepts (ref arXiv:2504.09459). Report per variant {independent, sequential, leaky-joint} |
| **Intervention Δaccuracy** | CBM + intervention | change in diagnosis when a concept value is overwritten (simulated clinician correction); report per corrected concept |
| **Concept-space vs embedding-space OOD** | concept-space detectors | AUROC/AUPR of MSP/Energy/Mahalanobis run **in concept space** vs the same detectors in embedding space, **with CIs**, stratified by shift type |
| **Covariate-shift response** | device / pediatric | for device LODO (if feasible, G4) and SPRSound: how much a detector fires on *known* diseases under new device/population (a good detector should NOT) — the physics-fragility check (I5) |
| **FM concept-faithfulness** | FM-probing (G5) | probing accuracy of the clinical concepts from an OPERA/M2D embedding, and whether it stays faithful under shift |

> **These are the numbers Gate G3 reads** to decide Path A vs Path B (`OWMTL_Merged_Decision_Roadmap.md`): the accuracy–interpretability tradeoff, the leakage, and the intervention effect. Produce all three before the G3 decision.

### Efficiency Metrics

| Metric | How |
|---|---|
| Total parameters | `sum(p.numel() for p in model.parameters())` |
| Trainable parameters | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Model size (MB) | Save `state_dict()` to temp file, check file size |
| Training time per epoch | `time.time()` around each epoch |
| Total training time | Sum of epoch times |
| GPU type | `torch.cuda.get_device_name(0)` |

**Optional but recommended:** inference time (ms/sample) — useful for the paper's efficiency discussion.

### Quick reference for model size:

```python
import tempfile, os, torch

def get_model_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        return round(os.path.getsize(tmp.name) / (1024 * 1024), 2)
```

---

## 4. Results JSON Format

Each completed model produces **one JSON file**: `results_<MODEL_ID>.json` (e.g., `results_M1.json`, `results_M23_aug.json`).

This is the format the final merge (§9 of `Model_Training_Reference.md`) expects. Stick to this structure so we don't have to reformat later.

```json
{
  "meta": {
    "model_id": "M1",
    "model_name": "Provisional CNN Backbone",
    "contributor": "Barshon",
    "date_completed": "2026-07-23",
    "is_augmented": false,
    "augmentation_method": "none",
    "notes": ""
  },

  "config": {
    "sample_rate": 16000,
    "n_mels": 128,
    "batch_size": 32,
    "num_epochs": 60,
    "lr": 0.001,
    "optimizer": "Adam",
    "scheduler": "StepLR",
    "architecture": "2D_CNN_4Block",
    "seed": 42
  },

  "environment": {
    "platform": "Kaggle",
    "gpu_name": "Tesla T4",
    "pytorch_version": "2.10.0",
    "python_version": "3.12.13"
  },

  "dataset_info": {
    "dataset": "ICBHI_2017",
    "train_samples": 6406,
    "test_samples": 492,
    "split_method": "patient_independent_60_40"
  },

  "efficiency": {
    "total_params": 421732,
    "trainable_params": 421732,
    "model_size_mb": 1.61,
    "training_time_total_s": 15660,
    "training_time_per_epoch_s_avg": 261,
    "gpu_name": "Tesla T4",
    "inference_time_ms_per_sample": null
  },

  "best_epoch": {
    "epoch": 47,
    "primary_metric": "icbhi_score",
    "primary_metric_value": 0.7144
  },

  "best_metrics": {
    "accuracy": 0.5200,
    "precision_macro": 0.4300,
    "recall_macro": 0.4800,
    "f1_macro": 0.4100,
    "specificity_macro": 0.8500,
    "icbhi_score": 0.6650,
    "per_class": {
      "Normal":  {"precision": 0.52, "recall": 0.90, "f1": 0.66, "support": 255},
      "Crackle": {"precision": 0.38, "recall": 0.23, "f1": 0.28, "support": 164},
      "Wheeze":  {"precision": 0.15, "recall": 0.21, "f1": 0.18, "support": 38},
      "Both":    {"precision": 0.29, "recall": 0.49, "f1": 0.36, "support": 35}
    },
    "confusion_matrix_raw": [[230, 15, 5, 5], [100, 38, 10, 16], [20, 5, 8, 5], [10, 5, 3, 17]],
    "confusion_matrix_normalized": []
  },

  "ablation": {
    "ablation_group": "backbone_architecture",
    "ablation_role": "variant",
    "baseline_model_id": null,
    "variable_changed": "backbone: 2D_CNN_4Block",
    "variables_held_constant": [
      "loss_function",
      "optimizer",
      "data_split",
      "augmentation: none",
      "seed: 42"
    ],
    "component_flags": {
      "has_sound_event_head": true,
      "has_disease_head": false,
      "has_cross_task_consistency": false,
      "has_cqkd_regularization": false,
      "has_openmax_rejection": false,
      "owl_stage": 0,
      "compression_clusters": null
    },
    "loss_weights": {
      "sound_event_weight": 1.0,
      "disease_weight": null,
      "consistency_weight": null
    }
  },

  "training_history": [
    {
      "epoch": 1,
      "train_loss": 1.23,
      "val_loss": 1.45,
      "train_accuracy": 0.45,
      "val_accuracy": 0.42,
      "train_f1_macro": 0.32,
      "val_f1_macro": 0.30,
      "lr": 0.001,
      "epoch_time_s": 261
    }
  ]
}
```

### What matters in this schema:
- **`meta`** — identifies who trained what and keeps `model_id` consistent with `Model_Training_Reference.md`'s model index. `contributor` is a free-text name — there is no fixed role or letter attached to it; anyone can pick up any chunk.
- **`best_metrics`** — the numbers that go in the paper. Must include all metrics from §3.
- **`training_history`** — one entry per epoch with at least loss, accuracy, and F1 for both train and val. This is what the plots are generated from.
- **`efficiency`** — params, model size, training time. Reviewers ask for these. `inference_time_ms_per_sample` is included (set to `null` if not measured, but **strongly recommended** — it feeds directly into the efficiency columns of the ablation table).
- **`ablation`** — **required for every model.** This block self-documents each run's role in the ablation study so the final table can be assembled programmatically instead of manually. See §4.1 for the full explanation.

You can add extra fields if your model needs them (e.g., `auroc` for open-set models, `compression_ratio` for compression sweeps). Just don't remove or rename the fields above.

**New-direction additions (concept-bottleneck engine):** add a **`concept_metrics`** object holding the §3.5 numbers relevant to the run (e.g. `concept_accuracy_per`, `tradeoff_vs_opaque`, `leakage_bits`, `intervention_delta`, `concept_space_auroc`, `embedding_space_auroc`, `covariate_fpr`), and attach a **`ci`** field to every headline metric (e.g. `"icbhi_score_official": 0.63, "icbhi_score_official_ci95": [0.60, 0.66]`) plus the paired-test result in `meta.notes` or a `stats` object (Essential #10). Set the new `component_flags` (§4.1) on every run so the merge can tell concept-bottleneck runs apart from legacy ones.

---

## 4.1 Ablation Metadata — How to Fill the `ablation` Block

The paper needs an ablation study table. An ablation table answers: *"What happens to performance when we add/remove/swap one component, holding everything else constant?"* The `ablation` block in the results JSON exists to make this table trivially assembable at merge time — **if you fill it in during training**, you won't have to reconstruct it from memory later.

### Fields explained:

| Field | What to put | Example |
|---|---|---|
| `ablation_group` | Which ablation table row-group does this model belong to? Use one of the canonical group names below. | `"backbone_architecture"` |
| `ablation_role` | Is this the full/baseline model (`"baseline"`) or a variant with something removed/changed (`"variant"`)? | `"variant"` |
| `baseline_model_id` | The model ID of the baseline this run is compared against. `null` if this *is* the baseline. | `"M4"` |
| `variable_changed` | Plain-English description of the one thing that differs from the baseline. | `"backbone: 2D_CNN_4Block"` |
| `variables_held_constant` | List of things deliberately kept the same so the comparison is fair. | `["loss_function", "optimizer", "seed: 42"]` |
| `component_flags` | Binary flags for which architectural components are active in this run. | See schema above |
| `loss_weights` | The actual loss-weighting values used. `null` if the head doesn't exist in this run. | `{"sound_event_weight": 1.0, ...}` |

### Canonical `ablation_group` names (use these exact strings):

| Group name | What it compares | Models involved |
|---|---|---|
| `backbone_architecture` | Backbone choice (CNN vs. MobileNet vs. transformer) | M2, M3, M4, M12 |
| `loss_weighting` | Loss weighting between heads (manual or auto-balanced — see Novelty Search §4.6) | M13, M15 |
| `cross_task_consistency` | With/without cross-task consistency | M13 (without) vs. M15 (with) |
| `rejection_method` | Cross-task consistency vs. OpenMax/Weibull vs. trivial post-hoc scores | M6 vs. M15 vs. M29 |
| `cqkd_regularization` | CQKD-regularized vs. unregularized | M17 variants |
| `owl_stage_count` | OWL stage count (1 vs. 2 vs. 3) | M15 (Stage 1) vs. M17 (Stage 2) |
| `compression_level` | Compression sweep | M18 variants |
| `augmentation_effect` | Augmented vs. clean | M2→M21, M3→M22, M4→M23, M15→M24, M18→M25, M17→M26 |
| `ood_generalization` | Full model vs. ablated on Coswara/SPRSound | M19, variants |
| `uncertainty_method` | Ensemble vs. MC-Dropout vs. SNGP vs. Evidential | M7, M8, M9, M10 |
| `calibration_method` | Temperature vs. vector vs. focal | M11 variants |

**Concept-bottleneck engine groups (new direction — use these for all Path A/B runs):**

| Group name | What it compares | Notes |
|---|---|---|
| `bottleneck_type` | independent-CBM vs. sequential-CBM vs. leaky-joint control vs. opaque baseline | the core accuracy–interpretability tradeoff (G3 input) |
| `concept_source` | physics/DSP-derived vs. learned/CLAP-derived vs. hybrid | the "physics-grounded" novelty claim; physics is the headline arm |
| `concept_set_size` | minimal {PAPR, flatness, wheeze-band} vs. extended {+ fine/coarse crackle, phase, rhonchi} | how bottleneck width trades against accuracy |
| `leakage_regularization` | with vs. without the leakage penalty | pairs with the leakage metric (§3.5) |
| `concept_space_vs_embedding_ood` | novelty detection in concept space vs. embedding space | detector family held constant (MSP/Energy/Mahalanobis) |
| `covariate_shift` | random split vs. leave-one-device-out (G4) vs. pediatric (SPRSound) | the robustness/physics-fragility axis |
| `fm_probing` | scratch-CNN encoder vs. OPERA/M2D embedding (frozen or LoRA) | optional, G5 — answers the deferred "Attack 7" |

> **Reuse note:** the old `rejection_method` / `cross_task_consistency` groups still exist, but the cross-task scorer (M15) is now **one scored detector inside `concept_space_vs_embedding_ood`**, not a standalone contribution.

Other novelty-driven work may still need group names (e.g. `training_strategy` for curriculum, `disease_head_architecture` for the prototypical head, `ensemble_fusion` for feature fusion). See `Novelty Search.md` §6 for proposed-but-not-yet-canonical names — add to these tables as a single batch edit when one is adopted, rather than one at a time.

### How `component_flags` works:

This is the key to building the ablation table automatically. Each flag answers a yes/no question about what's active in this particular run:

- `has_sound_event_head` — Is the 4-class sound-event head present?
- `has_disease_head` — Is the disease-diagnosis head present?
- `has_cross_task_consistency` — Is the cross-task disagreement mechanism active?
- `has_cqkd_regularization` — Is cluster-quantized distillation regularization applied?
- `has_openmax_rejection` — Is the OpenMax/Weibull rejection mechanism active (alternative to cross-task consistency)?
- `owl_stage` — Which OWL stage is this run evaluated at? (0, 1, or 2)
- `compression_clusters` — Number of clusters in CQKD compression. `null` if uncompressed.

**New-direction flags (add these for any concept-bottleneck engine run; set to `false`/`null` on legacy runs):**

- `has_concept_bottleneck` — Is the diagnosis routed *only* through the concept layer (a true bottleneck)?
- `bottleneck_type` — `"independent"` | `"sequential"` | `"leaky_joint"` | `"opaque"` | `null`.
- `concept_source` — `"physics"` | `"learned"` | `"hybrid"` | `null` (physics = DSP-derived clinically-named concepts).
- `has_leakage_measurement` — Was the information-theoretic leakage metric computed for this run?
- `has_concept_intervention` — Is the concept-intervention API exercised (overwrite-and-re-predict)?
- `concept_space_ood` — Is the OOD/novelty score computed in concept space (`true`) vs embedding space (`false`)?
- `fm_backbone` — `"none"` | `"OPERA"` | `"M2D"` | ... — the foundation-model encoder if used (G5).

These flags let the merge step auto-generate rows like:

| Variant | Sound Head | Disease Head | Cross-Task | CQKD | F1 | AUROC | Params | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| Full model | ✓ | ✓ | ✓ | ✓ | 0.xx | 0.xx | xxx | x.x |
| − Cross-task | ✓ | ✓ | ✗ | ✓ | 0.xx | 0.xx | xxx | x.x |
| − Disease head | ✓ | ✗ | ✗ | ✓ | 0.xx | — | xxx | x.x |

### `loss_weights` — why this matters for ablation:

The paper needs a "loss weighting between heads" ablation. If you're training a multi-head model (M13, M15, M17), record the actual weight values you used:
- `sound_event_weight` — weight on the sound-event classification loss
- `disease_weight` — weight on the disease-diagnosis loss
- `consistency_weight` — weight on the cross-task consistency loss term (M15+)

If you run the same model with different weight ratios (e.g., 1:1:0.5 vs. 1:1:1.0), each run gets its own results JSON with the weights recorded. The ablation table then shows the effect of rebalancing. (If you implement automatic loss-balancing — GradNorm or uncertainty-weighting, see `Novelty Search.md` §4.6 — record the *resulting* effective weights here, plus a note in `meta.notes` that they were learned, not hand-set.)

### Quick example — filling this for a backbone candidate (clean, no augmentation):

```json
"ablation": {
  "ablation_group": "backbone_architecture",
  "ablation_role": "variant",
  "baseline_model_id": "M12",
  "variable_changed": "backbone: AST_pretrained",
  "variables_held_constant": [
    "loss_function: CrossEntropyLoss",
    "optimizer: Adam",
    "data_split: patient_independent_60_40",
    "augmentation: none",
    "seed: 42",
    "preprocessing: 128mel_16kHz_8s"
  ],
  "component_flags": {
    "has_sound_event_head": true,
    "has_disease_head": false,
    "has_cross_task_consistency": false,
    "has_cqkd_regularization": false,
    "has_openmax_rejection": false,
    "owl_stage": 0,
    "compression_clusters": null
  },
  "loss_weights": {
    "sound_event_weight": 1.0,
    "disease_weight": null,
    "consistency_weight": null
  }
}
```

### Quick example — filling this for a cross-task consistency run (OWL Stage 1):

```json
"ablation": {
  "ablation_group": "cross_task_consistency",
  "ablation_role": "baseline",
  "baseline_model_id": null,
  "variable_changed": "cross_task_consistency: enabled",
  "variables_held_constant": [
    "backbone: M12_final",
    "disease_head: M13_architecture",
    "data_split: patient_independent_LOPO",
    "augmentation: none",
    "seed: 42"
  ],
  "component_flags": {
    "has_sound_event_head": true,
    "has_disease_head": true,
    "has_cross_task_consistency": true,
    "has_cqkd_regularization": false,
    "has_openmax_rejection": false,
    "owl_stage": 1,
    "compression_clusters": null
  },
  "loss_weights": {
    "sound_event_weight": 1.0,
    "disease_weight": 1.0,
    "consistency_weight": 0.5
  }
}
```

> **Tip:** If a model participates in multiple ablation groups (e.g., a cross-task consistency model is also the baseline for `rejection_method`), pick the *primary* group. The merge step can cross-reference by `model_id`. If you want to be thorough, add an optional `"secondary_ablation_groups": ["rejection_method"]` list.

---

## 5. Required Plots

Generate these for every model and save them alongside your results JSON.

| Plot | What it shows |
|---|---|
| **Loss curves** | Train loss + val loss vs. epoch (same axes) |
| **Accuracy curves** | Train accuracy + val accuracy vs. epoch |
| **F1 curves** | Train macro-F1 + val macro-F1 vs. epoch |
| **Confusion matrix** | Heatmap — at least the normalized version; raw counts version is a plus |

### Styling suggestions (not mandatory, but helps the paper look unified):
- 150 DPI or higher
- Mark the best epoch on the curves (dotted vertical line or similar)
- Use a consistent color for train (e.g., blue) and validation (e.g., orange/red)
- Include axis labels and a legend

---

## 6. Checkpoints & Disconnect Safety

Kaggle gives you ~9–12 hours; Colab free gives ~12 hours (less with GPU). Disconnects happen. Protect your training.

### The minimum you need:

1. **Save a checkpoint after every epoch**

2. **Auto-resume at the start of the training loop:**

3. **Save `best_model.pth` separately** whenever the primary metric improves — this is the checkpoint you evaluate on.

4. **Clean up old per-epoch checkpoints** (keep the last 2–3) to avoid filling disk.

### Platform-specific tips:

**Kaggle:**
- `/kaggle/working/` is writable but wiped after session ends
- Use **Save Version** (File → Save Version → Save & Run All) to persist outputs
- Download checkpoints from the output panel periodically

**Colab:**
- Mount Google Drive and copy checkpoints there after each epoch:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # After saving checkpoint:
  !cp /content/checkpoints/best_model.pth /content/drive/MyDrive/OWMTL/M1/
  ```

---

## 7. Data Augmentation

Augmentation runs are **separate experiments** from the clean baseline. The general approach:

- **Train with augmentation, validate/test without** — augmentation is training-time only.
- **Use the same seed** as the clean run so the only variable is the augmentation.
- **In your results JSON**, set `is_augmented: true` and note the method in `augmentation_method`.

Augmentation is a **stretch item**, not a required checkbox per chunk — see `Model_Training_Reference.md`'s current-status section. Only spend time on a full augmentation ablation once the chunk it augments is real and verified. If you do run one, common choices per chunk:

| Chunk | Typical augmentation |
|---|---|
| Sound-event backbone | SpecAugment (time + frequency masking on spectrograms) |
| Disease head | Class-balancing (noise injection, pitch shift, time stretch for minority classes) |
| Compression / OOD | Domain-robustness (simulated channel/device noise) |
| Calibration | Re-run calibration on the best augmented backbone |

Implementation details are up to whoever picks up the run.

---

## 8. Hyperparameter Tuning

Not every model needs a full HP search. Use your judgment, but keep this in consideration — and check `Novelty Search.md` §4.2/§4.6 before doing a plain grid/random search, since curriculum ordering and auto-balanced loss weighting are both cheap, novel alternatives that can substitute for (or complement) a manual sweep.

---

## 9. File Organization

Suggested (not mandatory) structure — the only hard requirement is that the results JSON and plots exist somewhere findable:

```
<ContributorName>/
├── M1/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── latest.pth
│   ├── results/
│   │   ├── results_M1.json
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── f1_curve.png
│   │   └── confusion_matrix.png
│   └── notebook_M1.ipynb
├── M2/
│   └── ...
```

`<ContributorName>` is whoever's working directory it is — not a role assignment. Nothing about which model ID goes in which folder is fixed to a person.

---

## 10. Before Calling a Model "Done"

Quick sanity check:

- [ ] Results JSON exists with all §3 metrics filled in
- [ ] Best model checkpoint exists
- [ ] Loss, accuracy, and F1 curve plots generated
- [ ] Confusion matrix plot generated
- [ ] Model size (MB) and parameter count recorded
- [ ] Training time recorded
- [ ] Patient-independent split was used — **and it is the official 60/40 split** for any number that will be compared or reported (Essential #8)
- [ ] **`confusion_matrix_raw` committed** (every model, so the score can be independently verified — Essential #9)
- [ ] **`icbhi_score_official` reported and led with** (macro `icbhi_score` may accompany it but is never the headline)
- [ ] **CI + paired test on every headline comparison** — bootstrap 95% CI + McNemar/Wilcoxon (Essential #10)
- [ ] **For concept-bottleneck engine runs:** `concept_metrics` object filled (§3.5) and the new `component_flags` set (§4.1)
- [ ] **Real data verified** — run `Asif's/audit/audit_project.py` and confirm no `synthetic_data_not_real_dataset` finding on this model
- [ ] For augmented runs: `is_augmented` and `augmentation_method` filled in
- [ ] **Ablation block filled in** — `ablation_group`, `ablation_role`, `baseline_model_id`, `variable_changed`, `component_flags`, and `loss_weights` are all populated (§4.1)
- [ ] **Inference time measured** (recommended) — `inference_time_ms_per_sample` in the `efficiency` block
- [ ] **PyTorch 2.6+ pickling safety verified** — explicit scalar casting (`int()`, `float()`) when saving and `weights_only=False` when loading (§11)
- [ ] **Kaggle persistence & eval_only fallback implemented** (§11)
- [ ] **Handoff download cell added at end of notebook** — clickable `FileLink`s or `.zip` bundle for `best_model.pth` and `results_M<ID>.json` (§11.D)

---

## 11. Checkpoint Safety, Disconnect Persistence & Kaggle Protocol

To prevent disconnects from losing progress, avoid PyTorch 2.6+ unpickling crashes, and allow committing notebooks via **"Save Version"** without re-training models from scratch, all training code should follow this protocol:

### A. PyTorch 2.6+ Checkpoint Serialization Rules
PyTorch 2.6 strictly enforces `weights_only=True` by default when loading `.pth` files. To ensure checkpoints can be cleanly saved and loaded across sessions without unpickling errors:
1. **Cast scalars to native Python types when saving:** When constructing the state dictionary in `save_checkpoint`, explicitly cast NumPy floats or integer metrics to native Python types (`int()` and `float()`). Never save raw `np.float64`, `np.float32`, or sklearn metric objects directly into the checkpoint dictionary.
   ```python
   # Correct:
   state = {
       "epoch": int(epoch),
       "best_score": float(best_score),
       "model_state": model.state_dict(),
       ...
   }
   ```
2. **Use `weights_only=False` when loading trusted checkpoints:** Since checkpoint files are self-generated and trusted, use `weights_only=False` in `torch.load` during auto-resume and evaluation to prevent PyTorch 2.6+ from rejecting custom metadata:
   ```python
   state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
   ```

### B. Kaggle Persistence Protocol (How to Never Lose Progress)
Interactive browser sessions on Kaggle time out after ~20–30 minutes of inactivity or network drops, which wipes the temporary `/kaggle/working/` directory.
1. **For Long Training Runs, Avoid Interactive Mode:**
   - Instead of running a multi-hour training loop interactively in your browser, click **Save Version** (top right) -> **Save & Run All (Commit)** -> **Save**.
   - This executes your notebook in an unattended background container (up to 12 hours) that never disconnects due to Wi-Fi drops, browser closing, or computer sleeping.
   - When finished, all checkpoints (`best_model.pth`, `epoch_*.pth`) and results JSONs are saved permanently in that version's **Output Files**.
2. **Auto-Resume Fallback (`/kaggle/input/` Discovery):**
   - Configure your checkpoint loader (`load_checkpoint`) so that if `/kaggle/working/checkpoints/` is empty, it automatically searches `/kaggle/input/` for existing checkpoint files from attached datasets or previous versions.
   - If found, copy them into `/kaggle/working/checkpoints/` and resume from the saved epoch.

### C. "Save Version" Without Re-Training Protocol
When you click **"Save & Run All (Commit)"**, Kaggle boots a clean, empty machine. To commit a final notebook version or generate reports without waiting hours for the model to re-train from scratch:
1. **Method 1 (Attach Previous Version Output):**
   - Open the right-hand sidebar in Kaggle -> **Add Data** -> **Your Work** -> **Output Files** -> click **Add** next to your previously trained notebook version.
   - When you click **Save & Run All**, your checkpoint loader will automatically discover the completed checkpoint (e.g., Epoch 60) in `/kaggle/input/`, see that `start_epoch = 61`, bypass the training loop in 1 second, and directly execute the evaluation cells.
2. **Method 2 (`eval_only` Mode):**
   - Include an `"eval_only": False` switch in your global configuration (`CFG`).
   - In your training loop, check `if CFG.get("eval_only", False):`. If `True`, skip the training loop (`start_epoch = CFG["num_epochs"] + 1`) and jump straight to final evaluation on whatever checkpoint is loaded.

### D. Cross-Contributor Handoff via Manual File Download / Upload
> [!NOTE]
> **Use this manual download/upload method whenever needed only.** If Kaggle's internal dataset linking works smoothly across accounts, use direct linking instead. Use manual download/upload whenever direct sharing is inconvenient or across different platforms.

When work happens across separate notebooks/accounts, sharing outputs via Kaggle's internal dataset linking can sometimes be inconvenient. Hand off models using **manual downloads**:
1. **Add a Handoff Download Cell at the End of Every Notebook:**
   At the very end of your notebook, include a dedicated code cell using `IPython.display.FileLink` to generate clickable download links for the exact output files a downstream model needs:
   - `best_model.pth`: Required by any downstream model that builds on this checkpoint.
   - `results_M<ID>.json`: Required for the final table reconciliation.
   - Training curves / plots (`.png`).

   **Copy-paste this exact code block as your final notebook cell:**
   ```python
   # ============================================================
   # FINAL CELL — TEAM HANDOFF & ONE-CLICK FILE DOWNLOADS
   # ============================================================
   import os
   import shutil
   import glob
   from IPython.display import display, FileLink

   print("=" * 60)
   print("OFFICIAL PROTOCOL OUTPUTS READY FOR DOWNLOAD")
   print("=" * 60)

   # 1. Grab exactly the files specified in §9 of Model_Training_Protocol.md
   protocol_files = sorted(
       glob.glob(os.path.join(CFG["ckpt_dir"], "best_model.pth")) +
       glob.glob(os.path.join(CFG["results_dir"], "results_M*.json")) +
       glob.glob(os.path.join(CFG["results_dir"], "*.png"))
   )

   for fpath in protocol_files:
       if os.path.exists(fpath):
           size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
           print(f"Ready: {os.path.basename(fpath):<25} ({size_mb} MB)")
           display(FileLink(fpath))
       else:
           print(f"Missing: {os.path.basename(fpath)}")

   # 2. Create a clean ZIP archive of only the official protocol files
   bundle_dir = "/kaggle/working/protocol_bundle"
   if protocol_files:
       os.makedirs(bundle_dir, exist_ok=True)
       for fpath in protocol_files:
           if os.path.exists(fpath):
               shutil.copy2(fpath, os.path.join(bundle_dir, os.path.basename(fpath)))
       
       zip_path = shutil.make_archive("/kaggle/working/model_handoff_bundle", 'zip', bundle_dir)
       size_zip = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
       print(f"\nOr download all official files in a single ZIP bundle ({size_zip} MB):")
       display(FileLink("model_handoff_bundle.zip"))
   print("=" * 60)
   ```
2. **How to Download:**
   - When the notebook run finishes, run the final download cell. Click the generated links in your browser to download the individual files (or the combined `.zip` bundle) directly to your local computer.
3. **How to Import in the Next Notebook:**
   - **On Kaggle:** In the destination notebook, click **Add Data** -> **New Dataset** -> upload the downloaded `best_model.pth` or results JSON (naming it something like `owmtl-m1-output`). The files will instantly be available at `/kaggle/input/owmtl-m1-output/`, ready for `torch.load()` or `json.load()`!
   - **On Colab / Local:** Place the downloaded files into your local checkpoint/results directory or mount Google Drive.

---

## 12. AI Model Notebook Generation Prompt (Instructions for LLMs)

> **SYSTEM INSTRUCTION FOR AI ASSISTANTS (e.g., Gemini, ChatGPT, Claude, GitHub Copilot):**
> When a user attaches or references this document (`Model_Training_Protocol.md`) and asks you to generate a Jupyter Notebook (`.ipynb`) for a specific model run (e.g., "Generate the notebook for M4" or "Create notebook for M15"), you must act as an expert AI medical imaging and audio signal processing engineer working on our research project: **"Cluster-Aware Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis" (OWMTL)**, targeting Q1 journal publication (*Biomedical Signal Processing and Control*).
>
> Do not ask the user for a separate template or to fill in placeholders. Instead, autonomously retrieve the model specifications from the workspace (such as `Model_Training_Reference.md`, `Novelty Search.md`, or the user's prompt) and generate a complete, professional, clean, and fully executable Jupyter Notebook (`.ipynb`) in Python/PyTorch that strictly conforms to all rules in §§1–11 of this protocol and the instructions below.
>
> **Before generating anything, check `Novelty Search.md`.** As of 2026, the supervisor does not expect a plain classification/model-count pipeline — every new model should trace to either the core mechanism finally running on real data, or one of the **selected** novelty items in §4.0 of that document. If the user's request doesn't obviously map to either, ask rather than assume.
>
> **Do not add novelty techniques the user didn't ask for.** The project deliberately implements 2–3 items, not the supervisor's full list — see `Novelty Search.md` §4.0. Bolting extra techniques into a notebook because they sound impressive makes the result unablatable and the paper buzzword-heavy. If you think an unselected technique belongs, say so and explain what it would replace; don't silently include it.

---

### Step 1: Retrieve Target Model Specifications
Before generating code, identify the assigned model run (e.g., M1, M4, M15, M18, M23, or a new novelty-driven ID) and check its exact specifications from `Model_Training_Reference.md`, `Novelty Search.md`, or the user's prompt:
* **Model ID & Name:** Identify the assigned model ID and descriptive name.
* **Chunk:** Which chunk (`Model_Training_Reference.md`'s current structure) does this belong to?
* **Prerequisite / Dependent Checkpoints:** Check if training starts from scratch or loads a prior checkpoint (e.g., M12's winning backbone, a teacher checkpoint for distillation).
* **Input Representation & Preprocessing:** Strictly default to **§2** (Log-mel spectrograms, 128 bins, 8.0s @ 16kHz) unless explicitly overridden by the model specification.
* **Assigned Augmentation:** Check if clean baseline or augmented (§7) — and remember augmentation is a stretch item, not a default requirement.
* **Architecture & Methodology Details:** Apply the exact architectural instructions from the spec.
* **Specific Outputs & Custom Metrics Required:** Automatically include the standard §3 metrics suite and §4.1 ablation block, plus any specialized requirements (unknown-detection AUROC/AUPR, inference latency ms/sample, compression ratio, forgetting curves).

---

### Step 2: Adhere to Project Scientific Context & Background
To ensure your implementation aligns with the core research methodology, keep the following foundational principles in mind:
1. **The Core Research Gap (Why we do this) — CURRENT DIRECTION:**
   Prior respiratory-audio deep learning is a black box evaluated mostly on closed-set accuracy at the published level (~0.60–0.65 official ICBHI), and its interpretability, when present, is *post-hoc* (Grad-CAM/SHAP) and unverified. This project routes diagnosis through a **physics-grounded, label-free acoustic concept bottleneck** — clinically-named concepts (fine/coarse crackle, wheeze pitch band, inspiratory phase, rhonchi, spectral flatness, PAPR) computed by DSP — to ask whether the model *actually listens to the clinical sounds*, whether that reasoning is **correctable** (intervention) and **faithful** (low leakage), and whether the clinical concept space is a **safer, more device/population-robust** place to flag unknown disease. **Retired framing (do not build a notebook around it):** cross-task disagreement as the unseen-disease detector — it lost to a trivial Energy baseline and is a published method; it now appears only as *one baseline detector*.
2. **Dataset & Task Formulation:** ICBHI 2017 (primary), Coswara/SPRSound (OOD + pediatric shift stress tests). Audio only — no imaging data exists in this pipeline. Official 60/40 patient-independent split for all reported numbers.
3. **Efficiency & Deployability:** Track parameter counts, model size in MB, training time per epoch, and inference latency (ms/sample) to support edge-deployment and clinical reliability claims.
4. **Novelty is a rigor/mechanism contribution, not an accuracy race.** The corrected numbers sit at the published level, so the contribution comes from interpretability, faithfulness, and honest evaluation — not from beating SOTA. See `OWMTL_Merged_Decision_Roadmap.md`, `OWMTL_Build_Sheet.md`, and `Novelty Search.md` for the current prioritized plan.

---

### Step 3: Strict Adherence to Protocol Rules (§§1–11)
**IMPORTANT:** You must strictly abide by all rules, coding conventions, schema specifications, and evaluation criteria laid out in §§1–11 of this document. Do not invent custom reporting formats or violate shared constraints. Specifically, ensure that:
1. **Patient-Independent Data Splits (§1):** Never leak cycles from the same patient across train, validation, or test splits. Use a strict patient-independent split or Leave-One-Patient-Out (LOPO) as dictated by sample size.
2. **Real data (§1):** The Dataset class must load actual audio files. No `torch.randn`/`np.random` placeholder data — this is not scaffolding you can commit as a result.
3. **Shared Audio Preprocessing (§2):** Strictly use `sample_rate=16000`, `duration=8.0s`, `n_mels=128`, `n_fft=1024`, `hop_length=160` (10ms), `win_length=400` (25ms), `f_min=50`, `f_max=2000`, and default `seed=42` unless the model specification explicitly overrides them.
4. **Checkpoints & Disconnect Protection (§6 & §11):** Save model weights after *every single epoch* to disk/temp storage so progress is never lost during Kaggle/Colab session timeouts. Always track and save the best checkpoint based on validation primary metric (`icbhi_score` or validation loss). Strictly follow §11.A for PyTorch 2.6+ pickling safety (casting scalars to `int()`/`float()`) and §11.B/C for Kaggle persistence.
5. **Required Metric Suite (§3):** Automatically compute and print Overall Accuracy, Macro/Per-class Precision, Recall, F1, Macro Specificity, ICBHI Score `((Sensitivity + Specificity) / 2)`, and raw + normalized Confusion Matrices. Compute all efficiency metrics (Total/Trainable parameters, Model Size in MB via temp file, Epoch duration, GPU name, and Inference time ms/sample).
6. **Standardized Results JSON (§4 & §4.1):** The notebook must culminate in exporting a single, perfectly formatted JSON file named `results_<MODEL_ID>.json` matching the exact schema in §4 of the protocol (including `meta`, `config`, `environment`, `dataset_info`, `efficiency`, `best_epoch`, `best_metrics`, `ablation`, and `training_history`). Ensure the `ablation` block (§4.1) is fully populated.
7. **Required Visualization (§5):** Generate and display clean inline plots using matplotlib/seaborn: (a) Train/Val Loss vs. Epoch, (b) Train/Val Accuracy/Score vs. Epoch, and (c) Annotated Confusion Matrix (Raw and Normalized).
8. **Team Handoff Download Cell (§11.D):** Include the exact one-click team handoff file download cell at the very end of the notebook.

---

### Step 4: Follow Required Notebook Structure & Organization
Generate the `.ipynb` file cleanly organized into consecutive markdown and code cells following this exact 8-part structure:
* **### Title & Meta-Information:** Descriptive markdown header with Model ID, Name, Author, and brief objective — including which chunk this belongs to and, if novelty-driven, which `Novelty Search.md` item it implements.
* **### Section 1: Environment Setup & Dependencies:** Device selection (`cuda` vs `cpu`), seed setting (`seed=42`), importing PyTorch, torchaudio, librosa, scikit-learn, seaborn, and helper libraries. Include Kaggle/Colab path autodetection if helpful.
* **### Section 2: Configuration & Hyperparameters:** A centralized dictionary or dataclass containing all audio parameters, learning rate, batch size, epoch count, paths, and model flags.
* **### Section 3: Dataset Loading & Patient-Independent Splitting:** Clean dataset class and splitting logic that guarantees zero patient leakage between train and test sets, and that loads real audio.
* **### Section 4: Model Architecture Definition:** Modular, well-documented PyTorch `nn.Module` classes for the backbone and task heads, including any assigned augmentation layers or custom loss functions.
* **### Section 5: Training & Validation Loop:** Robust training loop with `tqdm` progress bars, per-epoch metric logging, automatic best-model saving, checkpoint persistence to disk, and PyTorch 2.6+ scalar casting.
* **### Section 6: Comprehensive Evaluation & Visualization:** Loading the best checkpoint with `weights_only=False` and generating the full §3 metric suite and §5 plots (Loss curves, Accuracy curves, Confusion matrices).
* **### Section 7: Exporting Protocol-Compliant Results JSON & Handoff Cell:** Building the exact JSON data structure required by §4 and saving `results_<MODEL_ID>.json` to the working directory. Finally, include the exact one-click team handoff file download cell required by §11.D at the very end of the notebook.
* **### Section 8: Summary & Key Takeaways:** A concise markdown summary of final performance, efficiency numbers, and observations ready for project reporting — including an explicit note on what this result means for the relevant novelty claim.

---

### Step 5: Execute Coding Best Practices
* Write clean, idiomatic PyTorch code with clear variable names and inline docstrings.
* Handle potential division-by-zero or edge cases gracefully in custom metric calculations.
* Ensure the code is self-contained and runnable without manual intervention once dataset paths are set.
* Do not omit or truncate code blocks—provide the complete, functional implementation for every single cell.

---

## Quick Reference

```
SHARED AUDIO PARAMS:
  Sample rate: 16000 Hz | Duration: 8s | Mel bins: 128
  Hop: 160 | Window: 400 | Freq: 50–2000 Hz

DEFAULT SEED: 42

REQUIRED OUTPUTS PER MODEL:
  ✓ results_M<ID>.json     (metrics + config + efficiency + ablation)
  ✓ best_model.pth         (best checkpoint)
  ✓ loss_curve.png
  ✓ accuracy_curve.png
  ✓ f1_curve.png
  ✓ confusion_matrix.png

REQUIRED METRICS:
  Accuracy, Precision, Recall, F1 (macro + per-class)
  Confusion Matrix, Params, Model Size, Training Time

ABLATION BLOCK (in results JSON):
  ✓ ablation_group          (which table does this row go in?)
  ✓ ablation_role            (baseline or variant?)
  ✓ baseline_model_id        (compared against which model?)
  ✓ variable_changed         (what's different?)
  ✓ component_flags          (which heads/mechanisms are active?)
  ✓ loss_weights             (actual loss weights used)
  ○ inference_time_ms        (recommended for efficiency table)

NOT ABOUT MODEL COUNT:
  Check Novelty Search.md before starting a new run.
  Real data > more models. Novelty > more models.

NOT ABOUT NOVELTY COUNT EITHER:
  2-3 selected novelty items only (Novelty Search.md §4.0).
  Do NOT make this project buzzword-heavy.
  New technique? It REPLACES a selected item, it does not join it.

CURRENT DIRECTION (2026-08-14):
  Physics-grounded, label-free acoustic CONCEPT BOTTLENECK
  + faithfulness audit. One engine, A/B headline chosen by data at Gate G3.
  Cross-task disagreement = retired thesis, now one baseline detector.
  Source of truth: OWMTL_Merged_Decision_Roadmap.md + OWMTL_Build_Sheet.md

HARD REPORTING RULES (Essentials #8-10):
  ✓ Official 60/40 split + icbhi_score_official ONLY for comparisons (no 70/30)
  ✓ confusion_matrix_raw committed for EVERY model
  ✓ Bootstrap 95% CI + paired test (McNemar/Wilcoxon) on EVERY headline number

AI NOTEBOOK GENERATION:
  See §12 for direct instructions for LLM assistants to generate protocol-compliant .ipynb files.
```

---
