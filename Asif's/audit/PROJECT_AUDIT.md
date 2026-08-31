# OWMTL Project Audit

<<<<<<< HEAD
**Generated:** 2026-08-30 by `Asif's/audit/audit_project.py` · **Files scanned:** 59
=======
**Generated:** 2026-08-30 by `Asif's/audit/audit_project.py` · **Files scanned:** 51
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

Automated protocol-compliance and result-validity audit across every results file in the repository. Each check corresponds to a failure mode actually present in this repo, not a hypothetical one.

---

## Summary

| Severity | Count | Meaning |
|---|---|---|
<<<<<<< HEAD
| 🔴 CRITICAL | 10 | The result does not support the claim made on it |
| 🟡 WARNING | 74 | Needs resolving before submission |
| ⚪ INFO | 100 | Worth knowing, not blocking |
=======
| 🔴 CRITICAL | 19 | The result does not support the claim made on it |
| 🟡 WARNING | 87 | Needs resolving before submission |
| ⚪ INFO | 43 | Worth knowing, not blocking |
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

**Models with critical findings:** M14, M15, M19, M21, M33, M35, M36, M6

**Files with no findings:** M12, M18, M2, M22, M35

---

## 🔴 Critical findings

### M6

**`discrimination_at_or_below_chance`** — `best_metrics.open_set.auroc` = 0.4516 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.open_set.auroc = 0.4516`

<sub>source: `Barshon's/M6/result/results_M6.json`</sub>

### M14

<<<<<<< HEAD
=======
**`discrimination_at_or_below_chance`** — `best_metrics.auroc` = 0.4522 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.auroc = 0.4522`

<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`discrimination_at_or_below_chance`** — `best_metrics.test_auroc` = 0.4809 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.test_auroc = 0.4809`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

### M15

**`icbhi_score_unverifiable`** — `icbhi_score` = 0.0000 is reported with no committed `confusion_matrix_raw`, so neither the macro nor the official ICBHI score can be independently recomputed -- not by a teammate, not by this tool, not by a reviewer who asks. Re-export this run with the confusion matrix (protocol section 4) before the number is used in any table or claim.

> `icbhi_score = 0.0000, confusion_matrix_raw absent`

<sub>source: `Barshon's/M15/v6/results_M15.json`</sub>

<<<<<<< HEAD
=======
### M16

**`not_protocol_compliant`** — `M16_metrics.json` is not a §4 `results_M16.json`. It holds 3 loose field(s) instead of the required schema (meta/config/efficiency/best_metrics/ablation/training_history). The M28 merge cannot consume this, and it carries none of the efficiency, per-class or confusion-matrix data the paper needs.

> `fields: Student_Accuracy, Compression_Ratio, Student_Params`

<sub>source: `Barshon's/M16/M16_metrics.json`</sub>

### M18

**`not_protocol_compliant`** — `M18_metrics.json` is not a §4 `results_M18.json`. It holds 3 loose field(s) instead of the required schema (meta/config/efficiency/best_metrics/ablation/training_history). The M28 merge cannot consume this, and it carries none of the efficiency, per-class or confusion-matrix data the paper needs.

> `fields: Best_Accuracy, Best_Size_MB, Best_Pruning_Ratio`

<sub>source: `Barshon's/M18/M18_metrics.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
### M19

**`discrimination_at_or_below_chance`** — `best_metrics.auroc` = 0.3287 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.auroc = 0.3287`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

**`discrimination_at_or_below_chance`** — `best_metrics.per_dataset_results.Coswara_OOD.auroc` = 0.4881 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.per_dataset_results.Coswara_OOD.auroc = 0.4881`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

**`discrimination_at_or_below_chance`** — `best_metrics.per_dataset_results.SPRSound_OOD.auroc` = 0.3226 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `best_metrics.per_dataset_results.SPRSound_OOD.auroc = 0.3226`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

<<<<<<< HEAD
=======
**`not_protocol_compliant`** — `M19_metrics.json` is not a §4 `results_M19.json`. It holds 4 loose field(s) instead of the required schema (meta/config/efficiency/best_metrics/ablation/training_history). The M28 merge cannot consume this, and it carries none of the efficiency, per-class or confusion-matrix data the paper needs.

> `fields: Coswara_OOD_AUROC, SPRSound_OOD_AUROC, Overall_OOD_AUROC, Overall_OOD_AUPR`

<sub>source: `Barshon's/M19/M19_metrics.json`</sub>

**`discrimination_at_or_below_chance`** — `Coswara_OOD_AUROC` = 0.4881 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `Coswara_OOD_AUROC = 0.4881`

<sub>source: `Barshon's/M19/M19_metrics.json`</sub>

**`discrimination_at_or_below_chance`** — `SPRSound_OOD_AUROC` = 0.3226 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `SPRSound_OOD_AUROC = 0.3226`

<sub>source: `Barshon's/M19/M19_metrics.json`</sub>

**`discrimination_at_or_below_chance`** — `Overall_OOD_AUROC` = 0.3287 is at or below chance (0.5). A random scorer would do as well or better, so this value cannot support a detection claim.

> `Overall_OOD_AUROC = 0.3287`

<sub>source: `Barshon's/M19/M19_metrics.json`</sub>

### M20

**`not_protocol_compliant`** — `M20_metrics.json` is not a §4 `results_M20.json`. It holds 6 loose field(s) instead of the required schema (meta/config/efficiency/best_metrics/ablation/training_history). The M28 merge cannot consume this, and it carries none of the efficiency, per-class or confusion-matrix data the paper needs.

> `fields: ECE_Uncalibrated, ECE_Calibrated, Optimal_Temperature, NLL_Uncalibrated, NLL_Calibrated, ECE_Improvement`

<sub>source: `Barshon's/M20/M20_metrics.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
### M21

**`abnormal_detection_collapse`** — Official ICBHI sensitivity is 0.0186 -- the model detects almost no abnormal (Crackle/Wheeze/Both) events, which is the entire clinical point of the task. The reported score is being carried by the specificity term. This is majority-class collapse, not a working model.

> `Se = 0.0186, Sp = 0.9808`

<<<<<<< HEAD
<sub>source: `Barshon's\M21\M21_Updated\results_M21\results_M21.json`</sub>
=======
<sub>source: `Barshon's/M21/M21_metrics.json`</sub>

### M30

**`icbhi_score_unverifiable`** — `icbhi_score` = 0.8213 is reported with no committed `confusion_matrix_raw`, so neither the macro nor the official ICBHI score can be independently recomputed -- not by a teammate, not by this tool, not by a reviewer who asks. Re-export this run with the confusion matrix (protocol section 4) before the number is used in any table or claim.

> `icbhi_score = 0.8213, confusion_matrix_raw absent`

<sub>source: `Barshon's/M30/results_M30.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M33

**`normal_detection_collapse`** — Official ICBHI specificity is 0.0000 -- the model almost never classifies a Normal cycle correctly, so it would flag nearly every healthy patient. The reported score is being carried by the sensitivity term.

> `Se = 0.6660, Sp = 0.0000`

<sub>source: `Barshon's/M33/results_M33.json`</sub>

### M35

**`best_epoch_is_first`** — The best epoch is epoch 1 of 30. Training made the model worse from the very first update, which usually indicates a broken loss, label mismatch, or a learning rate far too high.

> `best_epoch=1, num_epochs=30`

<sub>source: `Barshon's/M35_v2/results_M35.json`</sub>

### M36

**`abnormal_detection_collapse`** — Official ICBHI sensitivity is 0.0932 -- the model detects almost no abnormal (Crackle/Wheeze/Both) events, which is the entire clinical point of the task. The reported score is being carried by the specificity term. This is majority-class collapse, not a working model.

> `Se = 0.0932, Sp = 0.9171`

<sub>source: `Barshon's/M36/results_M36.json`</sub>

---

## 🟡 Warnings

### ?

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_independent.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: precision_macro, recall_macro.

> `precision_macro, recall_macro`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_independent.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_leaky.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: precision_macro, recall_macro.

> `precision_macro, recall_macro`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_leaky.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_opaque.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: precision_macro, recall_macro.

> `precision_macro, recall_macro`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_opaque.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_sequential.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: precision_macro, recall_macro.

> `precision_macro, recall_macro`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_sequential.json`</sub>

### M2

**`no_better_than_majority_class`** — Accuracy 0.5116 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5116, majority prior=0.5729`

<<<<<<< HEAD
<sub>source: `Asif's\M2\m2_v4\results_M2.json`</sub>
=======
<sub>source: `Asif's/M2/17aug_run_v2/results_M2.json`</sub>

**`non_official_icbhi_score`** — `icbhi_score` = 0.6167 is the macro form ((recall_macro + specificity_macro)/2), not the ICBHI 2017 challenge metric, and is not comparable to published ICBHI results -- it is inflated by ~0.06-0.22 in this repo. No `icbhi_score_official` field is present. Run `Asif's/audit/icbhi_score_audit.py --write` to add it.

> `icbhi_score = 0.6167`

<sub>source: `Asif's/M2/17aug_run_v2/results_M2.json`</sub>

**`no_better_than_majority_class`** — Accuracy 0.4917 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.4917, majority prior=0.5729`

<sub>source: `Asif's/M2/29aug_run_v3/results_M2.json`</sub>

**`non_official_icbhi_score`** — `icbhi_score` = 0.5780 is the macro form ((recall_macro + specificity_macro)/2), not the ICBHI 2017 challenge metric, and is not comparable to published ICBHI results -- it is inflated by ~0.06-0.22 in this repo. No `icbhi_score_official` field is present. Run `Asif's/audit/icbhi_score_audit.py --write` to add it.

> `icbhi_score = 0.5780`

<sub>source: `Asif's/M2/29aug_run_v3/results_M2.json`</sub>

**`no_better_than_majority_class`** — Accuracy 0.5116 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5116, majority prior=0.5729`

<sub>source: `Asif's/M2/m2_v4/results_M2.json`</sub>

**`non_official_icbhi_score`** — `icbhi_score` = 0.5480 is the macro form ((recall_macro + specificity_macro)/2), not the ICBHI 2017 challenge metric, and is not comparable to published ICBHI results -- it is inflated by ~0.06-0.22 in this repo. No `icbhi_score_official` field is present. Run `Asif's/audit/icbhi_score_audit.py --write` to add it.

> `icbhi_score = 0.5480`

<sub>source: `Asif's/M2/m2_v4/results_M2.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M3

**`no_better_than_majority_class`** — Accuracy 0.5591 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5591, majority prior=0.5729`

<<<<<<< HEAD
<sub>source: `Asif's\M3\29 aug run\results_M3.json`</sub>
=======
<sub>source: `Asif's/M3/17aug_run_result/results_M3.json`</sub>

**`non_official_icbhi_score`** — `icbhi_score` = 0.6239 is the macro form ((recall_macro + specificity_macro)/2), not the ICBHI 2017 challenge metric, and is not comparable to published ICBHI results -- it is inflated by ~0.06-0.22 in this repo. No `icbhi_score_official` field is present. Run `Asif's/audit/icbhi_score_audit.py --write` to add it.

> `icbhi_score = 0.6239`

<sub>source: `Asif's/M3/17aug_run_result/results_M3.json`</sub>

**`no_better_than_majority_class`** — Accuracy 0.5591 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5591, majority prior=0.5729`

<sub>source: `Asif's/M3/29 aug run/results_M3.json`</sub>

**`non_official_icbhi_score`** — `icbhi_score` = 0.5639 is the macro form ((recall_macro + specificity_macro)/2), not the ICBHI 2017 challenge metric, and is not comparable to published ICBHI results -- it is inflated by ~0.06-0.22 in this repo. No `icbhi_score_official` field is present. Run `Asif's/audit/icbhi_score_audit.py --write` to add it.

> `icbhi_score = 0.5639`

<sub>source: `Asif's/M3/29 aug run/results_M3.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

**`best_epoch_very_early`** — Best epoch 4 of 40 (10% through the budget). The remaining 36 epochs only overfit. Worth reporting, and worth checking the model was given a fair chance to converge.

> `best_epoch=4/40`

<sub>source: `Asif's/M3/results_M3.json`</sub>

**`no_better_than_majority_class`** — Accuracy 0.5372 is not meaningfully above the majority-class rate 0.5918 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5372, majority prior=0.5918`

<sub>source: `Asif's\M3_v2\Results\results_M3_v2.json`</sub>

### M6

**`no_better_than_majority_class`** — Accuracy 0.7689 is not meaningfully above the majority-class rate 0.8994 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.7689, majority prior=0.8994`

<sub>source: `Barshon's/M6/result/results_M6.json`</sub>

### M7

**`schema_missing_blocks`** — Missing §4 block(s): efficiency, best_epoch, best_metrics, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `efficiency, best_epoch, best_metrics, training_history`

<sub>source: `Sami's/M7/M7_handoff/results_M7.json`</sub>

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Sami's/M7/M7_handoff/results_M7.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`schema_missing_blocks`** — Missing §4 block(s): efficiency, best_epoch, best_metrics, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `efficiency, best_epoch, best_metrics, training_history`

<sub>source: `Sami's/M7/M7_handoff/results_M7_aug.json`</sub>

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Sami's/M7/M7_handoff/results_M7_aug.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
### M11

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch, ablation, training_history`

<<<<<<< HEAD
<sub>source: `Barshon's\M11\results_M11.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Barshon's\M11\results_M11.json`</sub>

### M12

**`schema_missing_blocks`** — Missing §4 block(s): config, environment, best_epoch, best_metrics, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `config, environment, best_epoch, best_metrics, ablation, training_history`

<sub>source: `Asif's\M12_v2\results_M12_v2.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Asif's\M12_v2\results_M12_v2.json`</sub>
=======
<sub>source: `Barshon's/M11/results_M11 (3).json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M11/results_M11 (3).json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Barshon's/M11/results_M11 (3).json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M14

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

<<<<<<< HEAD
> `best_epoch, training_history`
=======
> `efficiency, best_epoch, training_history`

<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>

**`weak_discrimination`** — `baseline_comparisons.m15_auroc` = 0.5782 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `baseline_comparisons.m15_auroc = 0.5782`

<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>

**`weak_discrimination`** — `baseline_comparisons.m29_energy_auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `baseline_comparisons.m29_energy_auroc = 0.6466`

<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): efficiency, best_epoch, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `efficiency, best_epoch, training_history`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

**`weak_discrimination`** — `best_metrics.cal_auroc` = 0.5842 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.cal_auroc = 0.5842`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

**`weak_discrimination`** — `baseline_comparisons.m15_auroc` = 0.5782 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `baseline_comparisons.m15_auroc = 0.5782`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

**`weak_discrimination`** — `baseline_comparisons.m29_energy_auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `baseline_comparisons.m29_energy_auroc = 0.6466`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

### M15

<<<<<<< HEAD
=======
**`schema_missing_blocks`** — Missing §4 block(s): efficiency, best_epoch, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `efficiency, best_epoch, training_history`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

**`weak_discrimination`** — `best_metrics.auroc` = 0.5782 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.auroc = 0.5782`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

**`weak_discrimination`** — `baseline_comparisons.m29_energy_auroc` = 0.6005 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `baseline_comparisons.m29_energy_auroc = 0.6005`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`weak_discrimination`** — `best_metrics.open_set.auroc` = 0.5747 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.open_set.auroc = 0.5747`

<sub>source: `Barshon's/M15/v6/results_M15.json`</sub>

**`weak_discrimination`** — `best_metrics.open_set.baseline_energy_auroc` = 0.5948 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.open_set.baseline_energy_auroc = 0.5948`

<sub>source: `Barshon's/M15/v6/results_M15.json`</sub>

### M16

**`no_better_than_majority_class`** — Accuracy 0.9192 is not meaningfully above the majority-class rate 0.9026 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.9192, majority prior=0.9026`

<<<<<<< HEAD
<sub>source: `Barshon's\M16\M16_v2\results_M16\results_M16.json`</sub>
=======
<sub>source: `Barshon's/M16/results_M16.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M16/results_M16.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M16/results_M16.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M17

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

<<<<<<< HEAD
> `best_epoch, training_history`
=======
> `efficiency, best_epoch, training_history`

<sub>source: `Barshon's/M17/results_M17.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

<sub>source: `Barshon's/M17/results_M17.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M17/results_M17.json`</sub>

### M18

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<<<<<<< HEAD
<sub>source: `Barshon's\M18\M18_updated\results_M18\results_M18.json`</sub>
=======
<sub>source: `Barshon's/M18/results_M18.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M18/results_M18.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M18/results_M18.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M19

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch, training_history`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`weak_discrimination`** — `best_metrics.per_dataset_results.ICBHI_Unknown.auroc` = 0.5543 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.per_dataset_results.ICBHI_Unknown.auroc = 0.5543`

<sub>source: `Barshon's/M19/results_M19.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M19/results_M19.json`</sub>

### M20

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `best_epoch`

<<<<<<< HEAD
<sub>source: `Barshon's\M20\M20_updated\results_M20\results_M20.json`</sub>
=======
<sub>source: `Barshon's/M20/results_M20.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

**`no_better_than_majority_class`** — Accuracy 0.8983 is not meaningfully above the majority-class rate 0.9026 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.8983, majority prior=0.9026`

<<<<<<< HEAD
<sub>source: `Barshon's\M20\M20_updated\results_M20\results_M20.json`</sub>
=======
<sub>source: `Barshon's/M20/results_M20.json`</sub>

**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M20/results_M20.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M21

**`no_better_than_majority_class`** — Accuracy 0.5880 is not meaningfully above the majority-class rate 0.5918 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5880, majority prior=0.5918`

<<<<<<< HEAD
<sub>source: `Barshon's\M21\M21_Updated\results_M21\results_M21.json`</sub>
=======
<sub>source: `Barshon's/M21/results_M21.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M22

**`no_better_than_majority_class`** — Accuracy 0.5880 is not meaningfully above the majority-class rate 0.5918 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

<<<<<<< HEAD
> `accuracy=0.5880, majority prior=0.5918`
=======
<sub>source: `Barshon's/M21/results_M21.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

<sub>source: `Asif's\M22_v2\Results\results_M22_v2.json`</sub>

<<<<<<< HEAD
**`no_better_than_majority_class`** — Accuracy 0.5798 is not meaningfully above the majority-class rate 0.5729 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.5798, majority prior=0.5729`

<sub>source: `Asif's\M22_v2\Results\results_M22_v2_official.json`</sub>
=======
<sub>source: `Barshon's/M21/results_M21.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M24

**`schema_missing_blocks`** — Missing §4 block(s): best_epoch, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

<<<<<<< HEAD
> `best_epoch, ablation, training_history`
=======
> `efficiency, best_epoch, ablation, training_history`

<sub>source: `Barshon's/M24/results_M24.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

<sub>source: `Barshon's/M24/results_M24.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Barshon's/M24/results_M24.json`</sub>

**`weak_discrimination`** — `config.m15_baseline_auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `config.m15_baseline_auroc = 0.6466`

<sub>source: `Barshon's/M24/results_M24.json`</sub>

**`weak_discrimination`** — `best_metrics.m15_baseline_auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.m15_baseline_auroc = 0.6466`

<sub>source: `Barshon's/M24/results_M24.json`</sub>

### M28

**`schema_missing_blocks`** — Missing §4 block(s): config, environment, dataset_info, best_epoch, best_metrics, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

<<<<<<< HEAD
> `config, environment, dataset_info, best_epoch, best_metrics, ablation, training_history`
=======
> `config, environment, dataset_info, efficiency, best_epoch, best_metrics, ablation, training_history`

<sub>source: `Barshon's/M28/results_M28.json`</sub>

**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

<sub>source: `Barshon's/M28/results_M28.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Barshon's/M28/results_M28.json`</sub>

<<<<<<< HEAD
=======
**`weak_discrimination`** — `novelty_highlights.openmax_baseline_auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `novelty_highlights.openmax_baseline_auroc = 0.6466`

<sub>source: `Barshon's/M28/results_M28.json`</sub>

**`weak_discrimination`** — `novelty_highlights.m17_stage2_auroc` = 0.6120 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `novelty_highlights.m17_stage2_auroc = 0.6120`

<sub>source: `Barshon's/M28/results_M28.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`patient_independence_unclear`** — `dataset_info.split_method` = 'None' does not state a patient-independent split, and `patient_leakage_verified` is absent. Protocol §1 calls this a research-validity requirement, not a style choice -- it should be asserted in code, not assumed.

<sub>source: `Barshon's/M28/results_M28.json`</sub>

### M29

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's/M29/results_M29.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`weak_discrimination`** — `best_metrics.open_set.auroc` = 0.6466 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.open_set.auroc = 0.6466`

<sub>source: `Asif's/M29/results_M29.json`</sub>

### M30

**`best_epoch_very_early`** — Best epoch 3 of 30 (10% through the budget). The remaining 27 epochs only overfit. Worth reporting, and worth checking the model was given a fair chance to converge.

> `best_epoch=3/30`

<sub>source: `Asif's/M30_v2/results_M30.json`</sub>

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: precision_macro, recall_macro, f1_macro.

> `precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's/M30/results_M30.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
### M32

**`no_better_than_majority_class`** — Accuracy 0.4802 is not meaningfully above the majority-class rate 0.4653 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.4802, majority prior=0.4653`

<sub>source: `Barshon's/M32/results_M32.json`</sub>

### M33

**`no_better_than_majority_class`** — Accuracy 0.3561 is not meaningfully above the majority-class rate 0.4653 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.3561, majority prior=0.4653`

<sub>source: `Barshon's/M33/results_M33.json`</sub>

**`best_epoch_very_early`** — Best epoch 4 of 30 (13% through the budget). The remaining 26 epochs only overfit. Worth reporting, and worth checking the model was given a fair chance to converge.

> `best_epoch=4/30`

<sub>source: `Barshon's/M33/results_M33.json`</sub>

### M36

**`no_better_than_majority_class`** — Accuracy 0.4765 is not meaningfully above the majority-class rate 0.4653 (always predicting the largest class). The model may be learning little; report macro-F1 rather than accuracy and check the per-class breakdown.

> `accuracy=0.4765, majority prior=0.4653`

<sub>source: `Barshon's/M36/results_M36.json`</sub>

### M39

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's/M39/2nd_run_handoff/results_M39.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`weak_discrimination`** — `best_metrics.gate_g2.per_concept.crackle_score.auroc` = 0.5580 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.gate_g2.per_concept.crackle_score.auroc = 0.5580`

<sub>source: `Asif's/M39/2nd_run_handoff/results_M39.json`</sub>

**`weak_discrimination`** — `best_metrics.gate_g2.per_concept.wheeze_score.auroc` = 0.5340 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.gate_g2.per_concept.wheeze_score.auroc = 0.5340`

<sub>source: `Asif's/M39/2nd_run_handoff/results_M39.json`</sub>

<<<<<<< HEAD
=======
**`schema_missing_metrics`** — Missing §3 metric(s) in `best_metrics`: accuracy, precision_macro, recall_macro, f1_macro.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's/M39/M39_handoff/results_M39.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`weak_discrimination`** — `best_metrics.gate_g2.per_concept.crackle_score.auroc` = 0.5506 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.gate_g2.per_concept.crackle_score.auroc = 0.5506`

<sub>source: `Asif's/M39/M39_handoff/results_M39.json`</sub>

**`weak_discrimination`** — `best_metrics.gate_g2.per_concept.wheeze_score.auroc` = 0.5729 is weak. Conventionally <0.7 AUROC is considered poor discrimination; this needs framing as a negative or preliminary result, not a validated mechanism.

> `best_metrics.gate_g2.per_concept.wheeze_score.auroc = 0.5729`

<sub>source: `Asif's/M39/M39_handoff/results_M39.json`</sub>

### M40

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M40.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M40.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M40_aug.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M40_aug.json`</sub>

### M41

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M41.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M41.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M41_aug.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M41_aug.json`</sub>

### M42

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M42.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M42.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, ablation, training_history`

<sub>source: `M40_M43_transformers\Results\results_M42_aug.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `M40_M43_transformers\Results\results_M42_aug.json`</sub>

### M44

**`schema_missing_blocks`** — Missing §4 block(s): config, environment, best_epoch, best_metrics, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `config, environment, best_epoch, best_metrics, ablation, training_history`

<sub>source: `Asif's\M44\results_M44.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Asif's\M44\results_M44.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): config, environment, best_epoch, best_metrics, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `config, environment, best_epoch, best_metrics, ablation, training_history`

<sub>source: `Asif's\M44\results_M44_A0_wrap.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Asif's\M44\results_M44_A0_wrap.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): config, environment, best_epoch, best_metrics, ablation, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `config, environment, best_epoch, best_metrics, ablation, training_history`

<sub>source: `Asif's\M44\results_M44_P4_zero.json`</sub>

**`schema_missing_ablation`** — No §4.1 `ablation` block. Without it this run cannot be placed in the ablation table automatically at M28.

<sub>source: `Asif's\M44\results_M44_P4_zero.json`</sub>

### M45

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_A2.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_A3.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_A4.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_A5.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_A6.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_P1.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_P2.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_P3.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_P4.json`</sub>

**`schema_missing_blocks`** — Missing §4 block(s): environment, training_history. The M28 merge script expects every block; absent ones must be reconstructed by hand.

> `environment, training_history`

<sub>source: `Asif's\M45\results_M45_P5.json`</sub>

---

## ⚪ Informational

### ?

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_independent.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_official_60_40') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_independent.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_leaky.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_official_60_40') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_leaky.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_opaque.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_official_60_40') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_opaque.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_sequential.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_official_60_40') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `owmtl_concept_engine/owmtl_concept_engine/notebooks/002_concept_bottleneck_training/output/results_M13cbm_sequential.json`</sub>

### M1

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_60_40') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M1/results_M1.json`</sub>

### M6

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M6/result/results_M6.json`</sub>

### M7

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: deep ensemble handoff - carries per-member results and ensemble disagreement; it has no single top-level best_metrics block by design. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Sami's\M7\M7_handoff\results_M7.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `Sami's/M7/M7_handoff/results_M7.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_60_40_with_calibration_carveout') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Sami's/M7/M7_handoff/results_M7.json`</sub>

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: deep ensemble handoff - carries per-member results and ensemble disagreement; it has no single top-level best_metrics block by design. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Sami's\M7\M7_handoff\results_M7_aug.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `Sami's/M7/M7_handoff/results_M7_aug.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_60_40_with_calibration_carveout') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Sami's/M7/M7_handoff/results_M7_aug.json`</sub>

### M11

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: post-hoc calibrators fitted on frozen logits. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's\M11\results_M11.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's\M11\results_M11.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_60_20_20') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<<<<<<< HEAD
<sub>source: `Barshon's\M11\results_M11.json`</sub>

### M12

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, specificity_macro, icbhi_score, with a recorded reason: backbone re-decision - inference only over frozen checkpoints. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro, specificity_macro, icbhi_score`

<sub>source: `Asif's\M12_v2\results_M12_v2.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M12_v2\results_M12_v2.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_icbhi_60_40_patient_disjoint') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M12_v2\results_M12_v2.json`</sub>
=======
<sub>source: `Barshon's/M11/results_M11 (3).json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M13

**`schema_incomplete_ablation`** — `ablation` block missing: loss_weights.

> `loss_weights`

<sub>source: `Barshon's/M13/results_M13.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's/M13/results_M13.json`</sub>

### M14

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: conformal wrapper over M15's scores - no network of its own. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<<<<<<< HEAD
<sub>source: `Barshon's\M14\v2\results_M14.json`</sub>
=======
<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<<<<<<< HEAD
<sub>source: `Barshon's\M14\v2\results_M14.json`</sub>
=======
<sub>source: `Barshon's/M14/v1/results_M14.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

**`cites_subchance_reference_value`** — `baseline_comparisons.reference_m6_openmax_auroc` = 0.4516 is cited here as context from another model's already-audited result (it is at or below chance). This is not a claim about M14's own performance -- see that other model's audit entry for the underlying problem -- but do not treat it as a validity benchmark for M14.

> `baseline_comparisons.reference_m6_openmax_auroc = 0.4516`

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_60_20_20') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M14/v2/results_M14.json`</sub>

### M15

<<<<<<< HEAD
=======
**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

**`cites_subchance_reference_value`** — `baseline_comparisons.reference_m6_openmax_auroc` = 0.4516 is cited here as context from another model's already-audited result (it is at or below chance). This is not a claim about M15's own performance -- see that other model's audit entry for the underlying problem -- but do not treat it as a validity benchmark for M15.

> `baseline_comparisons.reference_m6_openmax_auroc = 0.4516`

<sub>source: `Barshon's/M15/v4/results_M15.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M15/v6/results_M15.json`</sub>

### M16

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<<<<<<< HEAD
<sub>source: `Barshon's\M16\M16_v2\results_M16\results_M16.json`</sub>
=======
<sub>source: `Barshon's/M16/results_M16.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M17

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: continual-learning forgetting curve - reports stage-0 retention and stage-2 plasticity accuracy plus AUROC, not 4-class metrics. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's\M17\results_M17.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Barshon's/M17/results_M17.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's\M17\results_M17.json`</sub>

### M18

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<<<<<<< HEAD
<sub>source: `Barshon's\M18\M18_updated\results_M18\results_M18.json`</sub>
=======
<sub>source: `Barshon's/M18/results_M18.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M19

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: cross-dataset evaluation of an existing checkpoint. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's\M19\results_M19.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's/M19/results_M19.json`</sub>

### M20

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<<<<<<< HEAD
<sub>source: `Barshon's\M20\M20_updated\results_M20\results_M20.json`</sub>
=======
<sub>source: `Barshon's/M20/results_M20.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M21

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<<<<<<< HEAD
<sub>source: `Barshon's\M21\M21_Updated\results_M21\results_M21.json`</sub>

### M22

**`non_patient_independent_by_design`** — Run declares itself NOT patient-independent and names the overlapping patients [156, 218]. This is the published ICBHI partition reproduced verbatim for comparability. Valid to report against the literature; NOT eligible as the project's best model, and it must never be described as patient-independent.

<sub>source: `Asif's\M22_v2\Results\results_M22_v2_official.json`</sub>
=======
<sub>source: `Barshon's/M21/results_M21.json`</sub>
>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9

### M24

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: open-world disease recognition - reports AUROC, AUPR and FPR@95TPR over known/unknown patients, not 4-class metrics. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's\M24\results_M24.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's\M24\results_M24.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M24/results_M24.json`</sub>

### M28

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: master merge of other models' results. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Barshon's\M28\results_M28.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's\M28\results_M28.json`</sub>

### M29

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: post-hoc OOD scoring over frozen M12 embeddings. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M29\results_M29.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's/M29/results_M29.json`</sub>

**`cites_subchance_reference_value`** — `best_metrics.reference_M6_openmax.auroc` = 0.4516 is cited here as context from another model's already-audited result (it is at or below chance). This is not a claim about M29's own performance -- see that other model's audit entry for the underlying problem -- but do not treat it as a validity benchmark for M29.

> `best_metrics.reference_M6_openmax.auroc = 0.4516`

<sub>source: `Asif's/M29/results_M29.json`</sub>

<<<<<<< HEAD
=======
### M30

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Barshon's/M30/results_M30.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M30/results_M30.json`</sub>

>>>>>>> 46798765c0fc827d2539b234a0f5e374d75278b9
### M31

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M31/results_M31.json`</sub>

### M32

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M32/results_M32.json`</sub>

### M33

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M33/results_M33.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M33_v2/results_M33.json`</sub>

### M34

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M34/results_M34.json`</sub>

### M35

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M35_v2/results_M35.json`</sub>

### M36

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M36/results_M36.json`</sub>

### M37

**`patient_independence_not_asserted`** — Split is declared patient-independent ('patient_independent_70_30') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Barshon's/M37_v2/results_M37.json`</sub>

### M39

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: concept-validity gate - scores DSP concepts by AUROC against cycle labels; it trains nothing and emits no 4-class prediction. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M39\2nd_run_handoff\results_M39.json`</sub>

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: concept-validity gate - scores DSP concepts by AUROC against cycle labels; it trains nothing and emits no 4-class prediction. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M39\M39_handoff\results_M39.json`</sub>

### M40

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M40.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M40.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M40_aug.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M40_aug.json`</sub>

### M41

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M41.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M41.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M41_aug.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M41_aug.json`</sub>

### M42

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M42.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M42.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `M40_M43_transformers\Results\results_M42_aug.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `M40_M43_transformers\Results\results_M42_aug.json`</sub>

### M44

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: XAI analysis of M22_v2 - explains a model, does not train one. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M44\results_M44.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M44\results_M44.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M44\results_M44.json`</sub>

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: XAI analysis of M22_v2 - explains a model, does not train one. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M44\results_M44_A0_wrap.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M44\results_M44_A0_wrap.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M44\results_M44_A0_wrap.json`</sub>

**`metrics_not_applicable`** — Missing §3 metric(s) accuracy, precision_macro, recall_macro, f1_macro, with a recorded reason: XAI analysis of M22_v2 - explains a model, does not train one. Accuracy/precision/recall/F1 are undefined for this file; it is not an incomplete model record.

> `accuracy, precision_macro, recall_macro, f1_macro`

<sub>source: `Asif's\M44\results_M44_P4_zero.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M44\results_M44_P4_zero.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M44\results_M44_P4_zero.json`</sub>

### M45

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_A2.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_A2.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_A2.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_A3.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_A3.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_A3.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_A4.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_A4.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_A4.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_A5.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_A5.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_A5.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_A6.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_A6.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_A6.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_P1.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_P1.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_P1.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_P2.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_P2.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_P2.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_P3.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_P3.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_P3.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_P4.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_P4.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_P4.json`</sub>

**`schema_incomplete_ablation`** — `ablation` block missing: component_flags, loss_weights.

> `component_flags, loss_weights`

<sub>source: `Asif's\M45\results_M45_P5.json`</sub>

**`no_inference_latency`** — `inference_time_ms_per_sample` not measured. Recommended by §4 and it feeds the efficiency columns of the ablation table.

<sub>source: `Asif's\M45\results_M45_P5.json`</sub>

**`patient_independence_not_asserted`** — Split is declared patient-independent ('official_60_40_patient_independent_corrected') but `patient_leakage_verified` is not set, so nothing checked it at runtime.

<sub>source: `Asif's\M45\results_M45_P5.json`</sub>

---

## What the checks look for

| Check | Catches |
|---|---|
| `synthetic_data_not_real_dataset` | a Dataset that fabricates its input instead of loading audio |
| `model_may_be_randomly_initialised` | a fallback that proceeds with an untrained model |
| `discrimination_at_or_below_chance` | AUROC ≤ 0.5 — a coin flip does as well |
| `comparison_against_subchance_baseline` | "beats baseline by N%" where the baseline is below chance |
| `perfect_metrics_implausible` | metrics of exactly 1.0 — leak or train-set evaluation |
| `metric_constant_across_sweep` | accuracy that does not move as capacity changes |
| `single_class_collapse` | one class at recall 1.0, the rest at 0.0 |
| `frozen_validation_metric` | validation score identical every epoch |
| `no_better_than_majority_class` | accuracy at the class prior |
| `best_epoch_is_first` / `_very_early` | the model never really trained |
| `missing_open_set_metrics` | M6/M15/M17 with no AUROC/AUPR |
| `not_protocol_compliant` | metrics files the M28 merge cannot read |
| `schema_*` | §4 / §4.1 blocks the merge expects |
| `patient_independence_*` | protocol §1, the one non-negotiable requirement |
| `non_official_icbhi_score` | the macro ICBHI score reported without the challenge metric |
| `icbhi_score_unverifiable` | an ICBHI score with no confusion matrix to recompute it from |
| `official_icbhi_score_mismatch` | stated official score disagrees with the file's own matrix |
| `abnormal_detection_collapse` | official Se < 0.15 — detects almost no crackles/wheezes |
| `normal_detection_collapse` | official Sp < 0.15 — flags nearly every healthy patient |
| `perfect_open_set_operating_point` | open-set precision/recall of exactly 1.0 at a threshold |

