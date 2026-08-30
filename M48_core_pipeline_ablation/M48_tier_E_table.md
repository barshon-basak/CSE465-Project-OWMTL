# M48 Tier E — evaluation-protocol ablation

Baseline **E0 = 0.5602** — E0 = M22_v2 on the corrected official 60/40 split, official metric.

Every score is recomputed from the run's committed raw confusion matrix.

| Row | Protocol component removed | Kind | Reported | Δ vs E0 | Se | Sp |
|---|---|---|---:|---:|---:|---:|
| `E0` | none — full corrected protocol | baseline | 0.5602 | — | 0.4089 | 0.7115 |
| `E1` | - official metric (report the macro variant instead) | rescore | 0.6140 | +0.0538 | 0.4089 | 0.7115 |
| `E2` | - official split (silent patient-id <= 111 fallback) | rerun | 0.6495 | +0.0893 | 0.5696 | 0.7294 |
| `E3` | - patient independence (published split verbatim, 2 patients leak) | rerun | 0.5641 | +0.0039 | 0.4562 | 0.6719 |
| `E1+E2` | both — the unaudited pipeline, as it would have been published | rescore of E2 | 0.7077 | +0.1475 | 0.5696 | 0.7294 |
| `E5` | − patient-level unit of analysis (test per cycle) | rescore | 7/10 significant | vs 1/10 | — | — |
| `E4` | − device-suffix-safe file join | defect count | 1 recording(s) dropped | — | — | — |

**E1 suite-wide:** across 20 verifiable runs the metric correction is mean 0.0948, max 0.2176, min 0.0025.

**E5:** 7 of 10 M45 rows are significant scored per cycle, 1 scored per patient. Rows that flip: A2, A5, A6, P2, P3, P4. The effective sample size is 47 patients, not 2636 cycles.

## Hypothesis tested by each row

- **E0** — reference: M22_v2, corrected patient-independent 60/40, official metric, raw matrix committed  
  <sub>results_M22_v2.json</sub>
- **E1** — the metric alone inflates the score, non-uniformly, so it does not cancel in a comparison and can reorder models  
  <sub>same predictions rescored; suite-wide over 20 verifiable runs: mean 0.0948, max 0.2176, min 0.0025</sub>
- **E2** — a silently substituted split is worth more than any architectural choice in the paper; 11 test patients and 7.1% of cycles read as a 40% test set  
  <sub>results_M22.json — same architecture, seed and schedule; split only</sub>
- **E3** — the leak is a VALIDITY fault, not a source of inflation — and reporting this null honestly is what makes E1 and E2 credible  
  <sub>results_M22_v2_official.json — patients 156, 218 on both sides</sub>
- **E1+E2** — the two faults compound rather than cancel; this row is the single number that states what the correction was worth  
  <sub>results_M22.json rescored under the macro variant</sub>
