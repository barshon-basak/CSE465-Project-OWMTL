"""
Statistical significance suite for the project's open-set / OOD-detection AUROC claims.

Why this exists: the project's own progress report (Research_Progress_Report.md, "Statistical
validity: 3/10") flags that none of the reported AUROCs anywhere in the repo have a confidence
interval or a significance test attached. With n=19 unknown-class patients, a point-estimate
AUROC alone cannot support "beats baseline" language.

Method: none of the results JSONs in the repo persist raw per-patient scores (only the
aggregate AUROC + class counts), so a bootstrap or DeLong test over real per-patient scores
isn't possible from committed artifacts alone. This script instead uses the Hanley & McNeil
(1982) closed-form variance estimator for AUROC, which needs only the AUROC point estimate and
the positive/negative class counts -- both of which ARE recorded. This is a standard, citable
approximation; it is not a substitute for a paired DeLong test on raw scores, and probably
somewhat *overstates* variance for pairs of methods evaluated on the same patients (an
independent-samples z-test is conservative relative to a paired one). See the report's
"Limitations" section for what would be needed to tighten this later.

Run: python3 compute_significance.py   (stdlib only, no deps)
"""
import json
import math
import os

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def hanley_mcneil_se(auc, n_pos, n_neg):
    """SE of AUROC via Hanley & McNeil (1982). n_pos = positive (unknown/OOD) class count,
    n_neg = negative (known/in-distribution) class count."""
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc ** 2)
        + (n_neg - 1) * (q2 - auc ** 2)
    ) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def ci95(auc, n_pos, n_neg):
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    lo, hi = auc - 1.96 * se, auc + 1.96 * se
    return se, max(0.0, lo), min(1.0, hi)


def z_test_independent(auc1, n_pos1, n_neg1, auc2, n_pos2, n_neg2):
    """Two-sided z-test for a difference between two AUROCs measured on independent samples.
    Conservative (wider CI than a paired DeLong test) when the two methods actually share
    patients, which is the case for every comparison below."""
    se1 = hanley_mcneil_se(auc1, n_pos1, n_neg1)
    se2 = hanley_mcneil_se(auc2, n_pos2, n_neg2)
    se_diff = math.sqrt(se1 ** 2 + se2 ** 2)
    z = (auc1 - auc2) / se_diff if se_diff > 0 else float("nan")
    # two-sided p-value from the standard normal CDF via erf
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p, se_diff


# ---------------------------------------------------------------------------
# Every AUROC in the repo with clean, unambiguous n_pos (unknown) / n_neg (known) counts,
# pulled directly from the committed results JSONs. Each entry cites its source file so the
# numbers can be re-verified against the repo at any time -- do not hand-edit these values.
# ---------------------------------------------------------------------------
ENTRIES = [
    dict(
        model="M6", label="OpenMax + Weibull (cycle-level)",
        auc=0.4516, n_pos=549, n_neg=1631, level="cycle",
        source="Barshon's/M6/result/results_M6.json:best_metrics.open_set",
    ),
    dict(
        model="M14v1", label="Conformal-wrapped disagreement v1",
        auc=0.4522, n_pos=19, n_neg=22, level="patient",
        source="Archive_Files (v4)/M14_v1_superseded/results_M14.json:best_metrics.auroc",
    ),
    dict(
        model="M14v2", label="Conformal-wrapped disagreement v2 (test)",
        auc=0.4809, n_pos=19, n_neg=22, level="patient",
        source="Barshon's/M14/v2/results_M14.json:best_metrics.test_auroc",
    ),
    dict(
        model="M15v6-Disagreement", label="Cross-task disagreement (v6, bug-fixed)",
        auc=0.5747, n_pos=19, n_neg=42, level="patient",
        source="Barshon's/M15/v6/results_M15.json:best_metrics.open_set.auroc",
        note="n_pos/n_neg not recorded in this file -- borrowed from M29's split (same "
             "104 known / 19 unknown patient pool) since v6 doesn't state its own test-set size.",
    ),
    dict(
        model="M15v6-EnergyRecomputed", label="Energy baseline, recomputed inside M15 v6",
        auc=0.5948, n_pos=19, n_neg=42, level="patient",
        source="Barshon's/M15/v6/results_M15.json:best_metrics.open_set.baseline_energy_auroc",
        note="Same n caveat as above. Sanity-check value: should be close to M29's own Energy "
             "AUROC (0.6466) since it's nominally the same method on the same backbone.",
    ),
    dict(
        model="M29-Energy", label="Energy score on frozen M12 backbone (best M29 scorer)",
        auc=0.6466, n_pos=19, n_neg=42, level="patient",
        source="Asif's/M29/results_M29.json:best_metrics.open_set",
    ),
    dict(
        model="M29-Entropy", label="Entropy score, M29 suite",
        auc=0.5789, n_pos=19, n_neg=42, level="patient",
        source="Asif's/M29/results_M29.json:best_metrics.all_scores[Entropy]",
    ),
    dict(
        model="M29-Mahalanobis_patient", label="Mahalanobis (patient-fit), M29 suite",
        auc=0.5689, n_pos=19, n_neg=42, level="patient",
        source="Asif's/M29/results_M29.json:best_metrics.all_scores[Mahalanobis_patient]",
    ),
    dict(
        model="M29-Mahalanobis_class", label="Mahalanobis (class-fit), M29 suite",
        auc=0.5376, n_pos=19, n_neg=42, level="patient",
        source="Asif's/M29/results_M29.json:best_metrics.all_scores[Mahalanobis_class]",
    ),
    dict(
        model="M29-MSP", label="Max softmax probability, M29 suite",
        auc=0.5025, n_pos=19, n_neg=42, level="patient",
        source="Asif's/M29/results_M29.json:best_metrics.all_scores[MSP]",
    ),
]

# Excluded entirely -- documented in the report instead of silently guessed at:
#   M15 main/v4 (results_M15.json): dataset_info states 22 unknown patients, which conflicts
#     with the 19-patient unknown group used by every other model (M6, M14, M29). No test-set
#     n_pos/n_neg is given alongside the AUROC itself, so a CI here would be built on an
#     assumption, not a recorded number.
#   M19 (all three datasets): num_samples is the count of ONE class only (the OOD/target set);
#     the paired in-distribution reference count needed for Hanley-McNeil isn't recorded.
#     Coswara_OOD in particular has num_samples=2 -- flagged qualitatively instead: no AUROC
#     computed from 2 samples can mean anything, regardless of its point value (0.4881).

# Head-to-head comparisons that matter for the paper's actual claims.
COMPARISONS = [
    ("M29-Energy", "M6",
     "Does the zero-training M29 baseline legitimately beat the M6 disease-head baseline?"),
    ("M29-Energy", "M15v6-Disagreement",
     "Headline question: does the selected novelty mechanism (cross-task disagreement) beat "
     "the trivial post-hoc baseline it's supposed to improve on?"),
    ("M29-Energy", "M15v6-EnergyRecomputed",
     "Sanity check: M15 v6 recomputes its own Energy baseline on what should be the same "
     "backbone/split as M29 -- these two numbers should agree if both pipelines are correct."),
    ("M29-Energy", "M14v2",
     "Does the conformal-calibrated score (selected novelty item 2) beat the trivial baseline?"),
]


def main():
    by_model = {e["model"]: e for e in ENTRIES}
    results = {"entries": [], "comparisons": []}

    print("=" * 78)
    print("HANLEY-McNEIL 95% CONFIDENCE INTERVALS")
    print("=" * 78)
    for e in ENTRIES:
        se, lo, hi = ci95(e["auc"], e["n_pos"], e["n_neg"])
        row = dict(e, se=round(se, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
                   crosses_chance=lo <= 0.5 <= hi)
        results["entries"].append(row)
        flag = "  <-- 95% CI includes chance (0.5)" if row["crosses_chance"] else ""
        print(f"{e['model']:28s} AUROC={e['auc']:.4f}  n=({e['n_pos']:>4d} pos / "
              f"{e['n_neg']:>4d} neg, {e['level']:6s})  95% CI=[{lo:.4f}, {hi:.4f}]{flag}")

    print()
    print("=" * 78)
    print("PAIRWISE SIGNIFICANCE TESTS (independent-samples z-test, conservative)")
    print("=" * 78)
    for name_a, name_b, question in COMPARISONS:
        a, b = by_model[name_a], by_model[name_b]
        z, p, se_diff = z_test_independent(a["auc"], a["n_pos"], a["n_neg"],
                                            b["auc"], b["n_pos"], b["n_neg"])
        sig = p < 0.05
        row = dict(a=name_a, b=name_b, question=question,
                   auc_a=a["auc"], auc_b=b["auc"], diff=round(a["auc"] - b["auc"], 4),
                   z=round(z, 4), p_value=round(p, 6), significant_at_05=sig)
        results["comparisons"].append(row)
        print(f"\n{name_a} ({a['auc']:.4f}) vs {name_b} ({b['auc']:.4f})")
        print(f"  Q: {question}")
        print(f"  diff={row['diff']:+.4f}  z={z:.3f}  p={p:.4f}  "
              f"{'SIGNIFICANT (p<0.05)' if sig else 'NOT significant (p>=0.05)'}")

    out_path = os.path.join(os.path.dirname(__file__), "significance_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
