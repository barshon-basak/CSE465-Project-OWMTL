#!/usr/bin/env python3
"""m45_paired.py - paired tests of each ablation row against the A0 baseline.

WHY THIS EXISTS SEPARATELY FROM THE TABLE
    `M45_ablation_table.json` reports each row's score and its delta from A0. A delta is
    not a result until it survives a paired test on the unit the protocol names, and for
    this project that unit is the PATIENT (Model_Training_Protocol.md section 1).

WHAT IT FOUND, AND WHY IT MATTERS MORE THAN THE DELTAS
    The two units disagree, and the disagreement is the point:

        row  delta    McNemar (cycle, n=2636)   paired bootstrap (patient, n=47)
        P1   -0.0089  p = 0.128  not sig        [-0.059, +0.034]  not shown to differ
        P2   -0.0355  p = 6.6e-6 SIGNIFICANT    [-0.089, +0.011]  not shown to differ
        P3   +0.0162  p = 0.040  SIGNIFICANT    [-0.017, +0.046]  not shown to differ

    Scored per cycle, two of three rows look conclusive. Scored per patient, none do.
    Cycles within a patient are not independent, so a cycle-level test is answering a
    question about 2,636 correlated draws and reporting it as though they were 2,636
    independent ones. This project's own Section 3 is about measurement faults that make
    results look stronger than they are; shipping the McNemar p-values alone would have
    been another one.

    Both are reported. The patient-level interval is the one that decides.

    python "Asif's/M45/m45_paired.py"
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
A0_PREDS = os.path.join(REPO, "Asif's", "M22_v2", "Results", "preds_M22_v2.csv")
OUT = os.path.join(HERE, "M45_paired_tests.json")
LABELS = [0, 1, 2, 3]


def official(cm):
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else np.nan
    abn = cm[1:].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else np.nan
    return (se + sp) / 2


def paired_patient_bootstrap(y, pred_a, pred_b, pid, n_boot=4000, seed=42):
    """Resample PATIENTS and score both models on the same resample.

    The same resample for both is what makes it paired: it cancels the patient-mix noise
    that dominates a 47-patient test set, which an unpaired comparison of two separate CIs
    does not.
    """
    from sklearn.metrics import confusion_matrix
    uq = np.unique(pid)
    ix = {p: np.flatnonzero(pid == p) for p in uq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = np.concatenate([ix[p] for p in rng.choice(uq, len(uq), replace=True)])
        da = official(confusion_matrix(y[s], pred_a[s], labels=LABELS))
        db = official(confusion_matrix(y[s], pred_b[s], labels=LABELS))
        if da == da and db == db:
            diffs.append(db - da)
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return round(float(lo), 4), round(float(hi), 4), round(float(min(p, 1.0)), 4)


def main() -> int:
    sys.path.insert(0, os.path.join(REPO, "Asif's", "Statistics"))
    from owmtl_scores import mcnemar
    from sklearn.metrics import confusion_matrix

    rows = list(csv.DictReader(open(A0_PREDS, encoding="utf-8")))
    y = np.array([int(r["y_true"]) for r in rows])
    a0 = np.array([int(r["y_pred"]) for r in rows])
    pid = np.array([int(r["patient_id"]) for r in rows])
    base = official(confusion_matrix(y, a0, labels=LABELS))

    doc = {"baseline": "A0 (M22_v2)", "baseline_official": round(float(base), 4),
           "split": "official_60_40_patient_independent_corrected",
           "n_test_cycles": int(len(y)), "n_test_patients": int(len(np.unique(pid))),
           "unit_note": "The patient-level interval decides. The cycle-level McNemar is "
                        "reported beside it because the two disagree, and that "
                        "disagreement is itself a finding about unit of analysis.",
           "rows": {}}

    print(f"  A0 = {base:.4f} | {len(y)} cycles from {len(np.unique(pid))} patients\n")
    print(f"  {'row':4s} {'score':>7s} {'delta':>8s}   {'patient 95% CI':<20s} {'p':>6s}  "
          f"{'McNemar p':>10s}  verdict")
    for row in sorted(f.split("_")[-1].split(".")[0]
                      for f in os.listdir(HERE) if f.startswith("preds_M45_")):
        f = os.path.join(HERE, f"preds_M45_{row}.npy")
        d = np.load(f, allow_pickle=True).item()
        if not np.array_equal(d["y_true"], y):
            print(f"  {row}: label order differs from A0 - refusing to pair")
            continue
        b = d["y_pred"]
        score = official(confusion_matrix(y, b, labels=LABELS))
        lo, hi, p = paired_patient_bootstrap(y, a0, b, pid)
        mc = mcnemar(y, a0, b, names=("A0", row))
        differs = lo * hi > 0
        doc["rows"][row] = {
            "official": round(float(score), 4), "delta_vs_A0": round(float(score - base), 4),
            "patient_paired_ci95": [lo, hi], "patient_paired_p": p,
            "cycle_mcnemar_p": round(float(mc["p_value"]), 6),
            "cycle_mcnemar_significant": bool(mc["significant_at_0.05"]),
            "verdict": ("differs from A0" if differs else "NOT shown to differ from A0"),
        }
        print(f"  {row:4s} {score:7.4f} {score-base:+8.4f}   [{lo:+.4f}, {hi:+.4f}]  {p:6.3f}  "
              f"{mc['p_value']:10.2e}  {'DIFFERS' if differs else 'not shown to differ'}")

    disagree = [r for r, v in doc["rows"].items()
                if v["cycle_mcnemar_significant"] and "NOT" in v["verdict"]]
    doc["unit_disagreement"] = sorted(disagree)
    print(f"\n  Rows the cycle-level test calls significant and the patient-level test does "
          f"not: {', '.join(disagree) if disagree else 'none'}")
    print("  Report the patient-level column. n=47, not n=2636.")

    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\n[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
