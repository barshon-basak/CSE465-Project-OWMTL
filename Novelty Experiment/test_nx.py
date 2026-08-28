"""
test_nx.py - one runnable self-check for the non-trivial logic in this folder.

    python test_nx.py

Asserts only, no framework. If a statistic here silently breaks, every result in
results/ is wrong, so this is the file to run before trusting a number.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def test_patient_frame():
    """Cycles must aggregate to patients, not stay as cycles."""
    d = {"X": np.array([[1.0, 2], [3, 4], [10, 20]], np.float32),
         "concept_names": ["a", "b"],
         "patient": np.array(["p1", "p1", "p2"]),
         "split": np.array(["train", "train", "test"]),
         "diagnosis": np.array(["COPD", "COPD", "Healthy"])}
    pf = C.patient_frame(d)
    assert len(pf["pid"]) == 2, "two patients expected"
    assert np.allclose(pf["C"][0], [2.0, 3.0]), "p1 must be the mean of its two cycles"
    assert pf["split"][1] == "test"
    print("  ok  patient_frame aggregates cycles -> patients")


def test_honest_verdict():
    assert "NOT SHOWN" in C.honest_verdict(0.45, 0.75)
    assert "ABOVE" in C.honest_verdict(0.55, 0.75)
    assert "BELOW" in C.honest_verdict(0.2, 0.45)
    print("  ok  honest_verdict refuses to call a chance-spanning CI a win")


def test_recall_at_fixed_tpr():
    """A perfectly separating score must catch every unknown at the 95% operating point."""
    y = np.r_[np.zeros(100, int), np.ones(20, int)]
    s = np.r_[np.zeros(100), np.ones(20) + 5]
    r = C.recall_at_fixed_tpr(y, s, 0.95)
    assert r["unknown_recall"] == 1.0, r
    # a useless score catches roughly the false-positive budget, not more
    rng = np.random.default_rng(0)
    r2 = C.recall_at_fixed_tpr(y, rng.standard_normal(120), 0.95)
    assert r2["unknown_recall"] < 0.5, r2
    print("  ok  recall_at_fixed_tpr separates a perfect score from a useless one")


def test_risk_coverage():
    """Confidence that tracks correctness must lower risk as coverage falls."""
    correct = np.r_[np.ones(80), np.zeros(20)]
    good_conf = np.r_[np.ones(80), np.zeros(20)]          # perfectly informative
    cov, risk, aurc_good = C.risk_coverage(correct, good_conf)
    assert risk[0] == 0.0, "the most confident item must be correct"
    assert abs(cov[-1] - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    _, _, aurc_bad = C.risk_coverage(correct, rng.standard_normal(100))
    assert aurc_good < aurc_bad, (aurc_good, aurc_bad)
    print("  ok  risk_coverage rewards informative confidence (lower AURC)")


def test_paired_bootstrap_diff():
    """A real difference must exclude zero; identical scores must not."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60, int), np.ones(60, int)]
    strong = y + rng.normal(0, 0.4, 120)
    weak = rng.normal(0, 1, 120)
    r = C.paired_bootstrap_diff(C.auroc, y, strong, weak)
    assert r["ci95"][0] > 0, r
    same = C.paired_bootstrap_diff(C.auroc, y, strong, strong)
    assert abs(same["diff"]) < 1e-9 and same["ci95"] == [0.0, 0.0], same
    print("  ok  paired_bootstrap_diff detects a real gap and reports zero for identity")


def test_permutation_p():
    """Signal -> small p. Noise -> p roughly uniform, certainly not tiny."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    r_sig = C.permutation_p(C.auroc, y, y + rng.normal(0, 0.3, 100), n_perm=200)
    r_noi = C.permutation_p(C.auroc, y, rng.normal(0, 1, 100), n_perm=200)
    assert r_sig["p"] < 0.05, r_sig
    assert r_noi["p"] > 0.05, r_noi
    print("  ok  permutation_p rejects signal from noise")


def test_ece():
    """A perfectly calibrated predictor must have near-zero ECE; an overconfident one must not."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    perfect = np.full((2000, 2), 0.5)
    e_perf, _ = C.ece_mce(perfect, y)
    over = np.zeros((2000, 2))
    over[np.arange(2000), rng.integers(0, 2, 2000)] = 1.0
    over = np.clip(over, 0.001, 0.999)
    e_over, _ = C.ece_mce(over, y)
    assert e_perf < 0.05, e_perf
    assert e_over > 0.3, e_over
    print("  ok  ece_mce flags an overconfident predictor")


def test_bottleneck_modes():
    """independent must ignore features; opaque must ignore concepts."""
    from N6_physics_bottleneck import BottleneckModel
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    Cm = np.c_[y + rng.normal(0, 0.3, n), rng.normal(0, 1, n)]     # concepts carry y
    Fm = rng.normal(0, 1, (n, 8))                                   # features carry nothing
    # Held-out evaluation. An MLP memorises 8 noise features on its own training rows, so
    # scoring on the fit data would compare two overfits, not two information channels.
    tr, te = slice(0, 300), slice(300, n)

    ind = BottleneckModel("independent", seed=0).fit(Fm[tr], Cm[tr], y[tr])
    opa = BottleneckModel("opaque", seed=0).fit(Fm[tr], Cm[tr], y[tr])
    acc_ind = float((ind.predict(Fm[te], Cm[te]) == y[te]).mean())
    acc_opa = float((opa.predict(Fm[te], Cm[te]) == y[te]).mean())
    assert acc_ind > 0.85, f"independent should read the informative concepts ({acc_ind})"
    assert acc_opa < acc_ind, f"opaque has only noise features ({acc_opa} vs {acc_ind})"

    # independent must be blind to features: changing them cannot change a prediction.
    p1 = ind.predict(Fm[te], Cm[te])
    p2 = ind.predict(rng.normal(0, 5, (n - 300, 8)), Cm[te])
    assert np.array_equal(p1, p2), "independent mode leaked the feature vector"
    print("  ok  bottleneck modes respect their information boundaries")


def test_intervention_curve():
    """Restoring true concepts must recover accuracy when the model truly uses them."""
    from N6_physics_bottleneck import BottleneckModel
    from N7_clinician_intervention import intervention_curve
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    Cm = np.c_[y + rng.normal(0, 0.2, n), rng.normal(0, 1, n)]
    m = BottleneckModel("independent", seed=0).fit(None, Cm, y)
    prior = np.tile(Cm.mean(0), (n, 1))
    curve = intervention_curve(m, None, Cm, prior, y, [0, 1])
    assert curve[-1] > curve[0] + 0.2, curve
    print("  ok  intervention_curve rises when concepts genuinely drive the prediction")


if __name__ == "__main__":
    print("self-check: Novelty Experiment")
    for fn in [test_patient_frame, test_honest_verdict, test_recall_at_fixed_tpr,
               test_risk_coverage, test_paired_bootstrap_diff, test_permutation_p,
               test_ece, test_bottleneck_modes, test_intervention_curve]:
        fn()
    print("\nall checks passed")
