"""Checks for the two things that silently broke the 2026-08-26 run.

  1. the official split loader   -- a hand-typed patient list was 15/47 correct
  2. the leakage estimator       -- returned "Path A" on data with a planted leak

Run: python3 tests/test_leakage_and_split.py
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

from owmtl.icbhi_data import load_split
from owmtl.leakage import estimate_leakage


def test_split():
    """Numbers come from the audit in Asif's/audit/official_split.py."""
    _, off = load_split(mode="official")
    assert off["n_recordings"] == 920, off["n_recordings"]
    assert off["n_patients"] == 126, off["n_patients"]
    assert (off["n_train"], off["n_test"]) == (539, 381), (off["n_train"], off["n_test"])
    assert off["leaking_patients"] == ["156", "218"], off["leaking_patients"]
    assert not off["is_patient_independent"]

    rows, pi = load_split(mode="patient_independent")
    assert (pi["n_train"], pi["n_test"]) == (551, 369), (pi["n_train"], pi["n_test"])
    assert pi["is_patient_independent"]

    # the property the whole protocol rests on: no patient on both sides
    sides = {}
    for stem, sp in rows.items():
        sides.setdefault(stem.split("_")[0], set()).add(sp)
    assert not [p for p, v in sides.items() if len(v) > 1]

    # and the guard against a hand-typed list ever passing again
    test_pat = {s.split("_")[0] for s, v in rows.items() if v == "test"}
    assert "103" not in test_pat, "103 is a TRAIN patient; the 2026-08-26 list called it test"
    assert "102" in test_pat, "102 is a TEST patient; the 2026-08-26 list called it train"
    print("split      OK  official 539/381, patient-independent 551/369, no leakage")


def _leak_case(delta, seed=0):
    """Cycle-level rows, 104 patients, concepts carry y; features carry delta*y extra."""
    rng = np.random.default_rng(seed)
    n = 6898
    g = rng.integers(0, 104, n)
    y = g % 3
    c = rng.standard_normal((n, 14))
    c[:, 0] += y
    f = rng.standard_normal((n, 128))
    f[:, 0] += delta * y
    return estimate_leakage(y, c, f, groups=g)


def test_leakage_recovers_a_planted_leak():
    none, weak, strong = _leak_case(0.0), _leak_case(0.8), _leak_case(3.0)
    for r in (none, weak, strong):
        assert r["estimator_valid"], r["interpretation"]
        assert r["patient_grouped_cv"]
    assert none["leakage_bits"] < 0.05, none["leakage_bits"]
    assert "LOW" in none["interpretation"]
    assert weak["leakage_bits"] > none["leakage_bits"]
    assert "MODERATE" in weak["interpretation"]
    assert strong["leakage_bits"] > 0.5, strong["leakage_bits"]
    assert "HIGH" in strong["interpretation"]
    print(f"leakage    OK  none {none['leakage_bits']:+.3f} < weak {weak['leakage_bits']:+.3f} "
          f"< planted {strong['leakage_bits']:+.3f} bits")


def test_leakage_refuses_when_underpowered():
    """The patient-level shape that produced the bad 2026-08-26 report must not
    quietly return a number."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 104)
    c = rng.standard_normal((104, 14))
    c[:, 0] += y
    f = rng.standard_normal((104, 128))
    f[:, 0] += 3 * y
    r = estimate_leakage(y, c, f)
    assert not r["estimator_valid"] or "UNDERPOWERED" in r["interpretation"], r
    print("underpower OK  n=104 patient-level estimate is flagged, not reported")


if __name__ == "__main__":
    test_split()
    test_leakage_recovers_a_planted_leak()
    test_leakage_refuses_when_underpowered()
    print("all checks passed")
