#!/usr/bin/env python3
"""
CC1 — canonical raw-score format + the paired tests it unlocks.

WHY THIS EXISTS
---------------
`SIGNIFICANCE_REPORT.md` limitation #1: *"No raw per-patient scores are saved anywhere in the
repo — every results JSON stores only the aggregate AUROC plus class counts. A proper paired
DeLong test needs the raw score vector per method."* So every comparison in that report had to
fall back on a conservative independent-samples z-test.

`Model_Training_Protocol.md` §1 essential #10 now *requires* a paired test on every headline
comparison. That requirement is unmeetable until notebooks dump raw scores. This module defines
the format and implements the tests, so the rule has a payoff rather than being a convention
nobody can cash in.

USE FROM A NOTEBOOK
-------------------
    dump_scores("scores_M35.csv", model_id="M35", unit_type="cycle",
                unit_ids=pids, scores=s, labels=y, score_name="crackle_prob")

Generators should splice `SCORE_DUMP_CELL` (bottom of this file) so every engine notebook emits
the same schema without importing anything — Colab runtimes do not have this repo on sys.path.

THEN, LOCALLY
-------------
    a = load_scores("scores_M35.csv"); b = load_scores("scores_M29.csv")
    print(delong_test(*align(a, b)))

THE UNIT-TYPE GUARD
-------------------
`align()` refuses to pair score sets whose `unit_type` differs. That is trap #6 from
`Asif's/CLAUDE.md` (cycle-level evaluation where the protocol says patient-level) encoded in the
tooling: a cycle-level AUROC and a patient-level AUROC are not the same quantity, and silently
comparing them is exactly how M6 ended up non-comparable to everything else.

Dependencies: numpy only. Normal tail probabilities use `math.erfc`, so scipy is not required.
"""
import csv
import math
import os

import numpy as np

SCHEMA = ["model_id", "split", "unit_type", "unit_id", "score_name", "score", "label"]
VALID_UNIT_TYPES = ("patient", "cycle", "recording")


# --------------------------------------------------------------------------- I/O
def dump_scores(path, model_id, unit_type, unit_ids, scores, labels,
                score_name="score", split="test", append=False):
    """Write raw per-unit scores in the canonical schema.

    unit_type must be one of patient/cycle/recording -- it is what makes a later comparison
    legitimate or meaningless, so it is required rather than inferred.
    labels: 1 = positive class (unknown/OOD/abnormal), 0 = negative.
    """
    if unit_type not in VALID_UNIT_TYPES:
        raise ValueError(f"unit_type must be one of {VALID_UNIT_TYPES}, got {unit_type!r}")
    unit_ids = list(unit_ids)
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    if not (len(unit_ids) == len(scores) == len(labels)):
        raise ValueError(f"length mismatch: {len(unit_ids)} ids, {len(scores)} scores, "
                         f"{len(labels)} labels")
    if len(set(labels.tolist())) < 2:
        raise ValueError("labels must contain both classes; a one-class score file cannot "
                         "support any of the tests this format exists for")
    if len(set(map(str, unit_ids))) != len(unit_ids):
        raise ValueError("unit_ids must be unique -- duplicates make paired alignment ambiguous")

    exists = os.path.exists(path) and append
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(SCHEMA)
        for u, s, y in zip(unit_ids, scores, labels):
            w.writerow([model_id, split, unit_type, str(u), score_name,
                        f"{float(s):.10g}", int(y)])
    return path


class ScoreSet(dict):
    """{unit_id: (score, label)} plus provenance. dict so it stays trivially inspectable."""

    def __init__(self, mapping, model_id, unit_type, score_name, split):
        super().__init__(mapping)
        self.model_id, self.unit_type = model_id, unit_type
        self.score_name, self.split = score_name, split

    def __repr__(self):
        return (f"ScoreSet({self.model_id}/{self.score_name}, {self.unit_type}-level, "
                f"n={len(self)}, split={self.split})")

    def arrays(self):
        ids = sorted(self)
        s = np.array([self[i][0] for i in ids], dtype=float)
        y = np.array([self[i][1] for i in ids], dtype=int)
        return ids, s, y


def load_scores(path, score_name=None, split="test"):
    """Read one score series. If the file holds several score_names, pass which one."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("split", "test") != split:
                continue
            if score_name is not None and r["score_name"] != score_name:
                continue
            rows.append(r)
    if not rows:
        raise ValueError(f"No rows in {path} for score_name={score_name!r} split={split!r}")

    names = {r["score_name"] for r in rows}
    if len(names) > 1:
        raise ValueError(f"{path} holds several score_names {sorted(names)} -- pass score_name=")
    types = {r["unit_type"] for r in rows}
    if len(types) > 1:
        raise ValueError(f"{path} mixes unit_types {sorted(types)} -- refusing to load")

    return ScoreSet({r["unit_id"]: (float(r["score"]), int(r["label"])) for r in rows},
                    model_id=rows[0]["model_id"], unit_type=types.pop(),
                    score_name=names.pop(), split=split)


def align(a, b, require_full=False):
    """Inner-join two ScoreSets on unit_id. Returns (y, score_a, score_b).

    Refuses to align across unit_types -- see the module docstring.
    """
    if a.unit_type != b.unit_type:
        raise ValueError(
            f"Refusing to pair {a.unit_type}-level scores ({a.model_id}) with "
            f"{b.unit_type}-level scores ({b.model_id}). These are different quantities: "
            f"patient-level aggregation averages out cycle noise and is not comparable to a "
            f"cycle-level AUROC. Re-evaluate one of them at the other's level first.")
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(f"No shared unit_ids between {a.model_id} and {b.model_id}")
    if require_full and (len(shared) != len(a) or len(shared) != len(b)):
        raise ValueError(f"Incomplete overlap: {len(shared)} shared vs {len(a)}/{len(b)}")

    ya = np.array([a[u][1] for u in shared])
    yb = np.array([b[u][1] for u in shared])
    if not np.array_equal(ya, yb):
        bad = [u for u in shared if a[u][1] != b[u][1]]
        raise ValueError(f"Labels disagree on {len(bad)} shared unit(s), e.g. {bad[:5]}. "
                         f"The two files describe different ground truth.")
    if len(shared) < len(a) or len(shared) < len(b):
        print(f"[align] using {len(shared)} shared units "
              f"({a.model_id}: {len(a)}, {b.model_id}: {len(b)})")
    return (ya,
            np.array([a[u][0] for u in shared], dtype=float),
            np.array([b[u][0] for u in shared], dtype=float))


# --------------------------------------------------------------------- statistics
def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _norm_sf2(z):
    """Two-sided normal tail probability, without scipy."""
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def _midrank(x):
    """Midranks (ties averaged), 1-based -- the core of the fast DeLong algorithm."""
    J = np.argsort(x, kind="mergesort")
    Z = np.asarray(x, dtype=float)[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(scores_by_method, n_pos):
    """DeLong AUCs + covariance (Sun & Xu 2014).

    scores_by_method: (k, n) array with the n_pos positive-class columns FIRST.
    Returns (aucs shape (k,), cov shape (k, k)).
    """
    m, k = n_pos, scores_by_method.shape[0]
    n = scores_by_method.shape[1] - m
    pos, neg = scores_by_method[:, :m], scores_by_method[:, m:]

    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(scores_by_method[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n            # positive-side structural components
    v10 = 1.0 - (tz[:, m:] - ty) / m      # negative-side
    sx = np.atleast_2d(np.cov(v01, ddof=1))
    sy = np.atleast_2d(np.cov(v10, ddof=1))
    return aucs, sx / m + sy / n


def auroc(y, s):
    """AUROC via the DeLong machinery, so it always matches the tests below."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")          # positives first
    a, _ = _fast_delong(np.asarray(s, dtype=float)[order][None, :], int(y.sum()))
    return float(a[0])


def delong_test(y, s1, s2, names=("A", "B")):
    """PAIRED DeLong test for two AUROCs on the SAME units.

    This is the test SIGNIFICANCE_REPORT.md could not run. Being paired, it accounts for the
    correlation between two methods scoring the same patients, so it is more powerful than the
    independent-samples z-test used there.
    """
    y = np.asarray(y).astype(int)
    if len(set(y.tolist())) < 2:
        raise ValueError("y must contain both classes")
    order = np.argsort(-y, kind="mergesort")
    mat = np.vstack([np.asarray(s1, float)[order], np.asarray(s2, float)[order]])
    aucs, cov = _fast_delong(mat, int(y.sum()))

    L = np.array([[1.0, -1.0]])
    var = float(np.asarray(L @ cov @ L.T).reshape(()))
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        z, p = (0.0, 1.0) if diff == 0 else (math.inf * math.copysign(1, diff), 0.0)
    else:
        z = diff / math.sqrt(var)
        p = _norm_sf2(z)
    return {
        "test": "paired_delong",
        "name_a": names[0], "name_b": names[1],
        "auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
        "diff": diff, "se_diff": math.sqrt(var) if var > 0 else 0.0,
        "z": float(z), "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "n_units": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
    }


def delong_ci(y, s, alpha=0.05):
    """95% CI for a single AUROC from the DeLong variance."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    aucs, cov = _fast_delong(np.asarray(s, float)[order][None, :], int(y.sum()))
    se = math.sqrt(max(float(cov[0, 0]), 0.0))
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else _z_for(alpha)
    a = float(aucs[0])
    return {"auroc": a, "se": se,
            "ci_lo": max(0.0, a - z * se), "ci_hi": min(1.0, a + z * se),
            "excludes_chance": bool(a - z * se > 0.5 or a + z * se < 0.5)}


def _z_for(alpha):
    """Inverse normal CDF at 1 - alpha/2, bisection (avoids a scipy dependency)."""
    target = 1.0 - alpha / 2.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        lo, hi = (mid, hi) if cdf < target else (lo, mid)
    return (lo + hi) / 2.0


def mcnemar(y, pred_a, pred_b, names=("A", "B")):
    """Exact-ish McNemar on paired CLASSIFICATION predictions (protocol §1 #10)."""
    y = np.asarray(y).astype(int)
    a_ok = np.asarray(pred_a).astype(int) == y
    b_ok = np.asarray(pred_b).astype(int) == y
    n01 = int(np.sum(~a_ok & b_ok))     # only B right
    n10 = int(np.sum(a_ok & ~b_ok))     # only A right
    n = n01 + n10
    if n == 0:
        p, stat, kind = 1.0, 0.0, "degenerate_no_discordant_pairs"
    elif n <= 1000:
        # binomial two-sided exact test at q=0.5, computed in log space. The direct form
        # sum(comb(n, i)) / 2**n overflows a float at n around 1030, which is reachable
        # whenever two cycle-level models are compared on this corpus (n can exceed 800).
        # It raised OverflowError rather than returning a wrong number, but it raised it
        # inside a paired comparison that callers had no reason to expect to fail.
        k = min(n01, n10)
        log_tail = _logsumexp([math.lgamma(n + 1) - math.lgamma(i + 1)
                               - math.lgamma(n - i + 1) - n * math.log(2.0)
                               for i in range(k + 1)])
        p = min(1.0, 2.0 * math.exp(log_tail))
        stat = (abs(n01 - n10) - 1.0) ** 2 / n
        kind = "mcnemar_exact"
    else:
        # Beyond that the normal approximation is indistinguishable from exact and cheap.
        stat = (abs(n01 - n10) - 1.0) ** 2 / n
        p = float(_norm_sf2(math.sqrt(stat)))
        kind = "mcnemar_chi2_continuity_corrected"
    return {"test": kind, "name_a": names[0], "name_b": names[1],
            "acc_a": float(a_ok.mean()), "acc_b": float(b_ok.mean()),
            "only_a_correct": n10, "only_b_correct": n01, "discordant": n,
            "chi2_cc": float(stat), "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05)}


def bootstrap_auroc_ci(y, s, B=1000, alpha=0.05, seed=42):
    """Stratified bootstrap CI -- the empirical cross-check on delong_ci."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(auroc(y[idx], s[idx]))
    vals = np.sort(vals)
    return {"auroc": auroc(y, s), "boot_mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(vals, 100 * (1 - alpha / 2))), "B": B}


# ------------------------------------------------------- the cell generators splice
SCORE_DUMP_CELL = r'''
# ============================================================
# CC1 — RAW PER-UNIT SCORE DUMP  (Model_Training_Protocol.md section 1, essential #10)
# ============================================================
# Self-contained on purpose: Colab has no access to this repo's modules, so the canonical
# schema is inlined rather than imported. Keep it byte-identical to
# Asif's/Statistics/owmtl_scores.py::SCHEMA -- the paired tests key on these column names.
#
# Without this file, no paired DeLong / McNemar test is possible and every comparison falls
# back to a conservative independent-samples z-test. That is the exact hole
# SIGNIFICANCE_REPORT.md documents.
import csv as _csv

_SCORE_SCHEMA = ["model_id", "split", "unit_type", "unit_id", "score_name", "score", "label"]


def dump_scores(path, model_id, unit_type, unit_ids, scores, labels,
                score_name="score", split="test", append=False):
    """unit_type MUST be 'patient', 'cycle' or 'recording' -- a cycle-level AUROC is not
    comparable to a patient-level one, and the loader refuses to pair across levels."""
    assert unit_type in ("patient", "cycle", "recording"), f"bad unit_type {unit_type!r}"
    unit_ids = [str(u) for u in unit_ids]
    scores = [float(x) for x in scores]
    labels = [int(x) for x in labels]
    assert len(unit_ids) == len(scores) == len(labels), "length mismatch"
    assert len(set(labels)) > 1, "need both classes"
    assert len(set(unit_ids)) == len(unit_ids), "unit_ids must be unique"
    import os as _os
    _exists = _os.path.exists(path) and append
    with open(path, "a" if append else "w", newline="") as _f:
        _w = _csv.writer(_f)
        if not _exists:
            _w.writerow(_SCORE_SCHEMA)
        for _u, _s, _y in zip(unit_ids, scores, labels):
            _w.writerow([model_id, split, unit_type, _u, score_name, f"{_s:.10g}", _y])
    print(f"[CC1] wrote {len(unit_ids)} {unit_type}-level scores -> {path}")
    return path
'''


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Paired tests over CC1 score files.")
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--score-a", default=None)
    ap.add_argument("--score-b", default=None)
    a = ap.parse_args()
    A = load_scores(a.file_a, a.score_a)
    B = load_scores(a.file_b, a.score_b)
    print(A); print(B)
    y, sa, sb = align(A, B)
    import json
    print(json.dumps(delong_test(y, sa, sb, (A.model_id, B.model_id)), indent=2))


if __name__ == "__main__":
    main()
