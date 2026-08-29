"""
N5 - Concept Leakage / Faithfulness Audit (hardened)
===================================================

FACULTY ASK: measure concept leakage / faithfulness.

WHY THIS IS A NEW EXPERIMENT (audit finding):
  This one is genuinely DONE - `owmtl/leakage.py` + notebook 03 produced
  `leakage_report.json`: leakage = I(y;f|c) = 0.2209 bits, 13.93% of base, concepts-only
  accuracy 0.9092 vs concepts+features 0.9212, patient-grouped CV over 6311 cycles,
  `estimator_valid: true`, verdict "MODERATE leakage". The estimator is careful work - it
  already fixed two failure modes (class_weight destroying the log-likelihood, and a
  patient-level fit that was underpowered to the point of sign errors).

  What it lacks is everything that makes a single number defensible:
    * NO uncertainty on 0.2209 bits. Is "MODERATE" distinguishable from zero?
    * NO null. If you shuffle the features, how many bits do you get anyway? A CV
      log-likelihood difference between a 14-d and a 782-d model has a positive bias by
      construction, and nobody has measured it.
    * NO per-concept breakdown. "Leakage is 0.2209" does not say WHICH concept fails to
      carry its share - which is the only actionable form of the result.

WHAT THIS ADDS:
    1. The headline reproduced by calling the existing estimator unchanged.
    2. A patient-level bootstrap CI on the leakage, obtained by resampling PATIENTS over
       the per-row held-out log-probabilities (no refit per replicate, so it is cheap and
       the folds stay honest).
    3. A permutation null: features are shuffled ACROSS patients, destroying the f->y link
       while leaving c intact. The resulting null distribution is the positive bias of the
       estimator, and the reported p-value is against that, not against zero.
    4. Drop-one-concept importance: for each concept, how much held-out information is
       lost when it is removed - the actionable ranking, and it flags dead concepts.

DEGRADED MODE:
  I(y;f|c) needs the frozen encoder features. `M2_features.npy` is NOT in the repo (it is
  a Kaggle dataset), so without --features this script runs items 3-4 on the concept side
  only and reports status PARTIAL rather than silently substituting something else.

RUNNING (CPU; ~1 min without features, ~5 min with):
    python N5_leakage_audit.py
    python N5_leakage_audit.py --features /path/to/M2_features.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N5_leakage_audit"


def held_out_logprob_rows(X, y, groups, n_splits=5, seed=0):
    """Per-row held-out log2 p(y_true), patient-grouped.

    Deliberately mirrors `owmtl.leakage._held_out_logprob` (same estimator, same
    LogisticRegressionCV selected by log-loss, same StratifiedGroupKFold) but returns the
    per-row vector instead of its mean. The library returns only the mean, and a CI over
    patients needs the rows.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    counts = np.bincount(y)
    n_splits = int(min(n_splits, counts[counts > 0].min()))
    if n_splits < 2:
        return None
    inner = max(2, min(5, int(counts[counts > 0].min())))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=10, cv=inner, scoring="neg_log_loss",
                                             max_iter=2000))
    cv = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
          if groups is not None else
          StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    proba = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")
    col = {c: i for i, c in enumerate(classes)}
    p = np.array([proba[i, col[y[i]]] for i in range(len(y))])
    return np.log2(np.clip(p, 1e-12, 1.0))


def patient_bootstrap_ci(rows_a, rows_b, groups, n_boot=2000, seed=0):
    """CI on mean(rows_a) - mean(rows_b), resampling PATIENTS (the independent unit)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_p = {p: np.flatnonzero(groups == p) for p in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_p[p] for p in pick])
        vals.append(float(rows_a[idx].mean() - rows_b[idx].mean()))
    point = float(rows_a.mean() - rows_b.mean())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point_bits": round(point, 4),
            "ci95_bits": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": int(n_boot),
            "resampled_unit": "patient"}


def permutation_null(C_mat, F_mat, y, groups, n_perm=5, seed=0):
    """Null distribution of the leakage estimate when features carry no information.

    Feature ROWS are permuted across patients, so f keeps its marginal distribution and
    its dimensionality (both of which drive the estimator's positive bias) but loses any
    link to y. n_perm is small on purpose: each replicate refits LogisticRegressionCV on
    ~780 dimensions (5 outer folds x 10 Cs x 5 inner folds), which costs minutes, not
    seconds - a 20-permutation null runs for hours. 5 replicates pin the null MEAN, which
    is the number the observed leakage must be compared against. The null CI at n_perm=5 is
    indicative only; raise --n_perm before quoting an interval.
    """
    rng = np.random.default_rng(seed)
    base = held_out_logprob_rows(C_mat, y, groups)
    if base is None:
        return None
    null = []
    for i in range(n_perm):
        Fp = F_mat[rng.permutation(len(F_mat))]
        rows = held_out_logprob_rows(np.c_[C_mat, Fp], y, groups, seed=i)
        if rows is not None:
            null.append(float(rows.mean() - base.mean()))
        print(f"    perm {i + 1}/{n_perm}: {null[-1]:+.4f} bits")
    null = np.asarray(null)
    return {"n_perm": int(null.size),
            "null_mean_bits": round(float(null.mean()), 4),
            "null_ci95_bits": [round(float(x), 4) for x in np.percentile(null, [2.5, 97.5])],
            "note": ("This is the estimator's positive bias from adding hundreds of "
                     "uninformative dimensions. Compare the observed leakage against THIS, "
                     "not against zero.")}


def drop_one_concepts(C_mat, y, groups, names, seed=0):
    """Held-out information lost when each concept is removed. Negative or ~0 = dead."""
    full = held_out_logprob_rows(C_mat, y, groups, seed=seed)
    if full is None:
        return []
    base = float(full.mean())
    rows = []
    for j, nm in enumerate(names):
        keep = [i for i in range(C_mat.shape[1]) if i != j]
        r = held_out_logprob_rows(C_mat[:, keep], y, groups, seed=seed)
        loss = base - float(r.mean()) if r is not None else float("nan")
        rows.append({"concept": nm, "bits_lost_if_removed": round(loss, 4)})
        print(f"    {nm:26s} {loss:+.4f} bits")
    rows.sort(key=lambda r: -r["bits_lost_if_removed"])
    dead = [r["concept"] for r in rows if r["bits_lost_if_removed"] <= 0.001]
    return {"per_concept": rows, "base_logprob_bits": round(base, 4),
            "dead_concepts": dead,
            "note": "A concept at or below 0 bits contributes nothing the other 13 do not "
                    "already carry. Report it; do not quietly keep it in the bottleneck."}


def make_figure(doc):
    """Drop-one concept importance, plus the leakage estimate against its null if present."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = doc.get("concept_importance") or {}
    rows = imp.get("per_concept") or []
    if not rows:
        return
    has_leak = "headline" in doc
    fig, axes = plt.subplots(1, 2 if has_leak else 1,
                             figsize=(13 if has_leak else 8, 5.5), squeeze=False)

    ax = axes[0][0]
    rows = sorted(rows, key=lambda r: r["bits_lost_if_removed"])
    vals = [r["bits_lost_if_removed"] for r in rows]
    cols = ["tab:red" if v <= 0.001 else "tab:blue" for v in vals]
    ax.barh(np.arange(len(rows)), vals, color=cols)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r["concept"] for r in rows], fontsize=8)
    ax.set_xlabel("held-out bits lost if this concept is removed")
    ax.set_title("N5 - drop-one concept importance\nred = dead (<= 0.001 bits)")
    ax.grid(alpha=0.3, axis="x")

    if has_leak:
        ax = axes[0][1]
        obs = doc["headline"]["leakage_bits"]
        ci = doc.get("leakage_ci", {}).get("ci95_bits")
        null = doc.get("permutation_null") or {}
        ax.axhline(0, color="k", lw=1)
        if ci:
            ax.plot([0, 0], ci, color="tab:blue", lw=3)
        ax.plot(0, obs, "o", color="tab:blue", ms=9, label=f"observed {obs} bits")
        if null.get("null_mean_bits") is not None:
            nm = null["null_mean_bits"]
            ax.plot(1, nm, "s", color="tab:red", ms=9,
                    label=f"permutation null {nm} bits")
            ncl = null.get("null_ci95_bits")
            if ncl:
                ax.plot([1, 1], ncl, color="tab:red", lw=3)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["observed", "null (shuffled f)"])
        ax.set_ylabel("leakage I(y;f|c), bits")
        ax.set_title("N5 - leakage vs the estimator's own bias")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    C.save_figure(fig, "N5_leakage_audit.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--figure-only", dest="figure_only", action="store_true",
                    help="redraw the figure from the saved JSON without recomputing "
                         "(the permutation null costs minutes per replicate)")
    ap.add_argument("--n_perm", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.figure_only:
        import json
        with open(os.path.join(C.RESULTS_DIR, f"{EXP_ID}.json"), encoding="utf-8") as fh:
            make_figure(json.load(fh))
        return

    C.banner("N5 - Concept Leakage / Faithfulness Audit (hardened)",
             "reproduce the headline, then put a CI, a null and a per-concept ranking on it")
    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))
    names = d["concept_names"]

    lab = {x: i for i, x in enumerate(C.KNOWN_DISEASES)}
    keep = np.isin(d["diagnosis"], C.KNOWN_DISEASES)
    Xc = d["X"][keep]
    y = np.array([lab[s] for s in d["diagnosis"][keep]])
    groups = d["patient"][keep]
    print(f"  cycles {len(y)} | patients {len(np.unique(groups))} | "
          f"class counts {np.bincount(y)}")

    doc = {"experiment": EXP_ID,
           "reproduces": {"source": "owmtl_concept_engine/.../03_concept_leakage_measurement/"
                                    "leakage_report.json",
                          "committed_leakage_bits": 0.2209,
                          "committed_verdict": "MODERATE leakage"},
           "dataset_info": {"dataset": "ICBHI_2017", "level": "cycle, patient-grouped CV",
                            "n_rows": int(len(y)),
                            "n_patients": int(len(np.unique(groups))),
                            "classes": C.KNOWN_DISEASES}}

    if F is not None:
        Fc = F[keep]
        print("\n  -- headline (existing estimator, unchanged)")
        from owmtl.leakage import estimate_leakage
        rep = estimate_leakage(y, Xc, Fc, groups=groups, n_splits=5, seed=args.seed)
        doc["headline"] = rep
        print(f"     leakage {rep['leakage_bits']} bits "
              f"({rep['leakage_frac_of_base']} of base) | valid={rep['estimator_valid']}")

        print("\n  -- patient bootstrap CI on the leakage")
        rows_c = held_out_logprob_rows(Xc, y, groups, seed=args.seed)
        rows_cf = held_out_logprob_rows(np.c_[Xc, Fc], y, groups, seed=args.seed)
        ci = patient_bootstrap_ci(rows_cf, rows_c, groups, args.n_boot, args.seed)
        doc["leakage_ci"] = ci
        print(f"     {ci['point_bits']} bits CI{ci['ci95_bits']} "
              f"excludes_zero={ci['excludes_zero']}")

        print(f"\n  -- permutation null ({args.n_perm} refits, slow)")
        doc["permutation_null"] = permutation_null(Xc, Fc, y, groups, args.n_perm, args.seed)
        if doc["permutation_null"]:
            nl = doc["permutation_null"]["null_ci95_bits"]
            doc["leakage_vs_null"] = {
                "observed_bits": rep["leakage_bits"], "null_ci95_bits": nl,
                "verdict": ("leakage exceeds the estimator's own bias"
                            if rep["leakage_bits"] > nl[1] else
                            "leakage is NOT distinguishable from the estimator's positive "
                            "bias - do not report 'MODERATE leakage' as a finding")}
            print(f"     {doc['leakage_vs_null']['verdict']}")
        doc["status"] = "OK"
    else:
        doc["status"] = "PARTIAL"
        doc["missing"] = {
            "what": "I(y;f|c) requires the frozen encoder features f",
            "file": "M2_features.npy (768-d, aligned to the 6898-cycle index)",
            "why_absent": "not committed to the repo; lives as a Kaggle dataset",
            "how_to_get_it": "owmtl.m2_features.export_features(records, audio_dir, load_m2)",
            "consequence": "the headline leakage number and its CI/null are NOT recomputed "
                           "here; the per-concept ranking below is unaffected"}
        print("\n  [PARTIAL] M2_features.npy not found - the concept-side analysis still "
              "runs;\n            pass --features to recompute I(y;f|c) with a CI and a null.")

    print("\n  -- drop-one-concept importance")
    doc["concept_importance"] = drop_one_concepts(Xc, y, groups, names, args.seed)

    try:
        make_figure(doc)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
