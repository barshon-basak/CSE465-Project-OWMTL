"""
owmtl.leakage  (Build step 03 — pre-G3, deterministic)
======================================================

Information-theoretic **concept leakage / sufficiency** for the physics concept
bottleneck. This is a Gate-G3 INPUT and one of the paper's core contributions (I2):
"does the diagnosis actually route through the clinical concepts, or does the model
need to bypass them?"

Definition (following the information-theoretic view, arXiv:2504.09459):
    leakage = I(y ; f | c)  = H(y | c) - H(y | c, f)
i.e. the extra predictive information the encoder features `f` carry about the label
`y` OVER AND ABOVE the concepts `c`. Estimated honestly with cross-validated held-out
log-likelihood so it is NOT inflated by overfitting:

    leakage_bits = ( E[log2 p(y | c, f)] - E[log2 p(y | c)] )   (held-out)

Interpretation:
  * leakage ~ 0  -> concepts are SUFFICIENT; the bottleneck is faithful (Path A story).
  * leakage large -> the model wants information outside the clinical concepts; the
    bottleneck loses diagnostic signal / can be bypassed (Path B story).

Also returns the accuracy/log-loss gain of a leaky (c+f) model over a concept-only
model, for a plain-language companion number. sklearn only.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _held_out_logprob(X, y, groups=None, n_splits=5, seed=0):
    """Mean held-out log2 p(y_true) from a CV logistic model. Higher = more info in X."""
    classes = np.unique(y)
    if len(classes) < 2 or np.min(np.bincount(y)) < 2:
        return float("nan"), float("nan")
    n_splits = int(min(n_splits, np.min(np.bincount(y))))
    if n_splits < 2:
        return float("nan"), float("nan")
    # NO class_weight="balanced" here. Balancing deliberately shifts the predicted
    # probabilities away from the true priors, which is fine for a macro-F1 classifier but
    # invalidates a LOG-LIKELIHOOD estimate: the 2026-08-26 run returned held-out log-probs
    # of -1.69/-1.63/-1.70 bits, all WORSE than uniform guessing (log2(1/3) = -1.585), so
    # the leakage figure was a difference of two miscalibrated numbers.
    # Regularisation strength is selected BY LOG-LOSS -- the exact quantity being
    # estimated. With n~100 patients and a ~128-d encoder, a fixed C=1.0 overfits so badly
    # that held-out log-prob drops below uniform and the estimate is unusable; letting the
    # inner CV shrink toward the prior is what makes the number mean anything.
    inner = max(2, min(5, int(np.min(np.bincount(y)))))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegressionCV(Cs=10, cv=inner, scoring="neg_log_loss",
                                             max_iter=2000, n_jobs=None))
    # groups=patient id => no patient appears in both folds. Without this, cycle-level
    # rows from one patient sit on both sides and "leakage" is really patient identity.
    if groups is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, groups=groups, cv=skf, method="predict_proba")
    # map class -> column
    col = {c: i for i, c in enumerate(classes)}
    p_true = np.array([proba[i, col[y[i]]] for i in range(len(y))])
    p_true = np.clip(p_true, 1e-12, 1.0)
    logp2 = np.log2(p_true)
    # accuracy of the CV model too
    pred = classes[np.argmax(proba, axis=1)]
    acc = float(np.mean(pred == y))
    return float(np.mean(logp2)), acc


def estimate_leakage(y, concepts, features, groups=None, n_splits=5, seed=0):
    """Return a leakage report dict.

    y        : (N,) int patient/cycle disease labels
    concepts : (N, Cc) physics concept matrix  c
    features : (N, Cf) encoder feature matrix   f  (e.g. frozen M2 embeddings)
    groups   : (N,) patient ids. REQUIRED whenever rows are cycles rather than patients,
               so cross-validation stays patient-independent.

    Run this at the CYCLE level (N ~ 6900), not the patient level (N ~ 104). Estimating
    I(y;f|c) for a ~128-d f from ~100 rows is underpowered to the point of sign errors: on
    synthetic data with a planted leak it returned "LOW leakage / Path A". Cycle-level rows
    with patient-grouped folds give the estimator something to work with.
    """
    y = np.asarray(y).astype(int)
    C = np.asarray(concepts, dtype=float)
    F = np.asarray(features, dtype=float)
    cf = np.concatenate([C, F], axis=1)

    lp_c, acc_c = _held_out_logprob(C, y, groups, n_splits, seed)     # concepts only
    lp_cf, acc_cf = _held_out_logprob(cf, y, groups, n_splits, seed)  # concepts + features
    lp_f, acc_f = _held_out_logprob(F, y, groups, n_splits, seed)     # features only (ref)

    leakage_bits = (lp_cf - lp_c) if (lp_cf == lp_cf and lp_c == lp_c) else float("nan")
    # normalise by the baseline uncertainty for an interpretable [0..1]-ish ratio
    base_bits = np.log2(len(np.unique(y)))
    return {
        "leakage_bits": round(float(leakage_bits), 4),
        "leakage_frac_of_base": round(float(leakage_bits / base_bits), 4) if base_bits else None,
        "logprob_bits": {"concepts_only": round(lp_c, 4),
                          "concepts_plus_features": round(lp_cf, 4),
                          "features_only": round(lp_f, 4)},
        "accuracy": {"concepts_only": round(acc_c, 4),
                     "concepts_plus_features": round(acc_cf, 4),
                     "features_only": round(acc_f, 4)},
        "accuracy_gain_from_features": round(float(acc_cf - acc_c), 4),
        "uniform_baseline_bits": round(float(-base_bits), 4),
        "n_rows": int(len(y)),
        "rows_per_feature": round(len(y) / max(cf.shape[1], 1), 1),
        "patient_grouped_cv": groups is not None,
        "estimator_valid": bool(lp_c >= -base_bits and lp_cf >= -base_bits),
        "interpretation": _interpret(leakage_bits, base_bits, lp_c, lp_cf, lp_f,
                                     len(y), cf.shape[1]),
        "method": "held-out CV log-likelihood; leakage = I(y;f|c) in bits (arXiv:2504.09459 view)",
    }


def _interpret(bits, base, lp_c=0.0, lp_cf=0.0, lp_f=0.0, n_rows=None, n_feat=1):
    if bits != bits:
        return "insufficient data (a class had too few samples for CV)."
    if n_rows is not None and n_rows / max(n_feat, 1) < 5:
        return (f"UNDERPOWERED: {n_rows} rows for {n_feat} predictors (<5 per predictor). "
                "Estimate I(y;f|c) at the CYCLE level with patient-grouped folds, not the "
                "patient level.")
    if min(lp_c, lp_cf) < -base:
        return ("INVALID: at least one held-out log-probability is worse than uniform "
                f"guessing ({-base:.3f} bits) -- the p(y|c) / p(y|c,f) models are "
                "miscalibrated, "
                "so this leakage estimate is a difference of two meaningless numbers. Do "
                "NOT read a Path A/B decision off it. Check calibration first.")
    if bits <= 0.05 * base:
        return ("LOW leakage: concepts are ~sufficient; the bottleneck is faithful "
                "(supports Path A).")
    if bits <= 0.25 * base:
        return ("MODERATE leakage: features add some signal beyond concepts; report and "
                "consider a leakage penalty.")
    return ("HIGH leakage: the model gains substantial information outside the clinical "
            "concepts -> concepts insufficient / bottleneck bypassable (supports Path B — "
            "the collapse/leakage IS the finding).")
