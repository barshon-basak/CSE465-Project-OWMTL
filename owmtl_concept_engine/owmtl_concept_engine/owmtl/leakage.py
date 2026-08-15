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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _held_out_logprob(X, y, n_splits=5, seed=0):
    """Mean held-out log2 p(y_true) from a CV logistic model. Higher = more info in X."""
    classes = np.unique(y)
    if len(classes) < 2 or np.min(np.bincount(y)) < 2:
        return float("nan"), float("nan")
    n_splits = int(min(n_splits, np.min(np.bincount(y))))
    if n_splits < 2:
        return float("nan"), float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    # map class -> column
    col = {c: i for i, c in enumerate(classes)}
    p_true = np.array([proba[i, col[y[i]]] for i in range(len(y))])
    p_true = np.clip(p_true, 1e-12, 1.0)
    logp2 = np.log2(p_true)
    # accuracy of the CV model too
    pred = classes[np.argmax(proba, axis=1)]
    acc = float(np.mean(pred == y))
    return float(np.mean(logp2)), acc


def estimate_leakage(y, concepts, features, n_splits=5, seed=0):
    """Return a leakage report dict.

    y        : (N,) int patient/cycle disease labels
    concepts : (N, Cc) physics concept matrix  c
    features : (N, Cf) encoder feature matrix   f  (e.g. frozen M2 embeddings)
    """
    y = np.asarray(y).astype(int)
    C = np.asarray(concepts, dtype=float)
    F = np.asarray(features, dtype=float)
    cf = np.concatenate([C, F], axis=1)

    lp_c, acc_c = _held_out_logprob(C, y, n_splits, seed)       # concepts only
    lp_cf, acc_cf = _held_out_logprob(cf, y, n_splits, seed)    # concepts + features
    lp_f, acc_f = _held_out_logprob(F, y, n_splits, seed)       # features only (ref)

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
        "interpretation": _interpret(leakage_bits, base_bits),
        "method": "held-out CV log-likelihood; leakage = I(y;f|c) in bits (arXiv:2504.09459 view)",
    }


def _interpret(bits, base):
    if bits != bits:
        return "insufficient data (a class had too few samples for CV)."
    if bits <= 0.05 * base:
        return ("LOW leakage: concepts are ~sufficient; the bottleneck is faithful "
                "(supports Path A).")
    if bits <= 0.25 * base:
        return ("MODERATE leakage: features add some signal beyond concepts; report and "
                "consider a leakage penalty.")
    return ("HIGH leakage: the model gains substantial information outside the clinical "
            "concepts -> concepts insufficient / bottleneck bypassable (supports Path B — "
            "the collapse/leakage IS the finding).")
