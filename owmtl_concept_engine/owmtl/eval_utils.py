"""
owmtl.eval_utils
================

The honest-evaluation utilities the new protocol makes mandatory (Essentials #8-10):

  * CC1: dump raw per-patient/per-cycle scores  -> unlocks proper paired tests
         (the SIGNIFICANCE_REPORT flagged that NO raw scores are saved anywhere).
  * bootstrap 95% CIs on any metric (AUROC, ICBHI score, accuracy).
  * McNemar paired test on two classifiers' predictions.
  * official ICBHI (Se+Sp)/2 from a 4x4 confusion matrix (pooled abnormal).
  * a §4-schema results-JSON writer that commits confusion_matrix_raw + CI fields.

numpy / scipy / sklearn only.
"""
from __future__ import annotations
import json, os, csv
import numpy as np
from scipy import stats

# ICBHI sound-event class order used throughout the project.
SOUND_CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


# ----------------------------------------------------------------------------- CC1
def dump_scores(path_prefix: str, ids, scores, labels=None, extra: dict | None = None):
    """Write raw per-item scores next to results_*.json so a paired DeLong test is
    possible later. Saves both .npy (fast) and .csv (human-readable)."""
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=float)
    labels = None if labels is None else np.asarray(labels)
    np.save(path_prefix + "_scores.npy",
            {"ids": ids, "scores": scores, "labels": labels, "extra": extra or {}},
            allow_pickle=True)
    with open(path_prefix + "_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "score", "label"])
        for i in range(len(ids)):
            w.writerow([ids[i], scores[i], "" if labels is None else labels[i]])
    return path_prefix + "_scores.npy"


# ------------------------------------------------------------------------ metrics
def auroc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def bootstrap_ci(metric_fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    """Generic paired bootstrap CI. metric_fn(*resampled_arrays) -> float."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = metric_fn(*[a[idx] for a in arrays])
            if v == v:  # not nan
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    point = metric_fn(*arrays)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def auroc_ci(y_true, scores, n_boot=1000, seed=0):
    return bootstrap_ci(auroc, y_true, scores, n_boot=n_boot, seed=seed)


def mcnemar(y_true, pred_a, pred_b):
    """Paired McNemar test between two classifiers' predictions. Returns (stat, p)."""
    y_true = np.asarray(y_true); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b01 = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    b10 = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    # exact binomial (robust for small n, e.g. n=19 regime)
    p = float(stats.binomtest(min(b01, b10), n, 0.5).pvalue)
    stat = (abs(b01 - b10) - 1) ** 2 / n
    return float(stat), p


def icbhi_official_from_confusion(cm: np.ndarray):
    """Official ICBHI (Se+Sp)/2 from a 4x4 confusion matrix over SOUND_CLASSES.
    Se = correct abnormal (Crackle+Wheeze+Both) / all abnormal;  Sp = correct Normal / all Normal."""
    cm = np.asarray(cm, dtype=float)
    assert cm.shape == (4, 4), "confusion matrix must be 4x4 in [Normal,Crackle,Wheeze,Both] order"
    normal_total = cm[0].sum()
    sp = cm[0, 0] / normal_total if normal_total > 0 else float("nan")
    abn_total = cm[1:].sum()
    abn_correct = cm[1, 1] + cm[2, 2] + cm[3, 3]      # each abnormal predicted as its own class
    se = abn_correct / abn_total if abn_total > 0 else float("nan")
    return {"Se": float(se), "Sp": float(sp), "icbhi_score_official": float((se + sp) / 2)}


def confusion_4x4(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])


# ------------------------------------------------------------------ §4 results JSON
def write_results_json(path, *, model_id, model_name, contributor, split,
                       best_metrics, ablation, efficiency=None, config=None,
                       environment=None, dataset_info=None, training_history=None,
                       concept_metrics=None, notes=""):
    """Write a protocol §4-compliant results_<id>.json. best_metrics MUST include
    confusion_matrix_raw and the official score; attach *_ci95 fields for headline
    numbers. concept_metrics carries the new-direction fields (§3.5)."""
    doc = {
        "meta": {"model_id": model_id, "model_name": model_name,
                 "contributor": contributor, "is_augmented": False,
                 "augmentation_method": "none", "notes": notes},
        "config": config or {},
        "environment": environment or {},
        "dataset_info": dataset_info or {"dataset": "ICBHI_2017", "split_method": split},
        "efficiency": efficiency or {},
        "best_metrics": best_metrics,
        "ablation": ablation,
        "training_history": training_history or [],
    }
    if concept_metrics is not None:
        doc["concept_metrics"] = concept_metrics
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path
