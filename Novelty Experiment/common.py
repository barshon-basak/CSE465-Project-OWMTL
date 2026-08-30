"""
common.py - shared plumbing for the eight Novelty Experiments (N1..N8).

Design rule: this file adds NOTHING that already exists in the repo. It locates the
existing `owmtl` package (concept extractors, leakage estimator, bottleneck head,
intervention API, eval_utils) and re-uses it, plus the few things that were genuinely
missing everywhere: patient-level aggregation, a permutation null, a paired bootstrap on
a metric *difference*, risk-coverage, and one honest-phrasing helper.

Every experiment writes results/<id>.json so the audit tool
(`Asif's/audit/audit_project.py`) can read them like any other model.

CPU-only. Nothing in this module imports torch.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")


# --------------------------------------------------------------------------- repo wiring
def repo_root() -> str:
    """Walk up until the repo marker directory appears."""
    d = HERE
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "owmtl_concept_engine")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(HERE)


def add_owmtl_to_path():
    """Put the existing owmtl package on sys.path. Also honours a Kaggle/Colab copy so the
    same script runs unchanged inside a notebook."""
    cands = [
        os.path.join(repo_root(), "owmtl_concept_engine"),
        "/kaggle/input/datasets/barshonbasak/owmtl-package",
        "/kaggle/working/owmtl-package",
        "/content/owmtl_concept_engine",
    ]
    for c in cands:
        if os.path.isdir(os.path.join(c, "owmtl")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


OWMTL_PATH = add_owmtl_to_path()


# --------------------------------------------------------------------------- data loading
CONCEPT_NPZ_CANDIDATES = [
    os.path.join(repo_root(), "owmtl_concept_engine", "notebooks",
                 "01_concept_extraction_and_validation", "concepts_all.npz"),
    "/kaggle/input/datasets/barshonbasak/concepts-all/concepts_all.npz",
    "/content/concepts_all.npz",
]

FEATURE_NPY_CANDIDATES = [
    os.path.join(HERE, "M2_features.npy"),
    os.path.join(repo_root(), "M2_features.npy"),
    "/kaggle/input/datasets/barshonbasak/m2-features/M2_features.npy",
    "/content/M2_features.npy",
]

# Project-wide disease groups. owmtl.icbhi_data is the single source of truth; these are a
# fallback so this module still imports on a machine with no owmtl checkout.
KNOWN_DISEASES = ["COPD", "Healthy", "URTI"]
UNKNOWN_DISEASES = ["Bronchiectasis", "Pneumonia", "Bronchiolitis"]
SOUND_CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


def _first_existing(cands, extra=None):
    for c in ([extra] if extra else []) + list(cands):
        if c and os.path.isfile(c):
            return c
    return None


def load_concepts(path=None) -> dict:
    """Load concepts_all.npz -> dict of aligned arrays.

    Keys: X (N,14), concept_names, patient, device, split, crackle, wheeze, sound_label,
    diagnosis. This one 6898-row file is the backbone of N2/N5/N6/N7/N8 - all of them run
    CPU-only from it with no audio.
    """
    p = _first_existing(CONCEPT_NPZ_CANDIDATES, path)
    if p is None:
        raise FileNotFoundError(
            "concepts_all.npz not found. Expected at "
            f"{CONCEPT_NPZ_CANDIDATES[0]!r} or a Kaggle/Colab copy. "
            "Do NOT substitute random data - regenerate it with notebook 01.")
    z = np.load(p, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["X"] = d["X"].astype(np.float32)
    d["concept_names"] = [str(s) for s in d["concept_names"]]
    d["_path"] = p
    return d


def load_features(path=None, n_expected=None):
    """Frozen M2 encoder features (N, 768) aligned to the cycle index, or None.

    M2_features.npy is NOT committed to the repo - it lives as a Kaggle dataset. Every
    script that can degrade without it does so loudly rather than silently substituting
    something else.
    """
    p = _first_existing(FEATURE_NPY_CANDIDATES, path)
    if p is None:
        return None
    F = np.load(p).astype(np.float32)
    if n_expected is not None and len(F) != n_expected:
        raise ValueError(f"M2 features ({len(F)}) are not aligned to the cycle index "
                         f"({n_expected}). Re-export with owmtl.m2_features.export_features.")
    return F


def patient_frame(d: dict, features=None, diseases=None, split=None) -> dict:
    """Aggregate cycles -> patients (mean per column).

    Patient is the unit of the disease task and of every open-set claim in this project.
    Cycle-level rows inflate n by ~60x and make every CI computed on them wrong.

    Returns dict with pid, y (index into `diseases`), diagnosis, C (concepts), F, split.
    """
    diseases = diseases or KNOWN_DISEASES
    keep = np.isin(d["diagnosis"], diseases)
    if split is not None:
        keep = keep & (d["split"] == split)
    pid_all = d["patient"][keep]
    pids = sorted(set(pid_all.tolist()))
    lab = {x: i for i, x in enumerate(diseases)}

    Xk = d["X"][keep]
    Fk = None if features is None else features[keep]
    dgk = d["diagnosis"][keep]
    spk = d["split"][keep]

    C = np.zeros((len(pids), d["X"].shape[1]), dtype=np.float32)
    F = None if features is None else np.zeros((len(pids), features.shape[1]), np.float32)
    y, diag, spl = [], [], []
    for i, p in enumerate(pids):
        m = pid_all == p
        C[i] = np.nanmean(Xk[m], axis=0)
        if F is not None:
            F[i] = np.nanmean(Fk[m], axis=0)
        diag.append(str(dgk[m][0]))
        y.append(lab[diag[-1]])
        spl.append(str(spk[m][0]))
    return {"pid": np.array(pids), "y": np.array(y), "diagnosis": np.array(diag),
            "C": np.nan_to_num(C), "F": F, "split": np.array(spl)}


# --------------------------------------------------------------------------- statistics
def auroc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _trapz(y, x):
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def bootstrap_ci(metric_fn, *arrays, n_boot=2000, seed=0, alpha=0.05, stratify=None):
    """Percentile bootstrap CI.

    `stratify` (e.g. the binary label) keeps both classes present in every resample.
    Without it, a resample of the 19-unknown-patient group can draw zero unknowns, the
    AUROC becomes NaN, those draws vanish, and the interval is biased.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    groups = None
    if stratify is not None:
        stratify = np.asarray(stratify)
        groups = [np.flatnonzero(stratify == v) for v in np.unique(stratify)]
    vals = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        try:
            v = metric_fn(*[a[idx] for a in arrays])
            if v == v:
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")], "n_boot": 0}
    point = metric_fn(*arrays)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_boot": len(vals)}


def paired_bootstrap_diff(metric_fn, y, scores_a, scores_b, n_boot=2000, seed=0):
    """CI on metric(a) - metric(b) using the SAME resample for both.

    This is the only correct way to say "concept space beats embedding space". Two
    independent CIs that overlap do not mean the difference is null, and two that do not
    overlap are not a test.
    """
    y = np.asarray(y)
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == v) for v in np.unique(y)]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        try:
            v = metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])
            if v == v:
                diffs.append(v)
        except Exception:
            pass
    if not diffs:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")],
                "p_two_sided": float("nan")}
    point = metric_fn(y, a) - metric_fn(y, b)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"diff": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided": round(min(p, 1.0), 4), "n_boot": int(diffs.size)}


def permutation_p(metric_fn, y, X, n_perm=1000, seed=0, groups=None):
    """Label-permutation null.

    `groups` shuffles WITHIN patient so that patient identity is not itself the thing
    being detected.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    observed = metric_fn(y, X)
    null = []
    for _ in range(n_perm):
        if groups is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for g in np.unique(groups):
                m = np.flatnonzero(groups == g)
                yp[m] = rng.permutation(y[m])
        try:
            v = metric_fn(yp, X)
            if v == v:
                null.append(v)
        except Exception:
            pass
    null = np.asarray(null)
    if null.size == 0:
        return {"observed": float(observed), "p": float("nan")}
    return {"observed": round(float(observed), 4),
            "p": round(float((np.sum(null >= observed) + 1) / (null.size + 1)), 4),
            "null_mean": round(float(null.mean()), 4),
            "null_ci95": [round(float(x), 4) for x in np.percentile(null, [2.5, 97.5])],
            "n_perm": int(null.size)}


def recall_at_fixed_tpr(y_unknown, scores, target_known_tpr=0.95):
    """Threshold chosen to retain `target_known_tpr` of KNOWN patients; report how many
    unknowns that threshold catches.

    The clinically meaningful operating point - you cannot flag a fifth of healthy
    patients as "unknown disease" and stay usable.
    """
    y = np.asarray(y_unknown).astype(bool)
    s = np.asarray(scores, dtype=float)
    if (~y).sum() == 0 or y.sum() == 0:
        return {"threshold": float("nan"), "unknown_recall": float("nan")}
    thr = float(np.quantile(s[~y], target_known_tpr))
    flagged = s > thr
    return {"threshold": round(thr, 6),
            "unknown_recall": round(float(flagged[y].mean()), 4),
            "known_fpr": round(float(flagged[~y].mean()), 4),
            "target_known_tpr": target_known_tpr}


def honest_verdict(ci_lo, ci_hi, chance=0.5) -> str:
    """The project's phrasing rule, made mechanical.

    With n=19 unknown patients every CI crosses 0.5, so a point estimate above chance is
    NOT a result. Never write "X beats Y" off a CI that spans chance.
    """
    if ci_lo != ci_lo:
        return "NOT ESTIMABLE"
    if ci_lo > chance:
        return "ABOVE CHANCE (CI excludes chance)"
    if ci_hi < chance:
        return "BELOW CHANCE (CI excludes chance)"
    return "NOT SHOWN TO BEAT CHANCE (CI spans chance) - do not write 'beats'"


# --------------------------------------------------------------- selective classification
def risk_coverage(correct, confidence):
    """Risk-coverage curve for selective prediction.

    Sort by confidence; at each coverage level report the error rate among the retained
    (most confident) items. Returns (coverage, risk, aurc). Lower AURC is better; this is
    the number that says whether abstention actually buys anything.
    """
    correct = np.asarray(correct).astype(float)
    conf = np.asarray(confidence, dtype=float)
    order = np.argsort(-conf)
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    coverage = k / len(c)
    risk = 1.0 - np.cumsum(c) / k
    return coverage, risk, _trapz(risk, coverage)


def ece_mce(probs, y_true, n_bins=10):
    """Expected and maximum calibration error (equal-width bins on max-prob)."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    mce = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        gap = abs(float(acc[m].mean()) - float(conf[m].mean()))
        e += float(m.mean()) * gap
        mce = max(mce, gap)
    return float(e), float(mce)


# --------------------------------------------------------------------------- output
def env_block() -> dict:
    return {"platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def save_result(exp_id: str, doc: dict) -> str:
    """Write results/<exp_id>.json with an environment stamp."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    doc = dict(doc)
    doc.setdefault("environment", {}).update(env_block())
    path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    print(f"[saved] {path}")
    return path


def save_figure(fig, name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
    return path


def banner(title: str, sub: str = ""):
    print("\n" + "=" * 78)
    print(title)
    if sub:
        print(sub)
    print("=" * 78)


def blocked(exp_id: str, reason: str, needs) -> dict:
    """Uniform "this experiment cannot run here" record.

    Written to results/ so a blocked experiment is still auditable and nobody has to
    rediscover why it did not run.
    """
    doc = {"experiment": exp_id, "status": "BLOCKED", "reason": reason,
           "needs": list(needs)}
    save_result(exp_id + "_BLOCKED", doc)
    print(f"\n[BLOCKED] {exp_id}: {reason}")
    for n in needs:
        print(f"          needs: {n}")
    return doc
