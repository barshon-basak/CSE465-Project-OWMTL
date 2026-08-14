"""
Open-world separability of a learned representation.

This module implements Member A's selection criterion, and it is the reason the
backbone workstream is a measurement contribution rather than a bake-off.

The standard move is: train three backbones on the 4-class sound-event task,
pick the one with the best macro-F1, hand it to Member B. That is a saturated
benchmark answered 135 times over, and it silently assumes closed-set accuracy
predicts open-world usefulness.

What we actually need from the shared encoder is that unseen diseases land
somewhere distinguishable in its embedding space. So we measure that directly:

    Freeze a backbone trained ONLY on sound-event labels. It has never seen a
    disease label. Embed the known-disease patients and the 19 held-out
    unseen-disease patients. Ask how separable they are.

Everything here is label-free with respect to disease: no disease head is
trained, nothing is fitted on the unknown patients. The unknown group is used
for scoring only, exactly as Member B will use it later.

The claim this produces:

    The backbone that maximises sound-event macro-F1 is not the backbone that
    maximises open-world separability, so accuracy is the wrong selection
    criterion for the shared encoder in an open-world pipeline.

`compare_backbones` builds the table that either supports or refutes it. Both
outcomes are publishable; a null result here is a real finding about the
benchmark, not a failed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_by_patient(
    embeddings: np.ndarray,
    patient_ids: Sequence[int],
    *,
    method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool cycle-level embeddings into one vector per patient.

    Returns (X, pids) with rows ordered by ascending patient id. `method` is
    "mean" (default) or "median"; median is more robust to the handful of cycles
    whose annotation boundaries are wrong.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    pids = np.asarray(patient_ids)
    uniq = np.unique(pids)
    reducer = {"mean": np.mean, "median": np.median}[method]
    X = np.stack([reducer(emb[pids == p], axis=0) for p in uniq])
    return X, uniq


def _l2_normalise(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Unknown-detection scores (all label-free w.r.t. disease)
# ---------------------------------------------------------------------------


@dataclass
class FittedScorer:
    """A scorer fitted on known-disease TRAIN patients only."""

    kind: str
    projector: Optional[object]
    centroids: np.ndarray
    precision: Optional[np.ndarray]
    reference: Optional[np.ndarray]
    k: int

    def _project(self, X: np.ndarray) -> np.ndarray:
        return self.projector.transform(X) if self.projector is not None else X

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more likely to be an unseen disease."""
        Z = self._project(np.asarray(X, dtype=np.float64))
        if self.kind == "mahalanobis":
            d = np.stack(
                [
                    np.einsum("ij,jk,ik->i", Z - c, self.precision, Z - c)
                    for c in self.centroids
                ],
                axis=1,
            )
            return np.sqrt(np.maximum(d.min(axis=1), 0.0))
        if self.kind == "cosine":
            sims = _l2_normalise(Z) @ _l2_normalise(self.centroids).T
            return -sims.max(axis=1)
        if self.kind == "knn":
            d = np.linalg.norm(Z[:, None, :] - self.reference[None, :, :], axis=2)
            k = min(self.k, d.shape[1])
            return np.sort(d, axis=1)[:, :k].mean(axis=1)
        raise ValueError(f"unknown scorer kind {self.kind!r}")


def fit_scorer(
    X_train: np.ndarray,
    y_train: Sequence[str],
    *,
    kind: str = "mahalanobis",
    n_components: Optional[int] = 32,
    k: int = 5,
    seed: int = 42,
) -> FittedScorer:
    """Fit an unknown-detection scorer on known-disease training patients.

    `n_components` PCA-reduces first. This is not cosmetic: with ~50 training
    patients and a 768-dim AST embedding the class covariance is singular, and
    an unregularised Mahalanobis distance there is meaningless. PCA to 32 dims
    plus Ledoit-Wolf shrinkage makes it well-posed. Set to None to skip (fine
    for cycle-level fitting, where n is in the thousands).
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)

    projector = None
    if n_components is not None and n_components < X.shape[1]:
        n_components = min(n_components, X.shape[0] - 1)
        projector = PCA(n_components=n_components, random_state=seed).fit(X)
        X = projector.transform(X)

    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])

    precision = None
    if kind == "mahalanobis":
        # Shared within-class covariance: centre each class, pool, then shrink.
        centred = np.concatenate([X[y == c] - X[y == c].mean(axis=0) for c in classes])
        precision = LedoitWolf(store_precision=True, assume_centered=True).fit(
            centred
        ).precision_

    return FittedScorer(
        kind=kind,
        projector=projector,
        centroids=centroids,
        precision=precision,
        reference=X if kind == "knn" else None,
        k=k,
    )


# ---------------------------------------------------------------------------
# Geometry of the representation (no scorer, no thresholds)
# ---------------------------------------------------------------------------


def fisher_ratio(X: np.ndarray, y: Sequence[str]) -> float:
    """Trace-ratio Fisher discriminant: tr(S_b) / tr(S_w).

    How linearly separable the *known* disease classes already are in a
    representation that was never trained on disease labels. High values mean
    the encoder has picked up disease-relevant structure for free.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    mu = X.mean(axis=0)
    s_b = 0.0
    s_w = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        s_b += len(Xc) * float(np.sum((mc - mu) ** 2))
        s_w += float(np.sum((Xc - mc) ** 2))
    return s_b / s_w if s_w > 0 else float("inf")


def silhouette_known_vs_unknown(X_known: np.ndarray, X_unknown: np.ndarray) -> float:
    """Silhouette score treating known vs unseen-disease as two clusters.

    Threshold-free and scale-free, so it is comparable across backbones with
    different embedding dimensions. Range [-1, 1]; > 0 means the unseen group
    occupies its own region.
    """
    from sklearn.metrics import silhouette_score

    X = np.concatenate([np.asarray(X_known), np.asarray(X_unknown)])
    labels = np.concatenate(
        [np.zeros(len(X_known), dtype=int), np.ones(len(X_unknown), dtype=int)]
    )
    if len(np.unique(labels)) < 2 or len(X) < 3:
        return float("nan")
    return float(silhouette_score(X, labels))


# ---------------------------------------------------------------------------
# The headline measurement
# ---------------------------------------------------------------------------


@dataclass
class SeparabilityResult:
    n_known_test: int
    n_unknown: int
    auroc: Dict[str, float]
    auprc: Dict[str, float]
    fisher_ratio_known: float
    silhouette: float
    scorer_kinds: List[str]

    def to_dict(self) -> Dict:
        return {
            "n_known_test_patients": self.n_known_test,
            "n_unknown_patients": self.n_unknown,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "fisher_ratio_known_diseases": self.fisher_ratio_known,
            "silhouette_known_vs_unknown": self.silhouette,
            "scorers": self.scorer_kinds,
            "note": (
                "Computed from a backbone trained on sound-event labels only. No "
                "disease head was trained and nothing was fitted on the unknown "
                "patients. n_unknown=19 is small: report AUROC as a point estimate "
                "with no bootstrap CI, per the project's statistical protocol."
            ),
        }


def open_world_separability(
    X_train: np.ndarray,
    y_train: Sequence[str],
    X_known_test: np.ndarray,
    X_unknown: np.ndarray,
    *,
    kinds: Sequence[str] = ("mahalanobis", "cosine", "knn"),
    n_components: Optional[int] = 32,
    seed: int = 42,
) -> SeparabilityResult:
    """Measure how separable unseen diseases are in a frozen representation.

    `X_train`/`y_train` are known-disease *training* patients (the only thing any
    scorer is fitted on). `X_known_test` and `X_unknown` are scored, never fitted.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    auroc: Dict[str, float] = {}
    auprc: Dict[str, float] = {}
    labels = np.concatenate(
        [np.zeros(len(X_known_test), dtype=int), np.ones(len(X_unknown), dtype=int)]
    )
    for kind in kinds:
        scorer = fit_scorer(
            X_train, y_train, kind=kind, n_components=n_components, seed=seed
        )
        scores = np.concatenate([scorer.score(X_known_test), scorer.score(X_unknown)])
        auroc[kind] = float(roc_auc_score(labels, scores))
        auprc[kind] = float(average_precision_score(labels, scores))

    return SeparabilityResult(
        n_known_test=len(X_known_test),
        n_unknown=len(X_unknown),
        auroc=auroc,
        auprc=auprc,
        fisher_ratio_known=fisher_ratio(X_train, y_train),
        silhouette=silhouette_known_vs_unknown(
            np.concatenate([X_train, X_known_test]), X_unknown
        ),
        scorer_kinds=list(kinds),
    )


def compare_backbones(
    results: Dict[str, Dict],
    *,
    accuracy_key: str = "f1_macro",
    separability_key: str = "mahalanobis",
) -> Dict:
    """Build the M12 selection table and test the criterion claim.

    `results` maps backbone name -> {"f1_macro": float, "separability":
    SeparabilityResult-or-dict}. Returns the ranking under each criterion, the
    rank correlation between them, and whether the two criteria disagree on the
    winner — which is the claim itself.
    """
    from scipy.stats import kendalltau

    rows = []
    for name, r in results.items():
        sep = r["separability"]
        sep = sep.to_dict() if hasattr(sep, "to_dict") else sep
        rows.append(
            {
                "backbone": name,
                "sound_event_f1_macro": float(r[accuracy_key]),
                "separability_auroc": float(sep["auroc"][separability_key]),
                "fisher_ratio": float(sep["fisher_ratio_known_diseases"]),
                "silhouette": float(sep["silhouette_known_vs_unknown"]),
            }
        )

    by_acc = sorted(rows, key=lambda r: -r["sound_event_f1_macro"])
    by_sep = sorted(rows, key=lambda r: -r["separability_auroc"])

    tau = float("nan")
    p = float("nan")
    if len(rows) >= 3:
        acc = [r["sound_event_f1_macro"] for r in rows]
        sep = [r["separability_auroc"] for r in rows]
        stat = kendalltau(acc, sep)
        tau, p = float(stat.statistic), float(stat.pvalue)

    return {
        "table": rows,
        "ranking_by_sound_event_f1": [r["backbone"] for r in by_acc],
        "ranking_by_open_world_separability": [r["backbone"] for r in by_sep],
        "winner_by_f1": by_acc[0]["backbone"],
        "winner_by_separability": by_sep[0]["backbone"],
        "criteria_disagree": by_acc[0]["backbone"] != by_sep[0]["backbone"],
        "kendall_tau": tau,
        "kendall_p": p,
        "interpretation": (
            "criteria_disagree=True supports the claim that closed-set accuracy is "
            "the wrong backbone-selection criterion for an open-world pipeline. "
            "False is still a reportable result: it says the two criteria coincide "
            "on ICBHI, which nobody has checked. With only 3-4 backbones the "
            "Kendall tau is descriptive, not a significance test — do not report a "
            "p-value as evidence."
        ),
    }
