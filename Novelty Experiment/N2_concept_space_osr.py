"""
N2 - Concept-Space Open-Set Recognition
======================================

FACULTY ASK: do open-set detectors work better in the interpretable CONCEPT space than in
the opaque embedding space?

WHY THIS IS A NEW EXPERIMENT (audit finding):
  M29 ran MSP / entropy / energy / Mahalanobis on the frozen M12 *embeddings* and got
  Energy AUROC 0.6466 with a CI that spans chance at n=19 unknown patients. M38 (the
  large-N version) has a generator and a notebook but was NEVER RUN - no results JSON
  exists. The concept-space arm - the same detectors on the 14-d physics concept vector -
  is STEP_SEQUENCE.md step 06 and was never built. That arm is the actual novelty: an
  open-set detector whose score is readable ("this patient is unusual in wheeze frequency
  and crackle rate") rather than a distance in a 768-d space nobody can inspect.

WHAT IT DOES:
  Known = COPD / Healthy / URTI (104 patients).  Unknown = Bronchiectasis / Pneumonia /
  Bronchiolitis (19 patients), used for EVALUATION ONLY - never fitted, at any stage.
  Detectors are fitted on the known-TRAIN patients only, then scored at PATIENT level
  (cycles mean-aggregated) on known-test + unknown.

  Spaces compared:  concepts (14-d) | embeddings (768-d) | concat
  Detectors:        MSP | entropy | energy | Mahalanobis | kNN
  Reported:         AUROC + stratified bootstrap CI, AUPR, unknown recall at the 95%
                    known-TPR operating point, and a PAIRED bootstrap on the
                    concept-minus-embedding difference.

WHY THE PAIRED TEST MATTERS:
  Two independent CIs that overlap do not mean the difference is null, and two that do not
  overlap are not a test. At n=19 the only defensible comparison is the paired difference,
  and its CI will almost certainly span zero - in which case the honest sentence is "the
  concept space is not shown to differ from the embedding space", which is still a
  publishable calibration of what this corpus can support.

RUNNING (CPU, ~1 min, no audio needed):
    python N2_concept_space_osr.py
    python N2_concept_space_osr.py --features /path/to/M2_features.npy   # adds the
                                                                        # embedding arm
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N2_concept_space_osr"
DETECTORS = ["msp", "entropy", "energy", "mahalanobis", "knn"]


def fit_detectors(Z_fit, y_fit, seed=0, k=5):
    """Fit every post-hoc OOD scorer on the KNOWN-FIT patients only.

    Returns score_fn(Z) -> {detector: score}, oriented so HIGHER = more likely unknown.
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.special import logsumexp, softmax

    sc = StandardScaler().fit(Z_fit)
    A = sc.transform(Z_fit)

    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(A, y_fit)

    # Shared shrunk covariance. On a 768-d space with ~60 patients the empirical
    # covariance is singular and Mahalanobis silently returns garbage; Ledoit-Wolf is the
    # standard fix and costs nothing on 14-d.
    mus = np.stack([A[y_fit == c].mean(0) for c in np.unique(y_fit)])
    lw = LedoitWolf().fit(A - mus[y_fit])
    P = lw.precision_

    def score(Z):
        B = sc.transform(Z)
        logits = clf.decision_function(B)
        if logits.ndim == 1:
            logits = np.c_[-logits, logits]
        prob = softmax(logits, axis=1)
        d = np.stack([np.einsum("ij,jk,ik->i", B - m, P, B - m) for m in mus], axis=1)
        dist = np.sqrt(np.maximum(np.min(d, axis=1), 0))
        nn = np.sort(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1) ** 0.5, axis=1)
        return {
            "msp": -prob.max(axis=1),                       # low max-prob -> unknown
            "entropy": -(prob * np.log(prob + 1e-12)).sum(1),
            "energy": -logsumexp(logits, axis=1),           # low logsumexp -> unknown
            "mahalanobis": dist,
            "knn": nn[:, min(k, nn.shape[1] - 1)],
        }

    return score


def evaluate_space(name, Z_fit, y_fit, Z_eval, y_unknown, seed=0, n_boot=2000):
    """Score one representation space and return per-detector metrics with CIs."""
    from sklearn.metrics import average_precision_score

    score = fit_detectors(Z_fit, y_fit, seed=seed)
    scores = score(Z_eval)
    out, raw = {}, {}
    for det in DETECTORS:
        s = scores[det]
        ci = C.bootstrap_ci(C.auroc, y_unknown, s, n_boot=n_boot, seed=seed,
                            stratify=y_unknown)
        op = C.recall_at_fixed_tpr(y_unknown, s, 0.95)
        out[det] = {
            "auroc": ci["point"], "auroc_ci95": ci["ci95"],
            "aupr": round(float(average_precision_score(y_unknown, s)), 4),
            "unknown_prevalence": round(float(np.mean(y_unknown)), 4),
            "unknown_recall_at_95_known_tpr": op["unknown_recall"],
            "verdict": C.honest_verdict(ci["ci95"][0], ci["ci95"][1], 0.5),
        }
        raw[det] = s
        print(f"    {det:12s} AUROC {ci['point']:.4f} CI{ci['ci95']}  "
              f"R@95 {op['unknown_recall']}  {out[det]['verdict']}")
    return out, raw


def make_figure(results, raw_scores, y_unknown):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spaces = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ypos, labels = [], []
    i = 0
    for sp in spaces:
        for det in DETECTORS:
            r = results[sp][det]
            lo, hi = r["auroc_ci95"]
            ax.plot([lo, hi], [i, i], color="0.4", lw=1.5)
            ax.plot(r["auroc"], i, "o", ms=6,
                    color="tab:green" if lo > 0.5 else "tab:red")
            labels.append(f"{sp}:{det}")
            ypos.append(i)
            i += 1
        i += 0.6
    ax.axvline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("AUROC (unknown-disease detection), 95% CI")
    ax.set_title("N2 - open-set detection by space\ngreen = CI excludes chance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    best_sp = spaces[0]
    s = raw_scores[best_sp]["energy"]
    ax.hist(s[y_unknown == 0], bins=20, alpha=0.6, label="known", density=True)
    ax.hist(s[y_unknown == 1], bins=20, alpha=0.6, label="unknown", density=True)
    ax.set_title(f"Energy score distribution - {best_sp}\n"
                 f"n_known={int((y_unknown == 0).sum())}  n_unknown={int(y_unknown.sum())}")
    ax.set_xlabel("energy score (higher = more unknown)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    C.save_figure(fig, "N2_concept_space_osr.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    C.banner("N2 - Concept-Space Open-Set Recognition",
             "patient level | unknown group is evaluation-only, never fitted")
    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))

    known = C.patient_frame(d, features=F, diseases=C.KNOWN_DISEASES)
    unknown = C.patient_frame(d, features=F, diseases=C.UNKNOWN_DISEASES)
    fit_m = known["split"] == "train"
    ev_m = ~fit_m

    print(f"  known patients   : {len(known['pid'])} "
          f"(fit {int(fit_m.sum())} / eval {int(ev_m.sum())})")
    print(f"  unknown patients : {len(unknown['pid'])}  <- evaluation only")
    if int(fit_m.sum()) < 10 or int(ev_m.sum()) < 5:
        return C.blocked(EXP_ID, "too few known patients on one side of the split",
                         ["check concepts_all.npz split column"])

    y_unknown = np.r_[np.zeros(int(ev_m.sum()), int), np.ones(len(unknown["pid"]), int)]
    eval_ids = np.r_[known["pid"][ev_m], unknown["pid"]]

    spaces = {"concept_14d": (known["C"][fit_m], np.r_[known["C"][ev_m], unknown["C"]])}
    if F is not None:
        spaces["embedding_768d"] = (known["F"][fit_m],
                                    np.r_[known["F"][ev_m], unknown["F"]])
        spaces["concat"] = (np.c_[known["C"][fit_m], known["F"][fit_m]],
                            np.c_[np.r_[known["C"][ev_m], unknown["C"]],
                                  np.r_[known["F"][ev_m], unknown["F"]]])
    else:
        print("\n  [note] M2_features.npy not found - running the concept arm only.\n"
              "         Pass --features to add the embedding comparison (the paired test\n"
              "         against the embedding space is the headline; without it this run\n"
              "         establishes the concept arm alone).")

    results, raw = {}, {}
    for name, (Zf, Ze) in spaces.items():
        print(f"\n  -- {name} ({Zf.shape[1]}-d)")
        results[name], raw[name] = evaluate_space(name, Zf, known["y"][fit_m], Ze,
                                                  y_unknown, args.seed, args.n_boot)

    # Paired comparison: the only defensible way to claim one space beats another.
    paired = {}
    if "embedding_768d" in raw:
        for det in DETECTORS:
            paired[det] = C.paired_bootstrap_diff(
                C.auroc, y_unknown, raw["concept_14d"][det],
                raw["embedding_768d"][det], n_boot=args.n_boot, seed=args.seed)
            paired[det]["reading"] = (
                "concept space differs from embedding space"
                if paired[det]["ci95"][0] > 0 or paired[det]["ci95"][1] < 0
                else "NOT shown to differ (CI spans zero)")
        print("\n  -- paired concept - embedding (AUROC difference)")
        for det, r in paired.items():
            print(f"    {det:12s} d={r['diff']:+.4f} CI{r['ci95']} p={r['p_two_sided']} "
                  f"| {r['reading']}")

    # Raw score dump so a later DeLong / McNemar against M29 is possible.
    try:
        from owmtl.eval_utils import dump_scores
        os.makedirs(C.RESULTS_DIR, exist_ok=True)
        for sp in raw:
            dump_scores(os.path.join(C.RESULTS_DIR, f"{EXP_ID}_{sp}_energy"),
                        eval_ids, raw[sp]["energy"], y_unknown,
                        extra={"space": sp, "detector": "energy"})
    except Exception as ex:
        print(f"  [warn] score dump skipped: {ex}")

    try:
        make_figure(results, raw, y_unknown)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    best = max(((sp, det, results[sp][det]["auroc"]) for sp in results for det in DETECTORS),
               key=lambda t: t[2])
    doc = {
        "experiment": EXP_ID, "status": "OK",
        "compares_against": {"M29": "embedding-space baseline, Energy AUROC 0.6466",
                             "M38": "never run - no results JSON committed"},
        "dataset_info": {"dataset": "ICBHI_2017", "evaluation_level": "patient",
                         "split_method": "patient_independent_official_60_40",
                         "known_classes": C.KNOWN_DISEASES,
                         "unknown_classes": C.UNKNOWN_DISEASES,
                         "n_known_fit": int(fit_m.sum()),
                         "n_known_eval": int(ev_m.sum()),
                         "n_unknown": int(len(unknown["pid"])),
                         "unknown_group_never_fitted": True},
        "spaces": results, "paired_concept_minus_embedding": paired,
        "best": {"space": best[0], "detector": best[1], "auroc": best[2]},
        "power_note": (f"n_unknown={len(unknown['pid'])}. Every CI at this sample size is "
                       "wide enough to span chance; report 'not shown to beat chance', "
                       "never 'beats'."),
    }
    C.save_result(EXP_ID, doc)
    print(f"\n  best: {best[0]} / {best[1]} AUROC {best[2]:.4f} "
          f"({results[best[0]][best[1]]['verdict']})")
    return doc


if __name__ == "__main__":
    main()
