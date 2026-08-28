"""
N3 - Prototypical Few-Shot Disease Head
=======================================

FACULTY ASK: a prototypical few-shot head for disease diagnosis.

WHY THIS IS A NEW EXPERIMENT (audit finding):
  `Barshon's/M13` already implements the prototypical head and it is real work: v4,
  nearest-class-mean over frozen M2/M12 embeddings, patient-level, 0.7209 accuracy /
  0.6061 macro-F1 (v3 scored URTI recall 0.026 on synthetic data). Three things stop it
  from being a FEW-SHOT claim:

    1. NO SHOT SWEEP. It runs one fixed 5-shot configuration. A few-shot method is a claim
       about the shape of the accuracy-vs-k curve; a single k is just a classifier.
    2. NO BASELINE. Nothing shows that prototypes beat a plain linear head on the same k
       support patients. Prototypical networks are supposed to win when k is small - if
       they do not, that is the finding, and if nobody checks, the novelty is unearned.
    3. WRONG HEADLINE METRIC. It reports `icbhi_score` 0.7137. The official ICBHI score is
       (Se+Sp)/2 over the FOUR-CLASS sound-event task with Normal vs pooled abnormal. It
       is undefined for a 3-class disease task, and the number in M13 is the inflated
       macro variant the 2026-08-16 audit was written to stop. This script refuses to emit
       it and reports macro-F1 + per-class recall + CIs instead.

WHAT IT DOES:
  For k in {1, 2, 5, 10, 20} support patients per class, drawn from the TRAIN side of the
  official split only, over R episodes:
      proto   : nearest-class-mean prototype (cosine on standardised features)
      linear  : logistic regression fitted on the identical k*3 support patients
  Both are evaluated on ALL 43 held-out TEST patients every episode, so the two curves are
  paired and the difference is testable. `full` (all train patients) is the upper bound.

  k is capped at the rarest TRAIN class count and the cap is recorded - URTI has ~14
  patients in total, so a "20-shot" claim would be silently fabricated otherwise.

RUNNING (CPU, ~30 s, no audio needed):
    python N3_prototypical_fewshot.py
    python N3_prototypical_fewshot.py --features /path/to/M2_features.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N3_prototypical_fewshot"
SHOTS = [1, 2, 5, 10, 20]


def _standardise(train, test):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def _proto_predict(sup_X, sup_y, qry_X, n_classes):
    """Nearest-class-mean on L2-normalised vectors == cosine to the prototype."""
    P = np.stack([sup_X[sup_y == c].mean(0) for c in range(n_classes)])
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    Q = qry_X / (np.linalg.norm(qry_X, axis=1, keepdims=True) + 1e-8)
    return (Q @ P.T).argmax(1)


def _linear_predict(sup_X, sup_y, qry_X):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(sup_y)) < 2:
        return np.full(len(qry_X), sup_y[0])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sup_X, sup_y)
    return clf.predict(qry_X)


def episode_sweep(Xtr, ytr, Xte, yte, n_classes, shots, episodes=200, seed=0):
    """Paired proto-vs-linear sweep over k. Returns per-k metric distributions."""
    from sklearn.metrics import f1_score, recall_score

    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(ytr == c) for c in range(n_classes)]
    min_train = min(len(ix) for ix in by_class)
    Xtr_s, Xte_s = _standardise(Xtr, Xte)

    out = {}
    for k in shots:
        if k > min_train:
            out[str(k)] = {"skipped": True,
                           "reason": f"k={k} exceeds the rarest TRAIN class ({min_train} "
                                     "patients) - claiming it would fabricate support data"}
            continue
        rec = {"proto": {"f1": [], "acc": []}, "linear": {"f1": [], "acc": []},
               "per_class_recall_proto": []}
        for e in range(episodes):
            sup = np.concatenate([rng.choice(ix, k, replace=False) for ix in by_class])
            sy = ytr[sup]
            for name, pred in (("proto", _proto_predict(Xtr_s[sup], sy, Xte_s, n_classes)),
                               ("linear", _linear_predict(Xtr_s[sup], sy, Xte_s))):
                rec[name]["f1"].append(f1_score(yte, pred, average="macro", zero_division=0))
                rec[name]["acc"].append(float(np.mean(pred == yte)))
            rec["per_class_recall_proto"].append(
                recall_score(yte, _proto_predict(Xtr_s[sup], sy, Xte_s, n_classes),
                             average=None, labels=list(range(n_classes)), zero_division=0))

        d = {}
        for name in ("proto", "linear"):
            f1 = np.array(rec[name]["f1"])
            ac = np.array(rec[name]["acc"])
            d[name] = {
                "f1_macro_mean": round(float(f1.mean()), 4),
                "f1_macro_ci95": [round(float(x), 4) for x in np.percentile(f1, [2.5, 97.5])],
                "accuracy_mean": round(float(ac.mean()), 4),
            }
        diff = np.array(rec["proto"]["f1"]) - np.array(rec["linear"]["f1"])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        d["proto_minus_linear_f1"] = {
            "mean": round(float(diff.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "verdict": ("prototypical WINS" if lo > 0 else
                        "linear WINS" if hi < 0 else
                        "NOT shown to differ (CI spans zero)")}
        d["per_class_recall_proto_mean"] = [
            round(float(x), 4) for x in np.mean(rec["per_class_recall_proto"], axis=0)]
        d["episodes"] = episodes
        out[str(k)] = d
    out["_min_train_class"] = int(min_train)
    return out


def full_shot(Xtr, ytr, Xte, yte, n_classes, seed=0):
    """Upper bound: every train patient used as support."""
    from sklearn.metrics import f1_score, recall_score, confusion_matrix
    Xtr_s, Xte_s = _standardise(Xtr, Xte)
    pred = _proto_predict(Xtr_s, ytr, Xte_s, n_classes)
    linp = _linear_predict(Xtr_s, ytr, Xte_s)
    boot = C.bootstrap_ci(lambda a, b: float(f1_score(a, b, average="macro", zero_division=0)),
                          yte, pred, n_boot=2000, seed=seed, stratify=yte)
    return {
        "proto_f1_macro": boot["point"], "proto_f1_macro_ci95": boot["ci95"],
        "proto_accuracy": round(float(np.mean(pred == yte)), 4),
        "linear_f1_macro": round(float(f1_score(yte, linp, average="macro",
                                                zero_division=0)), 4),
        "per_class_recall": [round(float(x), 4) for x in
                             recall_score(yte, pred, average=None,
                                          labels=list(range(n_classes)), zero_division=0)],
        "confusion_matrix_raw": confusion_matrix(
            yte, pred, labels=list(range(n_classes))).tolist(),
    }


def make_figure(results, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5), squeeze=False)
    for ax, (space, res) in zip(axes[0], results.items()):
        ks, pm, pl, ll = [], [], [], []
        for k in SHOTS:
            r = res["sweep"].get(str(k))
            if not r or r.get("skipped"):
                continue
            ks.append(k)
            pm.append(r["proto"]["f1_macro_mean"])
            pl.append(r["proto"]["f1_macro_ci95"])
            ll.append(r["linear"]["f1_macro_mean"])
        if not ks:
            continue
        pl = np.array(pl)
        ax.plot(ks, pm, "o-", label="prototypical", color="tab:blue")
        ax.fill_between(ks, pl[:, 0], pl[:, 1], alpha=0.2, color="tab:blue")
        ax.plot(ks, ll, "s--", label="linear head (same support)", color="tab:orange")
        ax.axhline(res["full"]["proto_f1_macro"], color="0.3", ls=":",
                   label=f"all-train proto ({res['full']['proto_f1_macro']})")
        ax.axhline(1.0 / len(classes), color="r", ls="--", lw=1, label="chance (macro-F1 floor)")
        ax.set_xlabel("support patients per class (k)")
        ax.set_ylabel("macro-F1 on 43 held-out test patients")
        ax.set_title(f"N3 - few-shot disease head\n{space}")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "N3_prototypical_fewshot.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C.banner("N3 - Prototypical Few-Shot Disease Head",
             "shot sweep + paired linear baseline | patient level | official 60/40")
    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))
    pf = C.patient_frame(d, features=F, diseases=C.KNOWN_DISEASES)
    tr, te = pf["split"] == "train", pf["split"] == "test"
    n_classes = len(C.KNOWN_DISEASES)

    print(f"  train patients {int(tr.sum())} | test patients {int(te.sum())}")
    print(f"  train class counts {np.bincount(pf['y'][tr], minlength=n_classes)} "
          f"({C.KNOWN_DISEASES})")

    spaces = {"concept_14d": pf["C"]}
    if pf["F"] is not None:
        spaces["m2_embedding_768d"] = pf["F"]
    else:
        print("  [note] M2_features.npy not found - concept space only. Pass --features to\n"
              "         reproduce M13's embedding arm under the shot sweep.")

    results = {}
    for name, Z in spaces.items():
        print(f"\n  -- {name} ({Z.shape[1]}-d)")
        sweep = episode_sweep(Z[tr], pf["y"][tr], Z[te], pf["y"][te], n_classes,
                              SHOTS, args.episodes, args.seed)
        full = full_shot(Z[tr], pf["y"][tr], Z[te], pf["y"][te], n_classes, args.seed)
        results[name] = {"sweep": sweep, "full": full}
        for k in SHOTS:
            r = sweep.get(str(k))
            if not r:
                continue
            if r.get("skipped"):
                print(f"    k={k:<3} SKIPPED - {r['reason']}")
                continue
            print(f"    k={k:<3} proto F1 {r['proto']['f1_macro_mean']:.4f} "
                  f"CI{r['proto']['f1_macro_ci95']} | linear "
                  f"{r['linear']['f1_macro_mean']:.4f} | "
                  f"{r['proto_minus_linear_f1']['verdict']}")
        print(f"    all-train proto F1 {full['proto_f1_macro']:.4f} "
              f"CI{full['proto_f1_macro_ci95']} (M13 reported 0.6061)")

    try:
        make_figure(results, C.KNOWN_DISEASES)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    doc = {
        "experiment": EXP_ID, "status": "OK",
        "supersedes": {"M13": "same head, now with a shot sweep, a paired linear baseline, "
                              "CIs, and without the undefined icbhi_score"},
        "metric_note": ("icbhi_score_official is (Se+Sp)/2 over the FOUR-CLASS sound-event "
                        "task and is UNDEFINED for this 3-class disease task. M13's "
                        "reported icbhi_score 0.7137 is the inflated macro variant; it is "
                        "deliberately not emitted here."),
        "dataset_info": {"dataset": "ICBHI_2017", "evaluation_level": "patient",
                         "split_method": "patient_independent_official_60_40",
                         "classes": C.KNOWN_DISEASES,
                         "n_train_patients": int(tr.sum()),
                         "n_test_patients": int(te.sum()),
                         "train_class_counts": np.bincount(
                             pf["y"][tr], minlength=n_classes).tolist()},
        "spaces": results,
    }
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
