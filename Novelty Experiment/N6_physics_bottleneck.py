"""
N6 - Physics-Derived Acoustic Concept Bottleneck (hardened)
==========================================================

FACULTY ASK: a physics-derived acoustic concept bottleneck.

WHY THIS IS A NEW EXPERIMENT (audit finding):
  This is DONE and it is the project's most important result - it is the one that FAILED
  its own pre-registered gate, twice, which is why `DECISION_2026-08-16_PIVOT.md` exists.
  The committed run (`results_M13cbm_*.json`) reports, on the official 60/40 split:

      opaque      accuracy 0.7209   macro-F1 0.5973
      leaky       accuracy 0.6512   macro-F1 0.5457
      sequential  accuracy 0.5581   macro-F1 0.4441
      independent accuracy 0.4884   macro-F1 0.4542
      -> interpretability cost 0.2326 accuracy / 0.1432 macro-F1

  Monotone degradation as the bottleneck tightens. That IS the finding. Four things stop
  it from surviving review:
    1. ONE SEED. Four numbers from one initialisation on 43 test patients.
    2. NO RANDOM-CONCEPT CONTROL. If a bottleneck over SHUFFLED concepts scores the same
       as one over real physics concepts, the "physics" claim is empty. Nobody checked,
       and this is the first thing a reviewer will ask.
    3. NO PAIRED TEST between independent and opaque - the interpretability cost is quoted
       as a bare difference of two point estimates.
    4. EMPTY SCHEMA FIELDS. `config`, `efficiency` and `training_history` are all `{}`, so
       the runs fail the project's own §4 audit.

WHAT THIS ADDS: seeds, stratified bootstrap CIs, the shuffled-concept control, McNemar
between independent and opaque, and a filled schema.

ENGINE NOTE:
  `owmtl/bottleneck.py` is the torch reference implementation of these four modes. This
  script reimplements the SAME four modes on sklearn's MLP (same 64-unit hidden layer) so
  it runs on a laptop with no torch and no GPU. The mode semantics are identical:
      independent : y = f(c)                      <- the true bottleneck, concepts only
      sequential  : c_hat = g(features); y = f(c_hat)
      leaky       : y = f(c, features)            <- can bypass the bottleneck
      opaque      : y = f(features)               <- upper bound, ignores concepts

RUNNING (CPU, ~1 min; the concepts-only modes need no features at all):
    python N6_physics_bottleneck.py
    python N6_physics_bottleneck.py --features /path/to/M2_features.npy   # all four modes
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N6_physics_bottleneck"
MODES = ["independent", "sequential", "leaky", "opaque"]
HIDDEN = 64  # matches owmtl.bottleneck's default hidden=64


class BottleneckModel:
    """The four bottleneck modes, sklearn engine. Mirrors owmtl.bottleneck.ConceptBottleneck.

    Exposes predict_proba(features, concepts) so N7 can intervene on the concept vector of
    a fitted model without knowing which mode it is.
    """

    def __init__(self, mode: str, seed: int = 0, hidden: int = HIDDEN):
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode = mode
        self.seed = seed
        self.hidden = hidden

    def _new_clf(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=2000,
                                           random_state=self.seed, early_stopping=False))

    def _inputs(self, Fm, Cm):
        if self.mode == "independent":
            return Cm
        if self.mode == "opaque":
            return Fm
        if self.mode == "leaky":
            return np.c_[Cm, Fm]
        return self.concept_predictor.predict(Fm)   # sequential: predicted concepts

    def fit(self, Fm, Cm, y):
        if self.mode in ("opaque", "leaky", "sequential") and Fm is None:
            raise ValueError(f"mode '{self.mode}' needs encoder features")
        if self.mode == "sequential":
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.concept_predictor = make_pipeline(
                StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0))).fit(Fm, Cm)
        self.clf = self._new_clf().fit(self._inputs(Fm, Cm), y)
        return self

    def predict_proba(self, Fm, Cm):
        return self.clf.predict_proba(self._inputs(Fm, Cm))

    def predict(self, Fm, Cm):
        return self.predict_proba(Fm, Cm).argmax(1)

    @property
    def n_params(self):
        """Trainable parameter count, for the efficiency block the committed runs left empty."""
        mlp = self.clf.steps[-1][1]
        return int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))


def run_mode(mode, Ftr, Ctr, ytr, Fte, Cte, yte, seeds, n_classes, n_boot=2000):
    """Fit one mode over several seeds; return metrics with CIs and the median-seed preds."""
    import time
    from sklearn.metrics import f1_score, confusion_matrix

    accs, f1s, preds, params, times = [], [], [], [], []
    for s in seeds:
        t0 = time.time()
        m = BottleneckModel(mode, seed=s).fit(Ftr, Ctr, ytr)
        times.append(time.time() - t0)
        p = m.predict(Fte, Cte)
        preds.append(p)
        accs.append(float((p == yte).mean()))
        f1s.append(float(f1_score(yte, p, average="macro", zero_division=0)))
        params.append(m.n_params)

    # Report the median seed, not the best - picking the best seed is a silent test-set read.
    med = int(np.argsort(f1s)[len(f1s) // 2])
    pred = preds[med]
    f1_ci = C.bootstrap_ci(
        lambda a, b: float(f1_score(a, b, average="macro", zero_division=0)),
        yte, pred, n_boot=n_boot, seed=0, stratify=yte)
    acc_ci = C.bootstrap_ci(lambda a, b: float((a == b).mean()), yte, pred,
                            n_boot=n_boot, seed=0, stratify=yte)
    return {
        "mode": mode, "n_seeds": len(seeds),
        "accuracy_median_seed": acc_ci["point"], "accuracy_ci95": acc_ci["ci95"],
        "f1_macro_median_seed": f1_ci["point"], "f1_macro_ci95": f1_ci["ci95"],
        "accuracy_across_seeds": {"mean": round(float(np.mean(accs)), 4),
                                  "std": round(float(np.std(accs)), 4),
                                  "min": round(float(np.min(accs)), 4),
                                  "max": round(float(np.max(accs)), 4)},
        "f1_across_seeds": {"mean": round(float(np.mean(f1s)), 4),
                            "std": round(float(np.std(f1s)), 4)},
        "per_class_f1": [round(float(x), 4) for x in
                         f1_score(yte, pred, average=None, labels=list(range(n_classes)),
                                  zero_division=0)],
        "confusion_matrix_raw": confusion_matrix(
            yte, pred, labels=list(range(n_classes))).tolist(),
        "efficiency": {"trainable_params": int(np.median(params)),
                       "training_time_total_s": round(float(np.sum(times)), 3),
                       "training_time_per_seed_s": round(float(np.mean(times)), 3),
                       "device": "cpu"},
    }, pred


def make_figure(results, cost):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m for m in ["opaque", "leaky", "sequential", "independent",
                         "independent_shuffled_concepts"] if m in results]
    vals = [results[m]["f1_macro_median_seed"] for m in order]
    los = [results[m]["f1_macro_ci95"][0] for m in order]
    his = [results[m]["f1_macro_ci95"][1] for m in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    colors = ["0.5" if m != "independent_shuffled_concepts" else "tab:red" for m in order]
    ax.bar(x, vals, color=colors, alpha=0.8)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor="k", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in order], fontsize=9)
    ax.set_ylabel("macro-F1 (43 held-out test patients)")
    title = "N6 - accuracy/interpretability tradeoff of the physics bottleneck"
    if cost:
        title += f"\ninterpretability cost = {cost.get('f1_macro')} macro-F1"
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    C.save_figure(fig, "N6_physics_bottleneck.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    C.banner("N6 - Physics Concept Bottleneck (hardened)",
             "seeds + CIs + shuffled-concept control + McNemar + filled schema")
    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))
    pf = C.patient_frame(d, features=F, diseases=C.KNOWN_DISEASES)
    tr, te = pf["split"] == "train", pf["split"] == "test"
    n_classes = len(C.KNOWN_DISEASES)
    seeds = list(range(args.seeds))

    Ftr = None if pf["F"] is None else pf["F"][tr]
    Fte = None if pf["F"] is None else pf["F"][te]
    print(f"  train {int(tr.sum())} | test {int(te.sum())} patients | "
          f"concepts {pf['C'].shape[1]}-d | features "
          f"{'none' if pf['F'] is None else pf['F'].shape[1]}-d")

    results, preds, skipped = {}, {}, {}
    for mode in MODES:
        if mode != "independent" and pf["F"] is None:
            skipped[mode] = ("needs M2_features.npy (768-d encoder features aligned to the "
                             "cycle index); not committed to the repo")
            print(f"  -- {mode:12s} SKIPPED - {skipped[mode]}")
            continue
        r, p = run_mode(mode, Ftr, pf["C"][tr], pf["y"][tr], Fte, pf["C"][te], pf["y"][te],
                        seeds, n_classes, args.n_boot)
        results[mode], preds[mode] = r, p
        print(f"  -- {mode:12s} acc {r['accuracy_median_seed']:.4f} "
              f"F1 {r['f1_macro_median_seed']:.4f} CI{r['f1_macro_ci95']} "
              f"(seed sd {r['f1_across_seeds']['std']:.4f})")

    # THE CONTROL the committed run is missing: same bottleneck, concepts shuffled across
    # patients. If this matches `independent`, the physics carries nothing.
    rng = np.random.default_rng(0)
    Csh_tr = pf["C"][tr][rng.permutation(int(tr.sum()))]
    Csh_te = pf["C"][te][rng.permutation(int(te.sum()))]
    r, p = run_mode("independent", None, Csh_tr, pf["y"][tr], None, Csh_te, pf["y"][te],
                    seeds, n_classes, args.n_boot)
    r["mode"] = "independent_shuffled_concepts"
    results["independent_shuffled_concepts"] = r
    preds["independent_shuffled_concepts"] = p
    print(f"  -- {'shuffled ctrl':12s} acc {r['accuracy_median_seed']:.4f} "
          f"F1 {r['f1_macro_median_seed']:.4f} CI{r['f1_macro_ci95']}")

    # Test on BOTH metrics: with 64/26/14 class counts, accuracy is dominated by COPD and
    # can hide a large macro-F1 gap. Reporting only one of the two would be cherry-picking.
    from sklearn.metrics import f1_score as _f1
    ctrl = {}
    for mname, mfn in (("accuracy", lambda a, b: float((a == b).mean())),
                       ("f1_macro", lambda a, b: float(_f1(a, b, average="macro",
                                                           zero_division=0)))):
        r = C.paired_bootstrap_diff(mfn, pf["y"][te], preds["independent"],
                                    preds["independent_shuffled_concepts"],
                                    n_boot=args.n_boot, seed=0)
        r["reading"] = ("real physics concepts beat shuffled ones"
                        if r["ci95"][0] > 0 else
                        "real concepts are NOT shown to beat shuffled concepts on this "
                        "metric")
        ctrl[mname] = r
        print(f"\n  real vs shuffled concepts ({mname}): {r['diff']:+.4f} "
              f"CI{r['ci95']} p={r['p_two_sided']} -> {r['reading']}")
    ctrl["overall"] = ("the physics concepts carry the bottleneck"
                       if any(v["ci95"][0] > 0 for v in ctrl.values()
                              if isinstance(v, dict)) else
                       "on neither metric do real concepts beat shuffled ones - the "
                       "'physics' in 'physics-derived bottleneck' is not carrying the "
                       "result on this corpus")

    cost, mcn = None, None
    if "opaque" in results:
        cost = {"accuracy": round(results["opaque"]["accuracy_median_seed"]
                                  - results["independent"]["accuracy_median_seed"], 4),
                "f1_macro": round(results["opaque"]["f1_macro_median_seed"]
                                  - results["independent"]["f1_macro_median_seed"], 4),
                "committed_run_reported": {"accuracy": 0.2326, "f1_macro": 0.1432}}
        try:
            from owmtl.eval_utils import mcnemar
            stat, pv = mcnemar(pf["y"][te], preds["independent"], preds["opaque"])
            mcn = {"statistic": round(stat, 4), "p_exact_binomial": round(pv, 4),
                   "verdict": ("bottleneck cost is significant" if pv < 0.05 else
                               "bottleneck cost is NOT significant at n=43")}
            print(f"  McNemar independent vs opaque: p={pv:.4f} -> {mcn['verdict']}")
        except Exception as ex:
            print(f"  [warn] McNemar skipped: {ex}")

    try:
        make_figure(results, cost)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    doc = {
        "experiment": EXP_ID, "status": "OK" if not skipped else "PARTIAL",
        "hardens": {"source": "owmtl_concept_engine/.../002_concept_bottleneck_training/"
                              "results_M13cbm_*.json",
                    "committed": {"opaque": 0.5973, "leaky": 0.5457,
                                  "sequential": 0.4441, "independent": 0.4542},
                    "committed_note": "one seed, no CI, no control, empty schema fields"},
        "gate_context": {"gate": "G2", "outcome": "FAILED twice",
                         "crackle_auroc": 0.5556, "wheeze_auroc": 0.5818,
                         "threshold": "AUROC >= 0.65 with CI excluding chance",
                         "decision_record": "DECISION_2026-08-16_PIVOT.md"},
        "config": {"engine": "sklearn MLPClassifier", "hidden_layer_sizes": [HIDDEN],
                   "max_iter": 2000, "seeds": seeds, "n_boot": args.n_boot,
                   "reference_implementation": "owmtl/bottleneck.py (torch)"},
        "dataset_info": {"dataset": "ICBHI_2017", "evaluation_level": "patient",
                         "split_method": "patient_independent_official_60_40",
                         "classes": C.KNOWN_DISEASES,
                         "n_train_patients": int(tr.sum()),
                         "n_test_patients": int(te.sum())},
        "modes": results, "skipped_modes": skipped,
        "shuffled_concept_control": ctrl,
        "interpretability_cost": cost, "mcnemar_independent_vs_opaque": mcn,
        "power_note": f"n_test_patients={int(te.sum())}. Every CI here is wide; the "
                      "monotone ordering across modes is the claim, not any single number.",
    }
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
