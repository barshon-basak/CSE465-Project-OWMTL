"""
N4 - Calibration-Aware Honest Operating Point
=============================================

FACULTY ASK: pick and defend a single honest operating point for the model.

WHY THIS IS A NEW EXPERIMENT (audit finding):
  The pieces exist; the deliverable never got assembled.
    * M11 does post-hoc calibration (ECE 0.446 -> 0.0878 with vector scaling) but on a
      60/20/20 split, and uncalibrated accuracy 0.3573 vs vector-scaled 0.6595 - a
      calibrator is not supposed to move accuracy by 30 points, so the underlying model
      was broken and the ECE improvement is not interpretable.
    * M14 v2 is the conformal wrapper: test AUROC 0.4809, empirical coverage 0.9545, and
      unknown detection 0.0% at 95% coverage - the "conformal paradox".
    * M29 states an operating point, but only for the open-set score.
  A grep for `risk_coverage` / `selective` / `abstain` / `deferral` across the repo returns
  nothing. STEP_SEQUENCE.md step 07 was never built.

THE POINT OF THIS SCRIPT:
  "Honest operating point" is not calibration, and it is not conformal coverage. It is the
  answer to: at the sensitivity a clinician would actually demand, what does this model
  cost, and how often must it abstain? Four things, in one place:

    1. CALIBRATION - ECE/MCE/Brier/NLL before and after temperature scaling, fitted on a
       held-out calibration split, plus the reliability diagram.
    2. SELECTIVE PREDICTION - the risk-coverage curve and AURC. This is the number that
       says whether abstaining actually buys accuracy, and no model in the repo has one.
    3. THE CLINICAL POINT - threshold chosen on the CALIBRATION split to hit a target
       sensitivity (default 0.90), then reported on TEST: achieved Se, Sp, PPV, NPV,
       referral rate, and expected cost under a stated FN:FP ratio (default 10:1). The
       threshold is never chosen on test - that is the error the corrected protocol exists
       to prevent.
    4. THE CONFORMAL HONESTY NOTE - split-conformal coverage IS guaranteed by
       construction, so reporting it as a result is circular. What is not guaranteed is
       whether unknown-disease patients get EMPTY prediction sets. That number, not
       coverage, is what M14 should have reported, and it is computed here.

RUNNING (CPU, ~20 s, no audio needed):
    python N4_honest_operating_point.py
    python N4_honest_operating_point.py --target_sensitivity 0.95 --fn_fp_ratio 20
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N4_honest_operating_point"


def temperature_scale(logits_cal, y_cal):
    """1-D temperature fitted by minimising NLL on the CALIBRATION split.

    Golden-section over log T - no torch, no gradient loop, and it cannot diverge.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import log_softmax

    def nll(logT):
        lp = log_softmax(logits_cal / np.exp(logT), axis=1)
        return -float(lp[np.arange(len(y_cal)), y_cal].mean())

    r = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(r.x))


def calibration_block(logits, y, T=1.0, n_bins=10):
    from scipy.special import softmax
    p = softmax(logits / T, axis=1)
    e, m = C.ece_mce(p, y, n_bins)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1
    return {
        "temperature": round(float(T), 4),
        "accuracy": round(float((p.argmax(1) == y).mean()), 4),
        "ece": round(e, 4), "mce": round(m, 4),
        "brier": round(float(((p - onehot) ** 2).sum(1).mean()), 4),
        "nll": round(float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), 4),
    }, p


def clinical_point(p_abnormal_cal, y_abn_cal, p_abnormal_test, y_abn_test,
                   target_se=0.90, fn_fp=10.0):
    """Threshold selected on CALIBRATION to reach `target_se`, then applied to TEST.

    Reports what a clinician actually asks for: at the sensitivity I require, how many
    healthy people do you refer, and what does a mistake cost?
    """
    order = np.sort(p_abnormal_cal[y_abn_cal == 1])
    if order.size == 0:
        return {"error": "no abnormal patients in the calibration split"}
    # lowest threshold whose calibration sensitivity is >= target
    idx = int(np.floor((1.0 - target_se) * order.size))
    thr = float(order[min(idx, order.size - 1)])

    pred = (p_abnormal_test >= thr).astype(int)
    tp = int(((pred == 1) & (y_abn_test == 1)).sum())
    fp = int(((pred == 1) & (y_abn_test == 0)).sum())
    fn = int(((pred == 0) & (y_abn_test == 1)).sum())
    tn = int(((pred == 0) & (y_abn_test == 0)).sum())
    se = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    se_ci = C.bootstrap_ci(lambda a, b: float(((b == 1) & (a == 1)).sum() / max((a == 1).sum(), 1)),
                           y_abn_test, pred, n_boot=2000, seed=0, stratify=y_abn_test)
    return {
        "threshold_selected_on": "calibration split (never on test)",
        "target_sensitivity": target_se, "threshold": round(thr, 6),
        "test_sensitivity": round(se, 4), "test_sensitivity_ci95": se_ci["ci95"],
        "test_specificity": round(sp, 4),
        "ppv": round(tp / max(tp + fp, 1), 4), "npv": round(tn / max(tn + fn, 1), 4),
        "referral_rate": round(float(pred.mean()), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "fn_fp_cost_ratio": fn_fp,
        "expected_cost_per_patient": round((fn_fp * fn + fp) / max(len(pred), 1), 4),
        "target_met_on_test": bool(se >= target_se),
        "note": ("Sensitivity on test may fall below target - that gap is the honest "
                 "result. A threshold re-tuned on test to hit the target would be the "
                 "exact error the corrected protocol forbids."),
    }


def conformal_block(p_cal, y_cal, p_test, p_unknown, alpha=0.10):
    """Split-conformal prediction sets, reported honestly.

    Coverage is guaranteed by construction, so it is not evidence of anything. The
    informative quantity is how often an UNKNOWN-disease patient receives an EMPTY
    prediction set - that is the actual novelty-detection claim M14 conflated with
    coverage.
    """
    s_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s_cal)
    q = float(np.quantile(s_cal, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)))
    sets_test = (1.0 - p_test) <= q
    out = {"alpha": alpha, "target_coverage": round(1 - alpha, 3),
           "conformal_quantile": round(q, 6),
           "mean_set_size_test": round(float(sets_test.sum(1).mean()), 4),
           "empty_set_rate_test_known": round(float((sets_test.sum(1) == 0).mean()), 4)}
    if p_unknown is not None and len(p_unknown):
        su = (1.0 - p_unknown) <= q
        out["empty_set_rate_unknown"] = round(float((su.sum(1) == 0).mean()), 4)
        out["mean_set_size_unknown"] = round(float(su.sum(1).mean()), 4)
        out["reading"] = (
            "Empty-set rate on unknown patients is the detection number. If it is ~0 the "
            "wrapper detects nothing, no matter how good the coverage looks - this is the "
            "M14 conformal paradox stated correctly.")
    return out


def make_figure(cal_before, cal_after, probs_test, y_test, cov, risk, aurc, cp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    conf = probs_test.max(1)
    acc = (probs_test.argmax(1) == y_test).astype(float)
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            xs.append(conf[m].mean())
            ys.append(acc[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(xs, ys, "o-", color="tab:blue", label="after temp. scaling")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability\nECE {cal_before['ece']:.3f} -> {cal_after['ece']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(cov, risk, color="tab:red")
    ax.set_xlabel("coverage (fraction of patients answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title(f"Risk-coverage\nAURC {aurc:.4f} (lower is better)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    if "error" in cp:
        ax.text(0.02, 0.5, cp["error"], fontsize=10)
    else:
        lines = [
            f"Clinical operating point (target Se = {cp['target_sensitivity']})",
            "threshold chosen on CALIBRATION, applied to TEST",
            "",
            f"  test sensitivity  {cp['test_sensitivity']:.4f}  CI{cp['test_sensitivity_ci95']}",
            f"  test specificity  {cp['test_specificity']:.4f}",
            f"  PPV / NPV         {cp['ppv']:.4f} / {cp['npv']:.4f}",
            f"  referral rate     {cp['referral_rate']:.4f}",
            f"  expected cost     {cp['expected_cost_per_patient']:.4f} /patient"
            f"  (FN:FP = {cp['fn_fp_cost_ratio']:g}:1)",
            f"  target met        {cp['target_met_on_test']}",
        ]
        ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    C.save_figure(fig, "N4_honest_operating_point.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--target_sensitivity", type=float, default=0.90)
    ap.add_argument("--fn_fp_ratio", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C.banner("N4 - Calibration-Aware Honest Operating Point",
             "calibration + risk-coverage + clinical threshold + honest conformal")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))
    known = C.patient_frame(d, features=F, diseases=C.KNOWN_DISEASES)
    unknown = C.patient_frame(d, features=F, diseases=C.UNKNOWN_DISEASES)

    Z = known["F"] if known["F"] is not None else known["C"]
    Zu = unknown["F"] if unknown["F"] is not None else unknown["C"]
    space = "m2_embedding" if known["F"] is not None else "concept_14d"

    tr = known["split"] == "train"
    te = ~tr
    # Split TRAIN into fit / calibration. Calibration must be disjoint from both fitting
    # and test, or the temperature and the conformal quantile are both fitted on data the
    # model has already seen.
    rng = np.random.default_rng(args.seed)
    idx_tr = np.flatnonzero(tr)
    cal_mask = np.zeros(len(known["y"]), bool)
    for c in np.unique(known["y"][tr]):
        ic = idx_tr[known["y"][idx_tr] == c]
        cal_mask[rng.choice(ic, max(1, int(round(0.3 * len(ic)))), replace=False)] = True
    fit = tr & ~cal_mask
    cal = cal_mask

    print(f"  space {space} | fit {int(fit.sum())} | calibration {int(cal.sum())} | "
          f"test {int(te.sum())} patients")
    if int(cal.sum()) < 6:
        return C.blocked(EXP_ID, "calibration split too small to fit a temperature",
                         ["more known patients, or reduce the calibration fraction"])

    sc = StandardScaler().fit(Z[fit])
    clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(Z[fit]), known["y"][fit])

    def logits(M):
        L = clf.decision_function(sc.transform(M))
        return np.c_[-L, L] if L.ndim == 1 else L

    L_cal, L_te, L_un = logits(Z[cal]), logits(Z[te]), logits(Zu)
    y_cal, y_te = known["y"][cal], known["y"][te]

    before, _ = calibration_block(L_te, y_te, T=1.0)
    T = temperature_scale(L_cal, y_cal)
    after, p_te = calibration_block(L_te, y_te, T=T)
    _, p_cal = calibration_block(L_cal, y_cal, T=T)
    _, p_un = calibration_block(L_un, np.zeros(len(L_un), int), T=T)
    print(f"  temperature {T:.4f} | ECE {before['ece']:.4f} -> {after['ece']:.4f} | "
          f"accuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f}")
    sanity = abs(after["accuracy"] - before["accuracy"]) < 1e-9

    cov, risk, aurc = C.risk_coverage((p_te.argmax(1) == y_te), p_te.max(1))
    sel = {f"coverage_{int(q * 100)}": round(float(1 - risk[max(int(q * len(risk)) - 1, 0)]), 4)
           for q in (1.0, 0.9, 0.8, 0.7, 0.5)}
    print(f"  AURC {aurc:.4f} | selective accuracy " +
          " ".join(f"{k.split('_')[1]}%:{v:.3f}" for k, v in sel.items()))

    healthy = C.KNOWN_DISEASES.index("Healthy")
    cp = clinical_point(1.0 - p_cal[:, healthy], (y_cal != healthy).astype(int),
                        1.0 - p_te[:, healthy], (y_te != healthy).astype(int),
                        args.target_sensitivity, args.fn_fp_ratio)
    if "error" not in cp:
        print(f"  clinical point: Se {cp['test_sensitivity']:.4f} "
              f"Sp {cp['test_specificity']:.4f} referral {cp['referral_rate']:.4f} "
              f"cost {cp['expected_cost_per_patient']:.4f} "
              f"| target met: {cp['target_met_on_test']}")

    conf = conformal_block(p_cal, y_cal, p_te, p_un, args.alpha)
    print(f"  conformal: empty-set rate known {conf['empty_set_rate_test_known']} vs "
          f"unknown {conf.get('empty_set_rate_unknown')}  <- the real detection number")

    try:
        make_figure(before, after, p_te, y_te, cov, risk, aurc, cp)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    doc = {
        "experiment": EXP_ID, "status": "OK", "space": space,
        "supersedes": {"M11": "calibration, now on the official split with an accuracy-"
                              "invariance check", "M14": "conformal, now reporting the "
                              "empty-set rate rather than the guaranteed coverage"},
        "dataset_info": {"dataset": "ICBHI_2017", "evaluation_level": "patient",
                         "split_method": "patient_independent_official_60_40 "
                                         "(train split further into fit/calibration)",
                         "n_fit": int(fit.sum()), "n_calibration": int(cal.sum()),
                         "n_test": int(te.sum()), "n_unknown": int(len(Zu))},
        "calibration": {"before": before, "after_temperature_scaling": after,
                        "temperature": round(T, 4),
                        "accuracy_invariant": sanity,
                        "sanity_note": ("Temperature scaling is monotone, so accuracy MUST "
                                        "be unchanged. M11 reported accuracy moving 0.3573 "
                                        "-> 0.6595 under calibration, which means the "
                                        "comparison there was not like-for-like.")},
        "selective_prediction": {"aurc": round(aurc, 4), "selective_accuracy": sel,
                                 "note": "AURC lower is better; no other model in the "
                                         "repo reports one."},
        "clinical_operating_point": cp,
        "conformal": conf,
    }
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
