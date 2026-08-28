"""
N7 - Clinician Concept Intervention
===================================

FACULTY ASK: let a clinician correct a concept and show the diagnosis responds.

WHY THIS IS A NEW EXPERIMENT (audit finding):
  The mechanism is DONE - `owmtl/intervention.py` + notebook 04 produced
  `intervention_report.json`: a per-concept sensitivity ranking (top
  `inspiratory_energy_fraction` 0.4376; `fine_crackle_ratio` exactly 0.0 - a dead concept)
  and two directed edits:
      crackle_presence -> COPD :  mean_delta_p +0.1398, 89.4% of patients increased
      wheeze_presence  -> COPD :  mean_delta_p -0.0008, 43.3% increased
  So one of the two directed interventions moves the model the WRONG WAY, which the
  committed report records but does not flag.

  Two things are missing:
    1. THE INTERVENTION CURVE. The canonical evidence for a concept bottleneck is
       accuracy as a function of HOW MANY concepts the clinician has corrected. A
       sensitivity ranking says the model reacts; only the curve says the reaction is
       CORRECT. Without it, "clinician intervention" is a demo, not a result.
    2. A REAL CLINICIAN. `CLINICIAN_LABELING_PACK.md` and `build_listening_pack.py` exist,
       but no returned labels are committed anywhere in the repo. Everything to date is
       simulated, and the write-up must say so.

WHAT THIS ADDS:
  * The intervention curve, done properly. Start from the "clinician has not listened yet"
    state - every concept replaced by its population mean - then restore k true concept
    values and re-predict. Two orders are compared:
        by sensitivity : correct the concepts the model reacts to most, first
        random         : the null ordering, averaged over repeats
    A bottleneck that is genuinely concept-driven improves faster under the sensitivity
    order than under random. If the two curves coincide, the ranking is decorative.
  * A RANDOM-CONCEPT control curve, so "accuracy rises with k" cannot be an artefact of
    simply moving away from the population mean.
  * An automatic dead-concept report (sensitivity <= 1e-6 - correcting it can never change
    a diagnosis, so it must not be shown to a clinician as actionable).
  * A real-clinician hook: if `clinician_corrections.csv` is present (columns
    patient,concept,value) those corrections are applied and the diagnosis change is
    reported per patient. Absent -> the run is explicitly labelled SIMULATED.

RUNNING (CPU, ~1 min):
    python N7_clinician_intervention.py
    python N7_clinician_intervention.py --features /path/to/M2_features.npy --mode leaky
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from N6_physics_bottleneck import BottleneckModel  # noqa: E402

EXP_ID = "N7_clinician_intervention"
CLINICIAN_CSV = os.path.join(C.HERE, "clinician_corrections.csv")


def sensitivity(model, Fm, Cm, names, low_pct=10, high_pct=90):
    """Per-concept sensitivity: mean total-variation change in the predicted class
    distribution when a concept is swept from its 10th to its 90th percentile.

    Same definition as owmtl.intervention.intervention_sensitivity, reimplemented against
    a predict_proba callable so it works on the sklearn engine too.
    """
    lo = np.percentile(Cm, low_pct, axis=0)
    hi = np.percentile(Cm, high_pct, axis=0)
    rows = []
    for j, nm in enumerate(names):
        Clo, Chi = Cm.copy(), Cm.copy()
        Clo[:, j] = lo[j]
        Chi[:, j] = hi[j]
        plo = model.predict_proba(Fm, Clo)
        phi = model.predict_proba(Fm, Chi)
        rows.append({"concept": nm,
                     "sensitivity": round(float(np.abs(phi - plo).sum(1).mean() / 2), 4),
                     "delta_by_class": [round(float(x), 4) for x in (phi - plo).mean(0)]})
    rows.sort(key=lambda r: -r["sensitivity"])
    return rows


def intervention_curve(model, Fm, C_true, C_prior, y, order, repeats=1, seed=0):
    """Accuracy after restoring the first k concepts of `order` (0..n_concepts).

    `C_prior` is the pre-intervention state - the population mean, i.e. "no auscultation
    findings recorded yet". Restoring a concept simulates a clinician supplying its true
    value.
    """
    rng = np.random.default_rng(seed)
    n = C_true.shape[1]
    accs = []
    for k in range(n + 1):
        reps = []
        for r in range(repeats):
            idx = order(rng)[:k] if callable(order) else order[:k]
            Cm = C_prior.copy()
            if k:
                Cm[:, idx] = C_true[:, idx]
            reps.append(float((model.predict(Fm, Cm) == y).mean()))
        accs.append(round(float(np.mean(reps)), 4))
    return accs


def make_figure(curves, n_concepts, base_acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    styles = {"by_sensitivity": ("o-", "tab:blue"), "random_order": ("s--", "tab:orange"),
              "shuffled_concepts_control": ("^:", "tab:red")}
    for name, ys in curves.items():
        fmt, col = styles.get(name, ("o-", "0.5"))
        ax.plot(range(n_concepts + 1), ys, fmt, color=col, label=name.replace("_", " "),
                ms=4)
    ax.axhline(base_acc, color="0.3", ls=":", label=f"all concepts true ({base_acc})")
    ax.set_xlabel("number of concepts corrected by the clinician (k)")
    ax.set_ylabel("disease accuracy on 43 held-out test patients")
    ax.set_title("N7 - concept intervention curve\n"
                 "flat = the bottleneck does not use the corrections")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.save_figure(fig, "N7_clinician_intervention.png")
    plt.close(fig)


def apply_clinician_csv(path, pids, names, C_test):
    """Apply real clinician corrections. Returns (C_corrected, n_applied, unmatched)."""
    idx_p = {p: i for i, p in enumerate(pids)}
    idx_c = {n: j for j, n in enumerate(names)}
    Cc = C_test.copy()
    applied, unmatched = 0, []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p, c = str(row.get("patient", "")).strip(), str(row.get("concept", "")).strip()
            if p in idx_p and c in idx_c:
                Cc[idx_p[p], idx_c[c]] = float(row["value"])
                applied += 1
            else:
                unmatched.append(f"{p}/{c}")
    return Cc, applied, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--mode", choices=["independent", "sequential", "leaky"],
                    default="independent")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C.banner("N7 - Clinician Concept Intervention",
             "sensitivity + the missing intervention CURVE + a real-clinician hook")
    d = C.load_concepts(args.concepts)
    F = C.load_features(args.features, n_expected=len(d["X"]))
    pf = C.patient_frame(d, features=F, diseases=C.KNOWN_DISEASES)
    names = d["concept_names"]
    tr, te = pf["split"] == "train", pf["split"] == "test"

    mode = args.mode
    if mode != "independent" and pf["F"] is None:
        print(f"  [note] mode '{mode}' needs M2_features.npy - falling back to "
              "'independent'.")
        mode = "independent"
    Ftr = None if pf["F"] is None else pf["F"][tr]
    Fte = None if pf["F"] is None else pf["F"][te]

    model = BottleneckModel(mode, seed=args.seed).fit(Ftr, pf["C"][tr], pf["y"][tr])
    base_acc = round(float((model.predict(Fte, pf["C"][te]) == pf["y"][te]).mean()), 4)
    print(f"  mode={mode} | test patients {int(te.sum())} | "
          f"accuracy with all concepts true: {base_acc}")

    print("\n  -- per-concept sensitivity")
    sens = sensitivity(model, Fte, pf["C"][te], names)
    for r in sens:
        print(f"    {r['concept']:26s} {r['sensitivity']:.4f}")
    dead = [r["concept"] for r in sens if r["sensitivity"] <= 1e-6]
    if dead:
        print(f"    DEAD (correcting these can never change a diagnosis): {dead}")

    # Pre-intervention state: population mean over the TRAIN patients. Using the test mean
    # would leak the test distribution into the starting point.
    prior = np.tile(pf["C"][tr].mean(0), (int(te.sum()), 1)).astype(np.float32)
    sens_order = [names.index(r["concept"]) for r in sens]

    print("\n  -- intervention curve")
    curves = {
        "by_sensitivity": intervention_curve(model, Fte, pf["C"][te], prior,
                                             pf["y"][te], sens_order),
        "random_order": intervention_curve(model, Fte, pf["C"][te], prior, pf["y"][te],
                                           lambda rng: rng.permutation(len(names)),
                                           repeats=args.repeats, seed=args.seed),
    }
    # Control: restore SHUFFLED concept values. Rules out "accuracy rises simply because we
    # moved off the population mean".
    rng = np.random.default_rng(args.seed)
    C_sh = pf["C"][te][rng.permutation(int(te.sum()))]
    curves["shuffled_concepts_control"] = intervention_curve(
        model, Fte, C_sh, prior, pf["y"][te],
        lambda r: r.permutation(len(names)), repeats=args.repeats, seed=args.seed)

    for nm, ys in curves.items():
        print(f"    {nm:28s} k=0 {ys[0]:.4f} -> k={len(names)} {ys[-1]:.4f} "
              f"(gain {ys[-1] - ys[0]:+.4f})")

    gain_real = curves["by_sensitivity"][-1] - curves["by_sensitivity"][0]
    gain_ctrl = curves["shuffled_concepts_control"][-1] - curves["shuffled_concepts_control"][0]
    auc_sens = float(np.mean(curves["by_sensitivity"]))
    auc_rand = float(np.mean(curves["random_order"]))
    # Order matters: a NEGATIVE gain is the dominant fact and must be reported as such.
    # Beating the shuffled control while still losing accuracy is not a working
    # intervention, and phrasing it as one is exactly the inflation the pivot forbids.
    if gain_real <= 0:
        verdict = ("correcting concepts does NOT improve the diagnosis - accuracy is flat "
                   f"or falls ({gain_real:+.4f} from k=0 to k={len(names)}). The "
                   "bottleneck is not usefully interventionable on this corpus; the "
                   "clinician demo cannot be presented as a working mechanism.")
    elif gain_real <= gain_ctrl:
        verdict = ("correcting REAL concepts is no better than correcting shuffled ones - "
                   "any apparent gain is an artefact of moving off the population mean")
    elif auc_sens > auc_rand + 0.01:
        verdict = "intervention works AND the sensitivity ranking is useful"
    else:
        verdict = ("intervention improves the diagnosis, but the sensitivity ranking is no "
                   "better than a random order - report the ranking as descriptive, not "
                   "actionable")
    print(f"\n  verdict: {verdict}")

    print("\n  -- directed interventions (clinician asserts a finding)")
    directed = {}
    for cname, target in (("crackle_presence", "COPD"), ("wheeze_presence", "COPD"),
                          ("rhonchi_presence", "COPD")):
        if cname not in names:
            continue
        j, t = names.index(cname), C.KNOWN_DISEASES.index(target)
        hi = float(np.percentile(pf["C"][te][:, j], 90))
        base = model.predict_proba(Fte, pf["C"][te])[:, t]
        Cm = pf["C"][te].copy()
        Cm[:, j] = hi
        new = model.predict_proba(Fte, Cm)[:, t]
        dlt = new - base
        directed[f"{cname}->{target}"] = {
            "asserted_value": round(hi, 4), "mean_delta_p": round(float(dlt.mean()), 4),
            "frac_increased": round(float((dlt > 0).mean()), 3),
            "direction_ok": bool(dlt.mean() > 0)}
        print(f"    {cname:20s} -> {target:8s} dp {dlt.mean():+.4f} "
              f"({(dlt > 0).mean():.1%} increased)"
              f"{'' if dlt.mean() > 0 else '   <-- MOVES THE WRONG WAY'}")

    clinician = {"source": "SIMULATED (DSP concept values used as the clinician oracle)",
                 "real_clinician_labels_present": False,
                 "how_to_supply": f"drop a CSV at {CLINICIAN_CSV} with columns "
                                  "patient,concept,value"}
    if os.path.isfile(CLINICIAN_CSV):
        Cc, applied, unmatched = apply_clinician_csv(CLINICIAN_CSV, pf["pid"][te], names,
                                                     pf["C"][te])
        acc_after = float((model.predict(Fte, Cc) == pf["y"][te]).mean())
        clinician = {"source": "REAL clinician corrections", "file": CLINICIAN_CSV,
                     "real_clinician_labels_present": True,
                     "n_corrections_applied": applied, "unmatched_rows": unmatched[:20],
                     "accuracy_before": base_acc, "accuracy_after": round(acc_after, 4),
                     "delta": round(acc_after - base_acc, 4)}
        print(f"\n  REAL clinician corrections applied: {applied} | "
              f"accuracy {base_acc} -> {acc_after:.4f}")
    else:
        print(f"\n  [SIMULATED] no clinician file at {CLINICIAN_CSV} - the curve above uses "
              "the DSP\n              concept values as the oracle. Say so in the write-up.")

    try:
        make_figure(curves, len(names), base_acc)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    doc = {
        "experiment": EXP_ID, "status": "OK", "mode": mode,
        "hardens": {"source": "owmtl_concept_engine/.../04_concept_intervention/"
                              "intervention_report.json",
                    "committed_top_sensitivity": {"inspiratory_energy_fraction": 0.4376},
                    "committed_dead_concept": "fine_crackle_ratio (0.0)",
                    "committed_wrong_way": "wheeze_presence->COPD mean_delta_p -0.0008",
                    "what_was_missing": "the intervention curve and any real clinician"},
        "dataset_info": {"dataset": "ICBHI_2017", "evaluation_level": "patient",
                         "split_method": "patient_independent_official_60_40",
                         "n_test_patients": int(te.sum()), "classes": C.KNOWN_DISEASES},
        "baseline_accuracy_all_concepts_true": base_acc,
        "sensitivity": sens, "dead_concepts": dead,
        "intervention_curve": {"k": list(range(len(names) + 1)), "curves": curves,
                               "mean_curve_by_sensitivity": round(auc_sens, 4),
                               "mean_curve_random_order": round(auc_rand, 4),
                               "gain_real": round(gain_real, 4),
                               "gain_shuffled_control": round(gain_ctrl, 4),
                               "verdict": verdict},
        "directed_interventions": directed,
        "clinician": clinician,
    }
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
