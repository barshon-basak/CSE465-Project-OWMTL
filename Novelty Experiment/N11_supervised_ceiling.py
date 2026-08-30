"""
N11 - The supervised ceiling: what CAN be validated against ICBHI cycle labels?
==============================================================================

WHY THIS EXISTS:
  Gate G2 failed twice and N10 came back UNDECIDED, which leaves the project's central
  claim resting on an ambiguity it could not resolve:

      (a) our DSP extractors are weak, or
      (b) AUROC 0.65 against ICBHI cycle labels is not reachable by any method.

  DECISION_2026-08-16_PIVOT.md states (a) is "true and demonstrated twice" and (b) is
  "unknown, and this experiment cannot decide it". This experiment decides it - not with a
  second clinician (none is available), but by asking a strictly easier question:

      How well does a model that is ALLOWED TO SEE THE LABELS do on the same task?

  A supervised probe fitted on the train patients and scored on the test patients is an
  upper bound on any unsupervised detector. If even that cannot clear 0.65, the binding
  constraint is the reference standard, not our DSP - and the claim stops depending on
  anyone accepting that our extractors were competently written.

WHY THIS IS NOT A THIRD GATE RUN:
  The gate asked "do OUR extractors reproduce the labels" and is closed at two reads. This
  asks "what is the best anyone could do here", which is a different question with a
  different estimator, and nothing here feeds back into extractor design. No extractor is
  tuned, revised, or re-thresholded. Report it as a ceiling estimate, never as a gate pass.

PRE-REGISTERED READINGS (fixed 2026-08-30, BEFORE the script was run):
  Let A* = the best supervised test AUROC over the three arms below.
    A* >= 0.80 : NO CEILING. These labels support a strong detector; our extractors are
                 simply weak and the paper says exactly that. The label-reliability
                 argument then rests on the published human benchmarks alone.
    0.65 <= A* < 0.80 : PARTIAL. The labels support a usable detector but the task is hard;
                 report the gap between 0.65 and A* as the headroom our DSP left on the
                 table.
    A* < 0.65  : CEILING CONFIRMED. The gate threshold is not reachable on this reference
                 standard even with label supervision, a foundation-model representation
                 and a task-trained encoder. G2's failure is then a property of the
                 benchmark, not only of our code.
  Writing the "no ceiling" branch down in advance is what makes the other two believable.

ARMS (all three fit on TRAIN patients only, scored on TEST patients, no tuning on test):
  1. ast_frozen  - 768-d frozen AudioSet AST embedding. The clean arm: this network has
                   never seen ICBHI or a respiratory label, so nothing about the test
                   patients can have leaked into its representation.
  2. m2_encoder  - 768-d frozen M2 embedding. STRONGER but not clean: M2 was trained on
                   these very crackle/wheeze labels, so treat it as an optimistic bound.
  3. dsp_concepts - the 14 physics concepts as a feature vector. The best LINEAR use of our
                   own extractors, as opposed to G2's single-score threshold. The gap
                   between this and arm 1/2 is how much our concept vocabulary throws away.

  Arm 3 answers a question the gate could not: G2 scored ONE concept against one label. If
  the 14-vector does much better, the concepts carry signal that the scalar score wasted.

INPUTS (all already on disk, CPU-only, no audio, ~30 s):
    concepts_all.npz                 labels, patient ids, split, the 14 concepts
    embeddings/ast_frozen.npy        arm 1   (optional; arm skipped with a note if absent)
    M2_features.npy                  arm 2   (optional; arm skipped with a note if absent)

RUNNING:
    python N11_supervised_ceiling.py
    python N11_supervised_ceiling.py --selftest   # validates the estimator, writes nothing
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N11_supervised_ceiling"
GATE = 0.65
SEED = 20260830


def _fit_probe(Xtr, ytr, Xte, seed=SEED):
    """Logistic regression with standardisation. Returns test scores.

    Deliberately linear: a linear probe is the standard "is the information present"
    estimator, and a stronger head would blur "the labels are learnable" into "our head is
    big". If a linear probe on a 768-d representation cannot clear the gate, no reviewer
    will believe a deeper one is the missing piece.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                           random_state=seed))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _patient_bootstrap_ci(y, s, patients, n_boot=2000, seed=SEED):
    """Resample PATIENTS, not cycles. Cycles within a patient are not independent."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    idx_by_pat = {p: np.flatnonzero(patients == p) for p in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_pat[p] for p in pick])
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        vals.append(C.auroc(yb, s[rows]))
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def run_arm(name, X, d, tr, te, label_key):
    y = d[label_key].astype(int)
    scores = _fit_probe(X[tr], y[tr], X[te])
    a = C.auroc(y[te], scores)
    ci = _patient_bootstrap_ci(y[te], scores, d["patient"][te])
    out = {"arm": name, "auroc": round(float(a), 4), "auroc_ci95_patient_bootstrap": ci,
           "n_train_cycles": int(tr.sum()), "n_test_cycles": int(te.sum()),
           "n_test_positive": int(y[te].sum()),
           "prevalence_test": round(float(y[te].mean()), 4),
           "clears_gate_0.65": bool(a >= GATE),
           "verdict_vs_chance": C.honest_verdict(ci[0], ci[1])}
    print(f"    {name:<14} AUROC {a:.4f}  CI {ci}  gate>={GATE}: "
          f"{'PASS' if a >= GATE else 'fail'}")
    return out


def selftest():
    """The estimator must recover a planted signal and find nothing in noise."""
    rng = np.random.default_rng(0)
    n, dim = 1200, 32
    patients = np.repeat(np.arange(60), 20).astype(str)
    y = rng.integers(0, 2, n)
    signal = rng.normal(size=(n, dim)) + y[:, None] * 1.2      # learnable
    noise = rng.normal(size=(n, dim))                          # not learnable
    tr = np.arange(n) < 800
    te = ~tr
    a_sig = C.auroc(y[te], _fit_probe(signal[tr], y[tr], signal[te]))
    a_noise = C.auroc(y[te], _fit_probe(noise[tr], y[tr], noise[te]))
    ci = _patient_bootstrap_ci(y[te], _fit_probe(noise[tr], y[tr], noise[te]), patients[te])
    print(f"  planted signal  AUROC {a_sig:.4f}   (expect > 0.9)")
    print(f"  pure noise      AUROC {a_noise:.4f}  CI {ci}  (expect ~0.5, CI spanning it)")
    ok = a_sig > 0.9 and abs(a_noise - 0.5) < 0.12 and ci[0] < 0.5 < ci[1]
    print("  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None, help="M2_features.npy (arm 2)")
    ap.add_argument("--ast", default=os.path.join(C.HERE, "embeddings", "ast_frozen.npy"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        C.banner("N11 selftest", "estimator validation only - writes nothing")
        return selftest()

    C.banner("N11 - supervised ceiling on ICBHI cycle labels",
             "how well can ANY model do against this reference standard?")

    d = C.load_concepts(args.concepts)
    n = len(d["X"])
    tr = d["split"] == "train"
    te = d["split"] == "test"
    print(f"  {n} cycles | train {tr.sum()} / test {te.sum()} | "
          f"{len(np.unique(d['patient'][te]))} test patients")

    arms = {"dsp_concepts": d["X"]}
    skipped = {}
    if os.path.exists(args.ast):
        A = np.load(args.ast).astype(np.float32)
        if len(A) != n:
            raise ValueError(f"ast_frozen.npy has {len(A)} rows, concepts have {n} - "
                             "not aligned; refusing to fit on mismatched rows.")
        arms["ast_frozen"] = A
    else:
        skipped["ast_frozen"] = f"not found at {args.ast}"

    F = C.load_features(args.features, n_expected=n)
    if F is not None:
        arms["m2_encoder"] = F
    else:
        skipped["m2_encoder"] = "M2_features.npy not found"

    doc = {
        "experiment": EXP_ID,
        "status": "OK",
        "question": "Gate G2 failed at 0.5506-0.5818. Is 0.65 reachable against these "
                    "labels by a model that is allowed to see them?",
        "why_not_a_third_gate_run": "The gate asked whether OUR extractors reproduce the "
                                    "labels and is closed at two test-split reads. This "
                                    "estimates the best achievable value with label "
                                    "supervision. No extractor is revised or re-thresholded "
                                    "here, and nothing feeds back into extractor design.",
        "prereg": {
            "fixed_utc": "2026-08-30",
            "before_running": True,
            "readings": {
                ">=0.80": "NO CEILING - the labels support a strong detector; our "
                          "extractors are weak and the paper says so",
                "0.65-0.80": "PARTIAL - usable but hard; report the headroom",
                "<0.65": "CEILING CONFIRMED - the gate threshold is unreachable on this "
                         "reference standard even with supervision"}},
        "gate_threshold": GATE,
        "g2_reference": {
            "pre_registered_gate_runs_M39": {"run1": {"crackle": 0.5506, "wheeze": 0.5729},
                                             "run2": {"crackle": 0.5580, "wheeze": 0.5340}},
            "concept_engine_14concept_validation": {"crackle": 0.5556, "wheeze": 0.5818},
            "note": "These are DIFFERENT extractor implementations (9-concept M39 gate vs "
                    "the later 14-concept engine), not two exports of one run. Do not mix "
                    "them in one table."},
        "split": "official_60_40_patient_independent (from concepts_all.npz)",
        "estimator": "logistic regression, standardised, class_weight=balanced, fit on "
                     "TRAIN patients, scored on TEST patients, patient-level bootstrap CI",
        "arms_skipped": skipped,
        "results": {},
    }

    for label_key in ("crackle", "wheeze"):
        print(f"\n  -- target: ICBHI '{label_key}' cycle label")
        doc["results"][label_key] = {k: run_arm(k, X, d, tr, te, label_key)
                                     for k, X in arms.items()}

    best = max((r["auroc"], f"{lk}/{arm}")
               for lk, per in doc["results"].items() for arm, r in per.items())
    a_star, best_name = best
    reading = (">=0.80" if a_star >= 0.80 else "0.65-0.80" if a_star >= GATE else "<0.65")
    doc["ceiling"] = {
        "best_supervised_auroc": a_star,
        "achieved_by": best_name,
        "prereg_reading_fired": reading,
        "verdict": doc["prereg"]["readings"][reading]}

    C.banner("CEILING", f"best supervised AUROC {a_star:.4f} ({best_name})")
    print(f"  pre-registered reading '{reading}' fires:\n    {doc['ceiling']['verdict']}")
    print("\n  Interpretation guard: this bounds CONCEPT-LEVEL VALIDATION AGAINST ICBHI "
          "CYCLE LABELS.\n  It is not a claim about respiratory ML overall, and not a "
          "claim that the labels are wrong.")

    C.save_result(EXP_ID, doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
