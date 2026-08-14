#!/usr/bin/env python3
"""
Phase-0 gate: is ICBHI's disease label predictable from recording metadata alone?

Why this has to run before anything else
----------------------------------------
COPD is 64 of ICBHI's 126 patients, and COPD recordings are not uniformly
distributed over stethoscopes or chest locations. Published "99% diagnosis
accuracy on ICBHI" results are widely suspected of learning the recording
signature rather than the pathology.

That is fatal specifically for this project. The novelty claim is that
disagreement between the sound-event head and the disease head detects unseen
diseases. If the disease head has learned "AKGC417L at the posterior-left
position => COPD", then the disagreement signal is measuring device and
placement mismatch, and the headline result is an artifact.

This probe uses NO audio. It trains a classifier on the recording metadata only:
stethoscope, chest location, acquisition mode, cycle duration statistics. If that
reaches accuracy comparable to an audio model, the shortcut is real.

    python scripts/shortcut_probe.py --index splits/cycles_index_v1.csv \
                                     --split splits/split_v1.json

Reading the result
------------------
  * Disease accuracy near the majority-class rate  -> no metadata shortcut.
  * Disease accuracy well above it, approaching your audio model              ->
    STOP. The disease head must be constrained (device-balanced sampling,
    device-adversarial training, or per-device reporting) before M13/M15 are
    worth running.
  * Sound-event accuracy above chance is a milder warning: it means device
    identity carries some crackle/wheeze information, which should be disclosed.

Requires scikit-learn. No audio, no GPU; runs in seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from owmtl import icbhi  # noqa: E402

METADATA_FIELDS = ("device", "chest_location", "acquisition_mode")


def crosstab(rows: List[dict], row_key: str, col_key: str) -> str:
    """Contingency table — usually more persuasive than the classifier itself."""
    table: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        table[r[row_key]][r[col_key]] += 1
    cols = sorted({c for counts in table.values() for c in counts})
    width = max((len(r) for r in table), default=8) + 2
    lines = [" " * width + "".join(f"{c:>14}" for c in cols) + f"{'total':>10}"]
    for rk in sorted(table):
        counts = table[rk]
        total = sum(counts.values())
        cells = "".join(
            f"{counts.get(c, 0):>8} ({100 * counts.get(c, 0) / total:>3.0f}%)"
            for c in cols
        )
        lines.append(f"{rk:<{width}}" + cells + f"{total:>10}")
    return "\n".join(lines)


def _encode(rows: List[dict], fields=METADATA_FIELDS) -> np.ndarray:
    """One-hot the categorical metadata and append duration statistics."""
    vocab = {f: sorted({r[f] for r in rows}) for f in fields}
    X = []
    for r in rows:
        vec: List[float] = []
        for f in fields:
            vec.extend(1.0 if r[f] == v else 0.0 for v in vocab[f])
        vec.append(float(r["duration"]))
        X.append(vec)
    return np.asarray(X, dtype=np.float64)


def _patient_level(rows: List[dict], fields=METADATA_FIELDS):
    """Aggregate a patient's recordings into one metadata feature vector."""
    vocab = {f: sorted({r[f] for r in rows}) for f in fields}
    by_pid: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["patient_id"]].append(r)

    pids, X, y = [], [], []
    for pid in sorted(by_pid):
        recs = by_pid[pid]
        vec: List[float] = []
        for f in fields:
            counts = Counter(r[f] for r in recs)
            n = sum(counts.values())
            vec.extend(counts.get(v, 0) / n for v in vocab[f])
        durations = [r["duration"] for r in recs]
        vec.extend([
            float(np.mean(durations)),
            float(np.std(durations)),
            float(len(recs)),
            float(len({r["stem"] for r in recs})),
        ])
        pids.append(pid)
        X.append(vec)
        y.append(recs[0]["diagnosis"])
    return np.asarray(X), np.asarray(y), np.asarray(pids)


def _cross_val_predict(X, y, groups, *, class_weight, seed: int, n_splits: int):
    """Grouped CV. Groups are patients, so the probe never sees a patient in both
    folds — the same rule the real models follow."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for tr, te in cv.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=300, random_state=seed, class_weight=class_weight
        )
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    return preds


def probe(X, y, groups, *, name: str, seed: int = 42) -> Dict:
    """Two probes, because one number cannot answer the question.

    An *unweighted* forest answers "can metadata beat the majority baseline on
    raw accuracy" — the direct can-you-cheat question. A *class-balanced* forest
    answers "does metadata identify the minority classes", which raw accuracy
    hides whenever one class dominates (COPD is 64 of 104 known patients, so a
    constant COPD predictor already scores 0.615).

    Comparing the balanced model's accuracy against the majority rate would be
    incoherent — balancing deliberately trades accuracy for minority recall — so
    each model is scored against its own appropriate baseline.
    """
    from sklearn.metrics import accuracy_score, f1_score

    counts = Counter(y)
    majority = max(counts.values()) / len(y)
    chance_f1 = 1.0 / len(counts)
    n_splits = min(5, min(counts.values()))
    if n_splits < 2:
        return {"probe": name, "skipped": "a class has fewer than 2 members"}

    preds_plain = _cross_val_predict(X, y, groups, class_weight=None,
                                     seed=seed, n_splits=n_splits)
    preds_bal = _cross_val_predict(X, y, groups, class_weight="balanced_subsample",
                                   seed=seed, n_splits=n_splits)

    acc = float(accuracy_score(y, preds_plain))
    f1_bal = float(f1_score(y, preds_bal, average="macro"))
    return {
        "probe": name,
        "n": int(len(y)),
        "n_classes": len(counts),
        "majority_class_rate": round(majority, 4),
        "metadata_only_accuracy": round(acc, 4),
        "accuracy_lift_over_majority": round(acc - majority, 4),
        "balanced_f1_macro": round(f1_bal, 4),
        "chance_f1_macro": round(chance_f1, 4),
        "f1_lift_over_chance": round(f1_bal - chance_f1, 4),
        "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="splits/cycles_index_v1.csv")
    ap.add_argument("--split", default="splits/split_v1.json")
    # Not results/ — that directory is gitignored, and this probe's output is a
    # Phase-0 decision record that belongs in the repo and in the paper.
    ap.add_argument("--out", default="reports/shortcut_probe.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disease-alarm", type=float, default=0.15,
                    help="lift over majority above which the disease shortcut is "
                         "treated as confirmed (default 0.15)")
    args = ap.parse_args()

    rows = icbhi.read_cycle_index(args.index)
    split = icbhi.load_split(args.split)

    print("=" * 78)
    print("PHASE-0 SHORTCUT PROBE — metadata only, no audio")
    print("=" * 78)

    print("\n1. Diagnosis x stethoscope (patient level)")
    patient_rows = {}
    for r in rows:
        patient_rows.setdefault((r["patient_id"], r["stem"]), r)
    per_recording = list(patient_rows.values())
    print(crosstab(per_recording, "diagnosis", "device"))

    print("\n2. Diagnosis x chest location (recording level)")
    print(crosstab(per_recording, "diagnosis", "chest_location"))

    results = []

    # --- Disease probe: known classes only, patient level ---------------------
    known = [r for r in rows if r["diagnosis"] in icbhi.KNOWN_DISEASES]
    Xd, yd, pids = _patient_level(known)
    print("\n3. Can metadata alone predict DISEASE? (patient level, known classes)")
    r_disease = probe(Xd, yd, pids, name="disease_from_metadata", seed=args.seed)
    results.append(r_disease)
    for k, v in r_disease.items():
        print(f"   {k:26s} {v}")

    # --- Sound-event probe: cycle level --------------------------------------
    print("\n4. Can metadata alone predict SOUND EVENT? (cycle level)")
    Xs = _encode(rows)
    ys = np.array([r["label_name"] for r in rows])
    gs = np.array([r["patient_id"] for r in rows])
    r_sound = probe(Xs, ys, gs, name="sound_event_from_metadata", seed=args.seed)
    results.append(r_sound)
    for k, v in r_sound.items():
        print(f"   {k:26s} {v}")

    # --- Verdict --------------------------------------------------------------
    # Either signal is sufficient: metadata beating the majority baseline on raw
    # accuracy, or metadata identifying minority diseases well above chance.
    acc_lift = r_disease.get("accuracy_lift_over_majority", 0.0)
    f1_lift = r_disease.get("f1_lift_over_chance", 0.0)
    shortcut = acc_lift >= args.disease_alarm or f1_lift >= args.disease_alarm
    print("\n" + "=" * 78)
    if shortcut:
        print(f"VERDICT: METADATA SHORTCUT CONFIRMED "
              f"(accuracy lift {acc_lift:+.3f}, balanced-F1 lift {f1_lift:+.3f})")
        print(
            "\nRecording metadata alone predicts diagnosis well above the majority\n"
            "baseline. Any disease head trained on this data can reach high accuracy\n"
            "without learning pathology, and cross-task disagreement would then be\n"
            "measuring device/placement mismatch rather than unseen disease.\n"
            "\nDo not start M13/M15 until one of these is in place:\n"
            "  a) report disease results per-device as well as pooled;\n"
            "  b) device-balanced or device-stratified sampling in the disease head;\n"
            "  c) a device-adversarial (gradient-reversal) branch on the backbone;\n"
            "  d) at minimum, disclose this probe's numbers in the paper as a\n"
            "     limitation — reviewers of ICBHI diagnosis papers ask about it."
        )
    else:
        print(f"VERDICT: no strong metadata shortcut "
              f"(accuracy lift {acc_lift:+.3f}, balanced-F1 lift {f1_lift:+.3f})")
        print(
            "\nDiagnosis is not trivially recoverable from stethoscope, chest\n"
            "location, or cycle duration. Report this probe in the paper anyway —\n"
            "it pre-empts the most predictable reviewer objection to any ICBHI\n"
            "diagnosis result."
        )
    print("=" * 78)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "probes": results,
                "disease_shortcut_detected": bool(shortcut),
                "alarm_threshold_lift": args.disease_alarm,
                "method": (
                    "RandomForest(300) on one-hot device / chest location / "
                    "acquisition mode plus duration statistics, StratifiedGroupKFold "
                    "grouped by patient. Reported twice: unweighted (accuracy vs the "
                    "majority-class baseline) and class-balanced (macro-F1 vs chance), "
                    "since raw accuracy hides a shortcut that only affects minority "
                    "classes. No audio is used."
                ),
            },
            fh,
            indent=2,
        )
        fh.write("\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
