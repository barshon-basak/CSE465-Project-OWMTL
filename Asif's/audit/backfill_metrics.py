#!/usr/bin/env python3
"""backfill_metrics.py - recover accuracy/precision/recall/F1 from committed confusion
matrices, and mark the files where those metrics do not apply.

RTK_requirements.md requirement 4 asks for accuracy, precision, recall and F1 on ALL
models. Two different things were making that fail, and they need opposite fixes:

    1. Runs that committed a raw confusion matrix but never wrote precision_macro /
       recall_macro. Every one of those four metrics is a deterministic function of the
       matrix, so they are recomputed here rather than re-run. Exact, no GPU.

    2. Files that are analyses rather than classifiers - a backbone re-decision, an XAI
       pack, a concept gate, a post-hoc calibrator, a results merge. They emit no 4-class
       prediction, so the metrics are not missing, they are meaningless. Writing zeros or
       leaving blanks both read as "incomplete work"; they get an explicit reason instead.

    The judgement of which files are in group 2 lives in ONE place, `NOT_A_MODEL` in
    backfill_efficiency.py, and is imported here rather than restated.

WHAT IT WILL NOT DO
    It never invents a metric for a file with no matrix and no reason to be exempt - that
    file is reported and left alone. A number nobody can trace back to a committed matrix
    is the exact failure this project's protocol was rewritten to stop.

    python "Asif's/audit/backfill_metrics.py"           # dry run
    python "Asif's/audit/backfill_metrics.py" --write   # additive edit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backfill_efficiency import NOT_A_MODEL  # noqa: E402  (single source of the judgement)

WANTED = ("accuracy", "precision_macro", "recall_macro", "f1_macro")

# A third group, distinct from NOT_A_MODEL: these ARE trained models with their own
# parameters, but the task they were built for is not 4-class sound-event classification,
# so the four metrics above are not the ones that describe them. Marking them "missing"
# would be as wrong as marking a regression model missing an F1.
DIFFERENT_METRIC_FAMILY = {
    "Barshon's/M17": "continual-learning forgetting curve - reports stage-0 retention and "
                     "stage-2 plasticity accuracy plus AUROC, not 4-class metrics",
    "Barshon's/M24": "open-world disease recognition - reports AUROC, AUPR and FPR@95TPR "
                     "over known/unknown patients, not 4-class metrics",
    "Sami's/M7": "deep ensemble handoff - carries per-member results and ensemble "
                 "disagreement; it has no single top-level best_metrics block by design",
}


def from_matrix(cm):
    """The four metrics, macro-averaged over classes present in the matrix.

    A class the model never predicts has an undefined precision. sklearn reports 0 and
    warns; we do the same and record how many classes it happened to, because a macro
    precision that quietly averages in a zero for an unpredicted class is exactly how a
    collapsed model (M33, M36 in this project) can still look mediocre rather than broken.
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.sum() == 0:
        return None
    tp = cm.diagonal()
    pred = cm.sum(axis=0)
    true = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / np.maximum(pred, 1e-12), 0.0)
        rec = np.where(true > 0, tp / np.maximum(true, 1e-12), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {
        "accuracy": round(float(tp.sum() / cm.sum()), 4),
        "precision_macro": round(float(prec.mean()), 4),
        "recall_macro": round(float(rec.mean()), 4),
        "f1_macro": round(float(f1.mean()), 4),
        "_n_classes": int(cm.shape[0]),
        "_unpredicted_classes": int((pred == 0).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    args = ap.parse_args()
    os.chdir(os.path.abspath(args.repo))

    filled, marked, stuck = [], [], []
    for f in sorted(glob.glob("**/results_M*.json", recursive=True)):
        if "Archive_" in f:
            continue
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        bm = doc.get("best_metrics")
        if not isinstance(bm, dict):
            bm = {}
        missing = [k for k in WANTED if bm.get(k) is None]
        if not missing:
            continue

        norm = f.replace("\\", "/")
        got = from_matrix(bm.get("confusion_matrix_raw")) if bm.get("confusion_matrix_raw") else None

        if got:
            note = (f"recomputed from confusion_matrix_raw ({got.pop('_n_classes')}x"
                    f"{len(bm['confusion_matrix_raw'])})")
            unpred = got.pop("_unpredicted_classes")
            if unpred:
                note += (f"; {unpred} class(es) never predicted, so their precision is "
                         "reported as 0 rather than undefined")
            for k in missing:
                bm[k] = got[k]
            doc["best_metrics"] = bm
            doc.setdefault("audit_notes", {})["metrics_source"] = note
            filled.append((f, missing, unpred))
        else:
            why = next((v for k, v in NOT_A_MODEL.items() if norm.startswith(k + "/")), None)
            kind = "not a classifier"
            if not why:
                why = next((v for k, v in DIFFERENT_METRIC_FAMILY.items()
                            if norm.startswith(k + "/")), None)
                kind = "different metric family"
            if not why:
                stuck.append(f)
                continue
            doc.setdefault("audit_notes", {})["metrics_not_applicable"] = (
                f"{why}. Accuracy/precision/recall/F1 are undefined for this file; it is "
                "not an incomplete model record.")
            marked.append((f, why, kind))

        if args.write:
            json.dump(doc, open(f, "w", encoding="utf-8"), indent=2, default=str)

    print(f"{'RECOMPUTED from the committed matrix':-<82}")
    for f, miss, unpred in filled:
        flag = f"  [{unpred} class(es) never predicted]" if unpred else ""
        print(f"  {f[:56]:<56} {','.join(miss)}{flag}")
    print(f"\n{'NOT APPLICABLE - these four metrics do not describe this file':-<82}")
    for f, why, kind in sorted(marked, key=lambda r: r[2]):
        print(f"  {f[:50]:<50} [{kind:<23}] {why[:40]}")
    if stuck:
        print(f"\n{'LEFT ALONE - no matrix and no exemption; needs a re-export':-<82}")
        for f in stuck:
            print(f"  {f}")
    print(f"\n{len(filled)} recomputed, {len(marked)} marked N/A, {len(stuck)} left alone.")
    if not args.write:
        print("\nDRY RUN - nothing written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
