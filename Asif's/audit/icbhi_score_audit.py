#!/usr/bin/env python3
"""
Recompute the OFFICIAL ICBHI 2017 challenge score for every model from its committed
confusion matrix, and compare it against the number the project currently reports.

The problem
-----------
Every model in this repo reports:

    icbhi_score = (recall_macro + specificity_macro) / 2

That is NOT the ICBHI 2017 challenge metric. The official metric is:

    Se    = (correctly classified ABNORMAL events) / (all abnormal events)
    Sp    = (correctly classified NORMAL   events) / (all normal events)
    Score = (Se + Sp) / 2

where "abnormal" pools Crackle + Wheeze + Both against Normal.

Why the difference matters: `specificity_macro` averages per-class specificity across all
four classes, and a class's specificity counts true negatives contributed by the other
three. Rare classes (Wheeze, Both) therefore score ~0.95 specificity almost automatically,
which drags the macro average up regardless of whether the model detects them at all. The
reported score is inflated by roughly 0.06-0.22 across this repo's models.

This is a publication-blocking issue, not a cosmetic one: the ~135 published ICBHI papers
report the official metric, so the project's numbers are not comparable to any of them,
and a reviewer familiar with the benchmark checks the metric definition immediately when a
4-class ICBHI score looks high.

Scope
-----
Only applies to the 4-class SOUND-EVENT task (Normal/Crackle/Wheeze/Both). Disease-head
models with 3x3 matrices (COPD/Healthy/URTI) have no ICBHI challenge score and are skipped
with a note. Models with no committed confusion matrix cannot be verified at all and are
listed separately -- that is itself a finding.

Usage
-----
    python3 icbhi_score_audit.py                 # report only (safe default)
    python3 icbhi_score_audit.py --write         # ALSO add official fields to each JSON

`--write` adds, next to the existing `icbhi_score` (which it never modifies or removes):

    best_metrics.icbhi_score_official        float
    best_metrics.icbhi_official_sensitivity  float
    best_metrics.icbhi_official_specificity  float
    best_metrics.icbhi_score_metric_note     str

Nothing is deleted or overwritten, so the change is reversible with `git checkout`.
"""
import argparse
import glob
import json
import os
import sys

SOUND_CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]

METRIC_NOTE = (
    "icbhi_score is (recall_macro + specificity_macro)/2, a project-internal metric that is "
    "NOT the ICBHI 2017 challenge score and is not comparable to published ICBHI results. "
    "icbhi_score_official is the challenge metric: (Se + Sp)/2 with Se = correctly-classified "
    "abnormal (Crackle+Wheeze+Both) / all abnormal, Sp = correctly-classified Normal / all "
    "Normal. Report the official figure in any paper, table, or comparison against prior work."
)


def official_icbhi(cm, class_order):
    """Official ICBHI 2017 challenge score from a 4x4 raw confusion matrix (rows = truth)."""
    if len(cm) != 4 or any(len(r) != 4 for r in cm):
        return None
    try:
        normal_idx = class_order.index("Normal")
    except ValueError:
        return None

    rows = [[float(x) for x in r] for r in cm]
    normal_total = sum(rows[normal_idx])
    sp = rows[normal_idx][normal_idx] / normal_total if normal_total else 0.0

    abn = [i for i in range(4) if i != normal_idx]
    abn_correct = sum(rows[i][i] for i in abn)
    abn_total = sum(sum(rows[i]) for i in abn)
    se = abn_correct / abn_total if abn_total else 0.0
    return se, sp, (se + sp) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="add official-metric fields to each results JSON (additive only)")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()

    repo = a.repo or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", ".."))
    os.chdir(repo)

    verified, skipped_disease, unverifiable = [], [], []

    for path in sorted(glob.glob("**/results_M*.json", recursive=True)):
        if "Archive_Work_Plan" in path:
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue

        bm = d.get("best_metrics") or {}
        meta = d.get("meta") or {}
        mid = meta.get("model_id") or os.path.dirname(path).split(os.sep)[-1]
        split = str((d.get("dataset_info") or {}).get("split_method") or "?")
        reported = bm.get("icbhi_score")
        cm = bm.get("confusion_matrix_raw")
        per_class = bm.get("per_class")
        order = list(per_class.keys()) if isinstance(per_class, dict) else SOUND_CLASSES

        if not isinstance(reported, (int, float)):
            continue

        if cm and len(cm) == 3:
            skipped_disease.append((mid, path, reported))
            continue
        if not cm:
            unverifiable.append((mid, path, reported, split))
            continue

        res = official_icbhi(cm, order)
        if res is None:
            unverifiable.append((mid, path, reported, split))
            continue

        se, sp, off = res
        verified.append(dict(model=mid, path=path, reported=round(float(reported), 4),
                             official=round(off, 4), delta=round(off - float(reported), 4),
                             se=round(se, 4), sp=round(sp, 4), split=split))

        if a.write:
            bm["icbhi_score_official"] = round(off, 4)
            bm["icbhi_official_sensitivity"] = round(se, 4)
            bm["icbhi_official_specificity"] = round(sp, 4)
            bm["icbhi_score_metric_note"] = METRIC_NOTE
            d["best_metrics"] = bm
            with open(path, "w") as f:
                json.dump(d, f, indent=2)

    # ---------------- console report ----------------
    print("=" * 92)
    print("OFFICIAL ICBHI 2017 SCORE AUDIT")
    print("=" * 92)
    print(f"{'model':<12}{'reported':>10}{'OFFICIAL':>10}{'delta':>9}{'Se':>8}{'Sp':>8}   split")
    print("-" * 92)
    for r in sorted(verified, key=lambda x: x["official"], reverse=True):
        print(f"{r['model']:<12}{r['reported']:>10.4f}{r['official']:>10.4f}"
              f"{r['delta']:>+9.4f}{r['se']:>8.4f}{r['sp']:>8.4f}   {r['split'][:34]}")

    if verified:
        worst = max(verified, key=lambda x: -x["delta"])
        mean_delta = sum(r["delta"] for r in verified) / len(verified)
        print("-" * 92)
        print(f"{len(verified)} models verified. Mean inflation {-mean_delta:+.4f}; "
              f"largest {-worst['delta']:.4f} ({worst['model']}).")

    if unverifiable:
        print(f"\n--- CANNOT VERIFY ({len(unverifiable)}): no usable confusion matrix ---")
        for mid, path, rep, split in unverifiable:
            print(f"  {mid:<12} reports icbhi_score={rep:.4f} but commits no 4x4 "
                  f"confusion_matrix_raw  [{path}]")
        print("  These numbers cannot be checked by anyone, including a reviewer who asks.")
        print("  Re-export them with the confusion matrix, per protocol section 4.")

    if skipped_disease:
        print(f"\n--- N/A ({len(skipped_disease)}): disease-head models, 3-class ---")
        for mid, path, rep in skipped_disease:
            print(f"  {mid:<12} 3x3 matrix; the ICBHI challenge score is a sound-event "
                  f"metric and does not apply.")

    # ---------------- markdown + json artifacts ----------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    md = [
        "# Official ICBHI Score Audit", "",
        f"**Generated by** `Asif's/audit/icbhi_score_audit.py` "
        f"(`--write` {'was' if a.write else 'was NOT'} used this run).", "",
        "## The problem", "",
        "Every model here reports `icbhi_score = (recall_macro + specificity_macro) / 2`. "
        "That is not the ICBHI 2017 challenge metric, which is `(Se + Sp) / 2` with **Se** "
        "over pooled abnormal events (Crackle+Wheeze+Both) and **Sp** over Normal.", "",
        "`specificity_macro` averages per-class specificity, and each class's specificity "
        "counts true negatives from the other three classes — so rare classes (Wheeze, Both) "
        "score ~0.95 almost regardless of whether the model detects them. That inflates the "
        "average.", "",
        "This matters because the ~135 published ICBHI papers use the official metric. As "
        "reported, the project's numbers are **not comparable to any prior work**, and they "
        "look implausibly high to anyone who knows the benchmark.", "",
        "## Recomputed from committed confusion matrices", "",
        "| Model | Reported | **Official** | Inflation | Se | Sp | Split |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(verified, key=lambda x: x["official"], reverse=True):
        md.append(f"| {r['model']} | {r['reported']:.4f} | **{r['official']:.4f}** | "
                  f"{-r['delta']:+.4f} | {r['se']:.4f} | {r['sp']:.4f} | `{r['split']}` |")

    md += ["",
           "Published ICBHI SOTA on the official 60/40 split is roughly **0.60–0.65**. Read "
           "the Official column against that, not the Reported column.", ""]

    if unverifiable:
        md += ["## Cannot be verified", "",
               "These report an `icbhi_score` but commit no 4×4 confusion matrix, so the "
               "number cannot be checked by us or by a reviewer:", ""]
        md += [f"- **{mid}** — `{path}` (reports {rep:.4f})"
               for mid, path, rep, _ in unverifiable]
        md += [""]

    if skipped_disease:
        md += ["## Not applicable", "",
               "Disease-head models (3-class COPD/Healthy/URTI). The ICBHI challenge score is "
               "a sound-event metric and does not apply to them:", ""]
        md += [f"- {mid} — `{path}`" for mid, path, _ in skipped_disease]
        md += [""]

    md += ["## What to do", "",
           "1. **Report the Official column** in the paper and in every comparison against "
           "prior work.",
           "2. **Keep both numbers** in the results JSON (run with `--write`) so the internal "
           "metric stays available for continuity but is never mistaken for the challenge score.",
           "3. **Do not compare across splits.** Models on `patient_independent_70_30` are not "
           "comparable to those on `patient_independent_official_60_40`, whichever metric is used.",
           "4. New runs should emit both — see `Model_Training_Protocol.md` §3.", ""]

    with open(os.path.join(out_dir, "ICBHI_SCORE_AUDIT.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(out_dir, "ICBHI_SCORE_AUDIT.json"), "w") as f:
        json.dump({"verified": verified,
                   "unverifiable": [dict(model=m, path=p, reported=r, split=s)
                                    for m, p, r, s in unverifiable],
                   "not_applicable": [dict(model=m, path=p, reported=r)
                                      for m, p, r in skipped_disease]}, f, indent=2)

    print(f"\nWrote {out_dir}/ICBHI_SCORE_AUDIT.md")
    print(f"Wrote {out_dir}/ICBHI_SCORE_AUDIT.json")
    if a.write:
        print(f"\n--write: added official-metric fields to {len(verified)} results JSON(s). "
              f"Nothing was removed; `git diff` shows exactly what changed.")
    else:
        print("\nReport only. Re-run with --write to add the official fields to the JSONs.")


if __name__ == "__main__":
    main()
