"""
M48 Tier E — ablation of the EVALUATION PROTOCOL itself.

WHY THIS EXISTS
    M45 ablates the model. Nothing in the repo ablates the thing the paper actually
    contributes: the corrected split loader and the official metric. Section 3 of the paper
    already contains every number, scattered across three subsections as prose. This script
    puts them in one table with one baseline, so that the pipeline figure, the methodology
    section and the ablation table describe the same object.

    Baseline E0 = the full corrected protocol on M22_v2 (0.5602). Each row removes exactly
    one protocol component and reports what the pipeline would then have reported.

TWO KINDS OF ROW, AND THE DIFFERENCE MATTERS
    RESCORE rows (E1, E5) re-score the SAME predictions under a different rule. Nothing is
        retrained, so the delta is exactly one variable and no seed noise enters.
    RERUN rows (E2, E3) cannot be rescorings: changing the split changes which cycles are
        in the test set, so the model must be retrained. Every other setting — architecture,
        optimiser, schedule, class weights, seed, SpecAugment parameters — is held fixed,
        and each source JSON records that explicitly.
    E4 is neither; it is a defect count, not a score.

INPUTS (all committed JSONs, no GPU, ~1 s)
    Asif's/audit/ICBHI_SCORE_AUDIT.json              suite-wide metric inflation
    Asif's/M22_v2/Results/results_M22_v2.json        E0
    Asif's/M22/result_M22/results_M22.json           E2 and E1+E2
    Asif's/M22_v2/Results/results_M22_v2_official.json  E3
    Asif's/M45/M45_paired_tests.json                 E5

RUNNING
    python tier_e_protocol_ablation.py
    python tier_e_protocol_ablation.py --selftest   # validates the two metrics, writes nothing
    python tier_e_protocol_ablation.py --audio-dir <ICBHI dir>   # also computes E4
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

# Windows consoles default to cp1252 and this table contains a delta sign.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

SRC = {
    "audit":   os.path.join(REPO, "Asif's", "audit", "ICBHI_SCORE_AUDIT.json"),
    "E0":      os.path.join(REPO, "Asif's", "M22_v2", "Results", "results_M22_v2.json"),
    "E2":      os.path.join(REPO, "Asif's", "M22", "result_M22", "results_M22.json"),
    "E3":      os.path.join(REPO, "Asif's", "M22_v2", "Results",
                            "results_M22_v2_official.json"),
    "paired":  os.path.join(REPO, "Asif's", "M45", "M45_paired_tests.json"),
}


# --------------------------------------------------------------------- the two metrics
def official(cm):
    """ICBHI 2017 challenge score: pooled abnormal recall, Normal recall, averaged."""
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se = np.trace(cm[1:, 1:]) / abn if abn else float("nan")
    return float((se + sp) / 2), float(se), float(sp)


def macro(cm):
    """The variant this project had been reporting: mean per-class recall and specificity.

    Each per-class specificity draws its true negatives from the other three classes, so a
    rare class scores ~0.95 whether or not the model ever predicts it. The inflation is
    therefore largest exactly where the model is worst.
    """
    cm = np.asarray(cm, float)
    n = cm.sum()
    rec, spec = [], []
    for c in range(cm.shape[0]):
        tp, fn = cm[c, c], cm[c].sum() - cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        tn = n - tp - fn - fp
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
        spec.append(tn / (tn + fp) if tn + fp else 0.0)
    return float((np.mean(rec) + np.mean(spec)) / 2)


# --------------------------------------------------------------------- helpers
def load(tag):
    p = SRC[tag]
    if not os.path.exists(p):
        raise FileNotFoundError(f"{tag}: {p}")
    return json.load(open(p, encoding="utf-8"))


def scores(doc):
    """(official, macro, cm) for a results JSON, recomputed from its committed matrix.

    A score is only admitted here if its raw confusion matrix is committed — the same rule
    the paper applies to every table. Reading the stored number instead would make this
    script a transcription rather than a recomputation.
    """
    bm = doc["best_metrics"]
    cm = bm.get("confusion_matrix_raw")
    if cm is None:
        raise ValueError(f"{doc['meta']['model_id']} commits no confusion_matrix_raw — "
                         "unverifiable, and this table does not transcribe scores")
    o, se, sp = official(cm)
    return o, macro(cm), se, sp, cm


def e4_device_suffix(audio_dir):
    """The fourth fault: a split-file stem whose device suffix is not on disk."""
    sys.path.insert(0, os.path.join(REPO, "Asif's", "audit"))
    from official_split import cross_check_audio, find_split_file  # noqa: E402
    only_in_split, only_on_disk, near = cross_check_audio(audio_dir, find_split_file())
    return {"only_in_split": only_in_split, "only_on_disk": only_on_disk,
            "near_misses": [list(p) for p in near],
            "recordings_silently_dropped_by_a_stem_join": len(only_in_split)}


def e5_unit_of_analysis():
    """How many ablation rows change verdict between cycle-level and patient-level tests."""
    d = load("paired")
    cyc = sum(1 for r in d["rows"].values() if r["cycle_mcnemar_significant"])
    pat = sum(1 for r in d["rows"].values() if r["verdict"].startswith("differs"))
    flipped = [k for k, r in d["rows"].items()
               if r["cycle_mcnemar_significant"] != r["verdict"].startswith("differs")]
    return {"n_rows": len(d["rows"]), "significant_at_cycle_level": cyc,
            "significant_at_patient_level": pat, "rows_that_flip_verdict": sorted(flipped),
            "n_test_cycles": d["n_test_cycles"], "n_test_patients": d["n_test_patients"],
            "effective_sample_size": d["n_test_patients"]}


# --------------------------------------------------------------------- the table
def build(audio_dir=None):
    e0 = load("E0"); e2 = load("E2"); e3 = load("E3")
    o0, m0, se0, sp0, _ = scores(e0)
    o2, m2, se2, sp2, _ = scores(e2)
    o3, m3, se3, sp3, _ = scores(e3)

    ver = load("audit")["verified"]
    deltas = [abs(r["delta"]) for r in ver if r.get("delta") is not None]

    rows = [
        dict(row="E0", kind="baseline",
             removed="none — full corrected protocol",
             reported=round(o0, 4), delta=None, se=round(se0, 4), sp=round(sp0, 4),
             hypothesis="reference: M22_v2, corrected patient-independent 60/40, "
                        "official metric, raw matrix committed",
             evidence="results_M22_v2.json"),

        dict(row="E1", kind="rescore",
             removed="- official metric (report the macro variant instead)",
             reported=round(m0, 4), delta=round(m0 - o0, 4),
             se=round(se0, 4), sp=round(sp0, 4),
             hypothesis="the metric alone inflates the score, non-uniformly, so it does "
                        "not cancel in a comparison and can reorder models",
             evidence=f"same predictions rescored; suite-wide over {len(deltas)} verifiable "
                      f"runs: mean {np.mean(deltas):.4f}, max {max(deltas):.4f}, "
                      f"min {min(deltas):.4f}"),

        dict(row="E2", kind="rerun",
             removed="- official split (silent patient-id <= 111 fallback)",
             reported=round(o2, 4), delta=round(o2 - o0, 4),
             se=round(se2, 4), sp=round(sp2, 4),
             hypothesis="a silently substituted split is worth more than any architectural "
                        "choice in the paper; 11 test patients and 7.1% of cycles read as "
                        "a 40% test set",
             evidence="results_M22.json — same architecture, seed and schedule; split only"),

        dict(row="E3", kind="rerun",
             removed="- patient independence (published split verbatim, 2 patients leak)",
             reported=round(o3, 4), delta=round(o3 - o0, 4),
             se=round(se3, 4), sp=round(sp3, 4),
             hypothesis="the leak is a VALIDITY fault, not a source of inflation — and "
                        "reporting this null honestly is what makes E1 and E2 credible",
             evidence="results_M22_v2_official.json — patients 156, 218 on both sides"),

        dict(row="E1+E2", kind="rescore of E2",
             removed="both — the unaudited pipeline, as it would have been published",
             reported=round(m2, 4), delta=round(m2 - o0, 4),
             se=round(se2, 4), sp=round(sp2, 4),
             hypothesis="the two faults compound rather than cancel; this row is the "
                        "single number that states what the correction was worth",
             evidence="results_M22.json rescored under the macro variant"),
    ]

    out = {
        "table": "M48 Tier E — evaluation-protocol ablation",
        "baseline": "E0 = M22_v2 on the corrected official 60/40 split, official metric",
        "baseline_official": round(o0, 4),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metric_note": "every score in this table is RECOMPUTED from the run's committed "
                       "confusion_matrix_raw, never transcribed",
        "row_kind_note": "RESCORE rows change the rule applied to fixed predictions. RERUN "
                         "rows change the split, which necessarily changes the test set, so "
                         "the model is retrained with every other setting held fixed.",
        "rows": rows,
        "E5_unit_of_analysis": e5_unit_of_analysis(),
        "suite_wide_metric_inflation": {
            "n_verifiable_runs": len(deltas), "mean": round(float(np.mean(deltas)), 4),
            "max": round(float(max(deltas)), 4), "min": round(float(min(deltas)), 4),
            "source": "ICBHI_SCORE_AUDIT.json"},
    }
    if audio_dir and os.path.isdir(audio_dir):
        out["E4_device_suffix_join"] = e4_device_suffix(audio_dir)
    else:
        out["E4_device_suffix_join"] = {
            "status": "not computed", "reason": "pass --audio-dir to compute",
            "known_case": "split lists 226_1b1_Pl_sc_Meditron; disk holds "
                          "226_1b1_Pl_sc_LittC2SE"}
    return out


def render(d):
    L = []
    L.append("# M48 Tier E — evaluation-protocol ablation\n")
    L.append(f"Baseline **E0 = {d['baseline_official']}** — {d['baseline']}.\n")
    L.append("Every score is recomputed from the run's committed raw confusion matrix.\n")
    L.append("| Row | Protocol component removed | Kind | Reported | Δ vs E0 | Se | Sp |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for r in d["rows"]:
        dl = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        L.append(f"| `{r['row']}` | {r['removed']} | {r['kind']} | {r['reported']:.4f} | "
                 f"{dl} | {r['se']:.4f} | {r['sp']:.4f} |")
    e5 = d["E5_unit_of_analysis"]
    L.append(f"| `E5` | − patient-level unit of analysis (test per cycle) | rescore | "
             f"{e5['significant_at_cycle_level']}/{e5['n_rows']} significant | "
             f"vs {e5['significant_at_patient_level']}/{e5['n_rows']} | — | — |")
    e4 = d["E4_device_suffix_join"]
    n4 = e4.get("recordings_silently_dropped_by_a_stem_join", "—")
    L.append(f"| `E4` | − device-suffix-safe file join | defect count | "
             f"{n4} recording(s) dropped | — | — | — |")
    L.append("")
    s = d["suite_wide_metric_inflation"]
    L.append(f"**E1 suite-wide:** across {s['n_verifiable_runs']} verifiable runs the metric "
             f"correction is mean {s['mean']}, max {s['max']}, min {s['min']}.\n")
    L.append(f"**E5:** {e5['significant_at_cycle_level']} of {e5['n_rows']} M45 rows are "
             f"significant scored per cycle, {e5['significant_at_patient_level']} scored per "
             f"patient. Rows that flip: {', '.join(e5['rows_that_flip_verdict'])}. The "
             f"effective sample size is {e5['effective_sample_size']} patients, not "
             f"{e5['n_test_cycles']} cycles.\n")
    L.append("## Hypothesis tested by each row\n")
    for r in d["rows"]:
        L.append(f"- **{r['row']}** — {r['hypothesis']}  \n  <sub>{r['evidence']}</sub>")
    return "\n".join(L) + "\n"


def latex(d):
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Ablation of the evaluation protocol. Row E0 is the full corrected "
         r"protocol; each row removes one protocol component and reports what the pipeline "
         r"would then have reported. Rescore rows re-score fixed predictions; rerun rows "
         r"retrain with only the split changed.}",
         r"\label{tab:protocolablation}", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Row & Component removed & ICBHI & $\Delta$ & $Se$ & $Sp$ \\", r"\midrule"]
    for r in d["rows"]:
        dl = "---" if r["delta"] is None else f"${r['delta']:+.4f}$"
        nm = r["removed"].replace("-", "$-$", 1).replace("&", r"\&").replace("%", r"\%")
        nm = nm.replace("<=", r"$\leq$")
        L.append(f"{r['row']} & {nm} & {r['reported']:.4f} & {dl} & "
                 f"{r['se']:.4f} & {r['sp']:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- selftest
def selftest():
    """The two metrics must disagree in the documented direction on a known pathology.

    M33 reported 0.5506 under the macro variant with an official score of 0.3330 because it
    never predicted Normal. If macro() and official() are implemented correctly, a matrix
    with Sp exactly 0 reproduces that gap: official collapses, macro does not.
    """
    ok = True
    # every Normal cycle predicted Crackle -> Sp = 0
    cm = np.array([[0, 1579, 0, 0],
                   [0,  400, 149, 100],
                   [0,  150, 200,  35],
                   [0,   60,  30,  53]], float)
    o, se, sp = official(cm)
    m = macro(cm)
    print(f"  Sp=0 pathology   official {o:.4f} (Se {se:.4f}, Sp {sp:.4f}) | macro {m:.4f}")
    ok &= abs(sp) < 1e-9 and m - o > 0.15

    # a perfect classifier must score 1.0 under both
    p = np.diag([100.0, 100, 100, 100])
    print(f"  perfect          official {official(p)[0]:.4f} | macro {macro(p):.4f} (want 1.0000)")
    ok &= abs(official(p)[0] - 1.0) < 1e-9 and abs(macro(p) - 1.0) < 1e-9

    # the committed M22_v2 numbers must be reproduced from its own matrix
    try:
        o0, m0, _, _, _ = scores(load("E0"))
        stored = load("E0")["best_metrics"]["icbhi_score_official"]
        print(f"  M22_v2 recompute official {o0:.4f} vs stored {stored:.4f} | macro {m0:.4f}")
        ok &= abs(o0 - stored) < 1e-3
    except (FileNotFoundError, ValueError) as e:
        print(f"  [skip] M22_v2 recompute: {e}")

    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audio-dir", default=os.environ.get(
        "ICBHI_AUDIO_DIR", r"C:\Users\Barshon\Desktop\ICBHI_final_database"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    d = build(a.audio_dir)
    md = render(d)
    print(md)
    json.dump(d, open(os.path.join(HERE, "M48_tier_E_table.json"), "w"), indent=2)
    open(os.path.join(HERE, "M48_tier_E_table.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(HERE, "M48_tier_E_table.tex"), "w", encoding="utf-8").write(latex(d))
    print("  wrote M48_tier_E_table.{json,md,tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
