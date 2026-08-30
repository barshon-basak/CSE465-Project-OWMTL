"""
N10 - Gate G2, re-run against the CLINICIAN reference
=====================================================

THIS IS THE EXPERIMENT THE PROJECT HAS BEEN UNABLE TO RUN.

Gate G2 asked whether the DSP concept extractors reproduce ICBHI's crackle/wheeze cycle
labels at AUROC >= 0.65. The pre-registered gate (M39) was run twice and failed twice
(crackle 0.5506 / 0.5580, wheeze 0.5729 / 0.5340); the later 14-concept engine scored
0.5556 / 0.5818 on the same test cycles with a DIFFERENT extractor set. Those are two
implementations, not two exports of one run - see DECISION_2026-08-30_CONSOLIDATION.md.
N11 has since shown these labels ARE learnable (AST frozen 0.71/0.76), so read the
numbers below as 'our extractors are weak', not as a reference-standard limit.
`DECISION_2026-08-16_PIVOT.md` recorded the honest ambiguity that followed:

    1. Our extractors are weak.                          TRUE, and demonstrated twice.
    2. The 0.65 target is reachable on this reference
       standard by ANY method.                            UNKNOWN - "this experiment
                                                           cannot decide it."

Nothing in the repo could decide between those two readings, because every validation
used the same reference standard whose reliability was in question. The clinician
listening study supplies a SECOND, INDEPENDENT reference for the same cycles. Scoring the
identical extractors against it separates the two readings directly:

    * extractors agree with the clinician NO BETTER than with ICBHI
        -> reading 1: the extractors are weak, independently of the labels.
    * extractors agree with the clinician BETTER than with ICBHI
        -> reading 2: the ICBHI labels were a binding constraint on G2, and the gate was
           measuring the reference standard as much as the detector.

Both outcomes are publishable and the experiment is pre-committed to reporting whichever
occurs. Everything is scored on the SAME clips, so the comparison is paired and the only
thing that changes is which reference standard is used.

ALSO RUN HERE - the first fine/coarse crackle validation on this corpus. G2 could only
validate crackle/wheeze PRESENCE, because those are the only labels ICBHI carries. The
clinician distinguished fine from coarse, so `fine_crackle_ratio` and
`coarse_crackle_ratio` - two of the fourteen concepts, previously unvalidatable by any
means - get their first reference here. This is the "fine-grained concept annotation"
asset that Gate G0 in the roadmap identified as the ceiling lever.

MAPPING, AND WHY IT IS SAFE. `clip_key.csv` gives (stem, cycle_idx); `concepts_all.npz`
is aligned to `owmtl.icbhi_data.build_cycle_index`. Both enumerate
`sorted(glob("*.wav"))` and then annotation order, 0-based, so (stem, cycle_idx) maps to
an npz row. The mapping is then VERIFIED rather than assumed: the key stores ICBHI's own
crackle/wheeze labels for every clip, so a correct mapping must reproduce them exactly.
If a single clip disagrees the script refuses to run.

RUNNING (CPU, seconds):
    python N10_gate_vs_clinician.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from N9_clinician_reliability import (CRACKLE_NEGATIVE, CRACKLE_POSITIVE, DEFAULT_KEY,
                                      DEFAULT_LABELS, WHEEZE_NEGATIVE, WHEEZE_POSITIVE,
                                      _binarise, read_key, read_labels)  # noqa: E402

EXP_ID = "N10_gate_vs_clinician"
GATE_THRESHOLD = 0.65          # the pre-registered G2 bar, unchanged


def map_clips_to_rows(key, d):
    """(stem, cycle_idx) -> row index in concepts_all.npz, verified against the key."""
    from owmtl.icbhi_data import build_cycle_index, find_split_file
    from make_m2_features import derive_diagnosis_file

    audio = os.environ.get("ICBHI_AUDIO_DIR", r"C:\Users\Barshon\Desktop\ICBHI_final_database")
    if not os.path.isdir(audio):
        raise FileNotFoundError(f"ICBHI audio directory not found: {audio!r}. "
                                "Set ICBHI_AUDIO_DIR.")
    diag, _ = derive_diagnosis_file(os.path.join(C.RESULTS_DIR,
                                                 "derived_patient_diagnosis.txt"))
    records = build_cycle_index(audio, find_split_file([audio]), diag)
    if len(records) != len(d["X"]):
        raise RuntimeError(f"cycle index {len(records)} != concepts {len(d['X'])}")

    seen, index = {}, {}
    for i, r in enumerate(records):
        n = seen.get(r.stem, 0)
        index[(r.stem, n)] = i
        seen[r.stem] = n + 1

    rows, missing, mismatched = {}, [], []
    for cid, k in key.items():
        i = index.get((k["stem"], int(k["cycle_idx"])))
        if i is None:
            missing.append(cid)
            continue
        # The verification that makes this mapping trustworthy.
        if (int(d["crackle"][i]) != int(k["icbhi_crackle"])
                or int(d["wheeze"][i]) != int(k["icbhi_wheeze"])):
            mismatched.append(cid)
            continue
        rows[cid] = i
    return rows, missing, mismatched


def auroc_block(scores, labels, name, n_boot=5000, seed=0):
    """AUROC of a continuous concept score against a binary reference, with CI + gate."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return {"reference": name, "n": int(len(labels)),
                "note": "reference has a single class here - not estimable"}
    # A constant score yields AUROC exactly 0.5 with a zero-width interval, which reads
    # like a precise estimate of "no signal" but is really "no measurement". Report it as
    # degenerate. fine_crackle_ratio is 99.7% zero corpus-wide and hits this path.
    if np.unique(scores).size < 2:
        return {"reference": name, "n": int(len(labels)),
                "n_positive": int(labels.sum()),
                "degenerate": True, "constant_value": round(float(scores[0]), 6),
                "note": "the concept score is CONSTANT across these clips - no AUROC is "
                        "estimable. This is a property of the extractor, not evidence "
                        "about the reference standard.",
                "passes_gate": False}
    ci = C.bootstrap_ci(C.auroc, labels, scores, n_boot=n_boot, seed=seed, stratify=labels)
    return {"reference": name, "n": int(len(labels)),
            "n_positive": int(labels.sum()), "prevalence": round(float(labels.mean()), 4),
            "auroc": ci["point"], "auroc_ci95": ci["ci95"],
            "verdict_vs_chance": C.honest_verdict(ci["ci95"][0], ci["ci95"][1], 0.5),
            "gate_threshold": GATE_THRESHOLD,
            "passes_gate": bool(ci["point"] >= GATE_THRESHOLD and ci["ci95"][0] > 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C.banner("N10 - Gate G2 re-run against the clinician reference",
             "same extractors, same clips, two reference standards")

    key = read_key(args.key)
    rows_raw, n_ans, src = read_labels(args.labels, None)
    if not rows_raw or n_ans == 0:
        return C.blocked(EXP_ID, "no answered clinician sheet found", [args.labels])
    labels = {str(r["clip_id"]).strip(): r for r in rows_raw}
    d = C.load_concepts(args.concepts)
    names = d["concept_names"]

    rowmap, missing, mismatched = map_clips_to_rows(key, d)
    print(f"  clips mapped to concept rows : {len(rowmap)}/{len(key)}")
    if missing:
        print(f"  [warn] {len(missing)} clips not found in the cycle index: {missing[:5]}")
    if mismatched:
        return C.blocked(EXP_ID,
                         f"{len(mismatched)} clips map to rows whose ICBHI labels differ "
                         "from clip_key.csv - the mapping is wrong",
                         ["do not proceed; a misaligned mapping invalidates every number",
                          f"first offenders: {mismatched[:5]}"])
    print("  mapping verified: every clip's ICBHI crackle/wheeze matches clip_key.csv")

    ci_crackle = names.index("crackle_presence")
    ci_wheeze = names.index("wheeze_presence")
    ci_fine = names.index("fine_crackle_ratio")
    ci_coarse = names.index("coarse_crackle_ratio")

    # Build the paired evaluation set: clips with BOTH a usable clinician answer and a
    # concept row. Using the same clips for both references is what makes the comparison a
    # comparison rather than two unrelated numbers.
    sets = {}
    for lbl, field, pos, neg, cidx in (("crackle", "crackles", CRACKLE_POSITIVE,
                                        CRACKLE_NEGATIVE, ci_crackle),
                                       ("wheeze", "wheeze", WHEEZE_POSITIVE,
                                        WHEEZE_NEGATIVE, ci_wheeze)):
        sc, y_clin, y_icbhi, pid = [], [], [], []
        for cid, i in rowmap.items():
            r = labels.get(cid)
            if not r:
                continue
            v = _binarise(r.get(field), pos, neg)
            if v not in (0, 1):
                continue
            sc.append(float(d["X"][i, cidx]))
            y_clin.append(v)
            y_icbhi.append(int(d[lbl][i]))
            pid.append(str(d["patient"][i]))
        sets[lbl] = (np.array(sc), np.array(y_clin), np.array(y_icbhi), np.array(pid))

    out = {"experiment": EXP_ID, "status": "OK",
           "question": ("G2 failed against ICBHI labels. Does the same extractor do better "
                        "against an independent clinical reference on the same clips?"),
           "gate": {"threshold": GATE_THRESHOLD,
                    "concept_engine_14concept_validation": {"crackle_score": 0.5556,
                                                            "wheeze_score": 0.5818},
                    "pre_registered_gate_M39": {"run1": {"crackle": 0.5506, "wheeze": 0.5729},
                                                "run2": {"crackle": 0.5580, "wheeze": 0.5340}},
                    "reference": "ICBHI cycle labels",
                    "n_cycles": 2636,
                    "do_not_mix": "the M39 gate used 9 concepts, the engine 14 - different "
                                  "implementations of the same idea",
                    "superseded_reading": "N11 shows a supervised probe reaches 0.71-0.87 "
                                          "on these same labels; the failure is the "
                                          "extractors, not the reference standard"},
           "mapping": {"clips_mapped": len(rowmap), "verified_against_clip_key": True},
           "comparability_warning": {
               "headline": "ABSOLUTE AUROC HERE IS NOT COMPARABLE TO THE ORIGINAL G2 NUMBER.",
               "why": ("The listening pack was not a random sample. build_listening_pack.py "
                       "filters to cycles of duration >= 0.9 s and then sorts each stratum "
                       "LONGEST-FIRST, so these 132 clips are systematically longer - and "
                       "therefore easier for a DSP detector - than the 2,636-cycle test "
                       "set G2 was scored on. It also over-samples abnormal strata "
                       "(42 normal / 45 crackle / 28 wheeze / 17 both) relative to the "
                       "corpus."),
               "consequence": ("A 'gate PASS' on this subset does NOT overturn the G2 "
                               "failure and must never be reported as doing so."),
               "what_is_valid": ("The PAIRED difference - the same score, on the same "
                                 "clips, against two different reference standards. "
                                 "Selection bias affects both arms identically and "
                                 "cancels. That comparison is the point of this "
                                 "experiment; the absolute numbers are not.")},
           "per_concept": {}}
    print("\n  [!] absolute AUROC on these 132 clips is NOT comparable to G2's 2,636-cycle\n"
          "      number - the pack is longest-first stratified, hence easier. Only the\n"
          "      PAIRED difference below is a valid comparison.")

    print("\n  -- same extractor, two reference standards (paired, identical clips)")
    for lbl in ("crackle", "wheeze"):
        sc, y_clin, y_icbhi, pid = sets[lbl]
        if len(sc) < 20:
            print(f"    {lbl}: only {len(sc)} usable clips - skipped")
            continue
        a_icbhi = auroc_block(sc, y_icbhi, "ICBHI cycle labels", args.n_boot, args.seed)
        a_clin = auroc_block(sc, y_clin, "clinician", args.n_boot, args.seed)
        paired = C.paired_bootstrap_diff(C.auroc, y_clin, sc, sc, n_boot=10, seed=0)
        # A genuine paired comparison needs the SAME score against two DIFFERENT labels,
        # which the generic helper cannot express; bootstrap the difference directly.
        rng = np.random.default_rng(args.seed)
        diffs = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, len(sc), len(sc))
            if len(np.unique(y_clin[idx])) < 2 or len(np.unique(y_icbhi[idx])) < 2:
                continue
            diffs.append(C.auroc(y_clin[idx], sc[idx]) - C.auroc(y_icbhi[idx], sc[idx]))
        diffs = np.asarray(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs.size else (np.nan, np.nan)
        pv = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if diffs.size else np.nan
        paired = {"auroc_clinician_minus_icbhi":
                  round(float(a_clin["auroc"] - a_icbhi["auroc"]), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "p_two_sided": round(float(min(pv, 1.0)), 4),
                  "reading": ("extractor agrees BETTER with the clinician"
                              if lo > 0 else
                              "extractor agrees BETTER with ICBHI" if hi < 0 else
                              "NOT shown to differ between the two reference standards")}

        out["per_concept"][lbl] = {"n_clips": int(len(sc)),
                                   "vs_icbhi": a_icbhi, "vs_clinician": a_clin,
                                   "paired_difference": paired}
        print(f"    {lbl}_presence  vs ICBHI     AUROC {a_icbhi['auroc']:.4f} "
              f"CI{a_icbhi['auroc_ci95']}  [subset-only, not vs G2]")
        print(f"    {lbl}_presence  vs clinician AUROC {a_clin['auroc']:.4f} "
              f"CI{a_clin['auroc_ci95']}  [subset-only, not vs G2]")
        print(f"      paired diff {paired['auroc_clinician_minus_icbhi']:+.4f} "
              f"CI{paired['ci95']} -> {paired['reading']}")

    # ---- first fine/coarse validation on this corpus -------------------------------
    print("\n  -- fine vs coarse crackle (no ICBHI equivalent exists)")
    fine_y, fine_s, coarse_s = [], [], []
    for cid, i in rowmap.items():
        v = (labels.get(cid, {}).get("crackles") or "").strip().lower()
        if v in ("fine", "coarse"):
            fine_y.append(1 if v == "fine" else 0)
            fine_s.append(float(d["X"][i, ci_fine]))
            coarse_s.append(float(d["X"][i, ci_coarse]))
    fine_y = np.array(fine_y)
    fc = {"n": int(len(fine_y)), "n_fine": int(fine_y.sum()),
          "n_coarse": int((1 - fine_y).sum())}
    if (len(fine_y) >= 10 and 0 < fine_y.sum() < len(fine_y)
            and np.unique(fine_s).size > 1):
        fc["fine_crackle_ratio_auroc"] = auroc_block(np.array(fine_s), fine_y,
                                                     "clinician fine-vs-coarse",
                                                     args.n_boot, args.seed)
        fc["coarse_crackle_ratio_auroc"] = auroc_block(-np.array(coarse_s), fine_y,
                                                       "clinician fine-vs-coarse (inverted)",
                                                       args.n_boot, args.seed)
        print(f"    fine_crackle_ratio   AUROC "
              f"{fc['fine_crackle_ratio_auroc']['auroc']:.4f} "
              f"CI{fc['fine_crackle_ratio_auroc']['auroc_ci95']}  (n={fc['n']}, "
              f"{fc['n_fine']} fine / {fc['n_coarse']} coarse)")
    else:
        const = np.unique(fine_s).size <= 1
        fc["degenerate_score"] = bool(const)
        fc["note"] = ((f"fine_crackle_ratio is CONSTANT across these {fc['n']} clips "
                       "(it is 99.7% zero corpus-wide), so no AUROC is estimable - the "
                       "detector has nothing to validate."
                       if const else
                       f"only {fc['n_fine']} fine and {fc['n_coarse']} coarse answers - too "
                       "few to estimate.")
                      + " The first fine/coarse reference labels on this corpus now exist "
                        "either way; they cannot validate a detector at this sample size.")
        print(f"    NOT ESTIMABLE - {fc['note']}")
    fc["significance"] = ("First fine/coarse crackle reference labels on ICBHI. G2 could "
                          "never test these two concepts because ICBHI carries no such "
                          "label; physicians agree on the distinction at kappa < 0.40.")
    out["fine_vs_coarse"] = fc

    # ---- the decision this experiment exists to make -------------------------------
    reads = [v["paired_difference"] for v in out["per_concept"].values()]
    better = sum(1 for r in reads if r["ci95"][0] > 0)
    worse = sum(1 for r in reads if r["ci95"][1] < 0)
    passes = [f"{k}:{ref}" for k, v in out["per_concept"].items()
              for ref in ("vs_icbhi", "vs_clinician") if v[ref].get("passes_gate")]
    if better and not worse:
        verdict = ("READING 2 SUPPORTED: the extractors track the clinician better than "
                   "they track ICBHI on the same clips, so the ICBHI cycle labels were a "
                   "binding constraint on G2. The gate was measuring the reference "
                   "standard as much as the detector.")
    elif worse and not better:
        verdict = ("READING 1 SUPPORTED: agreement is WORSE against the clinician, so the "
                   "extractors are weak independently of which reference is used.")
    else:
        verdict = ("UNDECIDED: no paired difference excludes zero. The extractors do not "
                   "demonstrably track either reference standard better than the other, "
                   "and the G2 ambiguity recorded in DECISION_2026-08-16_PIVOT.md stands "
                   "as written - now with a measurement behind it rather than an "
                   "assumption.")
    out["decision"] = {"n_better_vs_clinician": better, "n_better_vs_icbhi": worse,
                       "gate_passes": passes, "verdict": verdict,
                       "power_note": "single rater, ~110 clips per concept; every interval "
                                     "here is wide and the verdict is a direction, not a "
                                     "measurement of effect size."}
    print(f"\n  DECISION\n  {verdict}")
    if passes:
        print(f"\n  NOTE: {passes} clear {GATE_THRESHOLD} ON THIS EASY SUBSET. That does "
              "NOT\n  overturn the G2 failure - see comparability_warning in the JSON.")
    else:
        print(f"  Gate at {GATE_THRESHOLD}: not cleared against either reference standard.")

    C.save_result(EXP_ID, out)
    return out


if __name__ == "__main__":
    main()
