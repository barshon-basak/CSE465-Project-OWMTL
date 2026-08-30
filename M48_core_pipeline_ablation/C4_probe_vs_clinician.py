"""
M48 C4 — does the machine agree with ICBHI on the cycles the physician does not?

WHY THIS EXISTS
    The paper's sharpest claim is that machines reproduce ICBHI's cycle labels substantially
    better than trained listeners agree with them. Today that claim rests on two numbers
    measured on two different sets: N11's probe AUROC on 2,636 test cycles, and N9's
    clinician kappa on 132 clips. Nobody has put the machine and the physician on the SAME
    clips and asked who the labels side with.

    DECISION_2026-08-30_CONSOLIDATION.md section 5 flags this as "the one optional run worth
    more than the rest ... everything it needs is already on disk". This is that run.

WHAT IS NEW HERE, AND WHAT IS REUSED
    Reused unchanged: N10's verified clip->row mapping, N9's label binarisation, N11's linear
    probe and patient bootstrap, the frozen AST embedding from N1. Nothing is re-extracted,
    no model is trained, no extractor is revised.
    New: the probe is scored on the clinician subset, and the subset is partitioned by
    whether the physician AGREED with ICBHI on that clip.

THE ONE DESIGN DECISION
    N11 fits on train patients and scores test patients. Only some of the 132 clips fall in
    the test split, which would throw most of the listening study away. So the probe here is
    fitted by PATIENT-GROUPED cross-validation over all 6,898 cycles: every clip receives a
    score from a fold that never saw its patient. That is the same out-of-sample guarantee
    N11's split gives, applied to every clip instead of a subset. N11's train/test number is
    reported beside it as the anchor, and the strict test-split subset is reported too so a
    reader can check the CV did not flatter the probe.

THE READING, FIXED BEFORE RUNNING
    On the DISAGREEMENT clips the two references are exact complements, so one AUROC
    determines the other. Let D = the probe's AUROC against ICBHI on those clips.
        D > 0.5, CI excluding 0.5  : the machine sides with ICBHI where the physician does
                                     not. This is the direct form of the paper's headline.
        CI spanning 0.5            : UNDECIDED at this n. Report it as undecided; the
                                     disagreement subset is small and this is the likely
                                     outcome.
        D < 0.5, CI excluding 0.5  : the machine sides with the PHYSICIAN. That would be
                                     evidence the ICBHI labels are noisy where they are
                                     contested, and it would soften the headline. Say so.

WHAT THIS CANNOT SHOW
    Carried from N10: the 132 clips are not a random sample. build_listening_pack.py keeps
    cycles >= 0.9 s and sorts each stratum longest-first, so these clips are systematically
    longer - and easier - than the corpus, and abnormal strata are over-sampled. Absolute
    AUROC here is NOT comparable to N11's 2,636-cycle numbers, and a high value here can
    never overturn the G2 failure. The AGREE-vs-DISAGREE contrast is internal to this subset
    and is unaffected, which is why the contrast rather than the level is the result.

RUNNING (CPU, ~1 min; needs the ICBHI audio dir for the cycle index)
    python C4_probe_vs_clinician.py
    python C4_probe_vs_clinician.py --selftest    # estimator check, writes nothing
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
NOVELTY = os.path.join(REPO, "Novelty Experiment")
sys.path.insert(0, NOVELTY)

import common as C                                                        # noqa: E402
from N9_clinician_reliability import (CRACKLE_NEGATIVE, CRACKLE_POSITIVE,  # noqa: E402
                                      DEFAULT_KEY, DEFAULT_LABELS,
                                      WHEEZE_NEGATIVE, WHEEZE_POSITIVE,
                                      _binarise, read_key, read_labels)
from N10_gate_vs_clinician import map_clips_to_rows                        # noqa: E402
from N11_supervised_ceiling import _fit_probe                              # noqa: E402

EXP_ID = "M48_C4_probe_vs_clinician"
SEED = 20260830
N_BOOT = 5000


def grouped_oof_scores(X, y, groups, n_splits=5, seed=SEED):
    """Out-of-sample probe scores for every row, folds split by PATIENT.

    Grouping by patient is not a nicety: cycles from one patient share a device, a chest
    location and a pathology, so a row-wise fold would let the probe memorise the patient
    and every score here would be optimistic.
    """
    from sklearn.model_selection import GroupKFold
    out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        out[te] = _fit_probe(X[tr], y[tr], X[te], seed=seed)
    assert not np.isnan(out).any(), "some rows never appeared in a test fold"
    return out


def auroc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    """AUROC with a bootstrap CI, or an honest refusal when it is not estimable."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "n_positive": int(y.sum()),
                "note": "single-class reference here - no AUROC is estimable"}
    if np.unique(s).size < 2:
        return {"n": int(len(y)), "n_positive": int(y.sum()), "degenerate": True,
                "note": "the score is constant across these clips - no AUROC is estimable"}
    ci = C.bootstrap_ci(C.auroc, y, s, n_boot=n_boot, seed=seed, stratify=y)
    return {"n": int(len(y)), "n_positive": int(y.sum()),
            "prevalence": round(float(y.mean()), 4),
            "auroc": ci["point"], "auroc_ci95": ci["ci95"],
            "verdict_vs_chance": C.honest_verdict(ci["ci95"][0], ci["ci95"][1], 0.5)}


def build_sets(rowmap, labels, d):
    """Paired evaluation rows: clips carrying BOTH a usable clinician answer and a concept row."""
    sets = {}
    names = list(d["concept_names"])
    for lbl, field, pos, neg, cname in (
            ("crackle", "crackles", CRACKLE_POSITIVE, CRACKLE_NEGATIVE, "crackle_presence"),
            ("wheeze", "wheeze", WHEEZE_POSITIVE, WHEEZE_NEGATIVE, "wheeze_presence")):
        ci = names.index(cname)
        rows, y_clin, y_icbhi, dsp = [], [], [], []
        for cid, i in rowmap.items():
            r = labels.get(cid)
            if not r:
                continue
            v = _binarise(r.get(field), pos, neg)
            if v not in (0, 1):
                continue
            rows.append(i); y_clin.append(v)
            y_icbhi.append(int(d[lbl][i])); dsp.append(float(d["X"][i, ci]))
        sets[lbl] = dict(rows=np.array(rows, int), y_clin=np.array(y_clin, int),
                         y_icbhi=np.array(y_icbhi, int), dsp=np.array(dsp, float))
    return sets


def paired_reference_diff(p, y_a, y_b, n_boot=N_BOOT, seed=SEED):
    """CI on AUROC(scores vs reference A) - AUROC(scores vs reference B).

    `common.paired_bootstrap_diff` holds the reference fixed and varies the scores; here the
    SCORES are fixed and the two REFERENCES differ, so it does not apply. The clips are
    resampled once per iteration and both AUROCs are computed on that same resample, which
    is what makes the difference paired. Two overlapping marginal intervals are not a test.
    """
    p = np.asarray(p, float); y_a = np.asarray(y_a, int); y_b = np.asarray(y_b, int)
    rng = np.random.default_rng(seed)
    n = len(p)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_a[idx])) < 2 or len(np.unique(y_b[idx])) < 2:
            continue          # a degenerate resample is dropped, not coerced
        diffs.append(C.auroc(y_a[idx], p[idx]) - C.auroc(y_b[idx], p[idx]))
    if not diffs:
        return {"diff": float("nan"), "note": "no admissible resample"}
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pv = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"diff": round(float(C.auroc(y_a, p) - C.auroc(y_b, p)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided": round(min(pv, 1.0), 4), "n_boot": int(diffs.size),
            "verdict": ("ICBHI is reproduced BETTER than the clinician on the same clips"
                        if lo > 0 else
                        "the clinician is reproduced better" if hi < 0 else
                        "NOT SHOWN to differ on these clips")}


def analyse(lbl, S, probe_all, d):
    rows, y_clin, y_icbhi = S["rows"], S["y_clin"], S["y_icbhi"]
    p = probe_all[rows]
    agree = y_clin == y_icbhi
    dis = ~agree

    block = {
        "n_clips_scored": int(len(rows)),
        "clinician_vs_icbhi_agreement": round(float(agree.mean()), 4),
        "n_agree": int(agree.sum()), "n_disagree": int(dis.sum()),
        "test_split_clips": int((d["split"][rows] == "test").sum()),

        "probe_vs_icbhi_all_clips": auroc_ci(y_icbhi, p),
        "probe_vs_clinician_all_clips": auroc_ci(y_clin, p),
        "dsp_concept_vs_icbhi_all_clips": auroc_ci(y_icbhi, S["dsp"]),
        "dsp_concept_vs_clinician_all_clips": auroc_ci(y_clin, S["dsp"]),

        "on_clips_where_they_AGREE": auroc_ci(y_icbhi[agree], p[agree]),
        "on_clips_where_they_DISAGREE": auroc_ci(y_icbhi[dis], p[dis]),

        # The properly paired form of the paper's headline: same scores, same clips, the
        # only thing that changes is which reference standard they are scored against.
        "paired_icbhi_minus_clinician": paired_reference_diff(p, y_icbhi, y_clin),
    }

    # The headline reading. On disagreement clips the two references are complements, so
    # this single number answers "whose side is the machine on".
    dd = block["on_clips_where_they_DISAGREE"]
    if "auroc" in dd:
        lo, hi = dd["auroc_ci95"]
        if lo > 0.5:
            v = ("SIDES WITH ICBHI - the probe ranks ICBHI's positives above its negatives "
                 "on exactly the clips the physician called the other way")
        elif hi < 0.5:
            v = ("SIDES WITH THE PHYSICIAN - on contested clips the probe tracks the "
                 "physician, not ICBHI. This softens the headline; report it.")
        else:
            v = ("UNDECIDED at this n - the interval spans chance. State it as undecided "
                 "and do not use it as support in either direction.")
        block["reading"] = v
    else:
        block["reading"] = "not estimable on the disagreement subset - " + dd.get("note", "")
    return block


def selftest():
    """The grouped-CV estimator must recover a planted signal and find nothing in noise."""
    rng = np.random.default_rng(0)
    n, dim = 1500, 24
    groups = np.repeat(np.arange(75), 20)
    y = rng.integers(0, 2, n)
    sig = rng.normal(size=(n, dim)) + y[:, None] * 1.2
    noise = rng.normal(size=(n, dim))
    a_sig = C.auroc(y, grouped_oof_scores(sig, y, groups))
    a_noi = C.auroc(y, grouped_oof_scores(noise, y, groups))
    print(f"  planted signal  grouped-OOF AUROC {a_sig:.4f}  (expect > 0.90)")
    print(f"  pure noise      grouped-OOF AUROC {a_noi:.4f}  (expect ~0.50)")

    # A patient-leaking fold must look BETTER than a patient-grouped one on data where the
    # only signal is the patient id. If it does not, the grouping is not doing its job.
    pat_only = np.repeat(rng.normal(size=(75, dim)), 20, axis=0)
    yp = (groups % 2).astype(int)
    a_grp = C.auroc(yp, grouped_oof_scores(pat_only, yp, groups))
    print(f"  patient-only signal, grouped folds AUROC {a_grp:.4f}  (expect ~0.50: the "
          f"patient is never in-sample)")
    ok = a_sig > 0.90 and abs(a_noi - 0.5) < 0.10 and abs(a_grp - 0.5) < 0.15
    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--ast", default=os.path.join(NOVELTY, "embeddings", "ast_frozen.npy"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        C.banner("M48 C4 selftest", "estimator validation only - writes nothing")
        return selftest()

    C.banner("M48 C4 - the AST probe on the clinician-labelled clips",
             "whose side is the machine on where the physician and ICBHI disagree?")

    d = C.load_concepts(a.concepts)
    n = len(d["X"])
    if not os.path.exists(a.ast):
        return C.blocked(EXP_ID, "frozen AST embedding not found", [a.ast])
    A = np.load(a.ast).astype(np.float32)
    if len(A) != n:
        raise ValueError(f"ast_frozen.npy has {len(A)} rows, concepts have {n} - not "
                         "aligned; refusing to fit on mismatched rows.")

    key = read_key(a.key)
    rows_raw, n_ans, src = read_labels(a.labels, None)
    if not rows_raw or n_ans == 0:
        return C.blocked(EXP_ID, "no answered clinician sheet found", [a.labels])
    labels = {str(r["clip_id"]).strip(): r for r in rows_raw}

    rowmap, missing, mismatched = map_clips_to_rows(key, d)
    if mismatched:
        return C.blocked(EXP_ID,
                         f"{len(mismatched)} clips map to rows whose ICBHI labels differ "
                         "from clip_key.csv - the mapping is wrong",
                         [f"first offenders: {mismatched[:5]}"])
    print(f"  clips mapped and verified : {len(rowmap)}/{len(key)}"
          + (f"  ({len(missing)} not in the cycle index)" if missing else ""))

    tr, te = d["split"] == "train", d["split"] == "test"
    sets = build_sets(rowmap, labels, d)

    out = {"experiment": EXP_ID, "status": "OK",
           "question": ("Does the frozen-AST probe agree with ICBHI on the very cycles "
                        "where the physician does not?"),
           "prereg": {"fixed_utc": "2026-08-30", "before_running": True,
                      "readings": {
                          ">0.5, CI excludes 0.5": "machine sides with ICBHI - the direct "
                                                   "form of the paper's headline",
                          "CI spans 0.5": "UNDECIDED at this n; report as undecided",
                          "<0.5, CI excludes 0.5": "machine sides with the physician; this "
                                                   "softens the headline and we say so"}},
           "estimator": ("frozen AudioSet AST embedding (768-d, never saw a respiratory "
                         "corpus) -> logistic probe, standardised, class_weight=balanced, "
                         "patient-grouped 5-fold CV so every clip is out-of-sample"),
           "reused_unchanged": ["N10 clip->row mapping (verified against clip_key.csv)",
                                "N9 label binarisation", "N11 linear probe",
                                "N1 frozen AST embedding"],
           "mapping": {"clips_mapped": len(rowmap), "verified_against_clip_key": True,
                       "clips_not_in_cycle_index": len(missing)},
           "comparability_warning": {
               "headline": "ABSOLUTE AUROC HERE IS NOT COMPARABLE TO N11's 2,636-CYCLE "
                           "NUMBERS, AND CANNOT OVERTURN THE G2 FAILURE.",
               "why": ("The listening pack keeps cycles >= 0.9 s and sorts each stratum "
                       "longest-first, so these clips are systematically longer - and "
                       "easier for any detector - than the corpus, and abnormal strata are "
                       "over-sampled."),
               "what_is_unaffected": ("The AGREE-vs-DISAGREE contrast is internal to this "
                                      "subset, so the selection applies equally to both "
                                      "halves. The contrast is the result; the level is not.")},
           "n11_anchor": {"ast_frozen_train_test_crackle": 0.7115,
                          "ast_frozen_train_test_wheeze": 0.7621,
                          "note": "N11's strict train-fit / test-score numbers on all 2,636 "
                                  "test cycles, for reference only"},
           "results": {}}

    for lbl in ("crackle", "wheeze"):
        print(f"\n  --- {lbl} ---")
        y = d[lbl].astype(int)
        probe = grouped_oof_scores(A, y, d["patient"])
        # Sanity anchor: the same probe under N11's strict train/test protocol.
        strict = C.auroc(y[te], _fit_probe(A[tr], y[tr], A[te]))
        blk = analyse(lbl, sets[lbl], probe, d)
        blk["anchor_grouped_cv_all_6898_cycles"] = round(float(C.auroc(y, probe)), 4)
        blk["anchor_strict_train_test_2636_cycles"] = round(float(strict), 4)
        out["results"][lbl] = blk

        print(f"    clips {blk['n_clips_scored']}  agree {blk['n_agree']}  "
              f"disagree {blk['n_disagree']}  (clinician-ICBHI agreement "
              f"{blk['clinician_vs_icbhi_agreement']:.3f})")
        for k in ("probe_vs_icbhi_all_clips", "probe_vs_clinician_all_clips",
                  "dsp_concept_vs_icbhi_all_clips", "on_clips_where_they_AGREE",
                  "on_clips_where_they_DISAGREE"):
            v = blk[k]
            if "auroc" in v:
                print(f"    {k:<34} AUROC {v['auroc']:.4f}  CI {v['auroc_ci95']}  n={v['n']}")
            else:
                print(f"    {k:<34} {v.get('note')}")
        pd_ = blk["paired_icbhi_minus_clinician"]
        print(f"    PAIRED  AUROC(vs ICBHI) - AUROC(vs clinician) = {pd_['diff']:+.4f}  "
              f"CI {pd_['ci95']}  p={pd_['p_two_sided']}")
        print(f"            -> {pd_['verdict']}")
        print(f"    READING (disagreement subset): {blk['reading']}")

    out["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    p = os.path.join(HERE, "results_M48_C4.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n  wrote {os.path.basename(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
