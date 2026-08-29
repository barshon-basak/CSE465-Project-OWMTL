"""
N8 - Pediatric Physics-Fragility Analysis
=========================================

FACULTY ASK: does the physics degrade on children?

WHY THIS IS A NEW EXPERIMENT (audit finding):
  `Barshon's/Gap7/results_Gap7_OOD.json` is TWELVE LINES. It contains MMD only:

      Coswara   M2 0.8514  M30 0.8137  M35 0.8390
      SPRSound  M2 0.4236  M30 0.4093  M35 0.4434   <- physics loss is WORST on children

  That last number is the entire claimed headline. It has no CI, no schema fields, no
  downstream accuracy, no per-concept analysis, and M35 was trained on the 70/30 split.
  Worse: M35's physics LOSS is a different object from the 14 physics CONCEPTS, so
  "physics-fragility" has never actually been measured on the concepts at all.
  STEP_SEQUENCE.md step 08a was never built. The device axis (G4) already failed - only
  3/126 patients span devices - so pediatric shift is the ONLY covariate stress available.

WHAT MAKES THIS VERSION WORTH RUNNING - A PRE-REGISTERED MECHANISM:
  An MMD number says "the distributions differ". It does not say WHY, and a reviewer will
  ask. The physics does say why: a child's airway is shorter and narrower, so its resonant
  frequency is HIGHER. That is a DIRECTIONAL prediction, made in advance, on named
  concepts:

      wheeze_dominant_freq_hz   expected HIGHER in children
      dominant_freq_hz          expected HIGHER in children
      low_high_freq_ratio       expected LOWER  in children
      rhonchi_presence          expected LOWER  in children  (rhonchi are <300 Hz)

  The other ten concepts have no directional prediction and act as the control set. Four
  pre-registered signs, tested one-sided, plus a binomial test on how many matched. If the
  signs come out as predicted, "adult-tuned acoustic priors fail on pediatric airways via
  frequency scaling" becomes a mechanistic claim rather than a distance measurement. If
  they do not, that is equally reportable and far more honest than an unexplained MMD.

  The predictions below are FIXED IN CODE before any pediatric audio is read. Running the
  script with no SPRSound data emits the adult reference distribution and this prediction
  table - which is exactly how a pre-registration should look.

ALSO REPORTED: per-concept Cohen's d with bootstrap CI, KS statistic, adult-vs-child
separability AUROC per concept, and MMD with a CI (hardening the Gap7 row).

RUNNING:
    python N8_pediatric_fragility.py                              # pre-registration only
    python N8_pediatric_fragility.py --sprsound_dir /path/to/wavs # the full analysis
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N8_pediatric_fragility"

# ---------------------------------------------------------------------------------
# PRE-REGISTERED DIRECTIONAL PREDICTIONS. Fixed before any pediatric audio is read.
# "higher" = the concept should be LARGER in children than in adults.
# Rationale: resonant frequency scales inversely with airway calibre and length.
# ---------------------------------------------------------------------------------
PHYSICS_PREDICTIONS = {
    "wheeze_dominant_freq_hz": "higher",
    "dominant_freq_hz": "higher",
    "low_high_freq_ratio": "lower",
    "rhonchi_presence": "lower",
}


def cohens_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                / max(na + nb - 2, 1))
    return float((np.mean(b) - np.mean(a)) / (s + 1e-12))


def mmd_rbf(X, Y, gamma=None):
    """Unbiased RBF-kernel MMD^2. Median heuristic for the bandwidth if not given."""
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        Z = np.vstack([X, Y])
        sub = Z[np.random.default_rng(0).choice(len(Z), min(500, len(Z)), replace=False)]
        med = np.median(((sub[:, None] - sub[None]) ** 2).sum(-1)) + 1e-12
        gamma = 1.0 / med
    Kxx, Kyy, Kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean())


def extract_sprsound_concepts(root, sr=16000, max_files=None):
    """Run the SAME extractors on pediatric audio, on a matched acoustic unit.

    Using owmtl.concept_extractors verbatim is the whole point: a re-implementation would
    make every adult-vs-child difference uninterpretable (is it the airway or the code?).

    UNIT MATCHING. ICBHI concepts were computed on RAW, UNPADDED respiratory cycles
    (notebook 01: `extract_concept_vector(load_cycle_waveform(...))`). SPRSound ships
    record-level wavs with a JSON of typed events carrying start/end in MILLISECONDS, so
    each annotated event is cut the same way. Slicing SPRSound into arbitrary fixed windows
    instead would compare one breath cycle against several, and the whole frequency
    comparison would be measuring segmentation rather than airways.

    Records marked "Poor Quality" are dropped - they are annotated as unusable by the
    corpus authors, and keeping them would import their noise into the adult-vs-child
    contrast.

    Filenames encode metadata: <patientID>_<age>_<sex>_<position>_<recordID>.wav
    The age is what makes the within-cohort gradient test below possible.
    """
    import json as _json
    import soundfile as sf
    from owmtl.concept_extractors import extract_concept_vector, CONCEPT_NAMES

    jsons = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    pairs = []
    for j in jsons:
        w = j.replace("_json", "_wav").replace(".json", ".wav")
        if os.path.isfile(w):
            pairs.append((j, w))
    if not pairs:
        raise FileNotFoundError(
            f"no matching <*_json>/<*_wav> pairs under {root!r}. Point --sprsound_dir at "
            "the SPRSound checkout root (the folder containing BioCAS2022/).")
    if max_files:
        pairs = pairs[:max_files]

    rows, pids, ages, types = [], [], [], []
    n_poor = n_noage = 0
    for i, (jf, wf) in enumerate(pairs):
        try:
            meta = _json.load(open(jf, encoding="utf-8"))
            if meta.get("record_annotation") == "Poor Quality":
                n_poor += 1
                continue
            events = meta.get("event_annotation") or []
            if not events:
                continue
            parts = os.path.basename(wf)[:-4].split("_")
            pid = parts[0]
            try:
                age = float(parts[1])
            except (IndexError, ValueError):
                age = float("nan")
                n_noage += 1

            y, fs = sf.read(wf, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if fs != sr:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(fs), int(sr))
                y = resample_poly(y, sr // g, fs // g)

            for e in events:
                s0 = int(float(e["start"]) / 1000.0 * sr)      # ms -> samples
                s1 = int(float(e["end"]) / 1000.0 * sr)
                seg = y[max(s0, 0):min(s1, len(y))]
                if len(seg) < int(0.05 * sr):                  # < 50 ms is not a cycle
                    continue
                rows.append(extract_concept_vector(seg, sr=sr))
                pids.append(pid)
                ages.append(age)
                types.append(e.get("type", "?"))
        except Exception as ex:
            if i < 5:
                print(f"    warn {os.path.basename(wf)}: {ex}")
        if (i + 1) % 300 == 0:
            print(f"    {i + 1}/{len(pairs)} records -> {len(rows)} events")

    if not rows:
        raise RuntimeError("no concept vectors extracted from SPRSound")
    print(f"    dropped {n_poor} 'Poor Quality' records; {n_noage} without a parseable age")
    return (np.asarray(rows, dtype=np.float32), np.asarray(pids), np.asarray(ages),
            np.asarray(types), list(CONCEPT_NAMES))


# Within-cohort gradient predictions, DERIVED from PHYSICS_PREDICTIONS so the two can never
# drift apart. If a concept is "higher in children", then as a child grows toward adult
# size it must DECREASE with age - the same physics, tested with a continuous predictor.
AGE_PREDICTIONS = {k: ("negative" if v == "higher" else "positive")
                   for k, v in PHYSICS_PREDICTIONS.items()}


def age_gradient_test(X, ages, pids, names, n_boot=1000, seed=0):
    """Spearman correlation of each concept with AGE, within the pediatric cohort only.

    WHY THIS IS THE STRONGER TEST. The adult-vs-child comparison is confounded: different
    corpus, different stethoscopes, different recording protocol, different annotators. Any
    of those could move a spectral concept, and no amount of bootstrapping fixes it.

    The within-SPRSound age gradient has NONE of those confounds - same corpus, same
    devices, same protocol, same annotators - and it tests exactly the same physics. If
    resonant frequency really scales inversely with airway size, the frequency concepts
    must fall with age INSIDE the pediatric cohort too. Patient-level bootstrap, because
    events from one child are not independent.
    """
    from scipy import stats

    ok = np.isfinite(ages)
    X, ages, pids = X[ok], ages[ok], pids[ok]
    uniq = np.unique(pids)
    idx_by_p = {p: np.flatnonzero(pids == p) for p in uniq}
    rng = np.random.default_rng(seed)

    rows, matched, tested = [], 0, 0
    for j, nm in enumerate(names):
        v = X[:, j]
        m = np.isfinite(v)
        if m.sum() < 50 or np.std(v[m]) == 0:
            rows.append({"concept": nm, "skipped": "degenerate"})
            continue
        rho, p = stats.spearmanr(ages[m], v[m])
        boots = []
        for _ in range(min(n_boot, 500)):
            pick = rng.choice(uniq, len(uniq), replace=True)
            ii = np.concatenate([idx_by_p[q] for q in pick])
            ii = ii[np.isfinite(X[ii, j])]
            if len(ii) > 30:
                r2, _ = stats.spearmanr(ages[ii], X[ii, j])
                if r2 == r2:
                    boots.append(r2)
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots
                  else (float("nan"), float("nan")))
        row = {"concept": nm, "spearman_rho": round(float(rho), 4),
               "rho_ci95_patient_bootstrap": [round(float(lo), 4), round(float(hi), 4)],
               "p": float(f"{p:.3e}"),
               "excludes_zero": bool(lo > 0 or hi < 0)}
        if nm in AGE_PREDICTIONS:
            pred = AGE_PREDICTIONS[nm]
            observed = "negative" if rho < 0 else "positive"
            row["prereg"] = {"predicted": pred, "observed": observed,
                             "direction_matched": pred == observed}
            tested += 1
            matched += int(pred == observed)
        rows.append(row)

    return rows, verdict_from_predictions(rows, "within-cohort age gradient")


def compare(adult, child, names, n_boot=2000, seed=0):
    """Per-concept shift statistics plus the pre-registered directional test."""
    from scipy import stats

    rng = np.random.default_rng(seed)
    rows, matched, tested = [], 0, 0
    for j, nm in enumerate(names):
        a, b = adult[:, j], child[:, j]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5 or (np.std(a) == 0 and np.std(b) == 0):
            rows.append({"concept": nm, "skipped": "degenerate or too few samples"})
            continue
        d = cohens_d(a, b)
        boot = [cohens_d(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                for _ in range(min(n_boot, 500))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ks = stats.ks_2samp(a, b)
        sep = C.auroc(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])

        row = {"concept": nm, "adult_mean": round(float(a.mean()), 4),
               "child_mean": round(float(b.mean()), 4),
               "cohens_d": round(d, 4), "cohens_d_ci95": [round(float(lo), 4),
                                                          round(float(hi), 4)],
               "ks_stat": round(float(ks.statistic), 4),
               "ks_p": float(f"{ks.pvalue:.3e}"),
               "adult_vs_child_auroc": round(sep, 4),
               "shift_excludes_zero": bool(lo > 0 or hi < 0)}

        if nm in PHYSICS_PREDICTIONS:
            pred = PHYSICS_PREDICTIONS[nm]
            observed = "higher" if d > 0 else "lower"
            alt = "less" if pred == "higher" else "greater"   # H1: adult < child, or >
            p1 = float(stats.mannwhitneyu(a, b, alternative=alt).pvalue)
            row["prereg"] = {"predicted": pred, "observed": observed,
                             "direction_matched": pred == observed,
                             "one_sided_p": round(p1, 5)}
            tested += 1
            matched += int(pred == observed)
        rows.append(row)

    return rows, verdict_from_predictions(rows, "cross-corpus")


def verdict_from_predictions(rows, arm_label):
    """Judge the pre-registered mechanism on PER-CONCEPT evidence, not the sign test.

    WHY NOT THE SIGN TEST. With only 4 pre-registered predictions a one-sided binomial
    floors at p = 0.5^4 = 0.0625 - even a PERFECT 4/4 cannot reach p < 0.05. Using it as
    the primary criterion hardcodes "not supported" no matter what the data say. It is
    reported below for completeness, with that floor stated, but the criterion that
    actually carries information is per concept: did the direction match AND does its
    interval exclude zero?
    """
    from scipy import stats

    pre = [r for r in rows if "prereg" in r]
    if not pre:
        return None
    tested = len(pre)
    matched = sum(r["prereg"]["direction_matched"] for r in pre)
    confirmed = [r["concept"] for r in pre
                 if r["prereg"]["direction_matched"] and r.get("excludes_zero")
                 or (r["prereg"]["direction_matched"] and r.get("shift_excludes_zero"))]
    contradicted = [r["concept"] for r in pre
                    if not r["prereg"]["direction_matched"]
                    and (r.get("excludes_zero") or r.get("shift_excludes_zero"))]
    p = float(stats.binomtest(matched, tested, 0.5, alternative="greater").pvalue)

    if len(confirmed) >= 3 and not contradicted:
        v = (f"SUPPORTED ({arm_label}): {len(confirmed)}/{tested} pre-registered concepts "
             "moved in the predicted direction with intervals excluding zero, and none "
             "moved against it.")
    elif len(confirmed) >= 3:
        v = (f"MOSTLY SUPPORTED ({arm_label}): {len(confirmed)}/{tested} confirmed with "
             f"intervals excluding zero, but {contradicted} moved against prediction. "
             "Report both.")
    elif len(confirmed) >= 2:
        v = (f"MIXED ({arm_label}): only {len(confirmed)}/{tested} pre-registered concepts "
             "confirmed. Not enough to claim the mechanism.")
    else:
        v = (f"NOT SUPPORTED ({arm_label}): {len(confirmed)}/{tested} confirmed. Report the "
             "shift as an unexplained distribution difference, not as physics fragility.")

    return {"n_predictions": tested, "n_matched": matched,
            "confirmed_with_ci_excluding_zero": confirmed,
            "contradicted_with_ci_excluding_zero": contradicted,
            "binomial_p_one_sided": round(p, 4),
            "binomial_floor_note": ("With 4 predictions the one-sided binomial cannot go "
                                    "below p = 0.0625 even at 4/4, so it is reported as "
                                    "secondary, never as the criterion."),
            "verdict": v}


def make_figure(rows, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if "cohens_d" in r]
    if not ok:
        return
    ok.sort(key=lambda r: r["cohens_d"])
    y = np.arange(len(ok))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in enumerate(ok):
        lo, hi = r["cohens_d_ci95"]
        pre = r.get("prereg")
        col = ("tab:green" if pre and pre["direction_matched"] else
               "tab:red" if pre else "0.5")
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot(r["cohens_d"], i, "o", color=col, ms=6)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([(r["concept"] + (" *" if "prereg" in r else "")) for r in ok],
                       fontsize=8)
    ax.set_xlabel("Cohen's d  (adult ICBHI -> pediatric SPRSound), 95% CI")
    ax.set_title("N8 - pediatric physics fragility, per concept\n"
                 "* = pre-registered prediction; green = direction matched")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    C.save_figure(fig, "N8_pediatric_fragility.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--sprsound_dir", default=os.environ.get("SPRSOUND_DIR"))
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    C.banner("N8 - Pediatric Physics-Fragility Analysis",
             "pre-registered frequency-scaling mechanism, not just an MMD number")
    d = C.load_concepts(args.concepts)
    names = d["concept_names"]
    adult = d["X"]

    prereg = {"hypothesis": ("Pediatric airways are shorter and narrower, so resonant "
                             "frequency is higher. Adult-tuned acoustic priors should "
                             "therefore shift in named, predictable directions."),
              "predictions": PHYSICS_PREDICTIONS,
              "control_concepts": [n for n in names if n not in PHYSICS_PREDICTIONS],
              "test": "one-sided Mann-Whitney per concept + binomial sign test over the "
                      "4 predictions",
              "fixed_before_seeing_pediatric_data": True}
    adult_ref = {n: {"mean": round(float(np.nanmean(adult[:, j])), 4),
                     "std": round(float(np.nanstd(adult[:, j])), 4),
                     "median": round(float(np.nanmedian(adult[:, j])), 4)}
                 for j, n in enumerate(names)}

    doc = {"experiment": EXP_ID,
           "hardens": {"source": "Barshon's/Gap7/results_Gap7_OOD.json",
                       "committed": {"SPRSound_MMD": {"M2": 0.4236, "M30": 0.4093,
                                                      "M35": 0.4434}},
                       "committed_note": "MMD only - no CI, no schema, no per-concept "
                                         "analysis, and measured on M35's physics LOSS "
                                         "rather than on the 14 physics CONCEPTS",
                       "G4_device_axis": "failed (3/126 patients span devices) - pediatric "
                                         "shift is the only covariate stress available"},
           "pre_registration": prereg,
           "adult_reference_distribution": adult_ref,
           "dataset_info": {"adult": "ICBHI_2017", "n_adult_cycles": int(len(adult)),
                            "pediatric": "SPRSound"}}

    if not args.sprsound_dir or not os.path.isdir(args.sprsound_dir):
        doc["status"] = "PRE-REGISTERED"
        doc["blocked_on"] = {
            "what": "SPRSound pediatric audio",
            "how": "--sprsound_dir <dir of .wav>  (or set SPRSOUND_DIR)",
            "extra_deps": ["soundfile"],
            "note": "The adult reference distribution and the directional predictions "
                    "above are complete. Supplying the audio finishes the test without "
                    "changing a line - which is what makes this a pre-registration."}
        C.save_result(EXP_ID, doc)
        print("\n  PRE-REGISTERED. Adult reference distribution written; predictions fixed:")
        for k, v in PHYSICS_PREDICTIONS.items():
            print(f"    {k:26s} expected {v} in children")
        print(f"\n  [BLOCKED on data] pass --sprsound_dir to run the comparison.")
        return doc

    print(f"\n  extracting concepts from {args.sprsound_dir} with the SAME extractors")
    child, cpid, cage, ctype, cnames = extract_sprsound_concepts(
        args.sprsound_dir, max_files=args.max_files)
    if cnames != names:
        return C.blocked(EXP_ID, "concept vocabulary mismatch between the two corpora",
                         [f"ICBHI: {names}", f"SPRSound: {cnames}"])
    finite_age = cage[np.isfinite(cage)]
    print(f"  pediatric events: {len(child)} | children: {len(np.unique(cpid))} | "
          f"age {finite_age.min():.1f}-{finite_age.max():.1f} y "
          f"(median {np.median(finite_age):.1f})")

    rows, sign_test = compare(adult, child, names, args.n_boot, args.seed)
    for r in rows:
        if "cohens_d" not in r:
            continue
        tag = ""
        if "prereg" in r:
            tag = (f"  [prereg {r['prereg']['predicted']}/"
                   f"{r['prereg']['observed']} "
                   f"{'MATCH' if r['prereg']['direction_matched'] else 'MISS'}]")
        print(f"    {r['concept']:26s} d={r['cohens_d']:+.3f} CI{r['cohens_d_ci95']} "
              f"AUROC {r['adult_vs_child_auroc']:.3f}{tag}")

    print(f"\n  cross-corpus: {sign_test['n_matched']}/{sign_test['n_predictions']} "
          f"directions matched | confirmed (CI excludes zero): "
          f"{sign_test['confirmed_with_ci_excluding_zero']}")
    print(f"  {sign_test['verdict']}")

    # The confound-free arm: same corpus, same devices, same protocol, continuous predictor.
    print("\n  -- within-cohort AGE GRADIENT (no cross-corpus confound)")
    age_rows, age_sign = age_gradient_test(child, cage, cpid, names, seed=args.seed)
    for r in age_rows:
        if "spearman_rho" not in r:
            continue
        tag = ""
        if "prereg" in r:
            tag = (f"  [prereg {r['prereg']['predicted']}/{r['prereg']['observed']} "
                   f"{'MATCH' if r['prereg']['direction_matched'] else 'MISS'}]")
        print(f"    {r['concept']:26s} rho={r['spearman_rho']:+.3f} "
              f"CI{r['rho_ci95_patient_bootstrap']}{tag}")
    if age_sign:
        print(f"\n  age gradient: {age_sign['n_matched']}/{age_sign['n_predictions']} "
              f"directions matched | confirmed (CI excludes zero): "
              f"{age_sign['confirmed_with_ci_excluding_zero']}")
        print(f"  {age_sign['verdict']}")

    # MMD with a CI - the Gap7 row, hardened.
    rng = np.random.default_rng(args.seed)
    sa = adult[rng.choice(len(adult), min(1500, len(adult)), replace=False)]
    sb = child[rng.choice(len(child), min(1500, len(child)), replace=False)]
    sa = np.nan_to_num(sa)
    sb = np.nan_to_num(sb)
    mmd = mmd_rbf(sa, sb)
    boots = [mmd_rbf(sa[rng.integers(0, len(sa), len(sa))],
                     sb[rng.integers(0, len(sb), len(sb))]) for _ in range(100)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  concept-space MMD^2 {mmd:.4f} CI[{lo:.4f}, {hi:.4f}]  "
          f"(Gap7 reported embedding MMD 0.4236 for M2, no CI)")

    try:
        make_figure(rows, names)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    doc.update({"status": "OK",
                "dataset_info": dict(doc["dataset_info"],
                                     n_pediatric_events=int(len(child)),
                                     n_children=int(len(np.unique(cpid))),
                                     age_years={"min": round(float(finite_age.min()), 1),
                                                "max": round(float(finite_age.max()), 1),
                                                "median": round(float(np.median(finite_age)), 1)},
                                     pediatric_unit="annotated respiratory event "
                                                    "(matched to ICBHI's annotated cycle)",
                                     sprsound_dir=args.sprsound_dir),
                "per_concept_shift": rows,
                "prereg_sign_test": sign_test,
                "age_gradient": {
                    "predictions": AGE_PREDICTIONS,
                    "per_concept": age_rows,
                    "sign_test": age_sign,
                    "why_this_is_the_stronger_arm":
                        "The cross-corpus comparison confounds airway size with corpus, "
                        "device, protocol and annotator. The within-SPRSound age gradient "
                        "holds all of those fixed and tests the same physics with a "
                        "continuous predictor."},
                "concept_space_mmd": {"mmd2": round(mmd, 4),
                                      "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                      "space": "14-d physics concepts (not embeddings)"}})
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
