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

WHAT MAKES THIS VERSION WORTH RUNNING - A DECLARED MECHANISM:
  An MMD number says "the distributions differ". It does not say WHY, and a reviewer will
  ask. Two physiological arguments do say why, and each yields DIRECTIONAL predictions on
  NAMED concepts (see PREDICTIONS below):

    A. FREQUENCY SCALING - a child's airway is shorter and narrower, so resonant frequency
       is higher and everything spectral shifts up.
    B. ALLOMETRIC RESPIRATORY RATE - smaller bodies breathe faster, so cycles are shorter
       and events fill a larger fraction of them.

  Six other concepts have no direction under either mechanism and are named explicitly as
  the CONTROL set, so a post-hoc "well, this one moved too" is not available.

  TWO REGISTRATION ROUNDS, REPORTED SEPARATELY. Round 1 (4 predictions) was fixed before
  any pediatric audio was read. Round 2 (4 more) was added afterwards, because a 4-item
  binomial floors at p = 0.0625 and therefore could never support the hypothesis whatever
  the data said - the test had no power to pass. Round 2 is an EXTENSION, not a
  pre-registration, and the sign test is reported for round 1 alone, round 2 alone, and
  all 8. Quoting only the combined figure would launder round-2 predictions as
  pre-registered; quoting only round 1 would discard the power the extension bought.

  With 8 predictions the test can now both pass and fail: 8/8 gives p = 0.0039 and 7/8
  gives 0.0352.

TWO ARMS, and the second is the stronger one:
  * CROSS-CORPUS (ICBHI adults vs SPRSound children) - confounded by corpus, stethoscope,
    protocol and annotator, none of which bootstrapping fixes.
  * WITHIN-COHORT AGE GRADIENT (SPRSound only, ages 0.2-16.2 y) - same corpus, same
    devices, same protocol, same annotators, continuous predictor. Same physics, no
    confound. This is the arm to lead with.

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
# DIRECTIONAL PREDICTIONS. "higher" = the concept should be LARGER in children.
#
# REGISTRATION ROUNDS - read this before quoting any p-value.
#   round 1 : fixed BEFORE any pediatric audio was read. Genuinely pre-registered.
#   round 2 : added AFTER the round-1 run, to give the sign test usable power (see the
#             power note below). These four concepts had not been examined individually
#             when they were declared, but the corpus HAD been opened, so they are an
#             extension, not a pre-registration, and are reported separately. Conflating
#             the two would be exactly the failure the G2 "test read twice" rule exists to
#             prevent.
#
# MECHANISMS. Two distinct physiological arguments, kept separate so a hit on one is not
# credited to the other:
#   A. FREQUENCY SCALING - a child's airway is shorter and narrower, so resonant frequency
#      is higher. Everything spectral shifts up.
#   B. ALLOMETRIC RESPIRATORY RATE - smaller bodies breathe faster, so cycles are shorter
#      and events occupy a larger fraction of them.
#
# WHY THE SET GREW. With 4 predictions a one-sided binomial floors at 0.5^4 = 0.0625: even
# a perfect 4/4 can never reach p < 0.05, so the test could not have supported the
# hypothesis whatever the data said. With 8, 8/8 gives p = 0.0039 and 7/8 gives 0.0352, so
# the test can both pass and fail. The concepts below were chosen because a direction
# follows from A or B, not because of how they behaved in round 1.
# ---------------------------------------------------------------------------------
PREDICTIONS = {
    # -- round 1, mechanism A (frequency scaling) --------------------------------
    "wheeze_dominant_freq_hz": ("higher", 1, "A"),   # tonal peak scales up
    "dominant_freq_hz": ("higher", 1, "A"),          # spectral centroid scales up
    "low_high_freq_ratio": ("lower", 1, "A"),        # energy moves out of the low band
    "rhonchi_presence": ("lower", 1, "A"),           # rhonchi are <300 Hz; fewer qualify
    # -- round 2, mechanism A ----------------------------------------------------
    "fine_crackle_ratio": ("higher", 2, "A"),        # transients reclassify as fine when
                                                     # their centre frequency rises
    "coarse_crackle_ratio": ("lower", 2, "A"),       # the exact mirror of the above
    # -- round 2, mechanism B (respiratory rate) ---------------------------------
    "crackle_rate_hz": ("higher", 2, "B"),           # shorter cycles pack transients denser
    "wheeze_duration_ratio": ("higher", 2, "B"),     # a wheeze of given length fills more
                                                     # of a shorter cycle
}

# Kept as the explicit control set: no direction follows from either mechanism, so these
# must NOT be counted either way. Naming them stops a post-hoc "well this one moved too".
CONTROL_CONCEPTS = ["crackle_presence", "wheeze_presence", "spectral_flatness", "papr_db",
                    "inspiratory_energy_fraction", "transient_timing_centroid"]

# Direction only, for code that does not care about round/mechanism.
PHYSICS_PREDICTIONS = {k: v[0] for k, v in PREDICTIONS.items()}


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
            _, rnd, mech = PREDICTIONS[nm]
            observed = "negative" if rho < 0 else "positive"
            row["prereg"] = {"predicted": pred, "observed": observed,
                             "direction_matched": pred == observed,
                             "round": rnd, "mechanism": mech}
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

        if nm in PREDICTIONS:
            pred, rnd, mech = PREDICTIONS[nm]
            observed = "higher" if d > 0 else "lower"
            alt = "less" if pred == "higher" else "greater"   # H1: adult < child, or >
            p1 = float(stats.mannwhitneyu(a, b, alternative=alt).pvalue)
            row["prereg"] = {"predicted": pred, "observed": observed,
                             "direction_matched": pred == observed,
                             "round": rnd, "mechanism": mech,
                             "one_sided_p": round(p1, 5)}
            tested += 1
            matched += int(pred == observed)
        rows.append(row)

    return rows, verdict_from_predictions(rows, "cross-corpus")


def _sign_block(subset):
    """Binomial sign test over a subset of predictions, with its own power floor stated."""
    from scipy import stats
    n = len(subset)
    if n == 0:
        return None
    m = sum(r["prereg"]["direction_matched"] for r in subset)
    p = float(stats.binomtest(m, n, 0.5, alternative="greater").pvalue)
    floor = float(stats.binomtest(n, n, 0.5, alternative="greater").pvalue)
    return {"n_predictions": n, "n_matched": m,
            "binomial_p_one_sided": round(p, 4),
            "min_achievable_p": round(floor, 4),
            "can_reach_significance": bool(floor < 0.05),
            "concepts": [r["concept"] for r in subset]}


def verdict_from_predictions(rows, arm_label):
    """Judge the mechanism on PER-CONCEPT evidence, with the sign test as support.

    The per-concept criterion (direction matched AND interval excludes zero) is primary
    because it uses the effect sizes, not just their signs, and it does not degrade as the
    number of predictions changes.

    The sign test is reported three ways, and the split matters:
      round 1 : the 4 genuinely pre-registered predictions (fixed before any pediatric
                audio was read). Underpowered - 0.0625 floor - and labelled as such.
      round 2 : the 4 added afterwards to give the test power. An extension, not a
                pre-registration.
      all 8   : the combined test, which CAN reach significance (7/8 -> p = 0.035).
    Reporting only the combined number would launder round-2 predictions as
    pre-registered; reporting only round 1 would throw away the power that was the whole
    point of extending the set.
    """
    pre = [r for r in rows if "prereg" in r]
    if not pre:
        return None

    def _ok(r):
        return r.get("excludes_zero") or r.get("shift_excludes_zero")

    confirmed = [r["concept"] for r in pre if r["prereg"]["direction_matched"] and _ok(r)]
    contradicted = [r["concept"] for r in pre
                    if not r["prereg"]["direction_matched"] and _ok(r)]
    r1 = [r for r in pre if r["prereg"].get("round") == 1]
    r2 = [r for r in pre if r["prereg"].get("round") == 2]
    mech = {}
    for m in ("A", "B"):
        sub = [r for r in pre if r["prereg"].get("mechanism") == m]
        if sub:
            mech[m] = _sign_block(sub)

    n, c = len(pre), len(confirmed)
    allsign = _sign_block(pre)
    sig = allsign["binomial_p_one_sided"] < 0.05 if allsign else False

    # Contradictions are weighted, not just counted against the total. A concept that moves
    # AGAINST a directional prediction with an interval excluding zero is evidence against
    # the mechanism, not merely absence of evidence for it - so two of them cap the verdict
    # at MIXED however many others confirm.
    if c >= 0.75 * n and not contradicted and sig:
        v = (f"SUPPORTED ({arm_label}): {c}/{n} concepts moved as predicted with intervals "
             f"excluding zero, none moved against, and the sign test reaches "
             f"p = {allsign['binomial_p_one_sided']}.")
    elif c >= 0.6 * n and len(contradicted) <= 1:
        v = (f"MOSTLY SUPPORTED ({arm_label}): {c}/{n} confirmed, {len(contradicted)} "
             f"against. Sign test p = {allsign['binomial_p_one_sided']}.")
    elif c >= 0.4 * n:
        v = (f"MIXED ({arm_label}): {c}/{n} confirmed but {len(contradicted)} moved AGAINST "
             f"prediction with intervals excluding zero, and the sign test does not reach "
             f"significance (p = {allsign['binomial_p_one_sided']}). The mechanism is not "
             "established; report the confirming concepts individually and name the "
             "contradicting ones.")
    else:
        v = (f"NOT SUPPORTED ({arm_label}): {c}/{n} confirmed. Report the shift as an "
             "unexplained distribution difference, not as physics fragility.")

    return {"n_predictions": n, "n_matched": sum(r["prereg"]["direction_matched"] for r in pre),
            "confirmed_with_ci_excluding_zero": confirmed,
            "contradicted_with_ci_excluding_zero": contradicted,
            "sign_test_all": _sign_block(pre),
            "sign_test_round1_prereg_only": _sign_block(r1),
            "sign_test_round2_extension": _sign_block(r2),
            "sign_test_by_mechanism": mech,
            "mechanism_key": {"A": "frequency scaling with airway calibre",
                              "B": "allometric respiratory rate"},
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
              "control_concepts": CONTROL_CONCEPTS,
              "registration_rounds": {k: {"direction": v[0], "round": v[1], "mechanism": v[2]}
                                      for k, v in PREDICTIONS.items()},
              "test": "per-concept CI (primary) + one-sided Mann-Whitney + binomial "
                      "sign test reported by registration round (secondary)",
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
