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


def extract_sprsound_concepts(wav_dir, sr=16000, max_files=None, cycle_s=8.0):
    """Run the SAME extractors on pediatric audio.

    Using owmtl.concept_extractors verbatim is the whole point: a re-implementation would
    make every adult-vs-child difference uninterpretable (is it the airway or the code?).
    SPRSound ships whole recordings; each is cut into fixed windows because its per-event
    annotation format differs from ICBHI's and mixing annotation conventions would
    reintroduce exactly the confound this test is trying to isolate.
    """
    import soundfile as sf
    from owmtl.concept_extractors import extract_concept_vector, CONCEPT_NAMES

    files = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .wav files under {wav_dir!r}")

    rows, ids = [], []
    for i, f in enumerate(files):
        try:
            y, fs = sf.read(f, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if fs != sr:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(fs), int(sr))
                y = resample_poly(y, sr // g, fs // g)
            win = int(cycle_s * sr)
            for s0 in range(0, max(len(y) - win // 2, 1), win):
                seg = y[s0:s0 + win]
                if len(seg) < win // 2:
                    continue
                rows.append(extract_concept_vector(seg, sr=sr))
                ids.append(os.path.basename(f))
        except Exception as ex:
            if i < 5:
                print(f"    warn {os.path.basename(f)}: {ex}")
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(files)} files")
    if not rows:
        raise RuntimeError("no concept vectors extracted from SPRSound")
    return np.asarray(rows, dtype=np.float32), np.asarray(ids), list(CONCEPT_NAMES)


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

    sign_test = None
    if tested:
        p = float(stats.binomtest(matched, tested, 0.5, alternative="greater").pvalue)
        sign_test = {
            "n_predictions": tested, "n_matched": matched,
            "binomial_p_one_sided": round(p, 4),
            "verdict": ("the pre-registered frequency-scaling mechanism is SUPPORTED"
                        if p < 0.05 else
                        f"only {matched}/{tested} pre-registered signs matched "
                        f"(p={p:.3f}) - the frequency-scaling mechanism is NOT supported; "
                        "report the MMD as an unexplained distribution shift, not as "
                        "physics fragility")}
    return rows, sign_test


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
    child, ids, cnames = extract_sprsound_concepts(args.sprsound_dir,
                                                   max_files=args.max_files)
    if cnames != names:
        return C.blocked(EXP_ID, "concept vocabulary mismatch between the two corpora",
                         [f"ICBHI: {names}", f"SPRSound: {cnames}"])
    print(f"  pediatric segments: {len(child)}")

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

    print(f"\n  sign test: {sign_test['n_matched']}/{sign_test['n_predictions']} matched, "
          f"p={sign_test['binomial_p_one_sided']}")
    print(f"  {sign_test['verdict']}")

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
                                     n_pediatric_segments=int(len(child)),
                                     sprsound_dir=args.sprsound_dir),
                "per_concept_shift": rows,
                "prereg_sign_test": sign_test,
                "concept_space_mmd": {"mmd2": round(mmd, 4),
                                      "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                      "space": "14-d physics concepts (not embeddings)"}})
    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
