#!/usr/bin/env python3
"""
Analyse the clinician's returned labels.

    python3 analyze_clinician_labels.py --labels "labels v1 - labels.csv" \
                                        --key clip_key.csv \
                                        [--concepts concepts_M39.csv]

Runs the checks in the order `CLINICIAN_LABELING_PACK.md` Part 4 specifies, because the order
matters: if the rater disagrees with *themselves* on the hidden duplicates, nothing downstream is
worth computing.

  1. INTRA-RATER  — the 12 hidden duplicate pairs. Gate on this first.
  2. vs ICBHI     — does the clinician agree with the benchmark's own labels?
  3. vs OUR DSP   — AUROC of each extractor against the clinician as reference.
  4. INTER-RATER  — (--rater2) two clinicians vs each other AND vs ICBHI.

Handling of `ambiguous` (8 clips): OUR BUG, not a rater deviation. We shipped six calibration
exemplars, one named `example_ambiguous.wav`, but the answer options only listed
none/fine/coarse/both/unsure. The rater matched the sound to our exemplar and wrote our word for
it. Confirmed with him directly: it means "none of the named classes fit this", NOT "I can't
tell" — `unsure` was used once, `ambiguous` eight times, and his confidence differs between
them. Primary analysis excludes `ambiguous`; a sensitivity analysis treating it as positive
prints alongside, because which choice you make should be visible rather than buried.
Any v2 pack must give every reference exemplar a matching answer option.

Without --key only the label-only summary runs (distributions, schema, confidence calibration).
"""
import argparse
import collections
import csv
import math
import os
import random
import sys

CR_POS = {"fine", "coarse", "both"}
WZ_POS = {"wheeze", "rhonchi", "both"}
NONDEFINITE = {"ambiguous", "unsure", ""}


def g(r, c):
    return (r.get(c) or "").strip().lower()


def cohen_kappa(a, b):
    """Cohen's kappa for two aligned label lists."""
    assert len(a) == len(b) and a
    cats = sorted(set(a) | set(b))
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    exp = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0, obs


def boot_kappa_ci(a, b, n_boot=10000, seed=20260818, alpha=0.05):
    """Percentile bootstrap CI for Cohen's kappa, resampling CLIPS (the sampling unit).

    Degenerate resamples — draws that happen to contain a single category, so chance agreement is
    1.0 and kappa undefined — are dropped rather than coerced to 0 or 1. At small n and low
    prevalence that can be a non-trivial share, so the number kept is reported: if it falls well
    below n_boot the interval is doing less work than it looks like it is.
    """
    n = len(a)
    if n < 2:
        return float("nan"), float("nan"), 0
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        aa, bb = [a[i] for i in idx], [b[i] for i in idx]
        if len(set(aa)) < 2 and len(set(bb)) < 2:
            continue
        k, _obs = cohen_kappa(aa, bb)
        if k == k:
            ks.append(k)
    if len(ks) < 100:
        return float("nan"), float("nan"), len(ks)
    ks.sort()
    lo = ks[int(alpha / 2 * len(ks))]
    hi = ks[min(len(ks) - 1, int((1 - alpha / 2) * len(ks)))]
    return lo, hi, len(ks)


def boot_sens_ci(d, i, n_boot=10000, seed=20260818, alpha=0.05):
    """Percentile bootstrap CI for sensitivity, resampling CLIPS (not the positives alone).

    Resampling clips is the right unit: the number of ICBHI-positive clips is itself a random
    quantity here, and conditioning on it would understate the interval.
    """
    n = len(d)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        tp = sum(1 for j in idx if d[j] and i[j])
        pos = sum(1 for j in idx if i[j])
        if pos:
            vals.append(tp / pos)
    if len(vals) < 100:
        return float("nan"), float("nan")
    vals.sort()
    return (vals[int(alpha / 2 * len(vals))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def interpret(k):
    return ("poor" if k < 0.20 else "fair" if k < 0.40 else "moderate" if k < 0.60
            else "good" if k < 0.80 else "very good")


# ---------------------------------------------------------------- label-only summary
def summarise(rows):
    print("=" * 74)
    print("1. LABEL-ONLY SUMMARY  (no key required)")
    print("=" * 74)
    print(f"rows: {len(rows)}")

    SCHEMA = {"crackles": {"none", "fine", "coarse", "both", "unsure"},
              "wheeze": {"none", "wheeze", "rhonchi", "both", "unsure"},
              "confidence": {"low", "medium", "high"},
              "audio_quality": {"ok", "noisy", "unusable"}}
    off = collections.defaultdict(collections.Counter)
    for col, allowed in SCHEMA.items():
        for r in rows:
            v = g(r, col)
            if v and v not in allowed:
                off[col][v] += 1
    if off:
        print("\nOff-schema values (the clinician used terms our form did not offer):")
        for col, c in off.items():
            for v, n in c.most_common():
                print(f"   {col}.{v} = {n}")

    usable = [r for r in rows if g(r, "audio_quality") != "unusable"]
    print(f"\nusable: {len(usable)}/{len(rows)}  "
          f"({(len(rows)-len(usable))/len(rows)*100:.1f}% marked unusable)")

    cr = [r for r in usable if g(r, "crackles") not in NONDEFINITE]
    wz = [r for r in usable if g(r, "wheeze") not in NONDEFINITE]
    crp = sum(1 for r in cr if g(r, "crackles") in CR_POS)
    wzp = sum(1 for r in wz if g(r, "wheeze") in WZ_POS)
    print(f"\nPhysician positive rates (definite calls on usable clips):")
    print(f"   crackle+ : {crp}/{len(cr)} = {crp/len(cr)*100:.1f}%")
    print(f"   wheeze+  : {wzp}/{len(wz)} = {wzp/len(wz)*100:.1f}%")
    print(f"   [pack was stratified to ~45.8% crackle+ and ~33.3% wheeze+ by ICBHI labels]")

    print("\nConfidence calibration (share rated 'high'):")
    for lab in ["none", "fine", "coarse", "both", "ambiguous", "unsure"]:
        sub = [r for r in usable if g(r, "crackles") == lab]
        if sub:
            hi = sum(1 for r in sub if g(r, "confidence") == "high")
            print(f"   crackles={lab:<10} n={len(sub):>3}  high={hi/len(sub)*100:5.1f}%")

    n_coarse = sum(1 for r in usable if g(r, "crackles") == "coarse")
    n_fine = sum(1 for r in usable if g(r, "crackles") == "fine")
    print(f"\nfine={n_fine}  coarse={n_coarse}")
    if n_coarse < 8:
        print(f"   *** WARNING: only {n_coarse} coarse call(s). crackle_fine_ratio cannot be")
        print(f"       meaningfully validated at this n regardless of how good it is. ***")
    return usable


# ---------------------------------------------------------------- 1. intra-rater
def intra_rater(rows, key):
    print("\n" + "=" * 74)
    print("2. INTRA-RATER AGREEMENT  (the hidden duplicates -- CHECK THIS FIRST)")
    print("=" * 74)
    by_id = {r["clip_id"]: r for r in rows}
    pairs = [(k["clip_id"], k["is_duplicate_of"]) for k in key
             if (k.get("is_duplicate_of") or "").strip()]
    if not pairs:
        print("No duplicate pairs recorded in the key.")
        return None
    print(f"{len(pairs)} duplicate pair(s)\n")

    agree_cr = agree_wz = comparable = 0
    print(f"{'pair':<26}{'crackles':<26}{'wheeze':<24}")
    print("-" * 74)
    for b, a in pairs:
        ra, rb = by_id.get(a), by_id.get(b)
        if not ra or not rb:
            continue
        ca, cb = g(ra, "crackles"), g(rb, "crackles")
        wa, wb = g(ra, "wheeze"), g(rb, "wheeze")
        both_usable = (g(ra, "audio_quality") != "unusable"
                       and g(rb, "audio_quality") != "unusable")
        mark = "" if both_usable else "  (one unusable)"
        print(f"{a}+{b:<14}{ca+' / '+cb:<26}{wa+' / '+wb:<24}{mark}")
        if both_usable:
            comparable += 1
            agree_cr += (ca == cb)
            agree_wz += (wa == wb)

    if comparable:
        print(f"\ncomparable pairs: {comparable}")
        print(f"  crackles exact agreement : {agree_cr}/{comparable} "
              f"= {agree_cr/comparable*100:.0f}%")
        print(f"  wheeze   exact agreement : {agree_wz}/{comparable} "
              f"= {agree_wz/comparable*100:.0f}%")
        # collapse to presence/absence -- the literature says broad categories agree better
        cpres = sum(1 for b, a in pairs
                    if by_id.get(a) and by_id.get(b)
                    and (g(by_id[a], "crackles") in CR_POS) == (g(by_id[b], "crackles") in CR_POS))
        print(f"  crackles presence-only   : {cpres}/{len(pairs)} "
              f"= {cpres/len(pairs)*100:.0f}%")
        print(f"\n  Pack rule: >3 of 12 self-disagreements means the task is too hard from")
        print(f"  recordings and the fine-grained concepts stay proxies permanently.")
        print(f"  VERDICT: {'PASS' if (comparable-agree_cr) <= 3 else 'FAIL'} "
              f"({comparable-agree_cr} crackle self-disagreement(s))")
    return comparable


# ---------------------------------------------------------------- 2. vs ICBHI
def vs_icbhi(rows, key, mode="exclude_ambiguous"):
    print("\n" + "=" * 74)
    print(f"3. CLINICIAN vs ICBHI LABELS   [{mode}]")
    print("=" * 74)
    by_id = {r["clip_id"]: r for r in rows}
    doc_cr, icb_cr, doc_wz, icb_wz = [], [], [], []
    for k in key:
        r = by_id.get(k["clip_id"])
        if not r or g(r, "audio_quality") == "unusable":
            continue
        dc, dw = g(r, "crackles"), g(r, "wheeze")
        if mode == "ambiguous_as_positive":
            dc = "fine" if dc == "ambiguous" else dc
            dw = "wheeze" if dw == "ambiguous" else dw
        # Exclude PER COLUMN, not jointly: a clip whose wheeze call was non-definite still
        # carries a perfectly usable crackle call. Dropping it from both comparisons discards
        # real data and couples two independent measurements to each other.
        if dc not in NONDEFINITE:
            doc_cr.append(dc in CR_POS)
            icb_cr.append(k["icbhi_crackle"] in ("1", 1, True))
        if dw not in NONDEFINITE:
            doc_wz.append(dw in WZ_POS)
            icb_wz.append(k["icbhi_wheeze"] in ("1", 1, True))

    if not doc_cr:
        print("no comparable clips"); return
    for name, d, i in (("crackle", doc_cr, icb_cr), ("wheeze", doc_wz, icb_wz)):
        ds, is_ = [str(x) for x in d], [str(x) for x in i]
        k, obs = cohen_kappa(ds, is_)
        klo, khi, _nk = boot_kappa_ci(ds, is_)
        tp = sum(1 for a, b in zip(d, i) if a and b)
        fn = sum(1 for a, b in zip(d, i) if not a and b)
        fp = sum(1 for a, b in zip(d, i) if a and not b)
        sens = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n{name} presence  (n={len(d)})")
        print(f"   physician+ {sum(d):>3}   ICBHI+ {sum(i):>3}")
        print(f"   raw agreement {obs*100:5.1f}%   kappa {k:+.3f}  ({interpret(k)})"
              f"   95% CI [{klo:+.3f}, {khi:+.3f}]"
              f"{'  <-- INCLUDES ZERO' if klo <= 0 <= khi else ''}")
        slo, shi = boot_sens_ci(d, i)
        print(f"   sensitivity vs ICBHI: {sens*100:5.1f}%  95% CI [{slo*100:.1f}, {shi*100:.1f}]"
              f"   (caught {tp} of {tp+fn}); false-positives vs ICBHI: {fp}")
    print("\n  Benchmarks: 7 senior physicians reached 23.23% sensitivity / 47.77% ICBHI score")
    print("  on this corpus (Tzeng 2025). Physician-vs-physician kappa on crackle presence is")
    print("  ~0.62, and <0.40 for detailed descriptions (Aviles-Solis 2016).")


# ---------------------------------------------------------------- 3. vs our extractors
def vs_extractors(rows, key, concepts_csv):
    print("\n" + "=" * 74)
    print("4. OUR DSP EXTRACTORS vs THE CLINICIAN (as reference standard)")
    print("=" * 74)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "Statistics"))
    try:
        from owmtl_scores import auroc, delong_ci
    except Exception as e:
        print(f"could not import owmtl_scores ({e})"); return
    try:
        con = {r["unit_id"]: r for r in csv.DictReader(open(concepts_csv))}
    except Exception as e:
        print(f"could not read {concepts_csv} ({e})"); return

    by_id = {r["clip_id"]: r for r in rows}
    tests = [("crackle_score", "crackles", CR_POS),
             ("wheeze_score", "wheeze", WZ_POS),
             ("rhonchi_score", "wheeze", {"rhonchi"}),
             ("crackle_fine_ratio", "crackles", {"fine"})]
    for concept, col, positive in tests:
        y, s = [], []
        for k in key:
            r = by_id.get(k["clip_id"])
            c = con.get(k.get("unit_id", ""))
            if not r or not c or g(r, "audio_quality") == "unusable":
                continue
            v = g(r, col)
            if v in NONDEFINITE:
                continue
            if concept == "crackle_fine_ratio" and v not in CR_POS:
                continue          # fine-vs-coarse only among actual crackles
            try:
                s.append(float(c[concept]))
            except Exception:
                continue
            y.append(1 if v in positive else 0)
        if len(set(y)) < 2:
            print(f"\n{concept:<20} n={len(y):<4} -- only one class present, cannot evaluate")
            continue
        d = delong_ci(y, s)
        print(f"\n{concept:<20} n={len(y):<4} pos={sum(y):<3} "
              f"AUROC {d['auroc']:.4f}  95% CI [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
              f"{'  (excludes chance)' if d['excludes_chance'] else ''}")
        if sum(y) < 8 or len(y) - sum(y) < 8:
            print(f"   *** underpowered: {sum(y)} positive / {len(y)-sum(y)} negative ***")


# ---------------------------------------------------------------- 4. inter-rater
# PRE-REGISTERED 2026-08-18, BEFORE RATER 2'S LABELS EXISTED.
# The three readings below are fixed in advance precisely so the one we get cannot be chosen
# after seeing the number. Whichever fires, it goes in the paper.
INTER_RATER_READINGS = """
  (A) LABELS ARE THE OUTLIER   kappa_inter >= 0.40  AND  kappa_inter - kappa_icbhi >= 0.20
      Two clinicians agree with each other substantially more than either agrees with ICBHI.
      This is the strongest form of our claim: the benchmark's cycle labels, not human
      perception, are the unreliable term. Concept validation against them is bounded by an
      error we can now quantify.

  (B) THE TASK HAS A CEILING   kappa_inter < 0.40
      Clinicians do not agree with each other either. The ceiling is a property of listening to
      these recordings, not of ICBHI's annotation. Our argument survives but changes target:
      no reference standard obtainable from this corpus can support fine-grained concept
      validation. Weaker as a criticism of ICBHI, equally strong as a bound on the method.

  (C) NO CEILING FOUND         kappa_inter >= 0.40  AND  kappa_icbhi >= 0.40
      Clinicians agree with each other AND with ICBHI. The reliability-ceiling argument largely
      fails, and the honest conclusion becomes: our extractors are weak. THIS MUST BE REPORTED
      IF IT HAPPENS. It is the outcome that would cost us the paper's central claim, which is
      exactly why it is written down here in advance.
"""


def inter_rater(rows1, rows2, key, r2_name="rater2"):
    print("\n" + "=" * 74)
    print("5. INTER-RATER AGREEMENT  (rater 1 vs %s)" % r2_name)
    print("=" * 74)

    by1 = {r["clip_id"]: r for r in rows1}
    by2 = {r["clip_id"]: r for r in rows2}
    keyed = {k["clip_id"]: k for k in key}

    # collapse hidden duplicates to their source clip so no recording is counted twice
    canon, seen, units = {}, set(), []
    for cid in sorted(set(by1) & set(by2)):
        k = keyed.get(cid, {})
        src = (k.get("is_duplicate_of") or "").strip() or cid
        canon[cid] = src
        if src in seen:
            continue
        seen.add(src)
        units.append(cid)
    dropped_dup = len(set(by1) & set(by2)) - len(units)

    both, excl_unusable, excl_nondef = [], 0, 0
    for cid in units:
        r1, r2 = by1[cid], by2[cid]
        if g(r1, "audio_quality") == "unusable" or g(r2, "audio_quality") == "unusable":
            excl_unusable += 1
            continue
        both.append((cid, r1, r2))

    print(f"clips rated by both     : {len(set(by1) & set(by2))}")
    if dropped_dup:
        print(f"  - hidden duplicates   : {dropped_dup} collapsed to their source clip")
    print(f"  - either marked unusable: {excl_unusable}")
    print(f"comparable clips        : {len(both)}")
    if len(both) < 10:
        print("\n*** fewer than 10 comparable clips — nothing here is interpretable. ***")
        return
    if len(both) < 30:
        print("\n*** n < 30: treat every interval below as descriptive, not confirmatory. ***")

    summary = {}
    for name, col, pos in (("crackle", "crackles", CR_POS), ("wheeze", "wheeze", WZ_POS)):
        a, b, ic = [], [], []
        for cid, r1, r2 in both:
            v1, v2 = g(r1, col), g(r2, col)
            if v1 in NONDEFINITE or v2 in NONDEFINITE:
                continue
            a.append(str(v1 in pos))
            b.append(str(v2 in pos))
            kk = keyed.get(cid, {})
            ic.append(str(kk.get("icbhi_" + ("crackle" if col == "crackles" else "wheeze"))
                          in ("1", 1, True)))
        if len(a) < 10 or len(set(a) | set(b)) < 2:
            print(f"\n{name}: too few definite calls ({len(a)}) — skipped")
            continue

        k_ab, obs = cohen_kappa(a, b)
        lo, hi, nk = boot_kappa_ci(a, b)
        k_a, _ = cohen_kappa(a, ic)
        k_b, _ = cohen_kappa(b, ic)
        k_icbhi = (k_a + k_b) / 2
        summary[name] = (k_ab, k_icbhi)

        exact = sum(1 for cid, r1, r2 in both if g(r1, col) == g(r2, col))
        print(f"\n{name} presence  (n={len(a)}, both definite)")
        print(f"   rater1+ {a.count('True'):>3}   {r2_name}+ {b.count('True'):>3}"
              f"   ICBHI+ {ic.count('True'):>3}")
        print(f"   raw agreement {obs*100:5.1f}%   exact-category "
              f"{exact}/{len(both)} = {exact/len(both)*100:.0f}%")
        print(f"   INTER-RATER kappa      {k_ab:+.3f}  ({interpret(k_ab)})"
              f"   95% CI [{lo:+.3f}, {hi:+.3f}]   [{nk} valid resamples]")
        print(f"   rater1 vs ICBHI kappa  {k_a:+.3f}")
        print(f"   {r2_name} vs ICBHI kappa {k_b:+.3f}")
        print(f"   -> inter-rater exceeds mean-vs-ICBHI by {k_ab - k_icbhi:+.3f}")

    print("\n" + "-" * 74)
    print("PRE-REGISTERED READING (fixed 2026-08-18, before these labels existed)")
    print("-" * 74)
    print(INTER_RATER_READINGS)
    for name, (k_ab, k_icbhi) in summary.items():
        if k_ab >= 0.40 and k_ab - k_icbhi >= 0.20:
            verdict = "(A) LABELS ARE THE OUTLIER"
        elif k_ab < 0.40:
            verdict = "(B) THE TASK HAS A CEILING"
        elif k_icbhi >= 0.40:
            verdict = "(C) NO CEILING FOUND — report it, it costs us the central claim"
        else:
            verdict = "(A/B boundary) — kappa_inter >= 0.40 but the gap is < 0.20; report both"
        print(f"  {name:<9} kappa_inter {k_ab:+.3f}  vs  kappa_icbhi {k_icbhi:+.3f}  ->  {verdict}")
    print("\n  Literature anchor: physician-vs-physician kappa on ICBHI-style recordings is ~0.62")
    print("  (crackle presence) and <0.40 for detailed descriptions (Aviles-Solis 2016).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--key", default=None, help="clip_key.csv (kept out of the pack)")
    ap.add_argument("--concepts", default=None, help="concepts_M39.csv")
    ap.add_argument("--rater2", default=None,
                    help="second clinician's labels CSV (same schema) -> inter-rater agreement")
    ap.add_argument("--rater2-name", default="rater2", help="label for rater 2 in the output")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.labels)))
    summarise(rows)

    if not a.key:
        print("\n" + "=" * 74)
        print("NO --key SUPPLIED")
        print("=" * 74)
        print("clip_key.csv is required for the two measurements that matter:")
        print("  * intra-rater agreement (which clips were the hidden duplicates)")
        print("  * clinician vs ICBHI    (what the benchmark said about each clip)")
        print("It was written OUTSIDE the pack when you built it -- check the Colab session's")
        print("/content/clip_key.csv, or your Downloads folder.")
        return

    key = list(csv.DictReader(open(a.key)))
    intra_rater(rows, key)
    vs_icbhi(rows, key, "exclude_ambiguous")
    vs_icbhi(rows, key, "ambiguous_as_positive")
    if a.concepts:
        vs_extractors(rows, key, a.concepts)
    if a.rater2:
        rows2 = list(csv.DictReader(open(a.rater2)))
        missing = {"clip_id", "crackles", "wheeze", "audio_quality"} - set(rows2[0] or {})
        if missing:
            print(f"\n--rater2 file is missing required column(s): {sorted(missing)}")
        else:
            inter_rater(rows, rows2, key, a.rater2_name)


if __name__ == "__main__":
    main()
