"""
N9 - Clinician Label Reliability (and the real-clinician input for N7)
=====================================================================

WHY THIS EXISTS:
  Every other experiment in this folder runs on real data. ONE did not: N7's clinician
  intervention used the DSP concept values as a stand-in oracle and labelled itself
  SIMULATED. This script replaces that stand-in with the actual clinician listening study,
  and in doing so produces the measurement the 2026-08-16 pivot calls "the one piece of
  genuinely new evidence".

WHAT IT MEASURES:
  1. CLINICIAN vs ICBHI AGREEMENT - Cohen's kappa, accuracy, and sensitivity/specificity
     with ICBHI's cycle labels as the reference, with patient-level bootstrap CIs.
  2. INTRA-RATER RELIABILITY - the pack deliberately repeats 12 clips. Agreement between a
     rater and themselves is the ceiling on agreement with anyone else, and it is the
     cheapest sanity check that the labels mean anything.
  3. THE CEILING IMPLICATION - the project's central claim. Gate G2 asked the DSP
     extractors to reach AUROC >= 0.65 against ICBHI's crackle/wheeze labels and they
     reached 0.5556 / 0.5818 (14-concept engine; the pre-registered M39 gate reached
     0.5506-0.5729). This script measures how well a CLINICIAN agrees with those labels.

     SUPERSEDED READING (2026-08-30): this file originally argued that low clinician
     agreement bounds what any extractor can achieve. N11 refutes that - a supervised probe
     on a frozen AudioSet embedding, which never saw a respiratory corpus, reaches AUROC
     0.71 (crackle) / 0.76 (wheeze) on these same test cycles. The labels are learnable.
     The finding here is therefore the OPPOSITE and more interesting one: machines
     reproduce these labels far better than a trained physician does. Do not cite this
     script as a ceiling argument.
  4. It writes `clinician_corrections.csv` - exactly the file N7 already knows how to read,
     so N7 upgrades from SIMULATED to real with no code change.

INPUTS (both already exist in the repo):
    ../Asif's/ICBHI_Dataset_labeling/labels.csv   (or labels.xlsx) - the clinician's answers
    ../Asif's/clip_key.csv                        - clip_id -> stem/cycle_idx/ICBHI labels

  The key is stored separately from the labels ON PURPOSE: the clinician must not see the
  ICBHI answers while listening. Do not merge them.

ANSWER SCHEMA (from build_listening_pack.py, enforced here):
    crackles : none | fine | coarse | both | unsure
    wheeze   : none | wheeze | rhonchi | both | unsure
    confidence : low | medium | high
    audio_quality : ok | noisy | unusable

  "unsure" is EXCLUDED from agreement rather than coerced to a class - the study explicitly
  told the rater that a high unsure rate is a genuine finding. Coercing it would
  manufacture agreement that was never given. The unsure rate is reported as its own result.

THIS SCRIPT NEVER INVENTS A LABEL. If the sheet is blank it reports BLOCKED and stops.

RUNNING:
    python N9_clinician_reliability.py
    python N9_clinician_reliability.py --selftest   # validates the maths on a synthetic
                                                    # sheet; writes nothing to results/
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N9_clinician_reliability"

DEFAULT_LABELS = os.path.join(C.repo_root(), "Asif's", "labels v1 - labels.csv")
DEFAULT_XLSX = os.path.join(C.repo_root(), "Asif's", "ICBHI_Dataset_labeling", "labels.xlsx")
DEFAULT_KEY = os.path.join(C.repo_root(), "Asif's", "clip_key.csv")
CORRECTIONS_OUT = os.path.join(C.HERE, "clinician_corrections.csv")

CRACKLE_POSITIVE = {"fine", "coarse", "both"}
CRACKLE_NEGATIVE = {"none"}
WHEEZE_POSITIVE = {"wheeze", "rhonchi", "both"}
WHEEZE_NEGATIVE = {"none"}

# "ambiguous" is NOT in the pack's declared vocabulary (none/fine/coarse/both/unsure), but
# the returned sheet uses it 12 times. It plainly means the same thing as "unsure" - the
# rater could not decide - so it is treated as unsure: EXCLUDED from agreement, never
# coerced to a class. The deviation is counted and reported rather than absorbed silently,
# because a reader must be able to see that the answer vocabulary drifted.
UNSURE = {"unsure", "ambiguous"}
OFF_SCHEMA = {"ambiguous"}

# Clips the rater marked unusable cannot be judged by anyone, so agreement computed over
# them would be noise attributed to the rater. The primary analysis uses every answered
# clip; a sensitivity analysis drops anything not marked "ok".
GOOD_QUALITY = {"ok"}


def read_labels(csv_path=None, xlsx_path=None):
    """Read the clinician sheet. Prefers whichever file actually has answers in it.

    The pack ships a .csv and a .xlsx of the same template; a rater may fill either. Taking
    the one with more answers avoids silently reading the stale blank copy.
    """
    def _from_csv(p):
        if not p or not os.path.isfile(p):
            return None
        with open(p, newline="", encoding="utf-8-sig") as fh:
            return [dict(r) for r in csv.DictReader(fh)]

    def _from_xlsx(p):
        if not p or not os.path.isfile(p):
            return None
        try:
            import openpyxl
        except ImportError:
            return None
        wb = openpyxl.load_workbook(p, data_only=True)
        ws = wb["labels"] if "labels" in wb.sheetnames else wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return None
        head = [str(h).strip() if h is not None else "" for h in rows[0]]
        return [{head[i]: ("" if v is None else str(v).strip())
                 for i, v in enumerate(r) if i < len(head)} for r in rows[1:]]

    cands = [("csv", _from_csv(csv_path)), ("xlsx", _from_xlsx(xlsx_path))]
    best, best_n, best_src = None, -1, None
    for src, rows in cands:
        if not rows:
            continue
        n = sum(1 for r in rows if str(r.get("crackles") or "").strip()
                or str(r.get("wheeze") or "").strip())
        if n > best_n:
            best, best_n, best_src = rows, n, src
    return best, best_n, best_src


def read_key(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def _binarise(value, positive, negative):
    v = (value or "").strip().lower()
    if v in positive:
        return 1
    if v in negative:
        return 0
    if v in UNSURE:
        return None                      # excluded, never coerced
    return "INVALID" if v else None


def cohens_kappa(a, b):
    """Cohen's kappa for two binary raters."""
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float((a == 1).mean() * (b == 1).mean() + (a == 0).mean() * (b == 0).mean())
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def agreement_block(clin, icbhi, patients, label, n_boot=2000, seed=0):
    """Clinician-vs-ICBHI agreement, bootstrapped over PATIENTS (cycles from one patient
    are not independent)."""
    clin = np.asarray(clin)
    icbhi = np.asarray(icbhi)
    patients = np.asarray(patients)
    if len(clin) == 0:
        return {"label": label, "n": 0, "note": "no usable (non-unsure) answers"}

    tp = int(((clin == 1) & (icbhi == 1)).sum())
    fp = int(((clin == 1) & (icbhi == 0)).sum())
    fn = int(((clin == 0) & (icbhi == 1)).sum())
    tn = int(((clin == 0) & (icbhi == 0)).sum())

    uniq = np.unique(patients)
    idx = {p: np.flatnonzero(patients == p) for p in uniq}
    rng = np.random.default_rng(seed)
    ks, accs = [], []
    for _ in range(n_boot):
        pick = np.concatenate([idx[p] for p in rng.choice(uniq, len(uniq), replace=True)])
        k = cohens_kappa(clin[pick], icbhi[pick])
        if k == k:
            ks.append(k)
        accs.append(float((clin[pick] == icbhi[pick]).mean()))
    kq = np.percentile(ks, [2.5, 97.5]) if ks else [float("nan")] * 2
    aq = np.percentile(accs, [2.5, 97.5]) if accs else [float("nan")] * 2

    return {
        "label": label, "n": int(len(clin)), "n_patients": int(len(uniq)),
        "cohens_kappa": round(cohens_kappa(clin, icbhi), 4),
        "kappa_ci95_patient_bootstrap": [round(float(x), 4) for x in kq],
        "raw_agreement": round(float((clin == icbhi).mean()), 4),
        "raw_agreement_ci95": [round(float(x), 4) for x in aq],
        "sensitivity_vs_icbhi": round(tp / max(tp + fn, 1), 4),
        "specificity_vs_icbhi": round(tn / max(tn + fp, 1), 4),
        "confusion_vs_icbhi": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "kappa_interpretation": _kappa_words(cohens_kappa(clin, icbhi)),
    }


def _kappa_words(k):
    if k != k:
        return "not estimable"
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def intra_rater(rows_by_clip, key, label_field, positive, negative):
    """Agreement of the rater with THEMSELVES on the deliberately repeated clips."""
    pairs = []
    for cid, k in key.items():
        dup = (k.get("is_duplicate_of") or "").strip()
        if not dup:
            continue
        a = rows_by_clip.get(cid)
        b = rows_by_clip.get(dup)
        if not a or not b:
            continue
        va = _binarise(a.get(label_field), positive, negative)
        vb = _binarise(b.get(label_field), positive, negative)
        if va in (0, 1) and vb in (0, 1):
            pairs.append((va, vb))
    if not pairs:
        return {"n_pairs": 0, "note": "no usable duplicate pairs (both answered, neither unsure)"}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_pairs": len(pairs),
            "cohens_kappa": round(cohens_kappa(a, b), 4),
            "raw_agreement": round(float((a == b).mean()), 4),
            "interpretation": _kappa_words(cohens_kappa(a, b)),
            "note": "A rater's agreement with themselves upper-bounds their agreement with "
                    "any reference standard. If this is low, everything below it is noise."}


def write_corrections(rows_by_clip, key, path):
    """Convert cycle-level clinician answers into patient-level concept corrections.

    N7 intervenes on a PATIENT's mean concept vector, so cycle answers are averaged per
    patient. Only `crackle_presence` and `wheeze_presence` are written - they are the only
    two of the 14 concepts a listener was actually asked about, and inventing values for
    the other twelve is precisely what this script exists to stop.

    LIMITATION, recorded in the output: each patient contributes only the 1-3 cycles that
    were sampled into the pack, out of the ~60 in their record. This is a sparse correction,
    not a re-labelling of the patient.
    """
    per = {}
    for cid, r in rows_by_clip.items():
        k = key.get(cid)
        if not k:
            continue
        pid = k["stem"].split("_")[0]
        cr = _binarise(r.get("crackles"), CRACKLE_POSITIVE, CRACKLE_NEGATIVE)
        wh = _binarise(r.get("wheeze"), WHEEZE_POSITIVE, WHEEZE_NEGATIVE)
        d = per.setdefault(pid, {"crackle_presence": [], "wheeze_presence": []})
        if cr in (0, 1):
            d["crackle_presence"].append(cr)
        if wh in (0, 1):
            d["wheeze_presence"].append(wh)

    out = []
    for pid, d in sorted(per.items()):
        for concept, vals in d.items():
            if vals:
                out.append({"patient": pid, "concept": concept,
                            "value": round(float(np.mean(vals)), 4),
                            "n_cycles": len(vals)})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient", "concept", "value", "n_cycles"])
        w.writeheader()
        w.writerows(out)
    return len(out), len({r["patient"] for r in out})


def write_release_annotations(rows_by_clip, key, doc, out_dir):
    """Emit the released annotation asset: ICBHI cycles joined to clinician labels.

    Gate G0 in `OWMTL_Decision_Roadmap (v3).md` named exactly this as the project's ceiling
    lever - "the annotated subset becomes a released asset (first fine-grained concept
    annotations for ICBHI)". It is a deliverable, not an experiment, so it ships as a plain
    CSV plus a README that states provenance and limits rather than as a results JSON.

    The join is the only place the clinician answers and the ICBHI reference sit in one
    file. Everything needed to re-derive our numbers - and to disagree with them - is here.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ICBHI_clinician_annotations_v1.csv")
    cols = ["clip_id", "stem", "cycle_idx", "start_s", "end_s", "stratum",
            "icbhi_crackle", "icbhi_wheeze",
            "clinician_crackles", "clinician_wheeze", "clinician_confidence",
            "clinician_audio_quality", "is_duplicate_of"]
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for cid, k in sorted(key.items()):
            r = rows_by_clip.get(cid, {})
            w.writerow({
                "clip_id": cid, "stem": k["stem"], "cycle_idx": k["cycle_idx"],
                "start_s": k["start"], "end_s": k["end"], "stratum": k["stratum"],
                "icbhi_crackle": k["icbhi_crackle"], "icbhi_wheeze": k["icbhi_wheeze"],
                "clinician_crackles": (r.get("crackles") or "").strip(),
                "clinician_wheeze": (r.get("wheeze") or "").strip(),
                "clinician_confidence": (r.get("confidence") or "").strip(),
                "clinician_audio_quality": (r.get("audio_quality") or "").strip(),
                "is_duplicate_of": k.get("is_duplicate_of", "")})
            n += 1

    ag = doc["agreement_vs_icbhi"]
    ir = doc["intra_rater"]
    cov = doc["coverage"]
    readme = f"""# ICBHI clinician annotations, v1

{n} respiratory cycles from ICBHI 2017, independently labelled by one clinician who was
blind to the ICBHI annotations. To our knowledge these are the first fine-grained
(fine-vs-coarse crackle) reference labels released for this corpus.

## Files
- `ICBHI_clinician_annotations_v1.csv` - the labels, joined to the ICBHI reference.

## Provenance
Clips were sampled from ICBHI 2017 by `Asif's/engine/build_listening_pack.py` (seed 42),
stratified over normal / crackle-only / wheeze-only / both. **The sample is not random:**
cycles shorter than 0.9 s were excluded and each stratum was sorted longest-first, so these
clips are systematically longer, and easier to judge, than the corpus average. Any
detector scored on this subset will look better here than on ICBHI as a whole.

The rater listened with headphones after calibrating on reference exemplars, could replay
freely, and was instructed to answer `unsure` rather than guess. {len(key)} clips were
presented ({len({k['unit_id'] for k in key.values()})} unique cycles plus hidden repeats);
{cov['clips_answered']} carry an answer. The repeats are used to measure self-consistency.

## Schema
`clinician_crackles` : none | fine | coarse | both | unsure | ambiguous
`clinician_wheeze`   : none | wheeze | rhonchi | both | unsure | ambiguous
`clinician_confidence`   : low | medium | high
`clinician_audio_quality`: ok | noisy | unusable

`ambiguous` is not in the task's declared vocabulary but appears {doc['schema_deviations']['n_field_instances']} times; it means the
rater could not decide and should be treated as `unsure`. Blank rows are clips the rater
marked `unusable` and declined to score - that is a deliberate answer, not missing data.

## Measured reliability
| quantity | crackles | wheeze |
|---|---|---|
| vs ICBHI, Cohen's kappa | {ag['crackles']['cohens_kappa']} {ag['crackles']['kappa_ci95_patient_bootstrap']} | {ag['wheeze']['cohens_kappa']} {ag['wheeze']['kappa_ci95_patient_bootstrap']} |
| vs ICBHI, raw agreement | {ag['crackles']['raw_agreement']} | {ag['wheeze']['raw_agreement']} |
| sensitivity vs ICBHI | {ag['crackles']['sensitivity_vs_icbhi']} | {ag['wheeze']['sensitivity_vs_icbhi']} |
| specificity vs ICBHI | {ag['crackles']['specificity_vs_icbhi']} | {ag['wheeze']['specificity_vs_icbhi']} |
| intra-rater kappa (hidden duplicates) | {ir['crackles'].get('cohens_kappa')} | {ir['wheeze'].get('cohens_kappa')} |

The rater is far more consistent with themselves than with ICBHI. That gap is the reason
this file exists.

## Limitations - read before using
1. **One rater.** No inter-rater agreement can be computed. Published multi-rater studies
   (n=7 and n=12) carry the reliability argument; this adds granularity, not consensus.
2. **Not a random sample** - see Provenance. Do not compute a corpus-level detector score
   from it.
3. **Fine vs coarse is barely populated** ({cov.get('usable_crackle_answers')} usable crackle answers, of which only a
   handful are coarse). The distinction is released as a reference, not as a benchmark.
4. These labels are an *independent second opinion*, not a correction of ICBHI. Where they
   disagree, neither is established as right.

## Citation
Released as part of the OWMTL corrected-evaluation study. Cite the ICBHI database
(Rocha et al., Physiol. Meas. 40(3), 2019) alongside any use of these labels.
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    return csv_path, n


def analyse(rows, key, n_boot=2000, seed=0, quality_filter=None):
    """Score the sheet. `quality_filter` (e.g. GOOD_QUALITY) restricts to those clips."""
    rows_by_clip = {str(r.get("clip_id", "")).strip(): r for r in rows
                    if str(r.get("clip_id", "")).strip()}
    if quality_filter is not None:
        rows_by_clip = {k: r for k, r in rows_by_clip.items()
                        if (r.get("audio_quality") or "").strip().lower() in quality_filter}
    doc = {"experiment": EXP_ID, "status": "OK"}
    off = sum(1 for r in rows_by_clip.values() for f in ("crackles", "wheeze")
              if (r.get(f) or "").strip().lower() in OFF_SCHEMA)
    doc["schema_deviations"] = {
        "off_schema_values_seen": sorted(OFF_SCHEMA),
        "n_field_instances": off,
        "handling": "'ambiguous' is not in the pack's declared vocabulary; it is treated as "
                    "'unsure' (excluded from agreement, never coerced to a class)."}

    invalid, unsure_c, unsure_w, quality, conf = [], 0, 0, {}, {}
    cc, ci, cp, wc, wi, wp = [], [], [], [], [], []
    for cid, r in rows_by_clip.items():
        k = key.get(cid)
        if not k:
            invalid.append(f"{cid}: not in clip_key.csv")
            continue
        pid = k["stem"].split("_")[0]
        q = (r.get("audio_quality") or "").strip().lower()
        quality[q or "(blank)"] = quality.get(q or "(blank)", 0) + 1
        cf = (r.get("confidence") or "").strip().lower()
        conf[cf or "(blank)"] = conf.get(cf or "(blank)", 0) + 1

        v = _binarise(r.get("crackles"), CRACKLE_POSITIVE, CRACKLE_NEGATIVE)
        if v == "INVALID":
            invalid.append(f"{cid}: crackles={r.get('crackles')!r}")
        elif v is None and (r.get("crackles") or "").strip().lower() in UNSURE:
            unsure_c += 1
        elif v in (0, 1):
            cc.append(v)
            ci.append(int(k["icbhi_crackle"]))
            cp.append(pid)

        v = _binarise(r.get("wheeze"), WHEEZE_POSITIVE, WHEEZE_NEGATIVE)
        if v == "INVALID":
            invalid.append(f"{cid}: wheeze={r.get('wheeze')!r}")
        elif v is None and (r.get("wheeze") or "").strip().lower() in UNSURE:
            unsure_w += 1
        elif v in (0, 1):
            wc.append(v)
            wi.append(int(k["icbhi_wheeze"]))
            wp.append(pid)

    n_ans = len([1 for r in rows_by_clip.values()
                 if (r.get("crackles") or "").strip() or (r.get("wheeze") or "").strip()])
    doc["coverage"] = {
        "clips_in_pack": len(key), "clips_answered": n_ans,
        "usable_crackle_answers": len(cc), "usable_wheeze_answers": len(wc),
        "unsure_crackles": unsure_c, "unsure_wheeze": unsure_w,
        "unsure_rate_crackles": round(unsure_c / max(n_ans, 1), 4),
        "unsure_rate_wheeze": round(unsure_w / max(n_ans, 1), 4),
        "audio_quality": quality, "confidence": conf,
        "invalid_rows": invalid[:20],
        "note": "'unsure' is excluded from agreement, never coerced to a class.",
    }
    doc["agreement_vs_icbhi"] = {
        "crackles": agreement_block(cc, ci, cp, "crackles", n_boot, seed),
        "wheeze": agreement_block(wc, wi, wp, "wheeze", n_boot, seed),
    }
    doc["intra_rater"] = {
        "crackles": intra_rater(rows_by_clip, key, "crackles",
                                CRACKLE_POSITIVE, CRACKLE_NEGATIVE),
        "wheeze": intra_rater(rows_by_clip, key, "wheeze",
                              WHEEZE_POSITIVE, WHEEZE_NEGATIVE),
    }

    # The ceiling argument, computed rather than asserted.
    ks = {lbl: doc["agreement_vs_icbhi"][lbl].get("cohens_kappa")
          for lbl in ("crackles", "wheeze")}
    doc["ceiling_implication"] = {
        "gate_G2_threshold_auroc": 0.65,
        "gate_G2_observed_auroc": {"crackle_score": 0.5556, "wheeze_score": 0.5818},
        "clinician_kappa_vs_icbhi": ks,
        "published_human_benchmark": {
            "Tzeng_2025_JMIR_AI": "7 senior physicians score 47.77% ICBHI on this corpus",
            "Aviles-Solis_2016": "inter-observer kappa < 0.40 on detailed adventitious sounds"},
        "reading": _ceiling_reading(ks),
    }
    return doc, rows_by_clip


def _ceiling_reading(ks):
    vals = [v for v in ks.values() if v is not None and v == v]
    if not vals:
        return "not estimable from the answers provided"
    k = float(np.mean(vals))
    if k < 0.40:
        return (f"Mean clinician-vs-ICBHI kappa = {k:.3f} ({_kappa_words(k)}). A trained "
                "listener and the ICBHI annotation disagree substantially on the SAME "
                "cycles. This does NOT bound what a machine can do: N11 fits a supervised "
                "probe on a frozen AudioSet embedding - a network that never saw a "
                "respiratory corpus - and reaches AUROC 0.71 (crackle) / 0.76 (wheeze) on "
                "the same test cycles, well above the 0.65 gate. The finding is therefore "
                "the sharper one: models reproduce these labels far better than a trained "
                "physician agrees with them. Our G2 extractors were weak; the reference "
                "standard was not the binding constraint.")
    if k < 0.60:
        return (f"Mean clinician-vs-ICBHI kappa = {k:.3f} ({_kappa_words(k)}). The labels "
                "carry real but imperfect signal; report the extractor result against this "
                "ceiling rather than against 1.0.")
    return (f"Mean clinician-vs-ICBHI kappa = {k:.3f} ({_kappa_words(k)}). The reference "
            "standard is reasonably reliable, so the G2 failure cannot be blamed on the "
            "labels - the extractors are genuinely weak. Report that plainly.")


def selftest():
    """Validate the maths on a synthetic sheet. Writes NOTHING to results/.

    Exists so the pipeline is proven correct before real answers arrive - not to stand in
    for them. A rater who agrees with ICBHI on 90% of clips must yield a high kappa; one
    who answers at random must yield a kappa near zero.
    """
    C.banner("N9 selftest", "synthetic sheet - validates the statistics, writes nothing")
    key = read_key(DEFAULT_KEY)
    rng = np.random.default_rng(0)

    def sheet(agree_prob):
        out = []
        for cid, k in key.items():
            ic, iw = int(k["icbhi_crackle"]), int(k["icbhi_wheeze"])
            c = ic if rng.random() < agree_prob else 1 - ic
            w = iw if rng.random() < agree_prob else 1 - iw
            out.append({"clip_id": cid,
                        "crackles": "fine" if c else "none",
                        "wheeze": "wheeze" if w else "none",
                        "confidence": "high", "audio_quality": "ok", "notes": ""})
        return out

    hi, _ = analyse(sheet(0.90), key, n_boot=200)
    lo, _ = analyse(sheet(0.50), key, n_boot=200)
    kh = hi["agreement_vs_icbhi"]["crackles"]["cohens_kappa"]
    kl = lo["agreement_vs_icbhi"]["crackles"]["cohens_kappa"]
    print(f"  90%-agreeing rater -> kappa {kh:.3f} ({_kappa_words(kh)})")
    print(f"  random rater       -> kappa {kl:.3f} ({_kappa_words(kl)})")
    assert kh > 0.7, f"a 90%-agreeing rater must give high kappa, got {kh}"
    assert abs(kl) < 0.25, f"a random rater must give kappa near zero, got {kl}"
    assert hi["coverage"]["usable_crackle_answers"] == len(key)
    print("\n  OK - agreement, kappa and coverage behave correctly.")
    print("  This proves the pipeline; it is NOT a result and nothing was written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    C.banner("N9 - Clinician Label Reliability",
             "clinician vs ICBHI | intra-rater | the ceiling on any extractor")

    if not os.path.isfile(args.key):
        return C.blocked(EXP_ID, f"clip key not found: {args.key}",
                         ["the pack's clip_key.csv maps clip_id -> ICBHI cycle; without it "
                          "an answer sheet cannot be scored"])
    key = read_key(args.key)

    rows, n_answered, src = read_labels(args.labels, args.xlsx)
    if rows is None:
        return C.blocked(EXP_ID, "no clinician label file found",
                         [f"expected {args.labels} or {args.xlsx}"])

    print(f"  key    : {args.key} ({len(key)} clips, "
          f"{len({r['stem'].split('_')[0] for r in key.values()})} patients, "
          f"{sum(1 for r in key.values() if r['is_duplicate_of'])} duplicate pairs)")
    print(f"  labels : {args.labels if src == 'csv' else args.xlsx} "
          f"({src}) - {n_answered}/{len(key)} clips answered")

    if n_answered == 0:
        return C.blocked(
            EXP_ID,
            "the clinician sheet is the BLANK TEMPLATE - 0 of "
            f"{len(key)} clips carry an answer",
            ["the returned sheet from the clinician, with the `crackles` and `wheeze` "
             "columns filled in",
             "allowed crackles: none | fine | coarse | both | unsure",
             "allowed wheeze:   none | wheeze | rhonchi | both | unsure",
             "drop it at " + args.labels + " (or the .xlsx) and re-run this script",
             "NOTHING is computed from an empty sheet - no result is invented"])

    doc, rows_by_clip = analyse(rows, key, args.n_boot, args.seed)
    doc["inputs"] = {"labels_file": args.labels if src == "csv" else args.xlsx,
                     "labels_format": src, "key_file": args.key}

    cov = doc["coverage"]
    print(f"\n  usable answers: crackles {cov['usable_crackle_answers']} | "
          f"wheeze {cov['usable_wheeze_answers']}")
    print(f"  unsure rate   : crackles {cov['unsure_rate_crackles']:.1%} | "
          f"wheeze {cov['unsure_rate_wheeze']:.1%}")
    if cov["invalid_rows"]:
        print(f"  [warn] {len(cov['invalid_rows'])} unparseable answers: "
              f"{cov['invalid_rows'][:3]}")

    print("\n  -- clinician vs ICBHI")
    for lbl in ("crackles", "wheeze"):
        a = doc["agreement_vs_icbhi"][lbl]
        if a.get("n", 0) == 0:
            print(f"    {lbl:9s} {a['note']}")
            continue
        print(f"    {lbl:9s} kappa {a['cohens_kappa']:+.4f} "
              f"CI{a['kappa_ci95_patient_bootstrap']} ({a['kappa_interpretation']}) | "
              f"agree {a['raw_agreement']:.3f} | Se {a['sensitivity_vs_icbhi']:.3f} "
              f"Sp {a['specificity_vs_icbhi']:.3f}  n={a['n']}")

    print("\n  -- intra-rater (repeated clips)")
    for lbl in ("crackles", "wheeze"):
        ir = doc["intra_rater"][lbl]
        if ir.get("n_pairs", 0) == 0:
            print(f"    {lbl:9s} {ir['note']}")
        else:
            print(f"    {lbl:9s} kappa {ir['cohens_kappa']:+.4f} "
                  f"({ir['interpretation']}) on {ir['n_pairs']} repeated clips")

    print(f"\n  -- ceiling implication\n     {doc['ceiling_implication']['reading']}")

    # Sensitivity: restrict to clips the rater called "ok". If the two agree, audio quality
    # is not driving the headline; if they diverge, that divergence is itself a finding.
    sens, _ = analyse(rows, key, args.n_boot, args.seed, quality_filter=GOOD_QUALITY)
    doc["sensitivity_good_quality_only"] = {
        "n_crackle": sens["coverage"]["usable_crackle_answers"],
        "n_wheeze": sens["coverage"]["usable_wheeze_answers"],
        "crackles_kappa": sens["agreement_vs_icbhi"]["crackles"].get("cohens_kappa"),
        "wheeze_kappa": sens["agreement_vs_icbhi"]["wheeze"].get("cohens_kappa"),
        "note": "clips with audio_quality == 'ok' only"}
    print("\n  -- sensitivity: 'ok' audio only")
    print(f"    crackles kappa {sens['agreement_vs_icbhi']['crackles'].get('cohens_kappa')} | "
          f"wheeze kappa {sens['agreement_vs_icbhi']['wheeze'].get('cohens_kappa')}")

    rel_dir = os.path.join(C.HERE, "release_annotations")
    rel_csv, rel_n = write_release_annotations(rows_by_clip, key, doc, rel_dir)
    doc["release_asset"] = {
        "path": rel_csv, "rows": rel_n, "readme": os.path.join(rel_dir, "README.md"),
        "why": "Gate G0 in the roadmap named the annotated subset as the project's ceiling "
               "lever - the first fine-grained concept annotations released for ICBHI."}
    print(f"\n  release asset: {rel_n} annotated cycles -> release_annotations/")

    n_rows, n_pat = write_corrections(rows_by_clip, key, CORRECTIONS_OUT)
    doc["corrections_written"] = {
        "path": CORRECTIONS_OUT, "rows": n_rows, "patients": n_pat,
        "concepts": ["crackle_presence", "wheeze_presence"],
        "limitation": "each patient contributes only the 1-3 cycles sampled into the pack, "
                      "out of ~60 in their record - a sparse correction, not a re-labelling",
        "next": "re-run N7; it reads this file automatically and drops the SIMULATED label"}
    print(f"\n  wrote {n_rows} corrections for {n_pat} patients -> "
          f"{os.path.basename(CORRECTIONS_OUT)}")
    print("  now re-run:  python N7_clinician_intervention.py --features M2_features.npy")

    C.save_result(EXP_ID, doc)
    return doc


if __name__ == "__main__":
    main()
