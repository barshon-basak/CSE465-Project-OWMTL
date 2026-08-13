#!/usr/bin/env python3
"""
Feasibility gate for ANY device-stratified experiment on ICBHI.

Why this exists
---------------
Both strategy documents lean on ICBHI's recording-device structure:

  - `Novelty_Reassessment.md` §4.3 makes leave-one-device-out the core of its
    proposed benchmark ("uses ICBHI's 7 stethoscopes as a controlled covariate axis").
  - `New_Directions_Search.md` §6 keeps device stratification as experiment 6 of the
    currently-selected ACBD direction ("device-stratified + external-set evaluation").

Nobody has checked whether that is actually possible. Three assumptions have to hold, and
if any fails the experiment is not merely harder -- it is uninterpretable:

  1. There are enough devices to hold one out.
  2. Device is NOT confounded with diagnosis. If (say) COPD patients were recorded almost
     entirely on one stethoscope, then "leave-one-device-out" silently also holds out a
     disease, and any effect you measure is a disease effect wearing a device costume.
  3. Patients span multiple devices. If each patient was recorded on exactly one device,
     then holding out a device also holds out those specific patients, and device shift
     is inseparable from inter-patient variation.

Assumption 3 is the subtle one and the most likely to fail.

Also verifies a factual claim currently in the repo: `Research_Progress_Report.md` states
"7 different stethoscopes (Meditron, LittC2SE, Litt3200, AKGC417L, etc.)". ICBHI 2017
documents FOUR recording devices; 7 is the number of chest LOCATIONS. This script reports
what is actually in the filenames so the claim can be corrected from data.

ICBHI filename format
---------------------
    <patient>_<recording_idx>_<chest_location>_<acquisition_mode>_<equipment>.wav
e.g. 101_1b1_Al_sc_Meditron.wav

Usage
-----
    python3 check_device_structure.py --data-root /content/.../audio_and_txt_files \
                                      --diagnosis /content/.../patient_diagnosis.csv

Reads filenames and the diagnosis CSV only -- no audio is loaded, so it runs in ~1 second.
"""
import argparse
import collections
import glob
import json
import os
import sys

KNOWN = ["COPD", "Healthy", "URTI"]
UNKNOWN = ["Bronchiectasis", "Pneumonia", "Bronchiolitis"]


def parse_files(data_root):
    rows = []
    for wav in sorted(glob.glob(os.path.join(data_root, "*.wav"))):
        parts = os.path.splitext(os.path.basename(wav))[0].split("_")
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append(dict(patient_id=pid, rec_idx=parts[1], chest_location=parts[2],
                         acquisition_mode=parts[3], equipment="_".join(parts[4:])))
    return rows


def load_diagnoses(path):
    diag = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",") if "," in line else line.split()
            if len(p) < 2:
                continue
            try:
                diag[int(p[0])] = p[1].strip()
            except ValueError:
                continue
    return diag


def group_of(dx):
    return "known" if dx in KNOWN else "unknown" if dx in UNKNOWN else "excluded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--diagnosis", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = parse_files(a.data_root)
    if not rows:
        sys.exit(f"No parseable ICBHI filenames under {a.data_root}")
    diag = load_diagnoses(a.diagnosis)
    for r in rows:
        r["diagnosis"] = diag.get(r["patient_id"], "UNKNOWN_LABEL")
        r["group"] = group_of(r["diagnosis"])

    verdicts = {}
    print("=" * 78)
    print("ICBHI DEVICE-STRUCTURE FEASIBILITY GATE")
    print("=" * 78)
    print(f"Recordings parsed : {len(rows)}")
    print(f"Patients          : {len({r['patient_id'] for r in rows})}")

    # ---------------- 1. what the filenames actually contain ----------------
    dev_counts = collections.Counter(r["equipment"] for r in rows)
    loc_counts = collections.Counter(r["chest_location"] for r in rows)
    print(f"\n--- 1. Devices (equipment field) ---")
    print(f"DISTINCT DEVICES : {len(dev_counts)}")
    for d, n in dev_counts.most_common():
        print(f"   {d:<14} {n:>5} recordings  ({100 * n / len(rows):5.1f}%)")
    print(f"\nDISTINCT CHEST LOCATIONS : {len(loc_counts)}   "
          f"({', '.join(sorted(loc_counts))})")

    claim_ok = len(dev_counts) == 7
    verdicts["device_count_matches_repo_claim_of_7"] = claim_ok
    print(f"\n  Research_Progress_Report.md claims 7 stethoscopes. Actual: {len(dev_counts)}.")
    if not claim_ok:
        print(f"  >> CLAIM IS WRONG. Correct the report. Note chest locations = "
              f"{len(loc_counts)}, which is almost certainly where '7' came from.")

    # ---------------- 2. device x diagnosis confound ----------------
    print("\n--- 2. Device x diagnosis confound (the make-or-break check) ---")
    dev_dx = collections.defaultdict(collections.Counter)
    for r in rows:
        dev_dx[r["equipment"]][r["diagnosis"]] += 1

    all_dx = sorted({r["diagnosis"] for r in rows})
    w = max(len(d) for d in dev_counts) + 2
    print(f"{'device':<{w}}" + "".join(f"{d[:11]:>13}" for d in all_dx))
    for dev in sorted(dev_counts):
        line = f"{dev:<{w}}"
        for dx in all_dx:
            n = dev_dx[dev][dx]
            pct = 100 * n / dev_counts[dev]
            line += f"{n:>6} ({pct:3.0f}%)"
        print(line)

    # A device whose recordings are >=80% one diagnosis cannot be held out cleanly.
    skewed = []
    for dev, cnt in dev_counts.items():
        top_dx, top_n = dev_dx[dev].most_common(1)[0]
        if top_n / cnt >= 0.80:
            skewed.append((dev, top_dx, top_n / cnt))
    verdicts["device_diagnosis_confounded"] = bool(skewed)
    if skewed:
        print("\n  >> CONFOUNDED. These devices are dominated by a single diagnosis:")
        for dev, dx, frac in skewed:
            print(f"       {dev}: {100 * frac:.0f}% {dx}")
        print("     Holding one of these out also holds out a disease. A leave-one-device-out")
        print("     result would NOT isolate covariate shift.")
    else:
        print("\n  >> OK. No device is dominated by one diagnosis at the 80% threshold.")

    # ---------------- 3. do patients span multiple devices? ----------------
    print("\n--- 3. Patient x device (can device shift be separated from patient shift?) ---")
    pat_dev = collections.defaultdict(set)
    for r in rows:
        pat_dev[r["patient_id"]].add(r["equipment"])
    spread = collections.Counter(len(v) for v in pat_dev.values())
    multi = sum(n for k, n in spread.items() if k > 1)
    print(f"  patients on exactly 1 device : {spread.get(1, 0)}")
    for k in sorted(x for x in spread if x > 1):
        print(f"  patients on {k} devices        : {spread[k]}")
    frac_multi = multi / len(pat_dev)
    verdicts["patients_span_multiple_devices"] = frac_multi > 0.10
    print(f"\n  {multi}/{len(pat_dev)} patients ({100 * frac_multi:.1f}%) appear on >1 device.")
    if frac_multi <= 0.10:
        print("  >> BLOCKING. Device is essentially a patient-level property here, so")
        print("     leave-one-device-out == leave-those-patients-out. Any 'device effect'")
        print("     you measure is confounded with inter-patient variation and cannot be")
        print("     attributed to the microphone. A within-patient device comparison is the")
        print("     only clean design, and there is not enough data for it.")
    else:
        print("  >> USABLE. Enough patients recorded on multiple devices to support a")
        print("     within-patient device contrast (the clean design).")

    # ---------------- 4. is leave-one-device-out actually runnable? ----------------
    print("\n--- 4. Leave-one-device-out viability (needs known AND unknown per fold) ---")
    ok_folds = []
    for dev in sorted(dev_counts):
        pats = {r["patient_id"] for r in rows if r["equipment"] == dev}
        g = collections.Counter(group_of(diag.get(p, "")) for p in pats)
        usable = g["known"] >= 5 and g["unknown"] >= 3
        ok_folds.append(usable)
        print(f"  {dev:<14} patients: {len(pats):>3}  known={g['known']:>3} "
              f"unknown={g['unknown']:>3} excluded={g['excluded']:>2}   "
              f"{'usable fold' if usable else 'TOO SMALL'}")
    verdicts["enough_usable_folds"] = sum(ok_folds) >= 3
    print(f"\n  Usable folds: {sum(ok_folds)} of {len(ok_folds)} (need >= 3).")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    blocking = []
    if verdicts["device_diagnosis_confounded"]:
        blocking.append("device is confounded with diagnosis")
    if not verdicts["patients_span_multiple_devices"]:
        blocking.append("patients do not span devices")
    if not verdicts["enough_usable_folds"]:
        blocking.append("too few usable leave-one-device-out folds")

    if blocking:
        print("DEVICE-STRATIFIED EXPERIMENTS ARE NOT CLEANLY FEASIBLE ON ICBHI.")
        print("Blocking issues: " + "; ".join(blocking))
        print("\nWhat to do instead: drop device stratification from the experiment plan and")
        print("use the external sets (SPRSound / Coswara) as the ONLY shift axis, labelled")
        print("honestly as combined device+population+modality shift rather than as a")
        print("controlled covariate axis. Do not claim a controlled device contrast you")
        print("cannot actually run -- that is the exact overclaim a reviewer checks first.")
    else:
        print("DEVICE-STRATIFIED EXPERIMENTS APPEAR FEASIBLE. Proceed, and report the")
        print("device x diagnosis table above in the paper so reviewers can see the balance.")
    print("=" * 78)

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({
                "n_recordings": len(rows),
                "n_patients": len(pat_dev),
                "devices": dict(dev_counts),
                "chest_locations": dict(loc_counts),
                "device_by_diagnosis": {d: dict(c) for d, c in dev_dx.items()},
                "patients_per_device_count": dict(spread),
                "verdicts": verdicts,
                "blocking_issues": blocking,
            }, f, indent=2)
        print(f"\nWrote {a.json_out}")


if __name__ == "__main__":
    main()
