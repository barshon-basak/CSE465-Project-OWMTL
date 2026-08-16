#!/usr/bin/env python3
"""
The ICBHI 2017 official train/test split -- loader, leakage audit, and the project's
patient-independent variant.

Import this from notebooks (`from official_split import load_split`) or run it standalone
(`python3 official_split.py`) to print the audit.

WHY THIS FILE EXISTS
--------------------
Two problems were found on 2026-08-14 and both are addressed here.

1. Every model in Asif's workstream reported `split_method: patient_independent_official_60_40`
   while actually running a fallback rule (`patient_id <= 111 -> test`) that yields 11 test
   patients and 492 of 6898 cycles -- 7.1% of the data, not 40%. The official split file simply
   was not present in the Colab runtime, and the fallback was silent. It is now committed at
   `Asif's/ICBHI_challenge_train_test.txt`.

2. The official split is **not patient-independent.** It assigns RECORDINGS, not patients, and
   two patients have recordings on both sides:

       patient 156 -> 9 train / 8 test recordings
       patient 218 -> 4 train / 4 test recordings

   That directly conflicts with `Model_Training_Protocol.md` section 1, which calls
   patient-independent splits "a research validity requirement, not a style choice."

THE PROJECT'S POLICY
--------------------
Default mode is `"patient_independent"`: take the official split, then move every recording of a
leaking patient to TRAIN. Moving to train (rather than test) is the conservative direction -- the
test set is then guaranteed to contain no patient the model saw while training.

Impact is small and, if anything, improves the balance:

    official            539 train / 381 test   (58.6% / 41.4%)   2 patients leak
    patient_independent 551 train / 369 test   (59.9% / 40.1%)   0 patients leak

The corrected test set differs from the official one by 12 of 381 recordings (3.1%).

Mode `"official"` reproduces the published split verbatim, for a directly comparable number.
Report whichever you use, by name, and say which. Do not call the corrected split "official".
"""
import argparse
import collections
import os

DEFAULT_SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                  "ICBHI_challenge_train_test.txt")

SEARCH_NAMES = ("ICBHI_challenge_train_test.txt", "ICBHI_challenge_train_test.txt".lower(),
                "*train_test*.txt")


def find_split_file(extra_dirs=()):
    """Locate the official split file. Checks the committed repo copy first -- it is the one
    artifact guaranteed to exist regardless of which Kaggle mirror the audio came from."""
    import glob
    cands = [os.path.normpath(DEFAULT_SPLIT_FILE)]
    for d in list(extra_dirs) + ["/content", "/kaggle/input", "/kaggle/working", ".", "./data"]:
        if not d or not os.path.isdir(d):
            continue
        for name in SEARCH_NAMES:
            cands += sorted(glob.glob(os.path.join(d, "**", name), recursive=True))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def _parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1].lower() in ("train", "test"):
                rows.append((parts[0].replace(".wav", ""), parts[1].lower()))
    if not rows:
        raise ValueError(f"No usable rows parsed from {path}")
    return rows


def audit(path=None):
    """Return a dict describing the official split and its patient leakage."""
    path = path or find_split_file()
    if not path:
        raise FileNotFoundError("ICBHI_challenge_train_test.txt not found.")
    rows = _parse(path)

    by_patient = collections.defaultdict(set)
    for stem, split in rows:
        by_patient[int(stem.split("_")[0])].add(split)
    leaking = sorted(p for p, v in by_patient.items() if len(v) > 1)

    counts = collections.Counter(s for _, s in rows)
    leak_detail = {}
    for pid in leaking:
        c = collections.Counter(s for stem, s in rows if int(stem.split("_")[0]) == pid)
        leak_detail[pid] = {"train": c["train"], "test": c["test"]}

    corrected = {stem: ("train" if int(stem.split("_")[0]) in leaking else split)
                 for stem, split in rows}
    cc = collections.Counter(corrected.values())

    return {
        "path": path,
        "n_recordings": len(rows),
        "n_patients": len(by_patient),
        "official": {"train": counts["train"], "test": counts["test"],
                     "test_fraction": round(counts["test"] / len(rows), 4)},
        "patient_independent": {"train": cc["train"], "test": cc["test"],
                                "test_fraction": round(cc["test"] / len(rows), 4)},
        "leaking_patients": leaking,
        "leaking_detail": leak_detail,
        "test_recordings_moved": counts["test"] - cc["test"],
    }


def load_split(path=None, mode="patient_independent"):
    """Return ({stem: 'train'|'test'}, info_dict).

    mode="patient_independent" (default): official split with every recording of a leaking
        patient reassigned to train. Protocol section 1 compliant.
    mode="official": the published split verbatim. NOT patient-independent -- 2 patients appear
        on both sides. Use only when you specifically want the directly-comparable number, and
        say so explicitly wherever you report it.
    """
    if mode not in ("patient_independent", "official"):
        raise ValueError(f"unknown mode {mode!r}")
    info = audit(path)
    rows = _parse(info["path"])
    leaking = set(info["leaking_patients"])

    if mode == "official":
        mapping = dict(rows)
    else:
        mapping = {stem: ("train" if int(stem.split("_")[0]) in leaking else split)
                   for stem, split in rows}

    info = dict(info)
    info["mode"] = mode
    info["is_patient_independent"] = (mode == "patient_independent")
    info["policy"] = (
        "Official ICBHI 2017 split, with every recording of a patient appearing on both sides "
        f"reassigned to TRAIN (patients {sorted(leaking)}). Conservative direction: the test set "
        "contains no patient seen during training. Deviates from the published split by "
        f"{info['test_recordings_moved']} of {info['official']['test']} test recordings "
        f"({info['test_recordings_moved'] / info['official']['test'] * 100:.1f}%)."
        if mode == "patient_independent" else
        "Published ICBHI 2017 split, verbatim. WARNING: not patient-independent -- patients "
        f"{sorted(leaking)} have recordings in BOTH train and test, which violates "
        "Model_Training_Protocol.md section 1. Comparable to published work; do not present it "
        "as patient-independent.")
    return mapping, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    i = audit(a.file)

    print("=" * 76)
    print("ICBHI 2017 OFFICIAL SPLIT — AUDIT")
    print("=" * 76)
    print(f"File       : {i['path']}")
    print(f"Recordings : {i['n_recordings']}   Patients: {i['n_patients']}")
    print()
    print(f"{'mode':<22}{'train':>8}{'test':>8}{'test %':>9}   patient-independent?")
    print("-" * 76)
    o, p = i["official"], i["patient_independent"]
    print(f"{'official (published)':<22}{o['train']:>8}{o['test']:>8}"
          f"{o['test_fraction'] * 100:>8.1f}%   NO — {len(i['leaking_patients'])} patient(s) leak")
    print(f"{'patient_independent':<22}{p['train']:>8}{p['test']:>8}"
          f"{p['test_fraction'] * 100:>8.1f}%   YES")
    print()
    print("Leaking patients in the published split:")
    for pid, d in i["leaking_detail"].items():
        print(f"  patient {pid}: {d['train']} train + {d['test']} test recordings")
    print(f"\nCorrection moves {i['test_recordings_moved']} of {o['test']} test recordings "
          f"({i['test_recordings_moved'] / o['test'] * 100:.1f}%) into train.")
    print()
    print("=" * 76)
    print("WHAT THIS MEANS")
    print("=" * 76)
    print("The published ICBHI split assigns RECORDINGS, not patients, so it is not")
    print("patient-independent. Model_Training_Protocol.md section 1 calls patient independence")
    print("'a research validity requirement, not a style choice' -- so the two cannot both be")
    print("satisfied verbatim.")
    print()
    print("Project policy: default to `patient_independent`. It costs 3.1% of the test")
    print("recordings, lands at 59.9/40.1 (closer to nominal 60/40 than the published split),")
    print("and is protocol compliant. Report `official` alongside it only when a directly")
    print("comparable number is needed -- and never describe it as patient-independent.")
    print("=" * 76)


if __name__ == "__main__":
    main()
