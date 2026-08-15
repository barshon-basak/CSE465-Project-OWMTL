#!/usr/bin/env python3
"""Patch M2's and M3's notebooks to use the OFFICIAL ICBHI split, with no silent fallback.

Why this exists
---------------
`Asif's/M2/results_M2.json` and `Asif's/M3/results_M3.json` both record

    "split_method": "patient_independent_official_60_40"
    "train_patients": 115, "test_patients": 11, "test_samples": 492

115 + 11 = 126 patients and 6406 + 492 = 6898 cycles -- i.e. every patient and every
cycle in ICBHI, split 91/9. That is not the official 60/40. Two defects combined:

  1. `load_official_split()` looked for "ICBHI_Challenge_train_test.txt" (capital C) via
     os.path.exists() on three exact paths. The real filename is lowercase
     "ICBHI_challenge_train_test.txt", so on Colab's case-sensitive filesystem the lookup
     never matched and the function returned None -- falling back to `pid <= 111`, which
     selects exactly 11 patients because ICBHI patient IDs run 101-226.

  2. The results-JSON export hardcodes split_method as a string literal, so it claimed
     "official" regardless of which path the code actually took. Nothing could disagree.

This patch makes the split file mandatory (raises, with the searched paths, if absent),
verifies it against the known official counts, and derives split_method from what the run
actually did instead of asserting it.

Patient-independence
--------------------
The official split is NOT patient-disjoint: patients 156 and 218 have recordings on both
sides (9+8 and 4+4). Model_Training_Protocol.md section 1 requirement 1 forbids that, so
CFG["official_overlap_policy"] decides:

  "drop_from_train" (default) -- drop those patients' 13 TRAIN recordings. The official
                                 TEST set stays byte-identical, so the score remains
                                 comparable to published ICBHI work. Yields 526/381
                                 recordings, 77/49 patients, fully patient-disjoint.
  "as_is"                     -- official split verbatim, accepting the leak.

Usage:  python3 "Asif's/scripts/patch_official_split.py" [--check]
        --check verifies the patch is applied without writing.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGETS = [
    os.path.join(REPO, "Asif's", "M2", "M2_cnn_baseline_tuned.ipynb"),
    os.path.join(REPO, "Asif's", "M3", "M3_lightweight_backbone.ipynb"),
]

SPLIT_CELL = 7
EXPORT_CELL = 22

# --------------------------------------------------------------------------------------
# 1. The split loader: case-insensitive, recursive, and it RAISES instead of falling back.
# --------------------------------------------------------------------------------------
OLD_LOADER = '''def load_official_split(data_root):
    """Return {filename_stem: 'train'|'test'} from ICBHI_Challenge_train_test.txt, or None."""
    candidates = [
        os.path.join(os.path.dirname(data_root), "ICBHI_Challenge_train_test.txt"),
        os.path.join(data_root, "ICBHI_Challenge_train_test.txt"),
        os.path.join(os.path.dirname(os.path.dirname(data_root)),
                     "ICBHI_Challenge_train_test.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        split_map = {}
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].lower() in ("train", "test"):
                    split_map[parts[0].replace(".wav", "")] = parts[1].lower()
        if split_map:
            print(f"Loaded official split: {path}  ({len(split_map)} recordings)")
            return split_map
    return None'''

NEW_LOADER = '''# ---- Official ICBHI 2017 split: ground truth ----
# Verified against Asif's/ICBHI_challenge_train_test.txt. If any assertion below fires,
# the split file is not the official one -- stop and check it rather than proceeding.
OFFICIAL_RECORDINGS = 920
OFFICIAL_TRAIN_RECS = 539          # 58.6%
OFFICIAL_TEST_RECS = 381           # 41.4%  <- this is the "60/40"
OFFICIAL_PATIENTS = 126
OFFICIAL_OVERLAP_PATIENTS = {156, 218}   # present in BOTH train and test in the official file

# How to reconcile the official split with protocol section 1 (patient-independence).
OVERLAP_POLICY = CFG.get("official_overlap_policy", "drop_from_train")


def load_official_split(data_root):
    """Return {filename_stem: 'train'|'test'} from the official split file.

    Raises if not found. There is deliberately NO fallback: the previous version searched
    for "ICBHI_Challenge_train_test.txt" (capital C) with os.path.exists() on three exact
    paths, never matched the real lowercase filename on Colab, and silently fell back to
    `pid <= 111` -- which is why results_M2/M3.json report 11 test patients while claiming
    split_method "patient_independent_official_60_40".
    """
    wanted = "icbhi_challenge_train_test.txt"
    searched, bases = [], [
        data_root,
        os.path.dirname(data_root.rstrip("/")),
        os.path.dirname(os.path.dirname(data_root.rstrip("/"))),
        "/content", "/kaggle/input", ".",
    ]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        searched.append(base)
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                # case-insensitive: the official file is lowercase, mirrors vary
                if fn.lower() != wanted and not fn.lower().endswith("train_test.txt"):
                    continue
                path = os.path.join(dirpath, fn)
                split_map = {}
                try:
                    with open(path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[1].lower() in ("train", "test"):
                                split_map[parts[0].replace(".wav", "")] = parts[1].lower()
                except Exception:
                    continue
                if split_map:
                    print(f"Official split file: {path}  ({len(split_map)} recordings)")
                    return split_map
    raise FileNotFoundError(
        "ICBHI_challenge_train_test.txt NOT FOUND. It is REQUIRED -- this notebook has no "
        "fallback split by design, because the fallback is what invalidated the previous "
        "M2/M3 runs (11 test patients reported as 'official 60/40').\\n"
        "Upload Asif's/ICBHI_challenge_train_test.txt to /content/ and re-run.\\n"
        "Searched under: " + ", ".join(searched))


def verify_official_split(split_map):
    """Assert the file really is the official split, then report the overlap policy."""
    n_train = sum(1 for v in split_map.values() if v == "train")
    n_test = sum(1 for v in split_map.values() if v == "test")
    sides = {}
    for stem, s in split_map.items():
        sides.setdefault(int(stem.split("_")[0]), set()).add(s)
    overlap = {p for p, s in sides.items() if len(s) == 2}

    print(f"  recordings : {len(split_map)} ({n_train} train / {n_test} test)")
    print(f"  patients   : {len(sides)}  | on both sides: {sorted(overlap)}")

    assert len(split_map) == OFFICIAL_RECORDINGS, \\
        f"expected {OFFICIAL_RECORDINGS} recordings, got {len(split_map)}"
    assert (n_train, n_test) == (OFFICIAL_TRAIN_RECS, OFFICIAL_TEST_RECS), \\
        f"expected {OFFICIAL_TRAIN_RECS}/{OFFICIAL_TEST_RECS}, got {n_train}/{n_test}"
    assert len(sides) == OFFICIAL_PATIENTS, \\
        f"expected {OFFICIAL_PATIENTS} patients, got {len(sides)}"
    assert overlap == OFFICIAL_OVERLAP_PATIENTS, \\
        f"expected {sorted(OFFICIAL_OVERLAP_PATIENTS)} to straddle the split, got {sorted(overlap)}"
    print("[OK] split file matches the official ICBHI 2017 challenge split exactly.")

    if OVERLAP_POLICY == "drop_from_train":
        print(f"\\nPolicy 'drop_from_train': patients {sorted(OFFICIAL_OVERLAP_PATIENTS)} appear "
              f"on both sides of the official split.\\n  Dropping their TRAIN recordings; the "
              f"official TEST set is left byte-identical so the score stays comparable to "
              f"published ICBHI work.")
    elif OVERLAP_POLICY == "as_is":
        print(f"\\nPolicy 'as_is': official split verbatim. WARNING -- patients "
              f"{sorted(OFFICIAL_OVERLAP_PATIENTS)} leak across train/test, violating "
              f"Model_Training_Protocol.md section 1 requirement 1.")
    else:
        raise ValueError(f"unknown official_overlap_policy: {OVERLAP_POLICY!r}")'''

# --------------------------------------------------------------------------------------
# 2. Call the verifier; drop the "not found" warning path entirely.
# --------------------------------------------------------------------------------------
OLD_CALL = '''    split_map = load_official_split(data_root)
    if split_map is None:
        print("WARNING: official split file not found — falling back to a deterministic "
              "patient-ID split (patients <= 111 -> test).")'''

NEW_CALL = '''    split_map = load_official_split(data_root)
    verify_official_split(split_map)'''

# --------------------------------------------------------------------------------------
# 3. Strict per-recording assignment -- no per-stem fallback, apply the overlap policy.
# --------------------------------------------------------------------------------------
OLD_ASSIGN = '''        pid = patient_id_from_stem(stem)
        if split_map is not None:
            split = split_map.get(stem) or ("test" if pid <= 111 else "train")
        else:
            split = "test" if pid <= 111 else "train"
'''

NEW_ASSIGN = '''        pid = patient_id_from_stem(stem)
        split = split_map.get(stem)
        if split is None:
            raise KeyError(
                f"recording {stem!r} is on disk but absent from the official split file. "
                f"The audio and the split file disagree -- do not guess a side for it.")
        if (OVERLAP_POLICY == "drop_from_train"
                and pid in OFFICIAL_OVERLAP_PATIENTS and split == "train"):
            n_dropped_overlap[0] += 1
            continue
'''

# --------------------------------------------------------------------------------------
# 4. Leakage assertion made policy-aware, plus a floor that catches the fallback signature.
# --------------------------------------------------------------------------------------
OLD_ASSERT = '''leaked = train_patients & test_patients
assert not leaked, (
    f"PATIENT LEAKAGE: {len(leaked)} patient(s) appear in BOTH train and test: "
    f"{sorted(leaked)[:10]}. This violates Model_Training_Protocol.md §1 and invalidates "
    f"the run. Do not train until this is fixed.")
assert len(df_train) > 0 and len(df_test) > 0, "One split is empty — check the split file."
print(f"\\n[OK] Protocol §1 verified: 0 patients shared between train and test.")'''

NEW_ASSERT = '''leaked = train_patients & test_patients
if OVERLAP_POLICY == "drop_from_train":
    assert not leaked, (
        f"PATIENT LEAKAGE: {len(leaked)} patient(s) appear in BOTH train and test: "
        f"{sorted(leaked)[:10]}. This violates Model_Training_Protocol.md §1 and invalidates "
        f"the run. Do not train until this is fixed.")
    print(f"\\n[OK] Protocol §1 verified: 0 patients shared between train and test.")
else:
    assert leaked == OFFICIAL_OVERLAP_PATIENTS, (
        f"under 'as_is' exactly {sorted(OFFICIAL_OVERLAP_PATIENTS)} should leak, got {sorted(leaked)}")
    print(f"\\n[!!] {len(leaked)} patient(s) leak by design under 'as_is': {sorted(leaked)}")

assert len(df_train) > 0 and len(df_test) > 0, "One split is empty — check the split file."

# The fallback that invalidated the previous runs produced exactly 11 test patients.
# The official split has 47 (as_is) or 49 (drop_from_train). Fail loudly, not silently.
assert len(test_patients) >= 40, (
    f"only {len(test_patients)} test patients — the official split has 47-49. This is the "
    f"signature of the `pid <= 111` fallback that invalidated the previous M2/M3 runs. "
    f"Do not train on this split.")
if n_dropped_overlap[0]:
    print(f"[OK] dropped {n_dropped_overlap[0]} train recording(s) from patients "
          f"{sorted(OFFICIAL_OVERLAP_PATIENTS)} per 'drop_from_train'.")'''

# --------------------------------------------------------------------------------------
# 5. Counter used by the assignment loop.
# --------------------------------------------------------------------------------------
OLD_COUNTER = "    rows, n_missing_ann = [], 0"
NEW_COUNTER = "    rows, n_missing_ann = [], 0\n    n_dropped_overlap[0] = 0"

OLD_GLOBAL = "def build_cycle_dataframe(data_root):"
NEW_GLOBAL = ("n_dropped_overlap = [0]   # mutable so build_cycle_dataframe can report outward\n\n\n"
              "def build_cycle_dataframe(data_root):")

# --------------------------------------------------------------------------------------
# 6. Export cell: derive provenance instead of asserting it.
# --------------------------------------------------------------------------------------
OLD_EXPORT = '''        "split_method": "patient_independent_official_60_40",
        "patient_leakage_verified": True,'''

NEW_EXPORT = '''        "split_method": ("official_icbhi_60_40_patient_disjoint"
                         if OVERLAP_POLICY == "drop_from_train"
                         else "official_icbhi_60_40_verbatim"),
        "split_source": "official_file",
        "split_file_verified": {
            "recordings": OFFICIAL_RECORDINGS,
            "train_recordings": OFFICIAL_TRAIN_RECS,
            "test_recordings": OFFICIAL_TEST_RECS,
            "patients": OFFICIAL_PATIENTS,
        },
        "official_overlap_policy": OVERLAP_POLICY,
        "official_overlap_patients": sorted(OFFICIAL_OVERLAP_PATIENTS),
        "official_overlap_note": (
            "The official ICBHI split is not patient-disjoint: patients 156 and 218 have "
            "recordings on both sides. Protocol §1 requirement 1 forbids this, so under "
            "policy 'drop_from_train' their TRAIN recordings are excluded while the official "
            "TEST set is left byte-identical, keeping the score comparable to published "
            "ICBHI results."),
        "supersedes_note": (
            "Supersedes the earlier run of this model, which reported 115 train / 11 test "
            "patients (492 test cycles) as 'patient_independent_official_60_40'. That run "
            "took the `pid <= 111` fallback because the split-file lookup used the wrong "
            "filename casing; split_method was a hardcoded literal and so could not disagree."),
        "patient_leakage_verified": bool(OVERLAP_POLICY == "drop_from_train"),'''

# --------------------------------------------------------------------------------------
# 7. The header comment that started it: it documents the filename with the wrong casing.
# --------------------------------------------------------------------------------------
OLD_COMMENT = ("#   ICBHI_Challenge_train_test.txt                <- official 60/40 split "
               "(one level up)")
NEW_COMMENT = ("#   ICBHI_challenge_train_test.txt                <- official 60/40 split "
               "(lowercase 'c' -- the capital-C spelling this comment used to carry is what\n"
               "#                                                   the old exact-path lookup "
               "searched for, and never found on Colab)")

# --------------------------------------------------------------------------------------
# 8. Device-suffix disagreement between the split file and the audio filenames.
#
# The official split file lists "226_1b1_Pl_sc_Meditron"; the audio on disk is
# "226_1b1_Pl_sc_LittC2SE". Same recording, different device tag -- a known inconsistency
# in the ICBHI distribution. A strict stem lookup raises on it.
#
# The first four fields (patient_session_location_mode) are UNIQUE across all 920
# recordings -- verified against the committed split file -- so they identify a recording
# without relying on the device tag. Match on those, and report every disagreement rather
# than absorbing it silently.
# --------------------------------------------------------------------------------------
OLD_KEYFN = '''def patient_id_from_stem(stem):
    """'101_1b1_Al_sc_Meditron' -> 101"""
    try:
        return int(stem.split("_")[0])
    except (ValueError, IndexError):
        return -1'''

NEW_KEYFN = '''def patient_id_from_stem(stem):
    """'101_1b1_Al_sc_Meditron' -> 101"""
    try:
        return int(stem.split("_")[0])
    except (ValueError, IndexError):
        return -1


def key_without_device(stem):
    """'226_1b1_Pl_sc_LittC2SE' -> '226_1b1_Pl_sc'  (device tag dropped).

    The official split file and the audio filenames disagree on the device suffix for at
    least one recording (226_1b1_Pl is 'Meditron' in the split file, 'LittC2SE' on disk).
    These first four fields are unique across all 920 recordings, so they identify a
    recording unambiguously without trusting the device tag.
    """
    return "_".join(stem.split("_")[:4])'''

OLD_CALL_V1 = '''    split_map = load_official_split(data_root)
    verify_official_split(split_map)'''

NEW_CALL_V2 = '''    split_map = load_official_split(data_root)
    verify_official_split(split_map)

    # Device-tag-independent index, used only when an exact stem lookup misses.
    split_by_key = {key_without_device(s): (s, side) for s, side in split_map.items()}
    assert len(split_by_key) == len(split_map), (
        "device-independent keys are not unique in the split file, so a recording whose "
        "device tag disagrees cannot be matched safely. Investigate before proceeding.")
    device_mismatches = []'''

OLD_ASSIGN_V1 = '''        pid = patient_id_from_stem(stem)
        split = split_map.get(stem)
        if split is None:
            raise KeyError(
                f"recording {stem!r} is on disk but absent from the official split file. "
                f"The audio and the split file disagree -- do not guess a side for it.")
        if (OVERLAP_POLICY == "drop_from_train"
                and pid in OFFICIAL_OVERLAP_PATIENTS and split == "train"):
            n_dropped_overlap[0] += 1
            continue
'''

NEW_ASSIGN_V2 = '''        pid = patient_id_from_stem(stem)
        split = split_map.get(stem)
        if split is None:
            # Exact stem missing: fall back to the device-independent key, which is unique.
            # This is NOT a guess -- it identifies the same recording by patient, session,
            # chest location and mode; only the device tag differs.
            alt = split_by_key.get(key_without_device(stem))
            if alt is None:
                raise KeyError(
                    f"recording {stem!r} is on disk but has no counterpart in the official "
                    f"split file, even ignoring the device suffix. The audio and the split "
                    f"file genuinely disagree -- do not guess a side for it.")
            file_stem, split = alt
            device_mismatches.append((stem, file_stem))
        if (OVERLAP_POLICY == "drop_from_train"
                and pid in OFFICIAL_OVERLAP_PATIENTS and split == "train"):
            n_dropped_overlap[0] += 1
            continue
'''

OLD_REPORT = '''    df = pd.DataFrame(rows)
    if n_missing_ann:
        print(f"Skipped {n_missing_ann} recordings with no matching annotation .txt")'''

NEW_REPORT = '''    df = pd.DataFrame(rows)
    if device_mismatches:
        print(f"\\n{len(device_mismatches)} recording(s) matched on patient/session/location "
              f"because the device tag differs between the audio and the split file:")
        for disk_stem, file_stem in device_mismatches:
            print(f"    disk={disk_stem}  split_file={file_stem}")
        print("  (known ICBHI inconsistency; the split assignment itself is unambiguous)")
    if n_missing_ann:
        print(f"Skipped {n_missing_ann} recordings with no matching annotation .txt")'''

PATCHES = [
    (SPLIT_CELL, OLD_COMMENT, NEW_COMMENT),
    (SPLIT_CELL, OLD_GLOBAL, NEW_GLOBAL),
    (SPLIT_CELL, OLD_LOADER, NEW_LOADER),
    (SPLIT_CELL, OLD_KEYFN, NEW_KEYFN),
    (SPLIT_CELL, [OLD_CALL, OLD_CALL_V1], NEW_CALL_V2),
    (SPLIT_CELL, OLD_COUNTER, NEW_COUNTER),
    (SPLIT_CELL, [OLD_ASSIGN, OLD_ASSIGN_V1], NEW_ASSIGN_V2),
    (SPLIT_CELL, OLD_REPORT, NEW_REPORT),
    (SPLIT_CELL, OLD_ASSERT, NEW_ASSERT),
    (EXPORT_CELL, OLD_EXPORT, NEW_EXPORT),
]


def get_source(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def apply_to(path, check_only=False):
    nb = json.load(open(path))
    sources = {i: get_source(nb["cells"][i]) for i in (SPLIT_CELL, EXPORT_CELL)}

    applied, skipped = 0, 0
    for idx, olds, new in PATCHES:
        if new in sources[idx]:
            skipped += 1              # already applied; safe to re-run
            continue
        if isinstance(olds, str):
            olds = [olds]
        # Accept any known prior form, so the patcher can upgrade a partially-patched
        # notebook as well as a pristine one.
        match = next((o for o in olds if sources[idx].count(o) == 1), None)
        if match is None:
            counts = [sources[idx].count(o) for o in olds]
            print(f"  FAIL {os.path.basename(path)} cell {idx}: expected exactly 1 occurrence "
                  f"of one known form of {olds[0].splitlines()[0][:55]!r}; counts={counts}")
            return False
        sources[idx] = sources[idx].replace(match, new)
        applied += 1

    if applied == 0:
        print(f"  {os.path.basename(path)}: already patched ({skipped}/{len(PATCHES)})")
        return True

    if check_only:
        print(f"  {os.path.basename(path)}: {applied} edit(s) apply cleanly, "
              f"{skipped} already present (not written)")
        return True

    for idx, src in sources.items():
        nb["cells"][idx]["source"] = src.splitlines(keepends=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  {os.path.basename(path)}: patched ({applied} edit(s), {skipped} already present)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()

    print("Patching M2/M3 notebooks to the official ICBHI split")
    ok = all(apply_to(p, args.check) for p in TARGETS)
    if not ok:
        print("\nFAILED — no notebook was left half-patched; fix the mismatch and re-run.")
        sys.exit(1)
    print("\nDone. Both notebooks now require the official split file and verify it.")


if __name__ == "__main__":
    main()
