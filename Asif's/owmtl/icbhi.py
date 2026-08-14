"""
ICBHI 2017 metadata, cycle index, and the canonical OWMTL split.

Standard-library only, on purpose: every member imports this to get their data
partitions, so it must not drag in librosa/torch/pandas just to answer
"which patients are in my test set?". Audio decoding lives in `features.py`.

The two artifacts this module produces are the group's shared contract:

    cycles_index_v1.csv   one row per annotated respiratory cycle
    split_v1.json         patient -> partition assignment, frozen and committed

Both are deterministic. Regenerating them from the same ICBHI copy with the
same seed reproduces byte-identical files.


Why a custom split at all
-------------------------
The ICBHI 2017 official 60/40 challenge split is patient-independent and is what
published numbers are measured on, so we keep it and report against it.

But it is *not* safe for this project. Member B's open-world claim rests on 19
patients (Bronchiectasis 7 + Pneumonia 6 + Bronchiolitis 6) being genuinely
unseen diseases. Under the official split most of those patients land in the
training half — meaning the shared backbone would be trained on the audio of the
very patients later presented to it as "unknown". The unknown-detection result
would then be partly a memorisation artifact.

So we emit two partitionings:

    official   the ICBHI challenge split, for literature comparability
    owmtl      leak-free: the 19 unknown-disease patients are held out of every
               training set for every task and every member

Reported sound-event numbers use `official` (comparable to prior work).
Anything feeding the open-world claim uses `owmtl`.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Label / group definitions
# ---------------------------------------------------------------------------

# Sound-event classes, in the canonical index order used by every model.
SOUND_EVENT_CLASSES: Tuple[str, ...] = ("Normal", "Crackle", "Wheeze", "Both")

# Disease-task grouping. Counts are the full-corpus expectation (126 patients)
# and are asserted at build time so a truncated dataset copy fails loudly.
KNOWN_DISEASES: Tuple[str, ...] = ("COPD", "Healthy", "URTI")
UNKNOWN_DISEASES: Tuple[str, ...] = ("Bronchiectasis", "Pneumonia", "Bronchiolitis")
# Asthma (n=1) and LRTI (n=2) are too small to be either a known class or a
# credible unknown group. They are dropped from the disease task but kept as
# sound-event training data, where patient count does not matter.
EXCLUDED_DISEASES: Tuple[str, ...] = ("Asthma", "LRTI")

EXPECTED_DIAGNOSIS_COUNTS: Dict[str, int] = {
    "COPD": 64,
    "Healthy": 26,
    "URTI": 14,
    "Bronchiectasis": 7,
    "Pneumonia": 6,
    "Bronchiolitis": 6,
    "LRTI": 2,
    "Asthma": 1,
}

# Roles a patient can hold. `disease_role` drives Member B/C/D; `sound_event`
# drives Member A. They are derived from one another so they can never diverge.
DISEASE_ROLES = ("known_train", "known_calib", "known_test", "unknown_eval", "excluded")
SOUND_EVENT_SPLITS = ("train", "calib", "test", "held_out")


def sound_event_label(crackle: int, wheeze: int) -> int:
    """(crackle, wheeze) flags -> class index. Normal=0 Crackle=1 Wheeze=2 Both=3."""
    return (1 if crackle else 0) + (2 if wheeze else 0)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# 101_1b1_Al_sc_Meditron
#  |   |   |  |    `-- recording equipment
#  |   |   |  `------- acquisition mode (sc = single channel, mc = multichannel)
#  |   |   `---------- chest location (Tc Al Ar Pl Pr Ll Lr)
#  |   `-------------- recording index
#  `------------------ patient id (101..226)
_STEM_RE = re.compile(
    r"^(?P<pid>\d+)_(?P<rec>[^_]+)_(?P<loc>[^_]+)_(?P<mode>[^_]+)_(?P<device>.+)$"
)

CHEST_LOCATIONS: Tuple[str, ...] = ("Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr")
DEVICES: Tuple[str, ...] = ("AKGC417L", "LittC2SE", "Litt3200", "Meditron")


@dataclass(frozen=True)
class RecordingMeta:
    """Metadata decoded from an ICBHI filename stem."""

    stem: str
    patient_id: int
    recording_index: str
    chest_location: str
    acquisition_mode: str
    device: str


def parse_stem(stem: str) -> RecordingMeta:
    """Decode an ICBHI filename stem. Raises ValueError on an unexpected shape."""
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"unrecognised ICBHI filename stem: {stem!r}")
    return RecordingMeta(
        stem=stem,
        patient_id=int(m.group("pid")),
        recording_index=m.group("rec"),
        chest_location=m.group("loc"),
        acquisition_mode=m.group("mode"),
        device=m.group("device"),
    )


# ---------------------------------------------------------------------------
# Locating the dataset
# ---------------------------------------------------------------------------


@dataclass
class IcbhiPaths:
    """Resolved locations of the files this module reads."""

    audio_dir: str
    diagnosis_file: str
    official_split_file: Optional[str] = None
    demographic_file: Optional[str] = None


_AUDIO_DIR_CANDIDATES = (
    ".",
    "audio_and_txt_files",
    "ICBHI_final_database",
    "Respiratory_Sound_Database/audio_and_txt_files",
    "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files",
)
_DIAGNOSIS_CANDIDATES = (
    "patient_diagnosis.csv",
    "ICBHI_Challenge_diagnosis.txt",
    "Respiratory_Sound_Database/patient_diagnosis.csv",
    "Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",
)
_OFFICIAL_SPLIT_CANDIDATES = (
    "ICBHI_challenge_train_test.txt",
    "ICBHI_Challenge_train_test.txt",
    "Respiratory_Sound_Database/ICBHI_challenge_train_test.txt",
    "Respiratory_Sound_Database/Respiratory_Sound_Database/ICBHI_challenge_train_test.txt",
)
_DEMOGRAPHIC_CANDIDATES = (
    "demographic_info.txt",
    "Respiratory_Sound_Database/demographic_info.txt",
    "Respiratory_Sound_Database/Respiratory_Sound_Database/demographic_info.txt",
)


def _first_existing(root: str, candidates: Iterable[str], must_be_dir: bool = False) -> Optional[str]:
    for rel in candidates:
        path = os.path.normpath(os.path.join(root, rel))
        ok = os.path.isdir(path) if must_be_dir else os.path.isfile(path)
        if ok:
            return path
    return None


def locate(data_root: str) -> IcbhiPaths:
    """Find the ICBHI files under `data_root`, tolerating the common layouts.

    Handles the official release, the vbookshelf Kaggle mirror, and the doubly
    nested variant that mirror unzips into.
    """
    audio_dir = None
    for rel in _AUDIO_DIR_CANDIDATES:
        cand = os.path.normpath(os.path.join(data_root, rel))
        if os.path.isdir(cand):
            # An audio dir is one that actually contains ICBHI .wav files.
            if any(f.endswith(".wav") for f in os.listdir(cand)):
                audio_dir = cand
                break
    if audio_dir is None:
        raise FileNotFoundError(
            f"no directory containing .wav files found under {data_root!r}; "
            f"tried {list(_AUDIO_DIR_CANDIDATES)}"
        )

    diagnosis_file = _first_existing(data_root, _DIAGNOSIS_CANDIDATES)
    if diagnosis_file is None:
        # Some mirrors drop it beside the audio.
        diagnosis_file = _first_existing(audio_dir, ("../patient_diagnosis.csv", "patient_diagnosis.csv"))
    if diagnosis_file is None:
        raise FileNotFoundError(
            f"patient_diagnosis.csv / ICBHI_Challenge_diagnosis.txt not found under {data_root!r}"
        )

    return IcbhiPaths(
        audio_dir=audio_dir,
        diagnosis_file=diagnosis_file,
        official_split_file=_first_existing(data_root, _OFFICIAL_SPLIT_CANDIDATES),
        demographic_file=_first_existing(data_root, _DEMOGRAPHIC_CANDIDATES),
    )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_diagnoses(path: str) -> Dict[int, str]:
    """patient_id -> diagnosis. Accepts comma- or whitespace-separated files."""
    out: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue  # header row
            out[pid] = parts[1].strip()
    if not out:
        raise ValueError(f"no diagnoses parsed from {path!r}")
    return out


def read_official_split(path: str) -> Dict[str, str]:
    """filename stem -> 'train' | 'test' from the ICBHI challenge split file."""
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            stem = parts[0]
            if stem.endswith(".wav"):
                stem = stem[: -len(".wav")]
            tag = parts[1].strip().lower()
            if tag in ("train", "test"):
                out[stem] = tag
    return out


def read_annotation(path: str) -> List[Tuple[float, float, int, int]]:
    """Parse one ICBHI annotation .txt -> [(start, end, crackle, wheeze), ...]."""
    cycles: List[Tuple[float, float, int, int]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(float(parts[2])), int(float(parts[3]))
            except ValueError:
                continue
            if end <= start:
                continue  # malformed interval; ICBHI has a handful
            cycles.append((start, end, crackle, wheeze))
    return cycles


# ---------------------------------------------------------------------------
# Cycle index
# ---------------------------------------------------------------------------

CYCLE_INDEX_FIELDS: Tuple[str, ...] = (
    "cycle_uid",
    "stem",
    "patient_id",
    "diagnosis",
    "chest_location",
    "acquisition_mode",
    "device",
    "cycle_idx",
    "start",
    "end",
    "duration",
    "crackle",
    "wheeze",
    "label",
    "label_name",
)


@dataclass
class CycleRow:
    cycle_uid: str
    stem: str
    patient_id: int
    diagnosis: str
    chest_location: str
    acquisition_mode: str
    device: str
    cycle_idx: int
    start: float
    end: float
    duration: float
    crackle: int
    wheeze: int
    label: int
    label_name: str


def build_cycle_index(paths: IcbhiPaths) -> List[CycleRow]:
    """Parse every annotation file into a flat, sorted list of cycles.

    Reads only .txt files — no audio is decoded, so this takes seconds and can
    be re-run freely.
    """
    diagnoses = read_diagnoses(paths.diagnosis_file)

    wav_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(paths.audio_dir)
        if f.endswith(".wav")
    }

    rows: List[CycleRow] = []
    for stem in sorted(wav_stems):
        txt_path = os.path.join(paths.audio_dir, stem + ".txt")
        if not os.path.isfile(txt_path):
            continue
        meta = parse_stem(stem)
        diagnosis = diagnoses.get(meta.patient_id)
        if diagnosis is None:
            raise KeyError(f"patient {meta.patient_id} ({stem}) missing from diagnosis file")
        for i, (start, end, crackle, wheeze) in enumerate(read_annotation(txt_path)):
            label = sound_event_label(crackle, wheeze)
            rows.append(
                CycleRow(
                    cycle_uid=f"{stem}#{i:03d}",
                    stem=stem,
                    patient_id=meta.patient_id,
                    diagnosis=diagnosis,
                    chest_location=meta.chest_location,
                    acquisition_mode=meta.acquisition_mode,
                    device=meta.device,
                    cycle_idx=i,
                    start=start,
                    end=end,
                    duration=round(end - start, 4),
                    crackle=crackle,
                    wheeze=wheeze,
                    label=label,
                    label_name=SOUND_EVENT_CLASSES[label],
                )
            )
    if not rows:
        raise ValueError(f"no cycles parsed from {paths.audio_dir!r}")
    return rows


def write_cycle_index(rows: Sequence[CycleRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CYCLE_INDEX_FIELDS))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def read_cycle_index(path: str) -> List[dict]:
    """Load the frozen cycle index, coercing numeric columns."""
    ints = {"patient_id", "cycle_idx", "crackle", "wheeze", "label"}
    floats = {"start", "end", "duration"}
    out: List[dict] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k in ints:
                row[k] = int(row[k])
            for k in floats:
                row[k] = float(row[k])
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def disease_group(diagnosis: str) -> str:
    """'known' | 'unknown' | 'excluded' for the open-world disease task."""
    if diagnosis in KNOWN_DISEASES:
        return "known"
    if diagnosis in UNKNOWN_DISEASES:
        return "unknown"
    if diagnosis in EXCLUDED_DISEASES:
        return "excluded"
    raise ValueError(f"unrecognised ICBHI diagnosis: {diagnosis!r}")


def _stratified_take(
    by_class: Dict[str, List[int]], frac: float, rng: random.Random
) -> List[int]:
    """Deterministically draw `frac` of each class's patients.

    Rounds so that every class with >= 2 patients contributes at least one, which
    keeps URTI (n=14) from vanishing out of the calibration partition.
    """
    picked: List[int] = []
    for cls in sorted(by_class):
        members = sorted(by_class[cls])
        shuffled = members[:]
        rng.shuffle(shuffled)
        k = int(round(frac * len(members)))
        k = max(k, 1) if len(members) >= 2 and frac > 0 else k
        k = min(k, len(members) - 1) if len(members) >= 2 else k
        picked.extend(shuffled[:k])
    return sorted(picked)


@dataclass
class Split:
    """Patient-level partition assignments for both tasks."""

    seed: int
    test_frac: float
    calib_frac: float
    diagnosis: Dict[int, str] = field(default_factory=dict)
    disease_role: Dict[int, str] = field(default_factory=dict)
    sound_event: Dict[int, str] = field(default_factory=dict)
    official_by_stem: Dict[str, str] = field(default_factory=dict)
    official_by_patient: Dict[int, str] = field(default_factory=dict)
    official_conflicts: List[int] = field(default_factory=list)

    def patients(self, *, disease_role: Optional[str] = None,
                 sound_event: Optional[str] = None) -> List[int]:
        out = []
        for pid in sorted(self.diagnosis):
            if disease_role is not None and self.disease_role[pid] != disease_role:
                continue
            if sound_event is not None and self.sound_event[pid] != sound_event:
                continue
            out.append(pid)
        return out


def build_split(
    rows: Sequence[CycleRow],
    *,
    seed: int = 42,
    test_frac: float = 0.40,
    calib_frac: float = 0.15,
    official_by_stem: Optional[Dict[str, str]] = None,
    strict_counts: bool = True,
) -> Split:
    """Construct the canonical OWMTL split.

    Parameters
    ----------
    test_frac
        Fraction of *known-disease patients* held out for testing. 0.40 mirrors
        the official challenge ratio.
    calib_frac
        Fraction of the remaining known-disease patients carved off as a
        never-trained calibration partition (Member D's conformal/temperature
        fitting). Set to 0.0 to skip it.
    strict_counts
        Assert the per-diagnosis patient counts match the full ICBHI corpus.
        Turn off only when working against a deliberate subset.
    """
    diagnosis: Dict[int, str] = {}
    for r in rows:
        prev = diagnosis.setdefault(r.patient_id, r.diagnosis)
        if prev != r.diagnosis:
            raise ValueError(f"patient {r.patient_id} has conflicting diagnoses")

    counts = Counter(diagnosis.values())
    if strict_counts:
        mismatches = {
            d: (counts.get(d, 0), n)
            for d, n in EXPECTED_DIAGNOSIS_COUNTS.items()
            if counts.get(d, 0) != n
        }
        if mismatches:
            raise ValueError(
                "diagnosis counts do not match the full ICBHI corpus "
                f"(got/expected): {mismatches}. Pass strict_counts=False to override."
            )

    known_by_dx: Dict[str, List[int]] = defaultdict(list)
    unknown: List[int] = []
    excluded: List[int] = []
    for pid, dx in diagnosis.items():
        grp = disease_group(dx)
        if grp == "known":
            known_by_dx[dx].append(pid)
        elif grp == "unknown":
            unknown.append(pid)
        else:
            excluded.append(pid)

    rng = random.Random(seed)
    test_ids = set(_stratified_take(known_by_dx, test_frac, rng))

    remaining_by_dx = {
        dx: [p for p in pids if p not in test_ids] for dx, pids in known_by_dx.items()
    }
    calib_ids = set(_stratified_take(remaining_by_dx, calib_frac, rng)) if calib_frac > 0 else set()
    train_ids = {p for pids in remaining_by_dx.values() for p in pids} - calib_ids

    disease_role: Dict[int, str] = {}
    sound_event: Dict[int, str] = {}
    for pid in diagnosis:
        if pid in test_ids:
            disease_role[pid], sound_event[pid] = "known_test", "test"
        elif pid in calib_ids:
            disease_role[pid], sound_event[pid] = "known_calib", "calib"
        elif pid in train_ids:
            disease_role[pid], sound_event[pid] = "known_train", "train"
        elif pid in unknown:
            # The anti-leak rule: unseen-disease patients are unseen everywhere.
            disease_role[pid], sound_event[pid] = "unknown_eval", "held_out"
        else:
            # Asthma / LRTI: no disease-task role, but usable sound-event data.
            disease_role[pid], sound_event[pid] = "excluded", "train"

    split = Split(
        seed=seed,
        test_frac=test_frac,
        calib_frac=calib_frac,
        diagnosis=diagnosis,
        disease_role=disease_role,
        sound_event=sound_event,
    )

    if official_by_stem:
        split.official_by_stem = dict(official_by_stem)
        per_patient: Dict[int, set] = defaultdict(set)
        for stem, tag in official_by_stem.items():
            try:
                per_patient[parse_stem(stem).patient_id].add(tag)
            except ValueError:
                continue
        for pid, tags in per_patient.items():
            if len(tags) == 1:
                split.official_by_patient[pid] = next(iter(tags))
            else:
                split.official_conflicts.append(pid)
        split.official_conflicts.sort()

    return split


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def split_to_dict(split: Split, rows: Sequence[CycleRow], paths: IcbhiPaths) -> dict:
    """Assemble the committed split_v1.json payload, including audit counts."""
    by_role: Dict[str, List[int]] = defaultdict(list)
    for pid, role in sorted(split.disease_role.items()):
        by_role[role].append(pid)
    by_se: Dict[str, List[int]] = defaultdict(list)
    for pid, se in sorted(split.sound_event.items()):
        by_se[se].append(pid)

    cycles_by_se: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        cycles_by_se[split.sound_event[r.patient_id]][r.label_name] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated": date.today().isoformat(),
        "generator": "Asif's/owmtl/icbhi.py",
        "seed": split.seed,
        "test_frac": split.test_frac,
        "calib_frac": split.calib_frac,
        "source": {
            "audio_dir": paths.audio_dir,
            "diagnosis_file": os.path.basename(paths.diagnosis_file),
            "official_split_file": (
                os.path.basename(paths.official_split_file)
                if paths.official_split_file
                else None
            ),
        },
        "totals": {
            "patients": len(split.diagnosis),
            "recordings": len({r.stem for r in rows}),
            "cycles": len(rows),
        },
        "diagnosis_counts": dict(sorted(Counter(split.diagnosis.values()).items())),
        "class_definitions": {
            "sound_event_classes": list(SOUND_EVENT_CLASSES),
            "known_diseases": list(KNOWN_DISEASES),
            "unknown_diseases": list(UNKNOWN_DISEASES),
            "excluded_diseases": list(EXCLUDED_DISEASES),
        },
        "partitions": {
            "owmtl": {
                "disease_role": {k: by_role.get(k, []) for k in DISEASE_ROLES},
                "sound_event": {k: by_se.get(k, []) for k in SOUND_EVENT_SPLITS},
            },
            "official": {
                "available": bool(split.official_by_stem),
                "by_patient": {
                    str(pid): tag for pid, tag in sorted(split.official_by_patient.items())
                },
                "patients_spanning_both_halves": split.official_conflicts,
                "by_stem": dict(sorted(split.official_by_stem.items())),
            },
        },
        "cycle_counts_by_sound_event_split": {
            se: dict(sorted(cycles_by_se[se].items())) for se in SOUND_EVENT_SPLITS
        },
        "patients": {
            str(pid): {
                "diagnosis": split.diagnosis[pid],
                "disease_group": disease_group(split.diagnosis[pid]),
                "disease_role": split.disease_role[pid],
                "sound_event": split.sound_event[pid],
                "official": split.official_by_patient.get(pid),
            }
            for pid in sorted(split.diagnosis)
        },
    }


def write_split(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def load_split(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# The member-facing accessor
# ---------------------------------------------------------------------------


def select_cycles(
    index: Sequence[dict],
    split: dict,
    *,
    partition: str,
    scheme: str = "owmtl",
) -> List[dict]:
    """Return the cycle rows belonging to one partition. This is the one call
    every member needs.

        cycles = select_cycles(index, split, partition="train")            # owmtl
        cycles = select_cycles(index, split, partition="test", scheme="official")

    `scheme="owmtl"` accepts: train, calib, test, held_out.
    `scheme="official"` accepts: train, test — and resolves per recording, as the
    challenge split is defined at file level.
    """
    if scheme == "owmtl":
        if partition not in SOUND_EVENT_SPLITS:
            raise ValueError(f"partition must be one of {SOUND_EVENT_SPLITS}, got {partition!r}")
        assign = split["partitions"]["owmtl"]["sound_event"]
        wanted = set(assign[partition])
        return [r for r in index if r["patient_id"] in wanted]

    if scheme == "official":
        if partition not in ("train", "test"):
            raise ValueError("official scheme has only 'train' and 'test'")
        official = split["partitions"]["official"]
        if not official["available"]:
            raise RuntimeError(
                "the ICBHI official split file was not present when split_v1.json "
                "was built; use scheme='owmtl' or re-run build_index.py with the "
                "challenge split file available"
            )
        by_stem = official["by_stem"]
        return [r for r in index if by_stem.get(r["stem"]) == partition]

    raise ValueError(f"unknown scheme {scheme!r}")


def disease_patients(split: dict, role: str) -> List[int]:
    """Patient IDs for a disease-task role, e.g. 'known_train', 'unknown_eval'."""
    if role not in DISEASE_ROLES:
        raise ValueError(f"role must be one of {DISEASE_ROLES}, got {role!r}")
    return list(split["partitions"]["owmtl"]["disease_role"][role])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(split: dict, index: Sequence[dict]) -> List[str]:
    """Check the invariants every downstream result depends on.

    Returns a list of problem strings; empty means the split is sound.
    """
    problems: List[str] = []
    owmtl = split["partitions"]["owmtl"]

    # 1. Patient-independence: no patient in two sound-event partitions.
    seen: Dict[int, str] = {}
    for part, pids in owmtl["sound_event"].items():
        for pid in pids:
            if pid in seen:
                problems.append(
                    f"patient {pid} appears in both '{seen[pid]}' and '{part}'"
                )
            seen[pid] = part

    # 2. Every patient in the index is assigned.
    for pid in sorted({r["patient_id"] for r in index}):
        if pid not in seen:
            problems.append(f"patient {pid} present in cycle index but unassigned")

    # 3. The anti-leak rule: unknown-disease patients never train.
    unknown = set(owmtl["disease_role"]["unknown_eval"])
    trained = set(owmtl["sound_event"]["train"]) | set(owmtl["sound_event"]["calib"])
    leaked = sorted(unknown & trained)
    if leaked:
        problems.append(
            f"unknown-disease patients leaked into a training partition: {leaked}"
        )

    # 4. Every sound-event class has test support (a split that drops Both is
    #    unusable regardless of how balanced it looks by patient count).
    counts = split["cycle_counts_by_sound_event_split"]["test"]
    for cls in SOUND_EVENT_CLASSES:
        n = counts.get(cls, 0)
        if n == 0:
            problems.append(f"sound-event class {cls!r} has zero test cycles")
        elif n < 30:
            problems.append(
                f"sound-event class {cls!r} has only {n} test cycles — too few for a "
                f"stable per-class F1"
            )

    # 5. Known disease classes must survive into the test partition.
    dx = {int(p): v["diagnosis"] for p, v in split["patients"].items()}
    test_dx = Counter(dx[p] for p in owmtl["disease_role"]["known_test"])
    for cls in KNOWN_DISEASES:
        if test_dx.get(cls, 0) == 0:
            problems.append(f"known disease {cls!r} has no test patients")

    # 6. If the official split is present, flag patients spanning both halves —
    #    the challenge split is patient-independent, so this should be empty.
    official = split["partitions"]["official"]
    if official["available"] and official["patients_spanning_both_halves"]:
        problems.append(
            "official split assigns these patients to both halves: "
            f"{official['patients_spanning_both_halves']}"
        )

    return problems


def summarise(split: dict) -> str:
    """Human-readable summary, printed by build_index.py and the notebooks."""
    owmtl = split["partitions"]["owmtl"]
    dx = {int(p): v["diagnosis"] for p, v in split["patients"].items()}
    lines: List[str] = []
    t = split["totals"]
    lines.append(
        f"ICBHI 2017 — {t['patients']} patients, {t['recordings']} recordings, "
        f"{t['cycles']} cycles"
    )
    lines.append(f"seed={split['seed']}  test_frac={split['test_frac']}  "
                 f"calib_frac={split['calib_frac']}")
    lines.append("")
    lines.append("Sound-event partitions (Member A)")
    lines.append(f"  {'partition':<10} {'patients':>8} {'cycles':>8}   class distribution")
    for part in SOUND_EVENT_SPLITS:
        pids = owmtl["sound_event"][part]
        counts = split["cycle_counts_by_sound_event_split"][part]
        total = sum(counts.values())
        dist = "  ".join(f"{c}={counts.get(c, 0)}" for c in SOUND_EVENT_CLASSES)
        lines.append(f"  {part:<10} {len(pids):>8} {total:>8}   {dist}")
    lines.append("")
    lines.append("Disease-task roles (Members B/C/D)")
    for role in DISEASE_ROLES:
        pids = owmtl["disease_role"][role]
        by_dx = Counter(dx[p] for p in pids)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(by_dx.items())) or "-"
        lines.append(f"  {role:<14} {len(pids):>3} patients   {detail}")
    official = split["partitions"]["official"]
    lines.append("")
    lines.append(
        "Official ICBHI 60/40 split: "
        + ("available" if official["available"] else "NOT FOUND — 'official' scheme disabled")
    )
    return "\n".join(lines)
