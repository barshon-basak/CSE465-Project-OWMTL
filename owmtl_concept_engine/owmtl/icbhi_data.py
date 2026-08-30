"""
owmtl.icbhi_data
================

Minimal ICBHI 2017 loading helpers shared by the two Critical-Fix notebooks.
Kept separate so the OFFICIAL 60/40 split and patient-independence are enforced in
ONE place (Protocol Essentials #1, #8).

ICBHI layout (as distributed):
  audio_and_txt_files/
    <PatientNumber>_<RecIdx>_<Location>_<Mode>_<Device>.wav
    <same>.txt      # one line per respiratory cycle: start  end  crackle  wheeze
  ICBHI_challenge_train_test.txt   # <filename_stem>\t(train|test)   -> OFFICIAL split
  ICBHI_Challenge_diagnosis.txt    # <PatientNumber>\t<diagnosis>

librosa is imported lazily (present on Kaggle/Colab); the parsing functions need no
audio library and are import-safe anywhere.
"""
from __future__ import annotations
import os, glob
from dataclasses import dataclass, field


def read_annotation(txt_path: str):
    """Return list of (start_s, end_s, crackle01, wheeze01) for each cycle."""
    cycles = []
    with open(txt_path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4:
                cycles.append((float(p[0]), float(p[1]), int(p[2]), int(p[3])))
    return cycles


def sound_event_label(crackle: int, wheeze: int) -> int:
    """ICBHI 4-class sound-event label: 0 Normal, 1 Crackle, 2 Wheeze, 3 Both."""
    return {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}[(crackle, wheeze)]


# The published split file ships INSIDE this package so a notebook can never silently
# fall back to a hand-typed patient list. That failure produced the 2026-08-14 result
# ("7.1% of cycles reported as 60/40") and again on 2026-08-26 (a 15/47-correct list).
PACKAGED_SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "ICBHI_challenge_train_test.txt")


def _tokens(line: str):
    """ICBHI metadata files are inconsistently tab-, space-, or COMMA-separated
    (patient_diagnosis.csv is comma-separated on most Kaggle mirrors)."""
    return [t for t in line.replace("\t", " ").replace(",", " ").split() if t]


def find_split_file(extra_dirs=()) -> str:
    """Locate ICBHI_challenge_train_test.txt. The packaged copy wins -- it is the one
    artifact guaranteed to exist regardless of which mirror the audio came from."""
    import glob
    cands = [PACKAGED_SPLIT_FILE]
    for d in list(extra_dirs) + ["/kaggle/input", "/kaggle/working", "/content", "."]:
        if d and os.path.isdir(d):
            cands += sorted(glob.glob(os.path.join(d, "**", "ICBHI_challenge_train_test.txt"),
                                      recursive=True))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "ICBHI_challenge_train_test.txt not found. Do NOT hand-type a patient list; "
        "upload the committed copy (repo: Asif's/ICBHI_challenge_train_test.txt).")


def load_official_split(split_file: str) -> dict:
    """{stem: 'train'|'test'} exactly as published. Raises rather than returning {}."""
    d = {}
    with open(split_file) as fh:
        for line in fh:
            t = _tokens(line)
            if len(t) >= 2 and t[1].lower() in ("train", "test"):
                d[t[0].replace(".wav", "")] = t[1].lower()
    if not d:
        raise ValueError(f"no train/test rows parsed from {split_file}")
    return d


def load_split(split_file=None, mode: str = "patient_independent"):
    """Return ({stem: 'train'|'test'}, info). Project policy; the full audit and the
    rationale live in `Asif's/audit/official_split.py` -- this is the packaged
    implementation the Kaggle notebooks import.

    mode='patient_independent' (default): the published split, with every recording of a
        patient appearing on BOTH sides reassigned to TRAIN. Conservative: the test set
        then contains no patient seen during training (Model_Training_Protocol.md sect 1).
    mode='official': published verbatim. NOT patient-independent -- say so wherever you
        report it.
    """
    if mode not in ("patient_independent", "official"):
        raise ValueError(f"unknown mode {mode!r}")
    path = split_file or find_split_file()
    rows = load_official_split(path)

    sides = {}
    for stem, sp in rows.items():
        sides.setdefault(stem.split("_")[0], set()).add(sp)
    leaking = sorted(p for p, v in sides.items() if len(v) > 1)

    if mode == "patient_independent":
        rows = {stem: ("train" if stem.split("_")[0] in leaking else sp)
                for stem, sp in rows.items()}
        still_leaking = []
    else:
        still_leaking = leaking

    n_test = sum(v == "test" for v in rows.values())
    info = {"path": path, "mode": mode, "n_recordings": len(rows),
            "n_patients": len(sides), "n_train": len(rows) - n_test, "n_test": n_test,
            "test_fraction": round(n_test / len(rows), 4),
            "leaking_patients": leaking,
            "leaking_patients_still_present": still_leaking,
            "is_patient_independent": not still_leaking,
            "split_method": ("patient_independent_official_60_40" if mode == "patient_independent"
                             else "official_60_40_NOT_patient_independent")}
    return rows, info


def load_diagnoses(diag_file: str) -> dict:
    """{patient: diagnosis}. Raises on a missing/unparseable file -- returning {} silently
    made every downstream diagnosis 'UNKNOWN' and turned the CC2 device-confound check into
    a no-op that reported 'no confound'."""
    if not diag_file or not os.path.exists(diag_file):
        raise FileNotFoundError(f"diagnosis file not found: {diag_file!r}")
    d = {}
    with open(diag_file) as fh:
        for line in fh:
            t = _tokens(line)
            if len(t) >= 2:
                d[t[0]] = t[1]
    if not d:
        raise ValueError(f"no patient/diagnosis rows parsed from {diag_file}")
    return d


def parse_stem(stem: str) -> dict:
    p = stem.split("_")
    return {"patient": p[0], "device": p[4] if len(p) > 4 else "UNKNOWN",
            "location": p[2] if len(p) > 2 else "?"}


@dataclass
class CycleRecord:
    stem: str
    patient: str
    device: str
    split: str            # 'train' | 'test'
    start: float
    end: float
    crackle: int
    wheeze: int
    sound_label: int
    diagnosis: str = "UNKNOWN"


def build_cycle_index(audio_dir: str, split_file: str, diagnosis_file: str):
    """Enumerate every annotated cycle with its official split + patient + device +
    labels. Does NOT load audio (cheap). Returns list[CycleRecord]."""
    split = load_official_split(split_file)
    diag = load_diagnoses(diagnosis_file)
    records = []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        stem = os.path.basename(wav)[:-4]
        txt = os.path.join(audio_dir, stem + ".txt")
        if not os.path.exists(txt):
            continue
        meta = parse_stem(stem)
        sp = split.get(stem, "train")
        for (s, e, cr, wh) in read_annotation(txt):
            records.append(CycleRecord(
                stem=stem, patient=meta["patient"], device=meta["device"], split=sp,
                start=s, end=e, crackle=cr, wheeze=wh,
                sound_label=sound_event_label(cr, wh),
                diagnosis=diag.get(meta["patient"], "UNKNOWN")))
    return records


def load_cycle_waveform(audio_dir: str, rec: CycleRecord, sr: int = 16000):
    """Load and resample one cycle's waveform. Requires librosa (Kaggle/Colab)."""
    import librosa
    wav = os.path.join(audio_dir, rec.stem + ".wav")
    y, _ = librosa.load(wav, sr=sr, offset=rec.start,
                        duration=max(rec.end - rec.start, 0.01))
    return y


# Known ICBHI disease groups used by the project (kept here for one source of truth).
KNOWN_DISEASES = ["COPD", "Healthy", "URTI"]            # 104 patients
HELD_OUT_DISEASES = ["Bronchiectasis", "Pneumonia", "Bronchiolitis"]   # 19 patients
