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


def load_official_split(split_file: str) -> dict:
    d = {}
    with open(split_file) as fh:
        for line in fh:
            t = line.replace("\t", " ").split()
            if len(t) >= 2:
                d[t[0]] = t[1].lower()
    return d


def load_diagnoses(diag_file: str) -> dict:
    d = {}
    with open(diag_file) as fh:
        for line in fh:
            t = line.replace("\t", " ").split()
            if len(t) >= 2:
                d[t[0]] = t[1]
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
