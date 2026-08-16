"""
owmtl.device_check  (CC2)
=========================

Gate G4 feasibility: BEFORE planning any leave-one-device-out (LODO) / device-shift
experiment, verify (a) how many acquisition devices exist, (b) whether any patient was
recorded on more than one device, and (c) whether device is confounded with diagnosis.

Why this matters (from the project's own post-mortem): ICBHI has **4 devices**
(AKGC417L, LittC2SE, Litt3200, Meditron) -- the "7" that Direction 1 assumed were
stethoscopes are chest LOCATIONS. If patients do not span devices, "leave-one-device-out"
is just "leave-those-patients-out" and no device claim is possible.

ICBHI filename convention:
    PatientNumber_RecordingIndex_ChestLocation_AcquisitionMode_RecordingEquipment.wav
    e.g. 101_1b1_Al_sc_Meditron.wav  ->  device = 'Meditron'

Diagnosis file (optional): ICBHI_Challenge_diagnosis.txt with lines "patientID<TAB>diagnosis".

Pure stdlib + numpy. Use from a notebook or via scripts/check_device_structure.py.
"""
from __future__ import annotations
import os, glob
from collections import defaultdict, Counter


def parse_filename(fname: str):
    base = os.path.basename(fname)
    base = base[:-4] if base.lower().endswith(".wav") else base
    parts = base.split("_")
    if len(parts) < 5:
        return None
    return {"patient": parts[0], "rec": parts[1], "location": parts[2],
            "mode": parts[3], "device": parts[4]}


def _load_diagnoses(diag_path: str):
    d: dict[str, str] = {}
    if not diag_path or not os.path.exists(diag_path):
        return d
    with open(diag_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            toks = line.replace("\t", " ").split()
            if len(toks) >= 2:
                d[toks[0]] = toks[1]
    return d


def analyze(wav_dir_or_list, diagnosis_file: str | None = None):
    """Return a structured feasibility report dict."""
    if isinstance(wav_dir_or_list, (list, tuple)):
        files = list(wav_dir_or_list)
    else:
        files = glob.glob(os.path.join(wav_dir_or_list, "*.wav"))
    recs = [r for r in (parse_filename(f) for f in files) if r]

    devices = Counter(r["device"] for r in recs)
    patient_devices = defaultdict(set)
    for r in recs:
        patient_devices[r["patient"]].add(r["device"])

    multi_device_patients = {p: sorted(ds) for p, ds in patient_devices.items()
                             if len(ds) > 1}

    diag = _load_diagnoses(diagnosis_file)
    device_by_diag = defaultdict(Counter)   # device -> Counter(diagnosis)
    diag_by_device = defaultdict(Counter)    # diagnosis -> Counter(device)
    if diag:
        for p, ds in patient_devices.items():
            dg = diag.get(p, "UNKNOWN")
            for d in ds:
                device_by_diag[d][dg] += 1
                diag_by_device[dg][d] += 1

    # confound heuristic: a diagnosis is device-confounded if >90% of its patients
    # sit on a single device.
    confounded = {}
    for dg, dev_counts in diag_by_device.items():
        tot = sum(dev_counts.values())
        if tot:
            top_dev, top_n = dev_counts.most_common(1)[0]
            frac = top_n / tot
            if frac >= 0.90:
                confounded[dg] = {"device": top_dev, "fraction": round(frac, 3)}

    n_patients = len(patient_devices)
    lodo_feasible = (len(devices) >= 2) and (len(multi_device_patients) > 0)

    return {
        "n_files": len(files),
        "n_parsed": len(recs),
        "n_patients": n_patients,
        "n_devices": len(devices),
        "device_counts": dict(devices),
        "n_multi_device_patients": len(multi_device_patients),
        "multi_device_patients": multi_device_patients,
        "diagnosis_available": bool(diag),
        "device_by_diagnosis": {k: dict(v) for k, v in device_by_diag.items()},
        "confounded_diagnoses": confounded,
        "lodo_feasible": lodo_feasible,
        "verdict": _verdict(len(devices), len(multi_device_patients), confounded),
    }


def _verdict(n_devices, n_multi, confounded):
    if n_devices < 2:
        return ("NO device axis: <2 devices found. Drop device-shift; use pediatric "
                "(SPRSound) shift only for the covariate analysis.")
    if n_multi == 0:
        return ("LODO NOT separable: no patient spans >1 device, so leave-one-device-out "
                "== leave-those-patients-out. A device claim is not possible; rely on "
                "SPRSound pediatric shift for the covariate stress instead.")
    msg = ("LODO feasible: multiple devices and some patients span devices.")
    if confounded:
        msg += (f" CAUTION: {len(confounded)} diagnosis(es) are device-confounded "
                f"({', '.join(confounded)}). Report device and disease effects separately "
                f"and interpret with care.")
    return msg


def format_report(rep: dict) -> str:
    lines = ["=== ICBHI device-structure feasibility (Gate G4) ===",
             f"files parsed        : {rep['n_parsed']}/{rep['n_files']}",
             f"patients            : {rep['n_patients']}",
             f"devices ({rep['n_devices']}): {rep['device_counts']}",
             f"multi-device patients: {rep['n_multi_device_patients']}"]
    if rep["confounded_diagnoses"]:
        lines.append(f"device-confounded dx : {rep['confounded_diagnoses']}")
    lines.append("")
    lines.append("VERDICT: " + rep["verdict"])
    return "\n".join(lines)
