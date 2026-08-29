#!/usr/bin/env python3
"""
Generator for M38 — Large-N Open-Set Baseline Suite (near-OOD and far-OOD).

Per the workstream convention (Asif's/CLAUDE.md): notebooks are generated, not hand-edited,
so regeneration is reproducible. Run `python3 gen_M38.py` to emit M38_largeN_openset.ipynb.

M38 is M29's four post-hoc OOD scorers, evaluated against three OOD pools instead of one:
  1. ICBHI unknown patients (n=19)     -- reproduces M29 exactly; the existing bar
  2. SPRSound (large N)                -- NEAR-OOD: same modality, different population
  3. Coswara heavy cough (large N)     -- FAR-OOD: different modality entirely

The in-distribution side is held identical across all three (ICBHI known-test patients), so
the only variable is which OOD pool is being detected.

Watch the nested-triple-quote trap: code cells are wrapped in r'''...''' so any Python
docstrings inside must use \"\"\"...\"\"\", never the reverse.
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


# ===========================================================================
md(r"""
# M38 — Large-N Open-Set Baseline Suite (near-OOD and far-OOD)

**Model ID:** M38 · **Chunk:** C (open-set baselines) · **Contributor:** Asif
**Requires:** M12's frozen backbone checkpoint (`Asif's/M2/best_model.pth`) — no training happens here.

## Why this exists

`Novelty Search.md` §1 names the project's **strongest contribution**:

> "the statistically defensible evaluation redesign — coarse pooled unknown class + **large-N OOD
> stress tests**, instead of the fragile small-sample splits most papers in this space use."

`Asif's/Statistics/SIGNIFICANCE_REPORT.md` showed the project has not actually delivered that. At
n=19 unknown patients, **every** open-set AUROC in the repo — including M29's best (Energy, 0.6466)
— has a 95% CI that includes chance. The evaluation is still the fragile small-sample split the
contribution claims to replace.

M38 fixes the sample size, not the method. Same four scorers as M29, same frozen backbone, same
preprocessing — only the OOD pool changes.

## The three regimes

| Regime | OOD pool | Modality | What a good result means |
|---|---|---|---|
| **A — ICBHI unknown** | 19 patients | stethoscope, lung cycles | Reproduces M29. Sanity check. |
| **B — SPRSound (near-OOD)** | large N | stethoscope, lung cycles (pediatric) | **The real test.** Same modality, unseen population. |
| **C — Coswara (far-OOD)** | large N | phone mic, voluntary cough | Control. Should be *easy*. |

**Regime C is deliberately a control, not a headline result.** Coswara is heavy-cough audio recorded
on phone microphones; ICBHI is stethoscope auscultation of breathing cycles. A detector can separate
those on recording modality alone without knowing anything about disease. So a high AUROC on Coswara
is **not** evidence the cross-task mechanism detects unknown diseases — it is evidence the model can
tell a phone from a stethoscope. Reporting it as a success would be exactly the kind of claim
`Novelty Search.md` §2 Attack 3 warns about. Its real diagnostic value is inverted: **if a detector
cannot even clear far-OOD, near-OOD results from it are not worth interpreting.**

## Two things this fixes

1. **Confidence intervals are computed inline**, per regime, per scorer — not bolted on afterward.
2. **Raw per-sample scores are written to `scores_M38.csv`.** No model in this project has ever
   saved these, which is why `SIGNIFICANCE_REPORT.md` had to fall back on an analytic approximation
   instead of a proper paired DeLong test. Any future comparison against M38 can use the real test.

## Known bug this corrects

`Barshon's/M19/results_M19.json` reports Coswara with `num_samples: 2`. Its configured path ends in
`.../kaggle_data/pGxub66GjDdAaJDd95hGHo3BcnJ3` — a **single participant's folder**, not the dataset
root. M38 globs the parent directory recursively instead. M19's Coswara AUROC (0.4881) should not be
cited by anyone; it is computed from two files.
""")

# ===========================================================================
md(r"""
---
## Section 1 — Environment Setup

Detects Colab / Kaggle / local, mounts Drive when available, and sets up output directories.
Checkpoints and results go to persistent storage; the spectrogram cache stays on local disk.
""")

code(r'''
# ============================================================
# CELL 0 — ENVIRONMENT VERIFICATION & PATHS
# ============================================================
import os, sys, platform

print("=" * 70)
print("M38 (Large-N Open-Set Baselines) — ENVIRONMENT")
print("=" * 70)
print(f"Python       : {sys.version.split()[0]}  ({platform.platform()})")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA avail   : {cuda_ok}")
    if cuda_ok:
        print(f"GPU device   : {torch.cuda.get_device_name(0)}")
    else:
        print("NOTE: No GPU. M38 is inference-only, so CPU works — just slower.")
except ImportError:
    print("ERROR: PyTorch not installed.")

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
DRIVE_MOUNTED = False

if IN_COLAB:
    print("\n--- Google Colab environment detected ---")
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_MOUNTED = True
        print("Google Drive mounted at /content/drive")
    except Exception as e:
        print(f"Drive mount skipped ({e}). Falling back to ephemeral /content storage.")
elif os.path.exists("/kaggle/working"):
    print("\n--- Kaggle environment detected ---")
else:
    print("\n--- Local / standard Linux environment detected ---")

if IN_COLAB and DRIVE_MOUNTED:
    BASE_DIR = "/content/drive/MyDrive/OWMTL/M38"
elif IN_COLAB:
    BASE_DIR = "/content/OWMTL/M38"
elif os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
else:
    BASE_DIR = "./outputs_M38"

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = ("/content/owmtl_spec_cache" if IN_COLAB else
             "/kaggle/working/owmtl_spec_cache" if os.path.exists("/kaggle/working") else
             "./owmtl_spec_cache")

for d in (RESULTS_DIR, CACHE_DIR):
    os.makedirs(d, exist_ok=True)

print(f"\nResults dir    : {RESULTS_DIR}   (persistent)")
print(f"Spec cache dir : {CACHE_DIR}   (local disk — fast, disposable)")
print("=" * 70)
''')

# ===========================================================================
md(r"""
---
## Section 1b — Dataset Download (Kaggle)

Downloads all three datasets straight into the runtime. Each download is **idempotent** — if the
files are already unpacked, it skips them, so re-running the notebook after a disconnect costs
nothing.

> **Do not commit your real Kaggle key.** The cell below ships a placeholder and refuses to run
> until you replace it. Get a key from kaggle.com → Settings → API → Create New Token. Safer still:
> use Colab's **Secrets** manager (key icon, left sidebar) — the commented-out block does that
> without the key ever appearing in the notebook text.
""")

code(r'''
# ============================================================
# CELL 0b — DOWNLOAD DATASETS FROM KAGGLE (idempotent)
# ============================================================
import os

os.environ['KAGGLE_USERNAME'] = 'AsifM7'
os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KAGGLE_API_KEY_HERE'   # <-- replace before running

# ---- Safer alternative: Colab Secrets (never appears in the notebook text) ----
# from google.colab import userdata
# os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
# os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

import subprocess, sys, glob

_DL_ROOT = "/content" if os.path.exists("/content") else "./data"
os.makedirs(_DL_ROOT, exist_ok=True)

# (kaggle slug, a directory that exists once unpacked, human name)
_DATASETS = [
    ("vbookshelf/respiratory-sound-database", "Respiratory_Sound_Database", "ICBHI 2017"),
    ("mayarelghandour/sprsound-nosplit",      "sprsound",                   "SPRSound"),
    ("sarabhian/coswara-dataset-heavy-cough", "coswara_data",               "Coswara heavy cough"),
]


def _already_there(marker):
    """A dataset counts as present if its marker dir exists anywhere under _DL_ROOT
    and contains at least one .wav somewhere beneath it."""
    for d in glob.glob(os.path.join(_DL_ROOT, "**", marker), recursive=True):
        if glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
            return d
    return None


_need = [(slug, marker, name) for slug, marker, name in _DATASETS
         if _already_there(marker) is None]

if not _need:
    print("All three datasets already present — skipping download.")
else:
    if os.environ.get('KAGGLE_KEY', '') in ('', 'PASTE_YOUR_KAGGLE_API_KEY_HERE'):
        raise RuntimeError(
            "KAGGLE_KEY is still the placeholder. Paste your real Kaggle API key into this cell "
            "(or use the commented-out Colab Secrets block above) before running.")

    print("Installing kaggle CLI ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    for slug, marker, name in _need:
        print(f"\nDownloading {name}  ({slug}) ...")
        try:
            subprocess.check_call(["kaggle", "datasets", "download", "-d", slug,
                                   "-p", _DL_ROOT, "--unzip"])
        except subprocess.CalledProcessError as e:
            # A missing external OOD set degrades the run, it does not invalidate it --
            # the notebook reports whichever regimes it could actually evaluate.
            print(f"  WARNING: download failed for {name} ({e}). "
                  f"That regime will be reported as unavailable rather than faked.")

print("\n--- Dataset presence ---")
for slug, marker, name in _DATASETS:
    found = _already_there(marker)
    n = len(glob.glob(os.path.join(found, "**", "*.wav"), recursive=True)) if found else 0
    print(f"  {name:<22} {'OK' if found else 'MISSING':<8} {n:>6} .wav   {found or ''}")
''')

# ===========================================================================
md(r"""
---
## Section 2 — Configuration

Preprocessing is inherited **verbatim** from M2. This is not a style choice: if the log-mel
parameters differ by even one value, the embeddings are no longer the M12 backbone's embeddings and
the comparison against M29 is mis-specified.

`max_external_samples` caps how many files are pulled from each external dataset. It exists purely
to bound runtime — set it to `None` to use everything. Whatever it ends up being is recorded in the
results JSON, because a sample cap changes the width of every confidence interval below.
""")

code(r'''
# ============================================================
# CELL 1 — DEPENDENCIES & CONFIGURATION
# ============================================================
import subprocess, math

def pip_install(pkg, import_name=None):
    try:
        __import__(import_name or pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p, n in [("librosa", None), ("soundfile", None), ("scikit-learn", "sklearn"),
             ("tqdm", None), ("matplotlib", None), ("seaborn", None), ("pandas", None)]:
    pip_install(p, n)
print("Dependencies ready.\n")

import glob

def find_first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def find_dataset_dir(markers, must_contain_wav=True):
    """Locate a dataset root by marker directory name, searching the usual mount points.
    Returns the deepest-matching existing directory that actually contains audio."""
    roots = ["/content", "/kaggle/input", "./data", "."]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for marker in markers:
            for d in sorted(glob.glob(os.path.join(root, "**", marker), recursive=True)):
                if not os.path.isdir(d):
                    continue
                if not must_contain_wav:
                    return d
                if glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
                    return d
    return None


DATA_ROOT = find_dataset_dir(["audio_and_txt_files"])

DIAG_FILE = find_first([
    "/content/**/patient_diagnosis.csv", "/content/**/*diagnosis*.csv",
    "/kaggle/input/**/patient_diagnosis.csv", "./**/patient_diagnosis.csv",
])

M2_CKPT = find_first([
    "/content/drive/MyDrive/OWMTL/M2/checkpoints/best_model.pth",
    "/content/drive/MyDrive/OWMTL/M2/best_model.pth",
    "/content/best_model.pth", "/content/M2_best_model.pth",
    "/kaggle/input/**/best_model.pth", "./**/M2/best_model.pth",
])

# --- External OOD roots ---
# NOTE on Coswara: M19 pointed at .../kaggle_data/<ONE_PARTICIPANT_ID>/ and consequently
# evaluated 2 files. We resolve the PARENT (kaggle_data) and glob recursively, so every
# participant folder is included.
SPRSOUND_ROOT = find_dataset_dir(["sprsound", "SPRSound", "sprsound-nosplit"])
COSWARA_ROOT = find_dataset_dir(["kaggle_data", "coswara_data"])

print(f"DATA_ROOT     : {DATA_ROOT}")
print(f"DIAG_FILE     : {DIAG_FILE}")
print(f"M2_CKPT       : {M2_CKPT}")
print(f"SPRSOUND_ROOT : {SPRSOUND_ROOT}")
print(f"COSWARA_ROOT  : {COSWARA_ROOT}")

if not all([DATA_ROOT, DIAG_FILE, M2_CKPT]):
    raise RuntimeError(
        "Missing a REQUIRED input (ICBHI audio, patient_diagnosis.csv, or the M2/M12 checkpoint). "
        "Regime A cannot run without all three. Run Cell 0b, and upload the M2 checkpoint "
        "(Asif's/M2/best_model.pth) to /content/ or Drive.")

CFG = {
    # -- section 2 preprocessing: MUST match M2 exactly --
    "sample_rate": 16000, "duration_s": 8.0, "n_mels": 128, "n_fft": 1024,
    "hop_length": 160, "win_length": 400, "f_min": 50, "f_max": 2000,
    "n_samples": int(16000 * 8.0), "n_frames": None,

    # -- ICBHI open-set definition (Model_Training_Reference.md 2.8) --
    "known_classes": ["COPD", "Healthy", "URTI"],
    "unknown_classes": ["Bronchiectasis", "Pneumonia", "Bronchiolitis"],
    "excluded_classes": ["Asthma", "LRTI"],
    "sound_event_classes": ["Normal", "Crackle", "Wheeze", "Both"],
    "num_classes": 4,

    # -- evaluation --
    "known_test_fraction": 0.40,
    "batch_size": 32, "num_workers": 2, "seed": 42,
    "target_known_tpr": 0.95,
    "max_external_samples": 1500,   # per external dataset; None = no cap

    "data_root": DATA_ROOT, "diag_file": DIAG_FILE, "m2_ckpt": M2_CKPT,
    "sprsound_root": SPRSOUND_ROOT, "coswara_root": COSWARA_ROOT,
    "results_dir": RESULTS_DIR, "cache_dir": CACHE_DIR,
    "model_id": "M38", "contributor": "Asif",
}
CFG["n_frames"] = 1 + math.floor(CFG["n_samples"] / CFG["hop_length"])   # 801

print("\n" + "=" * 60)
for k, v in CFG.items():
    print(f"  {k:<22}: {v}")
print("=" * 60)
''')

code(r'''
# ============================================================
# CELL 2 — IMPORTS & SEED
# ============================================================
import json, time, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | seed {CFG['seed']}")
''')

# ===========================================================================
md(r"""
---
## Section 3 — Data

Three sources, one shared preprocessing path.

**ICBHI** is parsed exactly as M29 does: annotation files give respiratory cycles, `patient_diagnosis.csv`
gives the known/unknown grouping, and the known group is split patient-independently into a fit half
(used to fit the Mahalanobis scorers) and a test half (the in-distribution side of every comparison).
The 19 unknown patients are **evaluation-only and never fitted on** — asserted in code, not assumed.

**SPRSound and Coswara** have no cycle annotations, so each audio file is one sample: load the first
8 s, tile if shorter. Where a participant/patient grouping is recoverable from the directory layout
or filename it is used, so scores aggregate at the same level as ICBHI's; where it isn't, the
notebook says so explicitly rather than silently mixing evaluation levels. That mix-up is a mistake
this project has already made once — `Asif's/CLAUDE.md` lists cycle-vs-patient level confusion as
trap #6.
""")

code(r'''
# ============================================================
# CELL 3 — PARSE ICBHI: CYCLES + PATIENT DIAGNOSES
# ============================================================

def parse_annotation_file(txt_path):
    """One ICBHI annotation file -> list of cycles with 4-class sound-event labels."""
    cycles = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if end <= start:
                continue
            label = (0 if (crackle == 0 and wheeze == 0) else
                     1 if (crackle == 1 and wheeze == 0) else
                     2 if (crackle == 0 and wheeze == 1) else 3)
            cycles.append({"start": start, "end": end, "label": label})
    return cycles


def load_diagnoses(diag_path):
    """patient_id -> diagnosis string. Handles the .csv and whitespace .txt variants."""
    diag = {}
    with open(diag_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 2:
                continue
            try:
                diag[int(parts[0])] = parts[1].strip()
            except ValueError:
                continue          # header row
    return diag


def build_dataframe(data_root, diag_path, cfg):
    diag = load_diagnoses(diag_path)
    print(f"Loaded {len(diag)} patient diagnoses")

    rows = []
    for wav in sorted(glob.glob(os.path.join(data_root, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(data_root, stem + ".txt")
        if not os.path.exists(txt):
            continue
        try:
            pid = int(stem.split("_")[0])
        except ValueError:
            continue
        dx = diag.get(pid)
        if dx is None:
            continue
        if dx in cfg["known_classes"]:
            grp = "known"
        elif dx in cfg["unknown_classes"]:
            grp = "unknown"
        else:
            continue                      # Asthma / LRTI -- documented exclusion
        for c in parse_annotation_file(txt):
            rows.append({"wav_path": wav, "patient_id": pid, "diagnosis": dx,
                         "group": grp, **c})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No cycles parsed -- check DATA_ROOT and DIAG_FILE.")
    return df


print(f"Parsing ICBHI from {CFG['data_root']}\n")
df = build_dataframe(CFG["data_root"], CFG["diag_file"], CFG)

pat = df.groupby("patient_id").agg(group=("group", "first"),
                                   diagnosis=("diagnosis", "first"),
                                   n_cycles=("label", "size")).reset_index()
known_pat = pat[pat.group == "known"]
unknown_pat = pat[pat.group == "unknown"]

print(f"Cycles      : {len(df)}")
print(f"Known       : {len(known_pat):>3} patients / {int(known_pat.n_cycles.sum()):>5} cycles")
print(f"Unknown     : {len(unknown_pat):>3} patients / {int(unknown_pat.n_cycles.sum()):>5} cycles")

# ---- patient-independent split of the KNOWN group (unknowns are all evaluation) ----
rng = np.random.RandomState(CFG["seed"])
kp = known_pat.patient_id.values.copy()
rng.shuffle(kp)
n_test = max(1, int(round(len(kp) * CFG["known_test_fraction"])))
known_test_ids = set(kp[:n_test].tolist())
known_fit_ids = set(kp[n_test:].tolist())

assert not (known_fit_ids & known_test_ids), "Patient leakage between fit and test."
assert not (set(unknown_pat.patient_id) & known_fit_ids), \
    "An unknown patient is in the fitting set -- the unknown group must never be fitted on."
assert not (set(unknown_pat.patient_id) & known_test_ids), "Group overlap."
print(f"\n[OK] Protocol section 1 verified: fit {len(known_fit_ids)} / "
      f"test {len(known_test_ids)} known patients, disjoint; "
      f"{len(unknown_pat)} unknown patients used for EVALUATION ONLY.")

df["split"] = df.patient_id.map(
    lambda p: "fit" if p in known_fit_ids else ("known_test" if p in known_test_ids
                                                else "unknown_test"))
print(df.groupby("split").size().to_string())
''')

code(r'''
# ============================================================
# CELL 4 — LOG-MEL EXTRACTION (identical to M2)
# ============================================================
# Preprocessing MUST be byte-identical to M2's, otherwise these are not the
# M12 backbone's embeddings and the whole experiment is mis-specified.

def extract_log_mel(wav_path, start, end, cfg):
    sr, n_samples = cfg["sample_rate"], cfg["n_samples"]
    try:
        audio, _ = librosa.load(wav_path, sr=sr, offset=start,
                                duration=max(end - start, 0.05), mono=True)
    except Exception as e:
        # A silent all-zero spectrogram here would be trained on and
        # scored as a real cycle. Fail instead of substituting
        # (Model_Training_Protocol.md section 1.2).
        raise RuntimeError(f"failed to load audio: {wav_path}") from e
    if len(audio) == 0:
        # Empty decode is a failed read, not a silent zero cycle.
        raise RuntimeError(f"empty audio decoded from audio: {wav_path}")
    if len(audio) < n_samples:
        audio = np.tile(audio, math.ceil(n_samples / len(audio)))[:n_samples]
    else:
        audio = audio[:n_samples]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], win_length=cfg["win_length"],
        fmin=cfg["f_min"], fmax=cfg["f_max"], power=2.0)
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    T = lm.shape[1]
    lm = (np.pad(lm, ((0, 0), (0, cfg["n_frames"] - T)), mode="constant")
          if T < cfg["n_frames"] else lm[:, :cfg["n_frames"]])
    return lm[np.newaxis].astype(np.float32)


class CycleDataset(Dataset):
    """ICBHI: one item per annotated respiratory cycle."""

    def __init__(self, frame, cfg):
        self.df = frame.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        spec = extract_log_mel(r["wav_path"], r["start"], r["end"], self.cfg)
        return (torch.from_numpy(spec), int(r["label"]), int(r["patient_id"]))


class FileDataset(Dataset):
    """External OOD sets: one item per audio FILE (first 8 s, tiled if shorter).
    `group_idx` is the integer-encoded participant grouping so scores can be
    aggregated at the same level ICBHI uses."""

    def __init__(self, paths, group_idx, cfg):
        self.paths = list(paths)
        self.group_idx = list(group_idx)
        self.cfg = cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        spec = extract_log_mel(self.paths[i], 0.0, self.cfg["duration_s"], self.cfg)
        return (torch.from_numpy(spec), 0, int(self.group_idx[i]))


loader = DataLoader(CycleDataset(df, CFG), batch_size=CFG["batch_size"], shuffle=False,
                    num_workers=CFG["num_workers"], pin_memory=torch.cuda.is_available())
print(f"ICBHI batches: {len(loader)}  ({len(df)} cycles)")
_x, _y, _p = next(iter(loader))
print(f"Batch spec {tuple(_x.shape)} | expected (B, 1, {CFG['n_mels']}, {CFG['n_frames']})")
assert _x.shape[1:] == (1, CFG["n_mels"], CFG["n_frames"]), "Spectrogram shape mismatch vs M2."
''')

code(r'''
# ============================================================
# CELL 5 — EXTERNAL OOD SETS: SPRSound (near) + Coswara (far)
# ============================================================

def collect_external(root, name, cfg, group_from="parent"):
    """
    Recursively collect .wav files under `root` and derive a participant grouping.

    group_from:
      "parent"      -- the containing folder is the participant (Coswara's layout:
                       coswara_data/kaggle_data/<PARTICIPANT_ID>/*.wav)
      "stem_prefix" -- the filename prefix before the first '_' is the patient
                       (SPRSound's convention)

    Returns (paths, group_labels, grouping_level) where grouping_level is
    "participant" if the grouping is genuinely coarser than the file list, else "file".
    """
    if not root or not os.path.isdir(root):
        print(f"[{name}] root not available -- regime will be reported as unavailable.")
        return [], [], "unavailable"

    paths = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not paths:
        print(f"[{name}] no .wav files under {root} -- regime unavailable.")
        return [], [], "unavailable"

    if group_from == "parent":
        groups = [os.path.basename(os.path.dirname(p)) for p in paths]
    else:
        groups = [os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in paths]

    n_groups = len(set(groups))
    # If the "grouping" is 1:1 with files it isn't a grouping at all -- say so instead of
    # implying a patient-level evaluation we cannot actually perform.
    level = "participant" if n_groups < len(paths) else "file"
    if level == "file":
        print(f"[{name}] WARNING: could not recover a participant grouping "
              f"({n_groups} groups for {len(paths)} files). Evaluating at FILE level. "
              f"This is a coarser-grained comparison than ICBHI's patient level -- "
              f"noted in the results JSON.")

    # Deterministic cap so runtime is bounded and reproducible.
    cap = cfg.get("max_external_samples")
    if cap is not None and len(paths) > cap:
        idx = np.random.RandomState(cfg["seed"]).choice(len(paths), cap, replace=False)
        idx = np.sort(idx)
        paths = [paths[i] for i in idx]
        groups = [groups[i] for i in idx]
        print(f"[{name}] capped at {cap} files (of {len(idx)} sampled, seed {cfg['seed']}).")

    print(f"[{name}] {len(paths)} files / {len(set(groups))} {level}s under {root}")
    return paths, groups, level


sprsound_paths, sprsound_groups, sprsound_level = collect_external(
    CFG["sprsound_root"], "SPRSound (near-OOD)", CFG, group_from="stem_prefix")

coswara_paths, coswara_groups, coswara_level = collect_external(
    CFG["coswara_root"], "Coswara (far-OOD)", CFG, group_from="parent")

# Sanity check against the M19 bug this notebook exists partly to correct.
if coswara_paths and len(coswara_paths) < 10:
    print(f"\n*** WARNING: only {len(coswara_paths)} Coswara files found. M19 hit exactly this "
          f"failure by pointing at a single participant folder. Verify COSWARA_ROOT is the "
          f"PARENT directory containing many participant folders, not one of them. ***")

EXTERNAL = {}
if sprsound_paths:
    EXTERNAL["SPRSound"] = dict(paths=sprsound_paths, groups=sprsound_groups,
                                level=sprsound_level, kind="near_ood")
if coswara_paths:
    EXTERNAL["Coswara"] = dict(paths=coswara_paths, groups=coswara_groups,
                               level=coswara_level, kind="far_ood")

print(f"\nExternal OOD regimes available: {list(EXTERNAL.keys()) or 'NONE (regime A only)'}")
''')

# ===========================================================================
md(r"""
---
## Section 4 — The frozen M12 backbone

The architecture is read from the checkpoint's own `model_config` rather than hardcoded, so this
cell keeps working if the M12 selection is ever re-run with different hyperparameters. Nothing is
trained or fine-tuned here — every parameter stays frozen, exactly as in M29.
""")

code(r'''
# ============================================================
# CELL 6 — M2 / M12 BACKBONE (architecture read from the checkpoint)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(pool))

    def forward(self, x):
        return self.block(x)


class M2_CNN(nn.Module):
    """Identical definition to Asif's/M2 -- required to load its state_dict."""

    def __init__(self, num_classes=4, depth=4, base_width=32, dropout=0.5, fc_dim=128):
        super().__init__()
        channels = [base_width * (2 ** i) for i in range(depth)]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch)); in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(channels[-1], fc_dim),
                                  nn.ReLU(inplace=True), nn.Linear(fc_dim, num_classes))
        self.embedding_dim = channels[-1]

    def forward(self, x):
        return self.head(self.dropout(self.gap(self.encoder(x)).flatten(1)))

    def get_embedding(self, x):
        return self.gap(self.encoder(x)).flatten(1)


state = torch.load(CFG["m2_ckpt"], map_location=DEVICE, weights_only=False)   # protocol 11.A
mcfg = state.get("model_config", {"depth": 5, "base_width": 48, "dropout": 0.4})
print(f"Checkpoint model_config : {mcfg}")
print(f"Checkpoint epoch        : {state.get('epoch')}")
print(f"Checkpoint best_score   : {state.get('best_score')}")

backbone = M2_CNN(num_classes=CFG["num_classes"], depth=int(mcfg["depth"]),
                  base_width=int(mcfg["base_width"]),
                  dropout=float(mcfg.get("dropout", 0.4))).to(DEVICE)
missing, unexpected = backbone.load_state_dict(state["model_state"], strict=False)
assert not missing, f"Checkpoint is missing weights for: {missing[:5]}"
backbone.eval()

N_PARAMS = sum(p.numel() for p in backbone.parameters())
print(f"\nBackbone loaded: {N_PARAMS:,} params, embedding dim {backbone.embedding_dim}")
print("This is the frozen M12 backbone -- no fine-tuning happens in this notebook.")
''')

code(r'''
# ============================================================
# CELL 7 — EXTRACT LOGITS + EMBEDDINGS (ICBHI, then each external set)
# ============================================================

def extract_features(data_loader, desc):
    logits_l, embs_l, grp_l = [], [], []
    with torch.no_grad():
        for specs, _labels, grp in tqdm(data_loader, desc=desc):
            specs = specs.to(DEVICE, non_blocking=True)
            emb = backbone.get_embedding(specs)
            logits_l.append(backbone.head(emb).cpu().numpy())
            embs_l.append(emb.cpu().numpy())
            grp_l.append(np.asarray(grp))
    return (np.concatenate(logits_l), np.concatenate(embs_l), np.concatenate(grp_l))


t0 = time.time()
LOGITS, EMBS, PIDS = extract_features(loader, "ICBHI")
LABELS = df["label"].values
SPLIT = df["split"].values
assert len(SPLIT) == len(LOGITS), "Row alignment broken between df and extracted tensors."
print(f"ICBHI: {LOGITS.shape[0]} cycles in {time.time() - t0:.0f}s")

# --- inference latency, measured on ICBHI (protocol section 3 efficiency metrics) ---
_probe = torch.randn(CFG["batch_size"], 1, CFG["n_mels"], CFG["n_frames"]).to(DEVICE)
with torch.no_grad():
    for _ in range(3):
        backbone(_probe)                       # warm up
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    _t = time.time()
    for _ in range(10):
        backbone(_probe)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
INFER_MS = (time.time() - _t) / (10 * CFG["batch_size"]) * 1000
print(f"Inference: {INFER_MS:.3f} ms/sample")

# --- external sets ---
for name, blob in EXTERNAL.items():
    uniq = sorted(set(blob["groups"]))
    g2i = {g: i for i, g in enumerate(uniq)}
    ext_loader = DataLoader(
        FileDataset(blob["paths"], [g2i[g] for g in blob["groups"]], CFG),
        batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=torch.cuda.is_available())
    t0 = time.time()
    lg, em, gi = extract_features(ext_loader, name)
    blob.update(logits=lg, embs=em, group_idx=gi, index_to_group=uniq)
    print(f"{name}: {lg.shape[0]} files / {len(uniq)} {blob['level']}s "
          f"in {time.time() - t0:.0f}s")
''')

# ===========================================================================
md(r"""
---
## Section 5 — The four baseline scores

Identical to M29: MSP, entropy, energy, and Mahalanobis distance. Every scorer is **fit on ICBHI
known-fit cycles only**, then applied unchanged to all three OOD pools. Higher score = more likely
unknown, for all four.

Fitting on the known-fit half and nothing else is what makes this a legitimate open-set evaluation
rather than a leak: no OOD pool — ICBHI unknowns, SPRSound, or Coswara — contributes anything to
the fitted statistics.
""")

code(r'''
# ============================================================
# CELL 8 — SCORE FUNCTIONS (identical to M29)
# ============================================================

def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def score_msp(logits):
    """1 - max softmax probability (Hendrycks & Gimpel). Higher = more unknown."""
    return 1.0 - softmax_np(logits).max(axis=1)


def score_entropy(logits):
    """Shannon entropy of the softmax, normalised to [0, 1]."""
    p = softmax_np(logits)
    h = -(p * np.log(p + 1e-12)).sum(axis=1)
    return h / np.log(p.shape[1])


def score_energy(logits):
    """-logsumexp(logits) -- the energy-based OOD score (Liu et al.)."""
    m = logits.max(axis=1, keepdims=True)
    lse = (m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1)))
    return -lse


def fit_mahalanobis_class(embs_fit, labels_fit, n_classes):
    """Class-conditional Gaussians with a shared covariance. Fitted on known-fit cycles only."""
    means, centered = [], []
    for c in range(n_classes):
        m = embs_fit[labels_fit == c]
        if len(m) < 2:
            means.append(embs_fit.mean(axis=0))
            continue
        mu = m.mean(axis=0)
        means.append(mu)
        centered.append(m - mu)
    cov = np.cov(np.vstack(centered).T) if centered else np.cov(embs_fit.T)
    cov = np.atleast_2d(cov) + np.eye(np.atleast_2d(cov).shape[0]) * 1e-6
    return np.array(means), np.linalg.pinv(cov)


def score_mahalanobis_class(embs, means, prec):
    """Minimum Mahalanobis distance to any known class centroid."""
    d = [np.einsum("ij,jk,ik->i", embs - mu, prec, embs - mu) for mu in means]
    return np.min(np.stack(d, axis=1), axis=1)


def fit_mahalanobis_single(x_fit):
    """One Gaussian over the fitting distribution (used at group level)."""
    mu = x_fit.mean(axis=0)
    cov = np.atleast_2d(np.cov(x_fit.T))
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    return mu, np.linalg.pinv(cov)


def score_mahalanobis_single(x, mu, prec):
    diff = x - mu
    return np.einsum("ij,jk,ik->i", diff, prec, diff)


def aggregate_by_group(values, groups):
    """Sample-level -> group-level by mean. Returns (group_ids, scores) sorted by id."""
    order = np.unique(groups)
    return order, np.array([values[groups == g].mean() for g in order])


print("Score functions defined: MSP, entropy, energy, Mahalanobis (class + group).")
''')

code(r'''
# ============================================================
# CELL 9 — FIT ON ICBHI KNOWN-FIT ONLY, THEN SCORE EVERY POOL
# ============================================================
fit_mask = SPLIT == "fit"
print(f"Fitting on {fit_mask.sum()} known-fit cycles "
      f"({len(np.unique(PIDS[fit_mask]))} patients). "
      f"No OOD pool contributes to any fitted statistic.")

means_c, prec_c = fit_mahalanobis_class(EMBS[fit_mask], LABELS[fit_mask], CFG["num_classes"])
mu_p, prec_p = fit_mahalanobis_single(
    np.array([EMBS[fit_mask][PIDS[fit_mask] == p].mean(axis=0)
              for p in np.unique(PIDS[fit_mask])]))

SCORE_NAMES = ["MSP", "Entropy", "Energy", "Mahalanobis_class", "Mahalanobis_patient"]


def score_pool(logits, embs, groups):
    """All five scores for one pool, aggregated to group level.
    Returns {score_name: (group_ids, group_scores)}."""
    cyc = {
        "MSP": score_msp(logits),
        "Entropy": score_entropy(logits),
        "Energy": score_energy(logits),
        "Mahalanobis_class": score_mahalanobis_class(embs, means_c, prec_c),
    }
    out = {n: aggregate_by_group(v, groups) for n, v in cyc.items()}

    gids = np.unique(groups)
    gmeans = np.array([embs[groups == g].mean(axis=0) for g in gids])
    out["Mahalanobis_patient"] = (gids, score_mahalanobis_single(gmeans, mu_p, prec_p))
    return out


# --- in-distribution side: ICBHI known-test patients (shared by all three regimes) ---
kt = SPLIT == "known_test"
ID_SCORES = score_pool(LOGITS[kt], EMBS[kt], PIDS[kt])
N_ID = len(ID_SCORES["Energy"][0])

# --- OOD pools ---
OOD_SCORES = {}

ut = SPLIT == "unknown_test"
OOD_SCORES["ICBHI_unknown"] = dict(scores=score_pool(LOGITS[ut], EMBS[ut], PIDS[ut]),
                                   kind="near_ood", level="patient",
                                   source="ICBHI 2017 held-out disease classes")

for name, blob in EXTERNAL.items():
    OOD_SCORES[name] = dict(
        scores=score_pool(blob["logits"], blob["embs"], blob["group_idx"]),
        kind=blob["kind"], level=blob["level"],
        source=("SPRSound pediatric respiratory audio" if name == "SPRSound"
                else "Coswara heavy-cough (phone microphone)"))

print(f"\nIn-distribution : {N_ID} ICBHI known-test patients")
for name, blob in OOD_SCORES.items():
    n = len(blob["scores"]["Energy"][0])
    print(f"OOD pool        : {name:<16} n={n:<6} [{blob['kind']}, {blob['level']}-level]")
''')

# ===========================================================================
md(r"""
---
## Section 6 — Evaluation, with confidence intervals

Each regime pits the **same** in-distribution pool (ICBHI known-test patients) against a different
OOD pool, so the only thing that varies between regimes is what is being detected.

Every AUROC gets a Hanley & McNeil (1982) 95% confidence interval computed **here**, not after the
fact. A point estimate on its own is what produced the situation `SIGNIFICANCE_REPORT.md` had to
document; the interval is the number that actually answers "is this better than chance?"
""")

code(r'''
# ============================================================
# CELL 10 — AUROC + 95% CI PER REGIME, PER SCORE
# ============================================================

def hanley_mcneil_se(auc, n_pos, n_neg):
    """SE of AUROC, Hanley & McNeil (1982). n_pos = OOD count, n_neg = in-distribution count."""
    q1 = auc / (2 - auc)
    q2 = (2 * auc ** 2) / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def evaluate(id_scores, ood_scores, name):
    """One regime: every score, AUROC + AUPR + 95% CI + operating point."""
    rows = []
    for s in SCORE_NAMES:
        id_v = id_scores[s][1]
        ood_v = ood_scores[s][1]
        y = np.concatenate([np.zeros(len(id_v)), np.ones(len(ood_v))])
        v = np.concatenate([id_v, ood_v])
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, v))
        aupr = float(average_precision_score(y, v))
        se = hanley_mcneil_se(auc, len(ood_v), len(id_v))
        lo, hi = max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)

        # operating point: threshold retaining target_known_tpr of in-distribution samples
        thr = float(np.quantile(id_v, CFG["target_known_tpr"]))
        flagged = v >= thr
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / int((y == 1).sum())

        rows.append(dict(regime=name, score=s, auroc=round(auc, 4), aupr=round(aupr, 4),
                         se=round(se, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
                         beats_chance=bool(lo > 0.5),
                         n_ood=len(ood_v), n_id=len(id_v),
                         threshold=round(thr, 6),
                         unknown_precision=round(prec, 4), unknown_recall=round(rec, 4)))
    return rows


ALL_ROWS = []
for name, blob in OOD_SCORES.items():
    ALL_ROWS += evaluate(ID_SCORES, blob["scores"], name)

RESULTS_DF = pd.DataFrame(ALL_ROWS)

for name in OOD_SCORES:
    sub = RESULTS_DF[RESULTS_DF.regime == name].sort_values("auroc", ascending=False)
    blob = OOD_SCORES[name]
    print("\n" + "=" * 78)
    print(f"REGIME: {name}   [{blob['kind']}, {blob['level']}-level]   "
          f"n_ood={sub.n_ood.iloc[0]} vs n_id={sub.n_id.iloc[0]}")
    print("=" * 78)
    for _, r in sub.iterrows():
        mark = "  <-- CI excludes chance" if r.beats_chance else ""
        print(f"  {r.score:<22} AUROC={r.auroc:.4f}  "
              f"95% CI=[{r.ci95_lo:.4f}, {r.ci95_hi:.4f}]  AUPR={r.aupr:.4f}{mark}")

print("\n" + "=" * 78)
n_sig = int(RESULTS_DF.beats_chance.sum())
print(f"{n_sig} of {len(RESULTS_DF)} (regime x score) combinations have a 95% CI "
      f"excluding chance.")
print("=" * 78)
''')

code(r'''
# ============================================================
# CELL 11 — PLOTS: forest plot + ROC curves per regime
# ============================================================
plt.rcParams["figure.dpi"] = 150

# ---- Forest plot: every regime x score with its CI ----
plot_df = RESULTS_DF.sort_values(["regime", "auroc"]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 0.34 * len(plot_df) + 2))
ypos = np.arange(len(plot_df))
ax.errorbar(plot_df.auroc, ypos,
            xerr=[plot_df.auroc - plot_df.ci95_lo, plot_df.ci95_hi - plot_df.auroc],
            fmt="none", ecolor="#999999", elinewidth=1.4, capsize=3)
for i, r in plot_df.iterrows():
    ax.plot(r.auroc, i, "o", ms=6,
            color=("#1b7f3a" if r.beats_chance else "#c0392b"), zorder=5)
ax.axvline(0.5, color="black", ls="--", lw=1, label="chance")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.regime} / {r.score}" for _, r in plot_df.iterrows()], fontsize=7)
ax.set_xlim(0, 1)
ax.set_xlabel("AUROC (95% CI, Hanley-McNeil)")
ax.set_title("M38 — open-set AUROC by regime and score\n"
             "green = CI excludes chance, red = CI includes chance", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CFG["results_dir"], "auroc_forest_M38.png"))
plt.show()

# ---- ROC curves, best score per regime ----
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, blob in OOD_SCORES.items():
    sub = RESULTS_DF[RESULTS_DF.regime == name]
    if sub.empty:
        continue
    best = sub.sort_values("auroc", ascending=False).iloc[0]
    id_v = ID_SCORES[best.score][1]
    ood_v = blob["scores"][best.score][1]
    y = np.concatenate([np.zeros(len(id_v)), np.ones(len(ood_v))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([id_v, ood_v]))
    ax.plot(fpr, tpr, lw=1.8,
            label=f"{name} — {best.score} (AUROC {best.auroc:.3f}, n={best.n_ood})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
ax.set_xlabel("False positive rate (known flagged as unknown)")
ax.set_ylabel("True positive rate (unknown detected)")
ax.set_title("M38 — best scorer per regime")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(CFG["results_dir"], "roc_curves_M38.png"))
plt.show()

# ---- Score distributions for the best overall scorer ----
best_overall = RESULTS_DF.sort_values("auroc", ascending=False).iloc[0].score
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ID_SCORES[best_overall][1], bins=25, alpha=0.6, label=f"ICBHI known (n={N_ID})",
        density=True, color="#2c7fb8")
for name, blob in OOD_SCORES.items():
    v = blob["scores"][best_overall][1]
    ax.hist(v, bins=25, alpha=0.45, label=f"{name} (n={len(v)})", density=True)
ax.set_xlabel(f"{best_overall} score (higher = more unknown)")
ax.set_ylabel("density")
ax.set_title(f"M38 — score distributions ({best_overall})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CFG["results_dir"], "score_distributions_M38.png"))
plt.show()

print(f"Plots written to {CFG['results_dir']}")
''')

code(r'''
# ============================================================
# CELL 12 — RAW SCORE DUMP  (unlocks paired DeLong tests)
# ============================================================
# No model in this project has ever saved raw per-sample scores, which is why
# Asif's/Statistics/SIGNIFICANCE_REPORT.md had to fall back on an analytic
# approximation instead of a proper paired DeLong test. This fixes that for
# every future comparison against M38.

dump = []
for s in SCORE_NAMES:
    gids, vals = ID_SCORES[s]
    for g, v in zip(gids, vals):
        dump.append(dict(regime="in_distribution", pool="ICBHI_known_test",
                         group_id=str(g), score_name=s, score=float(v), is_ood=0))
    for name, blob in OOD_SCORES.items():
        gids, vals = blob["scores"][s]
        idx2g = EXTERNAL.get(name, {}).get("index_to_group")
        for g, v in zip(gids, vals):
            gid = idx2g[int(g)] if idx2g is not None else str(g)
            dump.append(dict(regime=name, pool=name, group_id=str(gid),
                             score_name=s, score=float(v), is_ood=1))

SCORES_CSV = os.path.join(CFG["results_dir"], "scores_M38.csv")
pd.DataFrame(dump).to_csv(SCORES_CSV, index=False)
print(f"Wrote {len(dump)} raw scores to {SCORES_CSV}")
print("\nA paired DeLong test against any other method now only needs that method to dump")
print("the same columns for the same group_ids. See SIGNIFICANCE_REPORT.md limitation 1.")
print(pd.DataFrame(dump).groupby(["regime", "score_name"]).size().head(12).to_string())
''')

# ===========================================================================
md(r"""
---
## Section 7 — Protocol-compliant results JSON
""")

code(r'''
# ============================================================
# CELL 13 — EXPORT results_M38.json
# ============================================================
best_row = RESULTS_DF.sort_values("auroc", ascending=False).iloc[0]

per_regime = {}
for name, blob in OOD_SCORES.items():
    sub = RESULTS_DF[RESULTS_DF.regime == name].sort_values("auroc", ascending=False)
    if sub.empty:
        continue
    top = sub.iloc[0]
    per_regime[name] = {
        "kind": blob["kind"],
        "evaluation_level": blob["level"],
        "source": blob["source"],
        "n_ood": int(top.n_ood),
        "n_in_distribution": int(top.n_id),
        "best_score_name": top.score,
        "auroc": float(top.auroc),
        "auroc_ci95": [float(top.ci95_lo), float(top.ci95_hi)],
        "auroc_ci_excludes_chance": bool(top.beats_chance),
        "aupr": float(top.aupr),
        "unknown_precision": float(top.unknown_precision),
        "unknown_recall": float(top.unknown_recall),
        "all_scores": sub.drop(columns=["regime"]).to_dict(orient="records"),
    }

results = {
    "meta": {
        "model_id": "M38",
        "model_name": "Large-N Open-Set Baseline Suite (near-OOD and far-OOD)",
        "contributor": CFG["contributor"],
        "date_completed": time.strftime("%Y-%m-%d"),
        "is_augmented": False,
        "augmentation_method": "none",
        "notes": (
            "M29's four post-hoc OOD scorers (MSP, entropy, energy, Mahalanobis) on the frozen "
            "M12 backbone, evaluated against THREE OOD pools instead of one, with a Hanley-McNeil "
            "95% CI on every AUROC. The in-distribution side (ICBHI known-test patients) is held "
            "identical across regimes so the only variable is what is being detected. No training "
            "occurs. Exists because Asif's/Statistics/SIGNIFICANCE_REPORT.md showed every "
            "open-set AUROC in the project has a 95% CI including chance at n=19 unknown "
            "patients -- the large-N stress test Novelty Search.md section 1 claims as the "
            "project's strongest contribution had never actually been run. "
            "IMPORTANT INTERPRETATION NOTE: the Coswara regime is a FAR-OOD CONTROL, not an "
            "unknown-disease result. Coswara is phone-recorded voluntary cough; ICBHI is "
            "stethoscope auscultation. A detector can separate them on recording modality alone. "
            "A high Coswara AUROC is evidence of domain separability, NOT of disease-level "
            "unknown detection -- see Novelty Search.md section 2 Attack 3. It is diagnostic in "
            "the negative direction: failing far-OOD invalidates near-OOD results from the same "
            "detector. "
            "Also corrects a bug in M19, which configured the Coswara path as "
            ".../kaggle_data/<ONE_PARTICIPANT_ID>/ and therefore evaluated 2 files; M19's "
            "Coswara AUROC of 0.4881 should not be cited."),
    },
    "config": {k: v for k, v in CFG.items() if k not in ("results_dir", "cache_dir")},
    "environment": {
        "platform": ("Google Colab" if IN_COLAB else
                     "Kaggle" if os.path.exists("/kaggle/working") else "Local"),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    },
    "dataset_info": {
        "dataset": "ICBHI_2017 + SPRSound + Coswara",
        "data_source": "real_audio",
        "known_classes": CFG["known_classes"],
        "unknown_classes": CFG["unknown_classes"],
        "excluded_classes": CFG["excluded_classes"],
        "excluded_rationale": (
            "Asthma (n=1) and LRTI (n=2) are too few to place in either group without "
            "distorting it, and the reference's open-set protocol names neither."),
        "n_known_patients": int(len(known_pat)),
        "n_unknown_patients": int(len(unknown_pat)),
        "n_fit_patients": int(len(known_fit_ids)),
        "n_known_test_patients": int(len(known_test_ids)),
        "total_icbhi_cycles": int(len(df)),
        "split_method": "patient_independent_known_60_40; all OOD pools evaluation-only",
        "patient_leakage_verified": True,
        "unknown_group_never_fitted": True,
        "external_ood_never_fitted": True,
        "evaluation_level": "patient (ICBHI); see per-regime evaluation_level for external sets",
        "max_external_samples": CFG["max_external_samples"],
        "regimes_available": list(OOD_SCORES.keys()),
    },
    "efficiency": {
        "total_params": int(N_PARAMS),
        "trainable_params": 0,
        "model_size_mb": round(os.path.getsize(CFG["m2_ckpt"]) / (1024 * 1024), 2),
        "training_time_total_s": 0,
        "training_time_per_epoch_s_avg": 0,
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "inference_time_ms_per_sample": round(float(INFER_MS), 3),
        "note": "Backbone frozen and inherited from M12; M38 trains nothing.",
    },
    "best_epoch": {
        "epoch": None,
        "primary_metric": "auroc",
        "primary_metric_value": float(best_row.auroc),
    },
    "best_metrics": {
        "open_set": {
            "best_regime": best_row.regime,
            "best_score_name": best_row.score,
            "auroc": float(best_row.auroc),
            "auroc_ci95": [float(best_row.ci95_lo), float(best_row.ci95_hi)],
            "auroc_ci_excludes_chance": bool(best_row.beats_chance),
            "aupr": float(best_row.aupr),
            "unknown_precision": float(best_row.unknown_precision),
            "unknown_recall": float(best_row.unknown_recall),
            "n_ood": int(best_row.n_ood),
            "n_in_distribution": int(best_row.n_id),
        },
        "per_regime": per_regime,
        "reference_M29_energy": {
            "auroc": 0.6466,
            "auroc_ci95": [0.4917, 0.8015],
            "n_ood": 19, "n_in_distribution": 42,
            "source": "Asif's/M29/results_M29.json; CI from Asif's/Statistics/",
            "note": ("Cited for context as this notebook's direct n=19 predecessor, not as a "
                     "validity benchmark -- its own CI includes chance."),
        },
    },
    "ablation": {
        "ablation_group": "ood_generalization",
        "ablation_role": "baseline",
        "baseline_model_id": "M29",
        "variable_changed": ("OOD evaluation pool: large-N external sets (SPRSound near-OOD, "
                             "Coswara far-OOD) in addition to ICBHI's 19 unknown patients"),
        "variables_held_constant": [
            "backbone: frozen M12 (M2 checkpoint)",
            "scorers: MSP/entropy/energy/Mahalanobis, identical to M29",
            "preprocessing: 128mel_16kHz_8s, identical to M2",
            "in_distribution_pool: ICBHI known-test patients",
            "seed: 42",
            "training: none (post-hoc scoring only)",
        ],
        "known_deviations": [
            ("External OOD sets have no cycle annotations, so each audio file is one sample "
             "(first 8 s, tiled if shorter) rather than one annotated respiratory cycle."),
            ("Where a participant grouping is not recoverable from an external dataset's layout, "
             "that regime is evaluated at file level rather than patient level and says so in "
             "per_regime.evaluation_level. Do not compare a file-level AUROC directly against a "
             "patient-level one."),
        ],
        "component_flags": {
            "has_sound_event_head": True,
            "has_disease_head": False,
            "has_cross_task_consistency": False,
            "has_cqkd_regularization": False,
            "has_openmax_rejection": False,
            "owl_stage": 1,
            "compression_clusters": None,
        },
        "loss_weights": {
            "sound_event_weight": 1.0, "disease_weight": None, "consistency_weight": None,
        },
        "purpose": ("Delivers the large-N OOD stress test that Novelty Search.md section 1 names "
                    "as the project's strongest contribution but which had never been run. "
                    "Supplies the trivial-baseline control that M19's external-dataset "
                    "evaluation lacks."),
    },
    "training_history": [],
    "raw_scores_file": "scores_M38.csv",
}

results_path = os.path.join(CFG["results_dir"], "results_M38.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {results_path}")
print(json.dumps(results["best_metrics"]["open_set"], indent=2))
''')

code(r'''
# ============================================================
# FINAL CELL — TEAM HANDOFF & ONE-CLICK FILE DOWNLOADS
# ============================================================
import shutil

try:
    from IPython.display import display, FileLink
except ImportError:                      # plain-python execution (e.g. CI / test harness)
    display = FileLink = None

print("=" * 60)
print("OFFICIAL PROTOCOL OUTPUTS READY FOR DOWNLOAD")
print("=" * 60)

protocol_files = sorted(
    glob.glob(os.path.join(CFG["results_dir"], "results_M38.json")) +
    glob.glob(os.path.join(CFG["results_dir"], "scores_M38.csv")) +
    glob.glob(os.path.join(CFG["results_dir"], "*.png")))

for fpath in protocol_files:
    size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 3)
    print(f"Ready: {os.path.basename(fpath):<28} ({size_mb} MB)")
    if display is not None:
        display(FileLink(fpath))

if protocol_files:
    bundle_dir = os.path.join(BASE_DIR, "M38_bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    for fpath in protocol_files:
        shutil.copy2(fpath, os.path.join(bundle_dir, os.path.basename(fpath)))
    zip_path = shutil.make_archive(os.path.join(BASE_DIR, "M38_handoff_bundle"),
                                   "zip", bundle_dir)
    print(f"\nZIP bundle ({round(os.path.getsize(zip_path) / (1024 * 1024), 2)} MB): {zip_path}")
    if display is not None:
        display(FileLink(zip_path))
print("=" * 60)
''')

# ===========================================================================
md(r"""
---
## Section 8 — Summary & how to read this

Run the cell below after everything above has completed.
""")

code(r'''
# ============================================================
# CELL 14 — SUMMARY
# ============================================================
print("=" * 78)
print("M38 — LARGE-N OPEN-SET BASELINE SUITE — SUMMARY")
print("=" * 78)

print(f"\nBackbone   : frozen M12 ({N_PARAMS:,} params), nothing trained")
print(f"In-dist    : {N_ID} ICBHI known-test patients (identical across all regimes)")
print(f"\n{'Regime':<18}{'kind':<11}{'level':<13}{'n_ood':>7}  {'best score':<20}"
      f"{'AUROC':>8}  {'95% CI':<20}")
print("-" * 78)
for name, blob in OOD_SCORES.items():
    sub = RESULTS_DF[RESULTS_DF.regime == name].sort_values("auroc", ascending=False)
    if sub.empty:
        continue
    t = sub.iloc[0]
    ci = f"[{t.ci95_lo:.3f}, {t.ci95_hi:.3f}]"
    print(f"{name:<18}{blob['kind']:<11}{blob['level']:<13}{t.n_ood:>7}  "
          f"{t.score:<20}{t.auroc:>8.4f}  {ci:<20}")

print("\n" + "-" * 78)
print("COMPARISON TO M29 (the n=19 predecessor)")
print("-" * 78)
icbhi = RESULTS_DF[RESULTS_DF.regime == "ICBHI_unknown"].sort_values("auroc", ascending=False)
if not icbhi.empty:
    t = icbhi.iloc[0]
    print(f"  M38 regime A : {t.score} AUROC {t.auroc:.4f}  "
          f"[{t.ci95_lo:.3f}, {t.ci95_hi:.3f}]  (n_ood={t.n_ood})")
    print(f"  M29 published: Energy   AUROC 0.6466  [0.492, 0.802]  (n_ood=19)")
    print("  Regime A should reproduce M29 closely -- same backbone, same split, same scorers.")
    print("  A large gap means something drifted; investigate before trusting regimes B/C.")

print("\n" + "-" * 78)
print("HOW TO READ THIS")
print("-" * 78)
sig = RESULTS_DF[RESULTS_DF.beats_chance]
if sig.empty:
    print("  NO regime/score combination has a 95% CI excluding chance.")
    print("  That is a real, reportable finding: these post-hoc scores do not detect")
    print("  distribution shift on this backbone at ANY of the sample sizes tested --")
    print("  which is a stronger statement than M29 could make, because large N means")
    print("  the null result is no longer explainable by insufficient power.")
else:
    print(f"  {len(sig)} combination(s) have a 95% CI excluding chance:")
    for _, r in sig.sort_values("auroc", ascending=False).iterrows():
        print(f"    {r.regime:<16} {r.score:<20} AUROC {r.auroc:.4f} "
              f"[{r.ci95_lo:.3f}, {r.ci95_hi:.3f}]")
    far = sig[sig.regime == "Coswara"]
    near = sig[sig.regime.isin(["SPRSound", "ICBHI_unknown"])]
    if len(far) and not len(near):
        print("\n  READ THIS CAREFULLY: the only significant results are on Coswara (far-OOD).")
        print("  Coswara is phone-recorded cough vs. ICBHI's stethoscope auscultation, so this")
        print("  most likely reflects RECORDING MODALITY, not unknown-disease detection.")
        print("  Do not report it as evidence for the open-world mechanism. It does confirm")
        print("  the scorers are functional, which makes the near-OOD nulls more meaningful.")
    elif len(near):
        print("\n  At least one NEAR-OOD result is significant. That is the meaningful one --")
        print("  same modality, unseen population. This is the number worth reporting.")

print("\n  Raw per-sample scores: scores_M38.csv -- enables paired DeLong tests against")
print("  any future method that dumps the same columns.")
print("=" * 78)
''')

# ===========================================================================
md(r"""
---
### What to do with these outputs

1. **Check regime A against M29 first.** It should reproduce `AUROC 0.6466` closely. If it doesn't,
   stop and find out why before reading anything into regimes B and C — a drift there means the
   backbone, split, or preprocessing changed.
2. **Regime B (SPRSound) is the result that matters.** Same modality, unseen population, large N.
   Whatever it says — positive or null — it says with statistical power the project has never had.
3. **Regime C (Coswara) is a control.** A strong result there is expected and is *not* a finding.
   A weak result there is a serious problem, because it means the scorers can't separate even
   trivially different audio.
4. **Commit `scores_M38.csv` alongside the JSON.** It's small, and it's what makes paired
   significance testing possible for everyone else on the project.
5. Re-run `python3 "Asif's/audit/audit_project.py"` and confirm M38 lands clean before reporting it.
""")

# ===========================================================================
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Notebook JSON stores `source` as a list of lines, each keeping its trailing newline
# except the last. Splitting with keepends preserves that exactly and avoids the
# missing-newline cell-boundary corruption class of bug.
for c in NB["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "M38_largeN_openset.ipynb")
with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)
print(f"Wrote {OUT}  ({len(NB['cells'])} cells)")
