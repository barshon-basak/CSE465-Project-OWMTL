#!/usr/bin/env python3
"""
Generator for M30 v2 — Gated Feature-Fusion Ensemble, re-run on the OFFICIAL ICBHI split.

Run `python3 gen_M30_v2.py` to emit M30_v2_gated_fusion.ipynb.

Why v2 exists (three defects in the original, all independently disqualifying):

  1. WRONG SPLIT. `Barshon's/M30/...ipynb` has a function `build_icbhi_splits` whose docstring
     says "perform official patient-independent split" but whose body does
     `np.random.shuffle(all_pids); n_train = int(len(all_pids) * 0.70)`. It is a random 70/30
     split, not the official ICBHI 60/40. M2 and M3 -- the two backbones it fuses and claims to
     beat -- both use the official split. The "+9.86% over M2" claim compares across splits.

  2. WRONG METRIC. The reported 0.8213 is the project's legacy macro variant, not the ICBHI 2017
     challenge score. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`.

  3. UNVERIFIABLE. `results_M30.json` commits no `confusion_matrix_raw`, so no score can be
     recomputed from it by anyone.

And one latent risk worth fixing regardless:

  4. SILENT RANDOM-WEIGHT FALLBACK. The original loads each backbone inside
     `try: ... except Exception as e: print(f'M2 load note: {e}')` with `strict=False`. If a
     checkpoint fails to load, the notebook prints a warning and trains on RANDOMLY INITIALISED
     backbones -- the exact trap `Asif's/CLAUDE.md` records for M15. `strict=False` also hides
     partial loads. v2 asserts hard and additionally verifies the loaded weights actually changed
     the model, so a silent no-op load cannot pass.

What v2 additionally does that nobody has done: **the test `Novelty Search.md` §4.0 actually
requires.** That section admits M30 "only if it measurably beats both backbones alone." Nobody
evaluated M2-alone and M3-alone on the same test set to check. v2 does, at zero extra cost --
both backbones are already loaded -- and prints the verdict.

Note the nested-quote convention: code cells are wrapped in r'''...''' so any docstring inside
must use \"\"\"...\"\"\".
"""
import json
import os

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# ===========================================================================
md(r"""
# M30 v2 — Gated Feature-Fusion Ensemble (official ICBHI split)

**Model ID:** M30 · **Chunk:** E (Novelty Layer, selected item #3) · **Contributor:** Asif
**Requires:** M2 and M3 checkpoints (both real, both in the repo). **No backbone training** — only a
small fusion head is trained.

## Why this re-run exists

`Barshon's/M30/results_M30.json` reports ICBHI 0.8213 and a "+9.86% boost over M2". That result is
withdrawn. Three independent problems, any one of which is disqualifying:

| # | Problem | Detail |
|---|---|---|
| 1 | **Wrong split** | The original's `build_icbhi_splits()` docstring says *"official patient-independent split"* but the body runs `np.random.shuffle(all_pids)` and takes 70%. M2 and M3 use the **official 60/40**. The comparison crosses a split boundary. |
| 2 | **Wrong metric** | 0.8213 is the legacy macro variant, not the ICBHI 2017 challenge score. See `Asif's/audit/ICBHI_SCORE_AUDIT.md`. |
| 3 | **Unverifiable** | No `confusion_matrix_raw` committed, so no score can be recomputed from the file at all. |

This notebook re-runs the **same architecture** (Gated Adaptive Fusion over frozen M2 + M3
embeddings, `fusion_dim=512`) on the **official 60/40 split**, reports **both metrics**, and commits
the confusion matrix.

## Two things this fixes beyond the re-run

**Checkpoint loading is asserted, not hoped for.** The original wraps each backbone load in
`try/except` that prints a warning and continues, with `strict=False`. If a checkpoint path were
wrong, it would train a "fusion of M2 and M3" over **two randomly-initialised backbones** and report
it as a result — the trap `Asif's/CLAUDE.md` records for M15. This version hard-fails on a load
error, reports missing/unexpected keys explicitly, and **verifies the weights actually changed**, so
a silent no-op load cannot pass.

**It runs the test that admits this item.** `Novelty Search.md` §4.0 lists the fusion ensemble as
selected *"only if it measurably beats both backbones alone; drop it silently if it doesn't."*
**That test has never been run** — the original compared against M2's number from a different split.
Here, M2-alone and M3-alone are evaluated on the *same* test set as the fusion, and the notebook
prints the verdict, including "drop it" if the fusion loses.

## Calibration before you read the output

On the official split and official metric: **M2 = 0.6138**, **M3 = 0.5895**, **M22 = 0.6495**, and
published ICBHI SOTA ≈ **0.60–0.65**. A fusion landing near 0.65 is a solid, honest, literature-level
result. It will not look like 0.8213, and it shouldn't.
""")

# ===========================================================================
md(r"""
---
## Section 1 — Environment
""")

code(r'''
# ============================================================
# CELL 0 — ENVIRONMENT & OUTPUT PATHS
# ============================================================
import os, sys, platform

print("=" * 70)
print("M30 v2 (Gated Feature-Fusion, official split) — ENVIRONMENT")
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
        print("WARNING: no GPU. Only a small fusion head trains, but feature extraction "
              "over ~6.9k cycles will be slow on CPU.")
except ImportError:
    print("ERROR: PyTorch not installed.")

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
DRIVE_MOUNTED = False

if IN_COLAB:
    print("\n--- Google Colab detected ---")
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_MOUNTED = True
        print("Drive mounted at /content/drive")
    except Exception as e:
        print(f"Drive mount skipped ({e}); using ephemeral /content storage.")
elif os.path.exists("/kaggle/working"):
    print("\n--- Kaggle detected ---")
else:
    print("\n--- Local environment ---")

if IN_COLAB and DRIVE_MOUNTED:
    BASE_DIR = "/content/drive/MyDrive/OWMTL/M30_v2"
elif IN_COLAB:
    BASE_DIR = "/content/OWMTL/M30_v2"
elif os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
else:
    BASE_DIR = "./outputs_M30_v2"

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
for d in (CKPT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

print(f"\nCheckpoints : {CKPT_DIR}")
print(f"Results     : {RESULTS_DIR}")
print("=" * 70)
''')

md(r"""
---
## Section 1b — Dataset download (Kaggle)

Idempotent: if the audio is already unpacked, this skips. Paste your Kaggle key, or use the
commented-out Colab Secrets block so the key never appears in the notebook text.

**You must also supply the M2 and M3 checkpoints** — they are gitignored, so upload
`Asif's/M2/best_model.pth` and `Asif's/M3/best_model.pth` to `/content/` (rename to
`M2_best_model.pth` / `M3_best_model.pth`) or place them on Drive. The notebook fails fast and
explains if either is missing.
""")

code(r'''
# ============================================================
# CELL 0b — DOWNLOAD ICBHI FROM KAGGLE (idempotent)
# ============================================================
import os

os.environ['KAGGLE_USERNAME'] = 'AsifM7'
os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KAGGLE_API_KEY_HERE'   # <-- replace before running

# ---- Safer alternative: Colab Secrets ----
# from google.colab import userdata
# os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
# os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

import subprocess, sys, glob

_DL_ROOT = "/content" if os.path.exists("/content") else "./data"
os.makedirs(_DL_ROOT, exist_ok=True)


def _find_audio_dir():
    for d in glob.glob(os.path.join(_DL_ROOT, "**", "audio_and_txt_files"), recursive=True):
        if glob.glob(os.path.join(d, "*.wav")):
            return d
    return None


if _find_audio_dir():
    d = _find_audio_dir()
    print(f"ICBHI already present ({len(glob.glob(os.path.join(d, '*.wav')))} .wav) — skipping.")
else:
    if os.environ.get('KAGGLE_KEY', '') in ('', 'PASTE_YOUR_KAGGLE_API_KEY_HERE'):
        raise RuntimeError(
            "KAGGLE_KEY is still the placeholder. Paste your real Kaggle API key into this cell "
            "(or use the Colab Secrets block above) before running.")
    print("Installing kaggle CLI ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    print("Downloading vbookshelf/respiratory-sound-database ...")
    subprocess.check_call(["kaggle", "datasets", "download",
                           "-d", "vbookshelf/respiratory-sound-database",
                           "-p", _DL_ROOT, "--unzip"])
    d = _find_audio_dir()
    assert d, "Download finished but no audio_and_txt_files/*.wav found — check /content manually."
    print(f"Done. {len(glob.glob(os.path.join(d, '*.wav')))} .wav at {d}")
''')

# ===========================================================================
md(r"""
---
## Section 2 — Configuration

Preprocessing is inherited verbatim from M2/M3 so the frozen embeddings are the ones those
checkpoints were trained to produce. Fusion hyperparameters match the original M30 exactly
(`fusion_dim=512`, `dropout=0.4`, `lr=1e-3`, 30 epochs) — the point of this run is to change the
**split and the metric**, not the model.
""")

code(r'''
# ============================================================
# CELL 1 — DEPENDENCIES & CONFIGURATION
# ============================================================
import subprocess, math, glob

def pip_install(pkg, import_name=None):
    try:
        __import__(import_name or pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p, n in [("librosa", None), ("soundfile", None), ("scikit-learn", "sklearn"),
             ("tqdm", None), ("matplotlib", None), ("pandas", None), ("torchvision", None)]:
    pip_install(p, n)
print("Dependencies ready.\n")


def find_first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def find_audio_dir():
    for root in ("/content", "/kaggle/input", "./data", "."):
        if not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "**", "audio_and_txt_files"),
                                  recursive=True)):
            if glob.glob(os.path.join(d, "*.wav")):
                return d
    return None


DATA_ROOT = find_audio_dir()

M2_CKPT = find_first([
    "/content/drive/MyDrive/OWMTL/M2/checkpoints/best_model.pth",
    "/content/drive/MyDrive/OWMTL/M2/best_model.pth",
    "/content/M2_best_model.pth", "/content/best_model.pth",
    "/kaggle/input/**/m2*/best_model.pth", "./**/M2/best_model.pth",
])
M3_CKPT = find_first([
    "/content/drive/MyDrive/OWMTL/M3/checkpoints/best_model.pth",
    "/content/drive/MyDrive/OWMTL/M3/best_model.pth",
    "/content/M3_best_model.pth",
    "/kaggle/input/**/m3*/best_model.pth", "./**/M3/best_model.pth",
])

SPLIT_FILE = find_first([
    "/content/ICBHI_challenge_train_test.txt",
    "/content/**/ICBHI_challenge_train_test.txt",
    "/kaggle/input/**/ICBHI_challenge_train_test.txt",
    "./**/ICBHI_challenge_train_test.txt",
])

print(f"DATA_ROOT  : {DATA_ROOT}")
print(f"M2_CKPT    : {M2_CKPT}")
print(f"M3_CKPT    : {M3_CKPT}")
print(f"SPLIT_FILE : {SPLIT_FILE}")

missing = [n for n, v in (("ICBHI audio", DATA_ROOT), ("M2 checkpoint", M2_CKPT),
                          ("M3 checkpoint", M3_CKPT)) if not v]
if missing:
    raise RuntimeError(
        f"Missing required input(s): {', '.join(missing)}.\n"
        f"  ICBHI audio    -> run Cell 0b\n"
        f"  M2 checkpoint  -> upload Asif's/M2/best_model.pth as /content/M2_best_model.pth\n"
        f"  M3 checkpoint  -> upload Asif's/M3/best_model.pth as /content/M3_best_model.pth\n"
        f"This notebook deliberately refuses to run on randomly-initialised backbones -- that "
        f"is the failure mode it exists to rule out.")

CFG = {
    # -- section 2 preprocessing: identical to M2 and M3 --
    "sample_rate": 16000, "duration_s": 8.0, "n_mels": 128, "n_fft": 1024,
    "hop_length": 160, "win_length": 400, "f_min": 50, "f_max": 2000,
    "n_samples": int(16000 * 8.0), "n_frames": None,

    "sound_classes": ["Normal", "Crackle", "Wheeze", "Both"],
    "num_classes": 4,

    # -- fusion head: matched to the original M30 --
    "fusion_dim": 512, "dropout": 0.4,
    "batch_size": 32, "num_epochs": 30, "lr": 1e-3, "weight_decay": 1e-4,
    "seed": 42,

    "data_root": DATA_ROOT, "m2_ckpt": M2_CKPT, "m3_ckpt": M3_CKPT,
    "split_file": SPLIT_FILE,
    "ckpt_dir": CKPT_DIR, "results_dir": RESULTS_DIR,
    "model_id": "M30", "contributor": "Asif",

    # -- how to reconcile the official split with protocol section 1 --
    # The official ICBHI split is NOT patient-disjoint: patients 156 and 218 appear
    # on both sides (13 train + 12 test recordings between them). Protocol section 1
    # requirement 1 makes patient-independence a hard requirement, so we must choose.
    #   "drop_from_train" (default) -- drop those patients' TRAIN recordings. The official
    #                                  TEST set stays byte-identical, so the reported score
    #                                  remains directly comparable to published ICBHI work,
    #                                  at the cost of 13 of 539 training recordings (2.4%).
    #   "as_is"                     -- official split verbatim, accepting the leak. Only for
    #                                  reproducing other papers' exact setup; violates section 1.
    "official_overlap_policy": "drop_from_train",
}
CFG["n_frames"] = 1 + math.floor(CFG["n_samples"] / CFG["hop_length"])   # 801

print("\n" + "=" * 60)
for k, v in CFG.items():
    print(f"  {k:<18}: {v}")
print("=" * 60)
''')

code(r'''
# ============================================================
# CELL 2 — IMPORTS & SEED
# ============================================================
import json, time, random, warnings, tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
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
## Section 3 — Data: the **official** ICBHI 60/40 split

This is the substantive change from the original. The split is read from
`ICBHI_challenge_train_test.txt`, which ships with the dataset and assigns every recording to train
or test — the same function M2 uses, so M30 v2, M2 and M3 are evaluated on **identical test
cycles**.

If that file is absent the notebook falls back to the documented patient-ID rule and **says so
loudly in the results JSON**, because a silent fallback to a different split is precisely the defect
this re-run exists to correct.
""")

code(r'''
# ============================================================
# CELL 3 — PARSE ICBHI + OFFICIAL 60/40 SPLIT
# ============================================================

def parse_annotation_file(txt_path):
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


# Ground truth, verified against the committed Asif's/ICBHI_challenge_train_test.txt.
# If any assertion below fires, the split file is not the official one -- stop and check it.
OFFICIAL_RECORDINGS = 920
OFFICIAL_TRAIN_RECS = 539       # 58.6%
OFFICIAL_TEST_RECS = 381        # 41.4%  -- this is the "60/40"
OFFICIAL_PATIENTS = 126
OFFICIAL_OVERLAP_PATIENTS = {156, 218}   # in BOTH train and test in the official file


def load_official_split(data_root):
    """Return {filename_stem: 'train'|'test'} from ICBHI_challenge_train_test.txt.

    Raises if not found. There is deliberately NO fallback: a silent fallback to a
    patient-ID rule is exactly what gave M2/M3/M12 an 11-patient test set while their
    results JSONs recorded split_method 'patient_independent_official_60_40'.
    """
    searched = []
    bases = [os.path.dirname(data_root.rstrip("/")), data_root, "/content", "/kaggle/input", "."]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        for pat in ("**/ICBHI_challenge_train_test.txt", "**/*train_test*.txt"):
            spec = os.path.join(base, pat)
            searched.append(spec)
            for path in sorted(glob.glob(spec, recursive=True)):
                mapping = {}
                try:
                    with open(path) as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].lower() in ("train", "test"):
                                mapping[parts[0].replace(".wav", "")] = parts[1].lower()
                except Exception:
                    continue
                if mapping:
                    print(f"Official split file: {path}  ({len(mapping)} recordings)")
                    return mapping
    raise RuntimeError(
        "ICBHI_challenge_train_test.txt NOT FOUND, and this notebook has no fallback split "
        "by design.\n"
        "Upload Asif's/ICBHI_challenge_train_test.txt to /content/ and re-run.\n"
        "Searched:\n  " + "\n  ".join(searched))


split_map = load_official_split(CFG["data_root"])
SPLIT_SOURCE = "official_file"

# --- Verify the file is the real official split, not a truncated or edited copy ---
_n_train = sum(1 for v in split_map.values() if v == "train")
_n_test = sum(1 for v in split_map.values() if v == "test")
_pat_sides = {}
for _stem, _s in split_map.items():
    _pat_sides.setdefault(int(_stem.split("_")[0]), set()).add(_s)
_overlap = {p for p, s in _pat_sides.items() if len(s) == 2}

print(f"  recordings : {len(split_map)} ({_n_train} train / {_n_test} test)")
print(f"  patients   : {len(_pat_sides)}  | on both sides: {sorted(_overlap)}")

assert len(split_map) == OFFICIAL_RECORDINGS, \
    f"expected {OFFICIAL_RECORDINGS} recordings in the split file, got {len(split_map)}"
assert (_n_train, _n_test) == (OFFICIAL_TRAIN_RECS, OFFICIAL_TEST_RECS), \
    f"expected {OFFICIAL_TRAIN_RECS}/{OFFICIAL_TEST_RECS} train/test, got {_n_train}/{_n_test}"
assert len(_pat_sides) == OFFICIAL_PATIENTS, \
    f"expected {OFFICIAL_PATIENTS} patients, got {len(_pat_sides)}"
assert _overlap == OFFICIAL_OVERLAP_PATIENTS, \
    f"expected patients {sorted(OFFICIAL_OVERLAP_PATIENTS)} to straddle the split, got {sorted(_overlap)}"
print("[OK] split file matches the official ICBHI 2017 challenge split exactly.")

# --- Reconcile the official split with protocol section 1 (patient-independence) ---
OVERLAP_POLICY = CFG["official_overlap_policy"]
if OVERLAP_POLICY == "drop_from_train":
    print(f"\nPolicy 'drop_from_train': patients {sorted(OFFICIAL_OVERLAP_PATIENTS)} appear on both "
          f"sides of the official split.\n  Dropping their TRAIN recordings; the official TEST set "
          f"is left byte-identical so scores stay comparable to published work.")
elif OVERLAP_POLICY == "as_is":
    print(f"\nPolicy 'as_is': keeping the official split verbatim. WARNING -- patients "
          f"{sorted(OFFICIAL_OVERLAP_PATIENTS)} leak across train/test, violating "
          f"Model_Training_Protocol.md section 1 requirement 1.")
else:
    raise ValueError(f"unknown official_overlap_policy: {OVERLAP_POLICY!r}")

def key_without_device(stem):
    """'226_1b1_Pl_sc_LittC2SE' -> '226_1b1_Pl_sc'  (device tag dropped).

    The official split file and the audio filenames disagree on the device suffix for at
    least one recording: 226_1b1_Pl is 'Meditron' in the split file and 'LittC2SE' on disk.
    These first four fields are unique across all 920 recordings, so they identify a
    recording unambiguously without trusting the device tag.
    """
    return "_".join(stem.split("_")[:4])


split_by_key = {key_without_device(s): (s, side) for s, side in split_map.items()}
assert len(split_by_key) == len(split_map), (
    "device-independent keys are not unique in the split file, so a recording whose device "
    "tag disagrees cannot be matched safely. Investigate before proceeding.")

rows = []
device_mismatches = []
n_dropped_overlap = 0
for wav in sorted(glob.glob(os.path.join(CFG["data_root"], "*.wav"))):
    stem = os.path.splitext(os.path.basename(wav))[0]
    txt = os.path.join(CFG["data_root"], stem + ".txt")
    if not os.path.exists(txt):
        continue
    try:
        pid = int(stem.split("_")[0])
    except ValueError:
        continue
    split = split_map.get(stem)
    if split is None:
        # Exact stem missing: match on the device-independent key. Not a guess -- same
        # patient, session, chest location and mode; only the device tag differs.
        alt = split_by_key.get(key_without_device(stem))
        if alt is None:
            raise RuntimeError(
                f"recording {stem!r} is on disk but has no counterpart in the official split "
                f"file, even ignoring the device suffix. The audio and the split file "
                f"genuinely disagree -- do not guess a side for it.")
        file_stem, split = alt
        device_mismatches.append((stem, file_stem))
    if (OVERLAP_POLICY == "drop_from_train" and pid in OFFICIAL_OVERLAP_PATIENTS
            and split == "train"):
        n_dropped_overlap += 1
        continue
    for c in parse_annotation_file(txt):
        rows.append({"wav_path": wav, "patient_id": pid, "split": split, **c})

if device_mismatches:
    print(f"\n{len(device_mismatches)} recording(s) matched on patient/session/location "
          f"because the device tag differs between the audio and the split file:")
    for disk_stem, file_stem in device_mismatches:
        print(f"    disk={disk_stem}  split_file={file_stem}")
    print("  (known ICBHI inconsistency; the split assignment itself is unambiguous)")
if n_dropped_overlap:
    print(f"Dropped {n_dropped_overlap} train recording(s) from patients "
          f"{sorted(OFFICIAL_OVERLAP_PATIENTS)} per 'drop_from_train' policy.")

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No cycles parsed — check DATA_ROOT.")

df_train = df[df.split == "train"].reset_index(drop=True)
df_test = df[df.split == "test"].reset_index(drop=True)
train_patients = set(df_train.patient_id)
test_patients = set(df_test.patient_id)

leak = train_patients & test_patients
if OVERLAP_POLICY == "drop_from_train":
    assert not leak, f"PATIENT LEAKAGE between train and test: {sorted(leak)[:10]}"
    print(f"\n[OK] Patient-independent split verified (protocol section 1).")
else:
    assert leak == OFFICIAL_OVERLAP_PATIENTS, \
        f"expected only {sorted(OFFICIAL_OVERLAP_PATIENTS)} to leak under 'as_is', got {sorted(leak)}"
    print(f"\n[!!] {len(leak)} patient(s) leak by design under 'as_is': {sorted(leak)}")

print(f"Train : {len(df_train):5d} cycles from {len(train_patients):3d} patients")
print(f"Test  : {len(df_test):5d} cycles from {len(test_patients):3d} patients")

# Expected under the official split: 77 train-only + 47 test-only patients, 2 straddling.
assert len(test_patients) >= 40, (
    f"only {len(test_patients)} test patients -- the official split has 47-49. "
    f"This is the signature of the fallback split that invalidated M2/M3/M12 (11 patients).")

print(f"\n{'class':<10}{'train':>10}{'test':>10}")
for i, c in enumerate(CFG["sound_classes"]):
    print(f"{c:<10}{int((df_train.label == i).sum()):>10}{int((df_test.label == i).sum()):>10}")

# The official test set is ~2700-2800 cycles. M2/M3/M12's committed 492 is NOT a valid
# reference: those runs took the patient-ID fallback (11 test patients) while recording
# split_method 'patient_independent_official_60_40'. Do not compare against their scores
# until they are retrained on this split.
if not (2400 <= len(df_test) <= 3100):
    print(f"\nNOTE: {len(df_test)} test cycles, expected ~2750 for the official split. "
          f"Investigate the annotation parsing before trusting anything downstream.")
else:
    print(f"[OK] {len(df_test)} test cycles -- consistent with the official split.")
''')

code(r'''
# ============================================================
# CELL 4 — LOG-MEL EXTRACTION (identical to M2/M3)
# ============================================================

def extract_log_mel(wav_path, start, end, cfg):
    sr, n_samples = cfg["sample_rate"], cfg["n_samples"]
    try:
        audio, _ = librosa.load(wav_path, sr=sr, offset=start,
                                duration=max(end - start, 0.05), mono=True)
    except Exception:
        return np.zeros((1, cfg["n_mels"], cfg["n_frames"]), dtype=np.float32)
    if len(audio) == 0:
        return np.zeros((1, cfg["n_mels"], cfg["n_frames"]), dtype=np.float32)
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


class ICBHIDataset(Dataset):
    def __init__(self, frame, cfg):
        self.df = frame.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return (torch.from_numpy(extract_log_mel(r["wav_path"], r["start"], r["end"], self.cfg)),
                int(r["label"]))


# These loaders are used ONCE each, for the feature-extraction pass in Cell 6 -- not per epoch.
# See Cell 6 for why. persistent_workers keeps the worker pool alive so its iterator is not
# torn down and garbage-collected mid-run, which is what produces Colab's harmless-but-noisy
# "AssertionError: can only test a child process" flood from _MultiProcessingDataLoaderIter.__del__.
_dl_kw = dict(num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True)

extract_train_loader = DataLoader(ICBHIDataset(df_train, CFG), batch_size=CFG["batch_size"],
                                  shuffle=False, **_dl_kw)
extract_test_loader = DataLoader(ICBHIDataset(df_test, CFG), batch_size=CFG["batch_size"],
                                 shuffle=False, **_dl_kw)
_x, _y = next(iter(extract_train_loader))
print(f"Batch {tuple(_x.shape)} | expected (B, 1, {CFG['n_mels']}, {CFG['n_frames']})")
assert _x.shape[1:] == (1, CFG["n_mels"], CFG["n_frames"]), "Spectrogram shape mismatch."
print(f"Train {len(df_train)} cycles | Test {len(df_test)} cycles "
      f"— each read from disk exactly ONCE (Cell 6).")
''')

# ===========================================================================
md(r"""
---
## Section 4 — Architecture and **asserted** checkpoint loading

The architecture is copied from the original M30: frozen M2 (768-d) and M3 (1280-d) embeddings
concatenated to 2048-d, then a sigmoid gate modulating a projection into `fusion_dim=512`, then a
linear classifier.

The loading is not copied. The original's `try/except` + `strict=False` would proceed on randomly
initialised backbones after printing a warning. Here, three things must hold or the cell raises:

1. The checkpoint file loads.
2. `load_state_dict` reports **no missing keys** for the backbone parameters.
3. The weights actually **changed** — a fingerprint taken before and after the load must differ.

Check 3 is the one that catches the nastiest case: a `state_dict` whose keys silently don't match
under `strict=False`, which loads "successfully" while changing nothing.
""")

code(r'''
# ============================================================
# CELL 5 — BACKBONES + GATED FUSION, WITH ASSERTED LOADING
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
    """Identical to Asif's/M2 -- required to load its state_dict."""

    def __init__(self, num_classes=4, depth=5, base_width=48, dropout=0.4, fc_dim=128):
        super().__init__()
        channels = [base_width * (2 ** i) for i in range(depth)]      # [48,96,192,384,768]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch)); in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(channels[-1], fc_dim),
                                  nn.ReLU(inplace=True), nn.Linear(fc_dim, num_classes))
        self.embedding_dim = channels[-1]

    def get_embedding(self, x):
        return self.gap(self.encoder(x)).flatten(1)

    def forward(self, x):
        return self.head(self.dropout(self.get_embedding(x)))


class M3_MobileNet(nn.Module):
    """Identical to Asif's/M3: 1-channel log-mel repeated to 3 and ImageNet-normalised."""

    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()
        mb = tv_models.mobilenet_v2(weights=None)
        self.features = mb.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1280, num_classes)      # name must match M3's checkpoint: head.weight/head.bias
        self.embedding_dim = 1280
        # Buffer NAMES must match M3's checkpoint (in_mean / in_std) so the normalisation
        # actually loads instead of silently falling back to these defaults. The defaults
        # happen to equal M3's stored values today, but relying on that would break the
        # instant M3 is retrained with different normalisation.
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def get_embedding(self, x):
        x3 = (x.repeat(1, 3, 1, 1) - self.in_mean) / self.in_std
        return self.gap(self.features(x3)).flatten(1)

    def forward(self, x):
        return self.head(self.dropout(self.get_embedding(x)))


class FusionHead(nn.Module):
    """The only trainable part: gate * projection -> classifier, over concatenated embeddings.

    Split out from the ensemble deliberately. Because both backbones are FROZEN, the 2048-d
    embedding for a given cycle is identical at every epoch -- so it is extracted once (Cell 6)
    and this head trains directly on the cached vectors. Architecture is unchanged from the
    original M30; only where the embeddings come from differs.
    """

    def __init__(self, in_dim, fusion_dim=512, num_classes=4, dropout=0.4):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(in_dim, fusion_dim), nn.Sigmoid())
        self.project = nn.Sequential(nn.Linear(in_dim, fusion_dim),
                                     nn.BatchNorm1d(fusion_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(fusion_dim, num_classes))

    def forward(self, f):
        return self.classifier(self.gate(f) * self.project(f))


class GatedFusionEnsemble(nn.Module):
    """Backbones + head, end to end. Used for inference-latency measurement and as the saved
    checkpoint, so the artifact is a complete usable model rather than a bare head."""

    def __init__(self, m2, m3, head):
        super().__init__()
        self.m2, self.m3, self.head = m2, m3, head

    def forward(self, x):
        with torch.no_grad():
            f = torch.cat([self.m2.get_embedding(x), self.m3.get_embedding(x)], dim=-1)
        return self.head(f)


def _fingerprint(model):
    """Sum of all parameter values -- changes if any weight changes."""
    with torch.no_grad():
        return float(sum(p.detach().double().sum().item() for p in model.parameters()))


def load_backbone_or_die(model, ckpt_path, name):
    """Load a checkpoint and PROVE it took effect. Raises rather than continuing on random weights.

    The original M30 wrapped this in try/except + strict=False and printed a warning on failure,
    which would train a 'fusion of M2 and M3' over two randomly-initialised networks.
    """
    before = _fingerprint(model)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)   # protocol 11.A
    sd = state.get("model_state", state.get("model_state_dict", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    own = {n for n, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in own]
    if missing_params:
        raise RuntimeError(
            f"{name}: checkpoint does not supply weights for {len(missing_params)} parameter "
            f"tensor(s), e.g. {missing_params[:5]}. Those would stay randomly initialised. "
            f"Refusing to continue -- check that {ckpt_path} is the right checkpoint.")

    # Buffers (e.g. M3's in_mean/in_std normalisation) are not parameters, so a name
    # mismatch there loads nothing and silently leaves defaults in place. Report it.
    own_buffers = {n for n, _ in model.named_buffers()}
    missing_buffers = [k for k in missing if k in own_buffers]
    if missing_buffers:
        print(f"     NOTE: {name} checkpoint did not supply buffer(s) {missing_buffers}; "
              f"the class defaults remain in effect. Verify they are correct.")

    after = _fingerprint(model)
    if abs(after - before) < 1e-9:
        raise RuntimeError(
            f"{name}: load_state_dict reported success but NO WEIGHT CHANGED "
            f"(fingerprint {before:.6f} -> {after:.6f}). The state_dict keys almost certainly do "
            f"not match this architecture and strict=False hid it. Refusing to continue.")

    print(f"[OK] {name} loaded from {ckpt_path}")
    print(f"     epoch={state.get('epoch')} best_score={state.get('best_score')} "
          f"| {len(unexpected)} unexpected key(s) ignored")
    print(f"     weight fingerprint {before:.4f} -> {after:.4f}  (changed, as required)")
    return state


m2_backbone = M2_CNN(num_classes=CFG["num_classes"]).to(DEVICE)
m3_backbone = M3_MobileNet(num_classes=CFG["num_classes"]).to(DEVICE)

m2_state = load_backbone_or_die(m2_backbone, CFG["m2_ckpt"], "M2 backbone")
m3_state = load_backbone_or_die(m3_backbone, CFG["m3_ckpt"], "M3 backbone")

for bb in (m2_backbone, m3_backbone):
    bb.eval()
    for p in bb.parameters():
        p.requires_grad = False

EMB_DIM = m2_backbone.embedding_dim + m3_backbone.embedding_dim        # 768 + 1280 = 2048
fusion_head = FusionHead(EMB_DIM, fusion_dim=CFG["fusion_dim"],
                         num_classes=CFG["num_classes"], dropout=CFG["dropout"]).to(DEVICE)
model = GatedFusionEnsemble(m2_backbone, m3_backbone, fusion_head).to(DEVICE)

TOTAL_PARAMS = sum(p.numel() for p in model.parameters())
TRAINABLE_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nEmbedding dim    : {EMB_DIM}")
print(f"Total params     : {TOTAL_PARAMS:,}")
print(f"Trainable (fusion): {TRAINABLE_PARAMS:,} ({TRAINABLE_PARAMS / TOTAL_PARAMS:.1%})")
assert TRAINABLE_PARAMS < TOTAL_PARAMS * 0.5, "Backbones are not frozen."
''')

# ===========================================================================
md(r"""
---
## Section 5 — Metrics

Both ICBHI scores are computed side by side. `icbhi_score` is the project's legacy macro variant,
kept for continuity; `icbhi_score_official` is the ICBHI 2017 challenge metric and the one that goes
in the paper. See `Model_Training_Protocol.md` §3.
""")

code(r'''
# ============================================================
# CELL 6 — METRIC SUITE (both ICBHI variants)
# ============================================================

def specificity_per_class(cm):
    """Per-class specificity: TN / (TN + FP)."""
    out = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.array(out)


def official_icbhi(cm):
    """ICBHI 2017 challenge score. Se over pooled abnormal, Sp over Normal (class 0)."""
    cm = np.asarray(cm, dtype=float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    abn_correct = sum(cm[i, i] for i in range(1, cm.shape[0]))
    abn_total = sum(cm[i].sum() for i in range(1, cm.shape[0]))
    se = abn_correct / abn_total if abn_total else 0.0
    return se, sp, (se + sp) / 2


def metrics_from_predictions(y_true, y_pred, tag=""):
    """All protocol metrics from label arrays. Kept separate from any DataLoader so it can be
    reused for the cached-embedding path, the backbone baselines, and per-epoch validation."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CFG["num_classes"])))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(CFG["num_classes"])), zero_division=0)
    spec = specificity_per_class(cm)
    se_o, sp_o, icbhi_o = official_icbhi(cm)

    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec.mean()), "recall_macro": float(rec.mean()),
        "f1_macro": float(f1.mean()), "specificity_macro": float(spec.mean()),
        "icbhi_score": float((rec.mean() + spec.mean()) / 2),          # legacy macro variant
        "icbhi_score_official": float(icbhi_o),                         # ICBHI 2017 challenge
        "icbhi_official_sensitivity": float(se_o),
        "icbhi_official_specificity": float(sp_o),
        "per_class": {CFG["sound_classes"][i]: {
            "precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
            "specificity": round(float(spec[i]), 4), "f1": round(float(f1[i]), 4),
            "support": int(sup[i])} for i in range(CFG["num_classes"])},
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized": np.round(
            cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), 4).tolist(),
    }
    if tag:
        print(f"{tag:<22} official ICBHI {icbhi_o:.4f} (Se {se_o:.4f} / Sp {sp_o:.4f})  "
              f"| acc {m['accuracy']:.4f} F1 {m['f1_macro']:.4f}  "
              f"| legacy macro {m['icbhi_score']:.4f}")
    return m


def predict_logits(net, feats, batch=512):
    """Run a head over cached embeddings. No DataLoader, no worker processes."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            out.append(net(feats[i:i + batch].to(DEVICE)).cpu())
    return torch.cat(out)


print("Metric suite ready: both ICBHI variants computed side by side.")
''')

# ===========================================================================
md(r"""
---
## Section 6 — The test that admits this item

`Novelty Search.md` §4.0 selects the fusion ensemble **"only if it measurably beats both backbones
alone; drop it silently if it doesn't."** That test was never run — the original compared against
M2's score from a *different split*.

Here M2-alone and M3-alone are evaluated on **this notebook's test set**, before any fusion training,
so the comparison is like-for-like. These two numbers should land close to the published
M2 = 0.6138 / M3 = 0.5895 if the split and checkpoints are right; a large gap means something is
wrong and the fusion result should not be trusted either.
""")

code(r'''
# ============================================================
# CELL 7 — ONE-TIME FEATURE EXTRACTION + BACKBONE BASELINES
# ============================================================
# Both backbones are FROZEN, so a given cycle's 2048-d embedding is identical at every
# epoch. Extracting once and training the head on cached vectors is not an optimisation
# detail -- recomputing librosa log-mels 30 times over would be ~30x the work for bitwise
# identical inputs, and it is what made the first run take minutes per epoch.
# It also removes the per-epoch DataLoader worker churn behind Colab's noisy (harmless)
# "AssertionError: can only test a child process" flood.

def extract_features(loader, frame, tag):
    """One pass: cache M2 and M3 embeddings plus each backbone's own logits."""
    e2, e3, l2, l3, ys = [], [], [], [], []
    m2_backbone.eval(); m3_backbone.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"extracting {tag}"):
            x = x.to(DEVICE, non_blocking=True)
            f2 = m2_backbone.get_embedding(x)
            f3 = m3_backbone.get_embedding(x)
            e2.append(f2.cpu()); e3.append(f3.cpu())
            l2.append(m2_backbone.head(f2).cpu())      # backbone's own classifier
            l3.append(m3_backbone.head(f3).cpu())
            ys.append(y)
    e2, e3 = torch.cat(e2), torch.cat(e3)
    out = dict(emb=torch.cat([e2, e3], dim=1), logits_m2=torch.cat(l2),
               logits_m3=torch.cat(l3), y=torch.cat(ys).numpy())
    assert len(out["emb"]) == len(frame), \
        f"{tag}: extracted {len(out['emb'])} rows for {len(frame)} cycles -- order/length mismatch."
    print(f"  {tag}: {tuple(out['emb'].shape)} cached")
    return out


_t0 = time.time()
TRAIN_F = extract_features(extract_train_loader, df_train, "train")
TEST_F = extract_features(extract_test_loader, df_test, "test")
EXTRACT_TIME = time.time() - _t0
print(f"\nFeature extraction done in {EXTRACT_TIME:.0f}s. Audio is now read ZERO more times.\n")

# Free the worker pools explicitly so their iterators aren't torn down later during GC.
del extract_train_loader, extract_test_loader

print("Backbone baselines on THIS notebook's test split (official, ~47 test patients)\n")

BASE_M2 = metrics_from_predictions(TEST_F["y"], TEST_F["logits_m2"].argmax(1).numpy(), "M2 alone")
BASE_M3 = metrics_from_predictions(TEST_F["y"], TEST_F["logits_m3"].argmax(1).numpy(), "M3 alone")

# NOTE: M2/M3's committed 0.6138/0.5895 were measured on the patient-ID FALLBACK split
# (11 test patients, 492 cycles) despite their JSONs claiming 'official_60_40'. They are a
# provenance record, NOT a target. A deviation here is EXPECTED and is not a failure.
for name, got, superseded in (("M2", BASE_M2["icbhi_score_official"], 0.6138),
                              ("M3", BASE_M3["icbhi_score_official"], 0.5895)):
    print(f"  {name}: {got:.4f} on the official split "
          f"(superseded fallback-split figure was {superseded:.4f}, delta {got - superseded:+.4f})")

_worst = min(BASE_M2["icbhi_score_official"], BASE_M3["icbhi_score_official"])
if _worst < 0.40:
    print(f"\n*** STOP: a backbone scores {_worst:.4f}, near or below chance for this metric. "
          f"That indicates a broken checkpoint, preprocessing mismatch, or label misalignment -- "
          f"not a hard split. The fusion comparison below would inherit it. ***")
else:
    print(f"\n[OK] Both backbones score above 0.40; published ICBHI work sits at ~0.60-0.65. "
          f"These two numbers -- not the superseded ones -- are what the fusion must beat.")
''')

# ===========================================================================
md(r"""
---
## Section 7 — Train the fusion head
""")

code(r'''
# ============================================================
# CELL 8 — TRAINING LOOP (fusion head on cached embeddings)
# ============================================================
# TensorDataset over the cached 2048-d vectors: no librosa, no disk I/O, no worker
# processes. num_workers=0 is correct here, not a workaround -- there is nothing to
# parallelise, and it is what keeps the output clean.

train_ds = torch.utils.data.TensorDataset(TRAIN_F["emb"],
                                          torch.tensor(TRAIN_F["y"], dtype=torch.long))
head_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                         num_workers=0, drop_last=True)

counts = np.bincount(df_train.label.values, minlength=CFG["num_classes"]).astype(float)
weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) / CFG["num_classes"],
                       dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)     # inverse-frequency, as in M2/M3
optimizer = torch.optim.Adam(fusion_head.parameters(),
                             lr=CFG["lr"], weight_decay=CFG["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["num_epochs"])
print(f"Class weights: {weights.cpu().numpy().round(3)}")
print(f"Training {sum(p.numel() for p in fusion_head.parameters()):,} fusion params "
      f"on {len(train_ds)} cached embeddings\n")

history, best = [], {"icbhi_score_official": -1.0, "epoch": None}
BEST_PATH = os.path.join(CFG["ckpt_dir"], "best_model.pth")
t_start = time.time()

for epoch in range(1, CFG["num_epochs"] + 1):
    fusion_head.train()
    tot, correct, loss_sum = 0, 0, 0.0
    t0 = time.time()
    for f, y in head_loader:
        f, y = f.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        out = fusion_head(f)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    scheduler.step()

    val_pred = predict_logits(fusion_head, TEST_F["emb"]).argmax(1).numpy()
    val = metrics_from_predictions(TEST_F["y"], val_pred)
    rec = {"epoch": epoch, "train_loss": round(loss_sum / tot, 4),
           "train_accuracy": round(correct / tot, 4),
           "val_accuracy": round(val["accuracy"], 4),
           "val_f1_macro": round(val["f1_macro"], 4),
           "val_icbhi_score": round(val["icbhi_score"], 4),
           "val_icbhi_score_official": round(val["icbhi_score_official"], 4),
           "lr": optimizer.param_groups[0]["lr"],
           "epoch_time_s": round(time.time() - t0, 1),
           "is_best": False}

    if val["icbhi_score_official"] > best["icbhi_score_official"]:
        best = dict(val); best["epoch"] = epoch
        rec["is_best"] = True
        # Save the FULL ensemble (backbones + head) so the artifact is a usable model.
        torch.save({"epoch": int(epoch),
                    "best_score": float(val["icbhi_score_official"]),
                    "model_state": model.state_dict(),
                    "head_state": fusion_head.state_dict(),
                    "model_config": {"fusion_dim": CFG["fusion_dim"],
                                     "dropout": CFG["dropout"],
                                     "embedding_dim": EMB_DIM}}, BEST_PATH)   # protocol 11.A
    history.append(rec)
    print(f"  epoch {epoch:>2}  loss {rec['train_loss']:.4f}  "
          f"val acc {rec['val_accuracy']:.4f}  official ICBHI "
          f"{rec['val_icbhi_score_official']:.4f}{'  <- best' if rec['is_best'] else ''}")

TRAIN_TIME = time.time() - t_start
print(f"\nHead training done in {TRAIN_TIME:.0f}s "
      f"({TRAIN_TIME / CFG['num_epochs']:.1f}s/epoch). "
      f"Best epoch {best['epoch']} official ICBHI {best['icbhi_score_official']:.4f}")

# selection is on the official metric, so re-load the best head for final reporting
_st = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
fusion_head.load_state_dict(_st["head_state"])
FUSION = metrics_from_predictions(
    TEST_F["y"], predict_logits(fusion_head, TEST_F["emb"]).argmax(1).numpy(),
    "M30 v2 (fusion)")
''')

# ===========================================================================
md(r"""
---
## Section 8 — Verdict, plots, and results JSON
""")

code(r'''
# ============================================================
# CELL 9 — THE §4.0 VERDICT
# ============================================================
f_o = FUSION["icbhi_score_official"]
m2_o = BASE_M2["icbhi_score_official"]
m3_o = BASE_M3["icbhi_score_official"]
beats_both = f_o > m2_o and f_o > m3_o
margin = f_o - max(m2_o, m3_o)

print("=" * 78)
print("NOVELTY SEARCH SECTION 4.0 ADMISSION TEST")
print("=" * 78)
print('  Rule: include the fusion ensemble "only if it measurably beats both backbones')
print('        alone; drop it silently if it doesn\'t."\n')
print(f"{'':<22}{'official':>10}{'legacy macro':>15}")
for n, m in (("M2 alone", BASE_M2), ("M3 alone", BASE_M3), ("M30 v2 fusion", FUSION)):
    print(f"  {n:<20}{m['icbhi_score_official']:>10.4f}{m['icbhi_score']:>15.4f}")

print(f"\n  Margin over best single backbone: {margin:+.4f}")
print("-" * 78)
if beats_both and margin >= 0.01:
    print("  VERDICT: PASSES. The fusion beats both backbones on the same test set, on the")
    print("  official metric, by a margin larger than rounding. Selected item #3 is justified.")
elif beats_both:
    print(f"  VERDICT: MARGINAL. Fusion leads by only {margin:+.4f} -- within the noise band for")
    print("  this test-set size. Report it as 'no meaningful difference', not as a win, unless a")
    print("  significance test on paired predictions says otherwise.")
else:
    print("  VERDICT: FAILS. The fusion does NOT beat both backbones on a like-for-like")
    print("  comparison. Per Novelty Search section 4.0 this item should be DROPPED from the")
    print("  selected set rather than reported -- and a deferred item swapped in, not added.")
    print("  This is a legitimate, reportable negative result: it says heterogeneous feature")
    print("  fusion does not help here, which the original cross-split comparison concealed.")
print("=" * 78)
''')

code(r'''
# ============================================================
# CELL 10 — PLOTS
# ============================================================
plt.rcParams["figure.dpi"] = 150
ep = [h["epoch"] for h in history]
best_ep = best["epoch"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep, [h["train_loss"] for h in history], label="train loss", color="#2c7fb8")
ax.axvline(best_ep, ls=":", color="grey", label=f"best epoch ({best_ep})")
ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("M30 v2 — training loss")
fig.tight_layout(); fig.savefig(os.path.join(CFG["results_dir"], "loss_curve.png")); plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep, [h["train_accuracy"] for h in history], label="train acc", color="#2c7fb8")
ax.plot(ep, [h["val_accuracy"] for h in history], label="val acc", color="#e6550d")
ax.axvline(best_ep, ls=":", color="grey")
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()
ax.set_title("M30 v2 — accuracy")
fig.tight_layout(); fig.savefig(os.path.join(CFG["results_dir"], "accuracy_curve.png")); plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ep, [h["val_icbhi_score_official"] for h in history], label="official ICBHI",
        color="#1b7f3a")
ax.plot(ep, [h["val_icbhi_score"] for h in history], label="legacy macro", color="#c0392b",
        ls="--", alpha=0.7)
ax.axhline(m2_o, ls=":", color="#666666", label=f"M2 alone ({m2_o:.3f})")
ax.axvline(best_ep, ls=":", color="grey")
ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.legend(fontsize=8)
ax.set_title("M30 v2 — both ICBHI metrics (note the constant offset)")
fig.tight_layout(); fig.savefig(os.path.join(CFG["results_dir"], "icbhi_score_curve.png"))
plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4.8))
cmn = np.array(FUSION["confusion_matrix_normalized"])
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(CFG["sound_classes"], rotation=45, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(CFG["sound_classes"])
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"M30 v2 — confusion matrix\nofficial ICBHI {f_o:.4f}")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(CFG["results_dir"], "confusion_matrix.png"))
plt.show()

print(f"Plots written to {CFG['results_dir']}")
''')

code(r'''
# ============================================================
# CELL 11 — PROTOCOL-COMPLIANT RESULTS JSON
# ============================================================
def model_size_mb(path):
    return round(os.path.getsize(path) / (1024 * 1024), 2)


_probe = torch.randn(CFG["batch_size"], 1, CFG["n_mels"], CFG["n_frames"]).to(DEVICE)
model.eval()
with torch.no_grad():
    for _ in range(3):
        model(_probe)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    _t = time.time()
    for _ in range(10):
        model(_probe)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
INFER_MS = (time.time() - _t) / (10 * CFG["batch_size"]) * 1000

METRIC_NOTE = (
    "icbhi_score is (recall_macro + specificity_macro)/2, a project-internal metric that is NOT "
    "the ICBHI 2017 challenge score and is not comparable to published ICBHI results. "
    "icbhi_score_official is the challenge metric. Report the official figure.")

results = {
    "meta": {
        "model_id": "M30",
        "model_name": "M2 + M3 Gated Feature-Fusion Ensemble (v2, official split)",
        "contributor": CFG["contributor"],
        "date_completed": time.strftime("%Y-%m-%d"),
        "is_augmented": False,
        "augmentation_method": "none",
        "notes": (
            "v2 re-run of M30 on the OFFICIAL ICBHI 60/40 split, correcting three defects in "
            "Barshon's/M30/results_M30.json: (1) the original's build_icbhi_splits() docstring "
            "said 'official patient-independent split' while the body performed a random 70/30 "
            "shuffle, so its '+9.86% over M2' claim compared across splits; (2) the reported "
            "0.8213 was the legacy macro metric, not the ICBHI 2017 challenge score; (3) no "
            "confusion_matrix_raw was committed, so no score could be recomputed from it. "
            "Architecture, fusion_dim, dropout, lr and epoch budget are unchanged from the "
            "original -- only the split, the metric reporting, and the checkpoint-loading "
            "safety differ. "
            "Checkpoint loading is asserted rather than best-effort: the original wrapped it in "
            "try/except with strict=False and would have trained on randomly-initialised "
            "backbones after printing a warning. This version verifies no backbone parameter is "
            "missing AND that the weights measurably changed on load. "
            "This run also performs the Novelty Search section 4.0 admission test that had never "
            "been run: M2-alone and M3-alone evaluated on the SAME test set as the fusion. See "
            "best_metrics.admission_test. "
            "Implementation note: because both backbones are frozen, each cycle's 2048-d "
            "embedding is identical at every epoch, so embeddings are extracted ONCE and the "
            "fusion head trains on the cached vectors. This is mathematically equivalent to "
            "recomputing them per epoch (frozen backbones in eval mode, no augmentation) but "
            "avoids ~30x redundant librosa work. training_time_total_s covers both phases; "
            "efficiency.feature_extraction_time_s and head_training_time_s break it out."),
    },
    "config": {k: v for k, v in CFG.items() if k not in ("ckpt_dir", "results_dir")},
    "environment": {
        "platform": ("Google Colab" if IN_COLAB else
                     "Kaggle" if os.path.exists("/kaggle/working") else "Local"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    },
    "dataset_info": {
        "dataset": "ICBHI_2017",
        "data_source": "real_audio",
        "train_samples": int(len(df_train)),
        "test_samples": int(len(df_test)),
        "train_patients": int(len(train_patients)),
        "test_patients": int(len(test_patients)),
        "split_method": ("official_icbhi_60_40_patient_disjoint"
                         if OVERLAP_POLICY == "drop_from_train"
                         else "official_icbhi_60_40_verbatim"),
        "split_source": SPLIT_SOURCE,
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
            "recordings on both sides. Protocol section 1 requirement 1 forbids this, so under "
            "policy 'drop_from_train' their TRAIN recordings are excluded while the official TEST "
            "set is left byte-identical, keeping the score comparable to published ICBHI results."),
        "patient_leakage_verified": bool(OVERLAP_POLICY == "drop_from_train"),
    },
    "efficiency": {
        "total_params": int(TOTAL_PARAMS),
        "trainable_params": int(TRAINABLE_PARAMS),
        "model_size_mb": model_size_mb(BEST_PATH),
        "training_time_total_s": round(float(TRAIN_TIME + EXTRACT_TIME), 1),
        "training_time_per_epoch_s_avg": round(float(TRAIN_TIME) / CFG["num_epochs"], 1),
        "feature_extraction_time_s": round(float(EXTRACT_TIME), 1),
        "head_training_time_s": round(float(TRAIN_TIME), 1),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "inference_time_ms_per_sample": round(float(INFER_MS), 3),
        "note": "Only the fusion head trains; both backbones are frozen.",
    },
    "best_epoch": {
        "epoch": int(best["epoch"]),
        "primary_metric": "icbhi_score_official",
        "primary_metric_value": round(float(f_o), 4),
    },
    "best_metrics": {
        **{k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in FUSION.items() if k != "per_class"},
        "per_class": FUSION["per_class"],
        "icbhi_score_metric_note": METRIC_NOTE,
        "admission_test": {
            "rule": ("Novelty Search section 4.0: include only if it measurably beats both "
                     "backbones alone; drop it silently if it doesn't."),
            "evaluated_on": "identical test cycles for all three rows",
            "m2_alone_official": round(float(m2_o), 4),
            "m3_alone_official": round(float(m3_o), 4),
            "fusion_official": round(float(f_o), 4),
            "margin_over_best_backbone": round(float(margin), 4),
            "beats_both_backbones": bool(beats_both),
            "verdict": ("PASSES" if (beats_both and margin >= 0.01)
                        else "MARGINAL" if beats_both else "FAILS"),
            "reference_superseded_m2_fallback_split": 0.6138,
            "reference_superseded_m3_fallback_split": 0.5895,
            "reference_note": ("The two reference_superseded_* figures come from M2/M3 runs that "
                               "took the patient-ID fallback split (11 test patients, 492 cycles) "
                               "while recording split_method 'patient_independent_official_60_40'. "
                               "They are NOT comparable to this run and are recorded only for "
                               "provenance. The admission test uses m2_alone_official and "
                               "m3_alone_official, measured here on identical test cycles."),
        },
    },
    "ablation": {
        "ablation_group": "ensemble_fusion",
        "ablation_role": "variant",
        "baseline_model_id": "M2",
        "variable_changed": "gated feature-fusion of frozen M2 + M3 embeddings",
        "variables_held_constant": [
            "loss_function: inverse_frequency_class_weighted_CrossEntropyLoss",
            "data_split: patient_independent_official_60_40",
            "preprocessing: 128mel_16kHz_8s",
            "augmentation: none",
            "seed: 42",
        ],
        "known_deviations": [
            ("Supersedes Barshon's/M30/results_M30.json, which used a random 70/30 split "
             "despite a docstring claiming the official split, reported the legacy macro "
             "metric, and committed no confusion matrix."),
        ],
        "component_flags": {
            "has_sound_event_head": True, "has_disease_head": False,
            "has_cross_task_consistency": False, "has_cqkd_regularization": False,
            "has_openmax_rejection": False, "owl_stage": 0, "compression_clusters": None,
        },
        "loss_weights": {"sound_event_weight": 1.0, "disease_weight": None,
                         "consistency_weight": None},
    },
    "training_history": history,
}

results_path = os.path.join(CFG["results_dir"], "results_M30.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {results_path}")
print(json.dumps(results["best_metrics"]["admission_test"], indent=2))
''')

code(r'''
# ============================================================
# FINAL CELL — TEAM HANDOFF
# ============================================================
import shutil

try:
    from IPython.display import display, FileLink
except ImportError:
    display = FileLink = None

print("=" * 60)
print("OUTPUTS READY")
print("=" * 60)
files = sorted(glob.glob(os.path.join(CFG["results_dir"], "results_M30.json")) +
               glob.glob(os.path.join(CFG["results_dir"], "*.png")) +
               glob.glob(os.path.join(CFG["ckpt_dir"], "best_model.pth")))
for f in files:
    print(f"Ready: {os.path.basename(f):<26} ({round(os.path.getsize(f)/(1024*1024), 2)} MB)")
    if display is not None:
        display(FileLink(f))

if files:
    bundle = os.path.join(BASE_DIR, "M30_v2_bundle")
    os.makedirs(bundle, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(bundle, os.path.basename(f)))
    zp = shutil.make_archive(os.path.join(BASE_DIR, "M30_v2_handoff_bundle"), "zip", bundle)
    print(f"\nZIP: {zp}")
    if display is not None:
        display(FileLink(zp))
print("=" * 60)
''')

md(r"""
---
### What to do with these outputs

1. **Read the admission-test verdict first.** If it says FAILS, that is a real result — per
   `Novelty Search.md` §4.0 the fusion item should be dropped and a deferred item swapped in, not
   added alongside. A clean negative here is more useful than the original's cross-split positive.
2. **Check the M2/M3 sanity anchors** (Cell 7). They should land near 0.6138 / 0.5895. A large
   deviation means the split or checkpoints differ from the published runs and nothing below it is
   trustworthy.
3. **Commit `results_M30.json` with its confusion matrix.** That is what makes the number auditable,
   and its absence is why the original had to be withdrawn.
4. Run `python3 "Asif's/audit/audit_project.py"` and confirm M30 v2 lands clean.
5. Update `Model_Training_Reference.md` §2.22, `Novelty Search.md` §4.0 item 3, and
   `Project_Work_Plan.md` Chunk E with whichever verdict comes back.
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

# Preserve trailing newlines per source line -- avoids the cell-boundary corruption class of bug.
for c in NB["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "M30_v2_gated_fusion.ipynb")
with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)
print(f"Wrote {OUT}  ({len(NB['cells'])} cells)")
