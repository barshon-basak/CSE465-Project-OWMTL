#!/usr/bin/env python3
"""
Generator for M41 — Swin-T sound-event backbone notebook.

Why a generator (Asif's/CLAUDE.md convention): the .ipynb is an artifact, this
script is the source. Regenerate with `python gen_M41.py` after any edit so the
notebook stays reproducible and diff-able.

M41 is Farhana's pre-trained transformer for the CSE465 implementation checklist
(RTK_requirements.md req. 2 + 3). It supersedes / absorbs M23. One notebook,
two runs:

    CFG["is_augmented"] = False  ->  results_M41.json       (clean baseline)
    CFG["is_augmented"] = True   ->  results_M41_aug.json    (SpecAugment)

Protocol compliance (Model_Training_Protocol.md):
  * split: Asif's/audit/official_split.py, mode="patient_independent" — RAISES if
    the split file is missing (no silent fallback — README trap #2)
  * metric: leads with icbhi_score_official; commits confusion_matrix_raw
  * §4 results-JSON schema incl. the §4.1 ablation block
  * §5 plots, §11 checkpoint/persistence + handoff cell
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "M41_SwinT_soundevent.ipynb")


def _src(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


CELLS = []

CELLS.append(md(r"""
# M41 — Swin-T Sound-Event Backbone

**Author:** Farhana · **Model ID:** M41 · **Supersedes:** M23 (AST + SpecAugment)

**What this is.** A recent pre-trained transformer (Swin Transformer Tiny, ImageNet-1k)
fine-tuned on ICBHI 2017 log-mel spectrograms for 4-class sound-event classification
(Normal / Crackle / Wheeze / Both). This is Farhana's entry for the CSE465 implementation
checklist — `RTK_requirements.md` requirement 2 (pre-trained model, one per member) and
requirement 3 (≥ X = 4 transformer models).

**One notebook, two runs.** Flip `CFG["is_augmented"]`:
- `False` → clean baseline → `results_M41.json`
- `True`  → SpecAugment on the training split only → `results_M41_aug.json`

The clean-vs-augmented pair is requirement 5 / 7.

**Protocol** (`Model_Training_Protocol.md`):
- Split: `Asif's/audit/official_split.py`, `mode="patient_independent"` — the corrected
  official 60/40 split. **The notebook raises if the split file is missing** — no fallback.
- Leads with `icbhi_score_official` = (Se + Sp) / 2, pooled-abnormal. Commits `confusion_matrix_raw`.
- Full §3 metric suite, §4 results JSON incl. the §4.1 `ablation` block, §5 plots,
  §11 checkpoint safety + handoff cell.

**Expectation.** Transformers are expected to *lose* to the M2 CNN on 920 recordings
(`RTK_requirements.md` §4: AST was ~24x the size for a worse score). Do not tune to win —
the accuracy/compute trade-off is the reportable finding.
"""))

CELLS.append(md(r"""
## Section 0 — Data + code bootstrap (Colab / Kaggle)

Runs first so **Run all** works after a re-upload — no hand-added cell needed. On both Colab and
Kaggle it locates the ICBHI `audio_and_txt_files` folder and assembles these 5 files into
`<workdir>/OWMTL/Asif's/` — from either a full repo copy **or** a flat folder (e.g. `m41_deps`):
`__init__.py`, `icbhi.py`, `features.py`, `official_split.py`, `ICBHI_challenge_train_test.txt`.

- **Colab:** searches your Drive (mounts it first).
- **Kaggle:** searches `/kaggle/input` — add the ICBHI dataset *and* your `m41_deps` dataset.
- **Local:** no-op — Section 1 walks up from the working dir.
"""))

CELLS.append(code(r"""
import os, sys, glob, shutil

ON_KAGGLE = os.path.isdir("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
if ON_KAGGLE:
    ON_COLAB = False                # Kaggle now ships a google.colab stub — check Kaggle first
else:
    try:
        import google.colab
        ON_COLAB = True
    except ImportError:
        ON_COLAB = False
REPO = None                                   # set on success; Section 1 reuses it
ICBHI_DIR = None                              # set on success; Section 3 reuses it
print("env  :", "Colab" if ON_COLAB else ("Kaggle" if ON_KAGGLE else "local"))

def _all(pat):
    return sorted(glob.glob(pat, recursive=True))

def _first(pats, need_wav=False):
    for pat in pats:
        for h in _all(pat):
            if need_wav and not glob.glob(os.path.join(h, "*.wav")):
                continue
            return h
    return None

if ON_COLAB or ON_KAGGLE:
    if ON_COLAB:
        if not os.path.isdir("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        SEARCH = ["/content/drive/MyDrive", "/content/drive/Shareddrives", "/content/drive"]
        WORK = "/content/OWMTL"
    else:
        SEARCH = ["/kaggle/input"]
        WORK = "/kaggle/working/OWMTL"

    # (1) ICBHI audio — pick the MOST COMPLETE copy found (ICBHI has 920 recordings)
    roots = (["/content"] if ON_COLAB else []) + SEARCH
    cand = []
    for r in roots:
        for h in _all(f"{r}/**/audio_and_txt_files"):
            cand.append((len(glob.glob(os.path.join(h, "*.wav"))), h))
    cand.sort(reverse=True)
    assert cand, ("ICBHI 'audio_and_txt_files' folder not found.\nSearched: " + ", ".join(roots))
    n_wav, aud = cand[0]
    ICBHI_DIR = os.path.dirname(aud)
    print(f"ICBHI : {ICBHI_DIR}  ({n_wav} wav files)")
    assert n_wav >= 900, (
        f"\n*** Only {n_wav} of ~920 ICBHI recordings present — this copy is INCOMPLETE. ***\n"
        "A Drive folder upload silently dropped files. Run the NEXT cell once "
        "(Kaggle CLI download straight to Drive), then re-run Section 0.")
    if ON_COLAB:
        if os.path.islink("/content/Respiratory_Sound_Database") or os.path.exists("/content/Respiratory_Sound_Database"):
            os.system("rm -rf /content/Respiratory_Sound_Database")
        os.symlink(ICBHI_DIR, "/content/Respiratory_Sound_Database")

    # (2) the 5 repo files -> WORK/Asif's/...
    NEED = {"owmtl/__init__.py": "__init__.py", "owmtl/icbhi.py": "icbhi.py",
            "owmtl/features.py": "features.py", "audit/official_split.py": "official_split.py",
            "ICBHI_challenge_train_test.txt": "ICBHI_challenge_train_test.txt"}
    root = f"{WORK}/Asif's"
    ready = (not os.path.islink(WORK)
             and all(os.path.isfile(os.path.join(root, p)) for p in NEED))
    if not ready:
        if os.path.islink(WORK):
            os.unlink(WORK)
        shutil.rmtree(WORK, ignore_errors=True)
        os.makedirs(f"{root}/owmtl"); os.makedirs(f"{root}/audit")
        rp = _first([f"{d}/**/Asif's/owmtl/icbhi.py" for d in SEARCH])
        if rp:                                          # a full repo copy
            base = rp.split("/Asif's/")[0] + "/Asif's"
            resolve = lambda rel: os.path.join(base, rel)
            print("code  : repo copy at", base)
        else:                                           # a flat deps folder
            osp = _first([f"{d}/**/official_split.py" for d in SEARCH])
            assert osp, ("Dependency files not found.\nSearched: " + ", ".join(SEARCH) + "\n"
                         "Need one folder containing: __init__.py  icbhi.py  features.py  "
                         "official_split.py  ICBHI_challenge_train_test.txt")
            dd = os.path.dirname(osp)
            resolve = lambda rel: os.path.join(dd, os.path.basename(rel))
            print("code  : flat deps folder at", dd)
        for dst_rel in NEED:
            s = resolve(dst_rel)
            assert os.path.isfile(s), (f"missing: {os.path.basename(s)} — expected next to "
                                       f"official_split.py at {os.path.dirname(s)}")
            shutil.copy(s, os.path.join(root, dst_rel))
    for p in (root, f"{root}/audit"):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)
    REPO = WORK
    print("repo  :", REPO, "| icbhi.py:", os.path.isfile(f"{root}/owmtl/icbhi.py"),
          "| split file:", os.path.isfile(f"{root}/ICBHI_challenge_train_test.txt"))
else:
    print("local — Section 1 discovers the repo from parent dirs")
""".strip("\n")))

CELLS.append(md(r"""
### One-time ICBHI download (run only if Section 0 says the dataset is INCOMPLETE)

Downloads the full 920-recording ICBHI dataset from Kaggle straight to your Drive, so it
persists and Section 0 picks it up automatically next time. You need a Kaggle API token:
kaggle.com -> Settings -> API -> **Create New Token** (downloads `kaggle.json`).

Set `RUN_ICBHI_DOWNLOAD = True`, run this cell, upload `kaggle.json` when prompted, then
re-run from Section 0. Leave it `False` for normal runs.
"""))

CELLS.append(code(r"""
RUN_ICBHI_DOWNLOAD = False   # set True once, then flip back to False

if RUN_ICBHI_DOWNLOAD:
    from google.colab import files
    print("Upload kaggle.json:")
    files.upload()
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    os.system("cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    os.system(f"{sys.executable} -m pip install -q kaggle")
    dest = "/content/drive/MyDrive/icbhi_full"
    os.makedirs(dest, exist_ok=True)
    os.system(f'kaggle datasets download -d vbookshelf/respiratory-sound-database -p "{dest}" --unzip')
    import glob as _g
    print("wav files now on Drive:",
          len(_g.glob(dest + "/**/audio_and_txt_files/*.wav", recursive=True)))
    print("done — now re-run Section 0 (it will prefer this complete copy)")
else:
    print("skipped (RUN_ICBHI_DOWNLOAD is False)")
""".strip("\n")))

CELLS.append(md("## Section 1 — Environment, repo discovery, dependencies"))

CELLS.append(code(r"""
# --- standard imports -------------------------------------------------------
import os, sys, glob, json, time, math, random, tempfile, datetime, warnings, platform, types
warnings.filterwarnings("ignore")

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- torch first ---------------------------------------------------------
import torch
# Some Kaggle images leave core submodules unbound on the `torch` namespace, so
# timm's dynamo import later fails with "module 'torch' has no attribute '_utils'".
# Importing them explicitly binds them to the parent module and fixes it.
for _m in ("torch._utils", "torch.utils", "torch.fx", "torch.overrides",
           "torch.serialization", "torch.storage", "torch._dynamo"):
    try:
        __import__(_m)
    except Exception as _e:
        print(f"  (note: {_m} -> {type(_e).__name__})")
import torch.nn as nn
import torch.nn.functional as F

# timm also pulls torchvision (only for an optional feature-extractor path Swin
# doesn't use). If that's broken too, stub it so timm's own except-ImportError fires.
try:
    import torchvision  # noqa: F401
except Exception as _e:
    print(f"torchvision broken ({type(_e).__name__}) — stubbing it (not needed for Swin).")
    sys.modules["torchvision"] = types.ModuleType("torchvision")

# --- pip (Kaggle/Colab) ----------------------------------------------------
def _pip(pkg):
    mod = pkg.split("[")[0].split("==")[0].replace("-", "_")
    try:
        __import__(mod)
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q {pkg}")

for _p in ("librosa", "soundfile", "scikit-learn", "seaborn", "tqdm"):
    _pip(_p)

# timm >= 1.0.15 imports torch._dynamo at module load (naflexvit's @disable_compiler),
# which crashes on Kaggle images with a half-broken torch. Pin below that.
def _load_timm():
    try:
        import timm
        if tuple(int(x) for x in timm.__version__.split(".")[:3]) >= (1, 0, 15):
            raise RuntimeError(f"timm {timm.__version__} too new")
        return timm
    except Exception as e:
        print(f"installing timm==1.0.11  (reason: {type(e).__name__}: {e})")
        os.system(f"{sys.executable} -m pip install -q --no-deps timm==1.0.11")
        for _k in [k for k in list(sys.modules) if k == "timm" or k.startswith("timm.")]:
            del sys.modules[_k]
        import timm
        return timm

timm = _load_timm()
print("timm", timm.__version__)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"Python {platform.python_version()} | torch {torch.__version__} | device {DEVICE} ({GPU_NAME})")
""".strip("\n")))

CELLS.append(code(r"""
# --- locate the OWMTL repo (need Asif's/owmtl/ and Asif's/audit/) -----------
def _find_repo():
    cwd = os.getcwd()
    cands = [cwd]
    cands += [os.path.abspath(os.path.join(cwd, *([".."] * k))) for k in range(1, 7)]
    cands += sorted(glob.glob("/kaggle/input/*"))
    cands += ["/content/OWMTL", "/content", "/content/drive/MyDrive/OWMTL",
              "/content/drive/MyDrive/CSE465-Project-OWMTL"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "Asif's", "owmtl", "icbhi.py")):
            return c
    return None

REPO = globals().get("REPO") or _find_repo()   # Section 0 sets REPO on Colab
if REPO is None:
    raise FileNotFoundError(
        "OWMTL repo not found. Run Section 0 first (Runtime > Run all). If Section 0 "
        "printed an error, fix that — it means the ICBHI folder or the 5 dep files "
        "are not where it looked on Drive.")
print("REPO:", REPO)

for _p in (os.path.join(REPO, "Asif's"), os.path.join(REPO, "Asif's", "audit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from owmtl import icbhi, features          # shared cycle index + preprocessing
from official_split import load_split      # corrected official 60/40 split (raises, no fallback)
""".strip("\n")))

CELLS.append(md("## Section 2 — Configuration"))

CELLS.append(code(r"""
CFG = {
    # ---- run mode: the only switch you change between the two runs ----------
    "is_augmented": False,          # True -> SpecAugment run -> results_M41_aug.json
    "eval_only": False,             # True -> skip training, evaluate whatever checkpoint is loaded

    # ---- audio (Model_Training_Protocol.md §2 — do not deviate) -------------
    "sample_rate": 16000, "duration_s": 8.0,
    "n_mels": 128, "n_fft": 1024, "hop_length": 160, "win_length": 400,
    "f_min": 50.0, "f_max": 2000.0,

    # ---- preprocessing stages (req. 1 "ALL preprocessing"; each is ablatable) ----
    "pp_bandpass": True,            # 4th-order Butterworth 50-2000 Hz, zero-phase (features.load_cycle)
    "pp_cyclic_pad": True,          # reflect/tile pad to 8 s, not zeros    (features.load_cycle)
    "pp_amplitude_norm": True,      # per-cycle peak normalisation           (features.load_cycle)
    "pp_denoise": True,             # spectral gating (implemented in Section 4)
    "pp_logmel": True,              # 128-mel, 50-2000 Hz, power_to_db       (features.log_mel)
    "pp_train_standardize": True,   # zero-mean/unit-var, stats fit on TRAIN only
    "pp_imagenet_norm": True,       # 3-channel replicate + ImageNet mean/std (Swin needs this)

    # ---- model ------------------------------------------------------------
    "model_name": "swin_tiny_patch4_window7_224",
    "img_size": 224, "pretrained": True, "num_classes": 4,
    "finetune_mode": "full",        # "frozen" (linear probe) | "last_block" | "full"
    "drop_path_rate": 0.2,          # stochastic depth, applied only when the backbone trains
    "head_dropout": 0.3,

    # ---- training -------------------------------------------------------
    # Backbone_lr is deliberately tiny (2e-5) — full fine-tune at 1e-4 diverged. With the
    # full dataset (~4k fit cycles) + drop_path + label smoothing this is stable. Monitor
    # is icbhi_score_official (not val_loss, which bottoms while the model just predicts
    # Normal). Do NOT raise backbone_lr to chase a number — report the trade-off honestly.
    "batch_size": 16, "num_epochs": 45,
    "backbone_lr": 2e-5, "head_lr": 8e-4, "weight_decay": 0.02,
    "warmup_epochs": 3, "early_stop_patience": 12, "min_epochs": 22,
    "grad_clip": 1.0, "use_amp": True,
    "class_weight_power": 0.7,      # 0=uniform, 1=full inverse-freq; 0.7 pushes Se up without the thrash
    "label_smoothing": 0.1,
    "monitor": "icbhi_score_official",   # best-ckpt / early-stop signal (max); val_loss=min if you switch back
    "seed": SEED,
}

MODEL_ID = "M41_aug" if CFG["is_augmented"] else "M41"
CFG["augmentation_method"] = "SpecAugment (time+freq masking, train only)" if CFG["is_augmented"] else "none"

# ---- paths ------------------------------------------------------------------
if os.path.isdir("/kaggle/working"):
    BASE = "/kaggle/working/M41"
elif os.path.isdir("/content"):
    BASE = "/content/drive/MyDrive/OWMTL/M41" if os.path.isdir("/content/drive/MyDrive") else "/content/M41"
else:
    BASE = os.path.join(REPO, "Farhana's", "M41", "run")

CFG["ckpt_dir"]   = os.path.join(BASE, "checkpoints", MODEL_ID)
CFG["results_dir"] = os.path.join(BASE, "results")
CFG["cache_dir"]  = os.path.join("/content/owmtl_cache" if os.path.isdir("/content") else
                                 ("/kaggle/working/owmtl_cache" if os.path.isdir("/kaggle/working")
                                  else os.path.join(BASE, "cache")))
for d in (CFG["ckpt_dir"], CFG["results_dir"], CFG["cache_dir"]):
    os.makedirs(d, exist_ok=True)

print(f"MODEL_ID = {MODEL_ID}   is_augmented = {CFG['is_augmented']}")
print("results ->", CFG["results_dir"])
print("ckpts   ->", CFG["ckpt_dir"])
""".strip("\n")))

CELLS.append(md("## Section 3 — Cycle index + corrected official 60/40 split"))

CELLS.append(code(r"""
from dataclasses import asdict

# --- locate ICBHI audio ----------------------------------------------------
_DATA_CANDS = [
    globals().get("ICBHI_DIR"),                 # set by Section 0
    os.path.dirname(globals().get("ICBHI_DIR") or "x"),
    "/content/Respiratory_Sound_Database/Respiratory_Sound_Database",
    "/content/Respiratory_Sound_Database",
    "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database",
    "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database",
    os.path.join(REPO, "data"),
]
PATHS = None
for c in [c for c in _DATA_CANDS if c] + sorted(glob.glob("/kaggle/input/*")):
    try:
        PATHS = icbhi.locate(c)
        DATA_ROOT = c
        break
    except Exception:
        continue
if PATHS is None:
    raise FileNotFoundError(
        "ICBHI audio not found. Add the 'Respiratory Sound Database' dataset "
        "(vbookshelf/respiratory-sound-database) and re-run.")
print("ICBHI audio:", PATHS.audio_dir)

# --- cycle index (annotations only, no audio decoded) --------------------
rows = icbhi.build_cycle_index(PATHS)
index = [asdict(r) for r in rows]
print(f"{len(index)} cycles / {len({r['stem'] for r in index})} recordings / "
      f"{len({r['patient_id'] for r in index})} patients")

# --- corrected official 60/40 split — RAISES if the split file is missing ----
split_map, split_info = load_split(mode="patient_independent")
print("\nSplit policy:\n ", split_info["policy"])

# A few recordings exist in the audio but not in the official split file (a known ICBHI
# quirk, e.g. 226_1b1_Pl_sc_LittC2SE). Assign each to the SAME partition as the rest of
# its patient's recordings — this is the patient-independence rule, not a guess. If a
# patient's recordings span both halves (156/218), official_split already moved them to
# train, so the set is {"train"} and the extra recording joins train too.
_pat_part = {}
for _stem, _part in split_map.items():
    _pat_part.setdefault(int(_stem.split("_")[0]), set()).add(_part)
_reassigned = []
for r in index:
    if r["stem"] not in split_map:
        parts = _pat_part.get(r["patient_id"])
        if not parts:
            continue                                   # whole patient absent — handled below
        split_map[r["stem"]] = next(iter(parts)) if len(parts) == 1 else "train"
        _reassigned.append(r["stem"])
if _reassigned:
    print(f"note: {len(_reassigned)} recording(s) not in the split file assigned to their "
          f"patient's partition: {sorted(set(_reassigned))}")

missing = {r["stem"] for r in index} - set(split_map)
if missing:
    raise RuntimeError(f"{len(missing)} recordings from patients entirely absent from the "
                       f"split file: {sorted(missing)[:5]} — refusing to guess.")

for r in index:
    r["partition"] = split_map[r["stem"]]

train_rows = [r for r in index if r["partition"] == "train"]
test_rows  = [r for r in index if r["partition"] == "test"]

# --- patient-independence assertion (Essential #1) -----------------------
p_tr = {r["patient_id"] for r in train_rows}
p_te = {r["patient_id"] for r in test_rows}
assert not (p_tr & p_te), f"patient leakage: {sorted(p_tr & p_te)}"

# --- patient-level val split carved from TRAIN --------------------------
# Monitored during training for early-stop / best-checkpoint. TEST is evaluated
# only once, in Section 10 — so the reported number never peeked at test.
_vp = sorted(p_tr)                        # sorted first -> reproducible shuffle
np.random.RandomState(SEED).shuffle(_vp)
val_patients = set(_vp[:max(1, round(0.15 * len(_vp)))])
val_rows = [r for r in train_rows if r["patient_id"] in val_patients]
fit_rows = [r for r in train_rows if r["patient_id"] not in val_patients]

CLASSES = list(icbhi.SOUND_EVENT_CLASSES)   # ["Normal","Crackle","Wheeze","Both"]

def _dist(rs):
    c = np.bincount([r["label"] for r in rs], minlength=4)
    return "  ".join(f"{CLASSES[i]}={c[i]}" for i in range(4))

print(f"\nFIT    {len(fit_rows):5d} cycles  {len(p_tr)-len(val_patients):3d} patients   {_dist(fit_rows)}")
print(f"VAL    {len(val_rows):5d} cycles  {len(val_patients):3d} patients   {_dist(val_rows)}")
print(f"TEST   {len(test_rows):5d} cycles  {len(p_te):3d} patients   {_dist(test_rows)}")
for cls in range(1, 4):
    n = sum(r["label"] == cls for r in test_rows)
    if n < 30:
        print(f"  ! only {n} test cycles for {CLASSES[cls]} — per-class F1 will be noisy")
""".strip("\n")))

CELLS.append(md(r"""
## Section 4 — Preprocessing pipeline + feature cache

Full stage list (`Model_Training_Protocol.md` §2, `RTK_requirements.md` §3a). Stages 1-6 come
from the shared `owmtl.features` module so this notebook is byte-identical to the other backbones;
stage 6b (denoising) is added here and is ablatable via `CFG["pp_denoise"]`.

| # | Stage | Where |
|---|---|---|
| 1 | Load + resample 16 kHz mono | `features.load_cycle` |
| 2 | Butterworth band-pass 50-2000 Hz, zero-phase | `features.load_cycle` (`pp_bandpass`) |
| 3 | Cycle segmentation from ICBHI annotation times | `icbhi.build_cycle_index` |
| 4 | Duration standardisation 8 s, reflect/tile pad | `features.load_cycle` (`pp_cyclic_pad`) |
| 5 | Per-cycle peak amplitude normalisation | `features.load_cycle` (`pp_amplitude_norm`) |
| 6a | Spectral-gating denoise | this cell (`pp_denoise`) |
| 6b | Log-mel 128 / 50-2000 Hz / `power_to_db` | `features.log_mel` |
| 7 | Per-spectrogram standardisation, stats fit on TRAIN only | Section 6 (`pp_train_standardize`) |
| 8 | 3-channel replicate + ImageNet mean/std | Section 6 (`pp_imagenet_norm`) |
| 9 | Label encoding (4-class sound event) | `icbhi.sound_event_label` |
| 10 | Class-imbalance handling (inverse-freq weighted CE) | Section 7 |
| 11 | Patient-independent split | Section 3 |
"""))

CELLS.append(code(r"""
import librosa

CFGA = features.AudioConfig()   # defaults already match Protocol §2 exactly
assert (CFGA.sample_rate, CFGA.n_mels, CFGA.n_fft, CFGA.hop_length,
        CFGA.win_length, CFGA.f_min, CFGA.f_max) == (16000, 128, 1024, 160, 400, 50.0, 2000.0)

def spectral_gate(y, sr=16000, n_fft=1024, hop=160, thresh=1.5):
    # Stage 6a — stationary spectral gating: estimate a per-frequency noise floor
    # from the quietest 10% of frames, null bins that do not clear thresh * floor.
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    floor = np.percentile(mag, 10, axis=1, keepdims=True)
    mag = mag * (mag > thresh * floor)
    out = librosa.istft(mag * np.exp(1j * phase), hop_length=hop, length=len(y))
    return out.astype(np.float32)

def cycle_to_logmel(wav_path, start, end, waveform=None):
    seg = features.load_cycle(wav_path, start, end, CFGA, waveform=waveform)  # stages 1-5
    if CFG["pp_denoise"]:
        seg = spectral_gate(seg, CFGA.sample_rate, CFGA.n_fft, CFGA.hop_length)
    return features.log_mel(seg, CFGA)   # stage 6b -> (128, ~801) float32 dB

# --- build / load the feature cache (decode each .wav once) ----------------
CACHE_KEY = f"logmel_dn{int(CFG['pp_denoise'])}"
arr_path = os.path.join(CFG["cache_dir"], CACHE_KEY + ".npy")
meta_path = os.path.join(CFG["cache_dir"], CACHE_KEY + ".meta.json")

if os.path.isfile(arr_path) and os.path.isfile(meta_path):
    SPEC = np.load(arr_path, mmap_mode="r")
    UID2IDX = {u: i for i, u in enumerate(json.load(open(meta_path))["cycle_uids"])}
    print("cache hit:", SPEC.shape)
else:
    probe = cycle_to_logmel(None, 0.0, 1.0, waveform=np.zeros(CFGA.n_samples, np.float32))
    SPEC = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float16,
                                     shape=(len(index), *probe.shape))
    by_stem = {}
    for i, r in enumerate(index):
        by_stem.setdefault(r["stem"], []).append(i)
    from tqdm.auto import tqdm
    for stem in tqdm(sorted(by_stem), desc="decode", unit="rec"):
        wp = os.path.join(PATHS.audio_dir, stem + ".wav")
        wav, _ = librosa.load(wp, sr=CFGA.sample_rate, mono=True)
        for i in by_stem[stem]:
            SPEC[i] = cycle_to_logmel(wp, index[i]["start"], index[i]["end"], waveform=wav).astype(np.float16)
    SPEC.flush()
    json.dump({"cycle_uids": [r["cycle_uid"] for r in index], "shape": list(SPEC.shape)},
              open(meta_path, "w"))
    UID2IDX = {r["cycle_uid"]: i for i, r in enumerate(index)}
    print("cache built:", SPEC.shape)
""".strip("\n")))

CELLS.append(md("## Section 5 — Dataset + DataLoaders"))

CELLS.append(code(r"""
from torch.utils.data import Dataset, DataLoader

# stage 7 — standardisation stats fit on the FIT cycles only (no val/test leak)
_fit_idx = np.sort(np.array([UID2IDX[r["cycle_uid"]] for r in fit_rows]))
_sample = _fit_idx if len(_fit_idx) <= 4000 else np.random.RandomState(SEED).choice(_fit_idx, 4000, replace=False)
NORM = features.fit_norm_stats(np.asarray(SPEC[np.sort(_sample)], dtype=np.float32))
print("fit-only norm stats:", NORM)

_IMEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_ISTD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class ICBHICycles(Dataset):
    def __init__(self, rs, augment=False):
        self.rs = rs
        self.augment = augment
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, k):
        r = self.rs[k]
        spec = np.asarray(SPEC[UID2IDX[r["cycle_uid"]]], dtype=np.float32)   # (128, W) log-mel dB
        spec = features.apply_norm(spec, NORM)                              # stage 7 — fit-only z-norm
        if self.augment:
            spec = features.spec_augment(spec, rng=self.rng, freq_axis=0)   # stage 8 (train only)
        t = torch.from_numpy(spec)[None, None]                              # (1,1,128,W)
        t = F.interpolate(t, size=(CFG["img_size"], CFG["img_size"]),
                          mode="bilinear", align_corners=False)[0]          # (1,224,224)
        t = t.clamp(-3.0, 3.0).add(3.0).div(6.0)                            # -> [0,1], deterministic (no per-sample min-max)
        t = t.repeat(3, 1, 1)
        t = (t - _IMEAN) / _ISTD                                            # stage 8 — ImageNet norm
        return t, r["label"]

train_ds = ICBHICycles(fit_rows,  augment=CFG["is_augmented"])
val_ds   = ICBHICycles(val_rows,  augment=False)
test_ds  = ICBHICycles(test_rows, augment=False)
NW = 2 if os.path.isdir("/kaggle") or os.path.isdir("/content") else 0
train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=NW, drop_last=True, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=NW, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=NW, pin_memory=True)

# stage 10 — class weights from the FIT split, tempered by class_weight_power
_cnt = np.bincount([r["label"] for r in fit_rows], minlength=4).astype(np.float64)
_w = (_cnt.sum() / (4 * _cnt)) ** CFG["class_weight_power"]
CLASS_W = torch.tensor(_w / _w.mean(), dtype=torch.float32)                 # normalised around 1.0
print("class weights:", [round(x, 3) for x in CLASS_W.tolist()])
""".strip("\n")))

CELLS.append(md(r"""
## Section 6 — Model: Swin-T (ImageNet-1k pretrained)

**Why the backbone is (mostly) frozen.** Swin-T is pretrained on ImageNet *photographs*; a
128-mel spectrogram upscaled to 224x224 is far out of that distribution. Full fine-tuning on
920 recordings destroys the pretrained features in ~2 epochs and then memorises the training
set (val loss diverges). `finetune_mode` controls how much adapts:

| mode | trains | use when |
|---|---|---|
| `"frozen"` | classifier head only (linear probe on frozen features) | safest; guaranteed not to diverge |
| `"last_block"` *(default)* | final Swin stage + final norm + head, backbone_lr 2e-5 | best stable number; still regularised |
| `"full"` | everything | only if you have a much larger dataset — expect divergence here |

The reportable finding is the CNN-vs-transformer accuracy/compute trade-off, **not** a tuned
transformer score (`RTK_requirements.md` §4).
"""))

CELLS.append(code(r"""
import timm

def build_model():
    dpr = CFG["drop_path_rate"] if CFG["finetune_mode"] != "frozen" else 0.0
    return timm.create_model(CFG["model_name"], pretrained=CFG["pretrained"],
                             num_classes=CFG["num_classes"], in_chans=3,
                             drop_path_rate=dpr, drop_rate=CFG["head_dropout"])

model = build_model().to(DEVICE)

# --- freeze policy -------------------------------------------------------
mode = CFG["finetune_mode"]
for p in model.parameters():
    p.requires_grad = (mode == "full")
for n, p in model.named_parameters():
    if n.startswith("head"):
        p.requires_grad = True                       # classifier head always trains
    elif mode == "last_block" and ("layers.3" in n or n.startswith("norm")):
        p.requires_grad = True                       # final Swin stage + final LayerNorm

_head = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
_back = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
groups = [{"params": _head, "lr": CFG["head_lr"]}]
if _back:
    groups.append({"params": _back, "lr": CFG["backbone_lr"]})
optimizer = torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])

def lr_lambda(ep):
    if ep < CFG["warmup_epochs"]:
        return (ep + 1) / CFG["warmup_epochs"]
    prog = (ep - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
    return 0.5 * (1 + math.cos(math.pi * prog))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(weight=CLASS_W.to(DEVICE), label_smoothing=CFG["label_smoothing"])
scaler = torch.cuda.amp.GradScaler(enabled=CFG["use_amp"])

TOTAL_PARAMS = sum(p.numel() for p in model.parameters())
TRAIN_PARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{CFG['model_name']} [{mode}]: {TOTAL_PARAMS/1e6:.2f} M params, "
      f"{TRAIN_PARAMS/1e6:.3f} M trainable")
""".strip("\n")))

CELLS.append(md("## Section 7 — Metric suite (§3)"))

CELLS.append(code(r"""
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

def evaluate(net, dl):
    net.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=CFG["use_amp"]):
                out = net(x)
                loss_sum += criterion(out, y).item() * len(y)
            n += len(y)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)

    acc = accuracy_score(y_true, y_pred)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # ICBHI 2017 official: Se = correct abnormal / all abnormal ; Sp = correct Normal / all Normal
    Sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    abn = cm[1:].sum()
    Se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else 0.0
    icbhi_official = (Se + Sp) / 2

    # project-internal macro variant (reported but never led with)
    spec_c = []
    for i in range(4):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec_c.append(tn / (tn + fp) if (tn + fp) else 0.0)
    icbhi_macro = (r_m + float(np.mean(spec_c))) / 2

    return {
        "loss": loss_sum / n,
        "accuracy": float(acc),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "specificity_macro": float(np.mean(spec_c)),
        "Se": float(Se), "Sp": float(Sp),
        "icbhi_score_official": float(icbhi_official),
        "icbhi_score": float(icbhi_macro),
        "per_class": {CLASSES[i]: {"precision": float(p_c[i]), "recall": float(r_c[i]),
                                   "f1": float(f_c[i]), "support": int(sup[i])} for i in range(4)},
        "confusion_matrix_raw": cm.astype(int).tolist(),
        "confusion_matrix_normalized": (cm / cm.sum(1, keepdims=True).clip(min=1)).round(4).tolist(),
    }
""".strip("\n")))

CELLS.append(md("## Section 8 — Checkpoint utilities (§11)"))

CELLS.append(code(r"""
def save_checkpoint(path, epoch, best_score, history, is_best=False):
    state = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "cfg": {k: v for k, v in CFG.items() if isinstance(v, (int, float, str, bool))},
    }
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(CFG["ckpt_dir"], "best_model.pth"))

def _discover_prior_ckpt():
    local = sorted(glob.glob(os.path.join(CFG["ckpt_dir"], "epoch_*.pth")))
    if local:
        return local[-1]
    for base in glob.glob("/kaggle/input/*"):
        hits = sorted(glob.glob(os.path.join(base, "**", "epoch_*.pth"), recursive=True))
        if hits:
            return hits[-1]
    return None

def load_checkpoint():
    p = _discover_prior_ckpt()
    if not p:
        print("no prior checkpoint — fresh start")
        return 0, -1.0, []
    st = torch.load(p, map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model_state"])
    try:
        optimizer.load_state_dict(st["optimizer_state"])
        scheduler.load_state_dict(st["scheduler_state"])
        scaler.load_state_dict(st["scaler_state"])
    except Exception as e:
        print("optimizer/scheduler state not restored:", e)
    print(f"resumed from {os.path.basename(p)} @ epoch {st['epoch']} (best {st['best_score']:.4f})")
    return st["epoch"], st["best_score"], st.get("history", [])
""".strip("\n")))

CELLS.append(md("## Section 9 — Training loop (auto-resume, monitors held-out val)"))

CELLS.append(code(r"""
from tqdm.auto import tqdm

MON = CFG["monitor"]
MON_MIN = (MON == "val_loss")
def _better(a, b):
    return a < b if MON_MIN else a > b

start_epoch, best_score, history = load_checkpoint()
if not history:
    best_score = float("inf") if MON_MIN else -1.0
if CFG["eval_only"]:
    start_epoch = CFG["num_epochs"]
    print("eval_only — skipping training")

bad_epochs = 0
epoch_times = []
for epoch in range(start_epoch, CFG["num_epochs"]):
    model.train()
    t0 = time.time()
    tr_loss, tr_correct, tr_n = 0.0, 0, 0
    tr_y, tr_p = [], []
    for x, y in tqdm(train_dl, desc=f"train {epoch+1}/{CFG['num_epochs']}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=CFG["use_amp"]):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        tr_loss += loss.item() * len(y)
        pred = out.argmax(1)
        tr_correct += (pred == y).sum().item(); tr_n += len(y)
        tr_y.append(y.cpu().numpy()); tr_p.append(pred.detach().cpu().numpy())
    scheduler.step()

    from sklearn.metrics import f1_score
    tr_y = np.concatenate(tr_y); tr_p = np.concatenate(tr_p)
    tr_f1 = f1_score(tr_y, tr_p, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    va = evaluate(model, val_dl)          # held-out val patients — NOT test
    dt = time.time() - t0
    epoch_times.append(dt)

    rec = {"epoch": epoch + 1,
           "train_loss": tr_loss / tr_n, "val_loss": va["loss"],
           "train_accuracy": tr_correct / tr_n, "val_accuracy": va["accuracy"],
           "train_f1_macro": float(tr_f1), "val_f1_macro": va["f1_macro"],
           "val_icbhi_score_official": va["icbhi_score_official"],
           "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(dt, 1)}
    history.append(rec)
    print(f"E{epoch+1:02d} | trL {rec['train_loss']:.3f} vaL {rec['val_loss']:.3f} | "
          f"vaAcc {va['accuracy']:.3f} vaF1 {va['f1_macro']:.3f} | "
          f"Se {va['Se']:.3f} Sp {va['Sp']:.3f} ICBHI* {va['icbhi_score_official']:.4f} | {dt:.0f}s")

    cur = va["loss"] if MON_MIN else va[MON]
    improved = _better(cur, best_score)
    if improved:
        best_score = cur
        bad_epochs = 0
    else:
        bad_epochs += 1
    save_checkpoint(os.path.join(CFG["ckpt_dir"], f"epoch_{epoch+1:03d}.pth"),
                    epoch + 1, best_score, history, is_best=improved)
    for old in sorted(glob.glob(os.path.join(CFG["ckpt_dir"], "epoch_*.pth")))[:-3]:
        os.remove(old)
    if epoch + 1 >= CFG["min_epochs"] and bad_epochs >= CFG["early_stop_patience"]:
        print(f"early stop — no val {MON} gain in {bad_epochs} epochs")
        break

print(f"\nbest val {MON} = {best_score:.4f}")
""".strip("\n")))

CELLS.append(md("## Section 10 — Final evaluation, efficiency, inference latency"))

CELLS.append(code(r"""
best_path = os.path.join(CFG["ckpt_dir"], "best_model.pth")
st = torch.load(best_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(st["model_state"])
BEST_EPOCH = int(st["epoch"])
history = history or st.get("history", [])   # keep the full loop history for the curves

FINAL = evaluate(model, test_dl)             # <-- the only time TEST is touched
print(f"TEST  acc {FINAL['accuracy']:.4f} | F1 {FINAL['f1_macro']:.4f} | "
      f"Se {FINAL['Se']:.4f} Sp {FINAL['Sp']:.4f} | "
      f"icbhi_score_official {FINAL['icbhi_score_official']:.4f}  (macro {FINAL['icbhi_score']:.4f})")
for c, m in FINAL["per_class"].items():
    print(f"  {c:<8} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  n={m['support']}")

# model size on disk
with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
    torch.save(model.state_dict(), tmp.name)
    SIZE_MB = round(os.path.getsize(tmp.name) / (1024 * 1024), 2)
os.unlink(tmp.name)

# inference latency (ms/sample), batch size 1
model.eval()
_x = torch.randn(1, 3, CFG["img_size"], CFG["img_size"], device=DEVICE)
with torch.no_grad():
    for _ in range(10):
        model(_x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    _t = time.time()
    for _ in range(50):
        model(_x)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
INF_MS = round((time.time() - _t) / 50 * 1000, 2)

TOT_TRAIN_S = float(sum(r["epoch_time_s"] for r in history))
AVG_EPOCH_S = round(TOT_TRAIN_S / max(1, len(history)), 1)
print(f"\nparams {TOTAL_PARAMS/1e6:.2f}M | size {SIZE_MB} MB | "
      f"{AVG_EPOCH_S}s/epoch | total {TOT_TRAIN_S/60:.1f} min | infer {INF_MS} ms/sample")
""".strip("\n")))

CELLS.append(md("## Section 11 — Protocol results JSON (§4 + §4.1)"))

CELLS.append(code(r"""
results = {
    "meta": {
        "model_id": MODEL_ID,
        "model_name": "Swin-T Sound-Event Backbone" + (" + SpecAugment" if CFG["is_augmented"] else ""),
        "contributor": "Farhana",
        "date_completed": datetime.date.today().isoformat(),
        "is_augmented": CFG["is_augmented"],
        "augmentation_method": CFG["augmentation_method"],
        "notes": ("Supersedes/absorbs M23. Pre-trained transformer for RTK_requirements.md req 2+3. "
                  "Split: official_split.py mode=patient_independent (corrected official 60/40). "
                  "Transformers are expected to underperform the M2 CNN on 920 recordings; "
                  "reported as the accuracy/compute trade-off, not tuned to win."),
    },
    "config": {
        "sample_rate": CFG["sample_rate"], "n_mels": CFG["n_mels"], "n_fft": CFG["n_fft"],
        "hop_length": CFG["hop_length"], "win_length": CFG["win_length"],
        "f_min": CFG["f_min"], "f_max": CFG["f_max"],
        "batch_size": CFG["batch_size"], "num_epochs": CFG["num_epochs"],
        "backbone_lr": CFG["backbone_lr"], "head_lr": CFG["head_lr"],
        "weight_decay": CFG["weight_decay"], "optimizer": "AdamW", "scheduler": "cosine+warmup",
        "architecture": CFG["model_name"], "pretrained": "ImageNet-1k",
        "finetune_mode": CFG["finetune_mode"], "drop_path_rate": CFG["drop_path_rate"],
        "head_dropout": CFG["head_dropout"], "label_smoothing": CFG["label_smoothing"],
        "class_weight_power": CFG["class_weight_power"], "monitor": CFG["monitor"],
        "img_size": CFG["img_size"], "seed": CFG["seed"],
        "preprocessing_stages": [
            "load_16k_mono", "butterworth_bandpass_50_2000" if CFG["pp_bandpass"] else "no_bandpass",
            "cycle_segmentation", "duration_8s_cyclic_pad" if CFG["pp_cyclic_pad"] else "zero_pad",
            "peak_amplitude_norm" if CFG["pp_amplitude_norm"] else "no_amp_norm",
            "spectral_gating_denoise" if CFG["pp_denoise"] else "no_denoise",
            "logmel_128_50_2000_db", "fit_only_standardize_clamp3",
            "3ch_imagenet_norm",
            "class_weighted_ce_label_smoothed", "patient_independent_official_60_40_corrected",
        ],
    },
    "environment": {
        "platform": "Kaggle" if os.path.isdir("/kaggle") else ("Colab" if os.path.isdir("/content") else "local"),
        "gpu_name": GPU_NAME, "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    },
    "dataset_info": {
        "dataset": "ICBHI_2017",
        "train_samples": len(fit_rows), "val_samples": len(val_rows), "test_samples": len(test_rows),
        "val_split": "15% of train patients, held out for early-stop/checkpoint selection",
        "split_method": "official_60_40_patient_independent_corrected",
        "split_policy": split_info["policy"],
    },
    "efficiency": {
        "total_params": int(TOTAL_PARAMS), "trainable_params": int(TRAIN_PARAMS),
        "model_size_mb": SIZE_MB,
        "training_time_total_s": round(TOT_TRAIN_S, 1),
        "training_time_per_epoch_s_avg": AVG_EPOCH_S,
        "gpu_name": GPU_NAME, "inference_time_ms_per_sample": INF_MS,
    },
    "best_epoch": {
        "epoch": BEST_EPOCH, "primary_metric": "icbhi_score_official",
        "primary_metric_value": FINAL["icbhi_score_official"],
    },
    "best_metrics": {
        "accuracy": FINAL["accuracy"],
        "precision_macro": FINAL["precision_macro"], "recall_macro": FINAL["recall_macro"],
        "f1_macro": FINAL["f1_macro"], "specificity_macro": FINAL["specificity_macro"],
        "Se": FINAL["Se"], "Sp": FINAL["Sp"],
        "icbhi_score_official": FINAL["icbhi_score_official"],
        "icbhi_score": FINAL["icbhi_score"],
        "per_class": FINAL["per_class"],
        "confusion_matrix_raw": FINAL["confusion_matrix_raw"],
        "confusion_matrix_normalized": FINAL["confusion_matrix_normalized"],
    },
    "ablation": {
        "ablation_group": "backbone_architecture",
        "ablation_role": "variant",
        "baseline_model_id": "M12",
        "variable_changed": f"backbone: Swin-T tiny (ImageNet-1k pretrained, {CFG['finetune_mode']} fine-tune)",
        "variables_held_constant": [
            "loss_function: class_weighted_CrossEntropy(label_smoothing=0.1)", "optimizer: AdamW",
            "data_split: official_60_40_patient_independent_corrected",
            f"augmentation: {'SpecAugment' if CFG['is_augmented'] else 'none'}",
            "seed: 42", "preprocessing: 128mel_16kHz_8s_50-2000Hz",
        ],
        "component_flags": {
            "has_sound_event_head": True, "has_disease_head": False,
            "has_cross_task_consistency": False, "has_cqkd_regularization": False,
            "has_openmax_rejection": False, "owl_stage": 0, "compression_clusters": None,
            "has_concept_bottleneck": False, "bottleneck_type": None,
            "concept_source": None, "has_leakage_measurement": False,
            "has_concept_intervention": False, "concept_space_ood": False, "fm_backbone": "none",
        },
        "loss_weights": {"sound_event_weight": 1.0, "disease_weight": None, "consistency_weight": None},
    },
    "training_history": history,
}

res_path = os.path.join(CFG["results_dir"], f"results_{MODEL_ID}.json")
json.dump(results, open(res_path, "w"), indent=2)
print("wrote", res_path)
""".strip("\n")))

CELLS.append(md("## Section 12 — Plots (§5)"))

CELLS.append(code(r"""
import matplotlib.pyplot as plt
import seaborn as sns

h = history
ep = [r["epoch"] for r in h]
be = BEST_EPOCH

def _mark(ax):
    ax.axvline(be, ls=":", c="grey", lw=1)
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("epoch")

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.plot(ep, [r["train_loss"] for r in h], c="tab:blue", label="train")
ax.plot(ep, [r["val_loss"] for r in h], c="tab:red", label="val")
ax.set_title(f"M41 Swin-T{' +SpecAug' if CFG['is_augmented'] else ''} — Loss"); _mark(ax)
fig.savefig(os.path.join(CFG["results_dir"], f"loss_curve_{MODEL_ID}.png"), bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.plot(ep, [r["train_accuracy"] for r in h], c="tab:blue", label="train")
ax.plot(ep, [r["val_accuracy"] for r in h], c="tab:red", label="val")
ax.set_title("M41 Swin-T — Accuracy"); _mark(ax)
fig.savefig(os.path.join(CFG["results_dir"], f"accuracy_curve_{MODEL_ID}.png"), bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.plot(ep, [r["train_f1_macro"] for r in h], c="tab:blue", label="train macro-F1")
ax.plot(ep, [r["val_f1_macro"] for r in h], c="tab:red", label="val macro-F1")
ax.plot(ep, [r["val_icbhi_score_official"] for r in h], c="tab:green", label="val ICBHI*")
ax.set_title("M41 Swin-T — F1 / ICBHI"); _mark(ax)
fig.savefig(os.path.join(CFG["results_dir"], f"f1_curve_{MODEL_ID}.png"), bbox_inches="tight"); plt.show()

cm = np.array(FINAL["confusion_matrix_raw"])
fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=axs[0])
axs[0].set_title("Confusion (raw)"); axs[0].set_ylabel("true"); axs[0].set_xlabel("pred")
sns.heatmap(cm / cm.sum(1, keepdims=True).clip(min=1), annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axs[1])
axs[1].set_title("Confusion (row-normalized)"); axs[1].set_xlabel("pred")
fig.savefig(os.path.join(CFG["results_dir"], f"confusion_matrix_{MODEL_ID}.png"), bbox_inches="tight"); plt.show()
""".strip("\n")))

CELLS.append(md("## Section 13 — Clean vs Augmented comparison (run after both runs exist)"))

CELLS.append(code(r"""
c_path = os.path.join(CFG["results_dir"], "results_M41.json")
a_path = os.path.join(CFG["results_dir"], "results_M41_aug.json")
if os.path.isfile(c_path) and os.path.isfile(a_path):
    c = json.load(open(c_path)); a = json.load(open(a_path))
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "Se", "Sp", "icbhi_score_official"]
    print(f"{'metric':<22}{'clean':>10}{'+SpecAug':>10}{'Δ':>9}")
    print("-" * 51)
    for k in keys:
        cv, av = c["best_metrics"][k], a["best_metrics"][k]
        print(f"{k:<22}{cv:>10.4f}{av:>10.4f}{av-cv:>+9.4f}")
    print(f"\n{'params (M)':<22}{c['efficiency']['total_params']/1e6:>10.2f}"
          f"{a['efficiency']['total_params']/1e6:>10.2f}")
    print(f"{'size (MB)':<22}{c['efficiency']['model_size_mb']:>10.2f}{a['efficiency']['model_size_mb']:>10.2f}")
    print(f"{'s/epoch':<22}{c['efficiency']['training_time_per_epoch_s_avg']:>10.1f}"
          f"{a['efficiency']['training_time_per_epoch_s_avg']:>10.1f}")
else:
    print("Run the notebook once with is_augmented=False and once with True, then re-run this cell.")
""".strip("\n")))

CELLS.append(md("## Section 14 — Team handoff (§11.D)"))

CELLS.append(code(r"""
import shutil
from IPython.display import display, FileLink

print("=" * 60)
print("OFFICIAL PROTOCOL OUTPUTS READY FOR DOWNLOAD")
print("=" * 60)
protocol_files = sorted(
    glob.glob(os.path.join(CFG["ckpt_dir"], "best_model.pth")) +
    glob.glob(os.path.join(CFG["results_dir"], f"results_{MODEL_ID}.json")) +
    glob.glob(os.path.join(CFG["results_dir"], f"*{MODEL_ID}.png"))
)
for fp in protocol_files:
    if os.path.exists(fp):
        print(f"Ready: {os.path.basename(fp):<32} ({os.path.getsize(fp)/1024/1024:.2f} MB)")
        display(FileLink(fp))

bundle = os.path.join(os.path.dirname(CFG["results_dir"]), f"handoff_{MODEL_ID}")
if protocol_files:
    os.makedirs(bundle, exist_ok=True)
    for fp in protocol_files:
        if os.path.exists(fp):
            shutil.copy2(fp, os.path.join(bundle, os.path.basename(fp)))
    zp = shutil.make_archive(bundle, "zip", bundle)
    print(f"\nZIP bundle ({os.path.getsize(zp)/1024/1024:.2f} MB):")
    display(FileLink(zp))
print("=" * 60)
""".strip("\n")))

CELLS.append(md(r"""
## Section 15 — Summary & report notes

Fill in after both runs:

- **Clean:** `icbhi_score_official` = ___ (Se ___ / Sp ___), macro-F1 ___, ___ M params, ___ MB, ___ s/epoch, ___ ms/sample.
- **+SpecAugment:** `icbhi_score_official` = ___ (Δ ___ vs clean).
- **Vs M2 (CNN, 0.6138 official):** Swin-T is ___x the params / ___x the latency for ___ the score.

**For the report.** M41 is the *pre-trained transformer* row of the backbone table and the
*transformer* half of the augmentation table (`RTK_requirements.md` §5, §7). Present it with the
**split column** = `official_60_40_corrected`, alongside M40/M42/M43. Frame any gap to M2 as the
accuracy/compute trade-off — the paper's contribution is the corrected evaluation protocol, not a
SOTA number.

**Before calling it done** (`Model_Training_Protocol.md` §10): run `python "Asif's/audit/audit_project.py"`
and confirm M41 has no `synthetic_data_not_real_dataset` or split-fallback flag. Hand
`results_M41.json` + `results_M41_aug.json` to Sami for the T2 master table.
"""))

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(NB, fh, indent=1)
print("wrote", OUT, "-", len(CELLS), "cells")
