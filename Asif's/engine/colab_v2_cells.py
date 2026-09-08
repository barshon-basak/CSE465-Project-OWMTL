#!/usr/bin/env python3
"""
The two Colab bootstrap cells, as one source of truth.

WHY THIS EXISTS
---------------
`patch_data_setup.py` and `patch_drive_setup.py` made M2 / M3 / M22 self-sufficient on Colab:
credentials from Colab Secrets, the ICBHI download automated, the official split embedded, and
output on Drive instead of ephemeral `/content`. The **v2 re-runs** (M22_v2, M3_v2) and the
cross-dataset notebook (M49) were written later and never got either patch, so they still ship
the Kaggle-only `CELL 3 — LOCATE DATA`, which on Colab raises

    FileNotFoundError: ICBHI audio not found. On Kaggle: Add Data -> ...

That is the failure this module fixes. Rather than copy the cell text a third and fourth time,
both cells live here and every patcher splices from this file.

TWO DELIBERATE DIFFERENCES from `patch_data_setup.py`'s original cell
--------------------------------------------------------------------
1. **The audio search never walks Drive.** The original globbed `/content/**` recursively. Once
   Drive is mounted that walks the user's entire Drive, which can take minutes and occasionally
   hangs on a stale FUSE handle. This version checks a short list of exact paths first, then
   globs only non-Drive roots.
2. **Output setup runs AFTER data setup.** Same reason: keep Drive unmounted while the audio
   search runs. The mount prompt still appears inside the first couple of minutes.
"""
import base64
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

DATA_MARKER  = "OWMTL_DATA_SETUP_V2"
DRIVE_MARKER = "OWMTL_DRIVE_OUTPUT_V2"

SPLIT_SRC = os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt")
with open(SPLIT_SRC, "rb") as _f:
    SPLIT_B64 = base64.b64encode(gzip.compress(_f.read(), 9)).decode()

# drive_setup.py is spliced in verbatim, minus its docstring and __main__ guard.
with open(os.path.join(HERE, "drive_setup.py")) as _f:
    _DRIVE_SRC = _f.read().split('if __name__ == "__main__":')[0].rstrip()
_DRIVE_SRC = _DRIVE_SRC.split('"""', 2)[-1].lstrip("\n")


def data_setup_cell():
    """Resolves DATA_ROOT and SPLIT_FILE on Colab, Kaggle or local. Downloads if needed."""
    return '''# ============================================================
# DATA SETUP — credentials, ICBHI audio, official split   [''' + DATA_MARKER + ''']
# ============================================================
# Fully automatic. Nothing to paste, nothing to upload.
#
# ONE-TIME (Colab): left sidebar -> key icon (Secrets) -> add
#     KAGGLE_USERNAME   your kaggle username
#     KAGGLE_KEY        kaggle.com -> Settings -> API -> Create New Token
# Toggle "Notebook access" ON for both. After that every notebook here just works.
#
# The key is never written into the notebook: these files are git-tracked and a Kaggle key is
# full account access. The official split file is EMBEDDED below (3.5 KB gzip+base64), so it
# never needs uploading either.
#
# Sets DATA_ROOT and SPLIT_FILE. Raises with precise instructions if it truly cannot.
import base64 as _b64, glob as _glob, gzip as _gzip, os as _os, subprocess as _sp, sys as _sys

_SPLIT_B64 = "''' + SPLIT_B64 + '''"


def _creds():
    """Kaggle credentials, in the order they should already live."""
    try:
        from google.colab import userdata
        u, k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if u and k:
            _os.environ["KAGGLE_USERNAME"], _os.environ["KAGGLE_KEY"] = u.strip(), k.strip()
            return "Colab Secrets"
    except Exception:
        pass
    if _os.environ.get("KAGGLE_USERNAME") and _os.environ.get("KAGGLE_KEY"):
        return "environment variables"
    import json as _j
    p = _os.path.expanduser("~/.kaggle/kaggle.json")
    if _os.path.isfile(p):
        try:
            d = _j.load(open(p))
            if d.get("username") and d.get("key"):
                _os.environ["KAGGLE_USERNAME"] = d["username"]
                _os.environ["KAGGLE_KEY"] = d["key"]
                return "~/.kaggle/kaggle.json"
        except Exception:
            pass
    return None


def _is_audio_dir(d):
    return bool(d) and _os.path.isdir(d) and len(_glob.glob(_os.path.join(d, "*.wav"))) >= 900


# Exact paths first — cheap, and they cover every layout this project has actually seen.
_EXACT = [
    "/content/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files",
    "/content/Respiratory_Sound_Database/audio_and_txt_files",
    "/content/data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files",
    "/content/audio_and_txt_files",
    "/content/drive/MyDrive/OWMTL_data/audio_and_txt_files",
    "/content/drive/MyDrive/ICBHI/audio_and_txt_files",
    "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/"
    "Respiratory_Sound_Database/audio_and_txt_files",
    "./data/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files",
]


def _find_audio():
    for d in _EXACT:
        if _is_audio_dir(d):
            return d
    # Glob only roots that are NOT Drive. A recursive walk of a mounted Drive can take
    # minutes and sometimes hangs on a stale FUSE handle.
    for root in ("/kaggle/input", "/content", "./data", "."):
        if not _os.path.isdir(root):
            continue
        for d in sorted(_glob.glob(_os.path.join(root, "**", "audio_and_txt_files"),
                                   recursive=True)):
            if "/drive/" in d.replace("\\\\", "/"):
                continue
            if _is_audio_dir(d):
                return d
    return None


def _valid_split(p):
    try:
        rows = [l.split() for l in open(p) if len(l.split()) >= 2]
        return len([r for r in rows if r[1].lower() in ("train", "test")]) == 920
    except Exception:
        return False


def _find_split():
    for root in ("/kaggle/input", "/content", ".", _os.path.expanduser("~")):
        if not _os.path.isdir(root):
            continue
        for pat in ("**/ICBHI_challenge_train_test.txt", "**/*train_test*.txt"):
            for p in sorted(_glob.glob(_os.path.join(root, pat), recursive=True)):
                if "/drive/" in p.replace("\\\\", "/"):
                    continue
                if _valid_split(p):
                    return p
    return None


print("=" * 70)
print("DATA SETUP")
print("=" * 70)

# ---- 1. ICBHI audio --------------------------------------------------------------
DATA_ROOT = _find_audio()
if DATA_ROOT:
    print(f"ICBHI audio : found -> {DATA_ROOT}")
else:
    _src = _creds()
    print(f"ICBHI audio : not present -> downloading  (credentials: {_src or 'NONE'})")
    if not _src:
        raise RuntimeError(
            "ICBHI audio is missing and no Kaggle credentials were found.\\n\\n"
            "Set them ONCE: Colab left sidebar -> key icon (Secrets) -> add\\n"
            "    KAGGLE_USERNAME   your kaggle username\\n"
            "    KAGGLE_KEY        kaggle.com -> Settings -> API -> Create New Token\\n"
            "Turn 'Notebook access' ON for BOTH, then re-run this cell.\\n"
            "You will never be asked again, in this or any other notebook here.")
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    _dest = "/content" if _os.path.isdir("/content") else "./data"
    _os.makedirs(_dest, exist_ok=True)
    try:
        _sp.check_call(["kaggle", "datasets", "download", "-d",
                        "vbookshelf/respiratory-sound-database", "-p", _dest, "--unzip"])
    except _sp.CalledProcessError as e:
        raise RuntimeError(
            f"Kaggle download failed (exit {e.returncode}).\\n"
            "Most common cause: the Kaggle account has not accepted the dataset's terms.\\n"
            "Open https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database "
            "once in a browser while signed in, then re-run this cell.") from e
    DATA_ROOT = _find_audio()
    if not DATA_ROOT:
        raise RuntimeError(f"Download finished but no audio_and_txt_files found under {_dest}")
    print(f"ICBHI audio : ready -> {DATA_ROOT}")

_n_wav = len(_glob.glob(_os.path.join(DATA_ROOT, "*.wav")))
_n_txt = len(_glob.glob(_os.path.join(DATA_ROOT, "*.txt")))
print(f"            {_n_wav} wav / {_n_txt} annotation txt")
assert _n_wav >= 900, f"only {_n_wav} wav files under {DATA_ROOT} — download looks incomplete"

# ---- 2. official split -----------------------------------------------------------
# This notebook REFUSES to fall back to a patient-id rule. That silent fallback gave four
# models an 11-patient test set while labelling itself "official 60/40", and correcting it
# is the whole point of the v2 re-runs (Model_Training_Protocol.md section 1).
SPLIT_FILE = _find_split()
if SPLIT_FILE:
    print(f"Split file  : found -> {SPLIT_FILE}")
else:
    for _cand in (_os.path.join(_os.path.dirname(DATA_ROOT.rstrip("/")),
                                "ICBHI_challenge_train_test.txt"),
                  "/content/ICBHI_challenge_train_test.txt",
                  "./ICBHI_challenge_train_test.txt"):
        try:
            with open(_cand, "wb") as _fh:
                _fh.write(_gzip.decompress(_b64.b64decode(_SPLIT_B64)))
            SPLIT_FILE = _cand
            break
        except Exception:
            continue
    if not SPLIT_FILE:
        raise RuntimeError("could not write the embedded split file anywhere")
    print(f"Split file  : written from embedded copy -> {SPLIT_FILE}")

_rows = [l.split() for l in open(SPLIT_FILE) if len(l.split()) >= 2]
_tr = sum(1 for r in _rows if r[1].lower() == "train")
_te = sum(1 for r in _rows if r[1].lower() == "test")
assert (len(_rows), _tr, _te) == (920, 539, 381), \\
    f"this is not the official split file: {len(_rows)} recordings, {_tr} train / {_te} test"
print(f"            verified: 920 recordings, {_tr} train / {_te} test")
print("=" * 70)
'''


def drive_output_cell():
    """Mounts Drive, resolves a guaranteed-writable output dir, and rewires CFG onto it."""
    return '''# ============================================================
# OUTPUT SETUP — Google Drive, checked not assumed   [''' + DRIVE_MARKER + ''']
# ============================================================
# Spliced from Asif's/engine/drive_setup.py -- do not hand-edit. Re-run
# `python3 "Asif\'s/engine/patch_colab_v2.py"` to refresh.
#
# On Colab this prompts for Drive access the first time. Accept it: without Drive every
# checkpoint and result lives in /content and disappears when the runtime disconnects, which
# on a 40-epoch run means losing about two hours.
#
# The old block in these notebooks did:
#     try:  drive.mount(...); DRIVE_MOUNTED = True
#     except Exception as e:  print("Falling back to ephemeral /content storage.")
# Two silent failures came out of that: re-mounting an already-mounted Drive raises and the
# except swallowed it, and a mounted-but-unwritable Drive passes makedirs() and only fails
# hours into training. This version PROBES that the directory is genuinely writable and
# RAISES if not. It never silently downgrades.
#
# RUN_MODE controls what happens to an existing OWMTL/<id> folder:
#   "auto"        completed run -> new version (<id>_v2, _v3, ...), old results preserved
#                 interrupted run (checkpoints, no results) -> reuse it so auto-resume works
#   "new_version" always a fresh versioned folder
#   "reuse"       use the folder as-is
#   "overwrite"   delete its contents first  (destructive)
import os, sys, shutil

RUN_MODE = "auto"     # <-- "reuse" to resume an interrupted run in place

''' + _DRIVE_SRC + '''

# ---------------------------------------------------------------- resolve
_OWMTL_ID = (CFG.get("model_id") if isinstance(globals().get("CFG"), dict) else None) or "OWMTL"
BASE_DIR, _OUT_INFO = setup_output_dir(_OWMTL_ID, mode=RUN_MODE)
CKPT_DIR    = _OUT_INFO["ckpt_dir"]
RESULTS_DIR = _OUT_INFO["results_dir"]
CACHE_DIR   = _OUT_INFO["cache_dir"]
IN_COLAB      = "google.colab" in sys.modules or os.path.exists("/content")
DRIVE_MOUNTED = _OUT_INFO["drive_mounted"]

# ---------------------------------------------------------------- rewire CFG
# Every downstream write goes through CFG, so redirecting it here is enough -- no other
# cell needs editing. The spectrogram cache deliberately stays on local disk: it is a
# multi-GB disposable memmap and writing it to Drive would be slow and eat the quota.
if isinstance(globals().get("CFG"), dict):
    _redirect = {"ckpt_dir": CKPT_DIR, "results_dir": RESULTS_DIR,
                 "out_dir": RESULTS_DIR, "cache_dir": CACHE_DIR}
    for _k, _v in _redirect.items():
        if _k in CFG:
            CFG[_k] = _v
            os.makedirs(_v, exist_ok=True)
    print("\\nCFG redirected to persistent storage:")
    for _k in ("ckpt_dir", "results_dir", "out_dir", "cache_dir"):
        if _k in CFG:
            print(f"  CFG[{_k!r}] = {CFG[_k]}")

if not _OUT_INFO["is_persistent"]:
    print("\\nRefusing to fail silently: outputs are EPHEMERAL. Mount Drive and re-run this")
    print("cell before starting a long training run, or accept that results vanish on")
    print("disconnect. (Change RUN_MODE and re-run this cell only -- nothing else changes.)")
'''


if __name__ == "__main__":
    for name, src in (("data_setup_cell", data_setup_cell()),
                      ("drive_output_cell", drive_output_cell())):
        compile(src, name, "exec")
        print(f"{name}: {len(src.splitlines())} lines, compiles")
