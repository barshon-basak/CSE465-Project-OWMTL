#!/usr/bin/env python3
"""
Make M2 / M3 / M22 fully self-sufficient on Colab.

    python3 patch_data_setup.py [--dry-run]

Before: M2 and M3 printed manual instructions ("upload your kaggle.json, then run these
commands") and set a fallback path that did not exist; M22 shipped
`PASTE_YOUR_KAGGLE_API_KEY_HERE`. All three additionally require
`ICBHI_challenge_train_test.txt`, which was another manual upload. Four manual steps, repeated
every session.

After: one cell that
  1. resolves Kaggle credentials from Colab Secrets -> env -> ~/.kaggle/kaggle.json,
  2. finds the ICBHI audio or downloads it,
  3. materialises the official split file from an EMBEDDED copy (3.5 KB gzip+base64) so it never
     needs uploading,
  4. sets DATA_ROOT and SPLIT_FILE, and fails with precise instructions if it truly cannot.

Idempotent — files already carrying the marker are skipped.
"""
import argparse
import base64
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_DATA_SETUP_V1"
SPLIT_SRC = os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt")

with open(SPLIT_SRC, "rb") as f:
    SPLIT_B64 = base64.b64encode(gzip.compress(f.read(), 9)).decode()

CELL = '''# ============================================================
# DATA SETUP — credentials, dataset, official split   [''' + MARKER + ''']
# ============================================================
# Fully automatic. Nothing to paste, nothing to upload.
#
# ONE-TIME (Colab): left sidebar -> key icon (Secrets) -> add
#     KAGGLE_USERNAME   your kaggle username
#     KAGGLE_KEY        kaggle.com -> Settings -> API -> Create New Token
# toggle "Notebook access" ON for both. After that every notebook here just works.
#
# The key is never hardcoded: these files are git-tracked and a Kaggle key is full account access.
# The official split file is EMBEDDED below (3.5 KB), so it never needs uploading either.
import base64 as _b64, glob as _glob, gzip as _gzip, os as _os, subprocess as _sp, sys as _sys

_SPLIT_B64 = "''' + SPLIT_B64 + '''"


def _creds():
    try:
        from google.colab import userdata
        u, k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if u and k:
            _os.environ["KAGGLE_USERNAME"], _os.environ["KAGGLE_KEY"] = u, k
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


def _find_audio():
    for root in ("/content", "/kaggle/input", "./data", "."):
        if not _os.path.isdir(root):
            continue
        for d in sorted(_glob.glob(_os.path.join(root, "**", "audio_and_txt_files"),
                                   recursive=True)):
            if _glob.glob(_os.path.join(d, "*.wav")):
                return d
    return None


def _find_split():
    for root in ("/content", "/kaggle/input", ".", _os.path.expanduser("~")):
        if not _os.path.isdir(root):
            continue
        for pat in ("**/ICBHI_challenge_train_test.txt", "**/*train_test*.txt"):
            for p in sorted(_glob.glob(_os.path.join(root, pat), recursive=True)):
                try:
                    rows = [l.split() for l in open(p) if len(l.split()) >= 2]
                    if len([r for r in rows if r[1].lower() in ("train", "test")]) == 920:
                        return p
                except Exception:
                    continue
    return None


print("=" * 70)
print("DATA SETUP")
print("=" * 70)

# ---- 1. audio -------------------------------------------------------------------
DATA_ROOT = _find_audio()
if DATA_ROOT:
    print(f"ICBHI audio : found ({len(_glob.glob(_os.path.join(DATA_ROOT, '*.wav')))} wav)")
else:
    src = _creds()
    print(f"ICBHI audio : not present -> downloading  (credentials: {src or 'NONE'})")
    if not src:
        raise RuntimeError(
            "ICBHI audio is missing and no Kaggle credentials were found.\\n"
            "Set them ONCE: Colab left sidebar -> key icon (Secrets) -> add\\n"
            "  KAGGLE_USERNAME  and  KAGGLE_KEY   (kaggle.com -> Settings -> API -> New Token)\\n"
            "Enable 'Notebook access' for both, then re-run this cell. You will never be asked "
            "again.")
    _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    _dest = "/content" if _os.path.isdir("/content") else "./data"
    _os.makedirs(_dest, exist_ok=True)
    _sp.check_call(["kaggle", "datasets", "download", "-d",
                    "vbookshelf/respiratory-sound-database", "-p", _dest, "--unzip"])
    DATA_ROOT = _find_audio()
    if not DATA_ROOT:
        raise RuntimeError(f"Download finished but no audio_and_txt_files found under {_dest}")
    print(f"ICBHI audio : ready ({len(_glob.glob(_os.path.join(DATA_ROOT, '*.wav')))} wav)")

# ---- 2. official split ----------------------------------------------------------
SPLIT_FILE = _find_split()
if SPLIT_FILE:
    print(f"Split file  : found -> {SPLIT_FILE}")
else:
    SPLIT_FILE = _os.path.join(_os.path.dirname(DATA_ROOT.rstrip("/")),
                               "ICBHI_challenge_train_test.txt")
    try:
        with open(SPLIT_FILE, "wb") as _f:
            _f.write(_gzip.decompress(_b64.b64decode(_SPLIT_B64)))
    except Exception:
        SPLIT_FILE = "/content/ICBHI_challenge_train_test.txt" if _os.path.isdir("/content") \\
            else "./ICBHI_challenge_train_test.txt"
        with open(SPLIT_FILE, "wb") as _f:
            _f.write(_gzip.decompress(_b64.b64decode(_SPLIT_B64)))
    print(f"Split file  : written from embedded copy -> {SPLIT_FILE}")

_rows = [l.split() for l in open(SPLIT_FILE) if len(l.split()) >= 2]
_tr = sum(1 for r in _rows if r[1].lower() == "train")
_te = sum(1 for r in _rows if r[1].lower() == "test")
assert (len(_rows), _tr, _te) == (920, 539, 381), \\
    f"split file is not the official one: {len(_rows)} recs, {_tr}/{_te}"
print(f"            verified: 920 recordings, {_tr} train / {_te} test")
print("=" * 70)
'''


def find_target_cell(nb):
    """The cell that resolves DATA_ROOT."""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "DATA_ROOT" in s and ("POSSIBLE_ROOTS" in s or "kaggle" in s.lower()):
            return i
    return None


def strip_old(src):
    """Remove the legacy DATA_ROOT resolution / manual-instruction / placeholder blocks."""
    src = re.sub(r"POSSIBLE_ROOTS\s*=\s*\[.*?\]\s*\n", "", src, flags=re.S)
    src = re.sub(r"DATA_ROOT\s*=\s*next\(\(p for p in POSSIBLE_ROOTS[^\n]*\n", "", src)
    src = re.sub(r"if DATA_ROOT is None:\n(?:[ \t]+[^\n]*\n|\s*\n)+?else:\n[ \t]+print\([^\n]*\n",
                 "", src)
    src = re.sub(r"os\.environ\['KAGGLE_USERNAME'\][^\n]*\n"
                 r"os\.environ\['KAGGLE_KEY'\][^\n]*\n(?:#[^\n]*\n|\s*\n)*", "", src)
    src = re.sub(r"if os\.environ\.get\('KAGGLE_KEY'[^\n]*:\n(?:[ \t]+[^\n]*\n)+", "", src)
    src = re.sub(r"_DATASETS\s*=\s*\[.*?\]\s*\n", "", src, flags=re.S)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=["M2", "M3", "M22"])
    a = ap.parse_args()

    paths = {
        "M2": "Asif's/M2/M2_cnn_baseline_tuned.ipynb",
        "M3": "Asif's/M3/M3_lightweight_backbone.ipynb",
        "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb",
    }
    for mid in a.only:
        p = os.path.join(REPO, paths[mid])
        nb = json.load(open(p))
        if MARKER in json.dumps(nb):
            print(f"[{mid}] already patched — skipped")
            continue
        idx = find_target_cell(nb)
        if idx is None:
            print(f"[{mid}] could not find the DATA_ROOT cell — skipped")
            continue

        old = "".join(nb["cells"][idx]["source"])
        rest = strip_old(old)
        new = CELL + "\n\n" + rest.lstrip("\n")
        compile(new, f"{mid}_data", "exec")

        if a.dry_run:
            print(f"[{mid}] would patch cell {idx} "
                  f"({len(old.splitlines())} -> {len(new.splitlines())} lines)")
            continue
        nb["cells"][idx]["source"] = new.splitlines(keepends=True)
        json.dump(nb, open(p, "w"), indent=1)
        print(f"[{mid}] patched cell {idx}")


if __name__ == "__main__":
    main()
