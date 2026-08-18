#!/usr/bin/env python3
"""
Patch M2 / M3 / M22 to use the robust output-directory setup.

    python3 patch_drive_setup.py [--dry-run]

Replaces the platform-detection + Drive-mount + BASE_DIR block in each notebook's CELL 0 with a
spliced copy of `drive_setup.py`, so all three share one implementation. Idempotent: a notebook
already carrying the marker is skipped.

Preserves the variables downstream cells expect: BASE_DIR, CKPT_DIR, RESULTS_DIR, CACHE_DIR,
IN_COLAB, DRIVE_MOUNTED.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "OWMTL_ROBUST_OUTPUT_SETUP_V1"

NOTEBOOKS = {
    "M2": "../M2/M2_cnn_baseline_tuned.ipynb",
    "M3": "../M3/M3_lightweight_backbone.ipynb",
    "M22": "../M22/M22_mobilenet_specaugment.ipynb",
}

with open(os.path.join(HERE, "drive_setup.py")) as f:
    MODULE_SRC = f.read().split('if __name__ == "__main__":')[0].rstrip()
# strip the module docstring; the cell carries its own explanation
MODULE_SRC = MODULE_SRC.split('"""', 2)[-1].lstrip("\n")


def build_cell(model_id):
    return f'''# ============================================================
# CELL 0 — ENVIRONMENT & OUTPUT SETUP    [{MARKER}]
# ============================================================
# Spliced from Asif's/engine/drive_setup.py -- do not hand-edit. Re-run
# `python3 Asif's/engine/patch_drive_setup.py` to refresh all three notebooks.
#
# Replaces the old block, which did:
#     try:  drive.mount(..., force_remount=False); DRIVE_MOUNTED = True
#     except Exception as e:  print("Falling back to ephemeral /content storage.")
#
# Two silent failures that caused:
#   1. Re-mounting an ALREADY-MOUNTED Drive raises "Mountpoint must not already contain
#      files". The except swallowed it, DRIVE_MOUNTED stayed False, and output quietly went
#      to ephemeral /content -- the run "saved" and then vanished on disconnect.
#   2. A mounted-but-unwritable Drive (quota / stale FUSE) passes makedirs() and only fails
#      hours into training.
#
# This version mounts robustly, PROBES that the directory is genuinely writable, and RAISES
# if not -- it never silently downgrades to ephemeral storage.
#
# It also handles an existing OWMTL/{model_id} folder (RUN_MODE below):
#   "auto"        completed run -> new version ({model_id}_v2, _v3, ...), preserving old results
#                 interrupted run (checkpoints, no results) -> reuse it so auto-resume works
#   "new_version" always a fresh versioned folder
#   "reuse"       use the folder as-is
#   "overwrite"   delete its contents first  (destructive)
import os, sys, platform

RUN_MODE = "auto"     # <-- change to "reuse" / "new_version" / "overwrite" if needed

print("=" * 70)
print("ENVIRONMENT VERIFICATION & PLATFORM DETECTION")
print("=" * 70)
print(f"Python       : {{sys.version.split()[0]}}  ({{platform.platform()}})")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"PyTorch      : {{torch.__version__}}")
    print(f"CUDA avail   : {{cuda_ok}}")
    if cuda_ok:
        print(f"GPU device   : {{torch.cuda.get_device_name(0)}}")
        print(f"GPU memory   : {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB")
    else:
        print("WARNING: No GPU detected -- training will be very slow on CPU.")
        print("         Colab: Runtime > Change runtime type > T4 GPU")
except ImportError:
    print("ERROR: PyTorch not installed.")

import shutil

{MODULE_SRC}

# ---------------------------------------------------------------- resolve
BASE_DIR, _OUT_INFO = setup_output_dir("{model_id}", mode=RUN_MODE)
CKPT_DIR = _OUT_INFO["ckpt_dir"]
RESULTS_DIR = _OUT_INFO["results_dir"]
CACHE_DIR = _OUT_INFO["cache_dir"]
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
DRIVE_MOUNTED = _OUT_INFO["drive_mounted"]

if not _OUT_INFO["is_persistent"]:
    print("\\nRefusing to fail silently: outputs are EPHEMERAL. Mount Drive and re-run this")
    print("cell before starting a long training run, or accept that results vanish on")
    print("disconnect. (Set RUN_MODE and re-run this cell only -- nothing else changes.)")
'''


def find_cell0(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "ENVIRONMENT VERIFICATION" in s or ("BASE_DIR" in s and "drive" in s.lower()):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for model_id, rel in NOTEBOOKS.items():
        path = os.path.normpath(os.path.join(HERE, rel))
        if not os.path.exists(path):
            print(f"[{model_id}] MISSING: {path}")
            continue
        nb = json.load(open(path))
        idx = find_cell0(nb)
        if idx is None:
            print(f"[{model_id}] could not locate the environment cell -- skipped")
            continue
        existing = "".join(nb["cells"][idx]["source"])
        if MARKER in existing:
            print(f"[{model_id}] already patched (cell {idx}) -- skipped")
            continue

        new_src = build_cell(model_id)
        # sanity: the downstream contract must survive
        for var in ("BASE_DIR", "CKPT_DIR", "RESULTS_DIR", "CACHE_DIR", "IN_COLAB",
                    "DRIVE_MOUNTED"):
            assert f"{var} =" in new_src or f"{var}," in new_src, var
        compile(new_src, f"{model_id}_cell0", "exec")

        if a.dry_run:
            print(f"[{model_id}] would patch cell {idx} "
                  f"({len(existing.splitlines())} -> {len(new_src.splitlines())} lines)")
            continue

        nb["cells"][idx]["source"] = new_src.splitlines(keepends=True)
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"[{model_id}] patched cell {idx}  ({path})")

    if not a.dry_run:
        print("\nDone. All three now share Asif's/engine/drive_setup.py.")


if __name__ == "__main__":
    main()
