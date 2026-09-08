#!/usr/bin/env python3
"""
Make the v2 notebooks run on Colab with no manual setup.

    python3 "Asif's/engine/patch_colab_v2.py" [--dry-run] [--only M22_v2 M3_v2 M49]

WHAT IT FIXES
-------------
`patch_data_setup.py` / `patch_drive_setup.py` made M2 / M3 / M22 self-sufficient on Colab. The
v2 re-runs and the cross-dataset notebook were written afterwards and never got either patch, so
they still carry the Kaggle-only `CELL 3 — LOCATE DATA`. On Colab that raises:

    FileNotFoundError: ICBHI audio not found. On Kaggle: Add Data -> ...

and even when the data is present, every checkpoint is written to ephemeral `/content` and lost on
disconnect -- which for a 40-epoch run is about two hours.

WHAT IT DOES, per notebook
--------------------------
1. Replaces the data-locating cell with the shared DATA SETUP cell: Kaggle credentials from Colab
   Secrets, the ICBHI download automated, the official split materialised from an embedded copy.
2. Inserts the shared OUTPUT SETUP cell straight after it: mounts Drive, probes that the target is
   genuinely writable, and rewires CFG's ckpt/results/out dirs onto it.

Order matters. Output setup runs *after* data setup so the audio search never has to walk a
mounted Drive.

Both cells come from `colab_v2_cells.py` -- one source of truth, shared with anything else that
needs them. Idempotent: a notebook already carrying the markers is skipped.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from colab_v2_cells import (DATA_MARKER, DRIVE_MARKER,          # noqa: E402
                            data_setup_cell, drive_output_cell)

NOTEBOOKS = {
    "M22_v2": "Asif's/M22_v2/m22-official-notebook.ipynb",
    "M3_v2":  "Asif's/M3_v2/M3_v2_clean_notebook.ipynb",
    "M49":    "Asif's/M49_cross_dataset/M49_SPRSound_transfer.ipynb",
}


def find_data_cell(nb):
    """The cell that resolves DATA_ROOT / ICBHI_AUDIO_DIR."""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "LOCATE DATA" in s:
            return i, "replace"
        if "ICBHI_AUDIO_DIR" in s and "_find_first" in s:
            return i, "prepend"          # M49: keep the cell, feed it DATA_ROOT
    return None, None


def find_cfg_cell(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and re.search(r"^CFG\s*=\s*\{", "".join(c["source"]), re.M):
            return i
    return None


def _mkcell(src):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": src.splitlines(keepends=True)}


def patch(path, dry_run=False):
    nb = json.load(open(path))
    blob = json.dumps(nb)
    if DATA_MARKER in blob and DRIVE_MARKER in blob:
        return "already patched — skipped"

    d_idx, mode = find_data_cell(nb)
    if d_idx is None:
        return "could not find the data-locating cell — skipped"
    cfg_idx = find_cfg_cell(nb)
    if cfg_idx is None:
        return "could not find the CFG cell — skipped"
    if cfg_idx > d_idx:
        return f"CFG (cell {cfg_idx}) comes after the data cell ({d_idx}) — skipped"

    data_src, drive_src = data_setup_cell(), drive_output_cell()
    compile(data_src, "data_setup", "exec")
    compile(drive_src, "drive_output", "exec")

    note = []
    if mode == "replace":
        # The old cell's whole contract is DATA_ROOT + SPLIT_FILE, which the new cell also
        # provides, so a straight replacement is safe and leaves nothing dead behind.
        old_lines = len("".join(nb["cells"][d_idx]["source"]).splitlines())
        nb["cells"][d_idx] = _mkcell(data_src)
        note.append(f"replaced data cell {d_idx} ({old_lines} -> {len(data_src.splitlines())} lines)")
        insert_at = d_idx + 1
    else:
        # M49 keeps its own discovery as a fallback; DATA_ROOT/SPLIT_FILE win when present.
        src = "".join(nb["cells"][d_idx]["source"])
        for var, gname in (("ICBHI_AUDIO_DIR", "DATA_ROOT"), ("ICBHI_SPLIT_FILE", "SPLIT_FILE")):
            new, n = re.subn(rf"^{var} = _find_first\(\[",
                             f'{var} = globals().get("{gname}") or _find_first([',
                             src, count=1, flags=re.M)
            if n != 1:
                return f"could not rewire {var} — skipped (notebook changed?)"
            src = new
        nb["cells"][d_idx]["source"] = src.splitlines(keepends=True)
        compile(src, "m49_gate", "exec")
        nb["cells"].insert(d_idx, _mkcell(data_src))
        note.append(f"inserted data cell at {d_idx}, rewired the gate to prefer DATA_ROOT")
        # Between the data cell and the gate, not after it: the gate is the first cell that
        # writes a spectrogram cache, so the redirect has to be in place before it runs.
        insert_at = d_idx + 1

    nb["cells"].insert(insert_at, _mkcell(drive_src))
    note.append(f"inserted output cell at {insert_at}")

    if dry_run:
        return "WOULD: " + "; ".join(note)

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    return "; ".join(note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(NOTEBOOKS))
    a = ap.parse_args()

    for mid in a.only:
        p = os.path.join(REPO, NOTEBOOKS[mid])
        if not os.path.exists(p):
            print(f"[{mid}] MISSING: {p}")
            continue
        print(f"[{mid}] {patch(p, a.dry_run)}")

    if not a.dry_run:
        print("\nDone. Verify with:")
        print('  python3 "Asif\'s/engine/check_undefined_names.py" <notebook>')


if __name__ == "__main__":
    main()
