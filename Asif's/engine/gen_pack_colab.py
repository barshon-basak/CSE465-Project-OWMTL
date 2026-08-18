#!/usr/bin/env python3
"""
Emit a standalone Colab notebook that builds the clinician listening pack.

    python3 gen_pack_colab.py    ->  build_listening_pack_colab.ipynb

Why a notebook: ICBHI lives in the Colab runtime, not on the laptop, so the pack has to be built
where the audio is. `build_listening_pack.py` is spliced in verbatim (single source of truth --
edit the module and regenerate, never hand-edit the notebook), so there is one file to upload and
nothing to paste.

The notebook keeps the pack and the answer key in SEPARATE downloads on purpose. `clip_key.csv`
carries ICBHI's label for every clip; sending it to the clinician would destroy the blinding the
whole exercise depends on.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "build_listening_pack.py")) as f:
    SRC = f.read()
SRC = SRC.split('if __name__ == "__main__":')[0].rstrip()

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


md(r"""
# Build the clinician listening pack

Run this in the **same Colab session as M39**, where ICBHI is already downloaded. If it isn't,
cell 1 will fetch it.

Produces two things, downloaded **separately and deliberately**:

| Download | Goes to | Contains |
|---|---|---|
| `ICBHI_listening_pack.zip` | **the physician** | 132 clips, `labels.xlsx`, reference sounds, instructions |
| `clip_key.csv` | **you only** | which cycle each clip came from + **ICBHI's own label** |

> ### ⚠️ Never send `clip_key.csv`
> It holds the answer for every clip. Sending it destroys the blinding, and a blind rating is the
> only kind that validates anything — if the clinician can see the expected answer, their agreement
> with it measures nothing.

`clip_key.csv` also records ICBHI's label per clip, which is what lets you later measure
**clinician-vs-ICBHI agreement**. That matters: M39's first run gave crackle AUROC 0.55, and there
are two possible reasons — our extractors are weak, or ICBHI's labels are noisy. These ratings
separate those.
""")

code(r'''
# ============================================================
# CELL 1 — FIND (or fetch) ICBHI
# ============================================================
import os, glob, subprocess, sys


def find_audio_dir():
    for root in ("/content", "/kaggle/input", "."):
        if os.path.isdir(root):
            for d in sorted(glob.glob(os.path.join(root, "**", "audio_and_txt_files"),
                                      recursive=True)):
                if glob.glob(os.path.join(d, "*.wav")):
                    return d
    return None


DATA_ROOT = find_audio_dir()

if DATA_ROOT:
    print(f"ICBHI already here: {DATA_ROOT}")
else:
    print("ICBHI not found in this runtime — downloading.")
    os.environ['KAGGLE_USERNAME'] = 'AsifM7'
    os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KAGGLE_API_KEY_HERE'   # <-- replace
    if os.environ['KAGGLE_KEY'] == 'PASTE_YOUR_KAGGLE_API_KEY_HERE':
        raise RuntimeError("Paste your Kaggle key above, or re-run this in the M39 session "
                           "where ICBHI is already downloaded.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    subprocess.check_call(["kaggle", "datasets", "download", "-d",
                           "vbookshelf/respiratory-sound-database", "-p", "/content", "--unzip"])
    DATA_ROOT = find_audio_dir()

assert DATA_ROOT, "Could not locate audio_and_txt_files."
print(f"  {len(glob.glob(os.path.join(DATA_ROOT, '*.wav')))} recordings")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "soundfile", "librosa", "openpyxl"])
print("Dependencies ready.")
''')

md(r"""
---
## The builder

Spliced verbatim from `Asif's/engine/build_listening_pack.py`. **Do not edit this cell** — edit the
module and regenerate with `gen_pack_colab.py`.
""")

code("# ============================================================\n"
     "# CELL 2 — BUILDER (spliced from Asif's/engine/build_listening_pack.py)\n"
     "# ============================================================\n" + SRC)

code(r'''
# ============================================================
# CELL 3 — BUILD THE PACK
# ============================================================
import random

OUT = "/content/ICBHI_listening_pack"
KEY = "/content/clip_key.csv"          # deliberately OUTSIDE the pack folder
SEED = 42

rng = random.Random(SEED)
cycles = collect_cycles(DATA_ROOT)
print(f"Found {len(cycles)} annotated cycles.\n")

chosen = []
for stratum, n in DEFAULT_STRATA.items():
    pool = [c for c in cycles if c["stratum"] == stratum and c["duration"] >= 0.9]
    if len(pool) < n:
        print(f"  WARNING: only {len(pool)} '{stratum}' available, wanted {n}")
        n = len(pool)
    pool.sort(key=lambda c: -c["duration"])
    top = pool[: max(n * 4, n)]
    rng.shuffle(top)
    seen_patients, picked = set(), []
    for c in top:
        pid = c["stem"].split("_")[0]
        if pid in seen_patients and len(picked) < n:
            continue
        picked.append(c); seen_patients.add(pid)
        if len(picked) == n:
            break
    for c in top:
        if len(picked) == n:
            break
        if c not in picked:
            picked.append(c)
    chosen += picked
    print(f"  {stratum:<14} {len(picked)} clips from {len(seen_patients)} patients")

MIN_SEP = 20
playlist = list(chosen); rng.shuffle(playlist)
half = len(playlist) // 2
dup_zone = max(1, min(half - MIN_SEP, len(playlist)))
dups = rng.sample(playlist[:dup_zone], min(N_DUPLICATES, dup_zone))
for d in dups:
    fa = playlist.index(d)
    lo = min(fa + MIN_SEP + 1, len(playlist))
    playlist.insert(rng.randrange(lo, len(playlist) + 1), d)

seen_pos, seps = {}, []
for i, c in enumerate(playlist):
    k = (c["stem"], c["cycle_idx"])
    if k in seen_pos:
        seps.append(i - seen_pos[k])
    else:
        seen_pos[k] = i
assert seps and min(seps) >= MIN_SEP, f"duplicates too close: {min(seps) if seps else 0}"
print(f"\n{len(chosen)} unique + {len(dups)} hidden duplicates = {len(playlist)} clips")
print(f"  duplicate separation: min {min(seps)}, median {sorted(seps)[len(seps)//2]} apart")

clips_dir = os.path.join(OUT, "clips"); refs_dir = os.path.join(OUT, "reference_sounds")
for d in (clips_dir, refs_dir):
    os.makedirs(d, exist_ok=True)

key_rows, label_rows = [], []
for i, c in enumerate(playlist, start=1):
    cid = f"clip_{i:03d}"
    sf.write(os.path.join(clips_dir, cid + ".wav"), load_clip(c, 16000), 16000)
    key_rows.append({"clip_id": cid, "stem": c["stem"], "cycle_idx": c["cycle_idx"],
                     "start": round(c["start"], 4), "end": round(c["end"], 4),
                     "stratum": c["stratum"], "icbhi_crackle": c["icbhi_crackle"],
                     "icbhi_wheeze": c["icbhi_wheeze"],
                     "unit_id": f"{c['stem']}__{c['cycle_idx']}", "is_duplicate_of": ""})
    label_rows.append({"clip_id": cid, "crackles": "", "wheeze": "",
                       "confidence": "", "audio_quality": "", "notes": ""})

seen = {}
for r in key_rows:
    if r["unit_id"] in seen:
        r["is_duplicate_of"] = seen[r["unit_id"]]
    else:
        seen[r["unit_id"]] = r["clip_id"]

with open(os.path.join(OUT, "labels.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(label_rows[0])); w.writeheader(); w.writerows(label_rows)
write_xlsx(os.path.join(OUT, "labels.xlsx"), label_rows)
with open(KEY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0])); w.writeheader(); w.writerows(key_rows)

picked_ids = {(x["stem"], x["cycle_idx"]) for x in chosen}
for name, stratum in {"example_normal": "normal", "example_fine_crackle": "crackle_only",
                      "example_coarse_crackle": "crackle_only", "example_wheeze": "wheeze_only",
                      "example_rhonchi": "wheeze_only", "example_ambiguous": "both"}.items():
    pool = [c for c in cycles
            if c["stratum"] == stratum and (c["stem"], c["cycle_idx"]) not in picked_ids]
    if pool:
        pool.sort(key=lambda c: -c["duration"])
        sf.write(os.path.join(refs_dir, name + ".wav"), load_clip(pool[0], 16000), 16000)

with open(os.path.join(OUT, "START_HERE.md"), "w") as f:
    f.write(START_HERE)

print(f"\nPack: {OUT}")
print(f"  clips/            {len(os.listdir(clips_dir))}")
print(f"  reference_sounds/ {len(os.listdir(refs_dir))}")
print(f"  labels.xlsx + labels.csv + START_HERE.md")
print(f"Key (private):      {KEY}")
''')

code(r'''
# ============================================================
# CELL 4 — SAFETY CHECK, THEN DOWNLOAD
# ============================================================
import zipfile

# The pack must not contain the answers, and no filename may hint at a label.
leaks = [p for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True)
         if os.path.basename(p) in ("clip_key.csv",)]
assert not leaks, f"ANSWER KEY IS INSIDE THE PACK: {leaks}"
bad_names = [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "clips", "*.wav"))
             if not os.path.basename(p).startswith("clip_")]
assert not bad_names, f"filenames may reveal labels: {bad_names[:5]}"
print("[OK] no answer key inside the pack")
print("[OK] all clip filenames are opaque")

ZIP = "/content/ICBHI_listening_pack.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True):
        if os.path.isfile(p):
            z.write(p, os.path.relpath(p, os.path.dirname(OUT)))
print(f"\n{ZIP}  ({round(os.path.getsize(ZIP)/1024**2, 1)} MB)")

try:
    from google.colab import files
    print("\nDownloading the PACK (send this to the physician) ...")
    files.download(ZIP)
    print("Downloading the KEY (keep this yourself — never send it) ...")
    files.download(KEY)
except Exception as e:
    print(f"\nAuto-download unavailable ({e}). Use the file browser on the left:")
    print(f"  send:  {ZIP}")
    print(f"  keep:  {KEY}")
''')

md(r"""
---
### After you send it

1. The physician gets **`ICBHI_listening_pack.zip`** only. They unzip, read `START_HERE.md`, and
   fill `labels.xlsx` — dropdowns are locked so answers can't arrive as `Fine` / `FINE` / `fine `.
2. **Check the 12 hidden duplicates first** when the file comes back. If they disagree with
   themselves on more than ~3 of 12, the task is too hard from recordings and the rest of the file
   is not worth much — that is itself a reportable finding.
3. Then join on `clip_key.csv` and compute both things it enables:
   - **clinician vs. our extractors** — did the DSP measure what the name claims?
   - **clinician vs. ICBHI labels** — are the benchmark's own labels reliable?

   The second is why M39's weak AUROC is not yet interpretable. Use
   `Asif's/Statistics/owmtl_scores.py` for the paired tests and CIs.
""")

NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.13"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
for c in NB["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT = os.path.join(HERE, "build_listening_pack_colab.ipynb")
with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)
print(f"Wrote {OUT}  ({len(NB['cells'])} cells)")
