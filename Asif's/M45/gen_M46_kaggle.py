#!/usr/bin/env python3
"""Generate M46_preprocessing_kaggle.ipynb - rows P1, P2, P3 of the ablation on Kaggle.

WHY A GENERATOR AND NOT A HAND-WRITTEN NOTEBOOK
    `Asif's/CLAUDE.md`: notebooks are generated, not hand-edited. This script embeds
    `m45_ablation.py` verbatim, so the notebook cannot drift from the module that produced
    rows A2-A6, P4 and P5. Regenerate instead of editing the .ipynb.

WHAT THE NOTEBOOK RUNS
    P1  + band-pass filter (50-2000 Hz, 4th-order Butterworth, zero-phase)
    P2  + spectral-gating denoising
    P3  + per-cycle peak amplitude normalisation

    These three stages are named in RTK_requirements.md section 3a and were never
    implemented. They are ADD rows: BASE keeps them off, so A0 and every delta already in
    the paper table stay valid, and each row reports what turning one stage on buys.

    Each row is one 40-epoch MobileNetV2 fine-tune on the corrected official split, plus a
    one-off spectrogram cache build (the cache key includes the three new flags, so no row
    can pick up another row's pixels). Budget roughly 25-40 min per row on a T4.

ON KAGGLE
    Add Data -> `vbookshelf/respiratory-sound-database`, set the accelerator to GPU, Run
    All. The notebook locates the audio and the split file by glob rather than by a
    hard-coded path, because Kaggle's dataset mirrors differ in layout.

OUTPUT TO BRING BACK
    results_M45_P1.json, results_M45_P2.json, results_M45_P3.json
    -> commit them into `Asif's/M45/`, then run `python m45_ablation.py --summarise`
       locally to rebuild M45_ablation_table.json with all twelve rows.

    python "Asif's/M45/gen_M46_kaggle.py"
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "m45_ablation.py")
OUT = os.path.join(HERE, "M46_preprocessing_kaggle.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def build():
    module_src = io.open(MODULE, encoding="utf-8").read()

    cells = [
        md("""# M46 --- preprocessing ablation rows P1, P2, P3

Closes the last open item in RTK requirement 1: the band-pass filter, denoising and
per-cycle amplitude normalisation named in the protocol but never implemented.

**These are ADD rows.** The baseline A0 (M22_v2, official ICBHI 0.5602) does *not* include
these stages, and it must not change --- every delta in the ablation table and in the paper
is measured against it. So each row turns exactly one stage **on** and reports what it
buys. A positive delta means adopt the stage; a negative one means its absence was a
reasonable design, not an oversight.

**Before running:** Add Data -> `vbookshelf/respiratory-sound-database`, and set the
accelerator to **GPU T4**. Then Run All.

Bring back `results_M45_P1.json`, `results_M45_P2.json`, `results_M45_P3.json`.
"""),
        code("""!pip -q install librosa==0.10.2 soundfile
import os, glob, sys
print(sys.version)
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""),
        md("""## 1. Locate the dataset

By glob, not by a hard-coded path --- Kaggle's mirrors of this dataset differ in layout,
and a wrong path that silently resolves to an empty directory is the failure mode this
project has already been bitten by."""),
        code("""def find_one(pattern, what):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(
            f"could not find {what} with {pattern!r}. Add the dataset "
            "'vbookshelf/respiratory-sound-database' via Add Data.")
    return sorted(hits)[0]

AUDIO_DIR  = os.path.dirname(find_one("/kaggle/input/**/audio_and_txt_files/*.wav", "the audio"))
SPLIT_FILE = find_one("/kaggle/input/**/ICBHI_challenge_train_test.txt", "the official split")
WORK = "/kaggle/working/M45"
os.makedirs(WORK, exist_ok=True)

print("audio :", AUDIO_DIR)
print("split :", SPLIT_FILE)
print("wavs  :", len(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))))
"""),
        md("""## 2. The ablation module

Embedded verbatim from `Asif's/M45/m45_ablation.py`, the same module that produced rows
A2--A6, P4 and P5. Nothing is re-implemented here, so the three new rows are trained and
scored by exactly the code that produced the rows they will be compared against."""),
        code("MODULE_SRC = r'''" + module_src.replace("'''", "\\'\\'\\'") + "'''\n"
             "\nwith open('/kaggle/working/m45_ablation.py', 'w', encoding='utf-8') as fh:\n"
             "    fh.write(MODULE_SRC)\n"
             "print('wrote m45_ablation.py', len(MODULE_SRC), 'chars')\n"),
        md("""## 3. Self-test the three new stages

Runs before any training. A preprocessing stage that silently does nothing would produce a
delta near zero and be written up as *"the stage does not help"* --- the wrong conclusion
from a no-op. This checks the band-pass keeps 500 Hz and rejects 6 kHz, the normaliser
reaches unit peak, the gate removes a stationary floor while preserving a transient, and
that the five cache keys are distinct."""),
        code("""import sys
sys.path.insert(0, "/kaggle/working")
import m45_ablation as M

M.HERE = WORK          # write results and cache into /kaggle/working
os.makedirs(os.path.join(WORK, "cache"), exist_ok=True)
assert M.selftest() == 0, "preprocessing self-test failed - do not run the rows"
"""),
        md("""## 4. Run P1, P2, P3

One 40-epoch fine-tune each, plus a one-off cache build per row (the cache key includes
the new flags, so no row can pick up another's spectrograms). Roughly 25--40 min per row on
a T4. Each row writes its own results JSON as soon as it finishes, so a timeout part-way
through still leaves you the completed rows."""),
        code("""import argparse, time

rows = M.corrected_split_index(AUDIO_DIR, SPLIT_FILE)
print("test cycles:", sum(1 for r in rows if r["split"] == "test"))

args = argparse.Namespace(audio_dir=AUDIO_DIR, split_file=SPLIT_FILE)
for rid in ["P1", "P2", "P3"]:
    t0 = time.time()
    print(f"\\n{'='*70}\\n  {rid} - {M.ROWS[rid]['_desc']}\\n{'='*70}")
    M.run_row(rid, rows, args)
    print(f"  {rid} done in {(time.time()-t0)/60:.1f} min")
"""),
        md("""## 5. Collect

Download the three JSONs from the output pane, commit them into `Asif's/M45/`, then run
locally:

```
python "Asif's/M45/m45_ablation.py" --summarise
```

which rebuilds `M45_ablation_table.json` with all twelve rows. The paper's ablation table
(rows P1--P3, currently marked *not implemented*) is then filled from it."""),
        code("""import json

for f in sorted(glob.glob(os.path.join(WORK, "results_M45_P*.json"))):
    d = json.load(open(f))
    b = d["best_metrics"]
    print(f"{d['meta']['row']}  {d['meta']['variable_changed']:<46} "
          f"official {b['icbhi_score_official']:.4f}  Se {b['icbhi_se_official']:.3f}  "
          f"Sp {b['icbhi_sp_official']:.3f}  F1 {b['f1_macro']:.3f}")
print("\\nBaseline A0 (M22_v2) = 0.5602 - a POSITIVE delta here means adopt the stage.")
"""),
    ]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}

    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
    print(f"  {len(cells)} cells, module embedded ({len(module_src)} chars)")


if __name__ == "__main__":
    build()
