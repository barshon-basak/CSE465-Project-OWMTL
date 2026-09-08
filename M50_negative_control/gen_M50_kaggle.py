#!/usr/bin/env python3
"""Generate the M50 CirCor negative-control notebook for Kaggle.

Same contract as `M49_cross_dataset/gen_M49_kaggle.py`: notebooks are generated, never
hand-edited, and the three modules are embedded as gzip+base64 so the notebook cannot drift
from the module the self-test covers. Regenerate; do not edit the .ipynb.

    python "M50_negative_control/gen_M50_kaggle.py"
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
M45_SRC = os.path.join(REPO, "Asif's", "M45", "m45_ablation.py")
M49_SRC = os.path.join(REPO, "M49_cross_dataset", "m49_xval.py")
M50_SRC = os.path.join(HERE, "m50_openset.py")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


def embed(path, name):
    """Embed a module as gzip+base64, byte-identical to the repo.

    Base64 has no quotes and no newlines, so the literal cannot be ended by its own contents
    -- the failure that made a triple-quoted embedding execute the module as cell code.
    """
    var = name.upper().replace(".PY", "") + "_B64"
    blob = base64.b64encode(gzip.compress(io.open(path, "rb").read())).decode("ascii")
    q = chr(39) * 3
    wrapped = "\n".join(blob[i:i + 76] for i in range(0, len(blob), 76))
    return (f"{var} = (\n{q}{wrapped}{q})\n\n"
            "import base64, gzip\n"
            f"_src = gzip.decompress(base64.b64decode({var}))\n"
            f"with open('/kaggle/working/{name}', 'wb') as fh:\n"
            "    fh.write(_src)\n"
            f"print('wrote {name}', len(_src), 'bytes')\n")


HEADER = """# M50 --- CirCor DigiScope as a **far-OOD negative control**

> **This is not a transfer test, and it must never be presented as one.**
> CirCor is a phonocardiogram corpus: it holds **heart** sounds, labelled murmur present /
> absent / unknown. A crackle/wheeze classifier has zero label overlap with that, so scoring
> it with the ICBHI metric would produce no generalization number at all. Cross-dataset
> transfer lives in `M49_cross_dataset/`.

**No CirCor label is read here.** Not the murmur annotation, not the segmentation `.tsv`, not
the demographics. The corpus enters as one thing only: audio that is definitionally not a
respiratory cycle. That is what makes this valid rather than a category error.

## The two questions

| | Question | How |
|---|---|---|
| **Rejection** | can the frozen model tell this is not its domain? | post-hoc energy score `-logsumexp(z)`, ICBHI-known vs CirCor-unknown, AUROC with a **patient-level** bootstrap |
| **Abstention** | when it is wrong, does it know? | max-softmax, entropy and the predicted-class histogram on heart sounds, against the same on ICBHI |

M49 found the model **wrong and confident** on SPRSound --- mean max-softmax 0.9435 at 0.2479
accuracy. This is the extreme point of that curve.

## The caveat that travels with the number

CirCor is **far-OOD**: a different organ. The paper's existing open-set arm reports
**AUROC 0.6466** on **near-OOD** unknowns --- held-out lung disease classes, **19 patients**.
Rejecting a heartbeat is an *easier* task than rejecting an unseen lung pathology.

So this number **does not replace, upgrade or supersede 0.6466.** It is a second, easier point
on a difficulty axis, and it earns its place by showing the detector functions at all --- which
n=19 could not. Both numbers get reported with their unknown sets named. The sentence ships
inside the results JSON as `openset.difficulty_note` so it cannot be lost in transcription.

## Before you press Run All

| | |
|---|---|
| **Accelerator** | GPU T4 |
| **Internet** | not needed --- everything is attached |
| **Add Data** | `bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2` |
| **Add Data** | `vbookshelf/respiratory-sound-database` (ICBHI --- the known side *and* the gate) |
| **Add Data** | the checkpoint: `Asif's/M22_v2/Results/best_model.pth` as a private dataset |
| **Runtime** | roughly 15 min |

### Bring back

`results_M50_circor.json`, `logits_M50_*.npy`, `M50_openset.png` --- zipped in the last cell.
Commit them into `M50_negative_control/`.
"""

SETUP = """!pip -q install librosa soundfile
import glob, json, os, sys, time
import numpy as np
print(sys.version)
import torch, librosa
print("torch", torch.__version__, "| librosa", librosa.__version__, "| cuda:",
      torch.cuda.is_available())

WORK = "/kaggle/working/M50"
os.makedirs(WORK, exist_ok=True)

# Set to a row count for a two-minute wiring check; None for the real thing.
SMOKE = None
# Cap the windows taken from any one recording. None keeps them all; the bootstrap groups by
# patient either way, so a long recording cannot dominate the interval.
MAX_WINDOWS = None
"""

FIND = '''def find_one(pattern, what, hint=""):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"could not find {what} with {pattern!r}. {hint}")
    return hits[0]

# The paper's best model. NOT best_model_official.pth (same folder, 0.5641, published split
# verbatim, leaks patients 156 and 218) -- m49_xval.load_checkpoint reads split_method out of
# the cfg and raises on it, so the wrong file cannot be used by accident.
CKPT_NAME = "best_model.pth"
CKPT = find_one(f"/kaggle/input/**/{CKPT_NAME}", "the checkpoint",
                f"Upload Asif's/M22_v2/Results/{CKPT_NAME} as a private Kaggle dataset.")
ICBHI_AUDIO = os.path.dirname(find_one(
    "/kaggle/input/**/audio_and_txt_files/*.wav", "the ICBHI audio",
    "Add Data -> vbookshelf/respiratory-sound-database."))
ICBHI_SPLIT = find_one("/kaggle/input/**/ICBHI_challenge_train_test.txt",
                       "the official ICBHI split file",
                       "It ships with the same dataset. A notebook that cannot find it must "
                       "raise, never fall back (Model_Training_Protocol.md 1.1).")

# CirCor: the Kaggle mirror nests the wavs under training_data/. Anchor on a file pattern that
# only CirCor has, then take its directory -- a hard-coded path that silently resolves to an
# empty directory is a failure mode this project has already been bitten by.
CIRCOR_ROOT = os.path.dirname(find_one(
    "/kaggle/input/**/training_data/*.wav", "the CirCor audio",
    "Add Data -> bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2"))
print("checkpoint :", CKPT)
print("ICBHI audio:", ICBHI_AUDIO, len(glob.glob(ICBHI_AUDIO + "/*.wav")), "wavs")
print("CirCor     :", CIRCOR_ROOT, len(glob.glob(CIRCOR_ROOT + "/*.wav")), "wavs")
'''

MODULES_MD = """## 3. The three modules

`m45_ablation.py` supplies the mel parameters and the corrected-split loader; `m49_xval.py`
supplies the checkpoint loader, the forward pass and the ICBHI gate; `m50_openset.py` adds
the CirCor index and the open-set scorers. Nothing is re-implemented --- a second copy of the
spectrogram code would make every number here a measurement of the gap between two scripts.

All three are carried as gzip+base64 and decoded byte-identical to the repo, so the cells
below look like blobs rather than source. That is deliberate: a quoted literal that ends one
character early runs the rest of the module as *cell* code."""

SELFTEST_MD = """## 4. Self-test --- runs before any real data

Builds a synthetic CirCor (two patients, four recordings) and pushes it through the real
index and the real scorers. It checks the things that fail *silently*: that the patient id is
parsed from the leading filename field so five locations of one child pool into one bootstrap
group rather than five, that the window loop never indexes past the end of a recording, that
no label is read, and that the AUROC is oriented so an unknown-is-high score gives ~1 --- a
scorer pointed the wrong way returns `1 - AUROC`, which looks like a result rather than a bug.

It also runs the M49 self-test, because the forward path and the gate come from that module."""

SELFTEST = """sys.path.insert(0, "/kaggle/working")
import m49_xval as X
import m50_openset as O

assert X.selftest() == 0, "M49 self-test failed - do not run the evaluation"
assert O.selftest() == 0, "M50 self-test failed - do not run the evaluation"
"""

RUN_MD = """## 5. The gate, then the two questions

`m50_openset.run` re-scores the checkpoint on the 2,636 ICBHI test cycles first and asserts it
reproduces its own stored score. An open-set number from an unverified forward path is not a
result.

**Gate tolerance.** The default is `0.005`, not the `1e-3` M49 shipped with. M49 reproduced
0.5585 against a stored 0.5602 --- a gap of 0.0017, about 4 cycles in 2,636, caused by a
librosa version difference (M22_v2 was trained with librosa unpinned). 0.0017 is **8x below
the paper's own three-seed noise floor of 0.0141**, and a structurally wrong forward path
misses by 0.05--0.30, not 0.0017. The reproduced value is written into the results JSON next
to the stored one, so the gap is reported rather than hidden."""

RUN = """doc = O.run(CIRCOR_ROOT, CKPT, WORK,
            icbhi_audio=ICBHI_AUDIO, icbhi_split=ICBHI_SPLIT,
            limit=SMOKE, batch_size=64, max_windows=MAX_WINDOWS)
"""

COLLECT_MD = """## 6. Collect

Download `M50_results.zip` from the Output panel, unzip it into `M50_negative_control/`, and
commit.

**When you write this up:** it belongs in its own subsection, framed as trustworthiness, never
in the M49 transfer table. If it appears as "we also tested on CirCor" without the far-OOD
distinction, a reviewer who knows the corpus will read it as a category error."""

COLLECT = """import shutil
shutil.make_archive("/kaggle/working/M50_results", "zip", WORK)
print("zipped:", os.path.getsize("/kaggle/working/M50_results.zip"), "bytes")

d = json.load(open(os.path.join(WORK, "results_M50_circor.json")))
o, c = d["openset"], d["confidence"]
print("\\nenergy AUROC ", o["energy"]["auroc_unknown_vs_known"], o["energy"]["auroc_ci95"])
print("near-OOD ref ", o["near_ood_reference"]["auroc"],
      f"(n={o['near_ood_reference']['unknown_patients']} patients, HARDER task)")
print("confidence   ", c["circor"]["mean_max_softmax"], "on heart sounds vs",
      c["icbhi_test"]["mean_max_softmax"], "on ICBHI")
print("calls them   ", c["circor"]["predicted_class_fraction"])
"""


def build():
    return [
        md(HEADER),
        md("## 1. Environment"),
        code(SETUP),
        md("## 2. Locate the checkpoint, ICBHI and CirCor\n\nBy glob, not by hard-coded "
           "path --- Kaggle's dataset mirrors differ in layout."),
        code(FIND),
        md(MODULES_MD),
        code(embed(M45_SRC, "m45_ablation.py")),
        code(embed(M49_SRC, "m49_xval.py")),
        code(embed(M50_SRC, "m50_openset.py")),
        md(SELFTEST_MD),
        code(SELFTEST),
        md(RUN_MD),
        code(RUN),
        md(COLLECT_MD),
        code(COLLECT),
    ]


def write(cells, path):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.basename(path)}  ({len(cells)} cells)")


if __name__ == "__main__":
    for src in (M45_SRC, M49_SRC, M50_SRC):
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
    write(build(), os.path.join(HERE, "M50_circor_kaggle.ipynb"))
