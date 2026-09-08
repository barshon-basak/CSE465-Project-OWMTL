#!/usr/bin/env python3
"""Generate the two M49 external-validation notebooks for Kaggle.

WHY A GENERATOR AND NOT TWO HAND-WRITTEN NOTEBOOKS
    `Asif's/CLAUDE.md`: notebooks are generated, not hand-edited. Both notebooks embed
    `m45_ablation.py` and `m49_xval.py` verbatim, so neither can drift from the module that
    trained the checkpoint or from the module the self-test covers. Regenerate; do not edit
    the .ipynb.

OUTPUT
    M49_sprsound_kaggle.ipynb   SPRSound (BioCAS2022), event level + record level
    M49_hflung_kaggle.ipynb     HF_Lung_V1, inhalation+exhalation cycles

    python "M49_cross_dataset/gen_M49_kaggle.py"
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
M45_SRC = os.path.abspath(os.path.join(HERE, "..", "Asif's", "M45", "m45_ablation.py"))
M49_SRC = os.path.join(HERE, "m49_xval.py")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


def embed(path, name):
    """Embed a module as gzip+base64, written to /kaggle/working byte-identical to the repo.

    It used to be a raw triple-quoted literal holding the module verbatim. That form sits one
    stray quote sequence away from ending early, and when it ends early the rest of the module
    runs as *cell* code - which is how `__file__` came to be evaluated at notebook scope and
    raised NameError. The base64 alphabet has no quotes and no newlines, so the literal cannot
    be terminated by its own contents, and reading in binary keeps the bytes exact rather than
    at the mercy of newline translation.
    """
    var = name.upper().replace(".PY", "") + "_B64"
    blob = base64.b64encode(gzip.compress(io.open(path, "rb").read())).decode("ascii")
    q = chr(39) * 3      # written this way so this line cannot end the literal it builds
    wrapped = "\n".join(blob[i:i + 76] for i in range(0, len(blob), 76))
    return (f"{var} = (\n{q}{wrapped}{q})\n\n"
            "import base64, gzip\n"
            f"_src = gzip.decompress(base64.b64decode({var}))\n"
            f"with open('/kaggle/working/{name}', 'wb') as fh:\n"
            "    fh.write(_src)\n"
            f"print('wrote {name}', len(_src), 'bytes')\n")


# ------------------------------------------------------------------ shared cells
HEADER = """# M49 --- external validation of the best ICBHI model on **{title}**

The best checkpoint on the corrected ICBHI official split is run, **unchanged**, over a
corpus it has never seen, and scored with the same official ICBHI metric. No fine-tuning,
no threshold tuning, no target label used to fit anything in the headline number.

{body}

## Before you press Run All

| | |
|---|---|
| **Accelerator** | GPU T4 (CPU works but the pass is slower) |
| **Internet** | **ON** --- {net} |
| **Add Data** | `vbookshelf/respiratory-sound-database` (ICBHI --- needed for the verification gate) |
| **Add Data** | the checkpoint: upload `Asif's/M22_v2/Results/best_model.pth` as a private Kaggle dataset |
| **Runtime** | roughly {runtime} |

### The gate

Before a single external number is computed, the notebook re-scores the checkpoint on the
2,636 ICBHI test cycles and **asserts it reproduces the score stored inside the checkpoint**
(M22_v2: 0.5602). That proves this notebook's preprocessing, channel expansion and ImageNet
normalisation are the training run's --- so a low external score can be read as domain shift
rather than as a bug in the harness. A mismatch aborts the notebook.

### Bring back

`results_M49_*.json`, `preds_M49_*.npy`, `confusion_M49_*.png` --- zipped in the last cell.
Commit them into `M49_cross_dataset/`.
"""

SETUP = """!pip -q install librosa soundfile
import glob, json, os, subprocess, sys, time
import numpy as np
print(sys.version)
import torch, librosa
print("torch", torch.__version__, "| librosa", librosa.__version__, "| cuda:",
      torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

WORK = "/kaggle/working/M49"
os.makedirs(WORK, exist_ok=True)

# Set to a row count for a 2-minute smoke run; None for the real thing.
SMOKE = None
# The frozen-feature probe uses target labels, so it is NOT a zero-shot number. It is
# reported in its own block and answers a question the headline number cannot: whether the
# representation is useless here, or only the ICBHI-fitted decision boundary is.
RUN_PROBE = True
"""

FIND_CKPT = '''md_find = """"""
def find_one(pattern, what, hint=""):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"could not find {what} with {pattern!r}. {hint}")
    return hits[0]

# M22_v2 (0.5602) is the model the paper reports as best, so it is the one this external
# validation is about. M45 P3 scores higher (0.5764) and the paper explicitly declines it:
# the delta is 1.1x the seed noise range (0.0141) and its patient-level interval spans zero.
# Any M45 / M22_v2 checkpoint loads here -- the loader reads the preprocessing flags out of
# the checkpoint's own cfg and the gate holds it to the score IT reports -- but swapping this
# to P3 puts the notebook at odds with the paper's ablation section.
#
# NOT best_model_official.pth: same folder, 0.5641, trained on the published split verbatim,
# which leaks patients 156 and 218 across train and test.
CKPT_NAME = "best_model.pth"
CKPT = find_one(f"/kaggle/input/**/{CKPT_NAME}", "the checkpoint",
                f"Upload Asif's/M22_v2/Results/{CKPT_NAME} as a private Kaggle dataset and "
                "attach it, or set CKPT_NAME to whichever checkpoint you attached.")
ICBHI_AUDIO = os.path.dirname(find_one(
    "/kaggle/input/**/audio_and_txt_files/*.wav", "the ICBHI audio",
    "Add Data -> vbookshelf/respiratory-sound-database."))
ICBHI_SPLIT = find_one("/kaggle/input/**/ICBHI_challenge_train_test.txt",
                       "the official ICBHI split file",
                       "It ships with the same dataset. A notebook that cannot find it "
                       "must raise, never fall back (Model_Training_Protocol.md 1.1).")
print("checkpoint  :", CKPT)
print("ICBHI audio :", ICBHI_AUDIO, len(glob.glob(ICBHI_AUDIO + "/*.wav")), "wavs")
print("ICBHI split :", ICBHI_SPLIT)
'''.replace('md_find = """"""\n', '')

MODULES_MD = """## 3. The two modules

`m45_ablation.py` is embedded **verbatim** --- it is the module that trained this checkpoint,
and it supplies the mel parameters, the official metric and the corrected-split loader.
`m49_xval.py` imports it rather than re-implementing any of it, because a second
implementation of the mel parameters would make every cross-dataset delta a measurement of
the gap between two scripts.

Both are carried as gzip+base64 and decoded to /kaggle/working, byte-identical to the repo.
The two cells below therefore look like blobs rather than source. That is deliberate: they
used to be triple-quoted literals holding the modules verbatim, and a literal that ends one
character early runs the rest of the module as *cell* code, which surfaces as a NameError on
`__file__`. Base64 has no quotes and no newlines, so it cannot end early."""

SELFTEST_MD = """## 4. Self-test on synthetic corpora --- runs before any real data

Builds a fake SPRSound (wav + JSON) and a fake HF_Lung_V1 (wav + `_label.txt`) in a temp
directory and pushes both through the real index builders, the real spectrogram function and
the real network. It checks the things that fail *silently*: that every label string lands in
the right ICBHI class, that an inhalation and its exhalation are paired into one cycle with
the right times, that an adventitious span outside a cycle does not label it, that the two
slices of one session share a bootstrap group, that an unknown label **raises** instead of
being bucketed into Normal, and that the metric being applied is the official one rather than
the inflated macro variant."""

SELFTEST = """sys.path.insert(0, "/kaggle/working")
import m49_xval as X

assert X.selftest() == 0, "self-test failed - do not run the evaluation"
"""

GATE_MD = """## 5. The gate --- reproduce the checkpoint's own ICBHI score

If this cell does not reproduce the score stored inside the checkpoint to within 1e-3, every
number after it is uninterpretable and the notebook stops here."""

GATE = """model, cfg, meta = X.load_checkpoint(CKPT)
print("checkpoint:", meta)
print("preprocessing:", {k: cfg[k] for k in
      ("n_mels", "duration_s", "padding", "minmax", "bandpass", "denoise", "ampnorm",
       "pretrained")})

# return_details keeps the 2,636-cycle pass this cell just made. The evaluation below reuses
# it for the calibration comparison and for ICBHI's own class prior, so those come from the
# verified forward pass rather than a second one that could differ.
GATE = X.verify_on_icbhi(model, cfg, meta, ICBHI_AUDIO, ICBHI_SPLIT,
                         tol=0.005, return_details=True, batch_size=64)
ICBHI_REF = GATE["score"]
"""

COLLECT_MD = """## Collect

Download `M49_results.zip` from the Output panel, unzip it into `M49_cross_dataset/`, and
commit. The results JSONs follow `Model_Training_Protocol.md` section 4, carry the full
taxonomy mapping in `dataset_info.label_mapping`, and name the bootstrap's grouping unit in
`best_metrics.icbhi_score_official_ci95_unit`."""

COLLECT = """import shutil
shutil.make_archive("/kaggle/working/M49_results", "zip", WORK)
print("zipped:", os.path.getsize("/kaggle/working/M49_results.zip"), "bytes")
for f in sorted(glob.glob(os.path.join(WORK, "results_M49_*.json"))):
    d = json.load(open(f))
    b, t = d["best_metrics"], d["transfer"]
    print(f"{d['meta']['model_id']:26s} n={d['dataset_info']['test_samples']:6d}  "
          f"ICBHI {X._s(b['icbhi_score_official'])} {b['icbhi_score_official_ci95']}  "
          f"Se {X._s(b['icbhi_se_official'])}  Sp {X._s(b['icbhi_sp_official'])}  "
          f"delta vs in-domain {X._s(t['delta'], sign=True)}")
"""


def common_cells(title, body, net, runtime, data_md, data_code):
    return [
        md(HEADER.format(title=title, body=body, net=net, runtime=runtime)),
        md("## 1. Environment"),
        code(SETUP),
        md("## 2. Locate the checkpoint and ICBHI\n\nBy glob, not by a hard-coded path --- "
           "Kaggle's dataset mirrors differ in layout, and a wrong path that silently "
           "resolves to an empty directory is a failure mode this project has already been "
           "bitten by."),
        code(FIND_CKPT),
        md(data_md),
        code(data_code),
        md(MODULES_MD),
        code(embed(M45_SRC, "m45_ablation.py")),
        code(embed(M49_SRC, "m49_xval.py")),
        md(SELFTEST_MD),
        code(SELFTEST),
        md(GATE_MD),
        code(GATE),
    ]


def tail_cells():
    return [md(COLLECT_MD), code(COLLECT)]


# ------------------------------------------------------------------ SPRSound
SPR_BODY = """**SPRSound (BioCAS 2022)** --- 2,683 recordings from 292 pediatric participants
(1 month to 18 years) at Shanghai Children's Medical Center, recorded with Yunting Model II
stethoscopes. Two shifts at once against ICBHI: a different population *and* different
hardware.

**Unit of analysis.** SPRSound's `event_annotation` entries are cycle-like segments and are
the headline unit, matching ICBHI's respiratory cycle. `record_annotation` is run as a
**secondary** row on a different unit (a whole ~9 s recording truncated to the model's 8 s
window) and is not comparable to the event-level number.

**Taxonomy.** `Normal` -> Normal; `Fine Crackle` / `Coarse Crackle` -> Crackle;
`Wheeze` / `Rhonchi` / `Stridor` -> Wheeze (all three are continuous adventitious sounds,
which is the class ICBHI calls Wheeze); `Wheeze+Crackle` -> Both. Record level:
`DAS` -> Crackle, `CAS` -> Wheeze, `CAS & DAS` -> Both, **`Poor Quality` excluded** (the
annotators judged the audio unusable). An unmapped label string raises.

**Confidence interval:** patient-level bootstrap --- the filename's first field is the
patient ID."""

SPR_DATA_MD = """## 2b. Get SPRSound

Tried in order: an attached Kaggle dataset first, then a sparse clone of the official GitHub
repository (only `BioCAS2022`, about 700 MB, so the 2023--2025 releases are not pulled).
Needs **Internet ON**."""

SPR_DATA = '''SPR_ROOT = None
hits = sorted(glob.glob("/kaggle/input/**/BioCAS2022", recursive=True))
if not hits:      # a mirror that flattened or renamed the release folder
    j = sorted(glob.glob("/kaggle/input/**/*2022_json/**/*.json", recursive=True))
    hits = [os.path.commonpath([os.path.dirname(p) for p in j])] if j else []
if hits:
    SPR_ROOT = hits[0]
    print("using the attached dataset:", SPR_ROOT)
else:
    dst = "/kaggle/working/SPRSound"
    if not os.path.exists(dst):
        url = "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git"
        t0 = time.time()
        rc = subprocess.call(["git", "clone", "--depth", "1", "--filter=blob:none",
                              "--sparse", url, dst])
        if rc == 0:
            rc = subprocess.call(["git", "-C", dst, "sparse-checkout", "set", "BioCAS2022"])
        if rc != 0:                      # older git without partial-clone support
            subprocess.call(["rm", "-rf", dst])
            rc = subprocess.check_call(["git", "clone", "--depth", "1", url, dst])
        print(f"cloned in {(time.time()-t0)/60:.1f} min")
    SPR_ROOT = os.path.join(dst, "BioCAS2022")

assert os.path.isdir(SPR_ROOT), (
    f"{SPR_ROOT} missing. Turn Internet ON, or attach a SPRSound mirror as a dataset.")
print("SPRSound root:", SPR_ROOT)
for sub in sorted(os.listdir(SPR_ROOT)):
    n = len(glob.glob(os.path.join(SPR_ROOT, sub, "**", "*.*"), recursive=True))
    print(f"  {sub:20s} {n}")
print("wav :", len(glob.glob(os.path.join(SPR_ROOT, "**", "*.wav"), recursive=True)))
print("json:", len(glob.glob(os.path.join(SPR_ROOT, "**", "*.json"), recursive=True)))
'''

SPR_EVAL_MD = """## 6. Event level --- the headline number

One forward pass over every annotated event. Spectrograms are computed on the fly: this is a
single pass, so an M45-style disk cache would cost gigabytes and buy nothing.

Read the printed label vocabulary before the score. It is the audit that the mapping covered
what the corpus actually contains --- and any string it does not cover has already raised."""

SPR_EVAL = """doc_event = X.run("sprsound", SPR_ROOT, CKPT, WORK, level="event",
                  icbhi_gate=GATE, limit=SMOKE, batch_size=64, probe=RUN_PROBE)
"""

SPR_REC_MD = """## 7. Record level --- secondary, and on a different unit

Whole recordings against `record_annotation`. The model's window is 8 s and a SPRSound record
is about 9 s, so this truncates; it is reported because record-level labels are what the
BioCAS 2022 challenge scored, **not** because it is comparable to the event-level row above.
Different unit, different number."""

SPR_REC = """doc_record = X.run("sprsound", SPR_ROOT, CKPT, WORK, level="record",
                   icbhi_gate=GATE, limit=SMOKE, batch_size=64, probe=False)
"""


def build_sprsound():
    cells = common_cells(
        "SPRSound (BioCAS 2022)", SPR_BODY, "the notebook clones SPRSound from GitHub",
        "10 min clone + 5 min evaluation on a T4", SPR_DATA_MD, SPR_DATA)
    cells += [md(SPR_EVAL_MD), code(SPR_EVAL), md(SPR_REC_MD), code(SPR_REC)]
    cells += tail_cells()
    return cells


# ------------------------------------------------------------------ HF_Lung_V1
HFL_BODY = """**HF_Lung_V1** --- 9,765 fifteen-second recordings from Taiwan, natively at
4 kHz, labelled with breath phases and adventitious-sound spans. The shift against ICBHI is
population, hardware **and** bandwidth: 4 kHz sampling puts Nyquist at 2,000 Hz, exactly the
model's mel `fmax`, so the top of the band arrives attenuated by the source anti-aliasing.
That is a real confound and it is recorded in `dataset_info.native_sample_rates_hz`.

**Unit of analysis.** HF_Lung_V1 annotates breath *phases*, not cycles, so each inhalation is
paired with the exhalation that follows it within 1 s to form one ICBHI-style cycle. An
unpaired phase becomes a cycle on its own rather than being dropped --- discarding lone phases
would throw away the breaths at the clip boundaries, which is where a 15 s window cuts a cycle
in half, and that loss is not random with respect to the label.

**Taxonomy.** A cycle is Crackle if a `D` span overlaps it by >= 50 ms, Wheeze if a
`Wheeze` / `Stridor` / `Rhonchi` span does, Both if both, Normal otherwise --- the same
"contains the sound" rule ICBHI uses. `I` and `E` build the cycle and are not labels. An
unmapped token raises.

**Confidence interval:** **recording-level**, not patient-level. HF_Lung_V1 filenames carry a
timestamp and no patient ID, so the bootstrap resamples recording sessions (all
`trunc_...-LX_N` slices of one session stay together). Calling it patient-level would
overstate what the resampling controls for, and the results JSON names the unit explicitly."""

HFL_DATA_MD = """## 2b. Get HF_Lung_V1

The corpus ships as split 7-Zip archives on GitLab (`train.7z.001`--`010`,
`test.7z.001`--`003`, about 1.1 GB total). Tried in order: an attached Kaggle dataset, then a
direct download. Needs **Internet ON**.

Set `HFL_PARTS = ("test",)` for a faster first pass over the test half only --- but then say
so, because it is a different corpus subset from the full run."""

HFL_DATA = '''HFL_PARTS = ("train", "test")     # ("test",) for a quicker first pass
HFL_ROOT = None

hits = sorted(glob.glob("/kaggle/input/**/*_label.txt", recursive=True))
if hits:
    # commonpath over every label file, so a mirror that flattened train/ and test/ into one
    # directory resolves just as well as one that kept them
    HFL_ROOT = os.path.commonpath([os.path.dirname(p) for p in hits])
    print("using the attached dataset:", HFL_ROOT, len(hits), "label files")
else:
    if subprocess.call(["which", "7z"]) != 0:
        subprocess.call(["apt-get", "-qq", "install", "-y", "p7zip-full"])
    assert subprocess.call(["which", "7z"]) == 0, (
        "7z is not available and could not be installed. Extract HF_Lung_V1 locally, "
        "upload it as a Kaggle dataset, and attach it - the cell above will find it.")
    HFL_ROOT = "/kaggle/working/HF_Lung_V1"
    os.makedirs(HFL_ROOT, exist_ok=True)
    base = "https://gitlab.com/techsupportHF/HF_Lung_V1/-/raw/master"
    n_parts = {"train": 10, "test": 3}
    for part in HFL_PARTS:
        if os.path.isdir(os.path.join(HFL_ROOT, part)):
            print(f"{part}/ already extracted, skipping")
            continue
        t0 = time.time()
        for i in range(1, n_parts[part] + 1):
            name = f"{part}.7z.{i:03d}"
            dst = os.path.join(HFL_ROOT, name)
            if not os.path.exists(dst):
                subprocess.check_call(["wget", "-q", "-O", dst,
                                       f"{base}/{name}?inline=false"])
            print(f"  {name} {os.path.getsize(dst)/1e6:.0f} MB")
        subprocess.check_call(["7z", "x", "-y", f"-o{HFL_ROOT}",
                               os.path.join(HFL_ROOT, f"{part}.7z.001")],
                              stdout=subprocess.DEVNULL)
        for f in glob.glob(os.path.join(HFL_ROOT, f"{part}.7z.*")):
            os.remove(f)
        print(f"  {part} ready in {(time.time()-t0)/60:.1f} min")

wavs = glob.glob(os.path.join(HFL_ROOT, "**", "*.wav"), recursive=True)
labs = glob.glob(os.path.join(HFL_ROOT, "**", "*_label.txt"), recursive=True)
assert wavs and labs, (f"nothing under {HFL_ROOT}. Turn Internet ON, or attach an "
                       "HF_Lung_V1 mirror as a dataset.")
print("HF_Lung_V1 root:", HFL_ROOT, "|", len(wavs), "wav |", len(labs), "label files")
print("a label file, verbatim:")
print(open(sorted(labs)[0]).read()[:300])
'''

HFL_EVAL_MD = """## 6. Cycle level --- the headline number

One forward pass over every inhalation+exhalation cycle. Read the printed label vocabulary and
the drop counts before the score: they are the audit that the cycle builder saw what the
corpus actually contains."""

HFL_EVAL = """doc_cycle = X.run("hflung", HFL_ROOT, CKPT, WORK,
                  icbhi_gate=GATE, limit=SMOKE, batch_size=64, probe=RUN_PROBE)
"""


def build_hflung():
    cells = common_cells(
        "HF_Lung_V1", HFL_BODY, "the notebook downloads HF_Lung_V1 from GitLab",
        "10 min download + 10 min evaluation on a T4", HFL_DATA_MD, HFL_DATA)
    cells += [md(HFL_EVAL_MD), code(HFL_EVAL)]
    cells += tail_cells()
    return cells


# ------------------------------------------------------------------
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
    for src in (M45_SRC, M49_SRC):
        if not os.path.exists(src):
            raise SystemExit(f"missing {src}")
    write(build_sprsound(), os.path.join(HERE, "M49_sprsound_kaggle.ipynb"))
    write(build_hflung(), os.path.join(HERE, "M49_hflung_kaggle.ipynb"))
