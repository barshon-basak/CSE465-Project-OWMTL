#!/usr/bin/env python3
"""
Generator for M39 — Physics-Derived Acoustic Concept Extraction & Validation (Gate G2).

Run `python3 gen_M39.py` to emit M39_concept_extraction.ipynb.

WHAT M39 IS
-----------
The first build of the new direction's shared engine. It computes the 9 physics-derived acoustic
concepts for every annotated ICBHI cycle and asks the Gate-G2 question: **do the extractors
actually measure what their names claim?**

Lineage: M35's `VectorizedAcousticPhysicsLoss` computed 2 of these (spectral flatness, temporal
PAPR) as a training regulariser on log-mels. M39 turns them into standalone per-cycle
measurements on the raw waveform and adds 7 more. M35's *number* is not reused (70/30 split,
legacy metric); only its DSP idea is.

DESIGN DECISIONS WORTH KNOWING
------------------------------
1. **Raw cycle audio, not the 8 s tiled version.** M2/M3 tile short cycles to a fixed 8 s for the
   CNN. Doing that here would repeat every crackle 3-4x and fabricate `crackle_rate_hz`. Concepts
   are computed on the cycle's true duration.
2. **The extractor module is inlined, not imported.** Colab has no access to this repo, so the
   generator splices `Asif's/engine/concept_extractors.py` verbatim. Single source of truth: edit
   the module, regenerate, never hand-edit the notebook.
3. **Only 2 of 9 concepts can be validated.** ICBHI labels crackle/wheeze *presence* only. The
   other 7 are proxies and the notebook reports them in a separate section that makes no
   validation claim. See `CLINICIAN_LABELING_PACK.md` for what would change that.
4. **CC1 score dumping is built in**, so paired DeLong tests against any later detector work.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

with open(os.path.join(REPO, "Asif's", "engine", "concept_extractors.py")) as f:
    EXTRACTOR_SRC = f.read()
# strip the module's __main__ demo -- the notebook has its own
EXTRACTOR_SRC = EXTRACTOR_SRC.split('if __name__ == "__main__":')[0].rstrip()

CELLS = []


def md(t):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")})


def code(t):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": t.strip("\n")})


# ===========================================================================
md(r"""
# M39 — Physics-Derived Acoustic Concept Extraction & Validation

**Gate:** G2 (the make-or-break technical gate) · **Contributor:** Asif · **Requires:** ICBHI audio
+ the committed official split. **No model, no training, no GPU** — this is pure DSP.

## The question this answers

The new direction routes diagnosis through a bottleneck of *clinically-named acoustic concepts*.
That only means anything if the concepts measure what their names say. G2 asks exactly that:

> Do the DSP extractors reproduce ICBHI's coarse crackle/wheeze labels materially better than
> chance?

## What can and cannot be validated here

ICBHI annotates each respiratory cycle for **crackle presence** and **wheeze presence** — nothing
finer. So of the 9 concepts:

| | Concepts | Validated against |
|---|---|---|
| **Validatable** (2) | `crackle_score`, `wheeze_score` | ICBHI cycle labels — reported with AUROC + CI |
| **Proxy** (5) | `crackle_fine_ratio`, `crackle_rate_hz`, `wheeze_pitch_hz`, `rhonchi_score`, `inspiratory_fraction` | **nothing** — no ground truth exists |
| **Descriptive** (2) | `spectral_flatness`, `temporal_papr` | n/a — not claims about a clinical entity |

The proxy section below reports distributions and makes **no accuracy claim**. Promoting any proxy
requires the clinician labeling exercise in `CLINICIAN_LABELING_PACK.md` (Gate G0).

**Never write "fine crackle" for `crackle_fine_ratio` in the paper.** Write "physics-derived proxy
for fine-crackle fraction". Naming a proxy after the thing it proxies is how the ICBHI-metric
problem started.

## Two things done differently from M35

**Raw waveform, not log-mels.** M35 computed flatness/PAPR from a normalised log-mel via an
`exp(spec * 3.0)` un-log approximation. Fine for a soft penalty; not a defensible basis for a
number the paper names "spectral flatness". Also crackles are 5–15 ms events and the shared mel
hop is 10 ms, so a mel frame barely resolves one.

**True cycle duration, not the 8 s tiled clip.** Tiling a 2 s cycle to 8 s repeats every crackle
four times — `crackle_rate_hz` would be pure fiction.

## Three bugs the synthetic tests caught before this ever ran

`Asif's/engine/test_concept_extractors.py` builds signals with *constructed* acoustic ground truth
and asserts the extractors recover it. It found:

1. A pure 400 Hz tone scoring **rhonchi 0.970** — the rhonchi band (60–300 Hz) still had a
   locally-prominent leakage bin. Fixed by requiring the in-band peak to be a real share of the
   frame's global peak.
2. A brick-wall FFT band-pass ringing (Gibbs), producing a **phantom crackle ~10 ms after every
   real one** — a 50% inflation of `crackle_rate_hz`. Fixed with raised-cosine band edges.
3. Detection firing on **numerical residue at 1e-16** when the inter-event signal was near-silent
   (median/MAD → 0). Fixed with an absolute energy floor: a crackle is an explosive sound, not
   merely a statistical outlier.

Running on real audio would have surfaced none of these — it only shows the code executes, never
that a detector named "crackle" responds to crackles.
""")

# ===========================================================================
md(r"""
---
## Section 1 — Environment
""")

code(r'''
# ============================================================
# CELL 0 — ENVIRONMENT & PATHS
# ============================================================
import os, sys, platform

print("=" * 70)
print("M39 — Concept Extraction & Validation (Gate G2)")
print("=" * 70)
print(f"Python : {sys.version.split()[0]}  ({platform.platform()})")
print("No GPU needed — this notebook is pure DSP (numpy only).")

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
DRIVE_MOUNTED = False
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_MOUNTED = True
        print("Drive mounted.")
    except Exception as e:
        print(f"Drive mount skipped ({e}).")

if IN_COLAB and DRIVE_MOUNTED:
    BASE_DIR = "/content/drive/MyDrive/OWMTL/M39"
elif IN_COLAB:
    BASE_DIR = "/content/OWMTL/M39"
elif os.path.exists("/kaggle/working"):
    BASE_DIR = "/kaggle/working"
else:
    BASE_DIR = "./outputs_M39"

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\nResults -> {RESULTS_DIR}")
print("=" * 70)
''')

md(r"""
---
## Section 1b — Data

Needs two things: the ICBHI audio (downloaded below) and **`ICBHI_challenge_train_test.txt`**,
which is committed at `Asif's/ICBHI_challenge_train_test.txt` — upload it to `/content/`.

The notebook **refuses to run without the split file.** Earlier models silently fell back to a
`patient_id <= 111` rule that produced an 11-patient test set while labelling itself "official
60/40". That silent fallback is exactly what this refusal exists to prevent.
""")

code(r'''
# ============================================================
# CELL 0b — DOWNLOAD ICBHI (idempotent)
# ============================================================
import os

os.environ['KAGGLE_USERNAME'] = 'AsifM7'
os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KAGGLE_API_KEY_HERE'   # <-- replace before running
# Safer: from google.colab import userdata; os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

import subprocess, sys, glob

_DL = "/content" if os.path.exists("/content") else "./data"
os.makedirs(_DL, exist_ok=True)


def _audio_dir():
    for d in glob.glob(os.path.join(_DL, "**", "audio_and_txt_files"), recursive=True):
        if glob.glob(os.path.join(d, "*.wav")):
            return d
    return None


if _audio_dir():
    print(f"ICBHI already present: {len(glob.glob(os.path.join(_audio_dir(), '*.wav')))} .wav")
else:
    if os.environ.get('KAGGLE_KEY', '') in ('', 'PASTE_YOUR_KAGGLE_API_KEY_HERE'):
        raise RuntimeError("Paste your Kaggle API key into this cell first.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
    subprocess.check_call(["kaggle", "datasets", "download",
                           "-d", "vbookshelf/respiratory-sound-database", "-p", _DL, "--unzip"])
    assert _audio_dir(), "Download finished but no audio found."
    print(f"Done: {len(glob.glob(os.path.join(_audio_dir(), '*.wav')))} .wav")
''')

code(r'''
# ============================================================
# CELL 1 — DEPENDENCIES, PATHS, CONFIG
# ============================================================
import subprocess, glob


def pip_install(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


for p, i in [("librosa", None), ("soundfile", None), ("numpy", None),
             ("pandas", None), ("matplotlib", None), ("tqdm", None)]:
    pip_install(p, i)
print("Dependencies ready.\n")


def find_first(pats):
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def find_audio_dir():
    for root in ("/content", "/kaggle/input", "./data", "."):
        if os.path.isdir(root):
            for d in sorted(glob.glob(os.path.join(root, "**", "audio_and_txt_files"),
                                      recursive=True)):
                if glob.glob(os.path.join(d, "*.wav")):
                    return d
    return None


DATA_ROOT = find_audio_dir()
SPLIT_FILE = find_first([
    "/content/ICBHI_challenge_train_test.txt",
    "/content/**/ICBHI_challenge_train_test.txt",
    "/kaggle/input/**/ICBHI_challenge_train_test.txt",
    "./**/ICBHI_challenge_train_test.txt",
])

print(f"DATA_ROOT  : {DATA_ROOT}")
print(f"SPLIT_FILE : {SPLIT_FILE}")

if not DATA_ROOT:
    raise RuntimeError("ICBHI audio not found — run Cell 0b.")
if not SPLIT_FILE:
    raise RuntimeError(
        "ICBHI_challenge_train_test.txt not found. It is committed at "
        "Asif's/ICBHI_challenge_train_test.txt -- upload it to /content/ and re-run. "
        "This notebook refuses to fall back to a patient-id rule: doing so silently is what "
        "gave M2/M3/M12/M22 an 11-patient test set mislabelled as the official 60/40 split.")

CFG = {
    "model_id": "M39",
    "contributor": "Asif",
    "sample_rate": 16000,
    "min_cycle_s": 0.15,        # below this a cycle carries no usable acoustic content
    "seed": 42,
    "data_root": DATA_ROOT,
    "split_file": SPLIT_FILE,
    "results_dir": RESULTS_DIR,
}
print("\n" + "=" * 60)
for k, v in CFG.items():
    print(f"  {k:<14}: {v}")
print("=" * 60)
''')

code(r'''
# ============================================================
# CELL 2 — IMPORTS & SEED
# ============================================================
import json, time, math, random, warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
print("Ready.")
''')

# ===========================================================================
md(r"""
---
## Section 2 — The concept extractors

Spliced verbatim from `Asif's/engine/concept_extractors.py` so Colab needs no repo access. **Do not
edit this cell** — edit the module and regenerate the notebook. The module is covered by
`test_concept_extractors.py` (43/43 on synthetic signals with constructed ground truth).
""")

code("# ============================================================\n"
     "# CELL 3 — CONCEPT EXTRACTORS  (spliced from Asif's/engine/concept_extractors.py)\n"
     "# ============================================================\n"
     + EXTRACTOR_SRC)

code(r'''
# ============================================================
# CELL 4 — SELF-CHECK: the extractors behave on known signals
# ============================================================
# Re-runs the core synthetic assertions here, so a Colab run proves the spliced copy behaves
# exactly like the tested module rather than assuming it.
_sr = 16000
_t = np.arange(_sr) / _sr


def _tone(hz):
    return np.sin(2 * np.pi * hz * _t)


def _crackles(width_ms, n=12, carrier=650.0, seed=0):
    sig = np.zeros(_sr)
    w = int(width_ms / 1000 * _sr)
    for c in np.linspace(0, _sr - w - 1, n).astype(int):
        sig[c:c + w] += np.sin(2 * np.pi * carrier * np.arange(w) / _sr) * np.hanning(w)
    return sig


_c400, _cnoise = extract_concepts(_tone(400), _sr), extract_concepts(
    np.random.RandomState(0).randn(_sr) * 0.1, _sr)
_cfine = extract_concepts(_crackles(5.0, carrier=650.0), _sr)
_ccoarse = extract_concepts(_crackles(15.0, carrier=300.0, seed=1), _sr)

_checks = [
    ("400 Hz tone -> wheeze fires", _c400["wheeze_score"] > 0.8),
    ("400 Hz tone -> pitch recovered", abs(_c400["wheeze_pitch_hz"] - 400) < 25),
    ("400 Hz tone -> rhonchi does NOT fire", _c400["rhonchi_score"] < 0.15),
    ("noise -> high flatness, no wheeze",
     _cnoise["spectral_flatness"] > 0.3 and _cnoise["wheeze_score"] < 0.2),
    ("fine train -> high fine_ratio", _cfine["crackle_fine_ratio"] > 0.6),
    ("coarse train -> low fine_ratio", _ccoarse["crackle_fine_ratio"] < 0.3),
    ("crackle rate ~ constructed 12/s", 9 <= _ccoarse["crackle_rate_hz"] <= 15),
]
for _n, _ok in _checks:
    print(f"  [{'OK ' if _ok else 'FAIL'}] {_n}")
assert all(ok for _, ok in _checks), "Spliced extractors do not match tested behaviour."
print("\nSelf-check passed — the spliced extractors behave as tested.")
''')

# ===========================================================================
md(r"""
---
## Section 3 — Cycles and the official split

Each ICBHI annotation line is one respiratory cycle with crackle/wheeze flags. Those flags are the
only ground truth G2 has.

The split is the official file with patients **156** and **218** reassigned wholly to train — they
have recordings on *both* sides of the published split, which violates Protocol §1's
patient-independence requirement. Cost: 12 of 381 test recordings (3.1%). See
`Asif's/audit/official_split.py`.
""")

code(r'''
# ============================================================
# CELL 5 — PARSE CYCLES + OFFICIAL SPLIT (patient-independent corrected)
# ============================================================
rows = []
for line in open(CFG["split_file"]):
    p = line.split()
    if len(p) >= 2 and p[1].lower() in ("train", "test"):
        rows.append((p[0].replace(".wav", ""), p[1].lower()))

by_patient = {}
for stem, sp in rows:
    by_patient.setdefault(int(stem.split("_")[0]), set()).add(sp)
LEAKING = sorted(p for p, v in by_patient.items() if len(v) > 1)
n_test_before = sum(1 for _, sp in rows if sp == "test")
SPLIT_MAP = {stem: ("train" if int(stem.split("_")[0]) in LEAKING else sp) for stem, sp in rows}
n_test_after = sum(1 for v in SPLIT_MAP.values() if v == "test")

print(f"Official split: {len(rows)} recordings, {len(by_patient)} patients")
print(f"  NOT patient-independent: patients {LEAKING} appear on both sides")
print(f"  Protocol section 1 fix -> reassigned to TRAIN; test {n_test_before} -> {n_test_after} "
      f"recordings ({(n_test_before - n_test_after) / n_test_before * 100:.1f}% moved)")


def parse_annotations(txt_path):
    out = []
    for line in open(txt_path):
        p = line.split()
        if len(p) < 4:
            continue
        try:
            s, e, cr, wh = float(p[0]), float(p[1]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if e > s:
            out.append({"start": s, "end": e, "crackle": cr, "wheeze": wh})
    return out


cycles = []
for wav in sorted(glob.glob(os.path.join(CFG["data_root"], "*.wav"))):
    stem = os.path.splitext(os.path.basename(wav))[0]
    txt = os.path.join(CFG["data_root"], stem + ".txt")
    if not os.path.exists(txt):
        continue
    sp = SPLIT_MAP.get(stem)
    if sp is None:
        continue                       # not in the official split file
    pid = int(stem.split("_")[0])
    for k, c in enumerate(parse_annotations(txt)):
        cycles.append({"stem": stem, "patient_id": pid, "split": sp,
                       "cycle_idx": k, "wav_path": wav, **c})

df = pd.DataFrame(cycles)
assert len(df), "No cycles parsed."
df["label4"] = df.crackle * 1 + df.wheeze * 2      # 0 normal,1 crackle,2 wheeze,3 both
df["unit_id"] = df.stem + "__" + df.cycle_idx.astype(str)
df["duration_s"] = df.end - df.start

tr_p, te_p = set(df[df.split == "train"].patient_id), set(df[df.split == "test"].patient_id)
assert not (tr_p & te_p), f"PATIENT LEAKAGE: {sorted(tr_p & te_p)[:5]}"

print(f"\nCycles: {len(df)}")
print(f"  train {int((df.split == 'train').sum()):>5} from {len(tr_p)} patients")
print(f"  test  {int((df.split == 'test').sum()):>5} from {len(te_p)} patients")
print("[OK] patient-independent verified\n")
print("Label distribution (all cycles):")
for i, nm in enumerate(["Normal", "Crackle", "Wheeze", "Both"]):
    n = int((df.label4 == i).sum())
    print(f"  {nm:<9}{n:>6}  ({n / len(df) * 100:5.1f}%)")
print(f"\ncrackle-positive: {int(df.crackle.sum())}   wheeze-positive: {int(df.wheeze.sum())}")
print(f"cycle duration: median {df.duration_s.median():.2f}s  "
      f"range {df.duration_s.min():.2f}-{df.duration_s.max():.2f}s")
''')

code(r'''
# ============================================================
# CELL 6 — EXTRACT CONCEPTS FOR EVERY CYCLE
# ============================================================
# Audio is loaded at the cycle's TRUE duration -- never tiled to a fixed 8 s. Tiling a 2 s cycle
# to 8 s would repeat each crackle ~4x and make crackle_rate_hz fiction.

_cache = {}


def load_full(path, sr):
    if path not in _cache:
        if len(_cache) > 40:                     # bounded: recordings are large
            _cache.clear()
        _cache[path] = librosa.load(path, sr=sr, mono=True)[0]
    return _cache[path]


recs, skipped = [], 0
t0 = time.time()
for _, r in tqdm(df.iterrows(), total=len(df), desc="extracting concepts"):
    try:
        y = load_full(r["wav_path"], CFG["sample_rate"])
        a, b = int(r["start"] * CFG["sample_rate"]), int(r["end"] * CFG["sample_rate"])
        seg = y[max(0, a):min(len(y), b)]
        if len(seg) < CFG["min_cycle_s"] * CFG["sample_rate"]:
            skipped += 1
            recs.append({k: np.nan for k in CONCEPT_NAMES})
            continue
        recs.append(extract_concepts(seg, CFG["sample_rate"]))
    except Exception:
        skipped += 1
        recs.append({k: np.nan for k in CONCEPT_NAMES})

EXTRACT_TIME = time.time() - t0
C = pd.DataFrame(recs, index=df.index)
df = pd.concat([df, C], axis=1)
_cache.clear()

print(f"\nExtracted in {EXTRACT_TIME:.0f}s ({EXTRACT_TIME / len(df) * 1000:.1f} ms/cycle)")
print(f"Skipped (too short / unreadable): {skipped} ({skipped / len(df) * 100:.2f}%)")
usable = df[CONCEPT_NAMES].notna().all(axis=1)
df = df[usable].reset_index(drop=True)
print(f"Usable cycles: {len(df)}")
print("\nConcept summary (all usable cycles):")
print(df[CONCEPT_NAMES].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())
''')

# ===========================================================================
md(r"""
---
## Section 4 — Gate G2: the validation that decides the plan

Only `crackle_score` and `wheeze_score` are tested — they are the only two with ICBHI ground truth.
Each gets an AUROC with a DeLong 95% CI on the **test split** (train-split figures are printed for
reference but the gate reads the test split).

**Gate rule:** a concept passes if its 95% CI excludes chance, i.e. it is *materially* better than
a coin flip, not merely above 0.5 by a point estimate.
""")

code(r'''
# ============================================================
# CELL 7 — AUROC + DeLong CI  (self-contained; mirrors Asif's/Statistics/owmtl_scores.py)
# ============================================================
def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = np.asarray(x, dtype=float)[J]
    N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def _fast_delong(mat, n_pos):
    m, k = n_pos, mat.shape[0]
    n = mat.shape[1] - m
    tx = np.vstack([_midrank(mat[r, :m]) for r in range(k)])
    ty = np.vstack([_midrank(mat[r, m:]) for r in range(k)])
    tz = np.vstack([_midrank(mat[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01, ddof=1)) / m + np.atleast_2d(np.cov(v10, ddof=1)) / n


def auprc(y, s):
    """Average precision. Reported alongside AUROC because the classes are imbalanced
    (~27% crackle-positive, ~17% wheeze-positive) and AUROC is optimistic under imbalance.
    Its chance baseline is the POSITIVE PREVALENCE, not 0.5."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = int(y.sum())
    return float((prec * y).sum() / n_pos) if n_pos else 0.0


def auroc_ci(y, s, alpha=0.05):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    order = np.argsort(-y, kind="mergesort")
    a, cov = _fast_delong(s[order][None, :], int(y.sum()))
    se = math.sqrt(max(float(cov[0, 0]), 0.0))
    z = 1.959963984540054
    lo, hi = max(0.0, float(a[0]) - z * se), min(1.0, float(a[0]) + z * se)
    return {"auroc": float(a[0]), "se": se, "ci_lo": lo, "ci_hi": hi,
            "excludes_chance": bool(lo > 0.5 or hi < 0.5),
            "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}


# TIGHTENED 2026-08-16. The first version passed a concept whenever its CI excluded chance.
# At n=2636 cycles that admits AUROC 0.52 -- statistically significant, practically useless. The
# first real run duly "passed" with crackle 0.5506 / wheeze 0.5729 while the crackle detector was
# firing on 100% of cycles. A gate that cannot fail is not a gate.
MIN_AUROC = 0.65

VALIDATABLE = [(k, v) for k, (s, v) in CONCEPT_VALIDATION.items() if s == "validatable"]
TARGETS = {"crackle_score": "crackle", "wheeze_score": "wheeze"}

G2 = {}
# TRAIN FIRST, deliberately. If an extractor needs work, diagnose it on train and leave the test
# split untouched -- tuning against the number the gate reads is fitting the gate, not passing it.
for split in ("train", "test"):
    if split == "test":
        print("\n" + "!" * 74)
        print("TEST SPLIT BELOW — treat this as your ONE look.")
        print("If you change the extractors after seeing these numbers and re-read them, the")
        print("gate no longer means anything. Diagnose on TRAIN, change, then look here once.")
        print("!" * 74)
    sub = df[df.split == split]
    print(f"\n{'=' * 74}\n{split.upper()} SPLIT  (n={len(sub)} cycles)\n{'=' * 74}")
    for concept, target in TARGETS.items():
        r = auroc_ci(sub[target].values, sub[concept].values)
        if r is None:
            continue
        mark = "PASS" if r["excludes_chance"] and r["auroc"] >= MIN_AUROC else "FAIL"
        print(f"  {concept:<16} vs ICBHI '{target}'   AUROC {r['auroc']:.4f}  "
              f"95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]   "
              f"n+={r['n_pos']} n-={r['n_neg']}   [{mark}]")
        ap = auprc(sub[target].values, sub[concept].values)
        prev = float(sub[target].mean())
        print(f"  {'':<16}    AUPRC {ap:.4f}  (chance = prevalence {prev:.4f})")
        if split == "test":
            G2[concept] = {**r, "target": target, "auprc": ap, "prevalence": prev,
                           "auprc_lift_over_prevalence": round(ap - prev, 4),
                           "min_auroc_required": MIN_AUROC,
                           "passes": bool(r["excludes_chance"] and r["auroc"] >= MIN_AUROC)}
''')

code(r'''
# ============================================================
# CELL 8 — THE G2 VERDICT
# ============================================================
n_pass = sum(1 for v in G2.values() if v["passes"])
print("=" * 74)
print("GATE G2 — CONCEPT-EXTRACTOR VALIDITY")
print("=" * 74)
print(f'Rule: AUROC >= {MIN_AUROC} AND its 95% CI excludes chance, on the TEST split.')
print('Both conditions are required: at this sample size a CI can exclude chance at AUROC 0.52,')
print('which is significant but useless. AUPRC is shown against its own chance baseline (the')
print('positive prevalence), since the classes are imbalanced.\n')
for c, v in G2.items():
    print(f"  {c:<16} AUROC {v['auroc']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]  "
          f"AUPRC {v['auprc']:.4f} (chance {v['prevalence']:.3f})  "
          f"{'PASS' if v['passes'] else 'FAIL'}")

print("-" * 74)
if n_pass == 2:
    VERDICT = "PASS"
    print("  VERDICT: PASS — both validatable extractors beat chance with CIs excluding 0.5.")
    print("  Proceed to the bottleneck head (M13 re-wire). The full plan is live.")
elif n_pass == 1:
    VERDICT = "PARTIAL"
    weak = [c for c, v in G2.items() if not v["passes"]]
    print(f"  VERDICT: PARTIAL — {weak} did not clear chance.")
    print("  Per roadmap G2 'if not satisfied (a)': shrink to the minimal reliable concept set and")
    print("  proceed with a thinner bottleneck. Do NOT tune thresholds against the test split to")
    print("  rescue the failing one -- that would be fitting the gate.")
else:
    VERDICT = "FAIL"
    print("  VERDICT: FAIL — neither extractor beats chance.")
    print("  Per roadmap G2 'if not satisfied (b)': pivot to the reliability-only paper")
    print("  (physics-fragility + honest operating point + optional FM probing), which does not")
    print("  depend on clean concepts. This is a graceful degradation, not a dead end.")
print("=" * 74)

print("\n" + "-" * 74)
print("HUMAN BENCHMARK -- read the numbers above against these")
print("-" * 74)
print("  7 senior physicians, blind, on clean ICBHI audio (Tzeng et al., JMIR AI 2025):")
print("      ICBHI score 47.77%   sensitivity 23.23%   mean confidence 2.88/5")
print("  12 physicians, DETAILED adventitious-sound descriptions (Aviles-Solis 2016):")
print("      kappa < 0.40 (poor-to-fair)")
print("      ... same study, COMBINED categories: crackles 0.62, wheezes 0.59")
print()
print("  So the ceiling here is set substantially by LABEL RELIABILITY, not only by detector")
print("  quality -- seven senior physicians miss ~77% of the cycles ICBHI marks abnormal.")
print("  crackle_fine_ratio targets a distinction humans agree on at kappa < 0.40, which is why")
print("  it stays a proxy permanently. See Papers/HUMAN_BENCHMARKS.md.")
print()
print("  This is CONTEXT, NOT AN EXCUSE. A weak detector is still weak. Report both numbers")
print("  together; never cite the human benchmark alone as cover for a failing gate.")

print("\nNOTE ON THE OTHER 7 CONCEPTS")
print("-" * 74)
for k, (status, meaning) in CONCEPT_VALIDATION.items():
    if status != "validatable":
        print(f"  {k:<22} [{status:<11}] {meaning}")
print("\n  These have NO ground truth in ICBHI and are NOT validated by this notebook.")
print("  Report them as proxies. Promoting them requires the clinician labeling exercise")
print("  in CLINICIAN_LABELING_PACK.md (Gate G0).")
''')

# ===========================================================================
md(r"""
---
## Section 5 — Proxy concepts: description only, no accuracy claim

These plots exist so the proxies can be sanity-checked by eye and reported honestly. **None of
this is validation.** A proxy behaving plausibly across label groups is face validity, which is
weaker evidence than it looks — the point of the clinician exercise is to replace it.
""")

code(r'''
# ============================================================
# CELL 9 — PROXY BEHAVIOUR ACROSS ICBHI LABEL GROUPS (descriptive)
# ============================================================
groups = {"Normal": (df.crackle == 0) & (df.wheeze == 0),
          "Crackle": (df.crackle == 1) & (df.wheeze == 0),
          "Wheeze": (df.crackle == 0) & (df.wheeze == 1),
          "Both": (df.crackle == 1) & (df.wheeze == 1)}

print(f"{'concept':<22}" + "".join(f"{g:>12}" for g in groups) + "   status")
print("-" * 86)
for c in CONCEPT_NAMES:
    line = f"{c:<22}"
    for g, m in groups.items():
        line += f"{df.loc[m, c].median():>12.3f}"
    print(line + f"   {CONCEPT_VALIDATION[c][0]}")
print("\n(medians; descriptive only — no test is implied by this table)")

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, c in zip(axes.ravel(), CONCEPT_NAMES):
    data = [df.loc[m, c].dropna().values for m in groups.values()]
    # positional tick labels: matplotlib renamed boxplot(labels=) -> tick_labels= in 3.9,
    # and Colab's version is not pinned. set_xticklabels works on every version.
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(list(groups))
    status = CONCEPT_VALIDATION[c][0]
    ax.set_title(f"{c}\n[{status}]", fontsize=9,
                 color=("#1b7f3a" if status == "validatable" else "#c0392b"))
    ax.tick_params(labelsize=7)
fig.suptitle("Concept distributions by ICBHI label group — green = validatable, red = proxy",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(CFG["results_dir"], "concept_distributions.png"), dpi=150)
plt.show()
''')

# ===========================================================================
md(r"""
---
## Section 6 — CC1 score dump + results JSON
""")

code(r'''
# ============================================================
# CELL 10 — CC1 RAW SCORE DUMP (Protocol section 1, essential #10)
# ============================================================
# Canonical schema, byte-identical to Asif's/Statistics/owmtl_scores.py::SCHEMA so the paired
# DeLong / McNemar tooling can consume it directly. Without this no paired test is possible.
import csv as _csv

_SCORE_SCHEMA = ["model_id", "split", "unit_type", "unit_id", "score_name", "score", "label"]


def dump_scores(path, model_id, unit_type, unit_ids, scores, labels,
                score_name="score", split="test", append=False):
    assert unit_type in ("patient", "cycle", "recording"), f"bad unit_type {unit_type!r}"
    unit_ids = [str(u) for u in unit_ids]
    scores = [float(x) for x in scores]
    labels = [int(x) for x in labels]
    assert len(unit_ids) == len(scores) == len(labels), "length mismatch"
    assert len(set(labels)) > 1, "need both classes"
    _exists = os.path.exists(path) and append
    with open(path, "a" if append else "w", newline="") as _f:
        _w = _csv.writer(_f)
        if not _exists:
            _w.writerow(_SCORE_SCHEMA)
        for _u, _s, _y in zip(unit_ids, scores, labels):
            _w.writerow([model_id, split, unit_type, _u, score_name, f"{_s:.10g}", _y])
    return path


SCORES_CSV = os.path.join(CFG["results_dir"], "scores_M39.csv")
first = True
for concept, target in TARGETS.items():
    for split in ("test", "train"):
        sub = df[df.split == split]
        dump_scores(SCORES_CSV, "M39", "cycle", sub.unit_id.values, sub[concept].values,
                    sub[target].values, score_name=concept, split=split, append=not first)
        first = False

print(f"[CC1] wrote {sum(1 for _ in open(SCORES_CSV)) - 1} rows -> {SCORES_CSV}")
print("     unit_type='cycle' -- these are CYCLE-level scores and the tooling will refuse to")
print("     pair them with patient-level scores from M29/M38. That guard is deliberate.")
print("\nUse: python3 \"Asif's/Statistics/owmtl_scores.py\" scores_M39.csv <other>.csv")
''')

code(r'''
# ============================================================
# CELL 11 — RESULTS JSON
# ============================================================
te = df[df.split == "test"]
results = {
    "meta": {
        "model_id": "M39",
        "model_name": "Physics-Derived Acoustic Concept Extraction & Validation (Gate G2)",
        "contributor": CFG["contributor"],
        "date_completed": time.strftime("%Y-%m-%d"),
        "is_augmented": False, "augmentation_method": "none",
        "notes": (
            "Gate G2 evidence for the concept-bottleneck direction. Computes 9 physics-derived "
            "acoustic concepts per ICBHI cycle by DSP (no model, no training, no labels) and "
            "validates the two that ICBHI can validate. "
            "Lineage: refactors M35's VectorizedAcousticPhysicsLoss (spectral flatness, temporal "
            "PAPR) from a training regulariser on log-mels into standalone per-cycle measurements "
            "on the RAW WAVEFORM, and adds 7 more concepts. M35's reported number is not reused "
            "(70/30 split, legacy metric). "
            "IMPORTANT: only crackle_score and wheeze_score are validatable -- ICBHI labels "
            "crackle/wheeze PRESENCE only. The other 7 are proxies with no ground truth and are "
            "reported descriptively. Do not name a proxy after the clinical entity it proxies; "
            "promoting them requires the clinician exercise in CLINICIAN_LABELING_PACK.md (G0). "
            "Concepts are computed on each cycle's TRUE duration, never tiled to 8 s -- tiling "
            "would repeat crackles and fabricate crackle_rate_hz."),
    },
    "config": {k: v for k, v in CFG.items() if k != "results_dir"},
    "environment": {
        "platform": ("Google Colab" if IN_COLAB else
                     "Kaggle" if os.path.exists("/kaggle/working") else "Local"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    },
    "dataset_info": {
        "dataset": "ICBHI_2017", "data_source": "real_audio",
        "unit": "respiratory_cycle",
        "total_cycles": int(len(df)),
        "train_cycles": int((df.split == "train").sum()),
        "test_cycles": int(len(te)),
        "train_patients": int(df[df.split == "train"].patient_id.nunique()),
        "test_patients": int(te.patient_id.nunique()),
        "split_method": "official_60_40_patient_independent_corrected",
        "split_source": "official_file_patient_independent",
        "split_details": {"leaking_patients": LEAKING,
                          "test_recordings_official": n_test_before,
                          "test_recordings_used": n_test_after},
        "patient_leakage_verified": True,
        "cycles_skipped_too_short": int(skipped),
    },
    "efficiency": {
        "total_params": 0, "trainable_params": 0, "model_size_mb": 0.0,
        "training_time_total_s": 0,
        "extraction_time_total_s": round(float(EXTRACT_TIME), 1),
        "extraction_ms_per_cycle": round(float(EXTRACT_TIME / max(len(df), 1) * 1000), 3),
        "gpu_name": "none (pure DSP)",
        "inference_time_ms_per_sample": round(float(EXTRACT_TIME / max(len(df), 1) * 1000), 3),
        "note": "No model. Cost is DSP over the raw waveform.",
    },
    "best_epoch": {"epoch": None, "primary_metric": "concept_validation_auroc",
                   "primary_metric_value": (round(max(v["auroc"] for v in G2.values()), 4)
                                            if G2 else None)},
    "best_metrics": {
        "gate_g2": {
            "verdict": VERDICT,
            "rule": (f"AUROC >= {MIN_AUROC} AND 95% DeLong CI excludes chance, on the test "
                     f"split. The AUROC floor is required because at n~2600 a CI can exclude "
                     f"chance at AUROC 0.52 -- significant but practically useless."),
            "min_auroc_required": MIN_AUROC,
            "n_validatable": len(G2), "n_passing": int(n_pass),
            "per_concept": G2,
        },
        "concept_validation_status": {k: v[0] for k, v in CONCEPT_VALIDATION.items()},
        "concept_summary_test": {
            c: {"mean": round(float(te[c].mean()), 4), "std": round(float(te[c].std()), 4),
                "median": round(float(te[c].median()), 4)} for c in CONCEPT_NAMES},
        "human_benchmark_context": {
            "note": ("Interpret the G2 AUROCs against human performance on the same task, not "
                     "against 1.0. This is context for the reader, not a pass condition."),
            "reference_physicians_vs_icbhi": {
                "source": "Tzeng et al., JMIR AI 2025;4:e67239, Table 4 (7 senior physicians, "
                          "blind, clean ICBHI audio)",
                "icbhi_score": 0.4777, "sensitivity": 0.2323, "specificity": 0.7232,
                "accuracy": 0.4940, "mean_confidence_1_to_5": 2.88},
            "reference_physician_interobserver": {
                "source": "Aviles-Solis et al. 2016, PMID 27158515 (12 physicians, 20 ERS "
                          "recordings)",
                "kappa_detailed_descriptions": "<0.40 (poor to fair)",
                "kappa_combined_crackles": 0.62, "kappa_combined_wheezes": 0.59,
                "implication": ("crackle_fine_ratio targets a distinction physicians agree on at "
                                "kappa < 0.40, so it stays a proxy permanently regardless of how "
                                "the extractor performs.")},
        },
        "unvalidated_proxy_warning": (
            "crackle_fine_ratio, crackle_rate_hz, wheeze_pitch_hz, rhonchi_score and "
            "inspiratory_fraction have NO ground truth in ICBHI. Any paper text must name them as "
            "physics-derived proxies, not as the clinical entities."),
    },
    "ablation": {
        "ablation_group": "concept_extraction",
        "ablation_role": "baseline",
        "baseline_model_id": None,
        "variable_changed": "physics-derived DSP concepts computed per cycle (no model)",
        "variables_held_constant": ["data_split: official_60_40_patient_independent_corrected",
                                    "sample_rate: 16000", "seed: 42"],
        "known_deviations": [
            "Concepts use each cycle's true duration, not the shared 8 s tiled clip, because "
            "tiling repeats transients and fabricates crackle_rate_hz.",
            "Computed from the raw waveform rather than the shared 128-mel spectrogram: at "
            "hop_length=160 (10 ms) a mel frame cannot resolve a 5 ms crackle.",
        ],
        "component_flags": {
            "has_sound_event_head": False, "has_disease_head": False,
            "has_cross_task_consistency": False, "has_cqkd_regularization": False,
            "has_openmax_rejection": False, "owl_stage": 0, "compression_clusters": None,
        },
        "loss_weights": {"sound_event_weight": None, "disease_weight": None,
                         "consistency_weight": None},
    },
    "training_history": [],
    "raw_scores_file": "scores_M39.csv",
}

path = os.path.join(CFG["results_dir"], "results_M39.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {path}")
print(json.dumps(results["best_metrics"]["gate_g2"], indent=2)[:1200])

df.to_csv(os.path.join(CFG["results_dir"], "concepts_M39.csv"), index=False)
print(f"Wrote per-cycle concept table ({len(df)} rows) -> concepts_M39.csv")
''')

code(r'''
# ============================================================
# FINAL CELL — HANDOFF
# ============================================================
import shutil
try:
    from IPython.display import display, FileLink
except ImportError:
    display = FileLink = None

files = sorted(glob.glob(os.path.join(CFG["results_dir"], "*.json")) +
               glob.glob(os.path.join(CFG["results_dir"], "*.csv")) +
               glob.glob(os.path.join(CFG["results_dir"], "*.png")))
print("=" * 60)
for f in files:
    print(f"Ready: {os.path.basename(f):<28} ({round(os.path.getsize(f) / 1024 ** 2, 2)} MB)")
    if display is not None:
        display(FileLink(f))
if files:
    b = os.path.join(BASE_DIR, "M39_bundle"); os.makedirs(b, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(b, os.path.basename(f)))
    z = shutil.make_archive(os.path.join(BASE_DIR, "M39_handoff"), "zip", b)
    print(f"\nZIP: {z}")
    if display is not None:
        display(FileLink(z))
print("=" * 60)
''')

md(r"""
---
### Reading the output

1. **The G2 verdict in Cell 8 is the deliverable.** PASS → build the bottleneck head. PARTIAL →
   shrink to the reliable concept subset. FAIL → pivot to the reliability-only paper. All three
   are live branches in the roadmap; none is a dead end.
2. **Do not tune thresholds to make a failing concept pass.** The extractor constants are
   clinically motivated (CORSA) and were fixed before seeing any ICBHI result. Adjusting them
   against the test split would be fitting the gate rather than passing it — and the constants
   live in `concept_extractors.py`, so any change is visible in the diff.
3. **Expect the proxy plots to look plausible.** That is face validity, not evidence. The
   clinician exercise is what turns any of them into a validated concept.
4. Commit `scores_M39.csv` — cycle-level, so the tooling will refuse to pair it with M29/M38's
   patient-level scores. That guard is intentional.
""")

# ===========================================================================
NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12.13"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

for c in NB["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT = os.path.join(HERE, "M39_concept_extraction.ipynb")
with open(OUT, "w") as f:
    json.dump(NB, f, indent=1)
print(f"Wrote {OUT}  ({len(NB['cells'])} cells, extractors spliced from the module)")
