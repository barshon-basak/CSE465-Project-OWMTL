#!/usr/bin/env python3
"""
Generator for M43 - AST (Audio Spectrogram Transformer) backbone, run on Kaggle.

Per the workstream convention (Asif's/CLAUDE.md): notebooks are generated, not hand-edited.
Run `python gen_M43_kaggle.py` to emit M43_AST_kaggle.ipynb.

WHY M43 IS NOT RUN LOCALLY
    AST attends over 1212 patches - roughly 9x the token count of the 224x224 vision
    transformers in M40-M42 - and does not fit at the shared batch size of 16 on a 6 GB
    laptop card. Locally it would need batch 4, which makes it the one model in the set
    whose recipe differs. On a Kaggle P100/T4 (16 GB) it runs at batch 16 like the other
    three, so moving it there does not just save time, it removes the deviation.

WHAT THE NOTEBOOK MUST REPRODUCE EXACTLY
    Everything upstream of the backbone is copied verbatim from
    `Asif's/M45/m45_ablation.py` (which `m40_m43_transformers.py` imports): the corrected
    official 60/40 split with the 156/218 overlap reassigned to train, the 128-mel 8 s
    log-mel with cyclic padding and per-spectrogram min-max, the official ICBHI score, and
    the patient-level bootstrap CI. It is duplicated rather than imported because Kaggle
    has no access to the repo - the asserts on split sizes (920 recordings, 551 train) are
    what catch a drift between the two copies.

    The emitted JSON matches the M40-M42 schema, so downloading it into
    `M40_M43_transformers/Results/` is the whole integration step - `--summarise` picks it
    up with no special-casing.

Watch the nested-triple-quote trap: code cells are wrapped in r'''...''' so any Python
docstrings inside must use \"\"\"...\"\"\", never the reverse.
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


# ===========================================================================
md(r"""
# M43 — AST (Audio Spectrogram Transformer) backbone

**Model ID:** M43 · **Owner:** Asif · **Requirement:** RTK §3 (transformer #4 of 4), §7 (augmentation pair)

## Why this runs on Kaggle and the other three run locally

M40 (ViT-B/16), M41 (Swin-T) and M42 (DeiT-S) resize the log-mel to 224×224, which is 197
patch tokens. AST does not resize — it takes the mel patch at its native 1024×128 and
attends over **1212 tokens**. At the shared batch size of 16 that does not fit in the 6 GB
of a laptop RTX 4050; locally it would have to drop to batch 4, making it the one model in
the set running a different recipe.

A 16 GB P100/T4 runs it at batch 16 like the other three. So this is not only faster — it
is what keeps the four-model comparison honest.

## What must not drift

Everything upstream of the backbone is copied from `Asif's/M45/m45_ablation.py`:

| Stage | Setting |
|---|---|
| Split | official 60/40, patients 156 and 218 (on both sides) reassigned to train |
| Front end | 128 mel × 8 s, cyclic padding, `fmin` 50 / `fmax` 2000, per-spectrogram min-max |
| Loss | class-weighted CE, weights normalised to mean 1 |
| Optimiser | AdamW, lr 1e-4 backbone / 1e-3 head, cosine, 40 epochs, seed 42 |
| Metric | `icbhi_score_official` = (Se + Sp)/2, patient-level bootstrap CI |

The asserts on `920` recordings and `551` train recordings are the tripwire: if the Kaggle
copy of the ICBHI split ever differs from the repo's, the notebook stops instead of
quietly producing a number on a different split.

## Two runs, not one

`RUN_VARIANTS` below trains **clean** then **SpecAugment**. That pair is what requirement 7
needs — the augmentation delta per model, not one augmented number. If the session runs out
of GPU time after the first, set `RUN_VARIANTS = ["aug"]` and run it again; each variant
writes its own JSON and the second run skips what already exists in `/kaggle/working`.

## Expect it to lose to MobileNetV2

M4 already measured AST at ~24× the parameters and ~30× the latency of M2 for a worse score,
and the local M22_v2 baseline sits at **0.5602**. 920 recordings is not enough to fine-tune
an 86 M-parameter transformer. A loss here is the reportable accuracy/compute trade-off —
**do not tune it until it wins.** Report what it does.
""")

# ===========================================================================
md(r"""
## 1 · Setup

Add the dataset **`vbookshelf/respiratory-sound-database`** to the notebook (Add Data →
search "Respiratory Sound Database"), and switch the accelerator to **GPU T4 ×2 or P100**.

The next cell locates the audio folder and the official split file by search rather than by
a hard-coded path, because Kaggle's dataset layouts differ between mirrors.
""")

code(r"""
import glob, json, math, os, time, datetime
import numpy as np

# --- what to run -----------------------------------------------------------
RUN_VARIANTS = ["clean", "aug"]      # set to ["aug"] alone if the session times out first
OUT_DIR      = "/kaggle/working"
CACHE_DIR    = "/kaggle/working/cache"

# --- the shared recipe (identical to M40-M42; do not edit) -----------------
SR        = 16000
N_MELS    = 128
DURATION  = 8.0
EPOCHS    = 40
BATCH     = 16          # the full recipe batch - this is why M43 moved off the laptop
SEED      = 42
LR_BACKBONE, LR_HEAD, WD = 1e-4, 1e-3, 1e-4
AST_CKPT  = "MIT/ast-finetuned-audioset-10-10-0.4593"

CLASSES  = ["Normal", "Crackle", "Wheeze", "Both"]
LABEL_OF = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
N_FRAMES = 1 + int(SR * DURATION) // 160          # 801

# --- locate the dataset ----------------------------------------------------
def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"could not find {what} with {pattern!r}. Add the dataset "
            f"'vbookshelf/respiratory-sound-database' to this notebook.")
    return hits[0]

AUDIO_DIR  = os.path.dirname(find_one("/kaggle/input/**/audio_and_txt_files/*.wav", "the audio"))
SPLIT_FILE = find_one("/kaggle/input/**/ICBHI_challenge_train_test.txt", "the official split")
os.makedirs(CACHE_DIR, exist_ok=True)

print("audio :", AUDIO_DIR)
print("split :", SPLIT_FILE)
print("wavs  :", len(glob.glob(os.path.join(AUDIO_DIR, "*.wav"))))
""")

# ===========================================================================
md(r"""
## 2 · Split and front end — copied verbatim from `Asif's/M45/m45_ablation.py`

Two things here exist because the project got them wrong before, and both are documented in
the repo README:

1. The "official split" was once a `pid <= 111` fallback that left 11 test patients. The
   assert on **920 recordings / 551 train** is what stops that silently recurring.
2. The real official split is **not patient-independent** — patients 156 and 218 appear on
   both sides. They are reassigned to train, which is the repo's default policy.

A failed audio read **raises** instead of substituting a zero spectrogram, per
`Model_Training_Protocol.md` §1.2 — a silent all-zero cycle would be trained on and scored
as if it were real.
""")

code(r"""
def corrected_split_index(audio_dir, split_file):
    split = {}
    for line in open(split_file):
        t = line.replace("\t", " ").replace(",", " ").split()
        if len(t) >= 2 and t[1].lower() in ("train", "test"):
            split[t[0].replace(".wav", "")] = t[1].lower()
    assert len(split) == 920, f"{len(split)} != 920 recordings"

    pid = lambda s: int(s.split("_")[0])
    sides = {}
    for s, v in split.items():
        sides.setdefault(pid(s), set()).add(v)
    overlap = {p for p, v in sides.items() if len(v) > 1}
    print("patients on both sides, reassigned to train:", sorted(overlap))
    split = {s: ("train" if pid(s) in overlap else v) for s, v in split.items()}
    assert sum(1 for v in split.values() if v == "train") == 551, "corrected split wrong"

    rows = []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(audio_dir, stem + ".txt")
        if stem not in split or not os.path.exists(txt):
            continue
        for line in open(txt):
            p = line.split()
            if len(p) >= 4:
                rows.append({"wav": wav, "stem": stem, "patient_id": pid(stem),
                             "start": float(p[0]), "end": float(p[1]),
                             "label": LABEL_OF[(int(p[2]), int(p[3]))],
                             "split": split[stem]})
    return rows


def log_mel(wav, start, end):
    import librosa
    n_samples = int(SR * DURATION)
    try:
        a, _ = librosa.load(wav, sr=SR, offset=start, duration=max(end - start, 0.05),
                            mono=True)
    except Exception as e:
        raise RuntimeError(f"failed to load {wav}") from e
    if len(a) == 0:
        raise RuntimeError(f"empty audio decoded from {wav}")

    if len(a) < n_samples:
        a = np.tile(a, math.ceil(n_samples / len(a)))[:n_samples]   # cyclic padding
    else:
        a = a[:n_samples]

    m = librosa.feature.melspectrogram(y=a, sr=SR, n_mels=N_MELS, n_fft=1024,
                                       hop_length=160, win_length=400, fmin=50,
                                       fmax=2000, power=2.0)
    lm = librosa.power_to_db(m, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)             # per-spectrogram min-max
    T = lm.shape[1]
    lm = np.pad(lm, ((0, 0), (0, N_FRAMES - T))) if T < N_FRAMES else lm[:, :N_FRAMES]
    return lm[None].astype(np.float32)


def build_cache(rows, tag):
    path = os.path.join(CACHE_DIR, f"{tag}_{N_MELS}m_{DURATION}s_wrap_1.npy")
    shape = (len(rows), 1, N_MELS, N_FRAMES)
    if os.path.exists(path):
        return np.memmap(path, dtype=np.float16, mode="r", shape=shape)
    print(f"  building cache {os.path.basename(path)} ({len(rows)} cycles) ...")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
    for i, r in enumerate(rows):
        mm[i] = log_mel(r["wav"], r["start"], r["end"]).astype(np.float16)
        if (i + 1) % 1500 == 0:
            print(f"    {i+1}/{len(rows)}")
    mm.flush()
    return np.memmap(path, dtype=np.float16, mode="r", shape=shape)


rows = corrected_split_index(AUDIO_DIR, SPLIT_FILE)
tr = [r for r in rows if r["split"] == "train"]
te = [r for r in rows if r["split"] == "test"]
print(f"cycles: {len(tr)} train / {len(te)} test")
print("test class counts:", np.bincount([r['label'] for r in te], minlength=4))
""")

# ===========================================================================
md(r"""
## 3 · Metrics

`icbhi_score_official` is the ICBHI 2017 challenge score, **(Se + Sp)/2** with Se over the
three abnormal classes pooled. The project-internal `(recall_macro + specificity_macro)/2`
runs ~0.11 higher and is not comparable to published work — it is deliberately not computed
here.

The CI resamples **patients**, not cycles, because cycles from one patient are not
independent.
""")

code(r"""
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def official(cm):
    cm = np.asarray(cm, float)
    sp  = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se  = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else float("nan")
    return float((se + sp) / 2), float(se), float(sp)


def patient_ci(y, pred, pid, n_boot=2000, seed=42):
    uq = np.unique(pid); ix = {p: np.flatnonzero(pid == p) for p in uq}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        s = np.concatenate([ix[p] for p in rng.choice(uq, len(uq), replace=True)])
        x = official(confusion_matrix(y[s], pred[s], labels=[0, 1, 2, 3]))[0]
        if x == x:
            v.append(x)
    return [round(float(z), 4) for z in np.percentile(v, [2.5, 97.5])]
""")

# ===========================================================================
md(r"""
## 4 · The model

`ASTForAudioClassification` with a fresh 4-way head. AST expects `(batch, frames, mels)` —
the cache is `(batch, 1, mels, frames)`, so it is squeezed and transposed, then padded from
801 frames to AST's `max_length` of 1024.

AST's own recipe normalises with the **training-set** mean and std and doubles the std,
which targets mean 0 / std 0.5. The AudioSet constants do not apply here because the input
is already min-max scaled by the shared front end, so the statistics are estimated from a
1-in-20 subsample of the training cache instead.

SpecAugment uses the same masks as M22/M45 (2 frequency masks ≤ 24 bins, 2 time masks ≤ 80
frames), so the augmentation delta is attributable to augmentation and nothing else.
""")

code(r"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ASTForAudioClassification


def specaug(x):
    for _ in range(2):
        f = int(torch.randint(0, 25, (1,)).item())
        if 0 < f < x.shape[1]:
            f0 = int(torch.randint(0, x.shape[1] - f + 1, (1,)).item()); x[:, f0:f0+f, :] = 0
    for _ in range(2):
        t = int(torch.randint(0, 81, (1,)).item())
        if 0 < t < x.shape[2]:
            t0 = int(torch.randint(0, x.shape[2] - t + 1, (1,)).item()); x[:, :, t0:t0+t] = 0
    return x


class DS(Dataset):
    def __init__(s, X, y, aug): s.X, s.y, s.aug = X, y, aug
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        x = torch.from_numpy(np.asarray(s.X[i], np.float32))
        return (specaug(x.clone()) if s.aug else x), torch.tensor(int(s.y[i]))


# (B, 1, mels, frames) -> (B, frames, mels) -> AST -> 4 logits.
class ASTWrapper(nn.Module):
    def __init__(s, net, mean, std, max_len):
        super().__init__()
        s.net, s.max_len = net, max_len
        s.register_buffer("mu", torch.tensor(float(mean)))
        s.register_buffer("sd", torch.tensor(float(std)))

    def forward(s, x):
        x = x.squeeze(1).transpose(1, 2)
        if x.shape[1] < s.max_len:
            x = F.pad(x, (0, 0, 0, s.max_len - x.shape[1]))
        else:
            x = x[:, :s.max_len]
        return s.net((x - s.mu) / (2 * s.sd)).logits


Xtr, Xte = build_cache(tr, "train"), build_cache(te, "test")
ytr = np.array([r["label"] for r in tr]); yte = np.array([r["label"] for r in te])
pid = np.array([r["patient_id"] for r in te])

_sub = np.asarray(Xtr[::20], np.float32)
AST_MEAN, AST_STD = float(_sub.mean()), float(_sub.std()) or 1.0
print(f"AST normalisation from train subsample: mean {AST_MEAN:.4f} std {AST_STD:.4f}")
""")

# ===========================================================================
md(r"""
## 5 · Train

Best epoch is selected on the official **test** score. That is test-set peeking and is not
defensible on its own — it is done here **only** because M22_v2 and every M45 ablation row
do it, and changing it for M43 alone would make the number non-comparable to the table it
has to sit in. Any M43 figure quoted outside that table must repeat this caveat.

Roughly 2–3 minutes per epoch on a P100, so ~1.5–2 h per variant. If the session dies
mid-way, re-running the notebook skips whichever variant already wrote its JSON.
""")

code(r"""
def run(aug):
    tag = "M43_aug" if aug else "M43"
    out_json = os.path.join(OUT_DIR, f"results_{tag}.json")
    if os.path.exists(out_json):
        print(f"{tag}: already done, skipping"); return json.load(open(out_json))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}\n  {tag}: AST ({'SpecAugment' if aug else 'clean'})  on {dev}\n{'='*70}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    dl_tr = DataLoader(DS(Xtr, ytr, aug), batch_size=BATCH, shuffle=True, num_workers=2)
    dl_te = DataLoader(DS(Xte, yte, False), batch_size=BATCH, num_workers=2)

    net = ASTForAudioClassification.from_pretrained(
        AST_CKPT, num_labels=4, ignore_mismatched_sizes=True)
    model = ASTWrapper(net, AST_MEAN, AST_STD, net.config.max_length).to(dev)

    head, backbone = [], []
    for n, p in model.named_parameters():
        (head if n.startswith("net.classifier") else backbone).append(p)
    assert head, "no head parameters matched"
    opt = torch.optim.AdamW([{"params": backbone, "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}], weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    c = np.maximum(np.bincount(ytr, minlength=4).astype(float), 1)
    w = c.sum() / (4 * c); w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    best, best_pred, best_ep, t0, ep_times = -1.0, None, 0, time.time(), []
    for ep in range(1, EPOCHS + 1):
        te0 = time.time(); model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sch.step(); ep_times.append(time.time() - te0)

        model.eval(); pr = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            for x, _ in dl_te:
                pr.append(model(x.to(dev)).argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        sc = official(confusion_matrix(yte, pr, labels=[0, 1, 2, 3]))[0]
        if sc > best:
            best, best_pred, best_ep = sc, pr, ep
        print(f"    ep {ep:02d}/{EPOCHS}  official {sc:.4f}  (best {best:.4f})  "
              f"{ep_times[-1]:.0f}s/ep", flush=True)

    cm = confusion_matrix(yte, best_pred, labels=[0, 1, 2, 3])
    sc, se, sp = official(cm)
    n_par = int(sum(p.numel() for p in model.parameters()))
    doc = {
        "meta": {"model_id": tag, "architecture": "AST", "owner": "Asif",
                 "model_type": "transformer", "pretrained_source": AST_CKPT,
                 "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                 "is_augmented": bool(aug),
                 "notes": "RTK req 2/3/7 transformer backbone. Trained on Kaggle because "
                          "AST's 1212 patches do not fit at batch 16 on the local 6 GB "
                          "card; recipe, split and front end are otherwise identical to "
                          "M40-M42. Best epoch selected on the official test score, as in "
                          "M45."},
        "config": {"n_mels": N_MELS, "duration_s": DURATION, "padding": "wrap",
                   "minmax": True, "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED,
                   "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD, "weight_decay": WD,
                   "optimizer": "AdamW", "scheduler": "cosine", "class_weighted": True,
                   "specaug": bool(aug), "pretrained": True},
        "dataset_info": {"dataset": "ICBHI_2017",
                         "split_method": "official_60_40_patient_independent_corrected",
                         "train_samples": len(tr), "test_samples": len(te),
                         "test_patients": int(len(np.unique(pid)))},
        "best_epoch": {"epoch": best_ep, "primary_metric": "icbhi_score_official",
                       "primary_metric_value": round(sc, 4)},
        "best_metrics": {"icbhi_score_official": round(sc, 4),
                         "icbhi_score_official_ci95": patient_ci(yte, best_pred, pid),
                         "icbhi_se_official": round(se, 4),
                         "icbhi_sp_official": round(sp, 4),
                         "accuracy": round(float(accuracy_score(yte, best_pred)), 4),
                         "f1_macro": round(float(f1_score(yte, best_pred, average="macro",
                                                          zero_division=0)), 4),
                         "confusion_matrix_raw": cm.tolist()},
        "efficiency": {"total_params": n_par,
                       "trainable_params": int(sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad)),
                       "model_size_mb": round(n_par * 4 / 1024 ** 2, 2),
                       "s_per_epoch": round(float(np.mean(ep_times)), 1),
                       "training_time_total_s": round(time.time() - t0, 1),
                       "gpu_name": (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else "cpu")},
    }
    torch.save({"epoch": int(best_ep), "best_score": float(best),
                "model_state": model.state_dict(), "model_id": tag},
               os.path.join(OUT_DIR, f"best_{tag}.pth"))
    json.dump(doc, open(out_json, "w"), indent=2)
    np.save(os.path.join(OUT_DIR, f"preds_{tag}.npy"),
            {"y_true": yte, "y_pred": best_pred, "patient_id": pid}, allow_pickle=True)
    print(f"    -> official {sc:.4f} {doc['best_metrics']['icbhi_score_official_ci95']}  "
          f"Se {se:.4f} Sp {sp:.4f}  params {n_par/1e6:.1f}M")
    return doc


results = {("M43_aug" if v == "aug" else "M43"): run(v == "aug") for v in RUN_VARIANTS}
""")

# ===========================================================================
md(r"""
## 6 · Result and what to bring back

Sanity-check the confusion matrix before trusting the score. **A row of zeros in every
abnormal class means the run collapsed to all-Normal** — that scores ~0.50 by predicting
one class and is a broken run, not a legitimate result. Report it as a collapse if it
happens; do not report the 0.50.
""")

code(r"""
for tag, d in results.items():
    bm, ef = d["best_metrics"], d["efficiency"]
    print(f"\n{tag}  ({'SpecAugment' if d['meta']['is_augmented'] else 'clean'})")
    print(f"  official  {bm['icbhi_score_official']:.4f}  CI95 {bm['icbhi_score_official_ci95']}"
          f"   (M22_v2 baseline 0.5602)")
    print(f"  Se {bm['icbhi_se_official']:.4f}  Sp {bm['icbhi_sp_official']:.4f}  "
          f"acc {bm['accuracy']:.4f}  F1 {bm['f1_macro']:.4f}   best epoch "
          f"{d['best_epoch']['epoch']}/{d['config']['epochs']}")
    print(f"  {ef['total_params']/1e6:.1f}M params  {ef['s_per_epoch']:.0f}s/epoch")
    print("  confusion matrix (rows = true Normal/Crackle/Wheeze/Both):")
    for r in bm["confusion_matrix_raw"]:
        print("   ", r)
    if all(sum(r[1:]) == 0 for r in bm["confusion_matrix_raw"][1:]):
        print("  !! COLLAPSED to all-Normal - report as a collapse, not as a score")

if len(results) == 2:
    d = (results["M43_aug"]["best_metrics"]["icbhi_score_official"]
         - results["M43"]["best_metrics"]["icbhi_score_official"])
    print(f"\nSpecAugment delta: {d:+.4f}")

print("\nfiles written to /kaggle/working:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith(("results_M43", "preds_M43", "best_M43")):
        print(f"  {f}  ({os.path.getsize(os.path.join(OUT_DIR, f))/1e6:.1f} MB)")
""")

# ===========================================================================
md(r"""
### Back in the repo

Download `results_M43.json`, `results_M43_aug.json` and the two `preds_M43*.npy` from the
notebook output, drop them into `M40_M43_transformers/Results/`, then:

```
python m40_m43_transformers.py --summarise
```

They land in the same table as M40–M42 with no special-casing — the schema is identical.
The `.pth` checkpoints are ~330 MB each; keep them out of git and link them the way
`Barshon's/M4/Best_model_pth_file Link.txt` does.

Then update the M43 row in `Model_Training_Reference.md` from *built, not run* to the
result, and re-run `python "Asif's/audit/audit_project.py"`.
""")

# ===========================================================================
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.13"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Notebook JSON stores `source` as a list of lines, each keeping its trailing newline
# except the last. Splitting with keepends preserves that exactly.
for c in NB["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "M43_AST_kaggle.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1)
print(f"Wrote {OUT}  ({len(NB['cells'])} cells)")
