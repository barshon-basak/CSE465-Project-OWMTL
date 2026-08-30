"""
M45 — Component + preprocessing ablation on the best model (RTK requirements 10 AND 1)

BASELINE: M22_v2 — MobileNetV2 + SpecAugment, official ICBHI 0.5602 [0.5081, 0.6137] on
`official_60_40_patient_independent_corrected`. One variable changes per row; everything
else is held at the M22_v2 recipe.

WHY ONE SCRIPT FOR TWO REQUIREMENTS
    RTK_requirements.md section 10 says so directly: "Merge M45 and M46 into one ablation
    section in the report; they are the same kind of evidence."

THE THREE MISSING STAGES ARE NOW ROWS, BUT THEY ADD RATHER THAN REMOVE
    RTK section 3b lists six preprocessing rows. Three of them ablate stages that did not
    exist in this pipeline: an explicit band-pass filter, denoising, and per-cycle
    amplitude normalisation. Section 3a asks for those stages to be written first.

    They are now implemented (`bandpass`, `denoise`, `ampnorm` in the config), and they are
    OFF in BASE. That is deliberate. Turning any of them on in BASE would change A0, and
    every delta in this table is measured against A0 — a silent baseline change would
    invalidate the seven rows already run and the paper table built from them.

    So P1-P3 are ADD rows, not REMOVE rows: each turns one stage on and reports what it
    buys. Read their delta in the opposite direction to A1-A6 and P4-P5. A positive delta
    on an ADD row is a recommendation to adopt the stage; a negative one is evidence that
    the stage is not missing from this pipeline by oversight but by merit.

A ROW THE SPEC DID NOT ASK FOR, ADDED ON EVIDENCE
    P4_zeropad. M44 measured attribution consistency across the wrap-padded repetitions of
    each cycle and found it low (mean r=0.098; only 32% of cycles above r=0.5). Every ICBHI
    cycle is shorter than the 8 s input (median 2.42 s), so the pipeline tiles it ~3x with
    `np.tile` — a hard concatenation that leaves a discontinuity at every seam.

    Cyclic padding to 8 s is standard practice on ICBHI, so this is not an idiosyncratic
    choice; but padding-driven shortcut learning is a documented failure mode in audio
    classification, and some published implementations tile WITH a fade in/out precisely to
    avoid seam artefacts. This row tests whether the padding scheme is doing work the
    acoustics should be doing. It is the one row motivated by a measurement rather than by
    a checklist.

ROWS
    A0  baseline            = M22_v2 (already run, reused)
    A1  - SpecAugment       = M3_v2  (already run, reused)
    A2  - ImageNet init     random initialisation
    A3  - class weighting   plain unweighted CE
    A4  frozen backbone     train the classifier only
    A5  64 mels             half the frequency resolution
    A6  4 s cycles          half the temporal context
    P4  zero-padding        pad with silence instead of tiling the cycle
    P5  - min-max norm      drop the per-spectrogram [0,1] rescale

USAGE
    python m45_ablation.py --row A2                 # one row
    python m45_ablation.py --all                    # every unrun row, sequentially
    python m45_ablation.py --summarise              # build the table from what exists
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]
LABEL_OF = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
SR = 16000

BASE = dict(n_mels=128, duration_s=8.0, padding="wrap", minmax=True,
            bandpass=False, denoise=False, ampnorm=False,
            pretrained=True, freeze=False, class_weighted=True, specaug=True,
            lr=5e-4, batch_size=16, epochs=40, dropout=0.3, seed=42)

ROWS = {
    "A2": dict(pretrained=False, _desc="- ImageNet pre-training (random init)"),
    "A3": dict(class_weighted=False, _desc="- class-weighted loss (plain CE)"),
    "A4": dict(freeze=True, _desc="frozen backbone, classifier only"),
    "A5": dict(n_mels=64, _desc="64 mels instead of 128"),
    "A6": dict(duration_s=4.0, _desc="4 s cycles instead of 8 s"),
    "P4": dict(padding="zero", _desc="zero-padding instead of cyclic tiling"),
    "P5": dict(minmax=False, _desc="- per-spectrogram min-max normalisation"),
    # ADD rows - see the docstring. Delta reads in the opposite direction to the rows above.
    "P1": dict(bandpass=True, _desc="+ band-pass filter (50-2000 Hz Butterworth)"),
    "P2": dict(denoise=True, _desc="+ spectral-gating denoising"),
    "P3": dict(ampnorm=True, _desc="+ per-cycle peak amplitude normalisation"),
}
ADD_ROWS = {"P1", "P2", "P3"}
REUSED = {
    "A0": (os.path.join(REPO, "Asif's", "M22_v2", "Results", "results_M22_v2.json"),
           "baseline (M22_v2): MobileNetV2 + SpecAugment"),
    "A1": (os.path.join(REPO, "Asif's", "M3_v2", "Results", "results_M3_v2.json"),
           "- SpecAugment (M3_v2)"),
}
NOT_APPLICABLE = {}   # P1-P3 were the entries here; they are ADD rows now (see docstring)


# ---------------------------------------------------------------- data
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


def butter_bandpass(a, lo=50.0, hi=2000.0, order=4):
    """Zero-phase 4th-order Butterworth band-pass (RTK section 3a stage 2).

    filtfilt, not lfilter: a causal filter shifts transients in time, and crackles are
    transients whose timing is the thing being measured.
    """
    from scipy.signal import butter, filtfilt
    nyq = SR / 2.0
    hi = min(hi, nyq * 0.999)
    b, a_ = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a_, a).astype(np.float32)


def peak_normalise(a):
    """Per-cycle peak normalisation (RTK section 3a stage 5) - removes device gain."""
    m = float(np.max(np.abs(a)))
    return (a / m).astype(np.float32) if m > 1e-8 else a


def spectral_gate(power, floor_pct=25.0, over=2.0):
    """Spectral-subtraction denoising on the power spectrogram (RTK section 3a stage 6).

    The noise floor is estimated per frequency bin as a low percentile over time, which
    assumes the noise is stationary within a cycle and the adventitious sound is not.
    Subtracted with an over-subtraction factor and floored at zero rather than at a small
    epsilon, so a bin that is pure noise becomes silent instead of becoming a small
    positive number the log stretches back into visible structure.

    The percentile and the factor are tied together and were set by the self-test, not by
    taste. Spectrogram power in a stationary bin is roughly exponential, so the 25th
    percentile sits near 0.29 of the mean; over=2.0 therefore subtracts a little over half
    the expected floor. An earlier setting (10th percentile, over=1.5) subtracted only
    about 0.16 of it — under a third as much — and would have shipped as a stage too weak
    to show up against the between-epoch noise in the ablation deltas. Pushing harder is
    possible but buys musical noise, which a CNN can learn as texture, so this stays
    deliberately conservative.
    """
    noise = np.percentile(power, floor_pct, axis=1, keepdims=True)
    return np.maximum(power - over * noise, 0.0)


def log_mel(wav, start, end, cfg):
    """The M22 preprocessing, with every ablatable stage switchable."""
    import librosa
    n_mels, dur = cfg["n_mels"], cfg["duration_s"]
    n_samples = int(SR * dur)
    n_frames = 1 + n_samples // 160
    try:
        a, _ = librosa.load(wav, sr=SR, offset=start, duration=max(end - start, 0.05),
                            mono=True)
    except Exception as e:
        # A silent all-zero spectrogram here would be trained on and
        # scored as a real cycle. Fail instead of substituting
        # (Model_Training_Protocol.md section 1.2).
        raise RuntimeError("failed to load audio") from e
    if len(a) == 0:
        # Empty decode is a failed read, not a silent zero cycle.
        raise RuntimeError(f"empty audio decoded from {wav}")

    if cfg.get("bandpass"):
        a = butter_bandpass(a)

    if len(a) < n_samples:
        if cfg["padding"] == "wrap":                       # tile the cycle (the default)
            a = np.tile(a, math.ceil(n_samples / len(a)))[:n_samples]
        else:                                              # pad with silence
            a = np.pad(a, (0, n_samples - len(a)))
    else:
        a = a[:n_samples]

    if cfg.get("ampnorm"):
        a = peak_normalise(a)

    if cfg.get("denoise"):
        # Gate in the linear STFT domain, then project through the mel filterbank, so the
        # subtraction happens at the resolution the noise was estimated at.
        spec = np.abs(librosa.stft(a, n_fft=1024, hop_length=160, win_length=400)) ** 2
        m = librosa.feature.melspectrogram(S=spectral_gate(spec), sr=SR, n_mels=n_mels,
                                           n_fft=1024, fmin=50, fmax=2000)
    else:
        m = librosa.feature.melspectrogram(y=a, sr=SR, n_mels=n_mels, n_fft=1024,
                                           hop_length=160, win_length=400, fmin=50,
                                           fmax=2000, power=2.0)
    lm = librosa.power_to_db(m, ref=np.max)
    if cfg["minmax"]:
        lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    else:
        lm = lm / 80.0 + 1.0                               # keep dB roughly in [0,1]
    T = lm.shape[1]
    lm = np.pad(lm, ((0, 0), (0, n_frames - T))) if T < n_frames else lm[:, :n_frames]
    return lm[None].astype(np.float32)


def build_cache(rows, cfg, tag):
    # Every stage that changes the pixels must be in the key. A cache hit on a stale key
    # would train the new row on the old row's spectrograms and report it as a result.
    key = (f"{cfg['n_mels']}m_{cfg['duration_s']}s_{cfg['padding']}_{int(cfg['minmax'])}"
           f"_bp{int(cfg.get('bandpass', False))}_dn{int(cfg.get('denoise', False))}"
           f"_an{int(cfg.get('ampnorm', False))}")
    n_frames = 1 + int(SR * cfg["duration_s"]) // 160
    path = os.path.join(HERE, "cache", f"{tag}_{key}.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shape = (len(rows), 1, cfg["n_mels"], n_frames)
    if os.path.exists(path):
        return np.memmap(path, dtype=np.float16, mode="r", shape=shape)
    print(f"    building cache {os.path.basename(path)} ...")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
    for i, r in enumerate(rows):
        mm[i] = log_mel(r["wav"], r["start"], r["end"], cfg).astype(np.float16)
        if (i + 1) % 1500 == 0:
            print(f"      {i+1}/{len(rows)}")
    mm.flush()
    return np.memmap(path, dtype=np.float16, mode="r", shape=shape)


# ---------------------------------------------------------------- metrics
def official(cm):
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else float("nan")
    return float((se + sp) / 2), float(se), float(sp)


def patient_ci(y, pred, pid, n_boot=2000, seed=42):
    from sklearn.metrics import confusion_matrix
    uq = np.unique(pid); ix = {p: np.flatnonzero(pid == p) for p in uq}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        s = np.concatenate([ix[p] for p in rng.choice(uq, len(uq), replace=True)])
        x = official(confusion_matrix(y[s], pred[s], labels=[0, 1, 2, 3]))[0]
        if x == x:
            v.append(x)
    return [round(float(z), 4) for z in np.percentile(v, [2.5, 97.5])]


# ---------------------------------------------------------------- run one row
def run_row(row_id, rows, args):
    import torch, torch.nn as nn, torchvision
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

    cfg = dict(BASE); cfg.update({k: v for k, v in ROWS[row_id].items() if not k.startswith("_")})
    desc = ROWS[row_id]["_desc"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_frames = 1 + int(SR * cfg["duration_s"]) // 160
    print(f"\n{'='*74}\n  {row_id}: {desc}\n{'='*74}")

    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr, Xte = build_cache(tr, cfg, "train"), build_cache(te, cfg, "test")
    ytr = np.array([r["label"] for r in tr]); yte = np.array([r["label"] for r in te])
    pid = np.array([r["patient_id"] for r in te])

    def spec_aug(x):
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
            return (spec_aug(x.clone()) if s.aug else x), torch.tensor(int(s.y[i]))

    dl_tr = DataLoader(DS(Xtr, ytr, cfg["specaug"]), batch_size=cfg["batch_size"],
                       shuffle=True, num_workers=0)
    dl_te = DataLoader(DS(Xte, yte, False), batch_size=cfg["batch_size"], num_workers=0)

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            w = "IMAGENET1K_V1" if cfg["pretrained"] else None
            s.features = torchvision.models.mobilenet_v2(weights=w).features
            if cfg["freeze"]:
                for p in s.features.parameters():
                    p.requires_grad = False
            s.gap = nn.AdaptiveAvgPool2d((1, 1)); s.dropout = nn.Dropout(cfg["dropout"])
            s.classifier = nn.Linear(1280, 4); s.norm = cfg["pretrained"]
        def forward(s, x):
            x = x.repeat(1, 3, 1, 1)
            if s.norm:
                x = (x - mean.to(x.device)) / std.to(x.device)
            return s.classifier(s.dropout(s.gap(s.features(x)).flatten(1)))

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model = Net().to(dev)
    if cfg["class_weighted"]:
        c = np.maximum(np.bincount(ytr, minlength=4).astype(float), 1)
        w = c.sum() / (4 * c); w = w / w.mean()
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best, best_pred, best_ep, t0 = -1.0, None, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        sch.step()
        model.eval(); pr = []
        with torch.no_grad():
            for x, _ in dl_te:
                pr.append(model(x.to(dev)).argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        sc = official(confusion_matrix(yte, pr, labels=[0, 1, 2, 3]))[0]
        if sc > best:
            best, best_pred, best_ep = sc, pr, ep
        if ep % 10 == 0 or ep == cfg["epochs"]:
            print(f"    ep {ep:02d}/{cfg['epochs']}  official {sc:.4f}  (best {best:.4f})")

    cm = confusion_matrix(yte, best_pred, labels=[0, 1, 2, 3])
    sc, se, sp = official(cm)
    doc = {"meta": {"model_id": f"M45_{row_id}", "row": row_id, "variable_changed": desc,
                    "baseline_model_id": "M22_v2",
                    "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "is_augmented": bool(cfg["specaug"]),
                    "notes": "M45 ablation row. One variable changed from the M22_v2 recipe."},
           "config": {k: v for k, v in cfg.items()},
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
           "efficiency": {"total_params": int(sum(p.numel() for p in model.parameters())),
                          "trainable_params": int(sum(p.numel() for p in model.parameters()
                                                      if p.requires_grad)),
                          "training_time_total_s": round(time.time() - t0, 1),
                          "gpu_name": (torch.cuda.get_device_name(0)
                                       if torch.cuda.is_available() else "cpu")},
           "ablation": {"ablation_group": "component_and_preprocessing_ablation",
                        "ablation_role": "variant", "baseline_model_id": "M22_v2",
                        "variable_changed": desc}}
    # Save the checkpoint. M44's attribution measures can then be re-run on any ablation
    # variant — which is the whole point of the P4 padding row: does removing the tiling
    # also remove the position-keying artefact? Without the weights that cannot be asked.
    torch.save({"epoch": int(best_ep), "best_score": float(best),
                "model_state": model.state_dict(), "cfg": cfg, "row": row_id},
               os.path.join(HERE, f"best_M45_{row_id}.pth"))
    json.dump(doc, open(os.path.join(HERE, f"results_M45_{row_id}.json"), "w"), indent=2)
    np.save(os.path.join(HERE, f"preds_M45_{row_id}.npy"),
            {"y_true": yte, "y_pred": best_pred, "patient_id": pid}, allow_pickle=True)
    print(f"    -> official {sc:.4f} {doc['best_metrics']['icbhi_score_official_ci95']}  "
          f"Se {se:.4f} Sp {sp:.4f}  ({time.time()-t0:.0f}s)")
    return doc


def summarise():
    rows = []
    for rid, (path, desc) in REUSED.items():
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8")); bm = d["best_metrics"]
            rows.append((rid, desc, bm["icbhi_score_official"],
                         bm.get("icbhi_score_official_ci95"), bm["icbhi_se_official"],
                         bm["icbhi_sp_official"], bm["f1_macro"]))
    for f in sorted(glob.glob(os.path.join(HERE, "results_M45_*.json"))):
        d = json.load(open(f, encoding="utf-8")); bm = d["best_metrics"]
        rows.append((d["meta"]["row"], d["meta"]["variable_changed"],
                     bm["icbhi_score_official"], bm["icbhi_score_official_ci95"],
                     bm["icbhi_se_official"], bm["icbhi_sp_official"], bm["f1_macro"]))
    rows.sort(key=lambda r: r[0])
    base = next((r[2] for r in rows if r[0] == "A0"), None)

    print("\n" + "=" * 100)
    print("  M45 — ABLATION TABLE (baseline A0 = M22_v2, corrected official split)")
    print("=" * 100)
    print(f"  {'row':4s} {'variable changed':44s} {'official':>8s} {'delta':>8s} "
          f"{'Se':>6s} {'Sp':>6s} {'F1':>6s}")
    print("  " + "-" * 96)
    for rid, desc, sc, ci, se, sp, f1 in rows:
        dl = f"{sc-base:+.4f}" if base is not None and rid != "A0" else "—"
        mark = " (add)" if rid in ADD_ROWS else ""
        print(f"  {rid:4s} {desc[:44]:44s} {sc:8.4f} {dl:>8s} {se:6.3f} {sp:6.3f} {f1:6.3f}{mark}")
    print("  " + "-" * 96)
    for rid, why in NOT_APPLICABLE.items():
        print(f"  {rid:4s} NOT APPLICABLE — {why}")
    print("=" * 100)
    json.dump({"baseline": "A0 (M22_v2)", "split": "official_60_40_patient_independent_corrected",
               "rows": [{"row": r[0], "variable_changed": r[1], "icbhi_score_official": r[2],
                         "ci95": r[3], "se": r[4], "sp": r[5], "f1_macro": r[6],
                         "delta_vs_A0": (round(r[2]-base, 4) if base and r[0] != "A0" else None),
                         "direction": ("add" if r[0] in ADD_ROWS else "remove")}
                        for r in rows],
               "row_direction_note": "P1-P3 ADD a stage that BASE does not have; every other "
                                     "row REMOVES or replaces one. Their deltas read in "
                                     "opposite directions.",
               "not_applicable": NOT_APPLICABLE},
              open(os.path.join(HERE, "M45_ablation_table.json"), "w"), indent=2)
    print(f"\n  wrote M45_ablation_table.json")


def selftest():
    """Prove the three new stages do what their names say, without touching ICBHI.

    A preprocessing stage that silently does nothing would show up as a delta of ~0 and be
    reported as "the stage does not help", which is the wrong conclusion from a no-op.
    """
    ok = True
    t = np.arange(int(SR * 1.0)) / SR

    lo = np.sin(2 * np.pi * 500 * t).astype(np.float32)      # inside the band
    hi = np.sin(2 * np.pi * 6000 * t).astype(np.float32)     # outside it
    keep = float(np.std(butter_bandpass(lo)) / np.std(lo))
    kill = float(np.std(butter_bandpass(hi)) / np.std(hi))
    print(f"  band-pass    500 Hz kept {keep:.3f} (want ~1) | 6 kHz kept {kill:.4f} (want ~0)")
    ok &= keep > 0.9 and kill < 0.05

    quiet = (lo * 0.01).astype(np.float32)
    peak = float(np.max(np.abs(peak_normalise(quiet))))
    print(f"  peak norm    max |a| = {peak:.4f} (want 1.0)")
    ok &= abs(peak - 1.0) < 1e-3

    # Exponential, not squared-Gaussian: that is the distribution of power in a
    # stationary STFT bin, and the percentile the gate keys on depends on which it is.
    rng = np.random.default_rng(0)
    noise = rng.exponential(1.0, (64, 100))                  # stationary noise floor
    noise[:, 50] += 400.0                                    # one transient frame
    gated = spectral_gate(noise)
    floor_left = float(gated[:, :50].mean() / noise[:, :50].mean())
    transient_left = float(gated[:, 50].mean() / noise[:, 50].mean())
    print(f"  spectral gate  floor kept {floor_left:.3f} (want <0.6) | "
          f"transient kept {transient_left:.3f} (want >0.9)")
    ok &= floor_left < 0.6 and transient_left > 0.9

    keys = set()
    for r in ("A2", "P1", "P2", "P3", "P4"):
        cfg = dict(BASE); cfg.update({k: v for k, v in ROWS[r].items() if not k.startswith("_")})
        keys.add((f"{cfg['n_mels']}m_{cfg['duration_s']}s_{cfg['padding']}_{int(cfg['minmax'])}"
                  f"_bp{int(cfg['bandpass'])}_dn{int(cfg['denoise'])}_an{int(cfg['ampnorm'])}"))
    print(f"  cache keys   {len(keys)} distinct across A2/P1/P2/P3/P4 (want 5)")
    ok &= len(keys) == 5

    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--row", choices=list(ROWS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--audio_dir", default=os.environ.get(
        "ICBHI_AUDIO_DIR", r"C:\Users\Barshon\Desktop\ICBHI_final_database"))
    ap.add_argument("--split_file",
                    default=os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt"))
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.summarise:
        return summarise()
    rows = corrected_split_index(args.audio_dir, args.split_file)
    todo = ([r for r in ROWS if not os.path.exists(os.path.join(HERE, f"results_M45_{r}.json"))]
            if args.all else [args.row])
    print(f"  test cycles {sum(1 for r in rows if r['split']=='test')} | rows to run: {todo}")
    for r in todo:
        run_row(r, rows, args)
    summarise()


if __name__ == "__main__":
    main()
