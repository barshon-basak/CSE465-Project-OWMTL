"""
M12_v2 — backbone re-decision on the corrected split, with a paired test.

WHY THIS EXISTS
    M12 selected the project's shared backbone by comparing candidates' committed
    `icbhi_score` values. Two things were wrong with that comparison:

      1. it used the MACRO variant of the ICBHI score, which the 2026-08-13 audit
         showed inflates by +0.11 on average and hides Se/Sp pathologies;
      2. every candidate number came from the 11-patient / 492-cycle fallback split.

    It also had no confidence interval and no paired test, so a 0.0243 margin was
    compared against a hand-set 0.0129 tolerance.

WHAT THIS DOES
    Re-evaluates the two candidates the team nominated — the 2026-08-28/29 re-runs —
    on the corrected split they were trained under, and decides on
    `icbhi_score_official` with a patient-level bootstrap CI and a paired McNemar.

    Inference only. No training. Both checkpoints are self-describing (`model_config`
    is stored inside them), so the architectures are rebuilt from the checkpoint
    rather than from a hard-coded guess.

CANDIDATES (chosen by the team, 2026-08-29)
    M2  Asif's/M2/m2_v4/best_model.pth          scratch CNN, depth 4, width 32
    M3  Asif's/M3/29 aug run/best_model.pth     mobilenet_v3_small, ImageNet-pretrained

    M4 (AST) is EXCLUDED: no checkpoint exists in the repo, only a link file. A
    candidate that cannot be evaluated cannot be compared, and the original M12 filled
    this gap with transcribed notebook output rather than a committed result.

SPLIT
    `official_icbhi_60_40_patient_disjoint` — the published split with the TRAIN
    recordings of the two straddling patients (156, 218) dropped, so the official TEST
    set stays byte-identical. This is the split both candidates were trained under, so
    it is the one they are scored on. It is NOT the same as the protocol-section-1
    `reassign_to_train` policy used by M22_v2/M3_v2 (369 test recordings); mixing the
    two would compare across partitions.

RUN
    python m12_v2_backbone_redecision.py --audio_dir <ICBHI audio_and_txt_files>
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]
LABEL_OF = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
OVERLAP = {156, 218}

CANDIDATES = {
    "M2": os.path.join(REPO, "Asif's", "M2", "m2_v4", "best_model.pth"),
    "M3": os.path.join(REPO, "Asif's", "M3", "29 aug run", "best_model.pth"),
}
EXCLUDED = {"M4": "no checkpoint committed (link file only) — cannot be evaluated"}


# ------------------------------------------------------------------ split + index
def build_test_index(audio_dir, split_file):
    """The official TEST set. drop_from_train leaves it byte-identical to published."""
    split_map = {}
    with open(split_file) as fh:
        for line in fh:
            t = line.replace("\t", " ").replace(",", " ").split()
            if len(t) >= 2 and t[1].lower() in ("train", "test"):
                split_map[t[0].replace(".wav", "")] = t[1].lower()
    assert len(split_map) == 920, f"{len(split_map)} recordings != 920"
    n_tr = sum(1 for v in split_map.values() if v == "train")
    assert (n_tr, len(split_map) - n_tr) == (539, 381), "not the official split"

    rows = []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        if split_map.get(stem) != "test":
            continue
        txt = os.path.join(audio_dir, stem + ".txt")
        if not os.path.exists(txt):
            continue
        pid = int(stem.split("_")[0])
        with open(txt) as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 4:
                    rows.append({"wav": wav, "stem": stem, "patient_id": pid,
                                 "start": float(p[0]), "end": float(p[1]),
                                 "label": LABEL_OF[(int(p[2]), int(p[3]))]})
    return rows


def log_mel(wav_path, start, end, n_mels=128, n_frames=801, sr=16000):
    """The M2/M3 preprocessing: power_to_db(ref=max) then per-sample min-max to [0,1],
    short cycles wrap-padded. Must match training or the frozen weights see wrong scale."""
    import librosa
    n_samples = sr * 8
    try:
        a, _ = librosa.load(wav_path, sr=sr, offset=start,
                            duration=max(end - start, 0.05), mono=True)
    except Exception as e:
        # A silent all-zero spectrogram here would be trained on and
        # scored as a real cycle. Fail instead of substituting
        # (Model_Training_Protocol.md section 1.2).
        raise RuntimeError(f"failed to load audio: {wav_path}") from e
    if len(a) == 0:
        # Empty decode is a failed read, not a silent zero cycle.
        raise RuntimeError(f"empty audio decoded from {wav_path}")
    a = (np.tile(a, math.ceil(n_samples / len(a)))[:n_samples] if len(a) < n_samples
         else a[:n_samples])
    mel = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=n_mels, n_fft=1024,
                                         hop_length=160, win_length=400,
                                         fmin=50, fmax=2000, power=2.0)
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    T = lm.shape[1]
    lm = (np.pad(lm, ((0, 0), (0, n_frames - T))) if T < n_frames else lm[:, :n_frames])
    return lm[None].astype(np.float32)


# ------------------------------------------------------------------ model rebuild
def build_from_checkpoint(path):
    """Rebuild a candidate from the config stored INSIDE its own checkpoint."""
    import torch
    import torch.nn as nn
    import torchvision

    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw["model_state"]
    mc = raw.get("model_config", {}) or {}

    if any(k.startswith("encoder.") for k in sd):                       # scratch CNN
        depth = mc.get("depth") or len([k for k in sd if k.endswith(".block.0.weight")])
        bw = mc.get("base_width") or int(sd["encoder.0.block.0.weight"].shape[0])
        drop = float(mc.get("dropout", 0.3))

        class ConvBlock(nn.Module):
            def __init__(s, i, o):
                super().__init__()
                s.block = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                        nn.MaxPool2d((2, 2)))
            def forward(s, x): return s.block(x)

        class M2CNN(nn.Module):
            def __init__(s):
                super().__init__()
                ch = [bw * 2 ** i for i in range(depth)]
                blocks, i = [], 1
                for o in ch:
                    blocks.append(ConvBlock(i, o)); i = o
                s.encoder = nn.Sequential(*blocks)
                s.gap = nn.AdaptiveAvgPool2d((1, 1))
                s.dropout = nn.Dropout(drop)
                s.head = nn.Sequential(nn.Linear(ch[-1], 128), nn.ReLU(inplace=True),
                                       nn.Linear(128, 4))
            def forward(s, x):
                return s.head(s.dropout(s.gap(s.encoder(x)).flatten(1)))

        m, desc = M2CNN(), f"scratch CNN depth={depth} width={bw}"
    else:                                                                # torchvision
        arch = mc.get("arch", "mobilenet_v3_small")
        drop = float(mc.get("dropout", 0.3))
        feat_dim = int(sd["head.weight"].shape[1])

        class TVNet(nn.Module):
            def __init__(s):
                super().__init__()
                base = getattr(torchvision.models, arch)(weights=None)
                s.features = base.features
                s.register_buffer("in_mean", torch.zeros(1, 3, 1, 1))
                s.register_buffer("in_std", torch.ones(1, 3, 1, 1))
                s.gap = nn.AdaptiveAvgPool2d((1, 1))
                s.dropout = nn.Dropout(drop)
                s.head = nn.Linear(feat_dim, 4)
            def forward(s, x):
                x = (x.repeat(1, 3, 1, 1) - s.in_mean) / s.in_std
                return s.head(s.dropout(s.gap(s.features(x)).flatten(1)))

        m, desc = TVNet(), f"{arch} (feat_dim {feat_dim})"

    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "num_batches" not in k], f"missing weights: {missing[:5]}"
    assert not unexpected, f"unexpected weights: {unexpected[:5]}"
    return m.eval(), desc, int(raw.get("epoch", -1)), float(raw.get("best_score", float("nan")))


# ------------------------------------------------------------------ metrics
def official(cm):
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else float("nan")
    return float((se + sp) / 2), float(se), float(sp)


def patient_bootstrap(y, pred, pid, n_boot=5000, seed=42):
    from sklearn.metrics import confusion_matrix
    uq = np.unique(pid); ix = {p: np.flatnonzero(pid == p) for p in uq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        sel = np.concatenate([ix[p] for p in rng.choice(uq, len(uq), replace=True)])
        v = official(confusion_matrix(y[sel], pred[sel], labels=[0, 1, 2, 3]))[0]
        if v == v:
            vals.append(v)
    return [round(float(x), 4) for x in np.percentile(vals, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir",
                    default=os.environ.get("ICBHI_AUDIO_DIR",
                                           r"C:\Users\Barshon\Desktop\ICBHI_final_database"))
    ap.add_argument("--split_file",
                    default=os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()

    import torch
    import pandas as pd
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    from scipy import stats

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 74)
    print("M12_v2 — BACKBONE RE-DECISION on the corrected split")
    print("=" * 74)
    print(f"  device      : {dev}")
    print(f"  split       : official_icbhi_60_40_patient_disjoint (drop_from_train)")

    rows = build_test_index(args.audio_dir, args.split_file)
    y = np.array([r["label"] for r in rows])
    pid = np.array([r["patient_id"] for r in rows])
    print(f"  test set    : {len(rows)} cycles / {len(np.unique(pid))} patients")
    if len(rows) != 2756:
        print(f"  [warn] expected 2756 cycles (what the candidates reported), got {len(rows)}")

    cache = os.path.join(HERE, "test_specs.npy")
    if os.path.exists(cache):
        X = np.load(cache, mmap_mode="r")
        print(f"  cache hit   : {os.path.basename(cache)} {X.shape}")
    else:
        print("  extracting log-mels ...")
        X = np.zeros((len(rows), 1, 128, 801), np.float32)
        for i, r in enumerate(rows):
            X[i] = log_mel(r["wav"], r["start"], r["end"])
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(rows)}")
        np.save(cache, X)

    results = {}
    for mid, ck in CANDIDATES.items():
        if not os.path.exists(ck):
            print(f"  [skip] {mid}: checkpoint not found at {ck}")
            continue
        model, desc, ep, best = build_from_checkpoint(ck)
        model = model.to(dev)
        preds = np.zeros(len(rows), int)
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                xb = torch.from_numpy(np.asarray(X[i:i + args.batch_size])).to(dev)
                preds[i:i + len(xb)] = model(xb).argmax(1).cpu().numpy()
        cm = confusion_matrix(y, preds, labels=[0, 1, 2, 3])
        sc, se, sp = official(cm)
        ci = patient_bootstrap(y, preds, pid, args.n_boot)
        results[mid] = {
            "checkpoint": os.path.relpath(ck, REPO), "architecture": desc,
            "trained_to_epoch": ep, "reported_best_score_at_train": round(best, 4),
            "icbhi_score_official": round(sc, 4), "icbhi_score_official_ci95": ci,
            "icbhi_se_official": round(se, 4), "icbhi_sp_official": round(sp, 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "f1_macro": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
            "confusion_matrix_raw": cm.tolist(),
            "_preds": preds,
        }
        print(f"\n  {mid}  {desc}")
        print(f"      official {sc:.4f}  CI95 {ci}   Se {se:.4f}  Sp {sp:.4f}")
        print(f"      accuracy {results[mid]['accuracy']:.4f}  macro-F1 "
              f"{results[mid]['f1_macro']:.4f}")
        pd.DataFrame({"stem": [r["stem"] for r in rows], "patient_id": pid,
                      "y_true": y, "y_pred": preds}).to_csv(
            os.path.join(HERE, f"preds_{mid}_corrected.csv"), index=False)

    # ---------------- paired comparison ----------------
    ids = list(results)
    paired = None
    if len(ids) == 2:
        a, b = ids
        pa, pb = results[a]["_preds"], results[b]["_preds"]
        ok_a, ok_b = pa == y, pb == y
        n_ab = int((ok_a & ~ok_b).sum()); n_ba = int((~ok_a & ok_b).sum())
        p_mc = float(stats.binomtest(min(n_ab, n_ba), n_ab + n_ba, 0.5).pvalue) \
            if (n_ab + n_ba) else 1.0
        uq = np.unique(pid); ix = {p: np.flatnonzero(pid == p) for p in uq}
        rng = np.random.default_rng(42); diffs = []
        for _ in range(args.n_boot):
            sel = np.concatenate([ix[p] for p in rng.choice(uq, len(uq), replace=True)])
            da = official(confusion_matrix(y[sel], pa[sel], labels=[0, 1, 2, 3]))[0]
            db = official(confusion_matrix(y[sel], pb[sel], labels=[0, 1, 2, 3]))[0]
            if da == da and db == db:
                diffs.append(da - db)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pboot = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
        delta = results[a]["icbhi_score_official"] - results[b]["icbhi_score_official"]
        sig = lo > 0 or hi < 0
        paired = {
            "comparison": f"{a} minus {b}", "delta": round(delta, 4),
            "delta_ci95_patient_bootstrap": [round(float(lo), 4), round(float(hi), 4)],
            "delta_p_bootstrap": round(float(min(pboot, 1.0)), 4),
            "mcnemar": {f"{a}_only_right": n_ab, f"{b}_only_right": n_ba,
                        "p_exact": float(f"{p_mc:.4g}"),
                        "note": "cycle-level and therefore anti-conservative; cycles from "
                                "one patient are not independent. Lead with the bootstrap."},
            "separable": bool(sig),
        }
        print(f"\n  paired {a} - {b}: {delta:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  "
              f"p={min(pboot,1.0):.4f}")
        print(f"  McNemar {a}-only-right={n_ab} {b}-only-right={n_ba} p={p_mc:.4g}")

    # ---------------- decision ----------------
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["icbhi_score_official"])
    winner = ranked[0][0]
    if paired and not paired["separable"]:
        verdict = (f"NO SEPARATION. {ranked[0][0]} leads on the point estimate "
                   f"({ranked[0][1]['icbhi_score_official']} vs "
                   f"{ranked[1][1]['icbhi_score_official']}) but the paired interval spans "
                   f"zero, so the candidates are not distinguishable on this test set. Any "
                   f"backbone choice between them must be justified on grounds other than "
                   f"the official score — efficiency, or a stated preference — and said so "
                   f"explicitly.")
    else:
        verdict = (f"{winner} SELECTED. It leads on icbhi_score_official and the paired "
                   f"interval excludes zero.")
    print("\n" + "=" * 74)
    print("  DECISION:", verdict)
    print("=" * 74)

    doc = {
        "meta": {"model_id": "M12_v2", "model_name": "Backbone Re-Decision (corrected split)",
                 "contributor": "OWMTL team",
                 "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                 "is_augmented": False, "augmentation_method": "none",
                 "notes": "M12_v2 is a DECISION, not a trained model — inference only. It "
                          "supersedes M12, which decided on the MACRO icbhi_score using "
                          "candidate numbers from the 11-patient fallback split, with no CI "
                          "and no paired test. Candidates here are the 2026-08-28/29 "
                          "re-runs nominated by the team; architectures were rebuilt from "
                          "the model_config stored inside each checkpoint."},
        "supersedes": {"M12": {"decided_on": "icbhi_score (macro variant)",
                               "candidate_split": "patient_id fallback, 11 test patients",
                               "selected": "M2", "margin": 0.0243,
                               "tie_tolerance_used": 0.0129,
                               "had_confidence_interval": False,
                               "had_paired_test": False}},
        "dataset_info": {"dataset": "ICBHI_2017",
                         "split_method": "official_icbhi_60_40_patient_disjoint",
                         "overlap_policy": "drop_from_train",
                         "test_samples": len(rows),
                         "test_patients": int(len(np.unique(pid))),
                         "evaluation": "inference only, frozen checkpoints"},
        "primary_metric": "icbhi_score_official",
        "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "_preds"}
                       for k, v in results.items()},
        "excluded_candidates": EXCLUDED,
        "paired_comparison": paired,
        "decision": {"leader_on_point_estimate": winner, "verdict": verdict},
    }
    out = os.path.join(HERE, "results_M12_v2.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"\n  wrote {out}")
    for mid in results:
        print(f"  wrote preds_{mid}_corrected.csv")


if __name__ == "__main__":
    main()
