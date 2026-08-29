"""
make_m2_features.py - generate M2_features.npy, the one file that unblocks five experiments.

WHAT THIS PRODUCES
    M2_features.npy : (6898, 768) float32, the frozen M2 encoder embedding for every
                      annotated ICBHI cycle, in EXACTLY the row order of concepts_all.npz.

WHY IT IS NOT JUST `owmtl.m2_features.export_features`
    That helper's `default_logmel` uses a plain `power_to_db` with no normalisation. M2's
    actual notebook (`Asif's/M2/M2_cnn_baseline_tuned.ipynb`) uses `power_to_db(ref=np.max)`
    followed by PER-SAMPLE MIN-MAX normalisation to [0, 1], and wrap-pads short cycles by
    repetition. Feeding a frozen network inputs on a different scale than it was trained on
    produces embeddings that look fine, load fine, and are silently wrong. This script
    reproduces M2's preprocessing exactly, and `export_features` should not be used for
    this checkpoint until its default is fixed.

ALIGNMENT IS THE WHOLE POINT
    Every downstream script indexes concepts and features by the same row. This script
    builds its cycle index with the same `owmtl.icbhi_data.build_cycle_index` that produced
    concepts_all.npz, and then VERIFIES row-by-row that the patient ids match before
    saving. If they do not, it refuses to write the file - a misaligned feature matrix
    would corrupt N2, N3, N4, N5 and N6 at once, and nothing downstream could detect it.

REQUIREMENTS
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install librosa

RUN
    python make_m2_features.py \
        --audio_dir "C:/Users/Barshon/Desktop/ICBHI_final_database" \
        --checkpoint "../Asif's/M2/M2_best_model.pth"

    Takes ~15-25 min on CPU for 6898 cycles. Add --limit 200 for a quick wiring check
    (writes nothing, just proves the pipeline runs end to end).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# M2's preprocessing constants (Protocol section 2 defaults, as used by the M2 notebook).
CFG = {"sample_rate": 16000, "duration_s": 8.0, "n_mels": 128, "n_fft": 1024,
       "hop_length": 160, "win_length": 400, "f_min": 50, "f_max": 2000}
CFG["n_samples"] = int(CFG["sample_rate"] * CFG["duration_s"])          # 128000
CFG["n_frames"] = CFG["n_samples"] // CFG["hop_length"] + 1            # 801


def build_m2(state_dict):
    """Rebuild M2_CNN and infer depth/base_width from the checkpoint itself.

    The class defaults are depth=4/base_width=32, but the run that was actually kept used
    the swept depth=5/base_width=48 (768-d embedding). Reading the shapes instead of
    trusting a default is what stops a silent architecture mismatch.
    """
    import torch
    import torch.nn as nn

    conv_shapes = [v.shape for k, v in state_dict.items()
                   if k.startswith("encoder.") and k.endswith(".block.0.weight")]
    if not conv_shapes:
        raise RuntimeError(
            "no encoder conv weights found in the checkpoint. Keys look like:\n  "
            + "\n  ".join(list(state_dict)[:12]))
    depth = len(conv_shapes)
    base_width = int(conv_shapes[0][0])
    embed_dim = int(conv_shapes[-1][0])
    print(f"  checkpoint says: depth={depth} base_width={base_width} "
          f"embedding_dim={embed_dim}")

    class ConvBlock(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                nn.ReLU(inplace=True), nn.MaxPool2d((2, 2)))

        def forward(self, x):
            return self.block(x)

    class M2_CNN(nn.Module):
        def __init__(self, num_classes=4, fc_dim=128, dropout=0.4):
            super().__init__()
            ch = [base_width * (2 ** i) for i in range(depth)]
            blocks, i = [], 1
            for o in ch:
                blocks.append(ConvBlock(i, o))
                i = o
            self.encoder = nn.Sequential(*blocks)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(ch[-1], fc_dim), nn.ReLU(inplace=True),
                                      nn.Linear(fc_dim, num_classes))
            self.embedding_dim = ch[-1]

        def get_embedding(self, x):
            return self.gap(self.encoder(x)).flatten(1)

    model = M2_CNN()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    if enc_missing:
        raise RuntimeError(f"encoder weights missing from the checkpoint: {enc_missing[:6]}"
                           "\nThis checkpoint does not match the M2 architecture.")
    print(f"  loaded. non-encoder missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval(), embed_dim


def log_mel(wav_path, start, end):
    """M2's exact cycle preprocessing. Do not 'simplify' this - see the module docstring."""
    import librosa
    sr, n_samples = CFG["sample_rate"], CFG["n_samples"]
    try:
        audio, _ = librosa.load(wav_path, sr=sr, offset=start,
                                duration=max(end - start, 0.05), mono=True)
    except Exception:
        return np.zeros((CFG["n_mels"], CFG["n_frames"]), np.float32)
    if len(audio) == 0:
        return np.zeros((CFG["n_mels"], CFG["n_frames"]), np.float32)

    # wrap-pad by repetition (M2/M1/M4 convention), then crop
    if len(audio) < n_samples:
        audio = np.tile(audio, math.ceil(n_samples / len(audio)))[:n_samples]
    else:
        audio = audio[:n_samples]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=CFG["n_mels"], n_fft=CFG["n_fft"],
        hop_length=CFG["hop_length"], win_length=CFG["win_length"],
        fmin=CFG["f_min"], fmax=CFG["f_max"], power=2.0)
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)     # per-sample min-max to [0,1]

    T = lm.shape[1]
    if T < CFG["n_frames"]:
        lm = np.pad(lm, ((0, 0), (0, CFG["n_frames"] - T)), mode="constant")
    else:
        lm = lm[:, :CFG["n_frames"]]
    return lm.astype(np.float32)


def find_diagnosis_file(audio_dir):
    roots = [audio_dir, os.path.dirname(audio_dir.rstrip("/\\")),
             os.path.dirname(os.path.dirname(audio_dir.rstrip("/\\")))]
    for r in roots:
        for name in ("ICBHI_Challenge_diagnosis.txt", "patient_diagnosis.csv",
                     "ICBHI_challenge_diagnosis.txt"):
            p = os.path.join(r, name)
            if os.path.isfile(p):
                return p
    return None


def derive_diagnosis_file(out_path):
    """Rebuild the patient->diagnosis table from concepts_all.npz.

    `build_cycle_index` requires a diagnosis file and raises without one, but the diagnosis
    label plays NO part in extracting an embedding - it only fills a field this script never
    reads. concepts_all.npz already stores the exact patient/diagnosis pairing that
    notebook 01 used, so deriving the file from it is both sufficient and strictly more
    consistent than sourcing a copy from elsewhere.
    """
    ref = C.load_concepts()
    seen = {}
    for p, dg in zip(ref["patient"], ref["diagnosis"]):
        seen.setdefault(str(p), str(dg))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for p in sorted(seen, key=int):
            fh.write(f"{p}\t{seen[p]}\n")
    return out_path, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True,
                    help="directory holding the 920 ICBHI .wav files and their .txt")
    ap.add_argument("--checkpoint", default=None,
                    help="M2 .pth (default: ../Asif's/M2/M2_best_model.pth)")
    ap.add_argument("--diagnosis_file", default=None)
    ap.add_argument("--out", default=os.path.join(C.HERE, "M2_features.npy"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="process only N cycles as a wiring check; writes nothing")
    args = ap.parse_args()

    C.banner("make_m2_features", "frozen M2 embeddings, aligned to concepts_all.npz")

    try:
        import torch
        import librosa  # noqa: F401
    except ImportError as ex:
        return C.blocked("make_m2_features", f"missing dependency: {ex.name}", [
            "pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "pip install librosa"])

    ckpt_path = args.checkpoint or os.path.join(C.repo_root(), "Asif's", "M2",
                                                "M2_best_model.pth")
    if not os.path.isfile(ckpt_path):
        return C.blocked("make_m2_features", f"checkpoint not found: {ckpt_path}",
                         ["--checkpoint <path to M2 best_model.pth>"])

    diag = args.diagnosis_file or find_diagnosis_file(args.audio_dir)
    derived = False
    if diag is None:
        diag, n_pat = derive_diagnosis_file(
            os.path.join(C.RESULTS_DIR, "derived_patient_diagnosis.txt"))
        derived = True
        print(f"  [note] no ICBHI diagnosis file on disk - derived one for {n_pat} patients "
              "from\n         concepts_all.npz. The label is not used to compute an "
              "embedding, so this\n         cannot affect the features.")

    from owmtl.icbhi_data import build_cycle_index, find_split_file
    split_file = find_split_file([args.audio_dir])
    print(f"  audio     : {args.audio_dir}")
    print(f"  split     : {split_file}")
    print(f"  diagnosis : {diag}{' (derived)' if derived else ''}")
    print(f"  checkpoint: {ckpt_path}")

    records = build_cycle_index(args.audio_dir, split_file, diag)
    print(f"  cycles indexed: {len(records)}")

    # ---- alignment gate, BEFORE spending 20 minutes of compute -----------------
    ref = C.load_concepts()
    if len(records) != len(ref["X"]):
        return C.blocked("make_m2_features",
                         f"cycle count {len(records)} != concepts_all.npz "
                         f"{len(ref['X'])}", [
                             "the audio directory is not the one notebook 01 used",
                             "check for missing/extra .wav or .txt files"])
    mism = [i for i in range(len(records))
            if records[i].patient != str(ref["patient"][i])][:5]
    if mism:
        return C.blocked("make_m2_features", "cycle ORDER differs from concepts_all.npz", [
            f"first mismatches at rows {mism}",
            "features must be row-aligned to the concepts or every downstream "
            "experiment is silently corrupted"])
    print("  alignment : OK (row order matches concepts_all.npz)")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    model, embed_dim = build_m2(sd)

    todo = records[:args.limit] if args.limit else records
    feats = np.zeros((len(todo), embed_dim), np.float32)
    n_fail = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        mels = np.stack([log_mel(os.path.join(args.audio_dir, r.stem + ".wav"),
                                 r.start, r.end) for r in chunk])
        n_fail += int(sum(1 for m in mels if not np.any(m)))
        with torch.no_grad():
            feats[i:i + len(chunk)] = model.get_embedding(
                torch.from_numpy(mels)[:, None]).numpy()
        if (i // args.batch_size) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(todo)}")

    print(f"  done. all-zero (unreadable) cycles: {n_fail}")
    if args.limit:
        print(f"\n  --limit {args.limit} was set: NOT writing {args.out}.")
        print(f"  Wiring is good (shape {feats.shape}). Re-run without --limit.")
        return

    np.save(args.out, feats)
    print(f"\n[saved] {args.out}  shape={feats.shape} dtype={feats.dtype}")
    print(f"        {os.path.getsize(args.out) / 1e6:.1f} MB")
    print("\nNext:")
    print("  python run_all.py --features M2_features.npy")


if __name__ == "__main__":
    main()
