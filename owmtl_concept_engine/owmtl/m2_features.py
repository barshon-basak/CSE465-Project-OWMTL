"""
owmtl.m2_features  (helper for Build step 02 — deterministic)
=============================================================

Export FROZEN M2 encoder features aligned to the cycle index from notebook 01, so
notebook 02 (and the leakage/intervention steps) can run the sequential/leaky/opaque
variants. This is NOT result-dependent -- it is a deterministic feature dump from an
already-selected, frozen backbone (M2, chosen by M12).

You must supply how to build the M2 model + return its penultimate embedding, because
that lives in your M2 notebook. Fill in `load_m2` (one function). Everything else is
generic: it re-uses owmtl.icbhi_data to load each cycle, computes the SAME log-mel the
project uses, runs the frozen model, and stores the pre-classifier embedding.

Requires torch, librosa (Kaggle/Colab).
"""
from __future__ import annotations
import numpy as np


def default_logmel(y, sr=16000, n_mels=128, n_fft=1024, hop=160, win=400,
                   fmin=50, fmax=2000):
    """The project's shared log-mel (Protocol §2)."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       win_length=win, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return librosa.power_to_db(S + 1e-10)


def export_features(records, audio_dir, load_m2, sr=16000, target_len=801,
                    logmel_fn=default_logmel, device=None, out_path=None,
                    batch_log=500):
    """records   : the SAME list[CycleRecord] used in notebook 01 (order == concept order)
    load_m2   : () -> (model, embed_fn) where model is a frozen nn.Module and
                embed_fn(model, mel_tensor) -> (B, D) penultimate embedding.
    Returns (N, D) float32 aligned to `records`, and optionally saves to out_path.
    """
    import torch
    from owmtl.icbhi_data import load_cycle_waveform
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, embed_fn = load_m2()
    model.eval().to(dev)

    feats = None
    for i, r in enumerate(records):
        try:
            y = load_cycle_waveform(audio_dir, r, sr=sr)
            mel = logmel_fn(y, sr=sr)                       # (n_mels, T)
            # pad/crop time to target_len for a fixed input
            T = mel.shape[1]
            if T < target_len:
                mel = np.pad(mel, ((0, 0), (0, target_len - T)))
            else:
                mel = mel[:, :target_len]
            x = torch.tensor(mel, dtype=torch.float32, device=dev)[None, None]  # (1,1,M,T)
            with torch.no_grad():
                emb = embed_fn(model, x).squeeze(0).cpu().numpy()
        except Exception as ex:
            emb = None
            if i < 5:
                print("warn", r.stem, ex)
        if feats is None and emb is not None:
            feats = np.zeros((len(records), emb.shape[-1]), dtype=np.float32)
        if emb is not None:
            feats[i] = emb
        if (i + 1) % batch_log == 0:
            print(f"M2 features {i+1}/{len(records)}")
    if feats is None:
        raise RuntimeError("no features extracted — check load_m2 / embed_fn")
    if out_path:
        np.save(out_path, feats)
        print("saved", out_path, feats.shape)
    return feats


# ---- Example load_m2 skeleton (copy into your notebook and fill in) --------------
EXAMPLE = '''
def load_m2():
    import torch
    from your_m2_module import M2CNN            # TODO: your M2 architecture class
    model = M2CNN(num_classes=4)
    sd = torch.load("M2_best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state"] if "model_state" in sd else sd)
    def embed_fn(m, x):
        # return the PRE-classifier embedding (global-pooled conv features)
        return m.forward_features(x)           # TODO: expose this in M2 (or hook it)
    return model, embed_fn
'''
