"""
Shared audio preprocessing for the OWMTL project.

⚠️ READ THIS BEFORE CITING OR RUNNING ANYTHING HERE (added 2026-08-30)

    This module is a REFERENCE implementation. It is NOT the chain that produced
    the committed results. An audit on 2026-08-30 found that nothing in the repo
    imports it — every model script re-implements its own log-mel — and that its
    defaults disagree with what the runs actually did:

        stage            this module              what the runs did
        padding          reflect (or tile)        np.tile (wrap), always
        spectrogram      power_to_db, no rescale  + per-sample min-max to [0,1]
        band-pass        ON, stage 3              never applied until M45 row P1
        denoising        absent                   added as M45 row P2

    So a model trained through this module would not reproduce any number in the
    paper, and describing the paper's pipeline from this file would misdescribe
    it. The AS-RUN chain is `Asif's/M45/m45_ablation.py::log_mel`, whose stages
    are individually ablated in `M45_ablation_table.json` (rows A5, A6, P1-P5).

    Kept, not deleted: the stage vocabulary and AudioConfig below are the
    project's written specification, and the gap between the specification and
    the runs is itself recorded in the paper's preprocessing section. If this
    module is ever adopted, its defaults must be reconciled with the as-run chain
    FIRST, or the new runs will silently disagree with the committed table.

The seven preprocessing stages are named explicitly here so they can be echoed
into each run's results JSON. The Aug 8 assignment marks "apply ALL the
preprocessing techniques" as a separate item — `PREPROCESSING_STAGES` is the list
to put on that slide.

    1. resample          -> mono, 16 kHz
    2. cycle segmentation-> slice [start, end) from the annotation file
    3. bandpass filter   -> 4th-order Butterworth, 50-2000 Hz
    4. fixed length      -> pad (reflect) / centre-crop to 8.0 s
    5. amplitude norm    -> peak normalisation to unit amplitude
    6. spectral features -> log-mel (128 bins) or MFCC (40 coeffs)
    7. per-channel norm  -> mean/std standardisation, statistics fit on TRAIN ONLY

SpecAugment is stage 8 and is training-time only — it is augmentation, not
preprocessing, and is reported as a separate ablation (M21-M23).

Deviations from the group spec are recorded, not silently applied:
  * `n_fft` is 1024 per Model_Training_Protocol.md §2. Barshon's M1 used 512;
    that is a real difference in spectral resolution and the two runs are not
    directly comparable until one is re-run.
  * AST uses its own Kaldi-fbank front-end (`ast_fbank`) rather than the librosa
    log-mel path, because its pretrained weights were learned on Kaldi fbank with
    AudioSet normalisation statistics. Feeding it librosa log-mel would handicap
    it and corrupt the backbone comparison. This is flagged in the results JSON
    as `frontend: "kaldi_fbank"`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PREPROCESSING_STAGES: Tuple[str, ...] = (
    "resample_16kHz_mono",
    "cycle_segmentation_from_annotations",
    "butterworth_bandpass_50_2000Hz_order4",
    "fixed_length_8s_reflect_pad_center_crop",
    "peak_amplitude_normalisation",
    "log_mel_128bins_or_mfcc_40",
    "per_channel_mean_std_normalisation_train_stats",
)

# AudioSet statistics the AST checkpoint was normalised with. Do not change.
AST_FBANK_MEAN = -4.2677393
AST_FBANK_STD = 4.5689974


@dataclass(frozen=True)
class AudioConfig:
    """Shared audio parameters (Model_Training_Protocol.md §2)."""

    sample_rate: int = 16000
    duration_s: float = 8.0
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 160      # 10 ms
    win_length: int = 400      # 25 ms
    f_min: float = 50.0
    f_max: float = 2000.0
    bandpass_low: float = 50.0
    bandpass_high: float = 2000.0
    bandpass_order: int = 4

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.duration_s))

    @property
    def n_frames(self) -> int:
        return 1 + self.n_samples // self.hop_length

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["n_samples"] = self.n_samples
        d["n_frames"] = self.n_frames
        d["preprocessing_stages"] = list(PREPROCESSING_STAGES)
        return d


DEFAULT_CONFIG = AudioConfig()


# ---------------------------------------------------------------------------
# Stages 1-5: waveform
# ---------------------------------------------------------------------------


def bandpass(signal: np.ndarray, cfg: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Stage 3 — 4th-order Butterworth bandpass, zero-phase (filtfilt).

    Removes heart-sound energy below 50 Hz and out-of-band noise above 2 kHz.
    Falls back to the unfiltered signal if the segment is shorter than the
    filter's padding requirement, which happens for a few very short cycles.
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * cfg.sample_rate
    low = max(cfg.bandpass_low / nyq, 1e-6)
    high = min(cfg.bandpass_high / nyq, 0.999)
    b, a = butter(cfg.bandpass_order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if signal.size <= padlen:
        return signal
    return filtfilt(b, a, signal).astype(np.float32)


def fix_length(signal: np.ndarray, cfg: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Stage 4 — pad or crop to exactly `duration_s`.

    Short cycles are reflect-padded rather than zero-padded: a zero-padded 1.5 s
    cycle sitting in an 8 s window gives the network a trivial "where does the
    silence start" cue that correlates with cycle duration, and cycle duration
    correlates with pathology. Reflection removes that cue.
    """
    n = cfg.n_samples
    if signal.size == n:
        return signal
    if signal.size > n:
        start = (signal.size - n) // 2
        return signal[start:start + n]
    deficit = n - signal.size
    if signal.size < 2:
        return np.pad(signal, (0, deficit))
    # np.pad's reflect mode requires pad width < signal length; tile if needed.
    reps = int(np.ceil(n / signal.size))
    tiled = np.tile(signal, reps)[:n] if reps > 2 else np.pad(
        signal, (deficit // 2, deficit - deficit // 2), mode="reflect"
    )
    return tiled[:n].astype(np.float32)


def peak_normalise(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Stage 5 — scale to unit peak amplitude.

    ICBHI recordings vary in gain by device; without this the network can read
    the stethoscope off the loudness envelope.
    """
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < eps:
        return signal.astype(np.float32)
    return (signal / peak).astype(np.float32)


def load_cycle(
    wav_path: str,
    start: float,
    end: float,
    cfg: AudioConfig = DEFAULT_CONFIG,
    *,
    waveform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stages 1-5 — one annotated respiratory cycle as a fixed-length waveform.

    Pass `waveform` (a full recording already loaded at `cfg.sample_rate`) to
    slice without re-decoding; `precompute_features` uses this so each of the 920
    recordings is decoded once rather than once per cycle.
    """
    if waveform is None:
        import librosa

        waveform, _ = librosa.load(wav_path, sr=cfg.sample_rate, mono=True)
    i0 = max(int(round(start * cfg.sample_rate)), 0)
    i1 = min(int(round(end * cfg.sample_rate)), waveform.size)
    seg = np.asarray(waveform[i0:i1], dtype=np.float32)
    if seg.size == 0:
        seg = np.zeros(1, dtype=np.float32)
    seg = bandpass(seg, cfg)
    seg = fix_length(seg, cfg)
    return peak_normalise(seg)


# ---------------------------------------------------------------------------
# Stage 6: spectral representations
# ---------------------------------------------------------------------------


def log_mel(signal: np.ndarray, cfg: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Stage 6 — log-mel spectrogram, shape (n_mels, n_frames). dB scale."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def mfcc(signal: np.ndarray, cfg: AudioConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Stage 6 (variant) — MFCC, shape (n_mfcc, n_frames). Representation ablation."""
    import librosa

    return librosa.feature.mfcc(
        S=log_mel(signal, cfg), n_mfcc=cfg.n_mfcc
    ).astype(np.float32)


def ast_fbank(signal: np.ndarray, cfg: AudioConfig = DEFAULT_CONFIG,
              target_frames: int = 1024) -> np.ndarray:
    """AST front-end — Kaldi filterbank, shape (target_frames, n_mels).

    Matches the AST checkpoint's pretraining front-end exactly (Gong et al. 2021):
    Kaldi fbank, 25 ms window / 10 ms shift, Hanning, then normalised by the
    AudioSet statistics. Padded to 1024 frames so the pretrained positional
    embeddings load unchanged — an 8 s cycle yields ~798 frames, the rest is
    zero-padded.
    """
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0)
    wav = wav - wav.mean()
    fb = kaldi.fbank(
        wav,
        htk_compat=True,
        sample_frequency=cfg.sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=cfg.n_mels,
        dither=0.0,
        frame_shift=10,
    )
    n = fb.shape[0]
    if n < target_frames:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, target_frames - n))
    elif n > target_frames:
        fb = fb[:target_frames]
    fb = (fb - AST_FBANK_MEAN) / (2 * AST_FBANK_STD)
    return fb.numpy().astype(np.float32)


FRONTENDS = {"log_mel": log_mel, "mfcc": mfcc, "ast_fbank": ast_fbank}


# ---------------------------------------------------------------------------
# Stage 7: normalisation statistics
# ---------------------------------------------------------------------------


def fit_norm_stats(features: np.ndarray) -> Dict[str, float]:
    """Stage 7 — global mean/std over the TRAIN partition only.

    Fitting these on all data is a subtle leak that shows up in every second
    ICBHI notebook. Fit here, save alongside the checkpoint, reuse at test time.
    """
    arr = np.asarray(features, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()) or 1.0}


def apply_norm(features: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return ((features - stats["mean"]) / (stats["std"] or 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 8: SpecAugment (training only)
# ---------------------------------------------------------------------------


def spec_augment(
    spec: np.ndarray,
    *,
    freq_mask_param: int = 24,
    time_mask_param: int = 96,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
    rng: Optional[np.random.Generator] = None,
    freq_axis: int = 0,
) -> np.ndarray:
    """SpecAugment masking (Park et al. 2019). Training data only.

    Defaults follow the AST paper's AudioSet recipe. `freq_axis=0` for librosa
    log-mel (n_mels, n_frames); `freq_axis=1` for AST fbank (n_frames, n_mels).
    """
    rng = rng or np.random.default_rng()
    out = np.array(spec, copy=True)
    time_axis = 1 - freq_axis
    n_freq, n_time = out.shape[freq_axis], out.shape[time_axis]
    fill = float(out.mean())

    def _mask(axis: int, size: int, param: int, count: int) -> None:
        for _ in range(count):
            width = int(rng.integers(0, min(param, size) + 1))
            if width == 0:
                continue
            start = int(rng.integers(0, size - width + 1))
            sl = [slice(None), slice(None)]
            sl[axis] = slice(start, start + width)
            out[tuple(sl)] = fill

    _mask(freq_axis, n_freq, freq_mask_param, n_freq_masks)
    _mask(time_axis, n_time, time_mask_param, n_time_masks)
    return out


def spec_augment_config(**kw) -> Dict:
    """The block to drop into results JSON `augmentation_method` for M21-M23."""
    cfg = {
        "method": "SpecAugment",
        "freq_mask_param": 24,
        "time_mask_param": 96,
        "n_freq_masks": 2,
        "n_time_masks": 2,
        "applied_to": "train_split_only",
        "reference": "Park et al. 2019; AST AudioSet recipe (Gong et al. 2021)",
    }
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# Precomputed feature cache
# ---------------------------------------------------------------------------


def precompute_features(
    index: Sequence[dict],
    audio_dir: str,
    out_dir: str,
    *,
    frontend: str = "log_mel",
    cfg: AudioConfig = DEFAULT_CONFIG,
    dtype: str = "float16",
    progress: bool = True,
) -> str:
    """Decode every cycle once into a single memory-mapped array.

    Writes `<out_dir>/<frontend>.npy` (N, H, W) plus `<frontend>.meta.json`
    mapping row index -> cycle_uid. Decoding on the fly costs ~3 minutes per
    epoch on Colab; this pays it once. Put `out_dir` on Colab local disk
    (/content/cache), not Drive — it regenerates in minutes and is far too much
    I/O for Drive.
    """
    if frontend not in FRONTENDS:
        raise ValueError(f"frontend must be one of {sorted(FRONTENDS)}")
    import librosa

    os.makedirs(out_dir, exist_ok=True)
    fn = FRONTENDS[frontend]

    probe = fn(np.zeros(cfg.n_samples, dtype=np.float32), cfg)
    shape = (len(index), *probe.shape)
    arr_path = os.path.join(out_dir, f"{frontend}.npy")
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.dtype(dtype), shape=shape)

    # Group by recording so each .wav is decoded exactly once.
    by_stem: Dict[str, List[int]] = {}
    for i, row in enumerate(index):
        by_stem.setdefault(row["stem"], []).append(i)

    stems = sorted(by_stem)
    iterator = stems
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(stems, desc=f"precompute[{frontend}]", unit="rec")
        except ImportError:
            pass

    for stem in iterator:
        wav_path = os.path.join(audio_dir, stem + ".wav")
        waveform, _ = librosa.load(wav_path, sr=cfg.sample_rate, mono=True)
        for i in by_stem[stem]:
            row = index[i]
            seg = load_cycle(wav_path, row["start"], row["end"], cfg, waveform=waveform)
            arr[i] = fn(seg, cfg).astype(dtype)
    arr.flush()

    meta = {
        "frontend": frontend,
        "dtype": dtype,
        "shape": list(shape),
        "config": cfg.to_dict(),
        "cycle_uids": [row["cycle_uid"] for row in index],
    }
    with open(os.path.join(out_dir, f"{frontend}.meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return arr_path


def load_feature_cache(out_dir: str, frontend: str = "log_mel") -> Tuple[np.ndarray, Dict]:
    """Open a precomputed cache read-only. Returns (memmap array, metadata)."""
    arr = np.load(os.path.join(out_dir, f"{frontend}.npy"), mmap_mode="r")
    with open(os.path.join(out_dir, f"{frontend}.meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return arr, meta
