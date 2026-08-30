"""
owmtl.concept_extractors
========================

Physics-grounded, LABEL-FREE acoustic CONCEPT extractors for respiratory cycles.

This is the core of the project's current direction: instead of routing a diagnosis
through opaque encoder features, we compute a small vector of **clinically-named,
signal-derived** acoustic concepts per breathing cycle and force the diagnosis
through them (see owmtl.bottleneck). The concepts are computed by DSP from the raw
waveform -- no human annotation and no CLIP/LLM concepts -- which is exactly what
distinguishes this from (a) generic MFCC/spectral-feature branches (already published)
and (b) label-free CBMs built on CLIP concepts (not clinically grounded).

Clinical grounding (measurable properties of adventitious lung sounds):
  * Crackles  : discontinuous, explosive TRANSIENTS. Fine = short (~5 ms), higher
                centre-frequency, mid/late inspiration; Coarse = longer (~15 ms),
                lower centre-frequency, early inspiration + expiration.
  * Wheezes   : continuous, MUSICAL/TONAL, sustained narrow-band energy, ~100-1000 Hz,
                duration > ~80-100 ms.
  * Rhonchi   : continuous, LOW-pitched (< ~300 Hz), snoring-like tonal energy.
  * Plus signal-physics scalars reused from M35: spectral flatness (tonality) and
    peak-to-average power ratio (impulsiveness/transients).

Every concept is a single interpretable float. The vector is intentionally small and
auditable. Thresholds have clinically-informed defaults but the CONTINUOUS scores are
what matter -- they are validated against ICBHI's crackle/wheeze cycle labels in
notebook 01 (Gate G2) via AUROC, i.e. "do the physics detectors track the labels?".

Dependencies: numpy, scipy only (no librosa/torch) -- so it runs anywhere and is
cheap to unit-test on synthetic signals (see tests/test_concepts_synthetic.py).

Author: OWMTL team scaffold, 2026-08. Breathing-phase proxy is intentionally simple;
a full phase detector (arXiv:1903.10251) can be dropped in later without changing the
concept interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal as sps

# ----------------------------------------------------------------------------------
# Canonical, ORDERED concept vocabulary. The order is load-bearing: the bottleneck
# consumes this vector, so keep it stable. Add new concepts at the END only.
# ----------------------------------------------------------------------------------
CONCEPT_NAMES = [
    "crackle_presence",          # [0..1] impulsive-transient score  -> maps to ICBHI 'crackle'
    "crackle_rate_hz",           # detected transient bursts per second
    "fine_crackle_ratio",        # fraction of transients that are 'fine' (short, high-freq)
    "coarse_crackle_ratio",      # fraction that are 'coarse' (long, low-freq)
    "wheeze_presence",           # [0..1] sustained narrow-band tonal score -> ICBHI 'wheeze'
    "wheeze_dominant_freq_hz",   # dominant tonal freq in 100-1000 Hz (0 if none)
    "wheeze_duration_ratio",     # fraction of cycle with sustained tonal energy
    "rhonchi_presence",          # [0..1] sustained LOW-freq (<300 Hz) tonal score
    "spectral_flatness",         # Wiener flatness (0 tonal .. 1 broadband) [M35]
    "papr_db",                   # peak-to-average power ratio in dB (impulsiveness) [M35]
    "inspiratory_energy_fraction",  # energy in first half of cycle (phase proxy)
    "transient_timing_centroid",    # [0..1] where transient energy sits (early..late)
    "dominant_freq_hz",          # spectral centroid in the physiological band (Hz)
    "low_high_freq_ratio",       # energy(<300 Hz) / energy(>=300 Hz)
]

N_CONCEPTS = len(CONCEPT_NAMES)

# Concepts whose CONTINUOUS score is directly validated against an ICBHI cycle label.
# (Gate G2 uses these to prove the extractors are clinically meaningful.)
CONCEPT_LABEL_MAP = {
    "crackle_presence": "crackle",  # ICBHI per-cycle crackle in {0,1}
    "wheeze_presence": "wheeze",    # ICBHI per-cycle wheeze in {0,1}
}


@dataclass
class ConceptConfig:
    sr: int = 16000
    n_fft: int = 512               # ~32 ms @ 16 kHz -- fine enough for transients
    hop: int = 128                 # ~8 ms
    fmin: float = 50.0
    fmax: float = 2000.0
    # Clinically-separated tonal bands: wheeze is high-pitched (musical, typically
    # >~400 Hz), rhonchi is low-pitched (snoring, <~250 Hz). A small gap avoids a
    # single mid tone firing both.
    wheeze_band: tuple = (200.0, 1000.0)
    rhonchi_band: tuple = (60.0, 250.0)
    # transient (crackle) detection -- deliberately selective so band noise does not
    # register as crackles.
    env_smooth_ms: float = 4.0
    transient_prominence: float = 5.0   # robust-z prominence a real transient must clear
    transient_min_height: float = 4.0   # robust-z envelope height at the peak
    transient_max_ms: float = 40.0      # crackles are impulsive (<~30-40 ms); wider bumps rejected
    fine_coarse_ms: float = 10.0        # duration threshold fine (<) vs coarse (>=)
    fine_coarse_hz: float = 500.0       # centre-freq threshold fine (>=) vs coarse (<)
    # tonality detection -- bandwidth-ROBUST: fraction of band energy in the single
    # strongest bin. A pure tone concentrates a large fraction regardless of band
    # width; broadband noise spreads it out.
    tonal_peakfrac: float = 0.25        # max-bin energy fraction to call a frame 'tonal'
    tonal_min_frames_ratio: float = 0.12  # min sustained fraction to count as continuous


def _safe(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).flatten()
    if y.size == 0:
        return np.zeros(1)
    y = y - np.mean(y)
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def _envelope(y: np.ndarray, sr: int, smooth_ms: float) -> np.ndarray:
    """Amplitude envelope via Hilbert magnitude, smoothed."""
    analytic = sps.hilbert(y)
    env = np.abs(analytic)
    w = max(1, int(sr * smooth_ms / 1000.0))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return env


def _power_spectrogram(y: np.ndarray, cfg: ConceptConfig):
    f, t, Z = sps.stft(y, fs=cfg.sr, nperseg=cfg.n_fft, noverlap=cfg.n_fft - cfg.hop,
                       boundary=None, padded=False)
    P = (np.abs(Z) ** 2)
    return f, t, P


def _band_mask(f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (f >= lo) & (f < hi)


def extract_concepts(y: np.ndarray, sr: int = 16000,
                     cfg: ConceptConfig | None = None) -> dict:
    """Return an ORDERED dict {concept_name: float} for one respiratory cycle.

    y  : mono waveform (any length; typically one cycle, ~0.5-4 s)
    sr : sample rate (resample to cfg.sr beforehand for consistency)
    """
    cfg = cfg or ConceptConfig(sr=sr)
    y = _safe(y)
    if y.size < cfg.n_fft:  # pad very short cycles
        y = np.pad(y, (0, cfg.n_fft - y.size))
    dur = y.size / cfg.sr

    f, t, P = _power_spectrogram(y, cfg)
    phys = _band_mask(f, cfg.fmin, cfg.fmax)
    Pp = P[phys]
    fp = f[phys]
    frame_energy = Pp.sum(axis=0) + 1e-12
    total_energy = frame_energy.sum() + 1e-12

    out = {name: 0.0 for name in CONCEPT_NAMES}

    # ---- Signal-physics scalars (M35) --------------------------------------------
    # Spectral flatness (Wiener): geometric mean / arithmetic mean of the band power,
    # averaged over frames. Low -> tonal, high -> broadband/noisy.
    with np.errstate(divide="ignore"):
        gm = np.exp(np.mean(np.log(Pp + 1e-12), axis=0))
    am = np.mean(Pp, axis=0) + 1e-12
    flat = np.clip(gm / am, 0.0, 1.0)
    out["spectral_flatness"] = float(np.average(flat, weights=frame_energy))

    # PAPR (impulsiveness) in dB
    p = y ** 2
    out["papr_db"] = float(10.0 * np.log10((np.max(p) + 1e-12) / (np.mean(p) + 1e-12)))

    # Spectral centroid (dominant freq) in the physiological band
    band_spec = Pp.sum(axis=1) + 1e-12
    out["dominant_freq_hz"] = float(np.sum(fp * band_spec) / np.sum(band_spec))

    # Low/high band energy ratio (<300 vs >=300 Hz)
    lo_e = P[_band_mask(f, cfg.fmin, 300.0)].sum()
    hi_e = P[_band_mask(f, 300.0, cfg.fmax)].sum()
    out["low_high_freq_ratio"] = float(lo_e / (hi_e + 1e-12))

    # ---- Crackles: transient detection on the envelope ---------------------------
    env = _envelope(y, cfg.sr, cfg.env_smooth_ms)
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-9
    env_norm = (env - med) / (1.4826 * mad)          # robust z-score
    # peaks = candidate crackle transients (selective: prominence AND height AND
    # impulsive width, so fluctuating band noise does not count as crackles)
    min_dist = int(cfg.sr * 0.005)                    # >=5 ms apart
    peaks, props = sps.find_peaks(env_norm, prominence=cfg.transient_prominence,
                                  height=cfg.transient_min_height,
                                  distance=max(1, min_dist))
    if len(peaks) > 0:
        widths_all, _, _, _ = sps.peak_widths(env_norm, peaks, rel_height=0.5)
        dur_ms_all = (widths_all / cfg.sr) * 1000.0
        keep = dur_ms_all <= cfg.transient_max_ms      # impulsive only
        peaks = peaks[keep]
        for k in list(props.keys()):
            try:
                props[k] = np.asarray(props[k])[keep]
            except Exception:
                pass
    n_tr = len(peaks)
    # transient "energy sharpness" -> presence score via a soft saturating map
    if n_tr > 0:
        widths, _, lefts, rights = sps.peak_widths(env_norm, peaks, rel_height=0.5)
        dur_ms = (widths / cfg.sr) * 1000.0
        # centre freq per transient: dominant physiological-band freq in a short window
        centre_hz = np.zeros(n_tr)
        half = cfg.n_fft // 2
        for i, pk in enumerate(peaks):
            a, b = max(0, pk - half), min(y.size, pk + half)
            seg = y[a:b] * np.hanning(b - a) if (b - a) > 1 else y[a:b]
            if seg.size >= 8:
                spec = np.abs(np.fft.rfft(seg, n=cfg.n_fft)) ** 2
                ff = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sr)
                bm = _band_mask(ff, cfg.fmin, cfg.fmax)
                centre_hz[i] = np.sum(ff[bm] * spec[bm]) / (np.sum(spec[bm]) + 1e-12)
        # fine vs coarse split
        is_fine = (dur_ms < cfg.fine_coarse_ms) & (centre_hz >= cfg.fine_coarse_hz)
        is_coarse = (~is_fine)
        out["crackle_rate_hz"] = float(n_tr / max(dur, 1e-6))
        out["fine_crackle_ratio"] = float(np.mean(is_fine)) if n_tr else 0.0
        out["coarse_crackle_ratio"] = float(np.mean(is_coarse)) if n_tr else 0.0
        # presence: saturating function of transient count weighted by prominence
        prom = props.get("prominences", np.ones(n_tr))
        strength = np.sum(np.tanh(prom / cfg.transient_prominence))
        out["crackle_presence"] = float(1.0 - np.exp(-strength / 2.0))
        # temporal centroid of transient energy (0 early .. 1 late)
        out["transient_timing_centroid"] = float(np.average(peaks / max(y.size - 1, 1),
                                                            weights=prom))
    # else: all crackle concepts stay 0.0

    # ---- Wheeze & rhonchi: sustained narrow-band tonality -------------------------
    def _tonal_scores(band):
        bm = _band_mask(f, band[0], band[1])
        if bm.sum() < 2:
            return 0.0, 0.0, 0.0
        Pb = P[bm]
        fb = f[bm]
        band_energy = Pb.sum(axis=0) + 1e-12
        n_bins = Pb.shape[0]
        uniform = 1.0 / n_bins                       # peak-fraction of flat noise
        peak_frac = Pb.max(axis=0) / band_energy
        # tonality normalised against the noise floor: ~0 for flat noise, ~1 for a
        # pure tone, INDEPENDENT of band width.
        tonality = np.clip((peak_frac - uniform) / (1.0 - uniform + 1e-9), 0.0, 1.0)
        sat = np.clip((tonality - 0.15) / (0.55 - 0.15), 0.0, 1.0)
        energetic = band_energy > np.median(band_energy)
        tonal_frames = (tonality > cfg.tonal_peakfrac) & energetic
        dur_ratio = float(np.mean(tonal_frames))
        presence = float(np.clip(np.average(sat, weights=band_energy), 0.0, 1.0))
        if dur_ratio < cfg.tonal_min_frames_ratio:      # not sustained -> damp
            presence *= dur_ratio / cfg.tonal_min_frames_ratio
        if tonal_frames.any():
            dom_bins = np.argmax(Pb[:, tonal_frames], axis=0)
            dom_hz = float(np.median(fb[dom_bins]))
        else:
            dom_hz = 0.0
        return presence, dom_hz, dur_ratio

    w_pres, w_hz, w_dur = _tonal_scores(cfg.wheeze_band)
    out["wheeze_presence"] = w_pres
    out["wheeze_dominant_freq_hz"] = w_hz
    out["wheeze_duration_ratio"] = w_dur

    r_pres, _r_hz, _r_dur = _tonal_scores(cfg.rhonchi_band)
    out["rhonchi_presence"] = r_pres

    # ---- Breathing-phase proxy (simple; swap in a real detector later) -----------
    half_idx = len(frame_energy) // 2
    first = frame_energy[:half_idx].sum()
    out["inspiratory_energy_fraction"] = float(first / total_energy)

    return out


def extract_concept_vector(y: np.ndarray, sr: int = 16000,
                           cfg: ConceptConfig | None = None) -> np.ndarray:
    """Return the concepts as an ordered numpy vector (len == N_CONCEPTS)."""
    d = extract_concepts(y, sr, cfg)
    return np.array([d[n] for n in CONCEPT_NAMES], dtype=np.float32)


def batch_extract(cycles, sr=16000, cfg: ConceptConfig | None = None) -> np.ndarray:
    """cycles: iterable of waveforms -> (N, N_CONCEPTS) float32 matrix."""
    return np.stack([extract_concept_vector(y, sr, cfg) for y in cycles], axis=0)
