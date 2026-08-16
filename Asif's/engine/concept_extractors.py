#!/usr/bin/env python3
"""
Physics-derived acoustic concept extractors for respiratory cycles — Gate G2.

WHAT THIS IS
------------
Per-cycle, label-free DSP that produces a small vector of *clinically-named* acoustic concepts.
This is the input side of the concept bottleneck: the diagnosis head sees only these numbers, so
they have to mean what their names say.

Refactored from M35's `VectorizedAcousticPhysicsLoss`, which computed two of these (spectral
flatness, temporal PAPR) as a *training regulariser* on log-mel spectrograms. Two changes:

  * They are now **standalone per-cycle measurements**, not a loss term.
  * They are computed from the **raw waveform**, not from a normalised log-mel via M35's
    `exp(spec * 3.0)` un-log approximation. That approximation was fine for a soft penalty but is
    not a defensible basis for a number the paper names "spectral flatness". Crackles are 5–15 ms
    events; at hop_length=160 (10 ms) a mel frame barely resolves one at all, which is the other
    reason this works on the waveform.

HONESTY ABOUT VALIDATION  (read before reporting any of these)
--------------------------------------------------------------
ICBHI annotates each cycle only for **crackle presence** and **wheeze presence**. So:

  VALIDATABLE against ICBHI labels ...... crackle_score, wheeze_score
  PROXIES with no ground truth .......... crackle_fine_ratio, crackle_rate_hz, wheeze_pitch_hz,
                                          rhonchi_score, inspiratory_fraction
  DESCRIPTIVE (no label concept) ........ spectral_flatness, temporal_papr

`CONCEPT_VALIDATION` below records this per concept, and the G2 notebook reports the two groups
separately. **Never write "fine crackle" for `crackle_fine_ratio` in the paper** — write
"physics-derived proxy for fine-crackle fraction". Naming a proxy after the thing it proxies is
how the ICBHI-metric problem started. If G0 returns a clinician, these proxies gain ground truth
and can be promoted.

CLINICAL GROUNDING
------------------
Thresholds follow the CORSA/ATS descriptive conventions for adventitious sounds:

  fine crackle    short (~5 ms two-cycle duration), higher centre frequency (~650 Hz)
  coarse crackle  longer (~10-15 ms), lower centre frequency (~350 Hz)
  wheeze          continuous >=100 ms, dominant frequency ~100-1000 Hz
  rhonchus        continuous >=100 ms, low pitched (<300 Hz)

These are *adult* conventions. `Gap7` already found the physics priors degrade on pediatric
airways (SPRSound) — expected, since children's airways are smaller and resonate higher. That
fragility is a finding to report, not a bug to tune away.

Dependencies: numpy only. STFT and band-pass are implemented directly so the module runs
anywhere (no librosa/scipy version coupling) and stays unit-testable on synthetic signals.
"""
import numpy as np

# name -> (validation status, one-line meaning)
CONCEPT_VALIDATION = {
    "crackle_score":        ("validatable", "strength of transient/discontinuous events"),
    "wheeze_score":         ("validatable", "strength of continuous tonal events 100-1000 Hz"),
    "crackle_fine_ratio":   ("proxy", "fraction of detected transients that are short+high-freq"),
    "crackle_rate_hz":      ("proxy", "detected transients per second"),
    "wheeze_pitch_hz":      ("proxy", "dominant tonal frequency when tonality is present"),
    "rhonchi_score":        ("proxy", "continuous tonal energy below 300 Hz"),
    "inspiratory_fraction": ("proxy", "share of cycle energy in its first (inspiratory) half"),
    "spectral_flatness":    ("descriptive", "Wiener flatness in the wheeze band (M35)"),
    "temporal_papr":        ("descriptive", "peak-to-average power ratio of the envelope (M35)"),
}

CONCEPT_NAMES = list(CONCEPT_VALIDATION)

# Paper-safe display names. The proxies are deliberately not called by the clinical term.
CONCEPT_DISPLAY = {
    "crackle_score":        "crackle presence (DSP)",
    "wheeze_score":         "wheeze presence (DSP)",
    "crackle_fine_ratio":   "proxy: fine-crackle fraction",
    "crackle_rate_hz":      "proxy: crackle rate (Hz)",
    "wheeze_pitch_hz":      "proxy: wheeze pitch (Hz)",
    "rhonchi_score":        "proxy: rhonchi (low-pitched continuous)",
    "inspiratory_fraction": "proxy: inspiratory energy fraction",
    "spectral_flatness":    "spectral flatness (wheeze band)",
    "temporal_papr":        "temporal PAPR",
}

# --- CORSA-derived constants -------------------------------------------------
FINE_MAX_WIDTH_MS = 8.0      # fine crackles are short; coarse run longer
FINE_MIN_CENTRE_HZ = 450.0   # and sit higher in frequency
WHEEZE_BAND = (100.0, 1000.0)
RHONCHI_BAND = (60.0, 300.0)
WHEEZE_MIN_MS = 100.0        # "continuous" by clinical convention
CRACKLE_BAND = (100.0, 2000.0)


# ============================================================ primitives
def _bandpass(x, sr, lo, hi, roll=0.3):
    """Zero-phase band-pass with raised-cosine edges (no filter-design dependency).

    The edges are TAPERED, not brick-wall. Zeroing FFT bins outright is a rectangular window in
    frequency, whose time-domain response is a sinc: every transient acquires ringing sidelobes.
    On the synthetic crackle train that produced a phantom detection ~10 ms after each real one
    and inflated crackle_rate_hz by 50%. A raised-cosine transition over `roll` x edge-frequency
    suppresses it.
    """
    n = len(x)
    if n == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    H = np.ones_like(f)

    lo_w = max(lo * roll, 1e-9)
    H[f < lo - lo_w] = 0.0
    m = (f >= lo - lo_w) & (f < lo)
    H[m] = 0.5 * (1.0 - np.cos(np.pi * (f[m] - (lo - lo_w)) / lo_w))

    hi_w = max(hi * roll, 1e-9)
    m = (f > hi) & (f <= hi + hi_w)
    H[m] = 0.5 * (1.0 + np.cos(np.pi * (f[m] - hi) / hi_w))
    H[f > hi + hi_w] = 0.0

    return np.fft.irfft(X * H, n=n)


def _frames(x, win, hop):
    """(n_frames, win) view; returns an empty array if the signal is shorter than one window."""
    if len(x) < win:
        return np.empty((0, win))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _stft_mag(x, sr, win_ms=40.0, hop_ms=10.0):
    """Magnitude STFT. Returns (mag [n_frames, n_bins], freqs)."""
    win = max(8, int(round(sr * win_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    fr = _frames(np.asarray(x, dtype=float), win, hop)
    if fr.shape[0] == 0:
        return np.zeros((0, win // 2 + 1)), np.fft.rfftfreq(win, 1.0 / sr)
    return np.abs(np.fft.rfft(fr * np.hanning(win)[None, :], axis=1)), np.fft.rfftfreq(win, 1.0 / sr)


def _envelope(x, sr, win_ms=2.0):
    """Short-window RMS envelope. 2 ms resolves a 5 ms fine crackle; a 10 ms mel hop does not."""
    win = max(4, int(round(sr * win_ms / 1000.0)))
    hop = max(1, win // 2)
    fr = _frames(np.asarray(x, dtype=float), win, hop)
    if fr.shape[0] == 0:
        return np.array([]), hop / sr
    return np.sqrt((fr ** 2).mean(axis=1)), hop / sr


def _robust_z(v):
    """Median/MAD standardisation — resists the very spikes we are hunting for."""
    if len(v) == 0:
        return v
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return (v - med) / (1.4826 * mad + 1e-12)


# ============================================================ detectors
def detect_crackles(audio, sr, z_thresh=3.5, refractory_ms=6.0, min_peak_frac=0.15):
    """Detect transient events. Returns a list of dicts: onset_s, width_ms, centre_hz.

    Crackles are discontinuous, explosive and brief. We band-pass to 100-2000 Hz, build a 2 ms
    RMS envelope, and take robustly-prominent peaks. Each peak's width is measured at half
    prominence and its centre frequency from the local spectrum, which is what separates fine
    (short, high) from coarse (long, low) per CORSA.

    `refractory_ms` merges detections closer together than a crackle can physiologically repeat.
    Without it a short high-frequency crackle splits into two counts: the RMS envelope window
    (2 ms) is close to the carrier period of a ~650 Hz fine crackle (1.5 ms), so the envelope
    still ripples at the carrier rate and crosses threshold twice. This inflated crackle_rate_hz
    by ~50% on the synthetic fine-crackle train, which is what caught it.
    """
    x = _bandpass(np.asarray(audio, dtype=float), sr, *CRACKLE_BAND)
    env, dt = _envelope(x, sr)
    if len(env) < 5:
        return []
    z = _robust_z(env)

    # A crackle is an EXPLOSIVE sound, so it must carry real energy -- not merely be a
    # statistical outlier. Without this floor the median/MAD z-score fires on numerical
    # residue whenever the inter-event signal is near-silent (MAD -> 0), which is exactly what
    # happened on the synthetic trains: 27 "crackles" detected in a 10-crackle signal.
    env_floor = min_peak_frac * float(env.max()) if env.size else 0.0

    events, i, n = [], 1, len(env)
    while i < n - 1:
        if (z[i] >= z_thresh and env[i] >= env_floor
                and z[i] >= z[i - 1] and z[i] > z[i + 1]):
            half = z[i] / 2.0
            a = i
            while a > 0 and z[a] > half:
                a -= 1
            b = i
            while b < n - 1 and z[b] > half:
                b += 1
            width_ms = (b - a) * dt * 1000.0

            c0, c1 = int(a * dt * sr), int(min(len(x), (b + 1) * dt * sr))
            centre = 0.0
            if c1 - c0 >= 16:
                seg = x[c0:c1] * np.hanning(c1 - c0)
                mag = np.abs(np.fft.rfft(seg))
                fq = np.fft.rfftfreq(len(seg), 1.0 / sr)
                if mag.sum() > 0:
                    centre = float((mag * fq).sum() / mag.sum())   # spectral centroid
            events.append({"onset_s": float(i * dt), "width_ms": float(width_ms),
                           "centre_hz": centre, "_z": float(z[i])})
            i = b + 1
        else:
            i += 1

    # Merge events inside the refractory window, keeping the most prominent of each group.
    merged = []
    for e in events:
        if merged and (e["onset_s"] - merged[-1]["onset_s"]) * 1000.0 < refractory_ms:
            if e["_z"] > merged[-1]["_z"]:
                merged[-1] = e
        else:
            merged.append(e)
    for e in merged:
        e.pop("_z", None)
    return merged


def detect_wheezes(audio, sr, band=WHEEZE_BAND, min_ms=WHEEZE_MIN_MS, prominence=3.0,
                   min_band_share=0.25):
    """Detect continuous tonal events. Returns dicts: start_s, duration_ms, freq_hz.

    A wheeze is a *sustained* narrowband peak. Per frame we find the strongest in-band bin and
    require two things: it stands out from the local in-band background (`prominence`), AND it
    is a real share of the frame's total spectral peak (`min_band_share`). A run of such frames
    at a stable frequency, lasting >= min_ms, is a wheeze.

    `min_band_share` exists because local prominence alone is not enough: for a pure 400 Hz tone
    the 60-300 Hz rhonchi band still contains a locally-dominant leakage bin, which made a clean
    wheeze register as a rhonchus. Requiring the in-band peak to be comparable to the global peak
    removes that -- caught by the synthetic-signal tests, which is what they are for.

    Note the bands overlap by clinical convention: a sustained 150 Hz tone is both "low-pitched
    wheeze" and "rhonchus", so both concepts firing on it is correct, not double counting.
    """
    mag, freqs = _stft_mag(audio, sr, win_ms=40.0, hop_ms=10.0)
    if mag.shape[0] == 0:
        return []
    sel = (freqs >= band[0]) & (freqs <= band[1])
    if not sel.any():
        return []

    hop_s = 0.010
    peak_f, tonal = np.zeros(mag.shape[0]), np.zeros(mag.shape[0], dtype=bool)
    for t in range(mag.shape[0]):
        row = mag[t, sel]
        if row.max() <= 0:
            continue
        k = int(np.argmax(row))
        peak_f[t] = freqs[sel][k]
        med = np.median(row) + 1e-12
        global_peak = mag[t].max() + 1e-12
        tonal[t] = ((row[k] / med) >= prominence) and ((row[k] / global_peak) >= min_band_share)

    out, t = [], 0
    while t < len(tonal):
        if not tonal[t]:
            t += 1
            continue
        s = t
        while (t + 1 < len(tonal) and tonal[t + 1]
               and abs(peak_f[t + 1] - peak_f[s]) <= 0.25 * max(peak_f[s], 1.0)):
            t += 1
        dur = (t - s + 1) * hop_s * 1000.0
        if dur >= min_ms:
            out.append({"start_s": float(s * hop_s), "duration_ms": float(dur),
                        "freq_hz": float(np.mean(peak_f[s:t + 1]))})
        t += 1
    return out


# ============================================================ scalar concepts
def spectral_flatness(audio, sr, band=WHEEZE_BAND):
    """Wiener flatness (geometric/arithmetic mean) in-band. Tone -> 0, noise -> 1. From M35."""
    mag, freqs = _stft_mag(audio, sr)
    if mag.shape[0] == 0:
        return 1.0
    sel = (freqs >= band[0]) & (freqs <= band[1])
    p = mag[:, sel] ** 2 + 1e-12
    if p.shape[1] == 0:
        return 1.0
    flat = np.exp(np.log(p).mean(axis=1)) / (p.mean(axis=1) + 1e-12)
    return float(np.clip(flat.mean(), 0.0, 1.0))


def temporal_papr(audio, sr):
    """Peak-to-average power ratio of the envelope. Transients push this up. From M35."""
    env, _ = _envelope(audio, sr)
    if len(env) == 0:
        return 1.0
    p = env ** 2
    return float(p.max() / (p.mean() + 1e-12))


def inspiratory_fraction(audio, sr):
    """Share of cycle energy in its first half.

    PROXY ONLY. ICBHI cycle annotations bound a whole respiratory cycle and do not mark the
    inspiration/expiration boundary, so there is no ground truth to check this against. The
    assumption — inspiration precedes expiration within an annotated cycle — is conventional but
    unverified here. A dedicated phase detector (arXiv:1903.10251) is the upgrade path.
    """
    env, _ = _envelope(audio, sr, win_ms=10.0)
    if len(env) < 2:
        return 0.5
    p = env ** 2
    tot = p.sum()
    return float(p[: len(p) // 2].sum() / tot) if tot > 0 else 0.5


# ============================================================ the vector
def extract_concepts(audio, sr):
    """Full per-cycle concept vector. Keys are exactly CONCEPT_NAMES, values are finite floats."""
    x = np.asarray(audio, dtype=float)
    if x.size == 0 or not np.any(np.isfinite(x)) or np.allclose(x, 0):
        return {k: 0.0 for k in CONCEPT_NAMES}
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    dur = max(len(x) / sr, 1e-6)

    cr = detect_crackles(x, sr)
    wz = detect_wheezes(x, sr)

    if cr:
        fine = [e for e in cr
                if e["width_ms"] <= FINE_MAX_WIDTH_MS and e["centre_hz"] >= FINE_MIN_CENTRE_HZ]
        fine_ratio = len(fine) / len(cr)
    else:
        fine_ratio = 0.0

    papr = temporal_papr(x, sr)
    # squashed so the concept is bounded in [0,1] like the others; 2.5 is M35's PAPR threshold
    crackle_score = float(np.clip(len(cr) / dur / 10.0, 0, 1) * 0.5
                          + np.clip((papr - 2.5) / 20.0, 0, 1) * 0.5)

    wheeze_ms = sum(w["duration_ms"] for w in wz)
    wheeze_score = float(np.clip(wheeze_ms / (dur * 1000.0), 0.0, 1.0))
    wheeze_pitch = float(np.mean([w["freq_hz"] for w in wz])) if wz else 0.0

    rh = detect_wheezes(x, sr, band=RHONCHI_BAND)
    rhonchi_ms = sum(w["duration_ms"] for w in rh)

    return {
        "crackle_score": crackle_score,
        "wheeze_score": wheeze_score,
        "crackle_fine_ratio": float(fine_ratio),
        "crackle_rate_hz": float(len(cr) / dur),
        "wheeze_pitch_hz": wheeze_pitch,
        "rhonchi_score": float(np.clip(rhonchi_ms / (dur * 1000.0), 0.0, 1.0)),
        "inspiratory_fraction": inspiratory_fraction(x, sr),
        "spectral_flatness": spectral_flatness(x, sr),
        "temporal_papr": float(papr),
    }


def concepts_to_vector(d):
    """Dict -> fixed-order array, so downstream matrices always line up with CONCEPT_NAMES."""
    return np.array([float(d.get(k, 0.0)) for k in CONCEPT_NAMES], dtype=np.float32)


if __name__ == "__main__":
    sr = 16000
    t = np.arange(sr) / sr
    demos = {
        "400 Hz tone (wheeze-like)": np.sin(2 * np.pi * 400 * t),
        "150 Hz tone (rhonchi-like)": np.sin(2 * np.pi * 150 * t),
        "white noise": np.random.RandomState(0).randn(sr) * 0.1,
    }
    def _crackle_train(width_ms, n=12, seed=0, carrier_hz=650.0):
        sig = np.zeros(sr)
        w = int(width_ms / 1000.0 * sr)
        rs = np.random.RandomState(seed)
        for c in np.linspace(0, sr - w - 1, n).astype(int):
            burst = np.sin(2 * np.pi * carrier_hz * np.arange(w) / sr) * np.hanning(w)
            sig[c:c + w] += burst * (1 + 0.1 * rs.randn())
        return sig

    demos["fine crackles (5 ms, 650 Hz)"] = _crackle_train(5.0, carrier_hz=650.0)
    demos["coarse crackles (15 ms, 300 Hz)"] = _crackle_train(15.0, carrier_hz=300.0, seed=1)

    print(f"{'signal':<28}" + "".join(f"{k[:11]:>13}" for k in CONCEPT_NAMES))
    for name, sig in demos.items():
        c = extract_concepts(sig, sr)
        print(f"{name:<28}" + "".join(f"{c[k]:>13.3f}" for k in CONCEPT_NAMES))
