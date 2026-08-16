"""Synthetic sanity checks for the concept extractors.

Proves the DSP discriminates the clinical sound classes BEFORE touching ICBHI:
  crackle  -> high crackle_presence / PAPR, high spectral_flatness (broadband), low wheeze
  wheeze   -> high wheeze_presence, dominant freq ~ tone, low spectral_flatness, low crackle
  rhonchi  -> high rhonchi_presence at low freq
  normal   -> low crackle & wheeze
Run: python3 tests/test_concepts_synthetic.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from owmtl.concept_extractors import extract_concepts, ConceptConfig

SR = 16000
rng = np.random.default_rng(42)  # fixed seed (no Math.random equivalent needed)


def make_normal(dur=2.0):
    n = int(SR * dur)
    x = rng.standard_normal(n)
    # colour it to the physiological band
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [80 / (SR / 2), 900 / (SR / 2)], btype="band")
    return filtfilt(b, a, x) * 0.2


def make_wheeze(dur=2.0, f0=400.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * f0 * t) * np.hanning(n)  # sustained musical tone
    return make_normal(dur) * 0.3 + tone


def make_rhonchi(dur=2.0, f0=150.0):
    return make_wheeze(dur, f0=f0)


def make_crackle(dur=2.0, n_clicks=6):
    base = make_normal(dur) * 0.3
    n = base.size
    for _ in range(n_clicks):
        c = rng.integers(int(0.1 * n), int(0.9 * n))
        w = int(SR * 0.004)  # ~4 ms fine crackle
        env = np.exp(-0.5 * ((np.arange(-w, w)) / (w / 3)) ** 2)
        click = env * np.sin(2 * np.pi * 650 * np.arange(-w, w) / SR)
        a, b = max(0, c - w), min(n, c + w)
        base[a:b] += 1.5 * click[: b - a]
    return base


def summarize(name, y):
    d = extract_concepts(y, SR)
    print(f"\n[{name}] "
          f"crackle={d['crackle_presence']:.2f} rate={d['crackle_rate_hz']:.1f} "
          f"fine={d['fine_crackle_ratio']:.2f} | "
          f"wheeze={d['wheeze_presence']:.2f} wf={d['wheeze_dominant_freq_hz']:.0f}Hz "
          f"wdur={d['wheeze_duration_ratio']:.2f} | "
          f"rhonchi={d['rhonchi_presence']:.2f} | "
          f"flat={d['spectral_flatness']:.3f} papr={d['papr_db']:.1f}dB")
    return d


def main():
    normal = summarize("normal", make_normal())
    wheeze = summarize("wheeze@400", make_wheeze(f0=400))
    rhonchi = summarize("rhonchi@150", make_rhonchi(f0=150))
    crackle = summarize("crackle", make_crackle())

    print("\n--- assertions ---")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS" if cond else "FAIL"), msg)
        ok = ok and cond

    # Crackle discriminates on transient/impulsiveness
    check(crackle["crackle_presence"] > max(normal["crackle_presence"], wheeze["crackle_presence"]),
          "crackle_presence highest for crackle signal")
    check(crackle["papr_db"] > normal["papr_db"], "PAPR higher for crackle than normal")
    # Wheeze discriminates on tonality
    check(wheeze["wheeze_presence"] > max(normal["wheeze_presence"], crackle["wheeze_presence"]),
          "wheeze_presence highest for wheeze signal")
    check(abs(wheeze["wheeze_dominant_freq_hz"] - 400) < 80,
          "wheeze dominant freq near 400 Hz")
    check(wheeze["spectral_flatness"] < normal["spectral_flatness"],
          "tonal wheeze has lower spectral flatness than noisy normal")
    # Rhonchi = low-freq tonal
    check(rhonchi["rhonchi_presence"] > normal["rhonchi_presence"],
          "rhonchi_presence higher for low tone than normal")
    check(rhonchi["dominant_freq_hz"] < wheeze["dominant_freq_hz"],
          "rhonchi dominant freq lower than wheeze")

    print("\nRESULT:", "ALL PASS ✅" if ok else "SOME FAILED ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
