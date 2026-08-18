#!/usr/bin/env python3
"""
Synthetic-signal validation for concept_extractors.py.

The point: build signals whose acoustic ground truth we *construct*, then assert the extractors
report it. Running on real ICBHI audio only tells you the code executes; it cannot tell you a
detector named "crackle" responds to crackles. That is what these do.

Run: python3 test_concept_extractors.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concept_extractors import (CONCEPT_NAMES, CONCEPT_VALIDATION, concepts_to_vector,
                                detect_crackles, detect_wheezes, extract_concepts,
                                spectral_flatness, temporal_papr)

SR = 16000
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def tone(hz, dur=1.0, sr=SR):
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * hz * t)


def noise(dur=1.0, sr=SR, seed=0):
    return np.random.RandomState(seed).randn(int(sr * dur)) * 0.1


def crackle_train(width_ms, n=12, carrier_hz=650.0, dur=1.0, sr=SR, seed=0):
    sig = np.zeros(int(sr * dur))
    w = int(width_ms / 1000.0 * sr)
    rs = np.random.RandomState(seed)
    for c in np.linspace(0, len(sig) - w - 1, n).astype(int):
        burst = np.sin(2 * np.pi * carrier_hz * np.arange(w) / sr) * np.hanning(w)
        sig[c:c + w] += burst * (1 + 0.1 * rs.randn())
    return sig


# ---------------------------------------------------------------- tonal concepts
print("\n--- wheeze / tonality ---")
c_tone = extract_concepts(tone(400), SR)
c_noise = extract_concepts(noise(), SR)
check("400 Hz tone -> high wheeze_score", c_tone["wheeze_score"] > 0.8,
      f"{c_tone['wheeze_score']:.3f}")
check("400 Hz tone -> pitch recovered", abs(c_tone["wheeze_pitch_hz"] - 400) < 25,
      f"{c_tone['wheeze_pitch_hz']:.1f} Hz")
check("noise -> no wheeze", c_noise["wheeze_score"] < 0.2, f"{c_noise['wheeze_score']:.3f}")
check("tone flatness << noise flatness",
      c_tone["spectral_flatness"] < 0.1 < c_noise["spectral_flatness"],
      f"tone {c_tone['spectral_flatness']:.3f} vs noise {c_noise['spectral_flatness']:.3f}")

for hz in (200, 400, 700):
    got = extract_concepts(tone(hz), SR)["wheeze_pitch_hz"]
    check(f"pitch tracks {hz} Hz", abs(got - hz) < 30, f"{got:.1f}")

print("\n--- rhonchi band (regression: 400 Hz must NOT register) ---")
check("150 Hz tone -> rhonchi fires", extract_concepts(tone(150), SR)["rhonchi_score"] > 0.8)
check("400 Hz tone -> rhonchi does NOT fire", c_tone["rhonchi_score"] < 0.15,
      f"{c_tone['rhonchi_score']:.3f} (was 0.970 before the min_band_share fix)")
check("800 Hz tone -> rhonchi does NOT fire",
      extract_concepts(tone(800), SR)["rhonchi_score"] < 0.15)
check("150 Hz fires BOTH wheeze and rhonchi (overlapping bands, by convention)",
      extract_concepts(tone(150), SR)["wheeze_score"] > 0.8)

# ---------------------------------------------------------------- transient concepts
print("\n--- crackles: detection and fine/coarse separation ---")
fine = extract_concepts(crackle_train(5.0, carrier_hz=650.0), SR)
coarse = extract_concepts(crackle_train(15.0, carrier_hz=300.0, seed=1), SR)
check("fine crackles -> high crackle_score", fine["crackle_score"] > 0.7,
      f"{fine['crackle_score']:.3f}")
check("coarse crackles -> high crackle_score", coarse["crackle_score"] > 0.7,
      f"{coarse['crackle_score']:.3f}")
check("fine train -> high fine_ratio", fine["crackle_fine_ratio"] > 0.6,
      f"{fine['crackle_fine_ratio']:.3f}")
check("coarse train -> low fine_ratio", coarse["crackle_fine_ratio"] < 0.3,
      f"{coarse['crackle_fine_ratio']:.3f}")
check("fine/coarse are separated", fine["crackle_fine_ratio"] - coarse["crackle_fine_ratio"] > 0.4,
      f"{fine['crackle_fine_ratio']:.2f} vs {coarse['crackle_fine_ratio']:.2f}")
check("tone -> no crackles", c_tone["crackle_score"] < 0.1, f"{c_tone['crackle_score']:.3f}")
check("crackle rate ~ construction (12 events/s)",
      8 <= coarse["crackle_rate_hz"] <= 16, f"{coarse['crackle_rate_hz']:.1f}/s")
check("transients -> PAPR >> tone PAPR",
      fine["temporal_papr"] > 5 * c_tone["temporal_papr"],
      f"{fine['temporal_papr']:.1f} vs {c_tone['temporal_papr']:.1f}")

print("\n--- detector internals ---")
ev = detect_crackles(crackle_train(5.0, n=10), SR)
check("detects roughly the constructed number of events", 6 <= len(ev) <= 14, f"{len(ev)}")
check("measured widths are short for a 5 ms train",
      np.median([e["width_ms"] for e in ev]) < 12,
      f"median {np.median([e['width_ms'] for e in ev]):.1f} ms")
wz = detect_wheezes(tone(400), SR)
check("one sustained wheeze event for a continuous tone", len(wz) == 1, f"{len(wz)}")
check("its duration ~ the signal length", wz[0]["duration_ms"] > 900,
      f"{wz[0]['duration_ms']:.0f} ms")
check("short tone burst (<100ms) is NOT a wheeze", len(detect_wheezes(tone(400, dur=0.05), SR)) == 0)

# ---------------------------------------------------------------- robustness
print("\n--- degenerate inputs ---")
for name, sig in (("empty", np.array([])), ("all zeros", np.zeros(SR)),
                  ("single sample", np.array([1.0])), ("very short", np.zeros(50))):
    c = extract_concepts(sig, SR)
    check(f"{name} -> no crash, all finite",
          set(c) == set(CONCEPT_NAMES) and all(np.isfinite(v) for v in c.values()))

nan_sig = tone(400).copy(); nan_sig[100:110] = np.nan
c = extract_concepts(nan_sig, SR)
check("NaNs -> finite output", all(np.isfinite(v) for v in c.values()))
inf_sig = tone(400).copy(); inf_sig[50] = np.inf
check("Infs -> finite output",
      all(np.isfinite(v) for v in extract_concepts(inf_sig, SR).values()))

print("\n--- contract ---")
c = extract_concepts(tone(400), SR)
check("keys exactly match CONCEPT_NAMES", list(c) == CONCEPT_NAMES)
v = concepts_to_vector(c)
check("vector length matches", len(v) == len(CONCEPT_NAMES))
check("vector order matches names",
      all(abs(v[i] - c[k]) < 1e-6 for i, k in enumerate(CONCEPT_NAMES)))
check("vector is float32", v.dtype == np.float32)

bounded = ["crackle_score", "wheeze_score", "crackle_fine_ratio", "rhonchi_score",
           "inspiratory_fraction", "spectral_flatness"]
for sig in (tone(400), noise(), crackle_train(5.0), crackle_train(15.0)):
    cc = extract_concepts(sig, SR)
    for k in bounded:
        if not (0.0 <= cc[k] <= 1.0):
            check(f"{k} stays in [0,1]", False, f"{cc[k]}")
            break
    else:
        continue
    break
else:
    check("all bounded concepts stay in [0,1] across signal types", True)

print("\n--- validation-status bookkeeping ---")
check("every concept has a validation status",
      set(CONCEPT_VALIDATION) == set(CONCEPT_NAMES))
check("exactly 2 concepts are validatable against ICBHI labels",
      sum(1 for s, _ in CONCEPT_VALIDATION.values() if s == "validatable") == 2,
      str([k for k, (s, _) in CONCEPT_VALIDATION.items() if s == "validatable"]))
check("crackle_score and wheeze_score are the validatable pair",
      CONCEPT_VALIDATION["crackle_score"][0] == "validatable"
      and CONCEPT_VALIDATION["wheeze_score"][0] == "validatable")

print("\n--- determinism ---")
sig = crackle_train(7.0, seed=5)
check("same input -> identical output",
      concepts_to_vector(extract_concepts(sig, SR)).tolist()
      == concepts_to_vector(extract_concepts(sig, SR)).tolist())

print("\n--- amplitude invariance (recording gain must not change the concepts) ---")
a = extract_concepts(crackle_train(5.0), SR)
b = extract_concepts(crackle_train(5.0) * 10.0, SR)
for k in ("crackle_fine_ratio", "crackle_rate_hz", "spectral_flatness", "temporal_papr"):
    # relative tolerance: the band-pass FFT round-trip introduces ~1e-5 relative float error on
    # a scaled input, which is numerical noise rather than a gain dependency
    check(f"{k} is gain-invariant", abs(a[k] - b[k]) <= 1e-4 * max(1e-6, abs(a[k])),
          f"{a[k]:.6f} vs {b[k]:.6f}")

n = sum(1 for _, ok, _ in CHECKS if ok)
print(f"\n{'=' * 72}\nRESULT: {n}/{len(CHECKS)} checks passed\n{'=' * 72}")
if n != len(CHECKS):
    for nm, ok, d in CHECKS:
        if not ok:
            print(f"  FAIL: {nm}  {d}")
    sys.exit(1)
