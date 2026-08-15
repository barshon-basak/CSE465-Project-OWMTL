#!/usr/bin/env python3
"""CC2 CLI: run the ICBHI device-structure feasibility check (Gate G4).

Usage:
    python3 scripts/check_device_structure.py --wav_dir /path/to/ICBHI/audio \
        [--diagnosis /path/to/ICBHI_Challenge_diagnosis.txt] [--json out.json]

If run with no args it executes a synthetic self-test so you can see the output shape.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from owmtl.device_check import analyze, format_report


def _selftest():
    # synthetic ICBHI-style filenames: 4 devices, one patient (105) spans two devices
    files = [
        "101_1b1_Al_sc_Meditron.wav", "101_2b2_Ar_sc_Meditron.wav",
        "102_1b1_Pl_sc_LittC2SE.wav", "103_1b1_Tc_sc_Litt3200.wav",
        "104_1b1_Ll_sc_AKGC417L.wav", "105_1b1_Lr_sc_Meditron.wav",
        "105_2b2_Lr_sc_AKGC417L.wav",
    ]
    diag = {"101": "COPD", "102": "COPD", "103": "Healthy",
            "104": "URTI", "105": "Pneumonia"}
    # write a temp diagnosis file
    import tempfile
    d = tempfile.mkdtemp()
    dp = os.path.join(d, "diag.txt")
    with open(dp, "w") as fh:
        for k, v in diag.items():
            fh.write(f"{k}\t{v}\n")
    rep = analyze(files, dp)
    print(format_report(rep))
    print("\n(self-test — replace with --wav_dir on the real ICBHI folder)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default=None)
    ap.add_argument("--diagnosis", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.wav_dir:
        rep = _selftest()
    else:
        rep = analyze(args.wav_dir, args.diagnosis)
        print(format_report(rep))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
