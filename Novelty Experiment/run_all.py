"""
run_all.py - run every novelty experiment that can run here, and say why the rest cannot.

    python run_all.py                          # the CPU-only subset (N2..N8)
    python run_all.py --features M2_features.npy --sprsound_dir <dir> --audio_dir <dir>

Writes results/RUN_SUMMARY.json and prints one line per experiment. Nothing here reruns a
completed experiment cheaply-but-differently: each script owns its own output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, extra args it understands, one-line what-it-needs)
EXPERIMENTS = [
    ("N1_fm_concept_probing.py", ["--stage", "probe"],
     "GPU + ICBHI audio for --stage embed/lora; probe alone needs cached embeddings"),
    ("N2_concept_space_osr.py", [], "CPU only"),
    ("N3_prototypical_fewshot.py", [], "CPU only"),
    ("N4_honest_operating_point.py", [], "CPU only"),
    ("N5_leakage_audit.py", [], "CPU only (full I(y;f|c) needs --features)"),
    ("N6_physics_bottleneck.py", [], "CPU only (3 of 4 modes need --features)"),
    ("N7_clinician_intervention.py", [], "CPU only"),
    ("N8_pediatric_fragility.py", [], "CPU only (full test needs --sprsound_dir)"),
    ("N9_clinician_reliability.py", [],
     "CPU only; needs the returned clinician sheet"),
    ("N10_gate_vs_clinician.py", [],
     "CPU only; needs the clinician sheet + ICBHI audio (ICBHI_AUDIO_DIR)"),
]

PASSTHROUGH = {"--features": ("N1", "N2", "N3", "N4", "N5", "N6", "N7"),
               "--sprsound_dir": ("N8",),
               "--audio_dir": ("N1",)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--sprsound_dir", default=None)
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these, e.g. --only N2 N6")
    ap.add_argument("--summary-only", dest="summary_only", action="store_true",
                    help="rebuild RUN_SUMMARY.json from the existing results/ files "
                         "without re-running anything (N5 alone costs ~25 min to redo)")
    args = ap.parse_args()

    summary = []
    for script, extra, needs in EXPERIMENTS:
        tag = script.split("_")[0]
        if args.only and tag not in args.only:
            continue
        cmd = [sys.executable, os.path.join(HERE, script)] + extra
        for flag, applies in PASSTHROUGH.items():
            val = getattr(args, flag.lstrip("-"))
            if val and tag in applies:
                cmd += [flag, val]

        if args.summary_only:
            r = subprocess.CompletedProcess(cmd, 0)
        else:
            print(f"\n{'=' * 78}\n>>> {tag}  ({needs})\n{'=' * 78}")
            r = subprocess.run(cmd, cwd=HERE)
        # A blocked experiment writes <id>_BLOCKED.json instead, and that is still a
        # result - "could not run, here is exactly what it needs" is auditable.
        status = "NOT WRITTEN"
        for suffix in ("", "_BLOCKED"):
            out = os.path.join(HERE, "results", f"{script[:-3]}{suffix}.json")
            if os.path.isfile(out):
                try:
                    status = json.load(open(out, encoding="utf-8")).get("status", "OK")
                except Exception:
                    status = "UNREADABLE"
                break
        summary.append({"experiment": tag, "script": script,
                        "exit_code": r.returncode, "status": status, "needs": needs})

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for s in summary:
        print(f"  {s['experiment']:4s} exit={s['exit_code']}  status={s['status']:<16s} "
              f"{s['needs']}")
    path = os.path.join(HERE, "results", "RUN_SUMMARY.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
