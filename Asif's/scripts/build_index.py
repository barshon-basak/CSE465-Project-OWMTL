#!/usr/bin/env python3
"""
Generate the group's shared data contract: cycles_index_v1.csv + split_v1.json.

Run this once, commit both artifacts, and never regenerate them mid-project
without a version bump — every member's numbers are only comparable because they
all read the same two files.

    python scripts/build_index.py \
        --data-root /content/Respiratory_Sound_Database \
        --out splits/

Standard library plus nothing. No audio is decoded, so this takes a few seconds
and can be run on a laptop with the annotation .txt files alone.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from owmtl import icbhi  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="directory containing the ICBHI audio/annotation files")
    ap.add_argument("--out", default="splits",
                    help="output directory for the two artifacts (default: splits/)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.40,
                    help="fraction of known-disease patients held out for test")
    ap.add_argument("--calib-frac", type=float, default=0.15,
                    help="fraction of remaining known patients reserved for "
                         "calibration (Member D). 0 to disable.")
    ap.add_argument("--version", default="v1", help="artifact version suffix")
    ap.add_argument("--no-strict", action="store_true",
                    help="allow diagnosis counts that differ from the full corpus")
    args = ap.parse_args()

    print(f"Locating ICBHI under {args.data_root} ...")
    paths = icbhi.locate(args.data_root)
    print(f"  audio_dir           : {paths.audio_dir}")
    print(f"  diagnosis_file      : {paths.diagnosis_file}")
    print(f"  official_split_file : {paths.official_split_file or 'NOT FOUND'}")
    if paths.official_split_file is None:
        print(
            "\n  NOTE: the ICBHI challenge split file is absent from this dataset\n"
            "  copy (the vbookshelf Kaggle mirror omits it). The 'official' scheme\n"
            "  will be unavailable and literature-comparable numbers cannot be\n"
            "  produced from this copy. The 'owmtl' scheme is unaffected.\n"
        )

    print("\nParsing annotations ...")
    rows = icbhi.build_cycle_index(paths)
    print(f"  {len(rows)} cycles from {len({r.stem for r in rows})} recordings")

    official = (
        icbhi.read_official_split(paths.official_split_file)
        if paths.official_split_file
        else None
    )

    print("Building split ...")
    split = icbhi.build_split(
        rows,
        seed=args.seed,
        test_frac=args.test_frac,
        calib_frac=args.calib_frac,
        official_by_stem=official,
        strict_counts=not args.no_strict,
    )
    payload = icbhi.split_to_dict(split, rows, paths)

    index_path = os.path.join(args.out, f"cycles_index_{args.version}.csv")
    split_path = os.path.join(args.out, f"split_{args.version}.json")
    icbhi.write_cycle_index(rows, index_path)
    icbhi.write_split(payload, split_path)

    print()
    print(icbhi.summarise(payload))
    print()

    problems = icbhi.validate(payload, [r.__dict__ for r in rows])
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        print("\nArtifacts were written but should not be committed until these "
              "are resolved.")
        return 1

    print("Validation passed.")
    print(f"\nWrote:\n  {index_path}\n  {split_path}")
    print("\nCommit both files. Every member reads them via:")
    print("    from owmtl import icbhi")
    print(f"    split = icbhi.load_split('{split_path}')")
    print(f"    index = icbhi.read_cycle_index('{index_path}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
