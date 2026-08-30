#!/usr/bin/env python3
"""
Make M22 actually FIND M3's corrected results file.

    python3 patch_m3_lookup.py [--dry-run]

THE PROBLEM
-----------
M22's comparison against M3 only searched four fixed paths:

    <parent>/M3/results/results_M3.json
    <parent>/M3/results_M3.json
    /content/drive/MyDrive/OWMTL/M3/results/results_M3.json
    /content/drive/MyDrive/OWMTL/M3/results_M3.json

But drive_setup.py's RUN_MODE="auto" writes each run to a VERSIONED folder -- M2's landed in
`OWMTL/M2_v3/results`, so M3's is in `OWMTL/M3_v<n>/results`, which none of those four patterns
match. Simulating the cell against the real results_M3.json (2026-08-29) confirmed it: the
lookup misses, M3_REFERENCE silently keeps its HARDCODED pre-correction values (ICBHI 0.6984,
11-patient split), and the ablation guard then correctly refuses to print a SpecAugment verdict.

Net effect: the run costs a full GPU training cycle and produces "COMPARISON NOT VALID" -- the
guard doing its job, but too late to be useful.

THE FIX
-------
Glob recursively for any `results_M3.json` under the project dir, Drive's OWMTL tree, and
/content, then choose deliberately rather than taking whatever comes first:

  1. prefer a file whose `split_method` MATCHES this run's split -- that is the run M22 is
     actually entitled to compare against;
  2. among those, take the most recently modified;
  3. if none match, still load the newest so the table renders, and let the existing guard
     refuse the verdict and say why.

Rule 1 matters more than recency: a newer M3 run on the WRONG split is worse than an older one
on the right split, and picking by mtime alone would silently prefer it.

Idempotent.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_M3_LOOKUP_V1"
PATH = "Asif's/M22/M22_mobilenet_specaugment.ipynb"

OLD = '''for _p in [
    os.path.join(os.path.dirname(BASE_DIR), "M3", "results", "results_M3.json"),
    os.path.join(os.path.dirname(BASE_DIR), "M3", "results_M3.json"),
    "/content/drive/MyDrive/OWMTL/M3/results/results_M3.json",
    "/content/drive/MyDrive/OWMTL/M3/results_M3.json",
]:'''

NEW = '''# [''' + MARKER + '''] Versioned run folders (OWMTL/M3_v2/results/...) meant the old
# fixed-path list never matched, so this silently used stale hardcoded M3 numbers. Search
# broadly, then PREFER a run whose split_method equals this run's split -- recency alone would
# happily pick a newer M3 trained on the wrong split.
import glob as _glob

_m3_found = []
for _pat in (
    os.path.join(os.path.dirname(BASE_DIR), "M3*", "**", "results_M3.json"),
    os.path.join(os.path.dirname(BASE_DIR), "**", "results_M3.json"),
    "/content/drive/MyDrive/OWMTL/**/results_M3.json",
    "/content/**/results_M3.json",
):
    try:
        _m3_found.extend(_glob.glob(_pat, recursive=True))
    except Exception:
        pass
_m3_found = sorted(set(_m3_found))

_m3_ranked = []
for _c in _m3_found:
    try:
        _cj = json.load(open(_c))
    except Exception:
        continue
    _csplit = _owmtl_find_key(_cj, "split_method")
    _m3_ranked.append((_csplit == THIS_SPLIT, os.path.getmtime(_c), _c))
# split match first, then newest
_m3_ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)

if _m3_found:
    print(f"[info] found {len(_m3_found)} results_M3.json candidate(s); "
          f"{sum(1 for r in _m3_ranked if r[0])} on this split ({THIS_SPLIT})")
    for _match, _mt, _c in _m3_ranked[:5]:
        print(f"         {'SPLIT-MATCH' if _match else 'other split'}  {_c}")
else:
    print("[info] no results_M3.json found by glob — will fall back to recorded M3 values")

for _p in [_r[2] for _r in _m3_ranked]:'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path = os.path.join(REPO, PATH)
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print("[M22] already patched — skipped")
        return
    hit = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if OLD in s:
            c["source"] = s.replace(OLD, NEW, 1).splitlines(keepends=True)
            hit += 1
    if hit != 1:
        print(f"[M22] FAILED — expected 1 match, got {hit}")
        raise SystemExit(1)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "M22", "exec")
    if not a.dry_run:
        json.dump(nb, open(path, "w"), indent=1)
    print(f"[M22] {'would patch' if a.dry_run else 'patched'} — M3 lookup now glob+split-aware")


if __name__ == "__main__":
    main()
