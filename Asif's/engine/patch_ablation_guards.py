#!/usr/bin/env python3
"""
Stop M3 and M22 from drawing ablation conclusions across a split boundary.

    python3 patch_ablation_guards.py [--dry-run]

THE PROBLEM
-----------
M22 exists to answer one question: does SpecAugment help? It answers it by subtracting M3's
numbers from its own and printing "SpecAugment HELPED" / "HURT". M3 does the same against M4 and
M2. Both read `icbhi_score` -- the macro alias -- and neither checks that the other run used the
same data split.

After the corrected-split re-runs this becomes actively wrong. If M22 is re-run before M3, it
falls back to HARDCODED M3 numbers from the pre-correction era (11-test-patient split, macro
metric) and prints a confident causal claim about SpecAugment that is actually measuring the
split change. The fallback announces itself with a single `[info]` line that is trivial to miss
in Colab output.

THE FIX
-------
1. Read `icbhi_score_official` when present, `icbhi_score` only as a legacy fallback, and say
   which was used.
2. Extract each run's `split_method` and compare. If the two runs are not on the same split -- or
   if the split of either is unknown -- the delta is still shown but the HELPED/HURT verdict is
   REPLACED by a refusal explaining what to re-run. A comparison that cannot be trusted should
   not produce a sentence that reads like a finding.

Idempotent.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_ABLATION_GUARD_V1"

HELPER = '''
# ---- ablation comparability guard  [''' + MARKER + '''] ----------------------------
def _owmtl_find_key(obj, key):
    """Recursively find `key` anywhere in a results JSON (nesting has changed over time)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _owmtl_find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _owmtl_find_key(v, key)
            if r is not None:
                return r
    return None


def _owmtl_official(metrics, label):
    """Official ICBHI score from a metrics dict, falling back to the legacy macro alias."""
    if isinstance(metrics, dict) and "icbhi_score_official" in metrics:
        return float(metrics["icbhi_score_official"]), "official"
    v = (metrics or {}).get("icbhi_score")
    print(f"[warn] {label}: no icbhi_score_official — using the NON-STANDARD macro value. "
          f"Re-run {label} with patch_official_metric.py applied before reporting this.")
    return (float(v) if v is not None else float("nan")), "macro_NONSTANDARD"


def _owmtl_comparable(split_a, split_b):
    a, b = (split_a or "?"), (split_b or "?")
    return (a != "?" and b != "?" and a == b), a, b


THIS_SPLIT = None
try:
    THIS_SPLIT = ("official_icbhi_60_40_patient_disjoint"
                  if 'OVERLAP_POLICY' in dir() and OVERLAP_POLICY == "drop_from_train"
                  else "official_icbhi_60_40_verbatim")
except Exception:
    pass
'''

M22_OLD_UPDATE = '''                "sp": _b["specificity_macro"], "icbhi": _b["icbhi_score"],'''
M22_NEW_UPDATE = '''                "sp": _b["specificity_macro"],
                "icbhi": _owmtl_official(_b, "M3")[0],
                "icbhi_kind": _owmtl_official(_b, "M3")[1],
                "split_method": _owmtl_find_key(_m3, "split_method"),'''

M22_OLD_VERDICT = '''print("\\nREADING THE RESULT")
print("-" * 88)
if d_icbhi > 0.01 and d_f1 > 0.01:'''
M22_NEW_VERDICT = '''print("\\nREADING THE RESULT")
print("-" * 88)

# [''' + MARKER + '''] Refuse a causal verdict when the two runs are not on the same split.
_ok, _sa, _sb = _owmtl_comparable(M3_REFERENCE.get("split_method"), THIS_SPLIT)
if not _ok:
    print("*** COMPARISON NOT VALID — no SpecAugment verdict printed. ***")
    print(f"    M3  split : {_sa}")
    print(f"    M22 split : {_sb}")
    print("    The delta above mixes two different data splits, so it measures the split")
    print("    change as much as SpecAugment. M3's hardcoded fallback numbers predate the")
    print("    split correction entirely.")
    print("    FIX: re-run M3 on this split first so results_M3.json is regenerated, then")
    print("    re-run this cell. The verdict appears automatically once the splits match.")
elif M3_REFERENCE.get("icbhi_kind") == "macro_NONSTANDARD":
    print("*** COMPARISON NOT VALID — M3's file predates the official-metric fix. ***")
    print("    Re-run M3 with patch_official_metric.py applied, then re-run this cell.")
elif d_icbhi > 0.01 and d_f1 > 0.01:'''

M3_OLD_M4 = '''d_icbhi = final_metrics["icbhi_score"] - m4["icbhi"]'''
M3_NEW_M4 = '''# [''' + MARKER + '''] M4's row is a transcribed pre-correction number on another split.
d_icbhi = final_metrics["icbhi_score_official"] - m4["icbhi"]
print("[warn] M4's reference values are pre-correction (macro metric, different split) and its "
      "results_M4.json is not in the repo. Treat any M3-vs-M4 delta as INDICATIVE ONLY until "
      "M4 is re-run — see PAPER_OUTLINE.md, transformer baseline.")'''

M3_OLD_M2 = '''    d_m2 = final_metrics["icbhi_score"] - _m2["icbhi"]'''
M3_NEW_M2 = '''    d_m2 = final_metrics["icbhi_score_official"] - _m2["icbhi"]
    if not _owmtl_comparable(_m2.get("split_method"), THIS_SPLIT)[0]:
        print("[warn] M2's numbers are not on this split — the M3-vs-M2 delta below is NOT a "
              "valid pretraining comparison. Re-run M2 on this split first.")'''

PATHS = {"M3": "Asif's/M3/M3_lightweight_backbone.ipynb",
         "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb"}
SUBS = {"M3": [(M3_OLD_M4, M3_NEW_M4), (M3_OLD_M2, M3_NEW_M2)],
        "M22": [(M22_OLD_UPDATE, M22_NEW_UPDATE), (M22_OLD_VERDICT, M22_NEW_VERDICT)]}


def patch_one(mid, path, dry):
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print(f"[{mid}] already patched — skipped")
        return True
    applied, helper_in = [], False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        o = s
        for old, new in SUBS[mid]:
            if old in s:
                if not helper_in:
                    s = HELPER + "\n" + s
                    helper_in = True
                s = s.replace(old, new, 1)
                applied.append(old[:28])
        if s != o:
            cell["source"] = s.splitlines(keepends=True)
    if len(applied) != len(SUBS[mid]):
        print(f"[{mid}] FAILED — matched {len(applied)}/{len(SUBS[mid])}")
        return False
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), mid, "exec")
    if not dry:
        json.dump(nb, open(path, "w"), indent=1)
    print(f"[{mid}] {'would patch' if dry else 'patched'} — {len(applied)} guard(s) installed")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ok = all(patch_one(m, os.path.join(REPO, p), a.dry_run) for m, p in PATHS.items())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
