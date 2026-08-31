#!/usr/bin/env python3
"""
Make load_official_split() use the split file DATA SETUP already resolved.

    python3 patch_split_lookup.py [--dry-run]

THE PROBLEM (two of them, one fatal)
------------------------------------
1. M22 still carried the ORIGINAL brittle lookup: three hardcoded os.path.exists() checks for
   "ICBHI_Challenge_train_test.txt" -- capital C. The DATA SETUP cell writes the file as
   "ICBHI_challenge_train_test.txt" (lowercase c, the official spelling). Linux is
   case-sensitive, so the lookup could never match, and M22 died with
   "Official split file not found" on Colab 2026-08-29 after the dataset had downloaded fine.
   M2 and M3 had been fixed for this months earlier; M22's port brought over the strict error
   and verify_official_split() but not the lookup itself. That is an incomplete port, and the
   strict no-fallback error is what turned it into a hard stop instead of a silent wrong split
   -- which is the behaviour we want, but it should never have been reachable.

2. ALL THREE notebooks ignored `SPLIT_FILE` -- the global the DATA SETUP cell sets after it has
   already located or materialised the official file from its embedded gzip+base64 copy AND
   asserted 920/539/381. So the notebook threw away a verified answer and re-derived it with a
   weaker search. Fixing only #1 would leave that redundancy in place.

THE FIX
-------
One canonical implementation for all three:
  1. use SPLIT_FILE when set and readable -- the verified path, so the normal path is a direct
     hit with no filesystem walk at all;
  2. otherwise fall back to a case-insensitive walk of the usual roots, for sessions where the
     DATA SETUP cell was skipped or edited;
  3. raise if neither works, listing what was searched. Still NO fallback split, by design.

Replacement is span-based rather than exact-text, so it applies to both the brittle M22 variant
and the longer M2/M3 one without two hardcoded patterns. The span ends at the NEXT top-level
`def` after load_official_split -- found by scanning, not by assuming which function comes next.

NOTE (2026-08-29, post-hoc): the first version of this script ended the span at a hardcoded
`def patient_id_from_stem`. In M22 that happens to be the next function, but in M2 and M3
`def verify_official_split` sits between the two -- so the replacement SILENTLY DELETED
verify_official_split from both notebooks. It was caught by a static undefined-name pass over
all three notebooks, not by this script's own compile check (the cell still compiled fine -- a
missing function is only a NameError at call time) and not by the accompanying test, which
extracted and exercised load_official_split alone and so never noticed the surrounding cell had
been damaged. Two lessons encoded below: find the span boundary by scanning for the next
top-level def, and assert that every other top-level def in the cell survives the edit.

Idempotent.
"""
import argparse
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_SPLIT_LOOKUP_V2"

PATHS = {
    "M2":  "Asif's/M2/M2_cnn_baseline_tuned.ipynb",
    "M3":  "Asif's/M3/M3_lightweight_backbone.ipynb",
    "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb",
}

NEW_FUNC = '''def load_official_split(data_root):
    """Return {filename_stem: 'train'|'test'} from the official split file.  [''' + MARKER + ''']

    Preference order:
      1. SPLIT_FILE -- the path the DATA SETUP cell resolved AND verified (920 recordings,
         539 train / 381 test). That cell carries an embedded gzip+base64 copy of the official
         file and writes it out when the Kaggle mirror does not ship one, so in a normal run
         this is a direct hit and no filesystem walk happens.
      2. A case-insensitive walk of the usual roots, for sessions where DATA SETUP was skipped.

    Raises if neither works. There is deliberately NO fallback split: the old `pid <= 111`
    fallback is what gave M2/M3/M22 an 11-patient test set (7.1% of cycles) while reporting
    split_method "patient_independent_official_60_40".

    History: M22 shipped a lookup that tested three exact paths for the CAPITAL-C spelling
    "ICBHI_Challenge_train_test.txt" while the real file is lowercase -- on a case-sensitive
    filesystem it never matched, and M22 could not start at all.
    """
    def _read_split(path):
        m = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1].lower() in ("train", "test"):
                        m[parts[0].replace(".wav", "")] = parts[1].lower()
        except Exception:
            return None
        return m or None

    # ---- 1. the path DATA SETUP already verified -----------------------------------------
    _sf = globals().get("SPLIT_FILE")
    if _sf and os.path.isfile(_sf):
        _m = _read_split(_sf)
        if _m:
            print(f"Official split file: {_sf}  ({len(_m)} recordings)  [via DATA SETUP]")
            return _m
        print(f"[warn] SPLIT_FILE={_sf} is set but unreadable/empty — falling back to search.")

    # ---- 2. case-insensitive search ------------------------------------------------------
    wanted = "icbhi_challenge_train_test.txt"
    searched, bases = [], [
        data_root,
        os.path.dirname(data_root.rstrip("/")),
        os.path.dirname(os.path.dirname(data_root.rstrip("/"))),
        "/content", "/kaggle/input", ".",
    ]
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        searched.append(base)
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                # the official file is lowercase; mirrors vary, so compare case-insensitively
                if fn.lower() != wanted and not fn.lower().endswith("train_test.txt"):
                    continue
                path = os.path.join(dirpath, fn)
                split_map = _read_split(path)
                if split_map:
                    print(f"Official split file: {path}  ({len(split_map)} recordings)")
                    return split_map

    raise FileNotFoundError(
        "ICBHI_challenge_train_test.txt NOT FOUND. It is REQUIRED -- this notebook has no "
        "fallback split by design, because the fallback is what invalidated the earlier "
        "M2/M3/M22 runs (11 test patients reported as 'official 60/40').\\n"
        "Re-run the DATA SETUP cell: it carries an embedded copy and writes it out.\\n"
        "Searched under: " + ", ".join(searched))


'''


def toplevel_defs(src):
    """Names of every top-level `def` in a cell, via AST (comments/strings can't fool it)."""
    try:
        return {n.name for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    except SyntaxError:
        return set()


def patch_one(mid, path, dry):
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print(f"[{mid}] already patched — skipped")
        return True
    hit = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        i = s.find("def load_official_split")
        if i == -1:
            continue
        # End the span at the NEXT top-level def, whatever it is -- never assume its name.
        j = s.find("\ndef ", i + 1)
        if j == -1:
            print(f"[{mid}] FAILED — no following top-level def to bound the span")
            return False
        j += 1
        before = toplevel_defs(s)
        new_s = s[:i] + NEW_FUNC + s[j:]
        after = toplevel_defs(new_s)
        lost = before - after
        if lost:
            print(f"[{mid}] FAILED — replacement would delete {sorted(lost)}")
            return False
        old_len = j - i
        cell["source"] = new_s.splitlines(keepends=True)
        hit += 1
        print(f"[{mid}] replaced load_official_split ({old_len} -> {len(NEW_FUNC)} chars); "
              f"other defs preserved: {sorted(before - {'load_official_split'})}")
    if hit != 1:
        print(f"[{mid}] FAILED — expected 1 definition, found {hit}")
        return False
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), mid, "exec")
    if not dry:
        json.dump(nb, open(path, "w"), indent=1)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(PATHS))
    a = ap.parse_args()
    ok = all(patch_one(m, os.path.join(REPO, PATHS[m]), a.dry_run) for m in a.only)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
