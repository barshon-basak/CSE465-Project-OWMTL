#!/usr/bin/env python3
"""
Remove every `PASTE_YOUR_KAGGLE_API_KEY_HERE` from the project.

    python3 patch_kaggle_auth.py [--dry-run]

Replaces the hardcoded-placeholder block in each notebook and generator with credential
resolution that tries, in order:

    1. Colab Secrets  (sidebar key icon -> KAGGLE_USERNAME / KAGGLE_KEY)  <- set once, done forever
    2. environment variables already set in the runtime
    3. ~/.kaggle/kaggle.json

and only errors — with instructions — if all three are empty. The key is never stored in a
git-tracked file.

Idempotent: files already carrying the marker are skipped.
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_KAGGLE_AUTH_V1"

AUTH_BLOCK = '''# ---- Kaggle credentials: resolved automatically, NEVER pasted here ----------------
# [''' + MARKER + ''']
# ONE-TIME SETUP (Colab): left sidebar -> key icon (Secrets) -> add two secrets
#     KAGGLE_USERNAME   your kaggle username
#     KAGGLE_KEY        from kaggle.com -> Settings -> API -> Create New Token
# toggle "Notebook access" ON for both. Every notebook in this project then picks them up
# silently -- you never type the key again.
#
# The key is deliberately NOT hardcoded: these files are git-tracked, and a Kaggle key grants
# full account access to anyone who reads the repo.
import os as _os


def _owmtl_load_kaggle_creds():
    """Returns the source it found credentials in, or None."""
    try:                                    # 1. Colab Secrets -- the one to use
        from google.colab import userdata
        _u, _k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if _u and _k:
            _os.environ["KAGGLE_USERNAME"], _os.environ["KAGGLE_KEY"] = _u, _k
            return "Colab Secrets"
    except Exception:
        pass
    if _os.environ.get("KAGGLE_USERNAME") and _os.environ.get("KAGGLE_KEY"):
        return "environment variables"      # 2. already exported
    import json as _json                    # 3. ~/.kaggle/kaggle.json
    _p = _os.path.expanduser("~/.kaggle/kaggle.json")
    if _os.path.isfile(_p):
        try:
            _d = _json.load(open(_p))
            if _d.get("username") and _d.get("key"):
                _os.environ["KAGGLE_USERNAME"] = _d["username"]
                _os.environ["KAGGLE_KEY"] = _d["key"]
                return "~/.kaggle/kaggle.json"
        except Exception:
            pass
    return None


KAGGLE_CRED_SOURCE = _owmtl_load_kaggle_creds()
print(f"Kaggle credentials: {KAGGLE_CRED_SOURCE or 'NOT FOUND (only needed if data is missing)'}")
'''

NEEDS_KEY_ERROR = '''        if not KAGGLE_CRED_SOURCE:
            raise RuntimeError(
                "The dataset is not present and no Kaggle credentials were found.\\n"
                "Set them ONCE in Colab: left sidebar -> key icon (Secrets) -> add\\n"
                "  KAGGLE_USERNAME  and  KAGGLE_KEY   (Settings -> API -> Create New Token)\\n"
                "and enable 'Notebook access' for both. Then re-run this cell.")
'''

# the exact placeholder assignment shipped in every notebook
OLD_RE = re.compile(
    r"os\.environ\['KAGGLE_USERNAME'\][^\n]*\n"
    r"os\.environ\['KAGGLE_KEY'\]\s*=\s*'PASTE_YOUR_KAGGLE_API_KEY_HERE'[^\n]*\n"
    r"(?:#[^\n]*\n|\s*\n)*"          # the commented-out Secrets suggestion that followed
)

GUARD_RE = re.compile(
    r"if os\.environ\.get\('KAGGLE_KEY',\s*''\)\s*(?:in\s*\([^)]*\)|==\s*'PASTE_YOUR_KAGGLE_API_KEY_HERE')\s*:\s*\n"
    r"(?:[ \t]+[^\n]*\n)+"
)


def patch_text(src):
    if MARKER in src:
        return src, False
    if "PASTE_YOUR_KAGGLE_API_KEY_HERE" not in src:
        return src, False
    new = OLD_RE.sub(AUTH_BLOCK + "\n", src, count=1)
    if new == src:                                   # generator files quote it differently
        new = src.replace(
            "os.environ['KAGGLE_USERNAME'] = 'AsifM7'\n"
            "os.environ['KAGGLE_KEY'] = 'PASTE_YOUR_KAGGLE_API_KEY_HERE'   # <-- replace before running",
            AUTH_BLOCK.rstrip(), 1)
    new = GUARD_RE.sub(NEEDS_KEY_ERROR, new)
    # any stragglers: make them fail loudly rather than look like a valid key
    new = new.replace("'PASTE_YOUR_KAGGLE_API_KEY_HERE'", "None")
    new = new.replace("PASTE_YOUR_KAGGLE_API_KEY_HERE", "")
    return new, new != src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = sorted(set(
        glob.glob(os.path.join(REPO, "**", "*.ipynb"), recursive=True) +
        glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True)))
    n_changed = 0

    for path in targets:
        if os.sep + ".git" + os.sep in path or "Archive_Files" in path:
            continue
        raw = open(path, encoding="utf-8").read()
        if "PASTE_YOUR_KAGGLE_API_KEY_HERE" not in raw:
            continue
        rel = os.path.relpath(path, REPO)

        if path.endswith(".ipynb"):
            nb = json.loads(raw)
            hit = False
            for c in nb["cells"]:
                if c["cell_type"] != "code":
                    continue
                s = "".join(c["source"])
                new, ch = patch_text(s)
                if ch:
                    compile(new, rel, "exec")
                    c["source"] = new.splitlines(keepends=True)
                    hit = True
            if hit and not a.dry_run:
                json.dump(nb, open(path, "w"), indent=1)
            if hit:
                print(f"{'would patch' if a.dry_run else 'patched'}: {rel}")
                n_changed += 1
        else:
            new, ch = patch_text(raw)
            if ch:
                compile(new, rel, "exec")
                if not a.dry_run:
                    open(path, "w", encoding="utf-8").write(new)
                print(f"{'would patch' if a.dry_run else 'patched'}: {rel}")
                n_changed += 1

    print(f"\n{n_changed} file(s) {'would be ' if a.dry_run else ''}changed.")
    if not a.dry_run:
        left = [p for p in targets
                if os.sep + ".git" + os.sep not in p and "Archive_Files" not in p
                and "PASTE_YOUR_KAGGLE_API_KEY_HERE" in open(p, encoding="utf-8").read()]
        print("Remaining placeholders:", len(left))
        for p in left:
            print("  ", os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
