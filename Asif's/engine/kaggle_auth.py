#!/usr/bin/env python3
"""
Kaggle credential resolution — set up once, never typed again.

WHY THIS EXISTS
---------------
Every notebook shipped a literal `PASTE_YOUR_KAGGLE_API_KEY_HERE`, so the key had to be re-pasted
on every run, in every notebook. That is pure friction, and it also pushed people toward pasting a
real key into a git-tracked file — where it gets committed, pushed to GitHub, and grants full
account access to anyone who reads the repo.

This resolves credentials from the places they should already live, in order:

  1. ~/.kaggle/kaggle.json      the Kaggle CLI standard. Set once, works forever, never in the repo.
  2. KAGGLE_USERNAME / KAGGLE_KEY environment variables.
  3. Colab Secrets (key icon in the left sidebar) — the Colab equivalent of (1).
  4. An interactive prompt, and it OFFERS to save to (1) so it is the last time you are asked.

Nothing is ever written into the notebook or the repo.

ONE-TIME SETUP (local)
----------------------
    python3 kaggle_auth.py --setup

or by hand: kaggle.com -> Settings -> API -> Create New Token, then

    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

ONE-TIME SETUP (Colab)
----------------------
Left sidebar -> key icon -> add `KAGGLE_USERNAME` and `KAGGLE_KEY`, enable for the notebook.
"""
import json
import os
import stat

KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")


def _from_file():
    if not os.path.isfile(KAGGLE_JSON):
        return None
    try:
        d = json.load(open(KAGGLE_JSON))
        if d.get("username") and d.get("key"):
            return d["username"], d["key"], f"~/.kaggle/kaggle.json"
    except Exception:
        pass
    return None


def _from_env():
    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    placeholder = ("", None, "PASTE_YOUR_KAGGLE_API_KEY_HERE")
    if u and k and k not in placeholder:
        return u, k, "environment variables"
    return None


def _from_colab_secrets():
    try:
        from google.colab import userdata
        u, k = userdata.get("KAGGLE_USERNAME"), userdata.get("KAGGLE_KEY")
        if u and k:
            return u, k, "Colab Secrets"
    except Exception:
        pass
    return None


def save_credentials(username, key):
    """Write ~/.kaggle/kaggle.json with 0600 so it is never asked for again."""
    os.makedirs(os.path.dirname(KAGGLE_JSON), exist_ok=True)
    with open(KAGGLE_JSON, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(KAGGLE_JSON, stat.S_IRUSR | stat.S_IWUSR)   # 0600, required by the kaggle CLI
    return KAGGLE_JSON


def ensure_kaggle_credentials(interactive=True, verbose=True):
    """Resolve credentials and export them to the environment for the kaggle CLI.

    Returns the source string. Raises RuntimeError if nothing is available and interactive=False.
    """
    for probe in (_from_file, _from_env, _from_colab_secrets):
        got = probe()
        if got:
            u, k, src = got
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
            if verbose:
                print(f"Kaggle credentials: loaded from {src} (user: {u})")
            return src

    msg = ("No Kaggle credentials found. Set them up ONCE and this never asks again:\n"
           "  local : kaggle.com -> Settings -> API -> Create New Token, then\n"
           "          mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
           "          chmod 600 ~/.kaggle/kaggle.json\n"
           "  Colab : left sidebar -> key icon -> add KAGGLE_USERNAME and KAGGLE_KEY\n"
           "\nDo NOT paste the key into a notebook -- these files are git-tracked and would "
           "publish it.")
    if not interactive:
        raise RuntimeError(msg)

    print(msg + "\n")
    try:
        import getpass
        u = input("Kaggle username: ").strip()
        k = getpass.getpass("Kaggle API key (hidden): ").strip()
    except Exception as e:
        raise RuntimeError(f"{msg}\n\n(prompt unavailable: {e})")
    if not u or not k:
        raise RuntimeError(msg)

    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
    try:
        if input(f"Save to {KAGGLE_JSON} so you are never asked again? [Y/n] ").strip().lower() \
                in ("", "y", "yes"):
            print(f"Saved -> {save_credentials(u, k)} (mode 0600)")
    except Exception:
        pass
    return "interactive prompt"


# ------------------------------------------------------------------ dataset download
DATASETS = {
    "icbhi": ("vbookshelf/respiratory-sound-database", "audio_and_txt_files"),
    "sprsound": ("mayarelghandour/sprsound-nosplit", "sprsound"),
    "coswara": ("sarabhian/coswara-dataset-heavy-cough", "coswara_data"),
}


def find_dataset(marker, roots=None):
    """Return a directory containing `marker` with at least one .wav beneath it, else None."""
    import glob
    for root in (roots or ["/content", "/kaggle/input", "./data", ".",
                           os.path.expanduser("~/owmtl_data")]):
        if not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "**", marker), recursive=True)):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
                return d
    return None


def ensure_dataset(name="icbhi", dest=None, verbose=True):
    """Return a path to the dataset, downloading only if it is not already present."""
    import subprocess
    import sys
    slug, marker = DATASETS[name]

    found = find_dataset(marker)
    if found:
        if verbose:
            print(f"{name}: already present at {found}")
        return found

    dest = dest or ("/content" if os.path.isdir("/content")
                    else os.path.expanduser("~/owmtl_data"))
    os.makedirs(dest, exist_ok=True)
    ensure_kaggle_credentials(verbose=verbose)

    try:
        import kaggle  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

    if verbose:
        print(f"{name}: downloading {slug} -> {dest} (this is a one-time cost)")
    subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"])

    found = find_dataset(marker, roots=[dest])
    if not found:
        raise RuntimeError(f"{name}: download finished but {marker!r} not found under {dest}")
    if verbose:
        print(f"{name}: ready at {found}")
    return found


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Kaggle credentials + dataset setup.")
    ap.add_argument("--setup", action="store_true", help="store credentials once")
    ap.add_argument("--download", nargs="*", metavar="NAME",
                    help=f"download dataset(s): {', '.join(DATASETS)}")
    a = ap.parse_args()

    if a.setup or not (a.setup or a.download):
        src = ensure_kaggle_credentials()
        print(f"\nCredentials OK (source: {src}).")
        if src != "~/.kaggle/kaggle.json" and os.path.isfile(KAGGLE_JSON):
            print("Stored at ~/.kaggle/kaggle.json -- no notebook will ask again.")
    for name in (a.download or []):
        ensure_dataset(name)
