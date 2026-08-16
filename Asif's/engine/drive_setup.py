#!/usr/bin/env python3
"""
Robust output-directory setup for Colab / Kaggle / local runs.

WHY THIS EXISTS
---------------
The old block in M2/M3/M22 did this:

    try:
        drive.mount("/content/drive", force_remount=False)
        DRIVE_MOUNTED = True
    except Exception as e:
        print(f"Drive mount skipped ({e}). Falling back to ephemeral /content storage.")

Two failure modes, both silent:

1. **Re-mounting an already-mounted Drive.** Colab raises
   `ValueError: Mountpoint must not already contain files` when /content/drive is non-empty
   from a previous mount. The except branch swallows it, `DRIVE_MOUNTED` stays False, and
   BASE_DIR quietly becomes ephemeral `/content/OWMTL/<id>`. The run then completes and
   "saves" — to storage that disappears on disconnect.
2. **A mounted-but-unwritable Drive** (quota, stale FUSE handle). `os.makedirs(exist_ok=True)`
   can succeed while later writes fail, so the failure surfaces hours into training.

This module refuses to guess. It mounts robustly, **probes that the directory is genuinely
writable**, and raises loudly if not — rather than silently degrading to ephemeral storage.
That is the same class of bug as the silent split fallback that gave four models an
11-patient test set while labelling itself "official 60/40".

COLLISION POLICY
----------------
An existing `OWMTL/<model_id>` folder is normal — previous results live there and must not be
clobbered. `mode` decides:

  "auto" (default)  Inspect the folder. If it holds a COMPLETED run (a results_*.json), start
                    a new version (`M2_v2`, `M2_v3`, ...) so old results survive. If it holds
                    only checkpoints, it is an INTERRUPTED run — reuse it so auto-resume works.
  "new_version"     Always a fresh versioned folder.
  "reuse"           Use the path as-is (resume, or add to it).
  "overwrite"       Delete the existing folder's contents and reuse the path. Destructive.

"auto" is the default because it does the right thing in both the cases that actually happen:
re-running a finished experiment, and resuming after a Colab disconnect.
"""
import os
import shutil

__all__ = ["setup_output_dir", "mount_drive_if_available"]


def mount_drive_if_available(verbose=True):
    """Return (mounted: bool, detail: str). Never raises."""
    in_colab = "google.colab" in __import__("sys").modules or os.path.exists("/content")
    if not in_colab:
        return False, "not a Colab runtime"

    # Already mounted? Do NOT call mount() again -- that is what raises
    # "Mountpoint must not already contain files".
    if os.path.isdir("/content/drive/MyDrive"):
        if verbose:
            print("Google Drive: already mounted at /content/drive")
        return True, "already mounted"

    try:
        from google.colab import drive
    except Exception as e:
        return False, f"google.colab unavailable ({e})"

    try:
        drive.mount("/content/drive")
        if verbose:
            print("Google Drive: mounted at /content/drive")
        return True, "mounted"
    except Exception as e1:
        # Most common cause: a stale, non-empty mountpoint. force_remount clears it.
        try:
            drive.mount("/content/drive", force_remount=True)
            if verbose:
                print(f"Google Drive: force-remounted (first attempt failed: {e1})")
            return True, "force-remounted"
        except Exception as e2:
            return False, f"mount failed: {e1} | force_remount failed: {e2}"


def _probe_writable(path):
    """Actually write, read back and delete a file. Returns (ok, error)."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".owmtl_write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        with open(probe) as f:
            if f.read() != "ok":
                return False, "read-back mismatch"
        os.remove(probe)
        return True, None
    except Exception as e:
        return False, str(e)


def _classify(path):
    """'absent' | 'empty' | 'completed' | 'in_progress'."""
    if not os.path.isdir(path):
        return "absent"
    entries = [e for e in os.listdir(path) if not e.startswith(".")]
    if not entries:
        return "empty"
    for dirpath, _d, files in os.walk(path):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                return "completed"
    for dirpath, _d, files in os.walk(path):
        for fn in files:
            if fn.endswith(".pth"):
                return "in_progress"
    return "in_progress"


def _next_version(root, name):
    """name -> name_v2 -> name_v3 ... first path that is absent or empty."""
    v = 2
    while True:
        cand = os.path.join(root, f"{name}_v{v}")
        if _classify(cand) in ("absent", "empty"):
            return cand
        v += 1
        if v > 99:
            raise RuntimeError(f"Refusing to version past {name}_v99 -- clean up {root}.")


def setup_output_dir(model_id, base_name="OWMTL", mode="auto", verbose=True):
    """Resolve a GUARANTEED-WRITABLE output directory.

    Returns (base_dir, info). Raises RuntimeError if nothing writable can be found -- never
    silently downgrades to ephemeral storage.
    """
    if mode not in ("auto", "new_version", "reuse", "overwrite"):
        raise ValueError(f"unknown mode {mode!r}")

    import sys
    in_colab = "google.colab" in sys.modules or os.path.exists("/content")
    mounted, detail = mount_drive_if_available(verbose=verbose) if in_colab else (False, "n/a")

    # Candidate roots, best first. Each is probed before use.
    if in_colab and mounted:
        roots = [(f"/content/drive/MyDrive/{base_name}", "Google Drive (persistent)"),
                 (f"/content/{base_name}", "Colab local (EPHEMERAL)")]
    elif in_colab:
        roots = [(f"/content/{base_name}", "Colab local (EPHEMERAL)")]
    elif os.path.exists("/kaggle/working"):
        roots = [("/kaggle/working", "Kaggle working")]
    else:
        roots = [("./outputs", "local")]

    root = kind = None
    problems = []
    for cand, label in roots:
        ok, err = _probe_writable(cand)
        if ok:
            root, kind = cand, label
            break
        problems.append(f"{cand}: {err}")

    if root is None:
        raise RuntimeError(
            "No writable output directory found. Tried:\n  " + "\n  ".join(problems) +
            "\n\nRefusing to continue: an earlier version silently fell back to ephemeral "
            "storage here, so runs 'succeeded' and then vanished on disconnect.")

    target = root if kind == "Kaggle working" else os.path.join(root, model_id)
    state = _classify(target)

    if mode == "auto":
        if state == "completed":
            base_dir = _next_version(root, model_id)
            action = f"existing run is COMPLETE -> new version {os.path.basename(base_dir)}"
        elif state == "in_progress":
            base_dir, action = target, "existing run is INCOMPLETE -> reusing it (auto-resume)"
        else:
            base_dir, action = target, f"{state} -> using it"
    elif mode == "new_version":
        if state in ("absent", "empty"):
            base_dir, action = target, f"{state} -> using it"
        else:
            base_dir = _next_version(root, model_id)
            action = f"forced new version -> {os.path.basename(base_dir)}"
    elif mode == "overwrite":
        if state not in ("absent", "empty"):
            shutil.rmtree(target, ignore_errors=True)
            action = "OVERWRITE -> previous contents deleted"
        else:
            action = f"{state} -> using it"
        base_dir = target
    else:  # reuse
        base_dir, action = target, f"{state} -> reusing as requested"

    ok, err = _probe_writable(base_dir)
    if not ok:
        raise RuntimeError(f"Chosen directory {base_dir} is not writable: {err}")

    ckpt_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    for d in (ckpt_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    cache_dir = ("/content/owmtl_spec_cache" if in_colab else
                 "/kaggle/working/owmtl_spec_cache" if os.path.exists("/kaggle/working")
                 else "./owmtl_spec_cache")
    os.makedirs(cache_dir, exist_ok=True)

    info = {"base_dir": base_dir, "ckpt_dir": ckpt_dir, "results_dir": results_dir,
            "cache_dir": cache_dir, "storage": kind, "drive_mounted": mounted,
            "drive_detail": detail, "existing_state": state, "action": action, "mode": mode,
            "is_persistent": "EPHEMERAL" not in kind}

    if verbose:
        print("=" * 70)
        print(f"OUTPUT LOCATION  (mode={mode})")
        print("=" * 70)
        print(f"  storage    : {kind}")
        print(f"  existing   : {state}")
        print(f"  decision   : {action}")
        print(f"  base       : {base_dir}")
        print(f"  checkpoints: {ckpt_dir}")
        print(f"  results    : {results_dir}")
        print(f"  spec cache : {cache_dir}   (local disk, disposable)")
        print(f"  write probe: PASSED")
        if not info["is_persistent"]:
            print("\n  *** WARNING: this is EPHEMERAL storage. Everything is lost when the")
            print("      runtime disconnects. Mount Drive before a long run. ***")
        print("=" * 70)
    return base_dir, info


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    print("### 1. fresh run -> uses M2")
    b, _ = setup_output_dir("M2")
    open(os.path.join(b, "checkpoints", "best_model.pth"), "w").write("x")

    print("\n### 2. interrupted run (checkpoint, no results) -> should REUSE M2")
    b2, i2 = setup_output_dir("M2")
    assert b2 == b, f"expected reuse of {b}, got {b2}"
    assert i2["existing_state"] == "in_progress"
    open(os.path.join(b2, "results", "results_M2.json"), "w").write("{}")

    print("\n### 3. completed run -> should make M2_v2")
    b3, i3 = setup_output_dir("M2")
    assert b3.endswith("M2_v2"), b3
    assert i3["existing_state"] == "completed"
    open(os.path.join(b3, "results", "results_M2.json"), "w").write("{}")

    print("\n### 4. two completed runs -> should make M2_v3")
    b4, _ = setup_output_dir("M2")
    assert b4.endswith("M2_v3"), b4

    print("\n### 5. explicit reuse -> back to M2")
    b5, _ = setup_output_dir("M2", mode="reuse")
    assert b5 == b, b5

    print("\n### 6. overwrite -> M2, emptied")
    b6, _ = setup_output_dir("M2", mode="overwrite")
    assert b6 == b and not os.path.exists(os.path.join(b6, "results", "results_M2.json"))

    print("\n" + "=" * 70)
    print("ALL COLLISION-POLICY CHECKS PASSED")
    print("=" * 70)
