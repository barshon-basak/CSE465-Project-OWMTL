#!/usr/bin/env python3
"""backfill_efficiency.py - fill the efficiency block from checkpoints, or say why it cannot.

RTK_requirements.md T1: `total_params`, `model_size_mb` and `training_time_total_s` are
missing from a chunk of the committed results JSONs, which makes those models uncountable
for course requirements 4 and 5.

WHAT IT RECOVERS AND WHAT IT REFUSES TO
    total_params     : counted from the checkpoint's state_dict. Exact.
    model_size_mb    : params x 4 bytes / 1e6, matching Model_Training_Protocol.md's
                       get_model_size_mb. Reported from the checkpoint's own tensors, not
                       from the file size on disk - a .pth also carries optimiser state and
                       would overstate the model.
    training_time_total_s : NOT RECOVERABLE after the fact. Written as null with a reason,
                       never guessed. RTK T1 item 3 says mark it, do not invent it.

Analysis artefacts (a decision record, an XAI pack, a merge) have no model of their own.
They get an explicit not_applicable note instead of a fabricated parameter count, so the
audit stops flagging them and a reader knows the difference between "missing" and "N/A".

    python "Asif's/audit/backfill_efficiency.py"           # dry run, prints the plan
    python "Asif's/audit/backfill_efficiency.py" --write   # additive edit, nothing removed
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REQUIRED = ("total_params", "model_size_mb", "training_time_total_s")

# Models whose results file describes an analysis, not a trained network.
NOT_A_MODEL = {
    "Asif's/M12_v2": "backbone re-decision - inference only over frozen checkpoints",
    "Asif's/M44": "XAI analysis of M22_v2 - explains a model, does not train one",
    "Barshon's/M28": "master merge of other models' results",
    "Barshon's/M11": "post-hoc calibrators fitted on frozen logits",
    "Barshon's/M14": "conformal wrapper over M15's scores - no network of its own",
    "Barshon's/M20": "temperature scaling on frozen logits",
    "Barshon's/M18": "compression sweep - per-config sizes are in the sweep CSV",
    "Barshon's/M19": "cross-dataset evaluation of an existing checkpoint",
    "Asif's/M29": "post-hoc OOD scoring over frozen M12 embeddings",
}

TRAIN_TIME_REASON = ("not recorded by the original run and not recoverable from a "
                     "checkpoint; re-run the model to obtain it")


def find_checkpoint(results_path: str) -> str | None:
    """The checkpoint that belongs to THIS result file, or None.

    Deliberately conservative: an ablation directory holds one checkpoint per row, so a
    bare same-directory glob would hand every row the first file it found and silently
    report one row's parameter count as another's. Match the row id first.
    """
    d = os.path.dirname(results_path)
    base = os.path.basename(results_path)[len("results_"):-len(".json")]  # e.g. M45_A3
    for pat in (f"best_{base}.pth", f"{base}_best_model.pth", f"best_{base}.pt"):
        hit = os.path.join(d, pat)
        if os.path.exists(hit):
            return hit
    generic = [p for p in sorted(glob.glob(os.path.join(d, "*.pth")))
               + sorted(glob.glob(os.path.join(d, "..", "*.pth")))]
    # Only accept a generic name when the directory holds exactly one candidate; more than
    # one means the mapping is ambiguous and guessing would put a wrong number in a table.
    uniq = {os.path.realpath(p) for p in generic}
    return generic[0] if len(uniq) == 1 else None


def count_params(ckpt_path: str):
    import torch
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        return None
    if not any(hasattr(v, "numel") for v in obj.values()):
        # One more level: some runs wrap the state dict under a run-specific key.
        nested = [v for v in obj.values()
                  if isinstance(v, dict) and any(hasattr(x, "numel") for x in v.values())]
        if len(nested) == 1:
            obj = nested[0]
    total = 0
    for v in obj.values():
        if hasattr(v, "numel"):
            total += int(v.numel())
    return total or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), "..", ".."))
    args = ap.parse_args()
    os.chdir(os.path.abspath(args.repo))

    filled, skipped, na = [], [], []
    for f in sorted(glob.glob("**/results_M*.json", recursive=True)):
        if "Archive_" in f:
            continue
        try:
            doc = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        eff = doc.get("efficiency")
        if not isinstance(eff, dict):
            eff = {}
        missing = [k for k in REQUIRED if eff.get(k) in (None, "")]
        if not missing:
            continue

        norm = f.replace("\\", "/")
        why_na = next((v for k, v in NOT_A_MODEL.items() if norm.startswith(k + "/")), None)
        note = {}

        if why_na and {"total_params", "model_size_mb"} & set(missing):
            note["efficiency_not_applicable"] = why_na
            eff.setdefault("total_params", None)
            eff.setdefault("model_size_mb", None)
            na.append((f, why_na))
        elif {"total_params", "model_size_mb"} & set(missing):
            ck = find_checkpoint(f)
            n = count_params(ck) if ck else None
            if n:
                eff["total_params"] = n
                eff["model_size_mb"] = round(n * 4 / 1e6, 3)
                note["efficiency_source"] = (
                    f"total_params counted from {os.path.basename(ck)}; model_size_mb = "
                    "params x 4 bytes (fp32), per Model_Training_Protocol.md section 3")
                filled.append((f, n, eff["model_size_mb"]))
            else:
                skipped.append((f, "no unambiguous checkpoint" if not ck
                                else "checkpoint holds no tensors"))
                continue

        if "training_time_total_s" in missing:
            eff["training_time_total_s"] = None
            note["training_time_not_recoverable"] = TRAIN_TIME_REASON

        doc["efficiency"] = eff
        doc.setdefault("audit_notes", {}).update(note)
        if args.write:
            json.dump(doc, open(f, "w", encoding="utf-8"), indent=2, default=str)

    print(f"{'RECOVERED from checkpoint':-<80}")
    for f, n, mb in filled:
        print(f"  {f[:58]:<58} {n:>12,} params  {mb:>8.3f} MB")
    print(f"\n{'NOT APPLICABLE - analysis artefact, no model of its own':-<80}")
    for f, why in na:
        print(f"  {f[:58]:<58} {why}")
    if skipped:
        print(f"\n{'LEFT ALONE - would have had to guess':-<80}")
        for f, why in skipped:
            print(f"  {f[:58]:<58} {why}")
    print(f"\n{len(filled)} recovered, {len(na)} marked N/A, {len(skipped)} left alone. "
          f"training_time_total_s is never invented.")
    if not args.write:
        print("\nDRY RUN - nothing written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
