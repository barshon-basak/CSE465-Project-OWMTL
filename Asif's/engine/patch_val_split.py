#!/usr/bin/env python3
"""
Stop the final training run from selecting checkpoints against the TEST set.

    python3 patch_val_split.py [--dry-run]

THE PROBLEM
-----------
In all three notebooks, the final 60-epoch training run does:

    final_run = train_model(..., train_loader=train_loader, val_loader=test_loader, ...)

`val_loader=test_loader`. `is_best = score > best_score` runs every epoch against that loader.
So `best_model.pth` -- the checkpoint every reported number comes from -- is chosen by which
epoch scored highest ON THE OFFICIAL TEST SET, checked 60 times. `Model_Training_Protocol.md`
writes "test/validation set" as one term throughout (§5, §6), so this was the intended design,
not an oversight -- but it means every "single held-out test evaluation" in this project has a
60-shot test-set-peeking advantage baked into checkpoint selection, on top of whatever the
final-evaluation cell reports.

Found while reviewing M2's first corrected-split run (2026-08-29): official score 0.4671, BELOW
the 0.50 a trivial always-predict-one-class model gets, DESPITE the 60-shot selection advantage.
That direction is what makes this worth fixing before M3/M22 rather than after: a properly
separated protocol can only score the same or lower, and the corrected-baseline table should not
carry an uncontrolled bias the paper's own §3 argues against elsewhere.

THE FIX
-------
Carve a patient-grouped validation slice OUT OF TRAIN ONLY (GroupShuffleSplit, 15%, seeded by
CFG["seed"]), asserted disjoint from both the fit set and the test set. Checkpoint selection reads
that slice; the test set is read exactly once, in the existing final-evaluation cell. The fit set
shrinks accordingly (~85% of the old training data) and class weights are recomputed on it, since
that is what is actually being trained on now.

M22-specific: `make_dataset(split, indices, augment=...)` defaults augment=True whenever
split=="train" -- which the validation slice also is, being carved FROM train. Left alone, the
held-out validation set would be SpecAugment-perturbed every epoch, which is its own quiet bug.
The val call passes augment=False explicitly; M22's own docstring already flags this exact case.

Idempotent.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_VAL_SPLIT_V1"

PATHS = {
    "M2":  "Asif's/M2/M2_cnn_baseline_tuned.ipynb",
    "M3":  "Asif's/M3/M3_lightweight_backbone.ipynb",
    "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb",
}

IMPORT_OLD = "from sklearn.model_selection import GroupKFold"
IMPORT_NEW = "from sklearn.model_selection import GroupKFold, GroupShuffleSplit"

RUN_OLD = '''final_run = train_model(
    config=best_config,
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=CFG["num_epochs"],
    class_weights_t=class_weights_tensor,
    ckpt_dir=CFG["ckpt_dir"],
    resume=True,
    verbose=True,
)'''


def val_block(mid):
    fit_call = 'make_dataset("train", _fit_idx)'
    val_call = ('make_dataset("train", _val_idx, augment=False)' if mid == "M22"
                else 'make_dataset("train", _val_idx)')
    return '''
# ---- Genuine held-out validation split  [''' + MARKER + '''] --------------------------------
# Was val_loader=test_loader: checkpoint selection read the official TEST set every epoch (60
# shots). This carves a patient-grouped slice OUT OF TRAIN ONLY; the test split is not touched
# until the single FINAL EVALUATION cell below.
_gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=CFG["seed"])
_fit_idx, _val_idx = next(_gss.split(np.arange(len(df_train)),
                                     groups=df_train["patient_id"].values))

_fit_patients = set(df_train["patient_id"].values[_fit_idx])
_val_patients = set(df_train["patient_id"].values[_val_idx])
assert not (_fit_patients & _val_patients), "fit/val patient overlap -- split failed"
assert not (_val_patients & test_patients), "validation patient leaked into the test set"

fit_loader = make_loader(''' + fit_call + ''', shuffle=True)
val_loader_heldout = make_loader(''' + val_call + ''', shuffle=False)
fit_class_weights = compute_class_weights(train_labels[_fit_idx], CFG["num_classes"])
fit_class_weights_tensor = torch.tensor(fit_class_weights, dtype=torch.float32, device=DEVICE)

print(f"Validation split : {len(_fit_patients)} fit patients ({len(_fit_idx)} cycles) / "
      f"{len(_val_patients)} val patients ({len(_val_idx)} cycles) -- carved from TRAIN only. "
      f"Test set ({len(test_patients)} patients) is not read until final evaluation.")

'''


RUN_NEW = '''final_run = train_model(
    config=best_config,
    train_loader=fit_loader,
    val_loader=val_loader_heldout,
    num_epochs=CFG["num_epochs"],
    class_weights_t=fit_class_weights_tensor,
    ckpt_dir=CFG["ckpt_dir"],
    resume=True,
    verbose=True,
)'''

JSON_OLD = '''"test_patients": int(len(test_patients)),'''
JSON_NEW = '''"test_patients": int(len(test_patients)),
        "val_patients": int(len(_val_patients)),
        "val_cycles": int(len(_val_idx)),
        "fit_patients": int(len(_fit_patients)),
        "fit_cycles": int(len(_fit_idx)),
        "validation_method": ("GroupShuffleSplit(test_size=0.15, seed=CFG['seed']) on TRAIN "
                              "patients only -- checkpoint selection does not read the test set"),
        "test_touched_times": 1,'''


def patch_one(mid, path, dry):
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print(f"[{mid}] already patched — skipped")
        return True

    done = {"import": False, "run": False, "json": False}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        o = s
        if not done["import"] and IMPORT_OLD in s:
            s = s.replace(IMPORT_OLD, IMPORT_NEW, 1)
            done["import"] = True
        if not done["run"] and RUN_OLD in s:
            s = s.replace(RUN_OLD, val_block(mid) + RUN_NEW, 1)
            done["run"] = True
        if not done["json"] and JSON_OLD in s:
            s = s.replace(JSON_OLD, JSON_NEW, 1)
            done["json"] = True
        if s != o:
            cell["source"] = s.splitlines(keepends=True)

    missing = [k for k, v in done.items() if not v]
    if missing:
        print(f"[{mid}] FAILED — could not find: {missing}")
        return False

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), mid, "exec")

    if not dry:
        json.dump(nb, open(path, "w"), indent=1)
    print(f"[{mid}] {'would patch' if dry else 'patched'} — import, val split, results JSON")
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
