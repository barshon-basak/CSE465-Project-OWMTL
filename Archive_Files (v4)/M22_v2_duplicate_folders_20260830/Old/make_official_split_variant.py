#!/usr/bin/env python3
"""
Generate `m22-official-notebook.ipynb` from `m22-v2-notebook.ipynb`, changing ONE thing:
the split becomes the PUBLISHED ICBHI split verbatim (539/381 recordings, 2,756 test cycles,
49 test patients) instead of the corrected patient-independent one (551/369, 2,636, 47).

WHY
---
`Papers/RESULT_COMPARISON.md` section E.2: the best model M22_v2 has no number on the exact
partition the literature reports on, so it cannot be placed in the same column as RespireNet,
Patch-Mix, PAFA, BTS++ and the rest. It cannot be obtained by re-scoring the existing
checkpoint, because patients 156 and 218 -- the two that straddle the published split -- are in
M22_v2's TRAINING set. A fresh run on the published split is the only way to get it.

Architecture, optimiser, schedule, class weights, seed, preprocessing and every SpecAugment
parameter are untouched, so the resulting number differs from M22_v2's 0.5602 by the split and
nothing else.

USAGE
-----
    python make_official_split_variant.py

Then upload `m22-official-notebook.ipynb` to Kaggle (same dataset attachments as the M22_v2
run), Run All, and commit the emitted `results_M22_v2_official.json`.

The output run is NOT patient-independent and the notebook says so in three places. Report it
only as the directly-comparable number, never as the project's protocol-compliant result.
"""
import json
import os
import sys

SRC = "m22-v2-notebook.ipynb"
DST = "m22-official-notebook.ipynb"

# (cell index, anchor found in that cell, replacement) -- every anchor must match exactly once.
PATCHES = [
    (4,
     '_mid, _mname, _aug = _IDS[VARIANT]\n',
     '_mid, _mname, _aug = _IDS[VARIANT]\n'
     '\n'
     '# ---- THE SECOND SWITCH -------------------------------------------------------\n'
     '# "corrected" -> 551/369 recs, 2,636 test cycles, 47 patients, patient-independent\n'
     '# "official"  -> 539/381 recs, 2,756 test cycles, 49 patients, VERBATIM published\n'
     '#                split; NOT patient-independent (156 and 218 straddle it). This is\n'
     '#                the partition the ICBHI literature reports on.\n'
     'SPLIT_MODE = "official"\n'
     'if SPLIT_MODE == "official":\n'
     '    _mid = _mid + "_official"\n'
     '    _mname = _mname.replace("corrected official split",\n'
     '                            "published official split, verbatim")\n'
     '# ------------------------------------------------------------------------------\n'),

    (4,
     '    "split_policy": "reassign_to_train",\n'
     '    "split_method": "official_60_40_patient_independent_corrected",\n',
     '    "split_policy": ("reassign_to_train" if SPLIT_MODE == "corrected"\n'
     '                     else "none_published_verbatim"),\n'
     '    "split_method": ("official_60_40_patient_independent_corrected" if SPLIT_MODE == "corrected"\n'
     '                     else "official_60_40_published_verbatim_NOT_patient_independent"),\n'),

    (7,
     'corrected = {s: ("train" if patient_id_from_stem(s) in overlap else v)\n'
     '             for s, v in split_map.items()}\n',
     'if SPLIT_MODE == "corrected":\n'
     '    corrected = {s: ("train" if patient_id_from_stem(s) in overlap else v)\n'
     '                 for s, v in split_map.items()}\n'
     'else:\n'
     '    corrected = dict(split_map)          # published split, verbatim -- no correction\n'),

    (7,
     'assert c_tr == CORRECTED_TRAIN_RECS, f"corrected train {c_tr} != {CORRECTED_TRAIN_RECS}"\n'
     'assert c_te == CORRECTED_TEST_RECS,  f"corrected test  {c_te} != {CORRECTED_TEST_RECS}"\n'
     'assert not still_leaking, f"patients still on both sides: {still_leaking}"\n',
     '_exp_tr, _exp_te = ((CORRECTED_TRAIN_RECS, CORRECTED_TEST_RECS) if SPLIT_MODE == "corrected"\n'
     '                    else (OFFICIAL_TRAIN_RECS, OFFICIAL_TEST_RECS))\n'
     'assert c_tr == _exp_tr, f"train {c_tr} != {_exp_tr}"\n'
     'assert c_te == _exp_te, f"test  {c_te} != {_exp_te}"\n'
     'if SPLIT_MODE == "corrected":\n'
     '    assert not still_leaking, f"patients still on both sides: {still_leaking}"\n'
     'else:\n'
     '    assert set(still_leaking) == OFFICIAL_OVERLAP, \\\n'
     '        f"expected {sorted(OFFICIAL_OVERLAP)} to straddle, got {sorted(still_leaking)}"\n'),

    (7,
     'print(f"[OK] corrected split: {c_tr} train / {c_te} test recordings "\n'
     '      f"({100 * c_te / len(corrected):.1f}% test), patient-independent.")\n',
     'if SPLIT_MODE == "corrected":\n'
     '    print(f"[OK] corrected split: {c_tr} train / {c_te} test recordings "\n'
     '          f"({100 * c_te / len(corrected):.1f}% test), patient-independent.")\n'
     'else:\n'
     '    print(f"[OK] PUBLISHED split verbatim: {c_tr} train / {c_te} test recordings "\n'
     '          f"({100 * c_te / len(corrected):.1f}% test).")\n'
     '    print("[WARN] this run is NOT patient-independent -- patients 156 and 218 have "\n'
     '          "recordings on both sides.")\n'
     '    print("       It exists solely to be directly comparable to published ICBHI work. "\n'
     '          "Never report it as the project\'s protocol-compliant result.")\n'),

    # -- Cell 5 re-checks patient overlap at cycle level. In official mode the overlap is
    # -- real and expected, so the assertion becomes "exactly the two known patients".
    (8,
     'rows, n_missing, n_unlisted = [], 0, 0\n',
     'rows, n_missing, n_unlisted, unlisted_stems = [], 0, 0, []\n'),

    (8,
     '        n_unlisted += 1; continue\n',
     '        n_unlisted += 1; unlisted_stems.append(stem); continue\n'),

    (8,
     'if n_unlisted: print(f"skipped {n_unlisted} recordings absent from the split file")\n',
     'if n_unlisted:\n'
     '    print(f"skipped {n_unlisted} recordings absent from the split file: {unlisted_stems}")\n'),

    (8,
     'leak = set(df_train.patient_id) & set(df_test.patient_id)\n'
     'assert not leak, f"PATIENT LEAKAGE: {sorted(leak)}"\n',
     'leak = set(df_train.patient_id) & set(df_test.patient_id)\n'
     'if SPLIT_MODE == "corrected":\n'
     '    assert not leak, f"PATIENT LEAKAGE: {sorted(leak)}"\n'
     'else:\n'
     '    assert leak == OFFICIAL_OVERLAP, \\\n'
     '        f"expected exactly {sorted(OFFICIAL_OVERLAP)} on both sides, got {sorted(leak)}"\n'
     '    _n_leak_cyc = int(df_test.patient_id.isin(leak).sum())\n'
     '    print(f"[WARN] patients {sorted(leak)} appear in BOTH splits: {_n_leak_cyc} of "\n'
     '          f"{len(df_test)} test cycles ({100 * _n_leak_cyc / len(df_test):.1f}%) come "\n'
     '          f"from a patient the model saw in training.")\n'
     '    print("       This is the published split\'s own flaw, reproduced deliberately so "\n'
     '          "the score is comparable to published work. Never report this run as "\n'
     '          "patient-independent.")\n'),

    (8,
     'print(f"patients: {df_train.patient_id.nunique()} train / {df_test.patient_id.nunique()} test  "\n'
     '      f"[no overlap]")\n',
     '_ov = "no overlap" if SPLIT_MODE == "corrected" else f"{len(leak)} OVERLAPPING patients"\n'
     'print(f"patients: {df_train.patient_id.nunique()} train / "\n'
     '      f"{df_test.patient_id.nunique()} test  [{_ov}]")\n'),

    # -- The emitted JSON must not claim patient-independence it does not have; the audit
    # -- tooling and the master table both read these fields.
    (21,
     '        "patient_leakage_verified": True,\n'
     '        "overlap_patients_reassigned_to_train": sorted(OFFICIAL_OVERLAP),\n',
     '        "patient_leakage_verified": (SPLIT_MODE == "corrected"),\n'
     '        "is_patient_independent": (SPLIT_MODE == "corrected"),\n'
     '        "overlap_patients_reassigned_to_train": (sorted(OFFICIAL_OVERLAP)\n'
     '                                                 if SPLIT_MODE == "corrected" else []),\n'
     '        "overlap_patients_present_in_both": ([] if SPLIT_MODE == "corrected"\n'
     '                                             else sorted(OFFICIAL_OVERLAP)),\n'
     '        "comparable_to_published_icbhi_split": (SPLIT_MODE == "official"),\n'),

    (21,
     '        "notes": (\n'
     '            "Re-run of M22 on the CORRECTED official split. ',
     '        "notes": (\n'
     '            ("PUBLISHED ICBHI SPLIT, VERBATIM (539/381 recordings; 2,756 test cycles from "\n'
     '             "49 test patients) so that the score is directly comparable to published work "\n'
     '             "on this benchmark. NOT patient-independent: patients 156 and 218 have "\n'
     '             "recordings in both train and test. Companion run to M22_v2, which is this "\n'
     '             "same model, seed and schedule on the corrected patient-independent split; "\n'
     '             "the two differ by the split and nothing else. " if SPLIT_MODE == "official"\n'
     '             else "") +\n'
     '            "Re-run of M22 on the CORRECTED official split. '),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, SRC)
    if not os.path.exists(src):
        sys.exit(f"error: {SRC} not found next to this script")

    with open(src, encoding="utf-8") as fh:
        nb = json.load(fh)

    for idx, anchor, replacement in PATCHES:
        cell = nb["cells"][idx]
        text = "".join(cell["source"])
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"error: anchor matched {n} times in cell {idx} (expected 1). "
                     f"The source notebook changed; update PATCHES.\nAnchor:\n{anchor}")
        cell["source"] = text.replace(anchor, replacement)

    # Drop stale outputs so the committed notebook shows only the new run's results.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    dst = os.path.join(here, DST)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)

    print(f"[OK] wrote {DST}  ({len(PATCHES)} patches applied)")
    print("     SPLIT_MODE   = 'official'")
    print("     model_id     -> M22_v2_official")
    print("     expected     -> 539/381 recordings, 2,756 test cycles, 49 test patients")
    print()
    print("Next: upload to Kaggle with the same dataset attachments as the M22_v2 run,")
    print("Run All, then commit results_M22_v2_official.json and preds_M22_v2_official.csv.")


if __name__ == "__main__":
    main()
