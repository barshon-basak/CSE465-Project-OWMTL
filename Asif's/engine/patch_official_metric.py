#!/usr/bin/env python3
"""
Make M2 / M3 / M22 report -- and SELECT ON -- the official ICBHI 2017 score, and dump per-cycle
scores so models can be compared with paired tests without retraining.

    python3 patch_official_metric.py [--dry-run] [--only M2 M3]

WHY THIS EXISTS
---------------
All three notebooks carry a cell headed "ICBHI Score = (Se + Sp) / 2", which looks correct, over
definitions that are not:

    macro_se = mean(per-class recall over all 4 classes)
    macro_sp = mean(per-class specificity over all 4 classes)
    icbhi_score = (macro_se + macro_sp) / 2

The official challenge metric pools the three abnormal classes against Normal:

    Se = correctly-classified abnormal events / all abnormal events   (Crackle+Wheeze+Both)
    Sp = correctly-classified Normal events   / all Normal events
    ICBHI score = (Se + Sp) / 2

`icbhi_score_audit.py` measured the gap at +0.11 mean / +0.22 worst across 14 models. The macro
variant also MASKS pathologies: a model with specificity 0.0000 scored 0.5506 under it.

The metric is not only reported -- it drives `best_model.pth` selection and early stopping. So a
re-run without this patch does not merely print an inflated number, it trains a differently
selected model. That is why this must land before the re-runs, not after.

WHAT IT DOES
------------
1. compute_metrics() gains icbhi_se_official / icbhi_sp_official / icbhi_score_official.
   The macro fields are KEPT and relabelled non-standard -- having both from one run is what
   lets the paper quote per-model inflation exactly.
2. Checkpoint selection, early stopping and the results JSON switch to the official score.
3. The final-evaluation cell dumps per-cycle probabilities in the owmtl_scores.py schema, keyed
   by `<wav_stem>#<start>-<end>` -- stable ACROSS notebooks, so M2/M3/M22 can be compared with
   paired DeLong without anyone retraining anything.

Idempotent: files carrying the marker are skipped.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_OFFICIAL_METRIC_V1"

PATHS = {
    "M2":  "Asif's/M2/M2_cnn_baseline_tuned.ipynb",
    "M3":  "Asif's/M3/M3_lightweight_backbone.ipynb",
    "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb",
}

# ---------------------------------------------------------------- 1. metric definition
OFFICIAL_BLOCK = '''
    # ---- OFFICIAL ICBHI 2017 SCORE  [''' + MARKER + '''] ----------------------------
    # Se = correctly-classified ABNORMAL events / all abnormal events (Crackle+Wheeze+Both
    #      POOLED -- not macro-averaged across the three classes)
    # Sp = correctly-classified Normal events   / all Normal events
    # This is the challenge's own definition and the one every published ICBHI number uses.
    # The macro_se/macro_sp pair below is a DIFFERENT, non-standard metric that runs ~0.11
    # higher on average (+0.22 worst case, measured across 14 of our own models). Both are
    # returned so the gap can be quoted per model from a single run.
    _cmf = cm.astype(float)
    _ni = list(class_names).index("Normal") if "Normal" in list(class_names) else 0
    _n_tot = float(_cmf[_ni].sum())
    _sp_off = float(_cmf[_ni, _ni] / _n_tot) if _n_tot else 0.0
    _abn = [i for i in range(n) if i != _ni]
    _a_tot = float(sum(_cmf[i].sum() for i in _abn))
    _se_off = float(sum(_cmf[i, i] for i in _abn) / _a_tot) if _a_tot else 0.0
    _score_off = (_se_off + _sp_off) / 2.0
'''

OLD_RETURN = '''    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(np.mean(precisions)), 4),
        "recall_macro": round(macro_se, 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "specificity_macro": round(macro_sp, 4),
        "icbhi_score": round((macro_se + macro_sp) / 2, 4),'''

NEW_RETURN = '''    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(np.mean(precisions)), 4),
        "recall_macro": round(macro_se, 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "specificity_macro": round(macro_sp, 4),
        # --- official challenge metric: THE one to report ---
        "icbhi_se_official": round(_se_off, 4),
        "icbhi_sp_official": round(_sp_off, 4),
        "icbhi_score_official": round(_score_off, 4),
        # --- non-standard macro variant, kept only to quantify the gap ---
        "icbhi_score_macro_NONSTANDARD": round((macro_se + macro_sp) / 2, 4),
        "icbhi_score": round((macro_se + macro_sp) / 2, 4),   # legacy alias; do NOT report
        "icbhi_inflation_macro_minus_official": round(
            (macro_se + macro_sp) / 2 - _score_off, 4),'''

OLD_HEADER = '''# ICBHI Score = (Se + Sp) / 2   [primary metric for this model]
#   Se = macro-average sensitivity (recall)
#   Sp = macro-average specificity, TN / (TN + FP) per class'''

NEW_HEADER = '''# PRIMARY METRIC: official ICBHI 2017 score = (Se + Sp) / 2 where
#   Se = correctly-classified ABNORMAL events / all abnormal (Crackle+Wheeze+Both POOLED)
#   Sp = correctly-classified Normal events   / all Normal
#
# A macro-averaged variant is ALSO computed and reported as
# `icbhi_score_macro_NONSTANDARD`. It is not the challenge metric and runs ~0.11 higher.
# Report `icbhi_score_official`. The key `icbhi_score` is a legacy alias for the macro
# variant, kept only so older plotting code keeps running.'''

OLD_PRINT = '''    print(f"{prefix}ICBHI Score       : {m['icbhi_score']:.4f}   <- primary metric")'''
NEW_PRINT = '''    print(f"{prefix}ICBHI OFFICIAL    : {m['icbhi_score_official']:.4f}   <- PRIMARY "
          f"(Se={m['icbhi_se_official']:.4f} Sp={m['icbhi_sp_official']:.4f})")
    print(f"{prefix}  macro variant   : {m['icbhi_score_macro_NONSTANDARD']:.4f}  "
          f"(NON-STANDARD, +{m['icbhi_inflation_macro_minus_official']:.4f} inflated)")'''

# ---------------------------------------------------------------- 2. selection metric
SEL_OLD = '''        score = va_m["icbhi_score"]              # primary metric'''
SEL_NEW = '''        score = va_m["icbhi_score_official"]     # primary metric: OFFICIAL ICBHI score
        # Selecting on the macro variant used to pick a different epoch. Checkpoint choice,
        # not just reporting, depends on this line.'''

BEST_OLD = '''        return int(max(hist, key=lambda h: h.get("val_icbhi_score", 0.0))["epoch"])'''
BEST_NEW = '''        return int(max(hist, key=lambda h: h.get("val_icbhi_score_official",
                                                 h.get("val_icbhi_score", 0.0)))["epoch"])'''

HIST_OLD = '''            "val_icbhi_score": va_m["icbhi_score"],'''
HIST_NEW = '''            "val_icbhi_score_official": va_m["icbhi_score_official"],
            "val_icbhi_se_official": va_m["icbhi_se_official"],
            "val_icbhi_sp_official": va_m["icbhi_sp_official"],
            "train_icbhi_score_official": tr_m["icbhi_score_official"],
            "val_icbhi_score": va_m["icbhi_score"],'''

EPOCH_OLD = '''                  f"Sp={va_m['specificity_macro']:.4f} ICBHI={score:.4f} "'''
EPOCH_NEW = '''                  f"Sp={va_m['icbhi_sp_official']:.4f} ICBHI={score:.4f} "'''

RESJSON_OLD = '''        "primary_metric": "icbhi_score",
        "primary_metric_value": float(final_metrics["icbhi_score"]),'''
RESJSON_NEW = '''        "primary_metric": "icbhi_score_official",
        "primary_metric_value": float(final_metrics["icbhi_score_official"]),
        "icbhi_se_official": final_metrics["icbhi_se_official"],
        "icbhi_sp_official": final_metrics["icbhi_sp_official"],
        "icbhi_score_official": final_metrics["icbhi_score_official"],
        "icbhi_score_macro_NONSTANDARD": final_metrics["icbhi_score_macro_NONSTANDARD"],
        "icbhi_metric_note": (
            "primary_metric_value is the OFFICIAL ICBHI 2017 score: Se over pooled abnormal "
            "events (Crackle+Wheeze+Both), Sp over Normal. The macro-averaged variant this "
            "notebook previously reported is retained as icbhi_score_macro_NONSTANDARD."),'''


def score_dump(model_id):
    return '''

# ============================================================
# PER-CYCLE SCORE DUMP  [''' + MARKER + ''']
# ============================================================
# Writes every test cycle's predicted probabilities in the owmtl_scores.py schema, so M2/M3/M22
# can be compared with PAIRED DeLong tests later without retraining anything.
#
# unit_id is `<wav_stem>#<start>-<end>`, taken from df_test -- deliberately NOT the row ordinal.
# All three notebooks derive cycles from the same annotation .txt files, so this key refers to
# the same physical cycle in every notebook, which is what makes the tests pairable. An ordinal
# would silently mis-pair if any notebook ever filtered or reordered its test set.
import csv as _csv

_MODEL_ID = "''' + model_id + '''"
model.eval()
_probs = []
with torch.no_grad():
    for _bx, _by in test_loader:
        _bx = _bx.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=USE_AMP):
            _lg = model(_bx)
        _probs.append(torch.softmax(_lg.float(), dim=1).cpu().numpy())
_probs = np.concatenate(_probs, axis=0)

assert len(_probs) == len(df_test), (
    f"score dump misaligned: {len(_probs)} predictions vs {len(df_test)} test rows. "
    "test_loader must be shuffle=False and cover df_test exactly.")
assert list(np.asarray(final_targets)) == list(df_test["label"].values.astype(int)), (
    "score dump misaligned: test_loader label order does not match df_test. "
    "Do NOT pair these scores with another model's until this is resolved.")

_dump = os.path.join(CFG["results_dir"], f"scores_{_MODEL_ID}.csv")
_names = list(CFG["classes"])
with open(_dump, "w", newline="") as _f:
    _w = _csv.writer(_f)
    _w.writerow(["model_id", "split", "unit_type", "unit_id", "score_name", "score", "label"])
    for _i in range(len(df_test)):
        _row = df_test.iloc[_i]
        _stem = os.path.splitext(os.path.basename(str(_row["wav_path"])))[0]
        _uid = f"{_stem}#{float(_row['start']):.3f}-{float(_row['end']):.3f}"
        _true = int(_row["label"])
        for _c, _cn in enumerate(_names):
            _w.writerow([_MODEL_ID, "test", "cycle", _uid, f"p_{_cn.lower()}",
                         f"{_probs[_i, _c]:.6f}", int(_true == _c)])
        _w.writerow([_MODEL_ID, "test", "cycle", _uid, "p_abnormal",
                     f"{1.0 - _probs[_i, 0]:.6f}", int(_true != 0)])

print(f"Per-cycle scores -> {_dump}  ({len(df_test)} cycles x {len(_names) + 1} score names)")
print("Pair across models with:  Asif's/Statistics/owmtl_scores.py")
'''


def patch_one(mid, path, dry):
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print(f"[{mid}] already patched — skipped")
        return True

    subs = [("metric header", OLD_HEADER, NEW_HEADER),
            ("metric return", OLD_RETURN, NEW_RETURN),
            ("metric print", OLD_PRINT, NEW_PRINT),
            ("selection metric", SEL_OLD, SEL_NEW),
            ("best-epoch recovery", BEST_OLD, BEST_NEW),
            ("history keys", HIST_OLD, HIST_NEW),
            ("epoch print", EPOCH_OLD, EPOCH_NEW),
            ("results json", RESJSON_OLD, RESJSON_NEW)]
    done = {k: False for k, _, _ in subs}

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        orig = s
        for label, old, new in subs:
            if not done[label] and old in s:
                s = s.replace(old, new, 1)
                done[label] = True
        # official block goes immediately before the metric return
        if "macro_sp = float(np.mean(specificities))" in s and MARKER not in s:
            s = s.replace("    return {\n        \"accuracy\":",
                          OFFICIAL_BLOCK + "\n    return {\n        \"accuracy\":", 1)
        if s != orig:
            cell["source"] = s.splitlines(keepends=True)

    # append the score dump to the final-evaluation cell
    dumped = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        if "final_metrics = compute_metrics(" in s:
            cell["source"] = (s + score_dump(mid)).splitlines(keepends=True)
            dumped = True
            break

    missing = [k for k, v in done.items() if not v] + ([] if dumped else ["score dump"])
    if missing:
        print(f"[{mid}] FAILED — could not find: {missing}")
        return False

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"{mid}", "exec")

    if not dry:
        json.dump(nb, open(path, "w"), indent=1)
    print(f"[{mid}] {'would patch' if dry else 'patched'} — 8 substitutions + score dump")
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
