#!/usr/bin/env python3
"""
Fix two places where patch_official_metric.py left the notebooks self-inconsistent.

    python3 patch_report_consistency.py [--dry-run]

1. PLOTS still drew the macro variant under the title "ICBHI Score (primary metric)", while
   checkpoint selection had switched to the official score. A reader would have seen a curve that
   did not correspond to the epoch actually selected. The Se/Sp panel had the same problem: it
   plotted macro-averaged Se/Sp labelled simply "Se"/"Sp".

2. Each notebook's OWN row in its primary summary table -- the one printed first, before any
   cross-model comparison -- was built from `final_metrics["icbhi_score"]`, the macro alias,
   even after checkpoint selection and every other display had switched to the official score.
   The M12 ABLATION TABLE compared the new run against M1/M4 reference rows that are
   (a) macro-metric numbers and (b) computed on a DIFFERENT split -- the audit records M1 as
   `patient_independent_60_40` and the old M2/M3/M22 as `..._patient_id_fallback`, i.e. the
   11-patient bug. Printing a "M2 vs M1" delta across that gap produces a number that looks
   meaningful and is not. Official values recovered from ICBHI_SCORE_AUDIT.md are shown instead,
   with the split mismatch stated and the delta suppressed.

   NOTE (2026-08-29, post-hoc): an earlier version of this script suppressed the `delta`
   assignment but left `if abs(delta) < spread:` a few lines further down still reading it --
   that block belonged to the SAME cell but wasn't part of the original match, so removing
   the definition without removing the use produced a live NameError, caught only when Asif
   actually ran cell 16 on Colab after M2's v3/v4 GPU runs. TABLE_OLD/TABLE_NEW now span both
   blocks so the two can't drift apart again; the spread print survives as an informational
   line, no longer gated on the deleted variable.

   NOTE (2026-08-29, second post-hoc): fix #1 above corrected the PLOTS but missed that each
   notebook's own row in its OWN primary table (`m2_row`/`m3_row`/`m22`) was populated from the
   same macro alias -- so M2's printed ABLATION TABLE showed 0.5480 in its ICBHI column on the
   very same run whose corrected score, printed three lines later by fix #2, was 0.4720. Not a
   crash, so nothing caught it until the numbers were compared by eye. ROW_SUBS below fixes the
   dict construction in all three notebooks so the row a reader sees FIRST already matches the
   number the corrected-comparison block prints after it.

Idempotent.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MARKER = "OWMTL_REPORT_CONSISTENCY_V1"

PATHS = {
    "M2":  "Asif's/M2/M2_cnn_baseline_tuned.ipynb",
    "M3":  "Asif's/M3/M3_lightweight_backbone.ipynb",
    "M22": "Asif's/M22/M22_mobilenet_specaugment.ipynb",
}

PLOT_SUBS = [
 ('''axes[0].plot(epochs, [h["val_icbhi_score"] for h in history], color="#2ca02c", linewidth=1.8,
             label="Val ICBHI Score")''',
  '''# [''' + MARKER + '''] plot the OFFICIAL score -- the one checkpoints are selected on.
_vk = "val_icbhi_score_official" if "val_icbhi_score_official" in history[0] else "val_icbhi_score"
_tk = "train_icbhi_score_official" if "train_icbhi_score_official" in history[0] else "train_icbhi_score"
axes[0].plot(epochs, [h[_vk] for h in history], color="#2ca02c", linewidth=1.8,
             label="Val ICBHI Score (official)")'''),
 ('''axes[0].plot(epochs, [h["train_icbhi_score"] for h in history], color=TRAIN_C, alpha=0.55,
             linewidth=1.4, label="Train ICBHI Score")''',
  '''axes[0].plot(epochs, [h[_tk] for h in history], color=TRAIN_C, alpha=0.55,
             linewidth=1.4, label="Train ICBHI Score (official)")'''),
 ('''axes[0].set_title("ICBHI Score (primary metric)"); axes[0].set_xlabel("Epoch")''',
  '''axes[0].set_title("Official ICBHI Score (primary metric — drives checkpoint selection)")
axes[0].set_xlabel("Epoch")'''),
 ('''axes[1].plot(epochs, [h["val_recall_macro"] for h in history], label="Se (macro sensitivity)")''',
  '''_sek = "val_icbhi_se_official" if "val_icbhi_se_official" in history[0] else "val_recall_macro"
_spk = "val_icbhi_sp_official" if "val_icbhi_sp_official" in history[0] else "val_specificity_macro"
axes[1].plot(epochs, [h[_sek] for h in history], label="Se (official: pooled abnormal)")'''),
 ('''axes[1].plot(epochs, [h["val_specificity_macro"] for h in history], label="Sp (macro specificity)")''',
  '''axes[1].plot(epochs, [h[_spk] for h in history], label="Sp (official: Normal)")'''),
]

TABLE_OLD = '''m1_icbhi = REFERENCE_ROWS[0]["icbhi"]
delta = final_metrics["icbhi_score"] - m1_icbhi
print(f"\\nM2 vs. M1 (the point of this run): ICBHI {final_metrics['icbhi_score']:.4f} "
      f"vs {m1_icbhi:.4f}  =>  {delta:+.4f}")
if sweep_results:
    spread = sweep_results[0]["std_icbhi"]
    print(f"Sweep cross-validation std for the winning config: +/- {spread:.4f}")
    if abs(delta) < spread:
        print("NOTE: the gap is smaller than the CV fold-to-fold spread. Report it as "
              "'comparable', not as an improvement — this is exactly the kind of claim "
              "a reviewer will check.")'''

TABLE_NEW = '''# [''' + MARKER + '''] The old "M2 vs M1" delta compared metrics across a split boundary.
# Per Asif's/audit/ICBHI_SCORE_AUDIT.md, M1 ran on `patient_independent_60_40` and the previous
# M2/M3/M22 on `..._patient_id_fallback` -- the 11-test-patient bug. This run is on the corrected
# official split. A difference across that gap measures the split change, not the model, so the
# delta is deliberately NOT printed.
print("\\n" + "=" * 104)
print("COMPARISON WITH EARLIER RUNS — READ THE CAVEAT")
print("=" * 104)
print(f"This run  (corrected official split) : OFFICIAL ICBHI "
      f"{final_metrics['icbhi_score_official']:.4f}  "
      f"(Se={final_metrics['icbhi_se_official']:.4f} Sp={final_metrics['icbhi_sp_official']:.4f})")
print(f"{'':38}  macro variant {final_metrics['icbhi_score_macro_NONSTANDARD']:.4f} "
      f"(+{final_metrics['icbhi_inflation_macro_minus_official']:.4f} inflated — do not report)")
print("")
print("Earlier runs, official scores recomputed by the audit from their confusion matrices:")
for _n, _off, _rep, _sp in (("M1 ", 0.6143, 0.7181, "patient_independent_60_40"),
                            ("M2 ", 0.6138, 0.7227, "60_40_patient_id_fallback  <- 11-patient bug"),
                            ("M3 ", 0.5895, 0.6984, "60_40_patient_id_fallback  <- 11-patient bug"),
                            ("M22", 0.6495, 0.7077, "60_40_patient_id_fallback  <- 11-patient bug")):
    print(f"   {_n}  official {_off:.4f}   (as-reported {_rep:.4f})   split: {_sp}")
print("")
print("NOT COMPARABLE to the number above: different split, and in three cases a buggy one.")
print("Published ICBHI SOTA on the official 60/40 split is ~0.60-0.65 — compare against THAT.")
if sweep_results:
    spread = sweep_results[0]["std_icbhi"]
    print(f"Sweep cross-validation std for the winning config: +/- {spread:.4f}  "
          "(fold-to-fold spread seen during hyperparameter selection; informational only -- "
          "the cross-split M1 comparison this used to gate on is no longer computed, see above)")'''


ROW_SUBS = [
 ('''"icbhi": final_metrics["icbhi_score"], "params": total_params,''',
  '''"icbhi": final_metrics["icbhi_score_official"], "params": total_params,'''),
 ('''"sp": final_metrics["specificity_macro"], "icbhi": final_metrics["icbhi_score"],''',
  '''"sp": final_metrics["specificity_macro"], "icbhi": final_metrics["icbhi_score_official"],'''),
]


def patch_one(mid, path, dry):
    nb = json.load(open(path))
    if MARKER in json.dumps(nb):
        print(f"[{mid}] already patched — skipped")
        return True
    done = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        s = "".join(cell["source"])
        o = s
        for old, new in PLOT_SUBS:
            if old in s:
                s = s.replace(old, new, 1)
                done.append("plot")
        for old, new in ROW_SUBS:
            if old in s:
                s = s.replace(old, new, 1)
                done.append("row")
        if TABLE_OLD.replace("M2 vs. M1", f"{mid} vs. M1") in s:
            s = s.replace(TABLE_OLD.replace("M2 vs. M1", f"{mid} vs. M1"), TABLE_NEW, 1)
            done.append("table")
        elif TABLE_OLD in s:
            s = s.replace(TABLE_OLD, TABLE_NEW, 1)
            done.append("table")
        if s != o:
            cell["source"] = s.splitlines(keepends=True)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), mid, "exec")
    if not dry:
        json.dump(nb, open(path, "w"), indent=1)
    print(f"[{mid}] {'would patch' if dry else 'patched'} — {done.count('plot')} plot fix(es), "
          f"{done.count('row')} row fix(es), {done.count('table')} table fix(es)")
    return bool(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(PATHS))
    a = ap.parse_args()
    ok = all(patch_one(m, os.path.join(REPO, PATHS[m]), a.dry_run) for m in a.only)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
