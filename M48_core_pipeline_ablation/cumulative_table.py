"""
M48 — the CUMULATIVE ablation ladder (baseline, +C1, +C1+C2, ... , full pipeline).

WHY A SECOND ABLATION TABLE
    M45 is leave-one-out: start from the full pipeline, remove one component, report the
    loss. A cumulative table starts from a bare baseline and adds components one at a time.
    The two answer different questions and only agree when components do not interact:

        leave-one-out delta : what this component is worth GIVEN everything else is present
        cumulative delta    : what this component is worth given only what came before it

    A component can be worth a great deal on its own and nothing at the margin (two stages
    fixing the same nuisance), or nothing on its own and a great deal at the margin. Report
    both; neither is the "true" contribution.

THE ORDER PROBLEM, AND WHY THIS FILE REPORTS TWO ORDERS
    Every cumulative table silently makes a claim: that the order components were added in
    is the natural one. It is not a neutral presentation - the delta credited to a component
    depends on what preceded it. Rather than caveat that, this file measures it. Two
    orderings are assembled that differ ONLY in whether SpecAugment or the class-weighted
    loss is added at rung 4, and both are fully populated from the same runs, because M45
    happens to contain both intermediate configurations (A3 and A1). The gap between what
    the two orderings credit to the same component IS the order effect, in real numbers.

THE LADDER
    rung   pretrain  fine-tune  class-weight  SpecAugment  amp-norm   source
    S0        -          -           -             -           -      M48 (new)
    S1        v          -           -             -           -      M48 (new)
    S2        v          v           -             -           -      M48 (new)
    S3        v          v      [order A: -  ] [order A: v ]    -     A3 (order A) / A1 (order B)
                              [order B: v  ] [order B: -  ]
    S4        v          v           v             v           -      M22_v2  (= M45 A0)
    S5        v          v           v             v           v      M45 P3

    Three of six rungs already exist. Only S0, S1 and S2 need the GPU, and they are rows in
    `m48_gpu_rows.py`, so the Kaggle notebook produces them alongside the seed band.

NO NUMBER IN THIS TABLE IS INVENTED
    A rung whose run does not exist is written `pending`, never filled with a plausible
    value. This project's paper is about reported numbers that could not be recomputed from
    a committed confusion matrix; a fabricated ablation row would be that same fault with
    the authors' knowledge. Every score here is recomputed from the source run's committed
    `confusion_matrix_raw`.

RUNNING
    python cumulative_table.py              # assemble whatever exists, mark the rest pending
    python cumulative_table.py --selftest   # toggle/consistency checks, writes nothing
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
M45 = os.path.join(REPO, "Asif's", "M45")

COMPONENTS = [
    ("pretrain",    "ImageNet pre-training"),
    ("finetune",    "Full backbone fine-tuning"),
    ("classweight", "Inverse-frequency class-weighted CE"),
    ("specaug",     "SpecAugment (training split only)"),
    ("ampnorm",     "Per-cycle peak amplitude normalisation"),
]

# rung -> (toggles, source json, human label)
def T(pre, ft, cw, sa, an):
    return dict(pretrain=pre, finetune=ft, classweight=cw, specaug=sa, ampnorm=an)


SHARED = {
    "S0": (T(0, 0, 0, 0, 0), os.path.join(HERE, "results_M48_S0.json"),
           "Baseline — MobileNetV2, random init, frozen, plain CE"),
    "S1": (T(1, 0, 0, 0, 0), os.path.join(HERE, "results_M48_S1.json"),
           "+ ImageNet pre-training"),
    "S2": (T(1, 1, 0, 0, 0), os.path.join(HERE, "results_M48_S2.json"),
           "+ full fine-tuning"),
    "S4": (T(1, 1, 1, 1, 0),
           os.path.join(REPO, "Asif's", "M22_v2", "Results", "results_M22_v2.json"),
           "+ the remaining one of {SpecAugment, class weighting}"),
    "S5": (T(1, 1, 1, 1, 1), os.path.join(M45, "results_M45_P3.json"),
           "+ amplitude normalisation — FULL PIPELINE"),
}

ORDERS = {
    "A": {"label": "SpecAugment before class weighting",
          "S3": (T(1, 1, 0, 1, 0), os.path.join(M45, "results_M45_A3.json"),
                 "+ SpecAugment"),
          "s3_component": "specaug", "s4_component": "classweight"},
    "B": {"label": "class weighting before SpecAugment",
          "S3": (T(1, 1, 1, 0, 0),
                 os.path.join(REPO, "Asif's", "M3_v2", "Results", "results_M3_v2.json"),
                 "+ class-weighted CE"),
          "s3_component": "classweight", "s4_component": "specaug"},
}


# ------------------------------------------------------------------ metrics
def official(cm):
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se = np.trace(cm[1:, 1:]) / abn if abn else float("nan")
    return float((se + sp) / 2), float(se), float(sp)


def read(path):
    """Recompute a rung's metrics from its committed raw confusion matrix, or return None."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    bm = d["best_metrics"]
    cm = bm.get("confusion_matrix_raw")
    if cm is None:
        return {"status": "unverifiable",
                "why": "the source run commits no confusion_matrix_raw"}
    sc, se, sp = official(cm)
    return {"status": "ok", "icbhi_score_official": round(sc, 4),
            "se": round(se, 4), "sp": round(sp, 4),
            "f1_macro": bm.get("f1_macro"), "accuracy": bm.get("accuracy"),
            "ci95": bm.get("icbhi_score_official_ci95"),
            "params": (d.get("efficiency") or {}).get("total_params"),
            "trainable_params": (d.get("efficiency") or {}).get("trainable_params"),
            "source": os.path.relpath(path, REPO).replace("\\", "/")}


def build_order(key):
    o = ORDERS[key]
    seq = ["S0", "S1", "S2", "S3", "S4", "S5"]
    rungs = []
    prev = None
    for r in seq:
        toggles, path, label = o["S3"] if r == "S3" else SHARED[r]
        if r == "S4":
            label = "+ " + dict(COMPONENTS)[o["s4_component"]].split(" (")[0]
        m = read(path)
        row = {"rung": r, "label": label, "toggles": toggles,
               "expected_path": os.path.relpath(path, REPO).replace("\\", "/")}
        if m is None:
            row.update(status="pending", note="run not on disk yet")
        else:
            row.update(m)
            if m["status"] == "ok" and prev is not None:
                row["delta_vs_previous"] = round(m["icbhi_score_official"] - prev, 4)
            if m["status"] == "ok":
                prev = m["icbhi_score_official"]
        rungs.append(row)
    return {"order": key, "label": o["label"], "rungs": rungs}


def leave_one_out():
    p = os.path.join(M45, "M45_ablation_table.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def seed_band():
    p = os.path.join(HERE, "M48_tier_A_table.json")
    if not os.path.exists(p):
        return None
    return (json.load(open(p, encoding="utf-8")) or {}).get("seed_band")


# ------------------------------------------------------------------ rendering
def tick(v):
    return "✓" if v else "✗"


def md_order(o, band):
    L = [f"### Order {o['order']} — {o['label']}", ""]
    head = "| Config | " + " | ".join(n for n, _ in COMPONENTS) + \
           " | ICBHI | Se | Sp | F1<sub>macro</sub> | Δ vs prev |"
    L += [head, "|---|" + "---|" * len(COMPONENTS) + "---:|---:|---:|---:|---:|"]
    done = [r for r in o["rungs"] if r.get("status") == "ok"]
    best = max((r["icbhi_score_official"] for r in done), default=None)
    for r in o["rungs"]:
        tg = " | ".join(tick(r["toggles"][k]) for k, _ in COMPONENTS)
        if r.get("status") != "ok":
            L.append(f"| **{r['rung']}** {r['label']} | {tg} | `pending` | — | — | — | — |")
            continue
        sc = r["icbhi_score_official"]
        cell = f"**{sc:.4f}**" if sc == best else f"{sc:.4f}"
        d = r.get("delta_vs_previous")
        ds = "—" if d is None else f"{d:+.4f}"
        if d is not None and band and abs(d) <= band["range"]:
            ds += " ⚠"
        L.append(f"| **{r['rung']}** {r['label']} | {tg} | {cell} | {r['se']:.4f} | "
                 f"{r['sp']:.4f} | {r['f1_macro']:.4f} | {ds} |")
    L.append("")
    return L


def render(doc):
    band = doc.get("seed_band")
    L = ["# M48 — cumulative ablation ladder", "",
         "Baseline first, one component added per row, on the **corrected official 60/40 "
         "patient-independent split** (2,636 test cycles, 47 patients). Every score is "
         "recomputed from the source run's committed raw confusion matrix. A rung whose run "
         "does not exist yet is `pending` and is **never** filled with a plausible value.", ""]
    if band:
        L += [f"⚠ marks a step smaller than the measured seed band "
              f"(range {band['range']:.4f} over seeds {band['seeds']}) — a step inside that "
              f"band is not distinguishable from run-to-run noise.", ""]
    else:
        L += ["_Seed band not yet measured; run the Kaggle notebook to annotate which steps "
              "clear run-to-run noise._", ""]

    for o in doc["orders"]:
        L += md_order(o, band)

    # the order effect
    oe = doc.get("order_effect")
    if oe:
        L += ["### The order effect, measured", "",
              "Both orderings above are built from the same runs and differ only in which of "
              "the two components is added at rung 4. What each is credited with therefore "
              "depends on where it sits, and this is that difference:", "",
              "| Component | credited in order A | credited in order B | shift |",
              "|---|---:|---:|---:|"]
        for k, v in oe.items():
            a = "pending" if v["order_A"] is None else f"{v['order_A']:+.4f}"
            b = "pending" if v["order_B"] is None else f"{v['order_B']:+.4f}"
            s = "—" if v["shift"] is None else f"{v['shift']:+.4f}"
            L.append(f"| {v['label']} | {a} | {b} | {s} |")
        L.append("")

    loo = doc.get("leave_one_out")
    if loo:
        L += ["### Leave-one-out companion (M45, already complete)", "",
              "Full pipeline minus one component. Reported beside the ladder because the two "
              "answer different questions — *worth given everything else* versus *worth given "
              "only what came before*.", "",
              "| Row | Change from full model | ICBHI | Δ vs A0 |", "|---|---|---:|---:|"]
        best = max(r["icbhi_score_official"] for r in loo["rows"])
        for r in loo["rows"]:
            sc = r["icbhi_score_official"]
            cell = f"**{sc:.4f}**" if sc == best else f"{sc:.4f}"
            d = r.get("delta_vs_A0")
            ds = "—" if d is None else f"{d:+.4f}"
            L.append(f"| `{r['row']}` | {r['variable_changed']} | {cell} | {ds} |")
        L += ["", "Rows P1–P3 **add** a stage the baseline does not have, so their sign reads "
                  "the other way: a positive delta is a recommendation to adopt.", ""]

    miss = [r["rung"] for o in doc["orders"] for r in o["rungs"]
            if r.get("status") == "pending"]
    if miss:
        L += ["### Still to run", "",
              f"`{'`, `'.join(sorted(set(miss)))}` — rows in `m48_gpu_rows.py`, produced by "
              "`M48_kaggle_tier_A.ipynb` alongside the seed band. About 18 minutes each; they "
              "share the spectrogram cache with every other row, so they add no cache time.", ""]
    return "\n".join(L) + "\n"


def latex(doc):
    o = doc["orders"][0]
    done = [r for r in o["rungs"] if r.get("status") == "ok"]
    best = max((r["icbhi_score_official"] for r in done), default=None)
    L = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Cumulative ablation. Each row adds one component to the row above, on the "
         r"corrected official 60/40 partition. \cmark/\xmark\ indicate module presence. "
         r"Rows marked \pending{} have not been run.}",
         r"\label{tab:cumulative}",
         r"\begin{tabular}{ll" + "c" * len(COMPONENTS) + r"rrrr}", r"\toprule",
         r"Config & Added & " + " & ".join(n.replace("_", r"\_") for n, _ in COMPONENTS)
         + r" & ICBHI & $Se$ & $Sp$ & $\Delta$ \\", r"\midrule"]
    for r in o["rungs"]:
        tg = " & ".join((r"\cmark" if r["toggles"][k] else r"\xmark")
                        for k, _ in COMPONENTS)
        lab = r["label"].replace("&", r"\&").replace("_", r"\_")
        if r.get("status") != "ok":
            L.append(f"{r['rung']} & {lab} & {tg} & "
                     r"\pending{} & --- & --- & --- \\")
            continue
        sc = r["icbhi_score_official"]
        cell = (r"\best{" + f"{sc:.4f}" + "}") if sc == best else f"{sc:.4f}"
        d = r.get("delta_vs_previous")
        ds = "---" if d is None else f"${d:+.4f}$"
        L.append(f"{r['rung']} & {lab} & {tg} & {cell} & {r['se']:.4f} & "
                 f"{r['sp']:.4f} & {ds} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ build
def build():
    orders = [build_order(k) for k in ("A", "B")]

    # What each of the two swapped components is credited with, in each ordering.
    oe = {}
    for comp in ("specaug", "classweight"):
        got = {}
        for o in orders:
            v = None
            for r in o["rungs"]:
                # the rung that turns this component on
                i = [x["rung"] for x in o["rungs"]].index(r["rung"])
                if i == 0:
                    continue
                prev = o["rungs"][i - 1]
                if r["toggles"][comp] and not prev["toggles"][comp]:
                    v = r.get("delta_vs_previous")
            got[o["order"]] = v
        shift = (None if got["A"] is None or got["B"] is None
                 else round(got["A"] - got["B"], 4))
        oe[comp] = {"label": dict(COMPONENTS)[comp], "order_A": got["A"],
                    "order_B": got["B"], "shift": shift}

    return {"table": "M48 cumulative ablation ladder",
            "split": "official_60_40_patient_independent_corrected",
            "n_test_cycles": 2636, "n_test_patients": 47,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "provenance_rule": ("every score recomputed from the source run's committed "
                                "confusion_matrix_raw; a missing run is 'pending', never "
                                "an estimate"),
            "components": [{"key": k, "name": n} for k, n in COMPONENTS],
            "orders": orders, "order_effect": oe,
            "seed_band": seed_band(), "leave_one_out": leave_one_out()}


def selftest():
    """The ladder must be monotone in toggles and consistent with the reused runs' configs."""
    ok = True
    for key in ("A", "B"):
        o = build_order(key)
        prev = None
        for r in o["rungs"]:
            t = r["toggles"]
            if prev is not None:
                added = [k for k in t if t[k] and not prev[k]]
                removed = [k for k in t if prev[k] and not t[k]]
                good = len(added) == 1 and not removed
                print(f"  order {key} {r['rung']}: adds {added or '-'}, "
                      f"removes {removed or '-'}  {'ok' if good else 'BAD'}")
                ok &= good
            prev = t
    # the two orders must end at the same full configuration
    a = build_order("A")["rungs"][-1]["toggles"]
    b = build_order("B")["rungs"][-1]["toggles"]
    print(f"  both orders end at the same full pipeline: {a == b}")
    ok &= a == b
    # and the reused sources must actually carry the toggles the ladder claims
    checks = [(os.path.join(M45, "results_M45_A3.json"), dict(class_weighted=False,
                                                             specaug=True)),
              (os.path.join(M45, "results_M45_P3.json"), dict(ampnorm=True)),
              (os.path.join(REPO, "Asif's", "M3_v2", "Results", "results_M3_v2.json"), None)]
    for path, want in checks:
        if not os.path.exists(path):
            print(f"  [skip] {os.path.basename(path)} not on disk")
            continue
        if want is None:
            print(f"  {os.path.basename(path)}: present (config not recorded; toggles "
                  f"asserted from M45's REUSED table)")
            continue
        cfg = json.load(open(path, encoding="utf-8")).get("config", {})
        good = all(cfg.get(k) == v for k, v in want.items())
        print(f"  {os.path.basename(path)}: config matches ladder claim {want}  "
              f"{'ok' if good else 'MISMATCH ' + str({k: cfg.get(k) for k in want})}")
        ok &= good
    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    doc = build()
    md = render(doc)
    print(md)
    json.dump(doc, open(os.path.join(HERE, "M48_cumulative_table.json"), "w"), indent=2)
    open(os.path.join(HERE, "M48_cumulative_table.md"), "w", encoding="utf-8").write(md)
    open(os.path.join(HERE, "M48_cumulative_table.tex"), "w",
         encoding="utf-8").write(latex(doc))
    print("  wrote M48_cumulative_table.{json,md,tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
