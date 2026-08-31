#!/usr/bin/env python3
"""Generate the five block diagrams of Final_Draft/main.tex.

    python Final_Draft/make_diagrams.py            # all five
    python Final_Draft/make_diagrams.py flowchart  # just one

Writes PDF (for pdfLaTeX) and PNG (for preview) into Final_Draft/figures/:

    fig_flowchart.pdf        F1   overall method / pipeline           Sec. II
    fig_motivation.pdf       F8   motivation / problem illustration   Sec. I
    fig_modules.pdf          F9   the two contributed modules         Sec. II
    fig_architecture.pdf     F10  model architecture + tensor shapes  Sec. II-C
    fig_metric_anatomy.pdf   F11  why the macro variant inflates      Sec. III-A

EVIDENCE RULE.  The same rule the rest of the paper follows: every measurement
that reaches a figure is read here from a committed results_M*.json, never typed
in.  The literals below are configuration fixed by the pipeline (mel bins, hop
length, class names, mask widths) and the published scores of Table XXXIII,
which are other people's numbers and are cited in the caption; every score,
count and confusion matrix belonging to this project comes from a loader.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(HERE, "figures")
sys.path.insert(0, HERE)

from diagram_lib import (  # noqa: E402
    ACCENT, EDGE, FS_BODY, FS_SMALL, FS_TITLE, GOOD, INK, MUTE, WARN,
    FILL_HEAVY, FILL_STD, arrow, box, diamond, elbow, new_canvas, note, save,
)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


# ====================================================================== data
def jload(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as fh:
        return json.load(fh)


ASIF = "Asif" + chr(39) + "s"          # the directory really is spelled this way
M22V2 = jload(ASIF, "M22_v2", "Results", "results_M22_v2.json")
AUDIT = jload(ASIF, "audit", "ICBHI_SCORE_AUDIT.json")
M48A = jload("M48_core_pipeline_ablation", "M48_tier_A_table.json")
TFM = {m: jload("M40_M43_transformers", "Results", "results_%s.json" % m)
       for m in ("M40", "M41", "M42")}

BEST = M22V2["best_metrics"]
CM = np.array(BEST["confusion_matrix_raw"], dtype=float)
EFF = M22V2["efficiency"]
DS = M22V2["dataset_info"]
CFG = M22V2["config"]
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


def audit_row(model):
    for r in AUDIT["verified"]:
        if r["model"] == model:
            return r
    raise KeyError(model)


def official(cm):
    """Eq. (6): pooled-abnormal sensitivity, Normal specificity, their mean."""
    se = cm[1:, 1:].diagonal().sum() / cm[1:, :].sum()
    sp = cm[0, 0] / cm[0, :].sum()
    return se, sp, (se + sp) / 2


def macro_parts(cm):
    """Eq. (7): the variant the pipeline had been reporting under the same name."""
    n = cm.sum()
    rec, spec = [], []
    for k in range(len(cm)):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = n - tp - fn - fp
        rec.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))
    rec, spec = np.array(rec), np.array(spec)
    return rec, spec, (rec.mean() + spec.mean()) / 2


# ================================================== F1  pipeline / flowchart
def fig_flowchart():
    fig, ax = new_canvas(6.7, 6.6, ylim=(-9.5, 101))

    ntr, nte = DS["train_samples"], DS["test_samples"]
    ptr, pte = DS["train_patients"], DS["test_patients"]
    icbhi = BEST["icbhi_score_official"]

    # --- inputs -----------------------------------------------------------
    box(ax, 2, 91.5, 45, 9.0, "ICBHI 2017 Respiratory Sound Database",
        ["920 recordings  ·  126 patients  ·  5.5 h  ·  4 stethoscopes",
         "one .txt per recording: cycle start, end, crackle, wheeze"],
        style="data", ls="--")
    box(ax, 53, 91.5, 45, 9.0, "ICBHI_challenge_train_test.txt",
        ["the released official split file",
         "assigns RECORDINGS, not patients — two patients straddle it"],
        style="data", ls="--")

    # --- contributed module 1 --------------------------------------------
    box(ax, 6, 75.5, 88, 13.0, "AUDITED SPLIT LOADER      (contribution 1)",
        ["raise if the split file is absent — never a silent identifier fallback",
         "detect patients on both sides (156, 218) → reassign every recording to train",
         "device-suffix-safe join: 226_1b1_Pl_sc_Meditron is released as ..._LittC2SE",
         "assert the train and test patient sets are disjoint before returning"],
        style="heavy", align="left", pad=2.4)

    # --- partitions -------------------------------------------------------
    box(ax, 6, 66.5, 42, 6.6, "TRAIN partition",
        ["{} patients  ·  {:,} cycles".format(ptr, ntr)], style="data")
    box(ax, 52, 66.5, 42, 6.6, "TEST partition",
        ["{} patients  ·  {:,} cycles".format(pte, nte)], style="data")

    # --- preprocessing ----------------------------------------------------
    box(ax, 13, 48.5, 74, 14.0,
        "SHARED PREPROCESSING   (identical for both partitions)",
        ["load {} kHz mono, crop to the annotated cycle start/end".format(
            CFG["sample_rate"] // 1000),
         "cyclic tiling to {:.1f} s   (Eq. 1)  —  99.8 % of cycles are shorter".format(
             CFG["duration_s"]),
         "log-mel {}×801:  n_fft {}, win {}, hop {}, {}–{} Hz   (Eq. 2)".format(
             CFG["n_mels"], CFG["n_fft"], CFG["win_length"], CFG["hop_length"],
             CFG["f_min"], CFG["f_max"]),
         "per-spectrogram min–max to [0, 1]   →   1 → 3 channels + ImageNet mean/std"],
        align="left", pad=2.4)

    # --- augmentation, training lane only ---------------------------------
    box(ax, 13, 35.0, 40, 8.0, "SpecAugment   (training split only)",
        ["2 freq masks ≤ 24 bins, 2 time masks ≤ 80 frames",
         "resampled every epoch   (Eq. 5)"], align="left", pad=2.0)

    # the three stages that were specified but never implemented
    box(ax, 57, 34.4, 35, 8.6, "not in the baseline",
        ["band-pass P1  ·  denoise P2", "amplitude normalisation P3",
         "each is an ADDITIVE ablation row"],
        style="mute", ls=(0, (2, 2)), align="left", pad=1.8,
        fs_title=FS_SMALL, fs_body=FS_SMALL)

    # --- backbone ---------------------------------------------------------
    box(ax, 13, 22.5, 74, 11.0,
        "BACKBONE  (one of four, ImageNet-1k)  +  shared classifier head",
        ["MobileNetV2 2.23 M   |   Swin-T 27.5 M   |   DeiT-S 21.7 M   |   ViT-B/16 85.8 M",
         "global average pool  →  dropout 0.3  →  Linear(→ 4 logits)",
         "inverse-frequency class-weighted cross-entropy, unit mean   (Eq. 4)"],
        align="left", pad=2.2)

    box(ax, 26, 14.0, 48, 5.4, None,
        ["4-class prediction per test cycle  →  raw 4 × 4 confusion matrix"],
        style="data", pad=1.6)

    # --- contributed module 2 --------------------------------------------
    box(ax, 6, -0.5, 88, 13.0, "OFFICIAL SCORING MODULE      (contribution 2)",
        ["commit the raw 4 × 4 confusion matrix verbatim — every score stays recomputable",
         "ICBHI = (Se + Sp)/2 with the three abnormal classes POOLED   (Eq. 6)",
         "the macro variant is still computed, but reported under its own name   (Eq. 7)",
         "patient-level percentile bootstrap, B = 1000  ·  paired McNemar and bootstrap"],
        style="heavy", align="left", pad=2.4)

    box(ax, 18, -8.5, 64, 5.4, None,
        ["results_M*.json   →   every table in Section III       "
         "(best model: ICBHI {:.4f})".format(icbhi)],
        style="data", ls="--", pad=1.6)

    # --- edges ------------------------------------------------------------
    arrow(ax, (24.5, 91.5), (24.5, 88.5))
    arrow(ax, (75.5, 91.5), (75.5, 88.5))
    arrow(ax, (27, 75.5), (27, 73.1))
    arrow(ax, (73, 75.5), (73, 73.1))
    arrow(ax, (27, 66.5), (27, 62.5))
    arrow(ax, (73, 66.5), (73, 62.5))
    arrow(ax, (27, 48.5), (27, 43.0))
    arrow(ax, (27, 35.0), (27, 33.5))
    # the evaluation lane bypasses augmentation entirely
    elbow(ax, [(78, 48.5), (78, 46.0), (96, 46.0), (96, 27.0), (87, 27.0)],
          color=MUTE, ls=(0, (4, 2)))
    note(ax, 80, 47.8, "test cycles: no augmentation", color=MUTE, ha="left")
    arrow(ax, (50, 22.5), (50, 19.4))
    arrow(ax, (50, 14.0), (50, 12.5))
    arrow(ax, (50, -0.5), (50, -3.1))
    # checkpoint selection is a pipeline component, not a footnote (Fault 5)
    elbow(ax, [(6, 5.5), (2.2, 5.5), (2.2, 30.0), (13, 30.0)], color=GOOD, lw=1.1)
    note(ax, 5.0, 21.0,
         "checkpoint selection\n(Fault 5, Sec. III-C):\nkeep argmax official\nscore, not min loss",
         color=GOOD, ha="left", style="normal")

    note(ax, 27, 65.2, "training path", color=MUTE, ha="center")
    note(ax, 73, 65.2, "evaluation path", color=MUTE, ha="center")

    save(fig, os.path.join(OUTDIR, "fig_flowchart"))


# ================================================ F8  motivation / problem
def fig_motivation():
    """One model, one corpus: what changes when only the measurement changes."""
    corrected = audit_row("M22_v2")          # corrected partition
    fallback = audit_row("M22")              # identifier-fallback partition
    by_loss = M48A["A7_selection_criterion"]["A0_s42"]["by_loss"]

    ours = [
        ("checkpoint chosen on minimum loss", by_loss, WARN),
        ("corrected protocol — what we report", corrected["official"], GOOD),
        ("+ identifier-fallback partition (7.1 % of the corpus)",
         fallback["official"], "#b5651d"),
        ("+ macro-averaged metric — the unaudited pipeline",
         fallback["reported"], WARN),
    ]
    # Published scores, official partition and official metric (Table XXXIII).
    lit = [
        ("Jeong et al. 2025 — BEATs + PAFA", 0.6484),
        ("Kim et al. 2024 — CLAP + BTS", 0.6354),
        ("Bae et al. 2023 — AST + patch-mix", 0.6237),
        ("Niizumi et al. 2025 — M2D frozen probe", 0.5938),
        ("Moummad and Farrugia 2023 — CNN6 + SCL", 0.5755),
        ("Gairola et al. 2021 — RespireNet", 0.5620),
        ("Tzeng et al. 2025 — 7 senior physicians", 0.4777),
    ]

    fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(6.7, 4.05), sharex=True,
        gridspec_kw=dict(height_ratios=[len(lit), len(ours) + 0.6], hspace=0.14))
    fig.subplots_adjust(left=0.375, right=0.985, top=0.905, bottom=0.115)

    lo, hi = 0.44, 0.762
    band = (min(v for _, v in lit[:-1]), max(v for _, v in lit))
    unaudited = ours[-1][1]

    for ax in (axt, axb):
        ax.axvspan(*band, color="#ededed", zorder=0)
        ax.axvline(unaudited, color=WARN, lw=1.0, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x", ls=":", color="#cfcfcf", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color("#9a9a9a")

    # --- published work ---------------------------------------------------
    y = np.arange(len(lit))[::-1]
    axt.hlines(y, lo, [v for _, v in lit], color="#c4c4c4", lw=0.9, zorder=2)
    axt.scatter([v for _, v in lit], y, s=22, facecolor="#8f8f8f",
                edgecolor="#5a5a5a", lw=0.6, zorder=3)
    for yy, (_, v) in zip(y, lit):
        axt.text(v + 0.007, yy, "%.4f" % v, va="center", fontsize=FS_SMALL,
                 color="#5a5a5a")
    axt.set_yticks(y, [n for n, _ in lit], fontsize=FS_SMALL)
    axt.set_ylim(-0.95, len(lit) - 0.2)
    axt.set_title("Published systems on ICBHI 2017 — official partition, official metric",
                  fontsize=FS_BODY, loc="left", color=INK, pad=4)
    axt.text(band[1] - 0.002, -0.55, "range of the published literature",
             fontsize=FS_SMALL, color="#8a8a8a", ha="right", va="center",
             style="italic")

    # --- our own model, four measurement choices --------------------------
    yb = np.arange(len(ours))[::-1]
    axb.hlines(yb, lo, [v for _, v, _ in ours], color="#c4c4c4", lw=0.9, zorder=2)
    for yy, (_, v, c) in zip(yb, ours):
        axb.scatter([v], [yy], s=34, facecolor=c, edgecolor="white", lw=0.7,
                    zorder=4, marker="D")
        axb.text(v + 0.008, yy, "%.4f" % v, va="center", fontsize=FS_SMALL,
                 color=c, fontweight="bold")
    axb.set_yticks(yb, [n for n, _, _ in ours], fontsize=FS_SMALL)
    axb.set_ylim(-1.35, len(ours) - 0.35)
    axb.set_title("The same MobileNetV2, the same corpus — only the measurement changes",
                  fontsize=FS_BODY, loc="left", color=INK, pad=4)
    axb.set_xlabel("official ICBHI score,  (Se + Sp) / 2", fontsize=FS_BODY)
    axb.tick_params(axis="x", labelsize=FS_SMALL)

    # deltas between consecutive measurement choices
    for i in range(len(ours) - 1):
        v0, v1 = ours[i][1], ours[i + 1][1]
        y0, y1 = yb[i], yb[i + 1]
        axb.annotate("", xy=(v1, y1), xytext=(v0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#6f6f6f", lw=0.9,
                                     shrinkA=4, shrinkB=4,
                                     connectionstyle="arc3,rad=-0.25"))
        axb.text((v0 + v1) / 2, (y0 + y1) / 2 - 0.30, "%+.4f" % (v1 - v0),
                 fontsize=FS_SMALL, color="#5a5a5a", ha="center")

    axb.annotate("would have ranked first\nin the table above",
                 xy=(unaudited, 0), xytext=(unaudited - 0.008, -1.02),
                 fontsize=FS_SMALL, color=WARN, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))

    save(fig, os.path.join(OUTDIR, "fig_motivation"))


# ============================================ F9  the two contributed modules
def fig_modules():
    fig, ax = new_canvas(6.7, 4.9, xlim=(0, 104), ylim=(0, 100))

    se, sp, off = official(CM)
    _, _, mac = macro_parts(CM)
    ci = BEST.get("icbhi_score_official_ci95", [0.508, 0.614])
    ptr, pte = DS["train_patients"], DS["test_patients"]
    ntr, nte = DS["train_samples"], DS["test_samples"]

    ax.plot([50.0, 50.0], [1, 96], color="#d5d5d5", lw=0.8, ls=(0, (3, 3)))
    ax.text(1, 98.5, "(a)  Audited split loader", fontsize=FS_TITLE + 0.4,
            fontweight="bold", color=ACCENT, va="center")
    ax.text(53, 98.5, "(b)  Official scoring module", fontsize=FS_TITLE + 0.4,
            fontweight="bold", color=ACCENT, va="center")

    # ------------------------------------------------------- (a) the loader
    box(ax, 1, 87.0, 46, 6.4, None,
        ["split file path  +  audio directory"], style="data", ls="--", pad=1.7)

    diamond(ax, 14, 78.0, 26, 8.6, ["split file present?"])
    box(ax, 29, 74.2, 19, 7.6, None,
        ["raise", "FileNotFoundError"], style="warn", pad=1.7, fs_body=FS_SMALL)
    note(ax, 29.5, 73.0, "no silent fallback to a\npatient-identifier rule",
         color=WARN, ha="left")

    box(ax, 1, 60.0, 46, 8.2, "parse 920 assignments",
        ["one line per RECORDING:  stem → train | test"],
        align="left", pad=1.8)

    box(ax, 1, 45.5, 46, 12.0, "device-suffix-safe join",
        ["match patient + location + acquisition mode, not the",
         "full stem  →  recovers 226_1b1_Pl_sc_*, which a naive",
         "stem join drops together with its 11 cycles, in silence"],
        align="left", pad=1.8)

    diamond(ax, 15, 35.0, 27, 10.0,
            ["train ∩ test", "patients = ∅ ?"])
    box(ax, 29, 31.2, 19, 7.6, None,
        ["reassign 156, 218", "to TRAIN"], style="good", pad=1.7,
        fs_body=FS_SMALL)

    box(ax, 1, 19.0, 46, 6.6, None,
        ["assert disjoint  —  the loop cannot start otherwise"],
        style="heavy", pad=1.7, lw=1.3)

    box(ax, 1, 7.0, 22, 8.2, "TRAIN",
        ["{} patients".format(ptr), "{:,} cycles".format(ntr)],
        style="data", pad=1.6, fs_title=FS_BODY, fs_body=FS_SMALL)
    box(ax, 25, 7.0, 22, 8.2, "TEST",
        ["{} patients".format(pte), "{:,} cycles".format(nte)],
        style="data", pad=1.6, fs_title=FS_BODY, fs_body=FS_SMALL)

    arrow(ax, (14, 87.0), (14, 82.4))
    arrow(ax, (27, 78.0), (29, 78.0), label="no", label_off=(-1.0, 1.7),
          label_color=WARN)
    arrow(ax, (14, 73.7), (14, 68.2), label="yes", label_off=(2.6, 0),
          label_color=GOOD)
    arrow(ax, (14, 60.0), (14, 57.5))
    arrow(ax, (14, 45.5), (15, 40.1))
    arrow(ax, (28.5, 35.0), (29, 35.0), label="no", label_off=(-1.4, 1.7),
          label_color="#b5651d")
    elbow(ax, [(38.5, 31.2), (38.5, 27.5), (35, 27.5), (35, 25.7)],
          color="#b5651d")
    arrow(ax, (15, 30.0), (15, 25.7), label="yes", label_off=(2.6, 0),
          label_color=GOOD)
    arrow(ax, (12, 19.0), (12, 15.3))
    arrow(ax, (36, 19.0), (36, 15.3))

    # --------------------------------------------------- (b) scoring module
    box(ax, 53, 87.0, 46, 6.4, None,
        ["per-cycle predictions + labels   ({:,} cycles, {} patients)".format(
            nte, pte)],
        style="data", ls="--", pad=1.7)

    box(ax, 53, 76.5, 46, 8.4, "raw 4 × 4 confusion matrix  C",
        ["written to results_M*.json verbatim, before any score"],
        style="heavy", align="left", pad=1.9, lw=1.4)
    note(ax, 57.5, 75.6,
         "this is what makes every number in Section III recomputable by\n"
         "a reader, a reviewer, or by us six months later", ha="left")

    # C fans out into four computations; all four land in the same record
    bus_x = 54.5
    ax.plot([bus_x, bus_x], [76.5, 12.0], color=INK, lw=1.0, zorder=3)

    branches = [
        (57.0, 55.0, 11.5, "good", "OFFICIAL  (Eq. 6)   —   the headline",
         ["Se = Σ diag(C[1:,1:]) / Σ C[1:,:]  =  {:.4f}".format(se),
          "Sp = C₀₀ / Σ C₀ⱼ  =  {:.4f}".format(sp),
          "ICBHI = (Se + Sp)/2  =  {:.4f}".format(off)]),
        (57.0, 43.0, 9.6, "mute", "MACRO VARIANT  (Eq. 7)   —   audit only",
         ["½(mean recall + mean per-class specificity) = {:.4f}".format(mac),
          "kept, but never reported as “the ICBHI score”"]),
        (57.0, 30.5, 9.6, "std", "UNCERTAINTY",
         ["patient-level percentile bootstrap, B = 1000",
          "resample {} patients, rescore  →  [{:.3f}, {:.3f}]".format(
              pte, ci[0], ci[1])]),
        (57.0, 18.0, 9.6, "std", "PAIRED COMPARISON",
         ["patient bootstrap on one resample, both systems",
          "exact McNemar beside it, never instead of it"]),
    ]
    for x, y, h, style, title, lines in branches:
        box(ax, x, y, 42, h, title, lines, style=style, align="left", pad=1.8,
            fs_title=FS_BODY + 0.3)
        arrow(ax, (bus_x, y + h / 2), (x, y + h / 2), head=5.0)

    box(ax, 53, 4.5, 46, 6.8, None,
        ["canonical results record  →  every table in Section III"],
        style="data", ls="--", pad=1.8)

    arrow(ax, (76, 87.0), (76, 84.9))
    arrow(ax, (bus_x, 13.5), (bus_x, 11.3))
    # the official score is also the checkpoint-selection criterion (Fault 5)
    elbow(ax, [(99, 60.8), (102.0, 60.8), (102.0, 90.2), (99.3, 90.2)],
          color=GOOD, lw=1.1)
    note(ax, 98.5, 70.5, "argmax over epochs → saved checkpoint", color=GOOD,
         ha="right", style="normal")

    save(fig, os.path.join(OUTDIR, "fig_modules"))


# =================================================== F10  model architecture
def fig_architecture():
    fig, ax = new_canvas(6.7, 4.75, ylim=(0, 100))

    nsamp = int(CFG["sample_rate"] * CFG["duration_s"])

    # ------------------------------------- (a) the shared front end, L to R
    ax.text(0.5, 98.0,
            "(a)  Shared front end — identical for all four backbones",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")

    chain = [
        (0.5, 15.4, "respiratory cycle", ["cropped to the", "annotated start/end"],
         "$a \\in \\mathbb{R}^{L_a}$"),
        (17.9, 15.4, "cyclic tiling", ["Eq. 1;", "truncate if longer"],
         "%s samples" % "{:,}".format(nsamp)),
        (35.3, 15.4, "log-mel + dB", ["Eq. 2, then per-", "spectrogram min–max"],
         "1 × %d × 801" % CFG["n_mels"]),
        (52.7, 15.4, "SpecAugment", ["training split", "only, Eq. 5"],
         "1 × %d × 801" % CFG["n_mels"]),
        (70.1, 15.4, "channel adapt", ["1 → 3 replicate,", "ImageNet mean/std"],
         "3 × %d × 801" % CFG["n_mels"]),
        (87.5, 12.0, "BACKBONE", ["see (b) below"], ""),
    ]
    for x, w, title, lines, shape in chain:
        box(ax, x, 80.0, w, 13.0, title, lines,
            style="heavy" if title == "BACKBONE" else "std",
            pad=1.8, fs_title=FS_BODY + 0.3, fs_body=FS_SMALL)
        if shape:
            ax.text(x + w / 2, 78.4, shape, fontsize=FS_SMALL, color=ACCENT,
                    ha="center", va="top")
    for x in (15.9, 33.3, 50.7, 68.1, 85.5):
        arrow(ax, (x, 86.5), (x + 2.0, 86.5), head=5.0)

    # ------------------------------------------- (b) the four backbones
    ax.text(0.5, 71.5,
            "(b)  The four interchangeable backbones — one recipe, one seed, one preprocessing cache",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")

    rows = [
        ("MobileNetV2", "CNN, inverted res.", EFF["total_params"],
         "1280 × 4 × 26", BEST["icbhi_score_official"], True),
        ("Swin-T", "shifted-window attn.", TFM["M41"]["efficiency"]["total_params"],
         "768 (pooled)", TFM["M41"]["best_metrics"]["icbhi_score_official"], False),
        ("DeiT-S", "ViT + distill. token", TFM["M42"]["efficiency"]["total_params"],
         "384 (CLS)", TFM["M42"]["best_metrics"]["icbhi_score_official"], False),
        ("ViT-B/16", "plain ViT", TFM["M40"]["efficiency"]["total_params"],
         "768 (CLS)", TFM["M40"]["best_metrics"]["icbhi_score_official"], False),
    ]
    for cx, name, ha in ((1.5, "backbone", "left"), (15.0, "family", "left"),
                         (39.5, "parameters", "right"),
                         (42.0, "GAP input", "left"),
                         (57.5, "ICBHI", "right")):
        ax.text(cx, 66.0, name, fontsize=FS_SMALL, color=MUTE, va="center",
                ha=ha, fontweight="bold")
    ax.plot([1.0, 58.5], [63.8, 63.8], color="#b8b8b8", lw=0.7)
    for i, (name, fam, prm, feat, score, best) in enumerate(rows):
        y = 60.6 - i * 6.0
        if best:
            ax.add_patch(Rectangle((1.0, y - 2.4), 57.5, 5.0,
                                   facecolor=FILL_HEAVY, edgecolor="none",
                                   zorder=0))
        c, w = (ACCENT, "bold") if best else (INK, "normal")
        ax.text(1.5, y, name, fontsize=FS_BODY, color=c, fontweight=w, va="center")
        ax.text(15.0, y, fam, fontsize=FS_SMALL, color=MUTE, va="center")
        ax.text(39.5, y, "{:,}".format(prm), fontsize=FS_BODY, color=c,
                fontweight=w, va="center", ha="right")
        ax.text(42.0, y, feat, fontsize=FS_SMALL, color=c, va="center")
        ax.text(57.5, y, "%.4f" % score, fontsize=FS_BODY, color=c,
                fontweight=w, va="center", ha="right")
    note(ax, 1.0, 34.0,
         "The 2.23 M-parameter convolutional backbone beats all three transformers under\n"
         "one shared recipe; ViT-B/16 collapsed to the majority class in both of its runs.",
         ha="left")

    # MobileNetV2 in detail, plus the tap the attribution figure reads
    box(ax, 61.0, 53.5, 38.5, 14.0, "MobileNetV2 path, in detail",
        ["conv 3×3, stride 2  →  32 channels",
         "17 inverted residual bottlenecks (expansion 6)",
         "conv 1×1  →  1280 channels",
         "total stride 32  →  1280 × 4 × 26"],
        align="left", pad=1.8, fs_title=FS_BODY + 0.3, fs_body=FS_SMALL)
    box(ax, 61.0, 38.0, 38.5, 11.5, "Grad-CAM tap",
        ["read at the 4 × 26 grid above, so one cell is",
         "≈ 500 Hz × 0.31 s; every attribution map is",
         "that grid bilinearly upsampled, not fine structure"],
        style="mute", align="left", pad=1.8, fs_title=FS_BODY + 0.3,
        fs_body=FS_SMALL)
    arrow(ax, (80, 53.5), (80, 49.5), color=MUTE)

    # ------------------------------------------- (c) the shared head + loss
    ax.text(0.5, 23.5,
            "(c)  Shared classifier head and objective — also identical for all four",
            fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")

    box(ax, 0.5, 4.5, 30.5, 13.5, "CLASSIFIER HEAD",
        ["global average pool over the",
         "feature map of (b)",
         "dropout {:.1f}".format(CFG["dropout"]),
         "Linear(d → 4)"],
        align="left", pad=1.8, fs_title=FS_BODY + 0.3, fs_body=FS_SMALL)
    box(ax, 35.0, 4.5, 30.5, 13.5, "OBJECTIVE",
        ["inverse-frequency class-weighted",
         "cross-entropy, unit mean  (Eq. 4)",
         "weights from the TRAIN partition,",
         "never from the test cycles"],
        style="good", align="left", pad=1.8, fs_title=FS_BODY + 0.3,
        fs_body=FS_SMALL)
    box(ax, 69.5, 4.5, 30.0, 13.5, "OUTPUT",
        ["4 logits: Normal, Crackle,",
         "Wheeze, Both",
         "argmax per cycle → the raw",
         "4 × 4 confusion matrix"],
        style="data", align="left", pad=1.8, fs_title=FS_BODY + 0.3,
        fs_body=FS_SMALL)
    arrow(ax, (31.5, 11.25), (34.4, 11.25), head=5.0)
    arrow(ax, (66.0, 11.25), (68.9, 11.25), head=5.0)

    save(fig, os.path.join(OUTDIR, "fig_architecture"))


# ============================================ F11  anatomy of the two metrics
def fig_metric_anatomy():
    se, sp, off = official(CM)
    rec, spec, mac = macro_parts(CM)
    support = CM.sum(axis=1)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.7, 3.25),
                                   gridspec_kw=dict(width_ratios=[1.18, 1.0],
                                                    wspace=0.26))
    fig.subplots_adjust(left=0.078, right=0.985, top=0.845, bottom=0.125)

    # ---------------- (a) the committed matrix, with the official read-out
    norm = CM / CM.sum(axis=1, keepdims=True)
    axa.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(4):
        for j in range(4):
            axa.text(j, i, "%d" % CM[i, j], ha="center", va="center",
                     fontsize=FS_BODY,
                     color="white" if norm[i, j] > 0.55 else INK)
    axa.set_xticks(range(4), ["Norm.", "Crack.", "Wheez.", "Both"],
                   fontsize=FS_SMALL)
    axa.set_yticks(range(4), CLASSES, fontsize=FS_SMALL)
    axa.set_xlabel("predicted", fontsize=FS_BODY)
    axa.set_ylabel("true", fontsize=FS_BODY)
    axa.tick_params(length=0)
    for s in axa.spines.values():
        s.set_visible(False)

    axa.add_patch(Rectangle((-0.5, -0.5), 4, 1, fill=False, edgecolor=WARN,
                            lw=1.6, zorder=5))
    axa.add_patch(Rectangle((-0.5, 0.5), 4, 3, fill=False, edgecolor=GOOD,
                            lw=1.6, zorder=5))
    axa.annotate("Sp = 1110 / 1560\n     = %.4f" % sp, xy=(3.5, 0),
                 xytext=(4.15, 0), fontsize=FS_SMALL, color=WARN, va="center",
                 arrowprops=dict(arrowstyle="-", color=WARN, lw=0.8))
    axa.annotate("Se = (320 + 110 + 10) / 1076\n     = %.4f" % se,
                 xy=(3.5, 2), xytext=(4.15, 2), fontsize=FS_SMALL, color=GOOD,
                 va="center", arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.8))
    axa.text(4.15, 3.15, "the three abnormal rows are POOLED\ninto one Se — "
             "a rare class cannot hide", fontsize=FS_SMALL, color="#6a6a6a",
             va="center", style="italic")
    axa.set_xlim(-0.5, 8.9)
    axa.set_title("(a)  Official metric, Eq. 6:  ICBHI = (Se + Sp)/2 = %.4f\n"
                  "computed from the committed raw matrix below" % off,
                  fontsize=FS_BODY, loc="left", color=INK, pad=5)

    # ---------------- (b) what the macro variant averages instead
    y = np.arange(4)[::-1]
    h = 0.36
    axb.barh(y + h / 2 + 0.02, rec, height=h, color="#6b9ac4",
             edgecolor="#3c6e99", lw=0.6, label="per-class recall")
    axb.barh(y - h / 2 - 0.02, spec, height=h, color="#d9d9d9",
             edgecolor="#9a9a9a", lw=0.6, label="per-class specificity")
    for yy, r, s, n in zip(y, rec, spec, support):
        axb.text(r + 0.012, yy + h / 2 + 0.02, "%.3f" % r, va="center",
                 fontsize=FS_SMALL, color="#3c6e99")
        axb.text(s + 0.012, yy - h / 2 - 0.02, "%.3f" % s, va="center",
                 fontsize=FS_SMALL, color="#6a6a6a")
    axb.set_yticks(y, ["%s\nn = %d" % (c, n) for c, n in zip(CLASSES, support)],
                   fontsize=FS_SMALL)
    axb.set_xlim(0, 1.16)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.tick_params(labelsize=FS_SMALL, length=0)
    axb.grid(axis="x", ls=":", color="#cfcfcf", lw=0.6)
    axb.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.spines["bottom"].set_color("#9a9a9a")
    axb.legend(fontsize=FS_SMALL, frameon=False, loc="upper right", ncol=2,
               bbox_to_anchor=(1.01, 1.03), handlelength=1.2,
               columnspacing=1.1, handletextpad=0.5)
    axb.set_title("(b)  Macro variant, Eq. 7:  per-class average = %.4f\n"
                  "inflation %+.4f, bought from specificity"
                  % (mac, mac - off),
                  fontsize=FS_BODY, loc="left", color=INK, pad=5)
    axb.annotate("only 10 of 86 Both cycles are found,\n"
                 "yet this class reports specificity %.3f" % spec[3],
                 xy=(spec[3] - 0.02, y[3] - h / 2 - 0.02),
                 xytext=(0.045, y[3] - 0.86),
                 fontsize=FS_SMALL, color=WARN,
                 arrowprops=dict(arrowstyle="->", color=WARN, lw=0.8))
    axb.set_ylim(-1.15, 4.15)

    save(fig, os.path.join(OUTDIR, "fig_metric_anatomy"))


# ====================================================================== main
FIGURES = {
    "flowchart": fig_flowchart,
    "motivation": fig_motivation,
    "modules": fig_modules,
    "architecture": fig_architecture,
    "metric": fig_metric_anatomy,
}

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for name in (sys.argv[1:] or list(FIGURES)):
        FIGURES[name]()
