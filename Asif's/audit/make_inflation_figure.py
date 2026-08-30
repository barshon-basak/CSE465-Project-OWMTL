#!/usr/bin/env python3
"""Redraw the paper's metric-inflation figure from ICBHI_SCORE_AUDIT.json.

The figure and Table II must show the same runs. The committed figure was drawn when
the audit verified fifteen; it now verifies twenty, after the corrected re-runs landed
and after the audit stopped counting archived duplicates as separate models. Drawing it
from the audit's own JSON is what keeps the two in step: there is no second list to
drift from.

    python "Asif's/audit/make_inflation_figure.py"
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "ICBHI_SCORE_AUDIT.json")
OUT = os.path.join(REPO, "DRAFT_PAPER", "figures", "fig_metric_inflation.png")
BAND = (0.60, 0.65)   # published state of the art on this benchmark


def main():
    rows = sorted(json.load(open(SRC, encoding="utf-8"))["verified"],
                  key=lambda r: r["official"], reverse=True)
    labels = [r["model"] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.2))
    ax.axvspan(*BAND, color="0.88", zorder=0, label="published SOTA band")
    for i, r in enumerate(rows):
        ax.plot([r["official"], r["reported"]], [i, i], c="0.7", lw=1.2, zorder=1)
    ax.scatter([r["official"] for r in rows], y, s=26, c="#1f77b4", zorder=2,
               label="official $(Se+Sp)/2$")
    ax.scatter([r["reported"] for r in rows], y, s=26, c="#d62728", marker="D",
               zorder=2, label="reported (macro variant)")

    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ICBHI score")
    ax.set_xlim(0.25, 0.85)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", ls=":", c="0.85")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    plt.close(fig)

    above_macro = sum(r["reported"] >= BAND[1] for r in rows)
    above_off = sum(r["official"] >= BAND[1] for r in rows)
    mean_infl = sum(abs(r["delta"]) for r in rows) / len(rows)
    print(f"[saved] {OUT}")
    print(f"  {len(rows)} runs | mean inflation {mean_infl:.4f} | "
          f"max {max(abs(r['delta']) for r in rows):.4f} | "
          f"min {min(abs(r['delta']) for r in rows):.4f}")
    print(f"  at or above {BAND[1]}: {above_macro} under the macro variant, "
          f"{above_off} under the official metric")
    print("  -> these must match the numbers in Table II and its figure caption")


if __name__ == "__main__":
    main()
