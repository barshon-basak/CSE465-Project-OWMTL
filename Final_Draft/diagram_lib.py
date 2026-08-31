#!/usr/bin/env python3
"""Shared drawing primitives for the block diagrams of Final_Draft/main.tex.

Kept separate from make_diagrams.py so the layout code below reads as layout and
not as matplotlib boilerplate. Nothing here holds a number; every quantity that
reaches a figure is loaded from a committed results_M*.json by make_diagrams.py,
which is the same rule FIGURES_SPEC.md imposes on the four existing generators.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

# ---------------------------------------------------------------- palette
# Greyscale-safe: the four fills below are separated in luminance as well as in
# hue, because the report may be printed in black and white (FIGURES_SPEC F1).
INK        = "#1b1b1b"   # default text / thin rules
EDGE       = "#4a4a4a"   # ordinary block border
FILL_STD   = "#f4f4f4"   # ordinary processing block
FILL_DATA  = "#ffffff"   # data artefact (dashed border)
ACCENT     = "#12395c"   # the two contributed modules
FILL_HEAVY = "#d7e4f0"
WARN       = "#8f2a2a"   # fault / abort path
FILL_WARN  = "#f6e2e2"
GOOD       = "#2c6444"
FILL_GOOD  = "#e2efe7"
MUTE       = "#8a8a8a"

FS_TITLE = 7.6
FS_BODY  = 6.7
FS_SMALL = 6.0


def new_canvas(w_in, h_in, xlim=(0, 100), ylim=(0, 100)):
    """A blank axes in an abstract 0-100 layout space."""
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), *, style="std", ls="-", lw=None,
        fs_title=FS_TITLE, fs_body=FS_BODY, align="center", title_color=None,
        pad=1.6, line_gap=None, zorder=2):
    """One block. `y` is the BOTTOM edge; `lines` are stacked under `title`."""
    fill, edge, tc = {
        "std":   (FILL_STD,   EDGE,   INK),
        "heavy": (FILL_HEAVY, ACCENT, ACCENT),
        "data":  (FILL_DATA,  EDGE,   INK),
        "warn":  (FILL_WARN,  WARN,   WARN),
        "good":  (FILL_GOOD,  GOOD,   GOOD),
        "mute":  ("#fafafa",  MUTE,   MUTE),
    }[style]
    if lw is None:
        lw = 1.9 if style == "heavy" else 0.9
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                           linewidth=lw, linestyle=ls, zorder=zorder))
    tx = {"center": x + w / 2, "left": x + pad}[align]
    ha = {"center": "center", "left": "left"}[align]
    cursor = y + h - pad
    if title:
        ax.text(tx, cursor, title, ha=ha, va="top", fontsize=fs_title,
                fontweight="bold", color=title_color or tc, zorder=zorder + 1)
        cursor -= fs_title * 0.155 + 0.9
    gap = line_gap if line_gap is not None else fs_body * 0.175 + 0.55
    for ln in lines:
        ax.text(tx, cursor, ln, ha=ha, va="top", fontsize=fs_body,
                color=INK, zorder=zorder + 1)
        cursor -= gap
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, lines, *, style="std", fs=FS_BODY):
    fill, edge = {"std": (FILL_STD, EDGE), "warn": (FILL_WARN, WARN)}[style]
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge,
                         linewidth=0.9, zorder=2))
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(cx, cy + (n - 1 - 2 * i) * (fs * 0.09), ln, ha="center",
                va="center", fontsize=fs, color=INK, zorder=3)


def arrow(ax, p0, p1, *, color=INK, lw=1.0, ls="-", head=6.0, zorder=4,
          rad=0.0, label=None, label_pos=0.5, label_off=(0, 1.2),
          label_fs=FS_SMALL, label_ha="center", label_color=None):
    style = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=style, shrinkA=0, shrinkB=0,
                                 zorder=zorder))
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * label_pos + label_off[0]
        my = p0[1] + (p1[1] - p0[1]) * label_pos + label_off[1]
        ax.text(mx, my, label, ha=label_ha, va="center", fontsize=label_fs,
                color=label_color or color, zorder=zorder + 1)


def elbow(ax, pts, *, color=INK, lw=1.0, ls="-", head=6.0, zorder=4):
    """Orthogonal poly-line; the arrow head sits on the final segment."""
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                solid_capstyle="butt", zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, head=head,
          zorder=zorder)


def note(ax, x, y, text, *, fs=FS_SMALL, color=MUTE, ha="left", va="top",
         style="italic", weight="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            style=style, fontweight=weight, zorder=6)


def save(fig, out_stem, dpi=400):
    import os
    for ext in ("pdf", "png"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=dpi, facecolor="white")
        print(f"[saved] {os.path.basename(path)}")
    plt.close(fig)
