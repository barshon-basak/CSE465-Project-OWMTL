#!/usr/bin/env python3
"""Regenerate the paper's best-model figures from M22_v2's committed results JSON.

The paper's Fig. 3 and Fig. 4 were drawn for M22 on the 492-cycle fallback partition.
M22_v2 is the best model under the rule stated in the paper (highest official ICBHI on
the corrected partition among runs committing a raw confusion matrix), so the figures
have to come from its record instead.

Everything here is read from `results_M22_v2.json` - the confusion matrix and the
per-epoch history are both committed, so nothing is retrained and no number in the
figures can drift from the number in the table.

    python "Asif's/M22_v2/make_paper_figures.py"
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "Results", "results_M22_v2.json")
OUT = os.path.join(REPO, "DRAFT_PAPER", "figures")
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]


def confusion(doc):
    cm = np.array(doc["best_metrics"]["confusion_matrix_raw"], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.3f}\n({int(cm[i, j])})", ha="center",
                    va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"M22-v2, corrected partition ({int(cm.sum())} cycles)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_confusion_best.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, norm.diagonal(), cm.sum(axis=1)


def curves(doc):
    h = doc["training_history"]
    ep = [e["epoch"] for e in h]
    best = doc["best_epoch"]["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for ax, (a, b, name) in zip(axes, [("train_loss", "val_loss", "Loss"),
                                       ("train_accuracy", "val_accuracy", "Accuracy")]):
        ax.plot(ep, [e[a] for e in h], label="train", lw=1.4)
        ax.plot(ep, [e[b] for e in h], label="validation", lw=1.4)
        ax.axvline(best, ls="--", c="0.4", lw=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend(fontsize=8, frameon=False)
    axes[0].annotate(f"selected epoch {best}", xy=(best, axes[0].get_ylim()[1]),
                     xytext=(-4, -10), textcoords="offset points", ha="right", fontsize=8,
                     color="0.35")
    fig.tight_layout()
    path = os.path.join(OUT, "fig_curves_best.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)

    vl = [e["val_loss"] for e in h]
    return path, ep[int(np.argmin(vl))], min(vl)


def main():
    os.makedirs(OUT, exist_ok=True)
    doc = json.load(open(SRC, encoding="utf-8"))
    cpath, diag, support = confusion(doc)
    kpath, min_ep, min_vl = curves(doc)
    print(f"[saved] {cpath}")
    print("  per-class recall:", ", ".join(
        f"{c} {d:.3f} (n={int(n)})" for c, d, n in zip(CLASSES, diag, support)))
    print(f"[saved] {kpath}")
    print(f"  validation loss minimum {min_vl:.4f} at epoch {min_ep}; "
          f"checkpoint selected at epoch {doc['best_epoch']['epoch']} on "
          f"{doc['best_epoch']['primary_metric']}")


if __name__ == "__main__":
    main()
