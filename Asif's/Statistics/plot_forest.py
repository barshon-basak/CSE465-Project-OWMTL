"""Forest plot of every AUROC in significance_results.json with its Hanley-McNeil 95% CI,
against the chance line (0.5). Run after compute_significance.py."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "significance_results.json")) as f:
    data = json.load(f)

entries = list(reversed(data["entries"]))  # top-to-bottom = table order
labels = [f"{e['model']}\n({e['label']})" for e in entries]
aucs = [e["auc"] for e in entries]
lo_err = [e["auc"] - e["ci95_lo"] for e in entries]
hi_err = [e["ci95_hi"] - e["auc"] for e in entries]
colors = ["#c0392b" if e["crosses_chance"] else "#1b7f3a" for e in entries]

fig, ax = plt.subplots(figsize=(9, 0.55 * len(entries) + 1.5))
y = range(len(entries))
ax.errorbar(aucs, y, xerr=[lo_err, hi_err], fmt="o", capsize=4, elinewidth=1.5,
            markersize=6, ecolor="#888888", markerfacecolor="#333333", markeredgecolor="#333333")
for yi, e, c in zip(y, entries, colors):
    ax.plot(e["auc"], yi, "o", color=c, markersize=7, zorder=5)

ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="chance (AUROC = 0.5)")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("AUROC (95% CI, Hanley-McNeil)")
ax.set_xlim(0, 1)
ax.set_title("Open-set / OOD detection AUROC — point estimate + 95% CI\n"
             "(red = CI includes chance; green = CI excludes chance)", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()

out_path = os.path.join(HERE, "auroc_forest_plot.png")
fig.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
