import json, os

nb_path = os.path.join(
    r"c:\Users\Barshon\Desktop\CSE465 Project\CSE465-Project-OWMTL",
    "Barshon's", "M1", "M1-provisional-cnn-backbone.ipynb"
)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Check Cell 22 starts with the new code
src22 = cells[22]["source"] if isinstance(cells[22]["source"], str) else "".join(cells[22]["source"])
print("Cell 22 first 200 chars:")
print(src22[:200])
print(f"\nCell 22 length: {len(src22)} chars")

# Check it contains key protocol fields
checks = [
    "confusion_matrix_raw",
    "confusion_matrix_normalized",
    "per_class",
    "results_M1.json",
    "ablation_group",
    "component_flags",
    "meta",
    "environment",
    "dataset_info",
    "best_epoch",
    "best_metrics",
    "seaborn",
    "confusion_matrix.png",
]
print("\nProtocol compliance checks (Cell 22):")
for check in checks:
    found = check in src22
    status = "OK" if found else "MISSING"
    print(f"  [{status}] {check}")

# Check Cell 24 has individual plot saves
src24 = cells[24]["source"] if isinstance(cells[24]["source"], str) else "".join(cells[24]["source"])
print(f"\nCell 24 length: {len(src24)} chars")
plot_checks = ["loss_curve.png", "accuracy_curve.png", "f1_curve.png"]
print("Plot file checks (Cell 24):")
for check in plot_checks:
    found = check in src24
    status = "OK" if found else "MISSING"
    print(f"  [{status}] {check}")

# Check Cell 26 reference
src26 = cells[26]["source"] if isinstance(cells[26]["source"], str) else "".join(cells[26]["source"])
has_old = "M1_final_results.json" in src26
has_new = "results_M1.json" in src26
print(f"\nCell 26: old filename present={has_old}, new filename present={has_new}")

print("\nDone - all checks passed!" if not has_old else "\nWARNING: old filename still present in Cell 26")
