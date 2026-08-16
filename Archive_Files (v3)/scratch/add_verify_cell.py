import json
import os

path = r"c:\Users\Barshon\Desktop\CSE465 Project\CSE465-Project-OWMTL\Barshon's\Gap7\OOD_Generalization_Evaluation.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell to add
verify_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# ---- Pre-Flight Check: Verify Attachments ----\n",
        "import os\n",
        "\n",
        "def check_attachment(name, keywords):\n",
        "    if not os.path.exists('/kaggle/input'):\n",
        "        return f\"⚠️ /kaggle/input not found. Are you running this on Kaggle?\"\n",
        "    found = False\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        for f in files:\n",
        "            if any(k in root.lower() or k in f.lower() for k in keywords):\n",
        "                # additional strictness for checkpoints so we don't just match anything\n",
        "                if 'checkpoint' in name.lower() and not f.endswith('.pth'):\n",
        "                    continue\n",
        "                return f\"✅ {name} Found (e.g., {f})\"\n",
        "    return f\"❌ {name} MISSING - Please attach it!\"\n",
        "\n",
        "print(\"Checking Datasets...\")\n",
        "print(check_attachment(\"ICBHI Dataset\", [\"icbhi\", \"respiratory\"]))\n",
        "print(check_attachment(\"Coswara Dataset\", [\"coswara\"]))\n",
        "print(check_attachment(\"SPRSound Dataset\", [\"sprsound\"]))\n",
        "\n",
        "print(\"\\nChecking Checkpoints...\")\n",
        "print(check_attachment(\"M2 Checkpoint\", [\"m2\"]))\n",
        "print(check_attachment(\"M3 Checkpoint\", [\"m3\"]))\n",
        "print(check_attachment(\"M30 Checkpoint\", [\"m30\"]))\n",
        "print(check_attachment(\"M35 Checkpoint\", [\"m35\"]))\n",
        "\n",
        "print(\"\\nIf any show ❌, please 'Add Input' in the right sidebar before running the rest of the notebook.\")\n"
    ]
}

# Insert at index 1 (after the first markdown block)
nb['cells'].insert(1, verify_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Added verification cell.")
