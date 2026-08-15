import nbformat as nbf, os
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as co
OUT = os.path.join(os.path.dirname(__file__), "notebooks"); os.makedirs(OUT, exist_ok=True)

# ---------- shared setup cell ----------
SETUP = """import sys, os
OWMTL_PKG = ".."
sys.path.insert(0, OWMTL_PKG)
import numpy as np, json
from collections import defaultdict
from owmtl.icbhi_data import KNOWN_DISEASES
OUT_DIR = "/kaggle/working"
Z = np.load(os.path.join(OUT_DIR, "concepts_all.npz"), allow_pickle=True)
X = Z["X"].astype("float32"); patient = Z["patient"]; split = Z["split"]
diagnosis = Z["diagnosis"]; concept_names = list(Z["concept_names"])
try:
    F = np.load(os.path.join(OUT_DIR, "M2_features.npy")).astype("float32")
    HAVE_FEATS = True
except Exception:
    F = np.zeros((len(X), 1), "float32"); HAVE_FEATS = False; print("NOTE: M2_features.npy missing")
# patient-level aggregation (known-class disease task)
lab_map = {d: i for i, d in enumerate(KNOWN_DISEASES)}
by = defaultdict(list)
for i in range(len(X)):
    if diagnosis[i] in lab_map: by[patient[i]].append(i)
pids, Xp, Fp, yp, spp = [], [], [], [], []
for pid, idxs in by.items():
    idxs = np.array(idxs); pids.append(pid)
    Xp.append(X[idxs].mean(0)); Fp.append(F[idxs].mean(0))
    yp.append(lab_map[diagnosis[idxs[0]]]); spp.append(split[idxs[0]])
pids, Xp, Fp, yp, spp = map(np.array, (pids, np.stack(Xp), np.stack(Fp), yp, spp))
print("patients", len(pids), "| classes", KNOWN_DISEASES)"""

# ================= 03 leakage =================
nb = new_notebook(); nb.cells = [
 md("""# 03 — Concept Leakage / Sufficiency  ·  Build step 04 (pre-G3) · Gate-G3 input · Novelty I2

Estimates **I(y ; features | concepts)** — the extra predictive information the encoder carries beyond the
clinical concepts — with honest cross-validated held-out log-likelihood. **Low** ⇒ concepts sufficient / faithful
(Path A). **High** ⇒ the model needs info outside the clinical concepts (Path B; the collapse/leakage is the finding).

Needs `concepts_all.npz` (nb01) and `M2_features.npy` (step 02). Output: `leakage_report.json`."""),
 co(SETUP),
 co("""from owmtl.leakage import estimate_leakage
if not HAVE_FEATS:
    raise SystemExit("Leakage needs M2 features (step 02). Export M2_features.npy first.")
rep = estimate_leakage(yp, Xp, Fp, n_splits=5, seed=42)
print(json.dumps(rep, indent=2))
with open(os.path.join(OUT_DIR, "leakage_report.json"), "w") as fh: json.dump(rep, fh, indent=2)"""),
 md("""**Read `leakage_bits` / `interpretation`.** This is a Gate-G3 input: combine it with the accuracy tradeoff
(nb02) and the intervention effect (nb04) to decide Path A vs Path B. Do **not** pick the path here — that is the
G3 decision, made once all three numbers exist.""")
]
nbf.write(nb, os.path.join(OUT, "03_concept_leakage_measurement.ipynb")); print("wrote 03")

# ================= 04 intervention =================
nb = new_notebook(); nb.cells = [
 md("""# 04 — Concept Intervention  ·  Build step 04 (pre-G3) · Gate-G3 input · Novelty I8 (clinician demo)

Trains the `independent` concept bottleneck and measures how the diagnosis responds when each concept is set from a
low to a high counterfactual value (does the diagnosis actually depend on the clinical concepts?), plus a directed
"clinician override" (e.g. assert a strong crackle → does COPD probability move as expected?).

Needs `concepts_all.npz` (nb01), `M2_features.npy` (step 02). Output: `intervention_report.json`."""),
 co(SETUP),
 co("""import torch
from owmtl.bottleneck import ConceptBottleneck
from owmtl.intervention import intervention_sensitivity, directed_intervention, summarize
tr = spp == "train"; K = len(KNOWN_DISEASES)
m = ConceptBottleneck(Xp.shape[1], Fp.shape[1], K, mode="independent", hidden=64)
xc, xf, ty = torch.tensor(Xp), torch.tensor(Fp), torch.tensor(yp).long()
cw = torch.tensor([(yp[tr]==k).sum() for k in range(K)], dtype=torch.float); cw = (cw.sum()/(cw+1e-6)); cw=cw/cw.sum()*K
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
trt = torch.tensor(tr)
for _ in range(200):
    opt.zero_grad(); lg,ch = m(xf[trt], xc[trt]); l = m.loss(lg, ty[trt], ch, xc[trt], class_weight=cw); l.backward(); opt.step()
rows = intervention_sensitivity(m, Fp, Xp, concept_names)
print(summarize(rows, top_k=6))"""),
 co("""# directed clinician-style interventions: assert a strong crackle / strong wheeze
name_idx = {n:i for i,n in enumerate(concept_names)}
hi = np.percentile(Xp, 90, axis=0)
report = {"gate": "G3_input", "sensitivity": rows, "directed": {}}
for concept, target in [("crackle_presence", "COPD"), ("wheeze_presence", "COPD")]:
    if target in KNOWN_DISEASES:
        di = directed_intervention(m, Fp, Xp, name_idx[concept], hi[name_idx[concept]],
                                   target_class=KNOWN_DISEASES.index(target))
        report["directed"][f"{concept}->{target}"] = di
        print(concept, "->", target, di)
with open(os.path.join(OUT_DIR, "intervention_report.json"), "w") as fh: json.dump(report, fh, indent=2)"""),
 md("""**Interpretation.** High total sensitivity + the *clinically-expected* concepts ranking top ⇒ the diagnosis is
genuinely driven by the clinical concepts (a Path-A positive). Near-zero sensitivity ⇒ the model ignores the concept
layer. This is the third Gate-G3 input; the A/B decision is made at G3, not here.""")
]
nbf.write(nb, os.path.join(OUT, "04_concept_intervention.ipynb")); print("wrote 04")
