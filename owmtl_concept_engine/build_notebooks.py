"""Generate the two Critical-Fix notebooks with nbformat."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

OUT = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(OUT, exist_ok=True)


def md(s): return new_markdown_cell(s)
def co(s): return new_code_cell(s)


# ============================================================ Notebook 01
nb1 = new_notebook()
nb1.cells = [
    md("""# 01 — Physics Concept Extraction & Validation  ·  Critical Fix (M35 → concepts) + Gate G2 + CC2

**Current project direction:** physics-grounded, *label-free* acoustic **concept bottleneck** + faithfulness audit.
This notebook is the **first Critical Fix**: it turns the M35 physics DSP into standalone, clinically-named
**per-cycle concept extractors**, runs them on ICBHI (official 60/40 split), and **validates them against ICBHI's
crackle/wheeze cycle labels (Gate G2)**. It also runs the cheap **device-structure check (CC2)**.

**What "pass" looks like (G2):** the continuous `crackle_presence` / `wheeze_presence` concepts should track the
ICBHI labels with AUROC clearly above 0.5. If they do, the physics concepts are clinically meaningful and you may
build the bottleneck (notebook 02). If not, fall back to the minimal reliable concept set.

Outputs: `concepts_all.npz` (concept vectors + metadata), `concept_validation_report.json`, device report.
"""),
    co("""# --- setup: make the owmtl package importable ---------------------------------
# On Kaggle: upload the `owmtl/` folder as a dataset and point OWMTL_PKG at it,
# or place owmtl/ next to this notebook.
import sys, os
OWMTL_PKG = ".."          # TODO: path that CONTAINS the `owmtl` folder
sys.path.insert(0, OWMTL_PKG)

import numpy as np, json, time
from owmtl import icbhi_data as D
from owmtl.concept_extractors import (extract_concept_vector, CONCEPT_NAMES,
                                       CONCEPT_LABEL_MAP, ConceptConfig)
from owmtl import eval_utils as E
print("concepts:", CONCEPT_NAMES)
"""),
    co("""# --- CONFIG: point these at the real ICBHI files -----------------------------
AUDIO_DIR = "/kaggle/input/icbhi-dataset/audio_and_txt_files"   # TODO
SPLIT_FILE = "/kaggle/input/icbhi-dataset/ICBHI_challenge_train_test.txt"  # TODO official 60/40
DIAG_FILE  = "/kaggle/input/icbhi-dataset/ICBHI_Challenge_diagnosis.txt"   # TODO
SR = 16000
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)
"""),
    md("### CC2 — device-structure feasibility (Gate G4). Cheap; decides the device axis now."),
    co("""from owmtl.device_check import analyze, format_report
dev_report = analyze(AUDIO_DIR, DIAG_FILE)
print(format_report(dev_report))
with open(os.path.join(OUT_DIR, "device_structure_report.json"), "w") as fh:
    json.dump(dev_report, fh, indent=2)
"""),
    md("### Build the cycle index on the OFFICIAL split (patient-independent by construction)."),
    co("""records = D.build_cycle_index(AUDIO_DIR, SPLIT_FILE, DIAG_FILE)
from collections import Counter
print("cycles:", len(records))
print("split :", Counter(r.split for r in records))
print("sound :", Counter(r.sound_label for r in records), "(0=N,1=C,2=W,3=B)")
print("device:", Counter(r.device for r in records))
"""),
    md("### Extract the physics concept vector for every cycle (label-free DSP)."),
    co("""cfg = ConceptConfig(sr=SR)
X = np.zeros((len(records), len(CONCEPT_NAMES)), dtype=np.float32)
meta = {k: [] for k in ("patient","device","split","crackle","wheeze","sound_label","diagnosis")}
t0 = time.time()
for i, r in enumerate(records):
    try:
        y = D.load_cycle_waveform(AUDIO_DIR, r, sr=SR)
        X[i] = extract_concept_vector(y, SR, cfg)
    except Exception as ex:
        X[i] = 0.0
        if i < 5: print("warn", r.stem, ex)
    for k in meta: meta[k].append(getattr(r, k))
    if (i+1) % 500 == 0: print(f"{i+1}/{len(records)}  ({time.time()-t0:.0f}s)")
meta = {k: np.array(v) for k, v in meta.items()}
np.savez_compressed(os.path.join(OUT_DIR, "concepts_all.npz"),
                    X=X, concept_names=np.array(CONCEPT_NAMES), **meta)
print("saved concepts_all.npz  shape", X.shape)
"""),
    md("""### Gate G2 — validate concepts against ICBHI labels
`crackle_presence` vs the ICBHI `crackle` bit, `wheeze_presence` vs `wheeze` bit, on the **test** split, with
bootstrap 95% CIs (Protocol Essential #10). AUROC clearly > 0.5 ⇒ the physics concept tracks the clinical label."""),
    co("""test = meta["split"] == "test"
report = {"gate": "G2", "n_test_cycles": int(test.sum()), "concept_label_auroc": {}}
name_idx = {n: i for i, n in enumerate(CONCEPT_NAMES)}
for concept, labelname in CONCEPT_LABEL_MAP.items():
    y = meta[labelname][test].astype(int)
    s = X[test, name_idx[concept]]
    pt, lo, hi = E.auroc_ci(y, s, n_boot=1000)
    report["concept_label_auroc"][concept] = {"vs_label": labelname,
        "auroc": round(pt,4), "ci95": [round(lo,4), round(hi,4)],
        "n_pos": int(y.sum()), "n_neg": int((1-y).sum())}
    print(f"{concept:18s} vs {labelname:8s}: AUROC {pt:.3f}  95% CI [{lo:.3f},{hi:.3f}]  (n+={y.sum()})")

# also: how well the full concept vector linearly separates each sound label (sanity)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
tr = ~test
for name, lab in [("crackle","crackle"),("wheeze","wheeze")]:
    try:
        clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X[tr], meta[lab][tr])
        auc = roc_auc_score(meta[lab][test], clf.predict_proba(X[test])[:,1])
        report.setdefault("full_vector_auroc", {})[lab] = round(float(auc),4)
        print(f"full concept vector -> {lab}: AUROC {auc:.3f}")
    except Exception as ex:
        print("skip", lab, ex)

with open(os.path.join(OUT_DIR, "concept_validation_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)
"""),
    md("""### G2 decision
- **PASS** (single-concept AUROC materially > 0.5, e.g. ≳ 0.6, CI lower bound > 0.5): the physics concepts are
  clinically meaningful → proceed to notebook 02 (bottleneck). The full-vector AUROCs should be higher still.
- **BORDERLINE/FAIL:** shrink to the most reliable concepts (PAPR, spectral_flatness, wheeze band) or add a
  learned concept-refinement step; re-validate before building the bottleneck.

This report (`concept_validation_report.json`) is the evidence a reviewer will ask for that "physics concepts" are real.
"""),
]
nbf.write(nb1, os.path.join(OUT, "01_concept_extraction_and_validation.ipynb"))
print("wrote 01_concept_extraction_and_validation.ipynb")


# ============================================================ Notebook 02
nb2 = new_notebook()
nb2.cells = [
    md("""# 02 — Concept Bottleneck Training  ·  Critical Fix (M13 → strict bottleneck) + Gate G3 inputs

Trains the **strict concept bottleneck** and its controls, producing the three numbers **Gate G3** reads to pick
Path A vs Path B:
- **tradeoff** = opaque − independent disease accuracy (the interpretability *cost*),
- **leakage proxy** = leaky − independent (how much the bottleneck can be bypassed),
- per-class disease F1 (watch for **COPD collapse** — the §2.5 validity-hole failure mode).

Variants: `independent` (disease from concepts only — the true bottleneck), `sequential` (features→concepts→disease),
`leaky` (concepts + features — the cheat control), `opaque` (features only — the accuracy upper bound).

**Requires:** `concepts_all.npz` from notebook 01, and **frozen M2 features** aligned to the same cycle index
(fill in `get_M2_features`). Outputs: `results_M13cbm_<variant>.json` (§4 schema, with confusion matrix, CIs,
per-patient score dumps) + a tradeoff summary.
"""),
    co("""import sys, os
OWMTL_PKG = ".."
sys.path.insert(0, OWMTL_PKG)
import numpy as np, json, torch
from collections import defaultdict
from owmtl.bottleneck import ConceptBottleneck, MODES
from owmtl import eval_utils as E
from owmtl.icbhi_data import KNOWN_DISEASES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "/kaggle/working"; SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
"""),
    co("""# --- load concepts from notebook 01 ------------------------------------------
Z = np.load(os.path.join(OUT_DIR, "concepts_all.npz"), allow_pickle=True)
X = Z["X"].astype("float32")                 # (N_cycles, N_concepts)
patient = Z["patient"]; split = Z["split"]; diagnosis = Z["diagnosis"]
concept_names = list(Z["concept_names"])
print("concept matrix", X.shape)
"""),
    co("""# --- M2 frozen features aligned to the SAME cycle order ----------------------
# TODO: return an (N_cycles, feature_dim) array from your frozen M2 encoder, in the
# exact order of X (i.e. the record order from notebook 01). Easiest: in nb01, also
# run each cycle's mel-spectrogram through frozen M2 and save M2_features.npy.
def get_M2_features():
    path = os.path.join(OUT_DIR, "M2_features.npy")
    if os.path.exists(path):
        return np.load(path).astype("float32")
    raise FileNotFoundError(
        "M2_features.npy not found. independent-CBM can run on concepts alone, but "
        "sequential/leaky/opaque need frozen M2 features. Export them aligned to the "
        "cycle index (same order as concepts_all.npz).")

try:
    F = get_M2_features(); HAVE_FEATS = True
except Exception as ex:
    print("NOTE:", ex); F = np.zeros((len(X), 1), dtype="float32"); HAVE_FEATS = False
FEATURE_DIM = F.shape[1]
"""),
    co("""# --- patient-level aggregation (disease is a patient-level label) ------------
# Known-class disease task (COPD/Healthy/URTI). Aggregate a patient's cycles by mean.
lab_map = {d: i for i, d in enumerate(KNOWN_DISEASES)}
keep = np.array([d in lab_map for d in diagnosis])
def agg(patients_mask):
    by = defaultdict(list)
    for i in np.where(patients_mask)[0]:
        by[patient[i]].append(i)
    pids, Xc, Ff, yy, sp = [], [], [], [], []
    for pid, idxs in by.items():
        idxs = np.array(idxs)
        pids.append(pid)
        Xc.append(X[idxs].mean(0)); Ff.append(F[idxs].mean(0))
        yy.append(lab_map[diagnosis[idxs[0]]]); sp.append(split[idxs[0]])
    return (np.array(pids), np.stack(Xc), np.stack(Ff), np.array(yy), np.array(sp))

pids, Xp, Fp, yp, spp = agg(keep)
tr, te = spp == "train", spp == "test"
print(f"patients: {len(pids)}  train {tr.sum()}  test {te.sum()}  classes {KNOWN_DISEASES}")
"""),
    co("""# --- train one variant, return metrics + per-patient predictions -------------
def run_variant(mode, epochs=150, lr=1e-3, hidden=64):
    Cdim, Fdim, K = Xp.shape[1], FEATURE_DIM, len(KNOWN_DISEASES)
    model = ConceptBottleneck(Cdim, Fdim, K, mode=mode, hidden=hidden).to(DEVICE)
    xc = torch.tensor(Xp).to(DEVICE); xf = torch.tensor(Fp).to(DEVICE)
    y = torch.tensor(yp).long().to(DEVICE)
    # class weights for imbalance (URTI is small)
    cw = torch.tensor([ (yp[tr]==k).sum() for k in range(K) ], dtype=torch.float)
    cw = (cw.sum()/(cw+1e-6)); cw = (cw/cw.sum()*K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_t = torch.tensor(tr);
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits, chat = model(xf[tr_t], xc[tr_t])
        loss = model.loss(logits, y[tr_t], chat, xc[tr_t], class_weight=cw)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(xf, xc)
        proba = torch.softmax(logits, -1).cpu().numpy()
        pred = proba.argmax(1)
    return model, pred, proba
"""),
    co("""# --- run all variants, assemble the G3 tradeoff -----------------------------
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
results = {}
for mode in MODES:
    if mode != "independent" and not HAVE_FEATS:
        print(f"skip {mode}: needs M2 features"); continue
    model, pred, proba = run_variant(mode)
    acc = accuracy_score(yp[te], pred[te])
    f1m = f1_score(yp[te], pred[te], average="macro")
    perclass_f1 = f1_score(yp[te], pred[te], average=None, labels=list(range(len(KNOWN_DISEASES))))
    cm = confusion_matrix(yp[te], pred[te], labels=list(range(len(KNOWN_DISEASES))))
    # bootstrap CI on macro-F1
    f1_ci = E.bootstrap_ci(lambda a,b: f1_score(a,b,average="macro",labels=list(range(len(KNOWN_DISEASES)))),
                           yp[te], pred[te], n_boot=1000, seed=SEED)
    # dump per-patient scores (max known-class prob as the confidence)
    E.dump_scores(os.path.join(OUT_DIR, f"M13cbm_{mode}"),
                  ids=pids[te], scores=proba[te].max(1), labels=yp[te])
    results[mode] = {"acc": float(acc), "macro_f1": float(f1m),
                     "macro_f1_ci95": [round(f1_ci[1],4), round(f1_ci[2],4)],
                     "per_class_f1": {KNOWN_DISEASES[i]: float(perclass_f1[i]) for i in range(len(KNOWN_DISEASES))},
                     "confusion_matrix_raw": cm.tolist(), "pred": pred}
    print(f"{mode:11s}: acc {acc:.3f}  macroF1 {f1m:.3f}  CI[{f1_ci[1]:.3f},{f1_ci[2]:.3f}]  "
          f"perclass {dict(zip(KNOWN_DISEASES, np.round(perclass_f1,3)))}")
"""),
    co("""# --- Gate G3 inputs: tradeoff, leakage proxy, COPD-collapse check ------------
g3 = {}
if "independent" in results and "opaque" in results:
    g3["interpretability_cost_acc"] = round(results["opaque"]["acc"] - results["independent"]["acc"], 4)
    g3["interpretability_cost_f1"]  = round(results["opaque"]["macro_f1"] - results["independent"]["macro_f1"], 4)
if "independent" in results and "leaky" in results:
    g3["leakage_proxy_acc"] = round(results["leaky"]["acc"] - results["independent"]["acc"], 4)
# COPD-collapse: does independent just predict COPD?
if "independent" in results:
    ind_pred = results["independent"]["pred"][te]
    g3["independent_predicts_COPD_frac"] = round(float(np.mean(ind_pred == 0)), 3)
    g3["independent_COPD_f1"] = results["independent"]["per_class_f1"]["COPD"]
print("GATE G3 INPUTS:", json.dumps(g3, indent=2))
"""),
    co("""# --- write §4-schema results JSON per variant -------------------------------
for mode, r in results.items():
    E.write_results_json(
        os.path.join(OUT_DIR, f"results_M13cbm_{mode}.json"),
        model_id=f"M13cbm_{mode}", model_name=f"Concept bottleneck ({mode})",
        contributor="Barshon", split="patient_independent_official_60_40",
        best_metrics={"accuracy": r["acc"], "f1_macro": r["macro_f1"],
                      "f1_macro_ci95": r["macro_f1_ci95"],
                      "per_class_f1": r["per_class_f1"],
                      "confusion_matrix_raw": r["confusion_matrix_raw"]},
        ablation={"ablation_group": "bottleneck_type", "ablation_role":
                  "baseline" if mode=="independent" else "variant",
                  "variable_changed": f"bottleneck_type: {mode}",
                  "component_flags": {"has_concept_bottleneck": mode in ("independent","sequential"),
                                      "bottleneck_type": mode,
                                      "concept_source": "physics"}},
        concept_metrics={"gate_G3": g3, "concept_names": concept_names},
        notes="Critical-Fix concept bottleneck; disease = f(concepts). Path A/B decided at G3.")
print("wrote results_M13cbm_*.json for:", list(results))
"""),
    md("""### G3 decision (read the printed `GATE G3 INPUTS`)
- **Path A (constructive)** if `independent` holds accuracy near `opaque` (small `interpretability_cost`),
  `leakage_proxy` is small, and it does **not** collapse to COPD (`independent_predicts_COPD_frac` not ≈1).
- **Path B (audit)** if the bottleneck costs a lot / leaks heavily / collapses toward COPD — that collapse is the
  *finding*, not a failure. Either way you have committed confusion matrices, CIs, and per-patient score dumps.
"""),
]
nbf.write(nb2, os.path.join(OUT, "02_concept_bottleneck_training.ipynb"))
print("wrote 02_concept_bottleneck_training.ipynb")
