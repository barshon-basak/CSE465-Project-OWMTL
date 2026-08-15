"""
owmtl.intervention  (Build step 04 — pre-G3, deterministic)
===========================================================

Clinician CONCEPT INTERVENTION on the bottleneck — a Gate-G3 input and the
clinician-facing demo (I8). "If a clinician corrects a mis-heard concept, does the
diagnosis change the way it should?"

Two things:
  1. intervention_sensitivity: for each concept, set it to a LOW vs HIGH counterfactual
     value and measure how much the predicted diagnosis distribution moves. A concept the
     diagnosis actually depends on will move it; a concept the model ignores will not.
     -> evidence that the diagnosis is genuinely a function of the clinical concepts.
  2. directed_intervention: set a specific concept to a specific value (simulating a
     clinician overriding, e.g., "there IS a coarse crackle") and report the change in a
     target class probability.

Only meaningful for modes that actually consume concepts (independent / sequential /
leaky); for `opaque` the sensitivity is ~0 by construction (a useful control).

torch + numpy.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def _predict_proba(model, features, concepts):
    model.eval()
    logits, _ = model(features, concepts)
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def intervention_sensitivity(model, features, concepts, concept_names,
                             low_pct=10, high_pct=90, device="cpu"):
    """Per-concept sensitivity = mean L1 change in predicted class-probability vector
    when a concept is set from its low percentile to its high percentile (across the
    batch), holding features fixed. Returns a ranked list of dicts."""
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    concepts = torch.as_tensor(concepts, dtype=torch.float32, device=device)
    C = concepts.shape[1]
    los = np.percentile(concepts.cpu().numpy(), low_pct, axis=0)
    his = np.percentile(concepts.cpu().numpy(), high_pct, axis=0)
    base = _predict_proba(model, features, concepts)
    rows = []
    for j in range(C):
        clo = concepts.clone(); clo[:, j] = float(los[j])
        chi = concepts.clone(); chi[:, j] = float(his[j])
        plo = _predict_proba(model, features, clo)
        phi = _predict_proba(model, features, chi)
        sens = float(torch.mean(torch.sum(torch.abs(phi - plo), dim=-1)).item()) / 2.0
        # signed effect on each class (mean prob change high vs low)
        delta_by_class = (phi - plo).mean(0).cpu().numpy()
        rows.append({"concept": concept_names[j],
                     "sensitivity": round(sens, 4),
                     "delta_by_class": [round(float(x), 4) for x in delta_by_class]})
    rows.sort(key=lambda r: -r["sensitivity"])
    return rows


@torch.no_grad()
def directed_intervention(model, features, concepts, concept_idx, new_value,
                          target_class, device="cpu"):
    """Set concepts[:, concept_idx] = new_value and report the mean change in
    p(target_class). Simulates a clinician asserting a concept value."""
    features = torch.as_tensor(features, dtype=torch.float32, device=device)
    concepts = torch.as_tensor(concepts, dtype=torch.float32, device=device)
    base = _predict_proba(model, features, concepts)[:, target_class]
    c2 = concepts.clone(); c2[:, concept_idx] = float(new_value)
    new = _predict_proba(model, features, c2)[:, target_class]
    d = (new - base).cpu().numpy()
    return {"concept_idx": int(concept_idx), "new_value": float(new_value),
            "target_class": int(target_class),
            "mean_delta_p": round(float(d.mean()), 4),
            "frac_increased": round(float((d > 0).mean()), 3)}


def summarize(sens_rows, top_k=5):
    lines = ["Concept intervention sensitivity (diagnosis responds to which concepts):"]
    for r in sens_rows[:top_k]:
        lines.append(f"  {r['concept']:24s} sensitivity={r['sensitivity']:.3f}")
    total = sum(r["sensitivity"] for r in sens_rows)
    lines.append(f"  (total responsiveness across concepts = {total:.3f}; "
                 f"~0 means the model ignores the concept layer)")
    return "\n".join(lines)
