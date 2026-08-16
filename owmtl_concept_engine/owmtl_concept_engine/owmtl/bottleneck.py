"""
owmtl.bottleneck
================

The strict CONCEPT BOTTLENECK head and its controls -- the second Critical Fix.

The diagnosis is forced to be a function ONLY of the interpretable acoustic-concept
vector (independent / sequential CBM). Two controls make the interpretability *cost*
and *leakage* measurable:

  * independent : disease predicted from the (fixed, DSP-derived) concept vector only.
                  This is the true bottleneck -- x -> c(DSP) -> y.
  * sequential  : a learned concept predictor maps encoder features -> concepts
                  (supervised by the DSP concepts), then disease from PREDICTED
                  concepts.  x -> f -> c_hat -> y.  Classic sequential CBM.
  * leaky       : disease sees concepts AND encoder features -> the model can bypass
                  the bottleneck. This is the control whose extra accuracy == leakage.
  * opaque      : disease from encoder features only (ignores concepts) -- the
                  upper-bound baseline for the accuracy--interpretability tradeoff.

Gate G3 reads three numbers produced with these: (1) tradeoff = opaque - independent
accuracy, (2) how much the leaky control beats independent (leakage proxy), and later
(3) intervention effect. This module produces (1) and (2); full information-theoretic
leakage and intervention are the downstream step, not a Critical Fix.

The module is encoder-agnostic: pass in per-item `features` (e.g. frozen M2 embeddings)
and `concepts` (from owmtl.concept_extractors). Patient-level aggregation is done by
the caller (mean/attention-pool a patient's cycles before or after the head).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("independent", "sequential", "leaky", "opaque")


def _mlp(in_dim, hidden, out_dim, p=0.3):
    if hidden and hidden > 0:
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )
    return nn.Linear(in_dim, out_dim)


class ConceptBottleneck(nn.Module):
    """Disease predictor with a selectable bottleneck mode (see module docstring)."""

    def __init__(self, concept_dim: int, feature_dim: int, n_classes: int,
                 mode: str = "independent", hidden: int = 64,
                 feature_proj: int = 32, dropout: float = 0.3):
        super().__init__()
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode = mode
        self.concept_dim = concept_dim
        self.feature_dim = feature_dim
        self.n_classes = n_classes

        # A learned concept predictor (only used by 'sequential').
        self.concept_predictor = _mlp(feature_dim, hidden, concept_dim, dropout)
        # Small projection of raw features for the 'leaky' control.
        self.feat_proj = nn.Linear(feature_dim, feature_proj)

        if mode == "independent":
            self.head = _mlp(concept_dim, hidden, n_classes, dropout)
        elif mode == "sequential":
            self.head = _mlp(concept_dim, hidden, n_classes, dropout)
        elif mode == "leaky":
            self.head = _mlp(concept_dim + feature_proj, hidden, n_classes, dropout)
        elif mode == "opaque":
            self.head = _mlp(feature_dim, hidden, n_classes, dropout)

    def forward(self, features: torch.Tensor, concepts: torch.Tensor):
        """Returns (logits, predicted_concepts_or_None)."""
        c_hat = None
        if self.mode == "independent":
            logits = self.head(concepts)
        elif self.mode == "sequential":
            c_hat = self.concept_predictor(features)
            logits = self.head(c_hat)
        elif self.mode == "leaky":
            logits = self.head(torch.cat([concepts, self.feat_proj(features)], dim=-1))
        else:  # opaque
            logits = self.head(features)
        return logits, c_hat

    def loss(self, logits, target, c_hat=None, concepts=None,
             concept_weight: float = 1.0, class_weight=None):
        """CE on disease + (sequential only) MSE that ties c_hat to the DSP concepts."""
        ce = F.cross_entropy(logits, target, weight=class_weight)
        if self.mode == "sequential" and c_hat is not None and concepts is not None:
            ce = ce + concept_weight * F.mse_loss(c_hat, concepts)
        return ce


def build_all_variants(concept_dim, feature_dim, n_classes, **kw):
    """Convenience: one instance per mode for the tradeoff/leakage comparison."""
    return {m: ConceptBottleneck(concept_dim, feature_dim, n_classes, mode=m, **kw)
            for m in MODES}
