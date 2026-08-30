"""
owmtl.m2_ready  (concrete M2 loader for Build step 02)
======================================================

Turn-key `load_m2` for the feature export -- resolves the one fill-in in
owmtl.m2_features. The M2 architecture is inlined verbatim from
`Asif's/M2/M2_cnn_baseline_tuned.ipynb`, which already exposes `get_embedding(x)`
(the penultimate GAP features). So step 02 needs only the checkpoint path.

Selected M2 config (from the M12 decision / Model_Training_Reference):
    depth=5, base_width=48, dropout=0.4, fc_dim=128  ->  embedding_dim = 768, ~3.63M params

If your committed config differs, pass the right values (confirm against
`Asif's/M2/results_M2.json` -> config, or `Asif's/M12/`).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(pool))

    def forward(self, x):
        return self.block(x)


class M2_CNN(nn.Module):
    """Verbatim M2 tuned CNN. Input (B,1,n_mels,T) -> logits; get_embedding -> (B, C_last)."""

    def __init__(self, num_classes=4, depth=5, base_width=48, dropout=0.4, fc_dim=128):
        super().__init__()
        channels = [base_width * (2 ** i) for i in range(depth)]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch)); in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(channels[-1], fc_dim),
                                  nn.ReLU(inplace=True), nn.Linear(fc_dim, num_classes))
        self.embedding_dim = channels[-1]

    def forward(self, x):
        return self.head(self.dropout(self.gap(self.encoder(x)).flatten(1)))

    def get_embedding(self, x):
        return self.gap(self.encoder(x)).flatten(1)


def ready_load_m2(checkpoint_path, num_classes=4, depth=5, base_width=48,
                  dropout=0.4, fc_dim=128):
    """Return (model, embed_fn) for owmtl.m2_features.export_features.
    Loads the frozen M2 checkpoint and uses its penultimate embedding as the feature."""
    def _load():
        model = M2_CNN(num_classes, depth, base_width, dropout, fc_dim)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = sd.get("model_state", sd.get("state_dict", sd))
        model.load_state_dict(state)
        for p in model.parameters():
            p.requires_grad_(False)

        def embed_fn(m, x):
            return m.get_embedding(x)
        return model, embed_fn
    return _load
