"""Smoke check: every M40-M43 backbone accepts the shared 128x801 log-mel and emits 4
logits, and the head/backbone parameter split actually finds a head. Downloads weights on
first run.

    python test_backbones.py
"""
import importlib.util
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "m4x", os.path.join(HERE, "m40_m43_transformers.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

x = torch.rand(2, 1, 128, 801)                      # what build_cache hands the model

for mid in m.MODELS:
    net, prefixes = m.build_model(mid, ast_stats=(0.5, 0.2))
    net.eval()
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 4), f"{mid}: {tuple(out.shape)} != (2, 4)"
    head = [n for n, _ in net.named_parameters()
            if any(n.startswith(p) for p in prefixes)]
    assert head, f"{mid}: head prefixes {prefixes} matched nothing"
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  {mid:4s} {m.MODELS[mid]['arch']:9s} ok  {n_par / 1e6:6.1f}M params  "
          f"head={len(head)} tensors")

y = m.specaug(torch.ones(1, 128, 801).clone())
assert (y == 0).any(), "specaug masked nothing"
assert (y == 1).any(), "specaug masked everything"
print("  specaug ok")
print("all backbones ok")
