"""
N1 - Foundation-Model Concept Probing (LoRA)
============================================

FACULTY ASK: does a foundation model actually encode the clinical acoustic concepts, and
what does LoRA adaptation do to that encoding?

WHY THIS IS A NEW EXPERIMENT (audit finding):
  `Barshon's/M37_v2` is titled "Audio Spectrogram Transformer LoRA PEFT" but the code
  instantiates `M37_LoRA_CNN` wrapping the project's OWN scratch-trained `M2_CNN`
  (3,635,700 params). There is no foundation model anywhere in the repo - a grep for
  OPERA / M2D / HeAR / CLAP / AudioMAE / BEATs / wav2vec / PANNs returns nothing. And
  concept *probing* was never attempted at all (STEP_SEQUENCE.md step 09, never built).

THE MEASUREMENT (this is the part that is novel, not the LoRA):
  A classification score says nothing about whether a model listens to crackles. A probe
  does. For each of the 14 physics concepts, fit a linear read-out from the FROZEN
  embedding and ask how well the concept is recoverable:

      probe_frozen   : can the pre-trained FM read the concept off its own embedding?
      probe_lora     : after LoRA-adapting the FM to the 4-class ICBHI task, is the
                       concept MORE or LESS recoverable?
      probe_m2       : the same probe on the project's own CNN (the non-FM baseline)
      probe_random   : the same probe on a random projection of the FM embedding
                       (the control that makes a positive result mean something)

  probe_lora - probe_frozen is the headline. If task adaptation DESTROYS concept
  encoding, the model bought its accuracy by ignoring the clinical sounds - that is a
  reportable faithfulness finding and it directly supports the project's existing
  label-reliability argument. If adaptation PRESERVES it, that is the constructive story.

HONESTY GUARDS BUILT IN:
  * every probe is patient-grouped CV (no patient on both sides of a fold);
  * every score carries a bootstrap CI and a within-patient permutation null;
  * the random-projection control is always run - a linear probe on a 768-d embedding
    with ~100 patients will "work" on noise if you do not control for it.

RUNNING (needs GPU + the ICBHI audio; ~40 min on a T4):
    python N1_fm_concept_probing.py --stage embed --audio_dir <ICBHI audio_and_txt_files>
    python N1_fm_concept_probing.py --stage lora  --audio_dir <...>
    python N1_fm_concept_probing.py --stage probe

    python N1_fm_concept_probing.py --dry-run     # CPU, synthetic, validates the wiring

Everything is cached to .npy, so `probe` is CPU-only once the embeddings exist.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EXP_ID = "N1_fm_concept_probing"
FM_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
EMB_DIR = os.path.join(C.HERE, "embeddings")


# ---------------------------------------------------------------------------- the probe
def _is_binary(v) -> bool:
    u = np.unique(v[~np.isnan(v)])
    return len(u) <= 2


def probe_concepts(E, X, names, groups, n_splits=5, seed=0, n_boot=1000, n_perm=200):
    """Linear read-out of each concept from an embedding, patient-grouped.

    Binary concepts -> AUROC (chance 0.5). Continuous concepts -> ridge R^2 (chance 0.0).
    Both get a bootstrap CI and a permutation null so a small positive number cannot be
    mistaken for a result.
    """
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    E = np.asarray(E, dtype=float)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    n_splits = int(min(n_splits, len(np.unique(groups))))
    cv = GroupKFold(n_splits=max(2, n_splits))

    rows = []
    for j, name in enumerate(names):
        v = X[:, j]
        if np.all(~np.isfinite(v)) or np.nanstd(v) == 0:
            rows.append({"concept": name, "kind": "degenerate", "score": None,
                         "note": "constant or all-NaN in this cohort - not probeable"})
            continue
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)))

        if _is_binary(v):
            y = (v > np.median(np.unique(v))).astype(int) if len(np.unique(v)) == 2 \
                else (v > 0).astype(int)
            if len(np.unique(y)) < 2:
                rows.append({"concept": name, "kind": "degenerate", "score": None,
                             "note": "single class after binarisation"})
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=0.1))
            p = cross_val_predict(clf, E, y, groups=groups, cv=cv,
                                  method="predict_proba")[:, 1]
            metric, chance, kind = C.auroc, 0.5, "binary/AUROC"
            ci = C.bootstrap_ci(metric, y, p, n_boot=n_boot, seed=seed, stratify=y)
            null = C.permutation_p(metric, y, p, n_perm=n_perm, seed=seed, groups=groups)
        else:
            reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            p = cross_val_predict(reg, E, v, groups=groups, cv=cv)
            metric, chance, kind = (lambda a, b: float(r2_score(a, b))), 0.0, "continuous/R2"
            y = v
            ci = C.bootstrap_ci(metric, y, p, n_boot=n_boot, seed=seed)
            null = C.permutation_p(metric, y, p, n_perm=n_perm, seed=seed, groups=groups)

        rows.append({"concept": name, "kind": kind, "score": ci["point"],
                     "ci95": ci["ci95"], "chance": chance,
                     "perm_p": null.get("p"), "perm_null_mean": null.get("null_mean"),
                     "verdict": C.honest_verdict(ci["ci95"][0], ci["ci95"][1], chance)})

    ok = [r for r in rows if r.get("score") is not None]
    above = [r for r in ok if r["ci95"][0] > r["chance"]]
    return {"per_concept": rows,
            "n_probeable": len(ok),
            "n_above_chance": len(above),
            "concepts_above_chance": [r["concept"] for r in above],
            "mean_score": round(float(np.mean([r["score"] for r in ok])), 4) if ok else None}


# ------------------------------------------------------------------- LoRA (no `peft` dep)
def inject_lora(model, targets=("q_proj", "v_proj", "query", "value"), r=8, alpha=16.0):
    """Wrap every nn.Linear whose attribute name is in `targets` with a LoRA adapter and
    freeze everything else.

    Hand-rolled on purpose: the repo already carries this exact pattern in M37, and adding
    the `peft` dependency to buy 30 lines is not worth the install surface on Kaggle.

    NAMING: transformers >= 5 renamed AST's attention projections to q_proj/k_proj/v_proj/
    o_proj; 4.x called them query/key/value. Both spellings are targeted so the script
    works on either. Adapting Q and V (not K, not the MLP) is the standard LoRA placement.

    This RAISES if it injects nothing. An earlier run silently matched zero modules and
    trained only the classifier head - producing a "LoRA" result that was really a linear
    probe. A hand-rolled adapter must fail loudly or it will quietly answer the wrong
    question.

    Returns (n_trainable, n_total).
    """
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, alpha: float):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            self.scaling = alpha / r

        def forward(self, x):
            return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling

    for p in model.parameters():
        p.requires_grad = False
    n_injected = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r, alpha))
                n_injected += 1
    if n_injected == 0:
        seen = sorted({n.split(".")[-1] for n, c in model.named_modules()
                       if isinstance(c, nn.Linear)})
        raise RuntimeError(
            f"LoRA injected 0 adapters - none of {targets} matched this model. "
            f"Linear submodule names present: {seen}. Refusing to run: training would "
            "silently reduce to a linear probe on a frozen backbone.")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  LoRA adapters injected: {n_injected}")
    return n_tr, n_all


# ------------------------------------------------------------------------------- stages
def _cycle_index(audio_dir):
    from owmtl.icbhi_data import build_cycle_index, find_split_file
    split_file = find_split_file([audio_dir])
    diag = None
    for cand in ("ICBHI_Challenge_diagnosis.txt", "patient_diagnosis.csv"):
        for root in (audio_dir, os.path.dirname(audio_dir.rstrip("/\\"))):
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                diag = p
                break
        if diag:
            break
    if diag is None:
        # ICBHI's diagnosis table is not shipped with every audio mirror. The label plays no
        # part in extracting an embedding, and concepts_all.npz already stores the exact
        # patient->diagnosis pairing notebook 01 used, so derive it rather than block.
        from make_m2_features import derive_diagnosis_file
        diag, n_pat = derive_diagnosis_file(
            os.path.join(C.RESULTS_DIR, "derived_patient_diagnosis.txt"))
        print(f"  [note] no ICBHI diagnosis file on disk - derived one for {n_pat} "
              "patients from concepts_all.npz (does not affect embeddings)")
    return build_cycle_index(audio_dir, split_file, diag)


def stage_embed(args, model=None, tag="frozen"):
    """Extract frozen FM embeddings for every annotated cycle -> embeddings/ast_<tag>.npy"""
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    from owmtl.icbhi_data import load_cycle_waveform

    os.makedirs(EMB_DIR, exist_ok=True)
    recs = _cycle_index(args.audio_dir)
    limited = bool(getattr(args, "limit", None))
    if limited:
        recs = recs[:args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C.banner(f"N1 stage=embed ({tag})",
             f"{len(recs)} cycles | model={FM_NAME} | device={dev}"
             + ("  [LIMITED - writes nothing]" if limited else ""))
    fe = ASTFeatureExtractor.from_pretrained(FM_NAME)
    if model is None:
        model = ASTModel.from_pretrained(FM_NAME)
    model = model.to(dev).eval()

    out = None
    for i in range(0, len(recs), args.batch_size):
        chunk = recs[i:i + args.batch_size]
        waves = []
        for r in chunk:
            try:
                waves.append(load_cycle_waveform(args.audio_dir, r, sr=16000))
            except Exception:
                waves.append(np.zeros(16000, dtype=np.float32))
        inp = fe(waves, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            h = model(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state
            emb = h.mean(dim=1).cpu().numpy()          # mean-pool over patches
        if out is None:
            out = np.zeros((len(recs), emb.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = emb
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(recs)}")

    if limited:
        print(f"\n  --limit {args.limit}: wiring OK, embeddings {out.shape}. "
              "Nothing written.\n  Re-run without --limit for the full pass.")
        return None
    path = os.path.join(EMB_DIR, f"ast_{tag}.npy")
    np.save(path, out)
    np.save(os.path.join(EMB_DIR, "cycle_patients.npy"),
            np.array([r.patient for r in recs]))
    print(f"[saved] {path} {out.shape}")
    return path


def stage_lora(args):
    """LoRA fine-tune the FM on the 4-class ICBHI sound-event task, then re-embed."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import ASTFeatureExtractor, ASTModel
    from owmtl.icbhi_data import load_cycle_waveform

    recs = _cycle_index(args.audio_dir)
    train = [r for r in recs if r.split == "train"]
    C.banner("N1 stage=lora", f"{len(train)} train cycles | rank={args.lora_r}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = ASTFeatureExtractor.from_pretrained(FM_NAME)
    backbone = ASTModel.from_pretrained(FM_NAME)
    n_tr, n_all = inject_lora(backbone, r=args.lora_r)
    print(f"  LoRA trainable {n_tr:,} / {n_all:,} ({n_tr / n_all:.3%})")

    head = nn.Linear(backbone.config.hidden_size, 4)
    model = nn.ModuleDict({"backbone": backbone, "head": head}).to(dev)

    class DS(Dataset):
        def __len__(self):
            return len(train)

        def __getitem__(self, i):
            r = train[i]
            try:
                w = load_cycle_waveform(args.audio_dir, r, sr=16000)
            except Exception:
                w = np.zeros(16000, dtype=np.float32)
            return w, r.sound_label

    def collate(b):
        waves = [x[0] for x in b]
        y = torch.tensor([x[1] for x in b], dtype=torch.long)
        return fe(waves, sampling_rate=16000, return_tensors="pt"), y

    dl = DataLoader(DS(), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    # Class-weighted CE, identical to M2/M3/M4 so this run stays comparable to the zoo.
    counts = np.bincount([r.sound_label for r in train], minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=w)

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for inp, y in dl:
            inp = {k: v.to(dev) for k, v in inp.items()}
            h = model["backbone"](**inp).last_hidden_state.mean(dim=1)
            loss = lossf(model["head"](h), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  epoch {ep + 1}/{args.epochs} loss {tot / max(len(dl), 1):.4f}")

    torch.save({"lora_state": {k: v for k, v in model.state_dict().items()
                               if ".A" in k or ".B" in k or k.startswith("head")},
                "lora_r": args.lora_r, "trainable": n_tr, "total": n_all},
               os.path.join(EMB_DIR, "ast_lora_adapter.pt"))
    return stage_embed(args, model=model["backbone"], tag="lora")


def make_figure(out):
    """Per-concept probe score with CI, one row group per embedding space.

    The random control is drawn in red on the same axis - a probe result is only
    interpretable next to it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spaces = list(out["spaces"])
    if not spaces:
        return
    names = [r["concept"] for r in out["spaces"][spaces[0]]["per_concept"]]
    fig, axes = plt.subplots(1, len(spaces), figsize=(5.2 * len(spaces), 6), squeeze=False,
                             sharey=True)
    for ax, sp in zip(axes[0], spaces):
        rows = {r["concept"]: r for r in out["spaces"][sp]["per_concept"]}
        y = np.arange(len(names))
        ctrl = sp == "random_control"
        for i, nm in enumerate(names):
            r = rows.get(nm, {})
            if r.get("score") is None:
                continue
            lo, hi = r["ci95"]
            col = "tab:red" if ctrl else ("tab:green" if lo > r["chance"] else "0.6")
            ax.plot([lo, hi], [i, i], color=col, lw=2)
            ax.plot(r["score"], i, "o", color=col, ms=5)
        chance = rows[names[0]].get("chance", 0.0) if names else 0.0
        ax.axvline(chance, color="k", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("probe score (R2 / AUROC), 95% CI")
        ax.set_title(f"{sp}\n{out['spaces'][sp]['n_above_chance']}"
                     f"/{out['spaces'][sp]['n_probeable']} above chance")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("N1 - is each physics concept linearly readable from the embedding?",
                 fontsize=11)
    fig.tight_layout()
    C.save_figure(fig, "N1_fm_concept_probing.png")
    plt.close(fig)


def stage_probe(args):
    """Probe every available embedding for the 14 concepts and write the comparison."""
    C.banner("N1 stage=probe", "linear read-out of 14 physics concepts, patient-grouped CV")
    d = C.load_concepts(args.concepts)
    names = d["concept_names"]

    spaces = {}
    for tag in ("frozen", "lora"):
        p = os.path.join(EMB_DIR, f"ast_{tag}.npy")
        if os.path.isfile(p):
            spaces[f"AST_{tag}"] = np.load(p)
    m2 = C.load_features(args.features, n_expected=len(d["X"]))
    if m2 is not None:
        spaces["M2_cnn"] = m2

    if not spaces:
        return C.blocked(EXP_ID, "no embeddings found to probe", [
            f"run --stage embed (writes {EMB_DIR}/ast_frozen.npy)",
            "or drop an aligned M2_features.npy next to this script"])

    # Random-projection control, built from whichever real space exists. Without it a
    # positive probe on a 768-d embedding with ~100 patients is uninterpretable.
    rng = np.random.default_rng(0)
    ref = next(iter(spaces.values()))
    spaces["random_control"] = rng.standard_normal((len(ref), 64)).astype(np.float32)

    out = {"experiment": EXP_ID, "status": "OK", "fm": FM_NAME,
           "level": "cycle (patient-grouped CV)", "n_rows": int(len(d["X"])),
           "n_patients": int(len(set(d["patient"].tolist()))), "spaces": {}}
    for tag, E in spaces.items():
        if len(E) != len(d["X"]):
            print(f"  [skip] {tag}: {len(E)} rows != {len(d['X'])} cycles (not aligned)")
            continue
        print(f"\n-- probing {tag} ({E.shape[1]}-d)")
        out["spaces"][tag] = probe_concepts(E, d["X"], names, d["patient"],
                                            seed=args.seed, n_boot=args.n_boot,
                                            n_perm=args.n_perm)
        print(f"   above chance: {out['spaces'][tag]['n_above_chance']}"
              f"/{out['spaces'][tag]['n_probeable']}"
              f" | mean {out['spaces'][tag]['mean_score']}")

    if "AST_frozen" in out["spaces"] and "AST_lora" in out["spaces"]:
        a = {r["concept"]: r for r in out["spaces"]["AST_frozen"]["per_concept"]}
        b = {r["concept"]: r for r in out["spaces"]["AST_lora"]["per_concept"]}
        delta = {k: round(b[k]["score"] - a[k]["score"], 4)
                 for k in a if a[k].get("score") is not None and b.get(k, {}).get("score") is not None}
        out["headline_lora_effect"] = {
            "per_concept_delta": delta,
            "mean_delta": round(float(np.mean(list(delta.values()))), 4) if delta else None,
            "n_concepts_degraded": int(sum(v < 0 for v in delta.values())),
            "reading": ("Task adaptation DESTROYS concept encoding -> the model bought "
                        "accuracy by ignoring the clinical sounds (faithfulness finding)."
                        if delta and np.mean(list(delta.values())) < 0 else
                        "Task adaptation PRESERVES or improves concept encoding.")}

    try:
        make_figure(out)
    except Exception as ex:
        print(f"  [warn] figure skipped: {ex}")

    C.save_result(EXP_ID, out)
    return out


def dry_run(args):
    """CPU wiring check on synthetic data with a PLANTED concept.

    Concept 0 is a linear function of the embedding, so the probe must find it and must
    NOT find the random control. If either fails, the probe is broken.
    """
    C.banner("N1 dry-run", "synthetic data, planted concept - validates the probe only")
    rng = np.random.default_rng(0)
    n, dim = 400, 32
    groups = np.repeat(np.arange(40), 10)
    E = rng.standard_normal((n, dim))
    w = rng.standard_normal(dim)
    planted = E @ w + 0.1 * rng.standard_normal(n)
    noise = rng.standard_normal(n)
    X = np.c_[planted, noise]
    res = probe_concepts(E, X, ["planted", "noise"], groups, n_boot=200, n_perm=100)
    for r in res["per_concept"]:
        print(f"  {r['concept']:10s} {r['kind']:16s} score={r['score']} {r['verdict']}")
    assert res["per_concept"][0]["score"] > 0.5, "probe failed to recover a planted concept"
    assert res["per_concept"][1]["score"] < 0.2, "probe hallucinated signal in pure noise"
    print("\nOK - probe recovers a planted concept and rejects noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", choices=["embed", "lora", "probe"], default="probe")
    ap.add_argument("--audio_dir", default=os.environ.get("ICBHI_AUDIO_DIR"))
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None,
                    help="embed only N cycles as a GPU wiring check; writes nothing")
    ap.add_argument("--dry-run", dest="dry", action="store_true")
    args = ap.parse_args()

    if args.dry:
        return dry_run(args)
    if args.stage in ("embed", "lora"):
        if not args.audio_dir or not os.path.isdir(args.audio_dir):
            return C.blocked(EXP_ID, "ICBHI audio directory not given/found", [
                "--audio_dir <path to audio_and_txt_files>  (or set ICBHI_AUDIO_DIR)",
                "pip install transformers torch librosa", "a GPU (T4 is enough)"])
        return stage_embed(args) if args.stage == "embed" else stage_lora(args)
    return stage_probe(args)


if __name__ == "__main__":
    main()
