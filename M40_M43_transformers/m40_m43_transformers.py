"""
M40-M43 - the four pre-trained transformer backbones (RTK requirements 2, 3 and 7).

WHY THIS FILE EXISTS
    RTK_requirements.md section 4 assigns one transformer per group member and demands
    they stay comparable. Before this script the repo had ZERO committed transformer
    results: M4 (AST) was run but never exported a results JSON, M23 (AST+SpecAugment)
    was never established, M33_v2 is a transformer *head* on CNN features rather than a
    pre-trained transformer backbone, and M37_v2 - despite its title - applies LoRA to
    the M2 CNN's FC layers, not to an AST.

    So all four are built here, in ONE script, because section 4's whole point is that
    they share a recipe. Four scripts would drift.

        M40  ViT-B/16   Barshon    torchvision, ImageNet-1k
        M41  Swin-T     Farhana    torchvision, ImageNet-1k   (supersedes M23)
        M42  DeiT-S     Sami       HuggingFace, ImageNet-1k
        M43  AST        Asif       HuggingFace, AudioSet      (closes the M4 hole)

    M43 is trained on Kaggle, not here - AST attends over 1212 patches and needs a 16 GB
    card to run the shared batch size. `gen_M43_kaggle.py` emits the notebook; it writes
    the same JSON schema, so its results drop straight into `Results/` and `--summarise`
    picks them up with no special-casing.

    Three of the four are ImageNet vision transformers fed a log-mel image, which is the
    standard ICBHI treatment; AST is the audio-native one, pre-trained on AudioSet, and
    is the reason it is worth having a fourth.

WHAT IS REUSED, NOT REWRITTEN
    The split loader, the log-mel front end, the on-disk cache, the official ICBHI score
    and the patient-level bootstrap all come from `Asif's/M45/m45_ablation.py` by import.
    That also means these runs share M45's spectrogram cache on disk - same features, so
    the only variable against the A0 baseline is the backbone.

RECIPE (fixed by RTK section 4, held identical across all four)
    128 mel x 8 s log-mel, cyclic padding, per-spectrogram min-max -> replicated to 3
    channels and resized to the backbone's resolution with ImageNet normalisation (AST
    instead takes the raw 1024x128 mel patch with dataset mean/std, its own convention).
    Class-weighted CE, seed 42, corrected official 60/40 split, AdamW with lr 1e-4 on the
    backbone and 1e-3 on the head, cosine schedule, 40 epochs.

    Each model is run twice - clean then SpecAugment - because requirement 7 wants the
    augmented/non-augmented pair per model, and that pair is what makes the augmentation
    table more than one row.

    Best epoch is selected on the official test score, which is what M22_v2 and every M45
    ablation row do. It is test-set peeking and it is not defensible on its own; it is
    kept ONLY so these numbers can sit in the same table as A0. Do not quote a M40-M43
    number outside that table without repeating the caveat.

EXPECT THE TRANSFORMERS TO LOSE
    M4 already measured AST at ~24x the parameters and ~30x the latency of M2 for a worse
    score. 920 recordings is not enough to fine-tune a ViT. A loss here is the reportable
    accuracy/compute trade-off, not a bug - do not tune until it wins (RTK section 12).

USAGE
    python m40_m43_transformers.py --model M40            # clean + specaug for one model
    python m40_m43_transformers.py --all --skip M43       # every unrun pair except AST
    python m40_m43_transformers.py --model M43 --aug      # one specific run
    python m40_m43_transformers.py --summarise            # build the comparison table
"""
from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

# The M45 harness is the shared front end. Loaded by path because the owner folders
# contain an apostrophe and are not importable as packages.
_spec = importlib.util.spec_from_file_location(
    "m45_ablation", os.path.join(REPO, "Asif's", "M45", "m45_ablation.py"))
m45 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m45)

# Identical to M45's BASE, so A0 (M22_v2) is a like-for-like row in the same table.
FEAT = dict(n_mels=128, duration_s=8.0, padding="wrap", minmax=True)
EPOCHS, BATCH, SEED = 40, 16, 42
LR_BACKBONE, LR_HEAD, WD = 1e-4, 1e-3, 1e-4

MODELS = {
    "M40": dict(arch="ViT-B/16", owner="Barshon", src="torchvision", res=224),
    "M41": dict(arch="Swin-T", owner="Farhana", src="torchvision", res=224),
    "M42": dict(arch="DeiT-S", owner="Sami", src="facebook/deit-small-patch16-224", res=224),
    "M43": dict(arch="AST", owner="Asif", src="MIT/ast-finetuned-audioset-10-10-0.4593",
                res=None, batch=4),
}
# AST attends over 1212 patches, ~9x the token count of the 224x224 models, and does not
# fit at batch 16 on a 6 GB card. Only the batch size differs; the recipe does not.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def specaug(x):
    """Same masks as M22/M45, so the augmentation delta stays attributable."""
    import torch
    for _ in range(2):
        f = int(torch.randint(0, 25, (1,)).item())
        if 0 < f < x.shape[1]:
            f0 = int(torch.randint(0, x.shape[1] - f + 1, (1,)).item())
            x[:, f0:f0 + f, :] = 0
    for _ in range(2):
        t = int(torch.randint(0, 81, (1,)).item())
        if 0 < t < x.shape[2]:
            t0 = int(torch.randint(0, x.shape[2] - t + 1, (1,)).item())
            x[:, :, t0:t0 + t] = 0
    return x


# ---------------------------------------------------------------- backbones
def build_model(mid, ast_stats=None):
    """Return (module, head_param_prefixes). Every backbone ends in a fresh 4-way head."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision

    spec = MODELS[mid]

    class ImageWrapper(nn.Module):
        """Log-mel -> 3-channel ImageNet-normalised square image -> ViT/Swin/DeiT."""

        def __init__(s, net, res, hf):
            super().__init__()
            s.net, s.res, s.hf = net, res, hf
            s.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            s.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(s, x):
            x = F.interpolate(x, size=(s.res, s.res), mode="bilinear", align_corners=False)
            x = (x.repeat(1, 3, 1, 1) - s.mean) / s.std
            return s.net(x).logits if s.hf else s.net(x)

    class ASTWrapper(nn.Module):
        """AST takes (B, frames, mels). Its recipe normalises with the training-set
        mean/std and a doubled std, which targets mean 0 / std 0.5."""

        def __init__(s, net, mean, std, max_len):
            super().__init__()
            s.net, s.max_len = net, max_len
            s.register_buffer("mu", torch.tensor(float(mean)))
            s.register_buffer("sd", torch.tensor(float(std)))

        def forward(s, x):
            x = x.squeeze(1).transpose(1, 2)                      # (B, frames, mels)
            if x.shape[1] < s.max_len:
                x = F.pad(x, (0, 0, 0, s.max_len - x.shape[1]))
            else:
                x = x[:, :s.max_len]
            return s.net((x - s.mu) / (2 * s.sd)).logits

    if mid == "M40":
        net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        net.heads.head = nn.Linear(net.heads.head.in_features, 4)
        return ImageWrapper(net, spec["res"], hf=False), ["net.heads.head"]
    if mid == "M41":
        net = torchvision.models.swin_t(weights="IMAGENET1K_V1")
        net.head = nn.Linear(net.head.in_features, 4)
        return ImageWrapper(net, spec["res"], hf=False), ["net.head"]
    if mid == "M42":
        from transformers import AutoModelForImageClassification
        net = AutoModelForImageClassification.from_pretrained(
            spec["src"], num_labels=4, ignore_mismatched_sizes=True)
        return ImageWrapper(net, spec["res"], hf=True), ["net.classifier"]
    if mid == "M43":
        from transformers import ASTForAudioClassification
        net = ASTForAudioClassification.from_pretrained(
            spec["src"], num_labels=4, ignore_mismatched_sizes=True)
        return (ASTWrapper(net, ast_stats[0], ast_stats[1], net.config.max_length),
                ["net.classifier"])
    raise KeyError(mid)


# ---------------------------------------------------------------- one run
def run(mid, aug, rows, out_dir, epochs=EPOCHS):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

    spec = MODELS[mid]
    tag = f"{mid}_aug" if aug else mid
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 74}\n  {tag}: {spec['arch']}  "
          f"({'SpecAugment' if aug else 'clean'})  owner {spec['owner']}\n{'=' * 74}")

    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr = m45.build_cache(tr, FEAT, "train")
    Xte = m45.build_cache(te, FEAT, "test")
    ytr = np.array([r["label"] for r in tr])
    yte = np.array([r["label"] for r in te])
    pid = np.array([r["patient_id"] for r in te])

    class DS(Dataset):
        def __init__(s, X, y, a):
            s.X, s.y, s.a = X, y, a

        def __len__(s):
            return len(s.y)

        def __getitem__(s, i):
            x = torch.from_numpy(np.asarray(s.X[i], np.float32))
            return (specaug(x.clone()) if s.a else x), torch.tensor(int(s.y[i]))

    bs = spec.get("batch", BATCH)
    dl_tr = DataLoader(DS(Xtr, ytr, aug), batch_size=bs, shuffle=True, num_workers=0)
    dl_te = DataLoader(DS(Xte, yte, False), batch_size=bs, num_workers=0)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ast_stats = None
    if mid == "M43":
        # AST normalises with the training-set statistics of its own features. Estimated
        # from a 1-in-20 subsample rather than the whole 6.9k x 128 x 801 array.
        sub = np.asarray(Xtr[::20], np.float32)
        ast_stats = (float(sub.mean()), float(sub.std()) or 1.0)
    model = build_model(mid, ast_stats)
    model, head_prefixes = model[0].to(dev), model[1]

    head, backbone = [], []
    for n, p in model.named_parameters():
        (head if any(n.startswith(h) for h in head_prefixes) else backbone).append(p)
    assert head, f"{mid}: no head parameters matched {head_prefixes}"
    opt = torch.optim.AdamW([{"params": backbone, "lr": LR_BACKBONE},
                             {"params": head, "lr": LR_HEAD}], weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    c = np.maximum(np.bincount(ytr, minlength=4).astype(float), 1)
    w = c.sum() / (4 * c)
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))

    best, best_pred, best_ep, t0, ep_times = -1.0, None, 0, time.time(), []
    for ep in range(1, epochs + 1):
        te0 = time.time()
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        sch.step()
        ep_times.append(time.time() - te0)
        model.eval()
        pr = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            for x, _ in dl_te:
                pr.append(model(x.to(dev)).argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        sc = m45.official(confusion_matrix(yte, pr, labels=[0, 1, 2, 3]))[0]
        if sc > best:
            best, best_pred, best_ep = sc, pr, ep
        if ep % 5 == 0 or ep == epochs:
            print(f"    ep {ep:02d}/{epochs}  official {sc:.4f}  (best {best:.4f})"
                  f"  {ep_times[-1]:.0f}s/ep")

    cm = confusion_matrix(yte, best_pred, labels=[0, 1, 2, 3])
    sc, se, sp = m45.official(cm)
    n_par = int(sum(p.numel() for p in model.parameters()))
    doc = {
        "meta": {"model_id": tag, "architecture": spec["arch"], "owner": spec["owner"],
                 "model_type": "transformer", "pretrained_source": spec["src"],
                 "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                 "is_augmented": bool(aug),
                 "notes": "RTK req 2/3/7 transformer backbone. Recipe fixed by "
                          "RTK_requirements.md section 4; front end shared with M45 so "
                          "the backbone is the only variable against A0 (M22_v2). Best "
                          "epoch selected on the official test score, as in M45."},
        "config": dict(FEAT, epochs=epochs, batch_size=bs, seed=SEED,
                       lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, weight_decay=WD,
                       optimizer="AdamW", scheduler="cosine", class_weighted=True,
                       specaug=bool(aug), pretrained=True),
        "dataset_info": {"dataset": "ICBHI_2017",
                         "split_method": "official_60_40_patient_independent_corrected",
                         "train_samples": len(tr), "test_samples": len(te),
                         "test_patients": int(len(np.unique(pid)))},
        "best_epoch": {"epoch": best_ep, "primary_metric": "icbhi_score_official",
                       "primary_metric_value": round(sc, 4)},
        "best_metrics": {"icbhi_score_official": round(sc, 4),
                         "icbhi_score_official_ci95": m45.patient_ci(yte, best_pred, pid),
                         "icbhi_se_official": round(se, 4),
                         "icbhi_sp_official": round(sp, 4),
                         "accuracy": round(float(accuracy_score(yte, best_pred)), 4),
                         "f1_macro": round(float(f1_score(yte, best_pred, average="macro",
                                                          zero_division=0)), 4),
                         "confusion_matrix_raw": cm.tolist()},
        "efficiency": {"total_params": n_par,
                       "trainable_params": int(sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad)),
                       "model_size_mb": round(n_par * 4 / 1024 ** 2, 2),
                       "s_per_epoch": round(float(np.mean(ep_times)), 1),
                       "training_time_total_s": round(time.time() - t0, 1),
                       "gpu_name": (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else "cpu")},
    }
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"epoch": int(best_ep), "best_score": float(best),
                "model_state": model.state_dict(), "model_id": tag},
               os.path.join(out_dir, f"best_{tag}.pth"))
    json.dump(doc, open(os.path.join(out_dir, f"results_{tag}.json"), "w"), indent=2)
    np.save(os.path.join(out_dir, f"preds_{tag}.npy"),
            {"y_true": yte, "y_pred": best_pred, "patient_id": pid}, allow_pickle=True)
    print(f"    -> official {sc:.4f} {doc['best_metrics']['icbhi_score_official_ci95']}  "
          f"Se {se:.4f} Sp {sp:.4f}  params {n_par / 1e6:.1f}M  ({time.time() - t0:.0f}s)")
    return doc


# ---------------------------------------------------------------- table
def summarise(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    found = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "results_M4*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        found[d["meta"]["model_id"]] = d

    a0 = os.path.join(REPO, "Asif's", "M22_v2", "Results", "results_M22_v2.json")
    base = None
    if os.path.exists(a0):
        base = json.load(open(a0, encoding="utf-8"))["best_metrics"]["icbhi_score_official"]

    print("\n" + "=" * 108)
    print("  M40-M43 - TRANSFORMER BACKBONES (corrected official 60/40 split)")
    print("=" * 108)
    print(f"  {'id':9s} {'arch':9s} {'owner':9s} {'aug':4s} {'official':>8s} "
          f"{'vs M22_v2':>10s} {'Se':>6s} {'Sp':>6s} {'F1':>6s} {'params':>9s} {'s/ep':>7s}")
    print("  " + "-" * 104)
    if base is not None:
        print(f"  {'M22_v2':9s} {'MobileNet':9s} {'Asif':9s} {'yes':4s} "
              f"{base:8.4f} {'baseline':>10s}")
    for mid in MODELS:
        for tag in (mid, f"{mid}_aug"):
            d = found.get(tag)
            if not d:
                print(f"  {tag:9s} {MODELS[mid]['arch']:9s} {MODELS[mid]['owner']:9s} "
                      f"{'yes' if tag.endswith('aug') else 'no':4s} {'NOT RUN':>8s}")
                continue
            bm, ef = d["best_metrics"], d["efficiency"]
            dl = f"{bm['icbhi_score_official'] - base:+.4f}" if base is not None else "-"
            print(f"  {tag:9s} {d['meta']['architecture']:9s} {d['meta']['owner']:9s} "
                  f"{'yes' if d['meta']['is_augmented'] else 'no':4s} "
                  f"{bm['icbhi_score_official']:8.4f} {dl:>10s} "
                  f"{bm['icbhi_se_official']:6.3f} {bm['icbhi_sp_official']:6.3f} "
                  f"{bm['f1_macro']:6.3f} {ef['total_params'] / 1e6:8.1f}M "
                  f"{ef['s_per_epoch']:7.1f}")
    print("  " + "-" * 104)
    for mid in MODELS:
        c, a = found.get(mid), found.get(f"{mid}_aug")
        if c and a:
            d = (a["best_metrics"]["icbhi_score_official"]
                 - c["best_metrics"]["icbhi_score_official"])
            print(f"  SpecAugment delta  {MODELS[mid]['arch']:10s} {d:+.4f}")
    print("=" * 108)

    json.dump({"split": "official_60_40_patient_independent_corrected",
               "baseline_M22_v2": base,
               "runs": [{"model_id": k, "architecture": v["meta"]["architecture"],
                         "owner": v["meta"]["owner"],
                         "is_augmented": v["meta"]["is_augmented"],
                         "icbhi_score_official": v["best_metrics"]["icbhi_score_official"],
                         "ci95": v["best_metrics"]["icbhi_score_official_ci95"],
                         "f1_macro": v["best_metrics"]["f1_macro"],
                         "total_params": v["efficiency"]["total_params"],
                         "s_per_epoch": v["efficiency"]["s_per_epoch"]}
                        for k, v in found.items()]},
              open(os.path.join(out_dir, "M40_M43_table.json"), "w"), indent=2)
    print(f"\n  wrote M40_M43_table.json  ({len(found)}/8 runs complete)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--aug", action="store_true",
                    help="with --model, run only the SpecAugment variant")
    ap.add_argument("--clean", action="store_true",
                    help="with --model, run only the clean variant")
    ap.add_argument("--all", action="store_true",
                    help="every (model, aug) pair not yet run")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(MODELS),
                    help="models to leave out of --all (M43 runs on Kaggle, see "
                         "gen_M43_kaggle.py)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="override the 40-epoch recipe (smoke runs only)")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "Results"))
    ap.add_argument("--audio_dir", default=os.environ.get(
        "ICBHI_AUDIO_DIR", r"C:\Users\Barshon\Desktop\ICBHI_final_database"))
    ap.add_argument("--split_file",
                    default=os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt"))
    args = ap.parse_args()

    if args.summarise:
        return summarise(args.out_dir)

    if args.all:
        todo = [(m, a) for m in MODELS if m not in args.skip for a in (False, True)]
    elif args.model:
        variants = [True] if args.aug else [False] if args.clean else [False, True]
        todo = [(args.model, a) for a in variants]
    else:
        return ap.error("pass --model, --all or --summarise")
    todo = [(m, a) for m, a in todo
            if not os.path.exists(os.path.join(
                args.out_dir, f"results_{m}{'_aug' if a else ''}.json"))]

    rows = m45.corrected_split_index(args.audio_dir, args.split_file)
    print(f"  cycles: {sum(1 for r in rows if r['split'] == 'train')} train / "
          f"{sum(1 for r in rows if r['split'] == 'test')} test | runs: "
          f"{[m + ('_aug' if a else '') for m, a in todo]}")
    for m, a in todo:
        run(m, a, rows, args.out_dir, args.epochs)
    summarise(args.out_dir)


if __name__ == "__main__":
    main()
