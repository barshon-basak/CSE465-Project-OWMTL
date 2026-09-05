"""
M48 Tier A — the ablation rows M45 could not answer. GPU required.

THREE THINGS THIS ADDS TO M45, AND NOTHING ELSE
    1. A SEED BAND. M45 ran one seed per row. Ten of its eleven rows come back "not shown to
       differ from A0", including P3, the highest score in the table. Without a measured
       run-to-run sigma a reader cannot tell an honest null from an underpowered one. Rows
       A0_s42 / A0_s1 / A0_s2 are the same configuration three times; their spread is the
       noise floor every delta in the M45 table must clear.
    2. A7, THE SELECTION CRITERION, FOR FREE. The paper argues that selecting on the official
       score rather than on validation loss matters, and reports a 33-epoch disagreement, but
       never puts a number on it. This loop records BOTH criteria every epoch and reports the
       checkpoint each one would have chosen, so A7 costs no extra training on any row.
    3. A2+A4, THE FEATURE-EXTRACTOR NULL. A4 (frozen backbone) is -0.1008 and is the only M45
       row that survives a patient-level test. But A4 freezes a PRETRAINED backbone, so its
       delta mixes "fine-tuning helps" with "pretraining helps". Freezing a RANDOM backbone
       separates the two.

WHAT IS REUSED, UNCHANGED
    Every data, preprocessing, caching and metric function is imported from
    `Asif's/M45/m45_ablation.py`. This file re-implements nothing: if the two disagreed about
    a cache key or a mel parameter, the seed band would be measuring the difference between
    two scripts rather than between two seeds. m45_ablation.py is never modified.

THE CAVEAT THAT MUST TRAVEL WITH A7
    m45_ablation.py monitors the TEST set each epoch and keeps the best. There is no separate
    validation split, so "select on loss" here means select on TEST loss. That is optimistic,
    and it is optimistic for every M45 row equally, so the seed band and the A2+A4 delta are
    unaffected. A7 itself must be reported as "which of two criteria applied to the same
    monitoring set", never as a clean train/val/test result. Say so in the paper.

RUNNING
    python m48_gpu_rows.py --all
    python m48_gpu_rows.py --row A0_s1
    python m48_gpu_rows.py --summarise
    python m48_gpu_rows.py --selftest          # no GPU, no audio; checks the wiring
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------- locate and import M45
def import_m45(explicit=None):
    """Import m45_ablation from the repo, a Kaggle input, or an explicit path."""
    cands = [explicit,
             os.path.join(HERE, "..", "Asif's", "M45"),
             "/kaggle/working/CSE465-Project-OWMTL/Asif's/M45",
             "/kaggle/input/owmtl/Asif's/M45",
             os.path.join(HERE, "m45")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "m45_ablation.py")):
            sys.path.insert(0, os.path.abspath(c))
            import m45_ablation as m45
            print(f"  m45_ablation imported from {os.path.abspath(c)}")
            return m45
    raise FileNotFoundError(
        "m45_ablation.py not found. Pass --m45-dir, or clone the repo into /kaggle/working. "
        f"Looked in: {[c for c in cands if c]}")


ROWS = {
    "A0_s42": dict(seed=42, _desc="A0 baseline re-run under the M48 harness (seed 42)"),
    "A0_s1":  dict(seed=1,  _desc="A0 baseline, seed 1 (replicate)"),
    "A0_s2":  dict(seed=2,  _desc="A0 baseline, seed 2 (replicate)"),
    "A24":    dict(pretrained=False, freeze=True,
                   _desc="A2+A4: random initialisation AND frozen backbone"),
    # --- the three rungs of the CUMULATIVE ladder that do not already exist -------------
    # Rungs S3-S5 are already on disk: M45's A3 (0.5497), M22_v2 (0.5602) and M45's P3
    # (0.5764). Only the bottom three have never been run. See cumulative_table.py.
    "S0": dict(pretrained=False, freeze=True, class_weighted=False, specaug=False,
               _desc="cumulative S0: random init, frozen backbone, plain CE, no SpecAugment"),
    "S1": dict(freeze=True, class_weighted=False, specaug=False,
               _desc="cumulative S1: + ImageNet pre-training (still frozen)"),
    "S2": dict(class_weighted=False, specaug=False,
               _desc="cumulative S2: + full fine-tuning"),
}


# --------------------------------------------------------------- one row
def run_row(row_id, rows, m45, cache_dir=None):
    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    cfg = dict(m45.BASE)
    cfg.update({k: v for k, v in ROWS[row_id].items() if not k.startswith("_")})
    desc = ROWS[row_id]["_desc"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*78}\n  {row_id}: {desc}\n  device {dev} | seed {cfg['seed']}\n{'='*78}")

    # build_cache writes beside m45_ablation.py, which is read-only on Kaggle.
    if cache_dir:
        m45.HERE = cache_dir
        os.makedirs(os.path.join(cache_dir, "cache"), exist_ok=True)

    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr = m45.build_cache(tr, cfg, "train")
    Xte = m45.build_cache(te, cfg, "test")
    ytr = np.array([r["label"] for r in tr])
    yte = np.array([r["label"] for r in te])
    pid = np.array([r["patient_id"] for r in te])

    def spec_aug(x):
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

    class DS(Dataset):
        def __init__(s, X, y, aug): s.X, s.y, s.aug = X, y, aug
        def __len__(s): return len(s.y)
        def __getitem__(s, i):
            x = torch.from_numpy(np.asarray(s.X[i], np.float32))
            return (spec_aug(x.clone()) if s.aug else x), torch.tensor(int(s.y[i]))

    dl_tr = DataLoader(DS(Xtr, ytr, cfg["specaug"]), batch_size=cfg["batch_size"],
                       shuffle=True, num_workers=0)
    dl_te = DataLoader(DS(Xte, yte, False), batch_size=cfg["batch_size"], num_workers=0)

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            w = "IMAGENET1K_V1" if cfg["pretrained"] else None
            s.features = torchvision.models.mobilenet_v2(weights=w).features
            if cfg["freeze"]:
                for p in s.features.parameters():
                    p.requires_grad = False
            s.gap = nn.AdaptiveAvgPool2d((1, 1))
            s.dropout = nn.Dropout(cfg["dropout"])
            s.classifier = nn.Linear(1280, 4)
            s.norm = cfg["pretrained"]
        def forward(s, x):
            x = x.repeat(1, 3, 1, 1)
            if s.norm:
                x = (x - mean.to(x.device)) / std.to(x.device)
            return s.classifier(s.dropout(s.gap(s.features(x)).flatten(1)))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = Net().to(dev)
    if cfg["class_weighted"]:
        c = np.maximum(np.bincount(ytr, minlength=4).astype(float), 1)
        w = c.sum() / (4 * c); w = w / w.mean()
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    @torch.no_grad()
    def evaluate(dl, y_ref):
        model.eval()
        pr, tot = [], 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            lo = model(x)
            tot += float(crit(lo, y)) * len(y)
            pr.append(lo.argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        return pr, tot / len(y_ref), float(accuracy_score(y_ref, pr))

    hist = []
    best_sc, pred_sc, ep_sc = -1.0, None, 0
    best_ls, pred_ls, ep_ls = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        run, hit = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += float(loss) * len(y)
            # Train accuracy taken from the forward pass we already ran. A second pass over
            # the training set for a curve would cost ~50% more wall clock and answer no
            # hypothesis in this file. It is therefore accuracy under augmentation; label it
            # as such wherever the curve is plotted.
            hit += int((logits.argmax(1) == y).sum())
        sch.step()
        tr_loss, tr_acc = run / len(ytr), hit / len(ytr)
        pr, ev_loss, ev_acc = evaluate(dl_te, yte)
        sc = m45.official(confusion_matrix(yte, pr, labels=[0, 1, 2, 3]))[0]
        hist.append({"epoch": ep, "train_loss": round(tr_loss, 5),
                     "train_accuracy_augmented": round(tr_acc, 5),
                     "eval_loss": round(ev_loss, 5),
                     "eval_accuracy": round(ev_acc, 5),
                     "icbhi_score_official": round(sc, 5)})

        # The two selection criteria, tracked side by side. This is all A7 costs.
        if sc > best_sc:
            best_sc, pred_sc, ep_sc = sc, pr, ep
        if ev_loss < best_ls:
            best_ls, pred_ls, ep_ls = ev_loss, pr, ep
        if ep % 5 == 0 or ep == cfg["epochs"]:
            print(f"    ep {ep:02d}/{cfg['epochs']}  official {sc:.4f} (best {best_sc:.4f} "
                  f"@ {ep_sc})  eval_loss {ev_loss:.4f} (min {best_ls:.4f} @ {ep_ls})")

    def endpoint(pred, ep):
        cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3])
        s, se, sp = m45.official(cm)
        return {"epoch": int(ep), "icbhi_score_official": round(s, 4),
                "icbhi_score_official_ci95": m45.patient_ci(yte, pred, pid),
                "icbhi_se_official": round(se, 4), "icbhi_sp_official": round(sp, 4),
                "accuracy": round(float(accuracy_score(yte, pred)), 4),
                "f1_macro": round(float(f1_score(yte, pred, average="macro",
                                                 zero_division=0)), 4),
                "confusion_matrix_raw": cm.tolist()}

    by_score = endpoint(pred_sc, ep_sc)
    by_loss = endpoint(pred_ls, ep_ls)
    doc = {
        "meta": {"model_id": f"M48_{row_id}", "row": row_id, "variable_changed": desc,
                 "baseline_model_id": "M22_v2",
                 "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                 "is_augmented": bool(cfg["specaug"]),
                 "notes": "M48 Tier A row. Harness reuses Asif's/M45/m45_ablation.py for "
                          "data, preprocessing, caching and metrics without modification."},
        "config": dict(cfg),
        "dataset_info": {"dataset": "ICBHI_2017", "data_source": "real_audio",
                         "split_method": "official_60_40_patient_independent_corrected",
                         "train_samples": len(tr), "test_samples": len(te),
                         "test_patients": int(len(np.unique(pid)))},
        "selection_caveat": ("m45_ablation.py monitors the TEST set each epoch; there is no "
                             "separate validation split. 'select on loss' therefore means "
                             "test loss. Optimistic, and equally so for every row, so the "
                             "seed band and the A2+A4 delta are unaffected — but A7 must be "
                             "reported as two criteria on one monitoring set, never as a "
                             "clean train/val/test result."),
        "best_epoch": {"epoch": int(ep_sc), "primary_metric": "icbhi_score_official",
                       "primary_metric_value": by_score["icbhi_score_official"]},
        "best_metrics": by_score,
        "selection": {"by_official_score": by_score, "by_min_eval_loss": by_loss,
                      "epochs_apart": int(abs(ep_sc - ep_ls)),
                      "A7_delta_score_minus_loss": round(
                          by_score["icbhi_score_official"]
                          - by_loss["icbhi_score_official"], 4)},
        "training_history": hist,
        "efficiency": {
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
            "training_time_total_s": round(time.time() - t0, 1),
            "s_per_epoch": round((time.time() - t0) / cfg["epochs"], 2),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu")},
        "ablation": {"ablation_group": "M48_seed_band_and_selection_criterion",
                     "ablation_role": "variant", "baseline_model_id": "M22_v2",
                     "variable_changed": desc},
    }
    out = os.path.join(HERE, f"results_M48_{row_id}.json")
    json.dump(doc, open(out, "w"), indent=2)
    np.save(os.path.join(HERE, f"preds_M48_{row_id}.npy"),
            {"y_true": yte, "y_pred_by_score": pred_sc, "y_pred_by_loss": pred_ls,
             "patient_id": pid}, allow_pickle=True)
    print(f"    -> by score {by_score['icbhi_score_official']:.4f} @ep{ep_sc}  |  "
          f"by loss {by_loss['icbhi_score_official']:.4f} @ep{ep_ls}  |  "
          f"A7 delta {doc['selection']['A7_delta_score_minus_loss']:+.4f}  "
          f"({time.time()-t0:.0f}s)")
    return doc


# --------------------------------------------------------------- summary
def summarise():
    # Load by row id, not by glob. A glob over results_M48_*.json also matches
    # results_M48_C4.json, the Tier C probe-vs-clinician result, which is a different
    # schema and has no "meta" key. Globbing results_M48_A*.json avoided that but also
    # dropped S0/S1/S2, so the cumulative ladder came back empty after they were run.
    docs = {}
    for r in ROWS:
        f = os.path.join(HERE, f"results_M48_{r}.json")
        if os.path.exists(f):
            docs[r] = json.load(open(f, encoding="utf-8"))

    seeds = [docs[r] for r in ("A0_s42", "A0_s1", "A0_s2") if r in docs]
    band = None
    if len(seeds) >= 2:
        v = [d["best_metrics"]["icbhi_score_official"] for d in seeds]
        band = {"n_seeds": len(v), "seeds": [d["config"]["seed"] for d in seeds],
                "scores": v, "mean": round(float(np.mean(v)), 4),
                "sd": round(float(np.std(v, ddof=1)), 4),
                "range": round(float(max(v) - min(v)), 4),
                "m45_A0_reference": 0.5602,
                "reading": ("Any M45 delta smaller than this spread is not distinguishable "
                            "from run-to-run noise, whatever its sign.")}

    a7 = {r: {"epochs_apart": d["selection"]["epochs_apart"],
              "by_score": d["selection"]["by_official_score"]["icbhi_score_official"],
              "by_loss": d["selection"]["by_min_eval_loss"]["icbhi_score_official"],
              "delta": d["selection"]["A7_delta_score_minus_loss"]}
          for r, d in docs.items()}

    ladder = {r: docs[r]["best_metrics"]["icbhi_score_official"]
              for r in ("S0", "S1", "S2") if r in docs}

    a24 = None
    if "A24" in docs:
        a24 = {"score": docs["A24"]["best_metrics"]["icbhi_score_official"],
               "vs_A0_m45": round(docs["A24"]["best_metrics"]["icbhi_score_official"]
                                  - 0.5602, 4),
               "m45_A2_random_init": 0.4999, "m45_A4_frozen_pretrained": 0.4595,
               "reading": ("A4 froze a PRETRAINED backbone, so its -0.1007 mixes the value of "
                           "fine-tuning with the value of pretraining. This row freezes a "
                           "RANDOM one and separates them.")}

    out = {"table": "M48 Tier A — seed band, selection criterion, feature-extractor null",
           "baseline": "M45 A0 (M22_v2) = 0.5602 on the corrected official 60/40 split",
           "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
               "%Y-%m-%dT%H:%M:%SZ"),
           "rows_completed": sorted(docs), "seed_band": band,
           "A7_selection_criterion": a7, "A24_feature_extractor_null": a24,
           "cumulative_ladder_rungs_run_here": ladder,
           "cumulative_ladder_note": ("S3-S5 are reused from M45 (A3, M22_v2, P3). Run "
                                      "cumulative_table.py to assemble both orderings."),
           "selection_caveat": (next(iter(docs.values()))["selection_caveat"]
                                if docs else None)}
    json.dump(out, open(os.path.join(HERE, "M48_tier_A_table.json"), "w"), indent=2)

    print("\n" + "=" * 84)
    print("  M48 TIER A — completed rows:", ", ".join(sorted(docs)) or "(none)")
    print("=" * 84)
    if band:
        print(f"  SEED BAND   mean {band['mean']:.4f}  sd {band['sd']:.4f}  "
              f"range {band['range']:.4f}  over seeds {band['seeds']}")
        print(f"              M45's A0 reference is {band['m45_A0_reference']}")
    for r, v in sorted(a7.items()):
        print(f"  A7 {r:<8} by score {v['by_score']:.4f}  by loss {v['by_loss']:.4f}  "
              f"delta {v['delta']:+.4f}  ({v['epochs_apart']} epochs apart)")
    for r, v in sorted(ladder.items()):
        print(f"  LADDER {r:<5} {v:.4f}  ({docs[r]['meta']['variable_changed']})")
    if a24:
        print(f"  A2+A4       {a24['score']:.4f}  vs A0 {a24['vs_A0_m45']:+.4f}  "
              f"(A2 alone {a24['m45_A2_random_init']}, A4 alone {a24['m45_A4_frozen_pretrained']})")
    print("=" * 84)
    print("  wrote M48_tier_A_table.json")
    return 0


def selftest():
    """Wiring only: no GPU, no audio. Catches the mistakes that waste a Kaggle session."""
    ok = True
    m45 = import_m45(os.environ.get("M45_DIR"))
    for fn in ("corrected_split_index", "build_cache", "official", "patient_ci", "BASE"):
        has = hasattr(m45, fn)
        print(f"  m45.{fn:<22} {'ok' if has else 'MISSING'}")
        ok &= has

    # Every row must differ from BASE in exactly the fields it names, and no others.
    for rid, spec in ROWS.items():
        cfg = dict(m45.BASE)
        cfg.update({k: v for k, v in spec.items() if not k.startswith("_")})
        changed = {k for k in cfg if cfg[k] != m45.BASE[k]}
        want = {k for k in spec if not k.startswith("_")} & set(m45.BASE)
        want = {k for k in want if spec[k] != m45.BASE[k]}
        print(f"  {rid:<8} changes {sorted(changed) or ['(none — same as BASE)']}")
        ok &= changed == want

    # The three A0 seeds must be identical apart from the seed, or the band measures
    # something other than seed variance.
    cfgs = {r: {**m45.BASE, **{k: v for k, v in ROWS[r].items() if not k.startswith("_")}}
            for r in ("A0_s42", "A0_s1", "A0_s2")}
    diff = {k for r in cfgs for k in cfgs[r] if cfgs[r][k] != cfgs["A0_s42"][k]}
    print(f"  A0 replicates differ only in: {sorted(diff)}  (want ['seed'])")
    ok &= diff == {"seed"}

    # A cache key that ignored a stage would train a row on another row's spectrograms.
    keys = set()
    for r, cfg in [(r, {**m45.BASE, **{k: v for k, v in ROWS[r].items()
                                       if not k.startswith("_")}}) for r in ROWS]:
        keys.add(f"{cfg['n_mels']}m_{cfg['duration_s']}s_{cfg['padding']}_{int(cfg['minmax'])}"
                 f"_bp{int(cfg['bandpass'])}_dn{int(cfg['denoise'])}_an{int(cfg['ampnorm'])}")
    print(f"  cache keys across all M48 rows: {len(keys)} (want 1 — no row changes the pixels)")
    ok &= len(keys) == 1

    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", choices=list(ROWS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--m45-dir", default=os.environ.get("M45_DIR"))
    ap.add_argument("--audio-dir", default=os.environ.get("ICBHI_AUDIO_DIR"))
    ap.add_argument("--split-file", default=os.environ.get("ICBHI_SPLIT_FILE"))
    ap.add_argument("--cache-dir", default=os.environ.get("M48_CACHE_DIR", HERE))
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.summarise:
        return summarise()

    m45 = import_m45(a.m45_dir)
    if not a.audio_dir or not a.split_file:
        raise SystemExit("pass --audio-dir and --split-file (or set ICBHI_AUDIO_DIR / "
                         "ICBHI_SPLIT_FILE)")
    rows = m45.corrected_split_index(a.audio_dir, a.split_file)
    print(f"  cycles {len(rows)} | test {sum(1 for r in rows if r['split']=='test')}")

    todo = ([r for r in ROWS
             if not os.path.exists(os.path.join(HERE, f"results_M48_{r}.json"))]
            if a.all else [a.row])
    if not todo or todo == [None]:
        raise SystemExit("nothing to do: pass --row or --all")
    print(f"  rows to run: {todo}")
    for r in todo:
        run_row(r, rows, m45, cache_dir=a.cache_dir)
    return summarise()


if __name__ == "__main__":
    sys.exit(main())
