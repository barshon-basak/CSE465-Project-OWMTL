"""
M44 — Interpretability analysis of the best model (RTK requirement 9)

BEST MODEL: M22_v2, MobileNetV2 + SpecAugment, official ICBHI 0.5602 [0.5081, 0.6137]
on `official_60_40_patient_independent_corrected` (2,636 test cycles / 47 patients).
Selected by the rule in RTK_requirements.md section 6: highest `icbhi_score_official` on
the corrected split among models with a committed raw confusion matrix.

WHAT THIS PRODUCES
    1. Grad-CAM over the final convolutional block, overlaid on the log-mel spectrogram.
    2. Occlusion sensitivity — a model-agnostic cross-check that needs no gradients.
    3. A four-panel figure per class (one correct example each of Normal / Crackle /
       Wheeze / Both) plus a misclassified example.
    4. TWO QUANTITATIVE ATTRIBUTION MEASURES (see the note below).

A CORRECTION TO THE SPEC, AND WHY
    RTK_requirements.md section 9 asks for a "pointing game": measure whether Grad-CAM
    mass falls inside the annotated crackle/wheeze event window, and report the hit rate.

    That is not implementable on ICBHI. Its annotation files carry ONE row per
    respiratory cycle — `start  end  crackle  wheeze` — which marks whether a cycle
    CONTAINS an adventitious sound, not where inside the cycle it occurs. The pipeline
    already crops to exactly that window, so the "annotated event window" IS the model
    input and a pointing game against it scores 100% by construction. There is no
    finer-grained annotation anywhere in the corpus.

    Two measures are substituted. Both are computable from cycle-level labels alone and
    both can fail:

    (a) BAND POINTING. Wheezes are sustained tonal energy in roughly 100-1000 Hz — the
        band this project's own `concept_extractors.py` uses. If the model has learned
        wheeze acoustics, its attribution on wheeze-positive cycles should sit in that
        band MORE than on wheeze-negative cycles. Reported as the difference in
        in-band Grad-CAM mass between the two groups, with a patient-level bootstrap CI.
        A uniform-attribution model scores the band's share of the mel axis (55.5%) on
        both groups and a difference of zero.

    (b) TILING CONSISTENCY. Every ICBHI cycle is shorter than the 8 s model input
        (100% of them; median 2.42 s), and the pipeline wrap-pads by TILING the cycle
        until it fills 8 s. The input therefore contains the same acoustic content
        repeated ~3x. A model reading acoustics should attribute near-identically to
        each repetition; a model keying on padding position will not. This is an
        artefact detector — it cannot show the model is right, only that it is
        consistent.

WHAT THIS IS NOT
    Evidence of clinical interpretability. This project's own results (N9: clinician vs
    ICBHI kappa 0.035 on crackles; N1: task adaptation degrades 13/14 concepts) say the
    labels these attributions are aligned to are themselves unreliable. Phrase every
    result as evidence about WHERE THE MODEL LOOKS, never as clinical validation.

RUN
    python m44_xai_best_model.py --audio_dir <ICBHI audio_and_txt_files>
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CKPT = os.path.join(REPO, "Asif's", "M22_v2", "Results", "best_model.pth")
PADDING = "wrap"      # overridden by --padding; must match how the model was trained
TAG = "M22_v2"        # overridden by --tag; names the output files
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]
LABEL_OF = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
N_MELS, N_FRAMES, SR = 128, 801, 16000
WHEEZE_BAND = (100.0, 1000.0)     # concept_extractors.py's wheeze definition


# --------------------------------------------------------------------- data
def corrected_test_index(audio_dir, split_file):
    """The corrected split's TEST side — patients 156/218 reassigned to train."""
    split = {}
    for line in open(split_file):
        t = line.replace("\t", " ").replace(",", " ").split()
        if len(t) >= 2 and t[1].lower() in ("train", "test"):
            split[t[0].replace(".wav", "")] = t[1].lower()
    pid = lambda s: int(s.split("_")[0])
    sides = {}
    for s, v in split.items():
        sides.setdefault(pid(s), set()).add(v)
    overlap = {p for p, v in sides.items() if len(v) > 1}
    split = {s: ("train" if pid(s) in overlap else v) for s, v in split.items()}

    rows = []
    for wav in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        if split.get(stem) != "test":
            continue
        txt = os.path.join(audio_dir, stem + ".txt")
        if not os.path.exists(txt):
            continue
        for line in open(txt):
            p = line.split()
            if len(p) >= 4:
                rows.append({"wav": wav, "stem": stem, "patient_id": pid(stem),
                             "start": float(p[0]), "end": float(p[1]),
                             "dur": float(p[1]) - float(p[0]),
                             "label": LABEL_OF[(int(p[2]), int(p[3]))]})
    return rows


def log_mel(wav, start, end):
    """M22's exact preprocessing — power_to_db(ref=max), per-sample min-max, wrap-pad."""
    import librosa
    n = SR * 8
    try:
        a, _ = librosa.load(wav, sr=SR, offset=start, duration=max(end - start, 0.05),
                            mono=True)
    except Exception as e:
        # A silent all-zero spectrogram here would be trained on and
        # scored as a real cycle. Fail instead of substituting
        # (Model_Training_Protocol.md section 1.2).
        raise RuntimeError("failed to load audio") from e
    if len(a) == 0:
        # Empty decode is a failed read, not a silent zero cycle.
        raise RuntimeError(f"empty audio decoded from {wav}")
    if len(a) < n:
        a = (np.tile(a, math.ceil(n / len(a)))[:n] if PADDING == "wrap"
             else np.pad(a, (0, n - len(a))))          # zero-pad variant
    else:
        a = a[:n]
    m = librosa.feature.melspectrogram(y=a, sr=SR, n_mels=N_MELS, n_fft=1024,
                                       hop_length=160, win_length=400, fmin=50,
                                       fmax=2000, power=2.0)
    lm = librosa.power_to_db(m, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    T = lm.shape[1]
    lm = np.pad(lm, ((0, 0), (0, N_FRAMES - T))) if T < N_FRAMES else lm[:, :N_FRAMES]
    return lm[None].astype(np.float32)


# --------------------------------------------------------------------- model
def load_model(device):
    import torch
    import torch.nn as nn
    import torchvision

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = raw["model_state"]
    # ablation checkpoints store the state under the same key, so this loads either
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.features = torchvision.models.mobilenet_v2(weights=None).features
            s.gap = nn.AdaptiveAvgPool2d((1, 1))
            s.dropout = nn.Dropout(0.3)
            s.classifier = nn.Linear(int(sd["classifier.weight"].shape[1]), 4)

        def forward(s, x):
            x = x.repeat(1, 3, 1, 1)
            x = (x - mean.to(x.device)) / std.to(x.device)
            return s.classifier(s.dropout(s.gap(s.features(x)).flatten(1)))

    m = Net()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "num_batches" not in k], f"missing {missing[:4]}"
    assert not unexpected, f"unexpected {unexpected[:4]}"
    return m.eval().to(device), int(raw.get("epoch", -1))


# --------------------------------------------------------------------- attribution
def grad_cam(model, x, cls, device):
    """Grad-CAM on the final conv block. Returns a (n_mels, n_frames) map in [0,1]."""
    import torch
    import torch.nn.functional as F

    acts, grads = {}, {}
    target = model.features[-1]
    h1 = target.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        xb = torch.as_tensor(x, device=device).unsqueeze(0).requires_grad_(True)
        out = model(xb)
        model.zero_grad(set_to_none=True)
        out[0, cls].backward()
        a, g = acts["a"], grads["g"]                       # (1,C,h,w)
        w = g.mean(dim=(2, 3), keepdim=True)               # channel importance
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(N_MELS, N_FRAMES), mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def occlusion(model, x, cls, device, pf=16, pt=80, stride_f=8, stride_t=40):
    """Model-agnostic cross-check: slide a zero patch, record the drop in p(cls)."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        base = F.softmax(model(torch.as_tensor(x, device=device).unsqueeze(0)), 1)[0, cls].item()
    heat = np.zeros((N_MELS, N_FRAMES), np.float32)
    cnt = np.zeros_like(heat) + 1e-8
    batch, coords = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            p = F.softmax(model(torch.stack(batch).to(device)), 1)[:, cls].cpu().numpy()
        for (f0, t0), pv in zip(coords, p):
            heat[f0:f0 + pf, t0:t0 + pt] += (base - pv)
            cnt[f0:f0 + pf, t0:t0 + pt] += 1
        batch.clear(); coords.clear()

    import torch as _t
    for f0 in range(0, N_MELS - pf + 1, stride_f):
        for t0 in range(0, N_FRAMES - pt + 1, stride_t):
            xm = _t.as_tensor(x).clone()
            xm[:, f0:f0 + pf, t0:t0 + pt] = 0.0
            batch.append(xm); coords.append((f0, t0))
            if len(batch) >= 64:
                flush()
    flush()
    h = heat / cnt
    h = h - h.min()
    return h / (h.max() + 1e-8)


# --------------------------------------------------------------------- measures
def band_mass(cam, mel_f, lo, hi):
    """Share of attribution mass inside a frequency band."""
    m = (mel_f >= lo) & (mel_f <= hi)
    tot = cam.sum()
    return float(cam[m].sum() / tot) if tot > 0 else float("nan")


def tiling_consistency(cam, dur_s):
    """Correlation between attribution on consecutive repetitions of the tiled cycle.

    The cycle occupies `dur_s` seconds and is repeated to fill 8 s. Two repetitions hold
    identical audio, so a model reading acoustics should attribute to them alike.
    """
    frames = int(round(dur_s * SR / 160))
    if frames < 40 or frames * 2 > N_FRAMES:
        return None
    prof = cam.mean(axis=0)                       # attribution over time
    a, b = prof[:frames], prof[frames:2 * frames]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def padding_attention(cam, dur_s):
    """Share of attribution mass falling BEYOND the first (real) cycle.

    Every ICBHI cycle is shorter than the 8 s input, so everything after `dur_s` is
    padding — tiled repeats under `wrap`, silence under `zero`. This is the one measure
    that compares the two schemes directly: how much of the model's attention lands on
    material it manufactured rather than recorded. A model reading only the real cycle
    scores ~0; uniform attribution scores 1 - dur_s/8.
    """
    cut = int(round(dur_s * SR / 160))
    if cut < 10 or cut >= N_FRAMES:
        return None, None
    prof = cam.sum(axis=0)
    tot = prof.sum()
    if tot <= 0:
        return None, None
    return float(prof[cut:].sum() / tot), float(1.0 - cut / N_FRAMES)


def patient_bootstrap_diff(vals_a, pid_a, vals_b, pid_b, n_boot=2000, seed=42):
    """CI on mean(a) - mean(b), resampling patients within each group."""
    rng = np.random.default_rng(seed)
    ia = {p: np.flatnonzero(pid_a == p) for p in np.unique(pid_a)}
    ib = {p: np.flatnonzero(pid_b == p) for p in np.unique(pid_b)}
    ua, ub = list(ia), list(ib)
    out = []
    for _ in range(n_boot):
        sa = np.concatenate([ia[p] for p in rng.choice(ua, len(ua), replace=True)])
        sb = np.concatenate([ib[p] for p in rng.choice(ub, len(ub), replace=True)])
        out.append(float(np.mean(vals_a[sa]) - np.mean(vals_b[sb])))
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((np.array(out) <= 0).mean(), (np.array(out) >= 0).mean())
    return (round(float(np.mean(vals_a) - np.mean(vals_b)), 4),
            [round(float(lo), 4), round(float(hi), 4)], round(float(min(p, 1.0)), 4))


# --------------------------------------------------------------------- figure
def panel_figure(examples, mel_f, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(examples)
    fig, ax = plt.subplots(n, 3, figsize=(15, 3.1 * n), squeeze=False)
    ext = [0, 8, mel_f[0], mel_f[-1]]
    for r, e in enumerate(examples):
        ok = e["true"] == e["pred"]
        ax[r, 0].imshow(e["spec"], origin="lower", aspect="auto", cmap="magma", extent=ext)
        ax[r, 0].set_ylabel(f"{CLASSES[e['true']]}\n{'correct' if ok else 'MISCLASSIFIED'}",
                            color="black" if ok else "crimson", fontsize=9)
        ax[r, 0].set_title("log-mel" if r == 0 else "", fontsize=10)

        ax[r, 1].imshow(e["spec"], origin="lower", aspect="auto", cmap="gray", extent=ext)
        ax[r, 1].imshow(e["cam"], origin="lower", aspect="auto", cmap="jet", alpha=0.5,
                        extent=ext)
        ax[r, 1].set_title("Grad-CAM" if r == 0 else "", fontsize=10)

        ax[r, 2].imshow(e["occ"], origin="lower", aspect="auto", cmap="viridis", extent=ext)
        ax[r, 2].set_title("occlusion sensitivity" if r == 0 else "", fontsize=10)

        for c in range(3):
            ax[r, c].set_xlabel("time (s)" if r == n - 1 else "")
            if c and r != n - 1:
                ax[r, c].set_xticks([])
        # mark the wheeze band and the end of the first (un-tiled) repetition
        for c in (1, 2):
            ax[r, c].axhline(WHEEZE_BAND[0], color="w", ls=":", lw=.8)
            ax[r, c].axhline(WHEEZE_BAND[1], color="w", ls=":", lw=.8)
            ax[r, c].axvline(e["dur"], color="cyan", ls="--", lw=1)
        ax[r, 0].text(0.02, 0.92, f"pred {CLASSES[e['pred']]} (p={e['p']:.2f})",
                      transform=ax[r, 0].transAxes, color="w", fontsize=8)
    fig.suptitle("M44 — Grad-CAM and occlusion sensitivity, M22_v2 on the corrected split\n"
                 "dotted = 100–1000 Hz wheeze band · dashed cyan = end of the first "
                 "repetition (everything right of it is wrap-padding)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=os.environ.get(
        "ICBHI_AUDIO_DIR", r"C:\Users\Barshon\Desktop\ICBHI_final_database"))
    ap.add_argument("--split_file",
                    default=os.path.join(REPO, "Asif's", "ICBHI_challenge_train_test.txt"))
    ap.add_argument("--n_analyse", type=int, default=400,
                    help="cycles used for the quantitative measures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None, help="analyse a different checkpoint")
    ap.add_argument("--padding", choices=["wrap", "zero"], default="wrap",
                    help="MUST match how that checkpoint was trained")
    ap.add_argument("--tag", default="M22_v2", help="names the output files")
    args = ap.parse_args()
    global CKPT, PADDING, TAG
    if args.ckpt:
        CKPT = args.ckpt
    PADDING, TAG = args.padding, args.tag

    import torch
    import torch.nn.functional as F
    import librosa

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mel_f = librosa.mel_frequencies(n_mels=N_MELS, fmin=50, fmax=2000)
    band_share = float(((mel_f >= WHEEZE_BAND[0]) & (mel_f <= WHEEZE_BAND[1])).mean())

    print("=" * 76)
    print("M44 — INTERPRETABILITY ANALYSIS  (best model: M22_v2)")
    print("=" * 76)
    model, ep = load_model(dev)
    print(f"  device {dev} | checkpoint epoch {ep}")

    rows = corrected_test_index(args.audio_dir, args.split_file)
    print(f"  corrected test set: {len(rows)} cycles / "
          f"{len({r['patient_id'] for r in rows})} patients")
    print(f"  wheeze band {WHEEZE_BAND[0]:.0f}-{WHEEZE_BAND[1]:.0f} Hz = {band_share:.1%} "
          f"of the mel axis (this is the uniform-attribution baseline)")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(rows), min(args.n_analyse, len(rows)), replace=False)

    recs = []
    print(f"\n  computing Grad-CAM for {len(sel)} cycles ...")
    for n, i in enumerate(sel):
        r = rows[i]
        x = log_mel(r["wav"], r["start"], r["end"])
        with torch.no_grad():
            p = F.softmax(model(torch.as_tensor(x, device=dev).unsqueeze(0)), 1)[0]
        pred = int(p.argmax())
        cam = grad_cam(model, x, pred, dev)
        recs.append({"i": int(i), "true": r["label"], "pred": pred,
                     "p": float(p[pred]), "pid": r["patient_id"], "dur": r["dur"],
                     "band": band_mass(cam, mel_f, *WHEEZE_BAND),
                     "tile": tiling_consistency(cam, r["dur"]),
                     "pad_attn": padding_attention(cam, r["dur"])[0],
                     "pad_uniform": padding_attention(cam, r["dur"])[1]})
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{len(sel)}")

    band = np.array([r["band"] for r in recs])
    pid = np.array([r["pid"] for r in recs])
    true = np.array([r["true"] for r in recs])
    wz = np.isin(true, [2, 3])            # Wheeze or Both = wheeze-positive
    nw = ~wz

    print("\n" + "-" * 76)
    print("  (a) BAND POINTING — Grad-CAM mass inside 100–1000 Hz")
    print("-" * 76)
    print(f"    uniform-attribution baseline      {band_share:.4f}")
    print(f"    wheeze-positive cycles (n={wz.sum():3d})    {band[wz].mean():.4f}")
    print(f"    wheeze-negative cycles (n={nw.sum():3d})    {band[nw].mean():.4f}")
    d, ci, pv = patient_bootstrap_diff(band[wz], pid[wz], band[nw], pid[nw])
    print(f"    difference                        {d:+.4f}  CI95 {ci}  p={pv}")
    band_verdict = ("the model attends MORE to the wheeze band when a wheeze is present"
                    if ci[0] > 0 else
                    "the model attends LESS to the wheeze band when a wheeze is present"
                    if ci[1] < 0 else
                    "NOT shown to differ — attribution does not track the wheeze band")
    print(f"    verdict: {band_verdict}")

    tiles = np.array([r["tile"] for r in recs if r["tile"] is not None])
    print("\n" + "-" * 76)
    print("  (b) TILING CONSISTENCY — attribution across identical repeated audio")
    print("-" * 76)
    print(f"    n evaluable {len(tiles)}  mean r = {tiles.mean():.4f}  "
          f"median {np.median(tiles):.4f}")
    print(f"    fraction with r > 0.5: {(tiles > 0.5).mean():.1%}")
    tile_verdict = ("attribution is consistent across identical repetitions — it tracks "
                    "acoustic content, not padding position" if tiles.mean() > 0.5 else
                    "attribution DIFFERS across identical repeated audio — the model is "
                    "partly keying on position within the 8 s window, not on the sound")
    print(f"    verdict: {tile_verdict}")

    pa = np.array([r["pad_attn"] for r in recs if r["pad_attn"] is not None])
    pu = np.array([r["pad_uniform"] for r in recs if r["pad_attn"] is not None])
    print("\n" + "-" * 76)
    print(f"  (c) PADDING ATTENTION — attribution mass beyond the real cycle "
          f"[{PADDING}-padding]")
    print("-" * 76)
    print(f"    observed          {pa.mean():.4f}")
    print(f"    uniform baseline  {pu.mean():.4f}   (padding is this share of the input)")
    print(f"    ratio             {pa.mean()/pu.mean():.3f}  "
          f"({'over' if pa.mean() > pu.mean() else 'under'}-attends the padded region)")
    pad_verdict = ("the model attends to padding MORE than its share of the input — "
                   "attribution is partly driven by manufactured signal"
                   if pa.mean() > pu.mean() * 1.05 else
                   "the model attends to padding LESS than its share of the input"
                   if pa.mean() < pu.mean() * 0.95 else
                   "padding attention is proportional to its share of the input")
    print(f"    verdict: {pad_verdict}")

    # ---- figure: one correct example per class + one misclassification ----
    print("\n  building the panel figure ...")
    examples = []
    for c in range(4):
        cand = [r for r in recs if r["true"] == c and r["pred"] == c]
        if cand:
            examples.append(max(cand, key=lambda r: r["p"]))
    wrong = [r for r in recs if r["true"] != r["pred"]]
    if wrong:
        examples.append(max(wrong, key=lambda r: r["p"]))     # confidently wrong
    for e in examples:
        r = rows[e["i"]]
        x = log_mel(r["wav"], r["start"], r["end"])
        e["spec"] = x[0]
        e["cam"] = grad_cam(model, x, e["pred"], dev)
        e["occ"] = occlusion(model, x, e["pred"], dev)
    panel_figure(examples, mel_f, os.path.join(HERE, f"M44_xai_panels_{TAG}.png"))

    doc = {
        "meta": {"model_id": "M44", "model_name": "Interpretability analysis (XAI)",
                 "contributor": "OWMTL team",
                 "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
                 "explains_model": "M22_v2", "checkpoint_epoch": ep,
                 "notes": "RTK requirement 9. Grad-CAM + occlusion on the best model under "
                          "the corrected split."},
        "spec_deviation": {
            "requested": "pointing game against annotated crackle/wheeze event windows",
            "why_not_implementable": (
                "ICBHI annotates one row per respiratory CYCLE (start, end, crackle, wheeze) "
                "and marks whether the cycle contains an adventitious sound, not where "
                "inside it. The pipeline crops to exactly that window, so the annotated "
                "window IS the model input and the hit rate is 100% by construction. No "
                "finer annotation exists in the corpus."),
            "substituted": ["band pointing (100-1000 Hz wheeze band)",
                            "tiling consistency across wrap-padded repetitions"]},
        "dataset_info": {"dataset": "ICBHI_2017",
                         "split_method": "official_60_40_patient_independent_corrected",
                         "test_cycles": len(rows),
                         "cycles_analysed": int(len(sel))},
        "band_pointing": {
            "band_hz": list(WHEEZE_BAND),
            "uniform_baseline": round(band_share, 4),
            "wheeze_positive_mean": round(float(band[wz].mean()), 4),
            "wheeze_negative_mean": round(float(band[nw].mean()), 4),
            "n_wheeze_positive": int(wz.sum()), "n_wheeze_negative": int(nw.sum()),
            "difference": d, "difference_ci95_patient_bootstrap": ci, "p": pv,
            "verdict": band_verdict},
        "tiling_consistency": {
            "n_evaluable": int(len(tiles)),
            "mean_r": round(float(tiles.mean()), 4),
            "median_r": round(float(np.median(tiles)), 4),
            "fraction_above_0.5": round(float((tiles > 0.5).mean()), 4),
            "verdict": tile_verdict},
        "padding_attention": {
            "padding_scheme": PADDING,
            "observed_mean": round(float(pa.mean()), 4),
            "uniform_baseline_mean": round(float(pu.mean()), 4),
            "ratio": round(float(pa.mean() / pu.mean()), 4),
            "n": int(len(pa)),
            "verdict": pad_verdict,
            "note": "Share of Grad-CAM mass after the real cycle ends. Directly "
                    "comparable between the wrap-padded and zero-padded models."},
        "interpretation_guard": (
            "This is evidence about WHERE THE MODEL LOOKS, not clinical validation. This "
            "project measured clinician-vs-ICBHI agreement at kappa 0.035 on crackles (N9) "
            "and found task adaptation degrades 13/14 acoustic concepts (N1). Attributions "
            "aligned to labels of that reliability cannot certify clinical reasoning."),
        "figure": f"M44_xai_panels_{TAG}.png", "analysed_checkpoint": CKPT,
    }
    out = os.path.join(HERE, f"results_M44_{TAG}.json")
    json.dump(doc, open(out, "w"), indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
