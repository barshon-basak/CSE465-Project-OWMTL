#!/usr/bin/env python3
"""M50 - CirCor DigiScope as a far-OOD negative control for the best ICBHI model.

WHAT THIS IS NOT
    It is NOT a transfer test. CirCor is a phonocardiogram corpus: it contains heart sounds,
    and its labels are murmur present / absent / unknown. A crackle/wheeze classifier has
    zero label overlap with that, so scoring it with the ICBHI metric would produce no
    generalization number at all. `M49_cross_dataset/` is where transfer lives.

    Nothing here reads a CirCor label. Not the murmur annotation, not the segmentation .tsv,
    not the demographics. CirCor enters as one thing only: audio that is definitionally not a
    respiratory cycle. That is what makes the experiment valid rather than a category error,
    and it is why `circor_index` has no label argument to get wrong.

WHAT IT MEASURES
    Two questions the transfer numbers cannot answer:

    1. REJECTION. Can the frozen model tell that this is not its domain? The post-hoc energy
       score `-logsumexp(logits)` (Liu et al., the same scorer as `Asif's/M29`) is computed
       over the ICBHI corrected-split test cycles (known) and over CirCor windows (unknown),
       and the two distributions are separated by AUROC with a patient-level bootstrap.

    2. ABSTENTION. When it is wrong, does it know? Mean and median max-softmax, entropy, and
       the predicted-class histogram on heart sounds, against the same quantities on ICBHI.
       M49 found the model wrong and confident on SPRSound (max-softmax 0.9435 at accuracy
       0.2479). This is the extreme point of that curve.

THE CAVEAT THAT MUST TRAVEL WITH THE NUMBER
    CirCor is FAR-OOD - a different organ. The paper's existing open-set arm
    (`final paper/main.tex` section on extensions, AUROC 0.6466) uses NEAR-OOD unknowns:
    held-out lung disease classes, 19 patients. Rejecting a heartbeat is an easier task than
    rejecting an unseen lung pathology, so a CirCor AUROC does NOT replace, upgrade or
    supersede the 0.6466. It is a second, easier point on a difficulty axis, and it earns its
    place by establishing that the detector functions at all - which n=19 could not show.
    Both numbers get reported, with their unknown sets named. `openset.difficulty_note` in
    the results JSON carries this sentence so it cannot be dropped in transcription.

SAMPLING RATE
    CirCor is recorded at 4 kHz, so its Nyquist is 2,000 Hz - exactly the model's mel `fmax`.
    The top of the band arrives attenuated by the recorder's anti-aliasing, the same confound
    already recorded for HF_Lung_V1. For a rejection experiment this cuts the safe way: it
    makes the audio look MORE alien, so a high AUROC is partly bandwidth and must be read as
    an upper bound. It is written to `dataset_info.native_sample_rates_hz`.

USAGE
    python m50_openset.py --selftest
    python m50_openset.py --circor_root .../circor --icbhi_audio .../audio_and_txt_files
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import platform
import sys
import time

import numpy as np

HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())

# m49_xval carries the loader, the forward pass and the ICBHI gate; m45_ablation carries the
# mel parameters and the corrected split. Importing both is the point - a second copy of the
# spectrogram code would make every number here a measurement of the gap between two scripts.
for _cand in (HERE, os.path.abspath(os.path.join(HERE, "..", "M49_cross_dataset"))):
    if os.path.exists(os.path.join(_cand, "m49_xval.py")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
import m49_xval as X                    # noqa: E402
M45 = X.M45                             # noqa: E402

CLASSES = X.CLASSES
WINDOW_S = 8.0                          # the model's input length; CirCor rows are windows


# ================================================================== index
def circor_patient(stem):
    """CirCor names files `<patient_id>_<auscultation_location>.wav` (e.g. 2530_AV).

    One patient contributes up to five locations, so the leading field is the only correct
    bootstrap unit. Resampling recordings instead would treat five views of one child's
    chest as five independent observations - the same inflation the paper's audit section
    is about.
    """
    return stem.split("_")[0]


def circor_index(root, window_s=WINDOW_S, max_windows=None, min_tail_s=1.0, verbose=True):
    """Index CirCor into M45-shaped rows. No labels are read, because none are used.

    Each recording is cut into consecutive `window_s` windows. A trailing fragment shorter
    than `min_tail_s` is dropped rather than tiled up to a full window: tiling a 0.3 s
    remainder repeats it ~27 times and manufactures a periodic transient, which is exactly
    the artefact this experiment would then be measuring instead of the audio.
    """
    wavs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not wavs:
        raise FileNotFoundError(
            f"no .wav under {root!r}. Attach the Kaggle dataset "
            "'bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2', or download "
            "https://physionet.org/content/circor-heart-sound/1.0.3/")

    rows, rates, dropped = [], {}, {"short_recording": 0, "tail_fragment": 0}
    per_rec = []
    for wav in wavs:
        stem = os.path.splitext(os.path.basename(wav))[0]
        dur, native_sr = X._sf_info(wav)
        rates[native_sr] = rates.get(native_sr, 0) + 1
        if dur < min_tail_s:
            dropped["short_recording"] += 1
            continue
        n = 0
        t = 0.0
        while t < dur:
            end = min(t + window_s, dur)
            if end - t < min_tail_s:
                dropped["tail_fragment"] += 1
                break
            rows.append({"wav": wav, "stem": stem, "patient_id": circor_patient(stem),
                         "start": t, "end": end, "native_sr": native_sr})
            n += 1
            if max_windows and n >= max_windows:
                break
            t += window_s
        per_rec.append(n)

    pats = {r["patient_id"] for r in rows}
    stats = {"recordings_seen": len(wavs), "recordings_used": len(per_rec),
             "windows": len(rows), "patients": len(pats),
             "windows_per_recording_median": float(np.median(per_rec)) if per_rec else 0.0,
             "window_seconds": window_s, "native_sample_rates_hz": sorted(rates),
             "dropped": dropped,
             "labels_read": "none - CirCor enters as unlabelled out-of-domain audio"}
    if verbose:
        print(f"  CirCor {len(rows)} windows from {len(per_rec)} recordings, "
              f"{len(pats)} patients")
        print(f"    native sample rates: {dict(sorted(rates.items()))}")
        print(f"    windows per recording (median): "
              f"{stats['windows_per_recording_median']:.1f}")
        print(f"    dropped: {dropped}")
    return rows, stats


# ================================================================== scores
def energy(logits):
    """-logsumexp(logits), the post-hoc energy OOD score (Liu et al.).

    Identical in form to `score_energy` in `Asif's/M29/M29_openset_baselines.ipynb`, so the
    far-OOD number here and the near-OOD number in the paper are the same statistic on the
    same frozen backbone. HIGH means more out-of-distribution.
    """
    from scipy.special import logsumexp
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)


def max_softmax(logits):
    """MSP - the maximum softmax probability. HIGH means more confident, so more in-domain."""
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).max(1)


def entropy(logits):
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(1)


def _auroc(score_known, score_unknown):
    """AUROC for separating unknown (label 1) from known (label 0) by `score`."""
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(score_known)), np.ones(len(score_unknown))]
    s = np.r_[np.asarray(score_known), np.asarray(score_unknown)]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def auroc_grouped_ci(score_known, groups_known, score_unknown, groups_unknown,
                     n_boot=2000, seed=42):
    """AUROC with a patient-level bootstrap: patients are resampled, not windows.

    Both sides are resampled independently, each over its own patient set. Resampling windows
    would treat every 8-second slice of one child's recording as an independent observation
    and would report an interval several times too narrow.
    """
    sk, gk = np.asarray(score_known), np.asarray(groups_known)
    su, gu = np.asarray(score_unknown), np.asarray(groups_unknown)
    point = _auroc(sk, su)
    uk, uu = np.unique(gk), np.unique(gu)
    ik = {g: np.flatnonzero(gk == g) for g in uk}
    iu = {g: np.flatnonzero(gu == g) for g in uu}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        a = np.concatenate([ik[g] for g in rng.choice(uk, len(uk), replace=True)])
        b = np.concatenate([iu[g] for g in rng.choice(uu, len(uu), replace=True)])
        x = _auroc(sk[a], su[b])
        if x is not None and np.isfinite(x):
            v.append(x)
    ci = [round(float(z), 4) for z in np.percentile(v, [2.5, 97.5])] if v else [None, None]
    return (round(point, 4) if point is not None else None), ci


def confidence_block(logits):
    """What the model claims about audio it has never seen."""
    msp, ent = max_softmax(logits), entropy(logits)
    pred = np.asarray(logits).argmax(1)
    return {
        "mean_max_softmax": round(float(msp.mean()), 4),
        "median_max_softmax": round(float(np.median(msp)), 4),
        "fraction_above_0.90": round(float((msp > 0.90).mean()), 4),
        "fraction_above_0.99": round(float((msp > 0.99).mean()), 4),
        "mean_entropy": round(float(ent.mean()), 4),
        "max_entropy_possible": round(float(np.log(len(CLASSES))), 4),
        "predicted_class_histogram": {CLASSES[i]: int((pred == i).sum())
                                      for i in range(len(CLASSES))},
        "predicted_class_fraction": {CLASSES[i]: round(float((pred == i).mean()), 4)
                                     for i in range(len(CLASSES))},
    }


# ================================================================== driver
def run(circor_root, ckpt, out_dir, icbhi_audio=None, icbhi_split=None, limit=None,
        batch_size=64, n_boot=2000, window_s=WINDOW_S, max_windows=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, meta = X.load_checkpoint(ckpt)
    print(f"  checkpoint {meta['path']} (epoch {meta['epoch']}, "
          f"reported ICBHI {meta['reported_icbhi_score']:.4f})")

    if not (icbhi_audio and icbhi_split):
        raise ValueError(
            "icbhi_audio and icbhi_split are required. The known side of this comparison IS "
            "the ICBHI test partition - without it there is no distribution to be out of.")

    # The gate IS the known side. `return_details` hands back the pass it already made, so
    # the verified forward pass and the one feeding the AUROC are literally the same tensor
    # rather than two runs that could disagree - and the 2,636 cycles are decoded once.
    gate = X.verify_on_icbhi(model, cfg, meta, icbhi_audio, icbhi_split,
                             tol=float(os.environ.get("M50_GATE_TOL", "0.005")),
                             return_details=True, batch_size=batch_size, verbose=verbose)
    icbhi_ref = gate["score"]
    known_rows, lo_known = gate["rows"], gate["logits"]
    if limit:
        known_rows, lo_known = known_rows[:limit], lo_known[:limit]
    print(f"  known side: {len(known_rows)} ICBHI test cycles (from the gate's own pass)")
    g_known = np.array([str(r["patient_id"]) for r in known_rows])

    unknown_rows, index_stats = circor_index(circor_root, window_s=window_s,
                                             max_windows=max_windows, verbose=verbose)
    if limit:
        unknown_rows = unknown_rows[:limit]
    print(f"  unknown side: {len(unknown_rows)} CirCor windows")
    lo_unknown = X.predict(model, unknown_rows, cfg, batch_size=batch_size,
                           verbose=verbose)["logits"]
    g_unknown = np.array([str(r["patient_id"]) for r in unknown_rows])

    openset = {}
    for name, fn, direction in (("energy", energy, "higher = more OOD"),
                                ("msp", max_softmax, "lower = more OOD"),
                                ("entropy", entropy, "higher = more OOD")):
        sk, su = fn(lo_known), fn(lo_unknown)
        # every scorer is oriented so that HIGH means unknown before the AUROC is taken
        if direction.startswith("lower"):
            sk, su = -sk, -su
        auc, ci = auroc_grouped_ci(sk, g_known, su, g_unknown, n_boot=n_boot)
        openset[name] = {"auroc_unknown_vs_known": auc, "auroc_ci95": ci,
                         "auroc_ci95_unit": "patient", "orientation": direction,
                         "mean_known": round(float(np.mean(sk)), 4),
                         "mean_unknown": round(float(np.mean(su)), 4)}
        print(f"  {name:8s} AUROC {auc} {ci}")

    openset["difficulty_note"] = (
        "CirCor is FAR-OOD: a different organ. The paper's existing open-set arm reports "
        "AUROC 0.6466 on NEAR-OOD unknowns - held-out lung disease classes, 19 patients. "
        "Rejecting a heartbeat is an easier task than rejecting an unseen lung pathology, so "
        "this number does not replace, upgrade or supersede that one. It is a second, easier "
        "point on a difficulty axis, and it shows the detector functions at all, which n=19 "
        "could not.")
    openset["near_ood_reference"] = {"auroc": 0.6466, "unknown_patients": 19,
                                     "source": "Asif's/M29 + final paper, extensions section",
                                     "unknown_set": "held-out ICBHI disease classes"}

    doc = {
        "meta": {
            "model_id": "M50_CirCor_negative_control",
            "model_name": f"far-OOD negative control of {meta['path']} on CirCor DigiScope",
            "contributor": "OWMTL team",
            "date_completed": datetime.date.today().isoformat(),
            "is_augmented": False,
            "augmentation_method": "none (inference only)",
            "notes": (
                "NOT a transfer test and NOT a generalization number. No CirCor label is "
                "read; the corpus enters as unlabelled audio from an organ the model was "
                "never trained on. Reports rejection (open-set AUROC) and abstention "
                "(confidence), nothing else. The checkpoint passed the ICBHI gate first: "
                f"reproduced {icbhi_ref} against a stored {meta['reported_icbhi_score']:.4f}."),
        },
        "config": dict(cfg, window_seconds=window_s),
        "environment": {
            "platform": platform.platform(), "python_version": platform.python_version(),
            "pytorch_version": __import__("torch").__version__,
            "librosa_version": __import__("librosa").__version__,
        },
        "dataset_info": {
            "dataset": "CirCor_DigiScope",
            "role": "far_out_of_distribution_negative_control",
            "unit_of_analysis": f"{window_s:g}s window",
            "split_method": "no split - every available recording is unknown by construction",
            "known_side": "ICBHI 2017 corrected official 60/40 test partition",
            "test_samples": len(unknown_rows),
            "test_groups": int(len(np.unique(g_unknown))),
            "test_group_kind": "patient",
            "known_samples": len(known_rows),
            "known_groups": int(len(np.unique(g_known))),
            "native_sample_rates_hz": index_stats["native_sample_rates_hz"],
            "resampled_to_hz": M45.SR,
            "bandwidth_note": (
                "CirCor is 4 kHz, so its Nyquist is 2,000 Hz - exactly the model's mel fmax. "
                "The top of the band arrives attenuated. For a rejection experiment this "
                "cuts the safe way: it makes the audio look more alien, so the AUROC is an "
                "upper bound rather than an understatement."),
            "index_stats": index_stats,
        },
        "source_model": {"checkpoint": meta["path"], "epoch": meta["epoch"],
                         "reported_icbhi_score": meta["reported_icbhi_score"],
                         "icbhi_score_reproduced_here": icbhi_ref},
        "openset": openset,
        "confidence": {"circor": confidence_block(lo_unknown),
                       "icbhi_test": confidence_block(lo_known)},
    }
    out = os.path.join(out_dir, "results_M50_circor.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, allow_nan=False)
    np.save(os.path.join(out_dir, "logits_M50_circor.npy"), lo_unknown)
    np.save(os.path.join(out_dir, "logits_M50_icbhi_known.npy"), lo_known)
    plot_scores(energy(lo_known), energy(lo_unknown), lo_known, lo_unknown,
                os.path.join(out_dir, "M50_openset.png"))
    print_summary(doc)
    print(f"  wrote {out}")
    return doc


def plot_scores(e_known, e_unknown, lo_known, lo_unknown, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(e_known, bins=60, alpha=0.6, density=True, label="ICBHI test (known)")
    ax[0].hist(e_unknown, bins=60, alpha=0.6, density=True, label="CirCor (unknown)")
    ax[0].set_xlabel("energy score  $-\\log\\sum e^{z}$"); ax[0].set_ylabel("density")
    ax[0].set_title("(a) rejection: energy"); ax[0].legend()

    ax[1].hist(max_softmax(lo_known), bins=40, range=(0.25, 1), alpha=0.6, density=True,
               label="ICBHI test")
    ax[1].hist(max_softmax(lo_unknown), bins=40, range=(0.25, 1), alpha=0.6, density=True,
               label="CirCor")
    ax[1].set_xlabel("max softmax"); ax[1].set_title("(b) abstention: confidence")
    ax[1].legend()

    p = np.asarray(lo_unknown).argmax(1)
    ax[2].bar(CLASSES, [(p == i).sum() for i in range(len(CLASSES))])
    ax[2].set_ylabel("CirCor windows"); ax[2].set_title("(c) what it calls a heartbeat")
    ax[2].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def print_summary(doc):
    o, c = doc["openset"], doc["confidence"]
    d = doc["dataset_info"]
    print("\n" + "=" * 78)
    print(f"  M50 negative control  |  {d['known_samples']} known ICBHI cycles vs "
          f"{d['test_samples']} CirCor windows ({d['test_groups']} patients)")
    print("=" * 78)
    for k in ("energy", "msp", "entropy"):
        print(f"  {k:8s} AUROC {X._s(o[k]['auroc_unknown_vs_known'])} {o[k]['auroc_ci95']}")
    print(f"  near-OOD reference (paper): {o['near_ood_reference']['auroc']} "
          f"at n={o['near_ood_reference']['unknown_patients']} patients - HARDER task")
    print(f"  confidence on heart sounds : mean max-softmax "
          f"{c['circor']['mean_max_softmax']}, "
          f"{100 * c['circor']['fraction_above_0.90']:.1f}% above 0.90")
    print(f"  confidence on ICBHI test   : mean max-softmax "
          f"{c['icbhi_test']['mean_max_softmax']}")
    print(f"  predicted classes on CirCor: {c['circor']['predicted_class_fraction']}")
    print("=" * 78)


# ================================================================== selftest
def selftest(tmp=None):
    """Synthetic CirCor + synthetic logits through the real index and the real scorers.

    The failure modes here are silent ones: a patient id parsed from the wrong filename
    field pools five recordings of one child as five patients and narrows every interval; a
    window loop that runs past the end of a recording indexes rows that decode empty; an
    AUROC taken with the score oriented the wrong way reports 1 - AUROC, which looks like a
    result rather than a bug.
    """
    import shutil, tempfile
    tmp = tmp or tempfile.mkdtemp()
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {got} (want {want})")

    root = os.path.join(tmp, "circor")
    os.makedirs(root, exist_ok=True)
    # one patient, three locations; 20 s each at 4 kHz -> 2 full 8 s windows + 4 s tail
    for loc in ("AV", "MV", "TV"):
        X._write_wav(os.path.join(root, f"2530_{loc}.wav"), 20.0, 4000, freq=110.0)
    # a second patient with a 3 s recording -> one short window, no tail fragment
    X._write_wav(os.path.join(root, "9999_PV.wav"), 3.0, 4000, freq=95.0)

    rows, stats = circor_index(root, verbose=False)
    chk("patient id from the leading field", sorted({r["patient_id"] for r in rows}),
        ["2530", "9999"])
    chk("three locations pool into one patient",
        len({r["stem"] for r in rows if r["patient_id"] == "2530"}), 3)
    chk("20 s -> 8 + 8 + 4 windows", len([r for r in rows if r["stem"] == "2530_AV"]), 3)
    chk("no window starts past the end of its recording",
        all(r["start"] < X._sf_info(r["wav"])[0] for r in rows), True)
    chk("no label was read", stats["labels_read"].startswith("none"), True)
    chk("native rate recorded", stats["native_sample_rates_hz"], [4000])

    # scorers, on logits whose answer is known by construction
    rng = np.random.default_rng(0)
    conf = rng.normal(0, 1, (200, 4)) + np.array([8.0, 0, 0, 0])    # peaky  -> low energy
    flat = rng.normal(0, 1, (200, 4)) * 0.01                        # flat   -> high energy
    chk("energy separates peaky from flat", bool(energy(conf).mean() < energy(flat).mean()),
        True)
    chk("msp higher on peaky logits",
        bool(max_softmax(conf).mean() > max_softmax(flat).mean()), True)
    chk("entropy higher on flat logits", bool(entropy(conf).mean() < entropy(flat).mean()),
        True)
    auc, ci = auroc_grouped_ci(energy(conf), np.repeat(np.arange(20), 10),
                               energy(flat), np.repeat(np.arange(20), 10), n_boot=200)
    chk("AUROC is oriented so unknown-is-high gives ~1", bool(auc > 0.99), True)
    chk("patient bootstrap returns a real interval", bool(ci[0] is not None), True)

    # a scorer pointed the wrong way must not quietly return 1 - AUROC as a result
    bad, _ = auroc_grouped_ci(-energy(conf), np.repeat(np.arange(20), 10),
                              -energy(flat), np.repeat(np.arange(20), 10), n_boot=100)
    chk("reversed orientation is visibly wrong, not plausible", bool(bad < 0.01), True)

    cb = confidence_block(conf)
    chk("confidence block counts the predicted class",
        cb["predicted_class_histogram"]["Normal"], 200)
    chk("max entropy reported for 4 classes", cb["max_entropy_possible"], 1.3863)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--circor_root")
    ap.add_argument("--ckpt", default=os.path.abspath(
        os.path.join(HERE, "..", "Asif's", "M22_v2", "Results", "best_model.pth")))
    ap.add_argument("--icbhi_audio")
    ap.add_argument("--icbhi_split", default=os.path.abspath(
        os.path.join(HERE, "..", "Asif's", "ICBHI_challenge_train_test.txt")))
    ap.add_argument("--out_dir", default=HERE)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--max_windows", type=int)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.circor_root:
        ap.error("--circor_root is required (or use --selftest)")
    run(a.circor_root, a.ckpt, a.out_dir, a.icbhi_audio, a.icbhi_split, a.limit,
        a.batch_size, a.n_boot, max_windows=a.max_windows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
