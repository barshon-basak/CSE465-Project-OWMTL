#!/usr/bin/env python3
"""
M49 - external validation of the best ICBHI model on SPRSound (2022) and HF_Lung_V1.

WHAT THIS IS
    Inference only. The checkpoint trained on the corrected ICBHI official split is run,
    unchanged and unadapted, over two corpora it has never seen, and scored with the same
    official ICBHI metric. No fine-tuning, no threshold tuning, no target label used to fit
    anything in the headline block. It is the "does this transfer" number the paper lacks.

WHY IT IMPORTS m45_ablation INSTEAD OF RE-IMPLEMENTING THE PIPELINE
    `log_mel`, `official`, `LABEL_OF` and the corrected-split loader come from
    `Asif's/M45/m45_ablation.py` verbatim - the module that trained the checkpoint. A second
    implementation of the mel parameters would turn every cross-dataset delta into a
    measurement of the gap between two scripts. The one thing that IS copied is the `Net`
    class, because it is defined inside `run_row` and cannot be imported; `load_state_dict`
    runs with strict=True so a structural drift raises rather than silently half-loading.

THE CHECK THAT MAKES THE REST BELIEVABLE
    `verify_on_icbhi()` re-scores the checkpoint on the 2,636 ICBHI test cycles and asserts
    the result equals the score stored inside the checkpoint (M22_v2: 0.5602, P3: 0.5764).
    That proves this file's preprocessing, channel expansion and ImageNet normalisation are
    the training run's, so a low external score reads as domain shift rather than as a bug
    in the harness. It runs BEFORE any external number is computed, and a mismatch aborts.

TAXONOMY MAPPING - THE ONE PLACE THIS EVALUATION CAN GO WRONG QUIETLY
    Neither corpus uses ICBHI's four labels, so both are mapped. Every mapping is spelled
    out below and copied into the results JSON, because a reader cannot check a
    cross-dataset score without it. An unknown label string RAISES with the offending
    vocabulary listed - it is never bucketed into Normal, which would inflate Sp.

    SPRSound event -> ICBHI        HF_Lung_V1 -> ICBHI
      Normal              Normal     I / E             phase, builds the cycle
      Fine/Coarse Crackle Crackle    D                 Crackle bit
      Wheeze / Rhonchi /  Wheeze     Wheeze / Stridor  Wheeze bit
        Stridor    (= CAS)             / Rhonchi (= CAS)
      Wheeze+Crackle      Both       both bits set     Both

    Rhonchi and stridor go to Wheeze because both source taxonomies define them as
    continuous adventitious sounds (CAS), which is the class ICBHI calls Wheeze;
    HF_Lung_V1's own paper pools W, S and R into CAS for exactly this reason.

UNIT OF ANALYSIS
    ICBHI scores respiratory cycles. SPRSound's event annotations are cycle-like segments
    and are the headline unit. HF_Lung_V1 annotates breath phases, so an inhalation and the
    exhalation that follows it are paired into one cycle - the ICBHI definition. SPRSound
    record-level labels are a secondary row on a DIFFERENT unit (a whole ~9 s recording
    truncated to the model's 8 s window) and are not comparable to the event-level number.

CONFIDENCE INTERVALS
    SPRSound filenames carry a patient ID, so its CI is a patient-level bootstrap, the same
    estimator the paper uses. HF_Lung_V1 filenames carry a timestamp and no patient ID, so
    its CI is a RECORDING-level bootstrap and is reported under that name. Calling it
    patient-level would overstate what the resampling controls for.

USAGE
    python m49_xval.py --selftest                       # synthetic corpora, no real data
    python m49_xval.py --dataset sprsound --root .../BioCAS2022   # --ckpt defaults to M22_v2
    python m49_xval.py --dataset hflung   --root .../HF_Lung_V1
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import platform
import re
import sys
import time

import numpy as np

# `__file__` is undefined when this source is exec'd in a notebook cell instead of being
# imported. On Kaggle the notebook writes this module into /kaggle/working, which is also
# the working directory, so cwd is the right answer there rather than merely a safe one.
HERE = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
        else os.getcwd())

# m45_ablation is the module that trained the checkpoint. On Kaggle the notebook writes it
# next to this file; locally it lives in the repo. Both are tried, and failing to find it is
# fatal - a fallback re-implementation is the exact drift this import exists to prevent.
for _cand in (HERE, os.path.abspath(os.path.join(HERE, "..", "Asif's", "M45"))):
    if os.path.exists(os.path.join(_cand, "m45_ablation.py")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
import m45_ablation as M45          # noqa: E402

CLASSES = M45.CLASSES               # ["Normal", "Crackle", "Wheeze", "Both"]
SR = M45.SR                         # 16000

# The committed P3 confusion matrix, used by the self-test to prove the metric being applied
# is the official one and not the inflated macro variant the project shipped before the audit.
P3_ICBHI_CM = [[1175, 231, 112, 42], [277, 308, 14, 18],
               [189, 40, 113, 31], [40, 15, 22, 9]]
P3_ICBHI_SCORE = 0.5764

# ICBHI's corrected-split TRAINING class balance, for the prior-correction diagnostic when the
# gate's own measurement is not carried over. Recomputed by the gate whenever it runs, and the
# results JSON records which of the two was used.
ICBHI_TRAIN_PRIOR = [0.5918, 0.2341, 0.1415, 0.0326]


# ================================================================== label taxonomies
def _norm(s):
    """Collapse whitespace and case so 'CAS & DAS' and 'cas  &  das' are one key."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


SPR_EVENT_TO_LABEL = {
    "normal": 0,
    "fine crackle": 1, "coarse crackle": 1, "crackle": 1,
    "wheeze": 2, "rhonchi": 2, "stridor": 2,
    "wheeze+crackle": 3, "wheeze + crackle": 3, "wheeze & crackle": 3,
    "wheeze and crackle": 3, "wheeze&crackle": 3,
}
SPR_RECORD_TO_LABEL = {"normal": 0, "das": 1, "cas": 2, "cas & das": 3, "cas&das": 3}
SPR_RECORD_EXCLUDE = {"poor quality"}   # unusable audio by the annotators' own judgement

# Rhonchi and stridor are the one judgement call in the mapping above. Both source taxonomies
# call them continuous adventitious sounds, which is the class ICBHI calls Wheeze, but ICBHI
# itself never annotated a low-pitched continuous sound or an upper-airway inspiratory one, so
# a reader can reasonably object that these rows are being scored against a class that was
# never trained to cover them. Rather than argue it, every row carries `strict_ok`: the STRICT
# arm drops the rows whose label depends on one of these tokens, the BROAD arm keeps them, and
# both are reported. Where they disagree, the disagreement is the finding.
SPR_STRICT_EXCLUDE = {"rhonchi", "stridor"}

HFL_PHASE = {"I", "E"}                          # inhalation / exhalation - build the cycle
HFL_CRACKLE = {"D"}                             # discontinuous adventitious sound
HFL_WHEEZE = {"Wheeze", "Stridor", "Rhonchi"}   # continuous adventitious sound (CAS)
HFL_KNOWN = HFL_PHASE | HFL_CRACKLE | HFL_WHEEZE

MAPPING_DOC = {
    "sprsound_event": {k: CLASSES[v] for k, v in SPR_EVENT_TO_LABEL.items()},
    "sprsound_record": {k: CLASSES[v] for k, v in SPR_RECORD_TO_LABEL.items()},
    "sprsound_record_excluded": sorted(SPR_RECORD_EXCLUDE),
    "hflung_phase_tokens": sorted(HFL_PHASE),
    "hflung_crackle_tokens": sorted(HFL_CRACKLE),
    "hflung_wheeze_tokens": sorted(HFL_WHEEZE),
    "strict_arm_excludes": sorted(SPR_STRICT_EXCLUDE),
    "rationale": ("Rhonchi and stridor are continuous adventitious sounds in both source "
                  "taxonomies and map to ICBHI's Wheeze class; HF_Lung_V1's own paper pools "
                  "W/S/R into CAS. That is the BROAD arm. The STRICT arm drops every row "
                  "whose label depends on one of those two tokens, because ICBHI never "
                  "annotated a low-pitched continuous sound or an upper-airway inspiratory "
                  "one. Both are reported; the gap between them prices the judgement call."),
}


# ================================================================== SPRSound
def _sf_info(path):
    """Header-only read: duration and native sample rate, without decoding the audio."""
    import soundfile as sf
    i = sf.info(path)
    return float(i.duration), int(i.samplerate)


def sprsound_index(root, level="event", min_event_s=0.05, verbose=True):
    """Index SPRSound into M45-shaped rows: wav / start / end / label / patient_id.

    `root` should be the BioCAS2022 directory (the 2022 release). The walk is recursive and
    pairing is by filename stem, not by directory, because test2022_json/ nests
    intra_test_json/ and inter_test_json/ while test2022_wav/ is flat.
    """
    wavs, jsons = {}, {}
    for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        wavs[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        jsons[os.path.splitext(os.path.basename(p))[0]] = p
    paired = sorted(set(wavs) & set(jsons))
    if not paired:
        raise FileNotFoundError(
            f"no wav/json pairs under {root!r}. Point --root at the BioCAS2022 directory "
            "(it must contain train2022_wav/ and train2022_json/).")

    rows, seen_types = [], {}
    dropped = {"short_or_reversed": 0, "past_eof": 0, "excluded_record_label": 0,
               "no_annotation": 0}
    for stem in paired:
        wav = wavs[stem]
        try:
            doc = json.load(open(jsons[stem], encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"unreadable annotation {jsons[stem]}") from e
        dur, native_sr = _sf_info(wav)
        pid = stem.split("_")[0]        # <patient>_<age>_<gender>_<location>_<record no>
        jp = jsons[stem].replace("\\", "/")
        subset = ("inter_test" if "inter_test" in jp else
                  "intra_test" if "intra_test" in jp else
                  "test" if "/test" in jp else "train")

        if level == "record":
            raw = _norm(doc.get("record_annotation", ""))
            seen_types[raw] = seen_types.get(raw, 0) + 1
            if raw in SPR_RECORD_EXCLUDE:
                dropped["excluded_record_label"] += 1
                continue
            if raw not in SPR_RECORD_TO_LABEL:
                if not raw:
                    dropped["no_annotation"] += 1
                    continue
                raise ValueError(f"unmapped SPRSound record label {raw!r} in {stem}. "
                                 f"Known: {sorted(SPR_RECORD_TO_LABEL)}")
            # The record taxonomy is already pooled into CAS/DAS by the annotators, so no
            # rhonchi/stridor row exists to drop and the two arms are identical here. The flag
            # is still written, because a scorer that had to special-case one level would be a
            # place for the two arms to diverge for a reason other than the taxonomy.
            rows.append({"wav": wav, "stem": stem, "patient_id": pid, "subset": subset,
                         "start": 0.0, "end": dur, "label": SPR_RECORD_TO_LABEL[raw],
                         "native_sr": native_sr, "strict_ok": True})
            continue

        events = doc.get("event_annotation") or []
        if not events:
            dropped["no_annotation"] += 1
            continue
        for ev in events:
            raw = _norm(ev.get("type", ""))
            seen_types[raw] = seen_types.get(raw, 0) + 1
            if raw not in SPR_EVENT_TO_LABEL:
                raise ValueError(
                    f"unmapped SPRSound event label {raw!r} in {stem}. "
                    f"Known: {sorted(SPR_EVENT_TO_LABEL)}. Bucketing an unknown class into "
                    "Normal would silently inflate specificity.")
            s, e = float(ev["start"]) / 1000.0, float(ev["end"]) / 1000.0   # ms -> s
            if e - s < min_event_s:
                dropped["short_or_reversed"] += 1
                continue
            if s >= dur:
                dropped["past_eof"] += 1
                continue
            rows.append({"wav": wav, "stem": stem, "patient_id": pid, "subset": subset,
                         "start": s, "end": min(e, dur),
                         "label": SPR_EVENT_TO_LABEL[raw], "native_sr": native_sr,
                         "strict_ok": raw not in SPR_STRICT_EXCLUDE})

    if verbose:
        print(f"  SPRSound[{level}] {len(rows)} rows from {len(paired)} recordings, "
              f"{len({r['patient_id'] for r in rows})} patients")
        print(f"    label vocabulary seen: {dict(sorted(seen_types.items()))}")
        print(f"    dropped: {dropped}")
    return rows, {"recordings_paired": len(paired), "label_vocabulary": seen_types,
                  "dropped": dropped}


# ================================================================== HF_Lung_V1
def _hfl_time(s):
    """Seconds from '1.5' or from 'HH:MM:SS.mmm'. Both forms appear in the wild."""
    if ":" in s:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(s)


def _overlap(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def hflung_group(stem):
    """Cluster the segments cut from one original recording.

    HF_Lung_V1 has no patient ID in the filename. `trunc_yyyy-mm-dd-HH-MM-ss-LX_N` is the
    Nth 15 s slice at auscultation site LX of ONE session, so all of its slices must land on
    the same side of a bootstrap resample or the interval comes out optimistic.
    """
    m = re.match(r"^(trunc_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", stem)
    return m.group(1) if m else stem


def hflung_parse_label(path):
    """`<label> <start> <end>`, whitespace-delimited, one event per line."""
    out = []
    for ln, raw in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        p = raw.split()
        if not p:
            continue
        if len(p) < 3:
            raise ValueError(f"{path}:{ln}: expected '<label> <start> <end>', got {raw!r}")
        out.append((p[0], _hfl_time(p[1]), _hfl_time(p[2])))
    return out


def hflung_cycles(events, pair_gap_s=1.0):
    """Pair each inhalation with the exhalation that follows it into one ICBHI-style cycle.

    Returns (start, end, phases) where `phases` is "I+E", "I" or "E".

    An unpaired phase becomes a cycle on its own rather than being discarded: dropping the
    lone phases would throw away the breaths at the clip boundaries, which is exactly where a
    15 s window cuts a cycle in half, and that loss is not random with respect to the label.

    It is also not a rare edge case. HF_Lung_V1 carries 34,095 inhalation labels against
    18,349 exhalation labels, so most inhalations have no annotated exhalation to pair with
    and the majority of rows end up being single phases rather than whole cycles. That is a
    real difference from ICBHI's unit and it is counted into `index_stats.phase_composition`
    instead of being smoothed over - a row that is one inhalation is roughly half the duration
    of the ICBHI cycle the model was trained on.
    """
    ph = sorted([e for e in events if e[0] in HFL_PHASE], key=lambda e: (e[1], e[2]))
    out, i = [], 0
    while i < len(ph):
        lab, s, e = ph[i]
        if (i + 1 < len(ph) and lab == "I" and ph[i + 1][0] == "E"
                and ph[i + 1][1] - e <= pair_gap_s):
            out.append((s, ph[i + 1][2], "I+E"))
            i += 2
        else:
            out.append((s, e, lab))
            i += 1
    return out


def hflung_index(root, min_overlap_s=0.05, pair_gap_s=1.0, min_cycle_s=0.05, verbose=True):
    """Index HF_Lung_V1 into M45-shaped rows. `root` holds the extracted train/ and test/."""
    wavs = {}
    for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        wavs[os.path.splitext(os.path.basename(p))[0]] = p
    if not wavs:
        raise FileNotFoundError(f"no .wav under {root!r}. Extract train.7z / test.7z first.")

    rows, seen_types = [], {}
    phases = {"I+E": 0, "I": 0, "E": 0}
    dropped = {"no_label_file": 0, "no_phase_labels": 0, "short_or_reversed": 0,
               "past_eof": 0}
    for stem in sorted(wavs):
        wav = wavs[stem]
        lab = os.path.join(os.path.dirname(wav), stem + "_label.txt")
        if not os.path.exists(lab):
            dropped["no_label_file"] += 1
            continue
        events = hflung_parse_label(lab)
        for t, _, _ in events:
            seen_types[t] = seen_types.get(t, 0) + 1
        unknown = {t for t, _, _ in events} - HFL_KNOWN
        if unknown:
            raise ValueError(f"unmapped HF_Lung_V1 label(s) {sorted(unknown)} in {lab}. "
                             f"Known: {sorted(HFL_KNOWN)}.")

        cycles = hflung_cycles(events, pair_gap_s)
        if not cycles:
            dropped["no_phase_labels"] += 1
            continue
        adv = [e for e in events if e[0] not in HFL_PHASE]
        dur, native_sr = _sf_info(wav)
        wp = wav.replace("\\", "/")
        subset = "test" if "/test/" in wp else "train"
        for s, e, ph in cycles:
            if e - s < min_cycle_s:
                dropped["short_or_reversed"] += 1
                continue
            if s >= dur:
                dropped["past_eof"] += 1
                continue
            crk = any(_overlap(s, e, a, b) >= min_overlap_s
                      for t, a, b in adv if t in HFL_CRACKLE)
            whz = any(_overlap(s, e, a, b) >= min_overlap_s
                      for t, a, b in adv if t in HFL_WHEEZE)
            # A cycle is strict-unsafe only when its Wheeze bit rests ENTIRELY on a stridor or
            # rhonchi span. One that also overlaps a real Wheeze span is unaffected by the
            # judgement call, and a cycle with no Wheeze bit at all never depended on it.
            true_whz = any(_overlap(s, e, a, b) >= min_overlap_s
                           for t, a, b in adv if t == "Wheeze")
            phases[ph] += 1
            rows.append({"wav": wav, "stem": stem, "patient_id": hflung_group(stem),
                         "subset": subset, "start": s, "end": min(e, dur), "phases": ph,
                         "label": M45.LABEL_OF[(int(crk), int(whz))],
                         "native_sr": native_sr,
                         "strict_ok": (not whz) or true_whz})

    if verbose:
        print(f"  HF_Lung_V1 {len(rows)} cycles from {len(wavs)} recordings, "
              f"{len({r['patient_id'] for r in rows})} recording groups")
        print(f"    label vocabulary seen: {dict(sorted(seen_types.items()))}")
        print(f"    phase composition: {phases}  "
              f"({100.0 * phases['I+E'] / max(len(rows), 1):.1f}% are whole I+E cycles)")
        print(f"    dropped: {dropped}")
    return rows, {"recordings_seen": len(wavs), "label_vocabulary": seen_types,
                  "phase_composition": phases,
                  "phase_composition_note": (
                      "HF_Lung_V1 annotates far more inhalations than exhalations, so most "
                      "rows are a single phase rather than a whole I+E cycle. A single-phase "
                      "row is roughly half the duration of the ICBHI cycle the model was "
                      "trained on; see dataset_info.segment_stats for the tiling this "
                      "implies."),
                  "dropped": dropped}


# ================================================================== model
def build_model(cfg, device):
    """The M22_v2 / M45 network, copied verbatim from `m45_ablation.run_row`.

    It cannot be imported (it is a closure over `cfg` inside `run_row`), so it is copied -
    and `load_checkpoint` uses strict=True so any structural drift raises instead of quietly
    loading a subset of the weights. `embed` is the only addition: it is the forward pass
    split at the penultimate layer so the frozen-feature probe can reuse it.
    """
    import torch, torch.nn as nn, torchvision

    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.features = torchvision.models.mobilenet_v2(weights=None).features
            s.gap = nn.AdaptiveAvgPool2d((1, 1))
            s.dropout = nn.Dropout(cfg["dropout"])
            s.classifier = nn.Linear(1280, 4)
            s.norm = cfg["pretrained"]

        def embed(s, x):
            x = x.repeat(1, 3, 1, 1)
            if s.norm:
                x = (x - mean.to(x.device)) / std.to(x.device)
            return s.gap(s.features(x)).flatten(1)

        def forward(s, x):
            return s.classifier(s.dropout(s.embed(x)))

    return Net().to(device)


def load_checkpoint(path, device=None):
    """Return (model, cfg, meta). cfg = M45.BASE overlaid with the checkpoint's own cfg.

    The overlay is what lets one loader serve both checkpoints: M45's cfg names every
    preprocessing flag, M22_v2's predates three of them, and BASE holds the values those
    flags had when M22_v2 was trained (wrap padding, min-max on, no amplitude normalisation).
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in ck:
        raise ValueError(f"{path} has keys {list(ck)} - expected a dict with 'model_state'")
    cfg = dict(M45.BASE)
    cfg.update(ck.get("cfg", {}))
    # A checkpoint trained on the published split verbatim leaks patients 156 and 218 across
    # train and test, so its ICBHI score is not comparable and an external delta measured
    # against it means nothing. The gate below cannot catch this on its own: it holds each
    # checkpoint to the score stored inside IT, and a leaking checkpoint reproduces its own
    # leaking score perfectly. M45 rows carry no split_method and are corrected by
    # construction, so an absent key passes.
    split = cfg.get("split_method")
    if split is not None and split != "official_60_40_patient_independent_corrected":
        raise ValueError(
            f"{os.path.basename(path)} was trained on split {split!r}, not the corrected "
            "patient-independent partition. Use Asif's/M22_v2/Results/best_model.pth "
            "(0.5602), not best_model_official.pth (0.5641, leaks patients 156 and 218).")
    model = build_model(cfg, device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    # Counted from the state_dict tensors, buffers included - the convention
    # `Asif's/audit/backfill_efficiency.py` used for every other results file in the repo
    # (2,263,160 for this architecture, not the 2,228,996 that `parameters()` returns).
    meta = {"path": os.path.basename(path), "row": ck.get("row"), "epoch": ck.get("epoch"),
            "reported_icbhi_score": ck.get("best_score"),
            "total_params": int(sum(int(v.numel()) for v in ck["model_state"].values()
                                    if hasattr(v, "numel")))}
    return model, cfg, meta


def predict(model, rows, cfg, device=None, batch_size=64, num_workers=None,
            want_features=False, verbose=True):
    """One forward pass over `rows`.

    Spectrograms are computed on the fly and not cached: a cross-dataset run is a single
    pass, so an M45-style disk cache would cost 2-5 GB of Kaggle working storage and buy
    nothing back.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    class DS(Dataset):
        def __len__(s):
            return len(rows)

        def __getitem__(s, i):
            r = rows[i]
            # M22_v2 and M45 both cached spectrograms as a float16 memmap, so the checkpoint
            # was trained AND scored on float16-quantised inputs. This path computes them on
            # the fly, so without the round-trip it feeds the model something the training
            # run never saw. The quantisation is ~2.4e-4 per bin, which is small - but the
            # gate reproduces 0.5585 against a stored 0.5602, a gap of about four cycles in
            # 2,636, and this is the one remaining named difference between the two paths.
            # (The librosa version is not it: SPRSound ran pinned at 0.10.2 and HF_Lung ran
            # unpinned at 0.11.0, and both produced exactly 0.5585.)
            x = M45.log_mel(r["wav"], r["start"], r["end"], cfg).astype(np.float16)
            return torch.from_numpy(x.astype(np.float32))

    dl = DataLoader(DS(), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, feats, t0 = [], [], time.time()
    with torch.no_grad():
        for bi, x in enumerate(dl):
            x = x.to(device)
            f = model.embed(x)
            logits.append(model.classifier(f).float().cpu().numpy())
            if want_features:
                feats.append(f.float().cpu().numpy())
            if verbose and (bi + 1) % 50 == 0:
                done = min((bi + 1) * batch_size, len(rows))
                print(f"      {done}/{len(rows)}  ({time.time() - t0:.0f}s)")
    out = {"logits": np.concatenate(logits), "seconds": time.time() - t0}
    if want_features:
        out["features"] = np.concatenate(feats)
    return out


# ================================================================== metrics
def _f(x, nd=4):
    """Round, or None if the value is not finite.

    `official()` returns NaN when a subset contains no Normal or no abnormal rows, and
    `json.dump` would happily write that as a bare `NaN`, which is not valid JSON and which
    every downstream reader would then either reject or misparse. A missing number has to
    look missing.
    """
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def group_bootstrap_ci(y, pred, groups, n_boot=2000, seed=42):
    """M45.patient_ci with the grouping unit named by the caller instead of assumed.

    A resample that happens to draw only Normal groups scores NaN and is skipped. If EVERY
    resample degenerates - which is what a `--limit` smoke run on a one-class slice does -
    the interval is `[None, None]` rather than a crash or, worse, a fabricated interval.
    `degenerate_resample_fraction` in the caller's block says how often it happened.
    """
    from sklearn.metrics import confusion_matrix
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    uq = np.unique(groups)
    ix = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        s = np.concatenate([ix[g] for g in rng.choice(uq, len(uq), replace=True)])
        x = M45.official(confusion_matrix(y[s], pred[s], labels=[0, 1, 2, 3]))[0]
        if np.isfinite(x):
            v.append(x)
    if not v:
        return [None, None]
    return [round(float(z), 4) for z in np.percentile(v, [2.5, 97.5])]


def binary_rescore(cm):
    """Rescore the SAME predictions as detect-only (any abnormal vs Normal).

    Sp is unchanged by construction; Se rises because a crackle called a wheeze now counts.
    The gap is the cost of sub-typing, the quantity the paper's error table reports
    in-domain, so the external and in-domain versions are directly comparable.
    """
    cm = np.asarray(cm, float)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    abn = cm[1:].sum()
    se = cm[1:, 1:].sum() / abn if abn else float("nan")
    return {"se": _f(se), "sp": _f(sp), "icbhi_score_official": _f((se + sp) / 2)}


def score_block(y, pred, groups, group_kind, n_boot=2000):
    from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    sc, se, sp = M45.official(cm)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3],
                                                   zero_division=0)
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    return {
        "icbhi_score_official": _f(sc),
        "icbhi_score_official_ci95": group_bootstrap_ci(y, pred, groups, n_boot),
        "icbhi_score_official_ci95_unit": group_kind,
        "icbhi_se_official": _f(se),
        "icbhi_sp_official": _f(sp),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision_macro": round(float(p.mean()), 4),
        "recall_macro": round(float(r.mean()), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "per_class": {CLASSES[i]: {"precision": round(float(p[i]), 4),
                                   "recall": round(float(r[i]), 4),
                                   "f1": round(float(f[i]), 4),
                                   "support": int(sup[i]),
                                   "predicted": int(cm[:, i].sum())} for i in range(4)},
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_normalized": np.round(rown, 4).tolist(),
        "label_distribution_true": np.bincount(np.asarray(y), minlength=4).tolist(),
        "label_distribution_pred": np.bincount(np.asarray(pred), minlength=4).tolist(),
        "binary_detection": binary_rescore(cm),
    }


# ================================================================== the ICBHI gate
def verify_on_icbhi(model, cfg, ckpt_meta, audio_dir, split_file, tol=1e-3,
                    return_details=False, **kw):
    """Re-score the checkpoint on ICBHI and require it to reproduce its own stored score.

    This is the gate. An external score produced by an unverified forward path is not a
    result, so a mismatch raises rather than warning.

    `return_details=True` hands back the test rows and the logits this pass already computed,
    for a caller that needs the ICBHI side as data rather than only as a verdict - M50 uses
    the same cycles as the known half of its open-set comparison. Returning them costs
    nothing and means the gate and that comparison come from one forward pass instead of two,
    so they cannot disagree.
    """
    from sklearn.metrics import confusion_matrix
    rows = M45.corrected_split_index(audio_dir, split_file)
    te = [r for r in rows if r["split"] == "test"]
    print(f"  ICBHI verification: {len(te)} test cycles, "
          f"{len({r['patient_id'] for r in te})} patients")
    y = np.array([r["label"] for r in te])
    logits = predict(model, te, cfg, **kw)["logits"]
    got = M45.official(confusion_matrix(y, logits.argmax(1), labels=[0, 1, 2, 3]))[0]
    exp = ckpt_meta.get("reported_icbhi_score")
    print(f"  reproduced {got:.4f} | checkpoint reports {exp:.4f} | tol {tol}")
    if exp is None or abs(got - exp) > tol:
        raise AssertionError(
            f"ICBHI reproduction failed: {got:.4f} vs {exp}. The forward path here does not "
            "match the training run - do NOT report any external number from it.")
    print("  PASS - forward path reproduces the training run.")
    score = round(float(got), 4)
    if return_details:
        # The train prior, not the test prior: prior-correction swaps the distribution the
        # model was FIT on for the target's, and the model was fit on the training split.
        tr = np.array([r["label"] for r in rows if r["split"] == "train"])
        prior = (np.bincount(tr, minlength=4) / max(len(tr), 1)).round(4).tolist()
        return {"score": score, "rows": te, "logits": logits, "y": y,
                "train_prior": prior, "n_train": int(len(tr))}
    return score


# ================================================================== optional probe
def feature_probe(features, y, groups, seed=42, n_splits=5):
    """Frozen-feature linear probe, grouped CV, every clip out-of-sample.

    Separates two failure modes one zero-shot number cannot tell apart: the representation
    carries nothing about this corpus, versus the representation is fine and only the
    ICBHI-fitted decision boundary fails to transfer. It uses target labels, so it is NOT a
    zero-shot number and lives in its own block.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    oof = np.full(len(y), -1)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(features, y, groups):
        # The scaler is inside the pipeline, so it is fitted on the training fold only and
        # the held-out fold never contributes to its statistics. Post-GAP activations are
        # all non-negative and vary by an order of magnitude across the 1,280 units, which
        # lbfgs converges on slowly and unevenly without it.
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        clf.fit(features[tr], y[tr])
        oof[te] = clf.predict(features[te])
    assert (oof >= 0).all(), "grouped CV left clips unassigned"
    cm = confusion_matrix(y, oof, labels=[0, 1, 2, 3])
    sc, se, sp = M45.official(cm)
    # SPRSound's Both class is thin (34 events in the full 2022 release, 4 in the test half),
    # and below n_splits the folds cannot be stratified on it - sklearn warns to stderr,
    # which nobody reads off a finished notebook. Put the number in the result instead.
    support = np.bincount(y, minlength=4)
    return {"icbhi_score_official": _f(sc),
            "icbhi_score_official_ci95": group_bootstrap_ci(y, oof, groups),
            "icbhi_se_official": _f(se), "icbhi_sp_official": _f(sp),
            "confusion_matrix_raw": cm.tolist(),
            "class_support": {CLASSES[i]: int(support[i]) for i in range(4)},
            "min_class_support": int(support.min()),
            "stratification_ok": bool(support.min() >= n_splits),
            "note": (f"frozen-backbone logistic probe, grouped {n_splits}-fold CV. Uses "
                     "target labels - not a zero-shot result, and not comparable to the "
                     "headline number. If stratification_ok is false, the rarest class has "
                     "fewer members than folds and its per-fold estimate is unstable.")}


# ================================================================== reporting
def segment_stats(rows, cfg):
    """How hard the 8 s window has to work on this corpus.

    Every segment shorter than the window is tiled with `np.tile`, which leaves a
    discontinuity at each seam - the artefact M44 measured and the P4 ablation row tested.
    ICBHI's cycles have a median of 2.42 s and are tiled about 3.3x; a corpus of shorter
    segments is tiled harder, so the model sees proportionally more seam and less acoustics.
    That is a confound on any cross-dataset comparison and belongs in the results file rather
    than in someone's memory.
    """
    d = np.array([r["end"] - r["start"] for r in rows], float)
    win = float(cfg["duration_s"])
    return {"segment_seconds_median": round(float(np.median(d)), 3),
            "segment_seconds_p5_p95": [round(float(np.percentile(d, 5)), 3),
                                       round(float(np.percentile(d, 95)), 3)],
            "segment_seconds_min_max": [round(float(d.min()), 3), round(float(d.max()), 3)],
            "model_window_seconds": win,
            "tiling_factor_median": round(float(win / np.median(d)), 2),
            "fraction_longer_than_window": round(float((d >= win).mean()), 4),
            "icbhi_reference_median_seconds": 2.42,
            "note": ("segments shorter than the window are cyclically tiled; a larger "
                     "tiling factor than ICBHI's ~3.3x means proportionally more seam "
                     "artefact per input.")}


# ================================================================== analysis arms
def trivial_baselines(y, groups, n_boot=500, seed=42):
    """What the corpus scores without a model. An external number cannot be read without them.

    `always Normal` is the one that matters: on this metric it scores exactly 0.50 by
    construction (Se 0, Sp 1), which is the floor any transfer number has to clear to mean
    anything. `prior-matched random` is the harder floor - it draws from the TARGET class
    distribution, so beating it means the model carries information about which cycle is
    which, not merely about how common each class is.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    prior = np.bincount(y, minlength=4) / max(len(y), 1)
    out = {}
    for name, pred in (
            ("always Normal", np.zeros_like(y)),
            ("always majority class", np.full_like(y, int(np.argmax(prior)))),
            ("uniform random", rng.integers(0, 4, len(y))),
            ("prior-matched random", rng.choice(4, len(y), p=prior)),
    ):
        blk = score_block(y, pred, groups, "group", n_boot=n_boot)
        out[name] = {"icbhi_score_official": blk["icbhi_score_official"],
                     "icbhi_se_official": blk["icbhi_se_official"],
                     "icbhi_sp_official": blk["icbhi_sp_official"],
                     "accuracy": blk["accuracy"]}
    out["_note"] = ("always-Normal scores 0.50 on this metric by construction and is the "
                    "floor. prior-matched random draws from the target prior, so it is the "
                    "floor that a model must clear to be carrying more than class frequency.")
    return out


def _softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def calibration_block(logits, y, n_bins=15):
    """Expected calibration error and friends: is the model wrong *confidently*?

    A model that fails while staying confident is worse than one that fails loudly, and for a
    paper about trustworthy evaluation that distinction is the point. `overconfidence` is mean
    confidence minus accuracy - positive means it claims more than it delivers.
    """
    p = _softmax(logits)
    conf, pred = p.max(1), p.argmax(1)
    y = np.asarray(y)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return {"ece": _f(ece), "mean_max_softmax": _f(conf.mean()),
            "median_max_softmax": _f(np.median(conf)),
            "fraction_above_0.90": _f((conf > 0.90).mean()),
            "mean_entropy": _f(ent.mean()),
            "max_entropy_possible": _f(np.log(4)),
            "accuracy": _f(acc), "overconfidence": _f(conf.mean() - acc)}


def prior_corrected(logits, source_prior, target_prior):
    """Re-decide each cycle after swapping the training prior for the target's.

    Subtracting log p_source and adding log p_target is the standard correction for a pure
    label-shift. It USES THE TARGET LABELS to build the target prior, so it is a diagnostic
    and never a zero-shot number - it lives in its own block and is labelled as one. What it
    answers is the first objection anyone raises: "the corpora have different class balance,
    so of course the score dropped." If the corrected score barely moves, they do not.
    """
    src = np.asarray(source_prior, dtype=np.float64)
    tgt = np.asarray(target_prior, dtype=np.float64)
    adj = np.asarray(logits, dtype=np.float64) - np.log(src + 1e-12) + np.log(tgt + 1e-12)
    return adj.argmax(1)


def analysis_arms(rows, y, logits, groups, group_kind, source_prior, icbhi_logits=None,
                  icbhi_y=None, n_boot=2000):
    """The four things a reviewer asks for that one zero-shot score cannot answer."""
    y = np.asarray(y)
    pred = np.asarray(logits).argmax(1)
    target_prior = (np.bincount(y, minlength=4) / max(len(y), 1)).round(4).tolist()

    strict = np.array([bool(r.get("strict_ok", True)) for r in rows])
    arms = {
        "zeroshot_4class_broad": dict(
            score_block(y, pred, groups, group_kind, n_boot=n_boot),
            n=int(len(y)),
            note="rhonchi and stridor counted as Wheeze - the headline arm"),
    }
    if strict.all():
        arms["zeroshot_4class_strict"] = {
            "identical_to_broad": True, "n": int(len(y)),
            "note": ("no row in this corpus/level depends on a rhonchi or stridor "
                     "annotation, so the strict and broad arms are the same rows.")}
    else:
        arms["zeroshot_4class_strict"] = dict(
            score_block(y[strict], pred[strict], groups[strict], group_kind, n_boot=n_boot),
            n=int(strict.sum()),
            n_dropped=int((~strict).sum()),
            note=("rows whose label depends on rhonchi or stridor are dropped, because "
                  "ICBHI never annotated either"))

    pc = prior_corrected(logits, source_prior, target_prior)
    arms["prior_corrected_4class_broad"] = dict(
        score_block(y, pc, groups, group_kind, n_boot=n_boot),
        n=int(len(y)),
        note=("DIAGNOSTIC, NOT A ZERO-SHOT NUMBER - it uses the target labels to build the "
              "target prior. It separates the part of the drop caused by class-balance shift "
              "from the part caused by the representation."))

    out = {"arms": arms,
           "baselines_on_target": trivial_baselines(y, groups),
           "source_class_prior": [round(float(v), 4) for v in source_prior],
           "target_class_prior": target_prior,
           "calibration": {"target": calibration_block(logits, y)}}
    if icbhi_logits is not None and icbhi_y is not None:
        out["calibration"]["icbhi_source"] = calibration_block(icbhi_logits, icbhi_y)
        t, i = out["calibration"]["target"], out["calibration"]["icbhi_source"]
        if t["ece"] is not None and i["ece"] is not None:
            out["calibration"]["delta_ece"] = _f(t["ece"] - i["ece"])
            out["calibration"]["delta_overconfidence"] = _f(
                t["overconfidence"] - i["overconfidence"])
    out["calibration"]["_note"] = (
        "A model that fails while staying confident is worse than one that fails loudly. "
        "overconfidence is mean max-softmax minus accuracy.")
    return out


def make_results(model_id, dataset, level, rows, y, pred, cfg, ckpt_meta, index_stats,
                 icbhi_reference, seconds, extra=None):
    import torch, librosa
    groups = np.array([r["patient_id"] for r in rows])
    group_kind = "patient" if dataset == "SPRSound" else "recording_group"
    doc = {
        "meta": {
            "model_id": model_id,
            "model_name": f"external validation of {ckpt_meta['path']} on {dataset}",
            "contributor": "OWMTL team",
            "date_completed": datetime.datetime.now().strftime("%Y-%m-%d"),
            "is_augmented": False,
            "augmentation_method": "none (inference only)",
            "notes": ("Zero-shot cross-dataset evaluation. No fine-tuning, no threshold "
                      "tuning, no target label used to fit anything in best_metrics. The "
                      "checkpoint reproduced its own ICBHI score before this ran."),
        },
        "config": dict(cfg),
        "environment": {"platform": platform.platform(),
                        "python_version": platform.python_version(),
                        "pytorch_version": torch.__version__,
                        "librosa_version": librosa.__version__,
                        "gpu_name": (torch.cuda.get_device_name(0)
                                     if torch.cuda.is_available() else "cpu")},
        "dataset_info": {
            "dataset": dataset,
            "unit_of_analysis": level,
            "split_method": "external_validation_zero_shot_all_available_data",
            "train_samples": 0,
            "test_samples": int(len(y)),
            "test_groups": int(len(np.unique(groups))),
            "test_group_kind": group_kind,
            "subset_counts": {s: int(sum(1 for r in rows if r["subset"] == s))
                              for s in sorted({r["subset"] for r in rows})},
            "native_sample_rates_hz": sorted({int(r["native_sr"]) for r in rows}),
            "resampled_to_hz": SR,
            "index_stats": index_stats,
            "segment_stats": segment_stats(rows, cfg),
            "label_mapping": MAPPING_DOC,
        },
        "source_model": {
            "checkpoint": ckpt_meta["path"],
            "ablation_row": ckpt_meta.get("row"),
            "selected_epoch": ckpt_meta.get("epoch"),
            "icbhi_score_official_in_domain": icbhi_reference,
            "trained_on": "ICBHI_2017, official_60_40_patient_independent_corrected",
        },
        "best_epoch": {"epoch": ckpt_meta.get("epoch"),
                       "primary_metric": "icbhi_score_official",
                       "primary_metric_value": None},
        "best_metrics": score_block(y, pred, groups, group_kind),
        "efficiency": {
            "total_params": ckpt_meta.get("total_params"),
            "trainable_params": 0,
            "model_size_mb": (round(ckpt_meta["total_params"] * 4 / 1e6, 3)
                              if ckpt_meta.get("total_params") else None),
            "training_time_total_s": 0,
            "inference_time_total_s": round(seconds, 1),
            "inference_time_ms_per_sample": round(1000.0 * seconds / max(len(y), 1), 3),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu"),
        },
        "ablation": {
            "ablation_group": "external_validation",
            "ablation_role": "variant",
            "baseline_model_id": ckpt_meta.get("row") or "M22_v2",
            "variable_changed": f"evaluation corpus: ICBHI_2017 -> {dataset}",
            "variables_held_constant": ["weights", "preprocessing", "official metric",
                                        "decision rule: argmax"],
            "component_flags": {"has_sound_event_head": True, "has_disease_head": False,
                                "has_cross_task_consistency": False,
                                "has_cqkd_regularization": False,
                                "has_openmax_rejection": False, "owl_stage": 0,
                                "compression_clusters": None},
            "loss_weights": {"sound_event_weight": 1.0, "disease_weight": None,
                             "consistency_weight": None},
        },
        "training_history": [],
    }
    ext = doc["best_metrics"]["icbhi_score_official"]
    doc["best_epoch"]["primary_metric_value"] = ext
    doc["transfer"] = {
        "icbhi_in_domain": icbhi_reference,
        "external": ext,
        "delta": (round(ext - icbhi_reference, 4) if ext is not None else None),
        "chance_reference": 0.5,
        "note": ("0.50 is what a constant always-Normal predictor scores on this metric, so "
                 "it is the floor an external number has to clear to mean anything."),
    }
    if extra:
        doc.update(extra)
    return doc


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{rown[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if rown[i, j] > 0.5 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def _s(x, sign=False):
    """Format a metric that may legitimately be absent (see `_f`)."""
    if x is None:
        return "  n/a "
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def print_summary(doc):
    b, t, d = doc["best_metrics"], doc["transfer"], doc["dataset_info"]
    print("\n" + "=" * 78)
    print(f"  {doc['meta']['model_id']}  |  {d['dataset']}  |  unit: {d['unit_of_analysis']}")
    print("=" * 78)
    print(f"  n = {d['test_samples']} over {d['test_groups']} {d['test_group_kind']}s")
    print(f"  ICBHI official   {_s(b['icbhi_score_official'])} "
          f"{b['icbhi_score_official_ci95']}   "
          f"({b['icbhi_score_official_ci95_unit']}-level CI)")
    print(f"  Se {_s(b['icbhi_se_official'])}   Sp {_s(b['icbhi_sp_official'])}   "
          f"acc {b['accuracy']:.4f}   macro-F1 {b['f1_macro']:.4f}")
    print(f"  detect-only      {_s(b['binary_detection']['icbhi_score_official'])}  "
          f"(Se {_s(b['binary_detection']['se'])}, Sp unchanged)")
    print(f"  in-domain ICBHI  {_s(t['icbhi_in_domain'])}   ->   "
          f"delta {_s(t['delta'], sign=True)}   (0.50 = always-Normal)")
    ss = d.get("segment_stats")
    if ss:
        print(f"  segments         median {ss['segment_seconds_median']:.2f} s -> tiled "
              f"{ss['tiling_factor_median']:.2f}x into the {ss['model_window_seconds']:.0f} s "
              f"window (ICBHI median 2.42 s, ~3.3x)")
    print("  distribution over" + "".join(f"{c:>10s}" for c in CLASSES))
    print("    true         " + "".join(f"{v:10d}" for v in b["label_distribution_true"]))
    print("    predicted    " + "".join(f"{v:10d}" for v in b["label_distribution_pred"]))
    if "feature_probe" in doc:
        p = doc["feature_probe"]
        print(f"  frozen-feature probe (uses target labels): "
              f"{_s(p['icbhi_score_official'])} {p['icbhi_score_official_ci95']}")
    if b["icbhi_score_official"] is None:
        print("  !! the official score is undefined here - this slice has no Normal rows or "
              "no abnormal rows. Not a reportable result.")
    print("=" * 78)


# ================================================================== self-test
def _write_wav(path, seconds, sr, freq=440.0):
    import soundfile as sf
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


def selftest(tmp=None):
    """Build synthetic SPRSound and HF_Lung_V1 corpora and run the whole path over them.

    Nothing here touches a real dataset or a GPU. It exists because the two expensive
    failure modes of this file are silent: a taxonomy mapping that buckets a class into the
    wrong bin, and a cycle builder that produces the right NUMBER of rows with the wrong
    times. Both would yield a plausible score that is simply not the quantity claimed.
    """
    import shutil
    import tempfile
    ok = True
    tmp = tmp or tempfile.mkdtemp(prefix="m49_selftest_")

    # -- the metric applied is the official one, not the inflated macro variant
    got = M45.official(P3_ICBHI_CM)[0]
    print(f"  metric        P3 matrix -> {got:.4f} (want {P3_ICBHI_SCORE})")
    ok &= abs(got - P3_ICBHI_SCORE) < 1e-4

    # -- SPRSound: one record per event class, plus a Poor Quality that must be excluded
    spr = os.path.join(tmp, "BioCAS2022", "train2022_wav")
    sprj = os.path.join(tmp, "BioCAS2022", "train2022_json")
    os.makedirs(spr, exist_ok=True)
    os.makedirs(sprj, exist_ok=True)
    cases = ["Normal", "Fine Crackle", "Coarse Crackle", "Wheeze", "Rhonchi", "Stridor",
             "Wheeze+Crackle"]
    for i, typ in enumerate(cases):
        stem = f"9000000{i}_5.0_0_p1_{100 + i}"
        _write_wav(os.path.join(spr, stem + ".wav"), 9.2, 8000)
        json.dump({"record_annotation": "Normal",
                   "event_annotation": [{"start": "1000", "end": "2500", "type": typ},
                                        {"start": "3000", "end": "4200", "type": typ}]},
                  open(os.path.join(sprj, stem + ".json"), "w"))
    stem = "90000099_5.0_0_p1_199"
    _write_wav(os.path.join(spr, stem + ".wav"), 9.2, 8000)
    json.dump({"record_annotation": "Poor Quality", "event_annotation": []},
              open(os.path.join(sprj, stem + ".json"), "w"))

    rows, st = sprsound_index(os.path.join(tmp, "BioCAS2022"), "event", verbose=False)
    counts = np.bincount([r["label"] for r in rows], minlength=4).tolist()
    print(f"  sprsound      event labels {counts} (want [2, 4, 6, 2])")
    ok &= counts == [2, 4, 6, 2]
    ok &= all(abs((r["end"] - r["start"]) - (1.5 if r["start"] == 1.0 else 1.2)) < 1e-9
              for r in rows)
    ok &= len({r["patient_id"] for r in rows}) == 7
    ok &= st["dropped"]["short_or_reversed"] == 0
    rrows, _ = sprsound_index(os.path.join(tmp, "BioCAS2022"), "record", verbose=False)
    print(f"  sprsound      record rows {len(rrows)} (want 7 - Poor Quality excluded)")
    ok &= len(rrows) == 7

    # -- HF_Lung_V1: one file per target class, built from I/E phases + adventitious spans
    hfl = os.path.join(tmp, "HF_Lung_V1", "train")
    os.makedirs(hfl, exist_ok=True)
    specs = {
        "steth_20190101_00_00_01": (["I 1.0 2.0", "E 2.1 3.5"], 0),
        "steth_20190101_00_00_02": (["I 1.0 2.0", "E 2.1 3.5", "D 1.5 1.6"], 1),
        "steth_20190101_00_00_03": (["I 1.0 2.0", "E 2.1 3.5", "Wheeze 2.5 3.0"], 2),
        "steth_20190101_00_00_04": (["I 1.0 2.0", "E 2.1 3.5", "Rhonchi 2.5 3.0"], 2),
        "steth_20190101_00_00_05": (["I 1.0 2.0", "E 2.1 3.5", "D 1.5 1.6",
                                     "Stridor 2.5 3.0"], 3),
        # an adventitious span outside the cycle must NOT label it
        "steth_20190101_00_00_06": (["I 1.0 2.0", "E 2.1 3.5", "D 9.0 9.2"], 0),
    }
    for stem, (lines, _) in specs.items():
        _write_wav(os.path.join(hfl, stem + ".wav"), 15.0, 4000)
        open(os.path.join(hfl, stem + "_label.txt"), "w").write("\n".join(lines) + "\n")
    for n in (1, 2):        # two slices of one session - they must share a bootstrap group
        stem = f"trunc_2019-01-01-00-00-07-L1_{n}"
        _write_wav(os.path.join(hfl, stem + ".wav"), 15.0, 4000)
        open(os.path.join(hfl, stem + "_label.txt"), "w").write("I 1.0 2.0\nE 2.1 3.5\n")

    hrows, hst = hflung_index(os.path.join(tmp, "HF_Lung_V1"), verbose=False)
    by_stem = {r["stem"]: r for r in hrows}
    print(f"  hflung        {len(hrows)} cycles from 8 files (want 8 - one cycle each)")
    ok &= len(hrows) == 8
    for stem, (_, want) in specs.items():
        gotl = by_stem[stem]["label"]
        ok &= gotl == want
        if gotl != want:
            print(f"    MISMATCH {stem}: got {CLASSES[gotl]} want {CLASSES[want]}")
    c = by_stem["steth_20190101_00_00_01"]
    print(f"  hflung        I+E paired into [{c['start']}, {c['end']}] as {c['phases']!r} "
          "(want [1.0, 3.5] 'I+E')")
    ok &= abs(c["start"] - 1.0) < 1e-9 and abs(c["end"] - 3.5) < 1e-9 and c["phases"] == "I+E"
    # an inhalation with no exhalation must survive as its own row, tagged as one phase -
    # HF_Lung_V1 has nearly twice as many I labels as E labels, so this is the common case
    lone = hflung_cycles([("I", 1.0, 2.0), ("I", 5.0, 6.0), ("E", 6.1, 7.0)])
    print(f"  hflung        unpaired I kept: {lone} (want I, then I+E)")
    ok &= [x[2] for x in lone] == ["I", "I+E"] and abs(lone[1][1] - 7.0) < 1e-9
    g = {r["stem"]: r["patient_id"] for r in hrows}
    same = g["trunc_2019-01-01-00-00-07-L1_1"] == g["trunc_2019-01-01-00-00-07-L1_2"]
    print(f"  hflung        two slices of one session share a group: {same} (want True)")
    ok &= same and hst["dropped"]["no_label_file"] == 0

    # -- an unknown label must raise, never be bucketed into Normal
    bad = os.path.join(tmp, "bad", "train")
    os.makedirs(bad, exist_ok=True)
    _write_wav(os.path.join(bad, "steth_20190101_00_00_09.wav"), 15.0, 4000)
    open(os.path.join(bad, "steth_20190101_00_00_09_label.txt"), "w").write("Squawk 1 2\n")
    try:
        hflung_index(os.path.join(tmp, "bad"), verbose=False)
        print("  unknown label did NOT raise (want ValueError)")
        ok = False
    except ValueError:
        print("  unknown label raises as required")

    # -- the tensor the model actually receives
    cfg = dict(M45.BASE)
    cfg["ampnorm"] = True
    x = M45.log_mel(rows[0]["wav"], rows[0]["start"], rows[0]["end"], cfg)
    print(f"  log_mel       shape {x.shape} range [{x.min():.2f}, {x.max():.2f}] "
          "(want (1, 128, 801) inside [0, 1])")
    ok &= x.shape == (1, 128, 801) and x.min() >= -1e-6 and x.max() <= 1 + 1e-6

    # -- forward path
    try:
        import torch
        m = build_model(cfg, "cpu")
        with torch.no_grad():
            o = m(torch.from_numpy(np.stack([x, x])))
        print(f"  forward       {tuple(o.shape)} (want (2, 4))")
        ok &= tuple(o.shape) == (2, 4)
    except ImportError as e:                                  # torch absent locally
        print(f"  forward       SKIPPED ({e})")

    # -- the estimators
    y = np.array([0] * 40 + [1] * 20 + [2] * 20 + [3] * 20)
    grp = np.repeat(np.arange(20), 5)
    lo, hi = group_bootstrap_ci(y, y.copy(), grp, n_boot=200)
    print(f"  bootstrap     perfect predictions -> [{lo}, {hi}] (want [1.0, 1.0])")
    ok &= lo == 1.0 and hi == 1.0
    # An all-Normal slice makes the official score undefined. It must come back missing,
    # not as a crash and not as a fabricated interval.
    z = np.zeros(10, int)
    deg = group_bootstrap_ci(z, z.copy(), np.arange(10), n_boot=50)
    print(f"  bootstrap     one-class slice -> {deg} (want [None, None])")
    ok &= deg == [None, None] and _f(float("nan")) is None and _f(0.123456) == 0.1235
    br = binary_rescore([[8, 1, 1, 0], [2, 5, 3, 0], [1, 4, 5, 0], [0, 0, 0, 0]])
    print(f"  detect-only   Se {br['se']} (want 0.85 - cross-type errors now count)")
    ok &= abs(br["se"] - 0.85) < 1e-9

    # ---- analysis arms -------------------------------------------------------
    yb = np.array([0] * 60 + [1] * 25 + [2] * 10 + [3] * 5)
    gb = np.repeat(np.arange(20), 5)
    bl = trivial_baselines(yb, gb, n_boot=50)
    print(f"  baseline      always-Normal {bl['always Normal']['icbhi_score_official']} "
          "(want 0.5 exactly - Se 0, Sp 1 by construction)")
    ok &= bl["always Normal"]["icbhi_score_official"] == 0.5
    ok &= bl["always Normal"]["icbhi_se_official"] == 0.0
    print(f"  baseline      prior-matched random "
          f"{bl['prior-matched random']['icbhi_score_official']} (want below always-Normal)")
    ok &= bl["prior-matched random"]["icbhi_score_official"] < 0.5

    # calibration: right-and-confident is calibrated; wrong-and-confident is not
    right = np.full((100, 4), -6.0); right[:, 0] = 6.0
    cal_ok = calibration_block(right, np.zeros(100, int))
    cal_bad = calibration_block(right, np.ones(100, int))
    print(f"  calibration   confident+right ece {cal_ok['ece']}, overconf "
          f"{cal_ok['overconfidence']} (want ~0)")
    ok &= cal_ok["ece"] < 0.01 and abs(cal_ok["overconfidence"]) < 0.01
    print(f"  calibration   confident+wrong ece {cal_bad['ece']}, overconf "
          f"{cal_bad['overconfidence']} (want ~1 - this is the finding, not a bug)")
    ok &= cal_bad["ece"] > 0.99 and cal_bad["overconfidence"] > 0.99

    # prior correction: identical priors must not move a single prediction
    lg = np.random.default_rng(0).normal(0, 2, (200, 4))
    same = prior_corrected(lg, [.25, .25, .25, .25], [.25, .25, .25, .25])
    print(f"  prior corr.   identical priors change "
          f"{int((same != lg.argmax(1)).sum())} predictions (want 0)")
    ok &= bool((same == lg.argmax(1)).all())
    shifted = prior_corrected(lg, [.97, .01, .01, .01], [.01, .01, .01, .97])
    print(f"  prior corr.   shifting the prior to class 3 moves mass to it: "
          f"{int((shifted == 3).sum())} > {int((lg.argmax(1) == 3).sum())}")
    ok &= int((shifted == 3).sum()) > int((lg.argmax(1) == 3).sum())

    # strict/broad marking - the subtlest logic in the index builders
    spr_rows, _ = sprsound_index(os.path.join(tmp, "BioCAS2022"), level="event", verbose=False)
    by_type = {r["label"]: r for r in spr_rows}
    rho = [r for r in spr_rows if not r["strict_ok"]]
    print(f"  strict arm    SPRSound marks {len(rho)} rhonchi/stridor rows unsafe "
          "(want 4 - the synthetic corpus writes two events per record)")
    ok &= len(rho) == 4
    ok &= all(r["label"] == 2 for r in rho)
    hfl_rows, _ = hflung_index(os.path.join(tmp, "HF_Lung_V1"), verbose=False)
    unsafe = [r for r in hfl_rows if not r["strict_ok"]]
    wheezy = [r for r in hfl_rows if r["label"] in (2, 3)]
    print(f"  strict arm    HF_Lung marks {len(unsafe)} of {len(wheezy)} wheeze-bearing "
          "cycles unsafe (want only the stridor/rhonchi-only ones)")
    ok &= len(unsafe) < len(wheezy) and all(r["label"] in (2, 3) for r in unsafe)
    print(f"  strict arm    a Normal cycle is never strict-unsafe: "
          f"{all(r['strict_ok'] for r in hfl_rows if r['label'] == 0)}")
    ok &= all(r["strict_ok"] for r in hfl_rows if r["label"] == 0)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n  SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================== driver
def run(dataset, root, ckpt, out_dir, level="event", icbhi_audio=None, icbhi_split=None,
        limit=None, batch_size=64, n_boot=2000, probe=False, verbose=True, icbhi_ref=None,
        icbhi_gate=None):
    os.makedirs(out_dir, exist_ok=True)
    model, cfg, meta = load_checkpoint(ckpt)
    print(f"  checkpoint {meta['path']} (row {meta['row']}, epoch {meta['epoch']}, "
          f"reported ICBHI {meta['reported_icbhi_score']:.4f})")
    print(f"  preprocessing: n_mels={cfg['n_mels']} dur={cfg['duration_s']}s "
          f"pad={cfg['padding']} minmax={cfg['minmax']} ampnorm={cfg.get('ampnorm')} "
          f"bandpass={cfg.get('bandpass')} denoise={cfg.get('denoise')}")

    if icbhi_gate is not None:
        # The notebook runs the gate in its own cell so a failure stops there rather than
        # after a corpus has been indexed. Handing the dict back means the calibration
        # comparison and the source prior come from that same verified pass.
        icbhi_ref = icbhi_gate["score"]
        print(f"  in-domain reference {icbhi_ref:.4f} from the caller's gate "
              f"({len(icbhi_gate['y'])} ICBHI test cycles carried over).")
    elif icbhi_ref is not None:
        print(f"  in-domain reference {icbhi_ref:.4f} supplied by the caller "
              "(gate already passed this session).")
    elif icbhi_audio and icbhi_split:
        icbhi_gate = verify_on_icbhi(model, cfg, meta, icbhi_audio, icbhi_split,
                                     return_details=True, batch_size=batch_size)
        icbhi_ref = icbhi_gate["score"]
    else:
        icbhi_ref = round(float(meta["reported_icbhi_score"]), 4)
        print("  ICBHI verification SKIPPED - the in-domain reference is the checkpoint's "
              "own stored score, not a reproduction.")

    if dataset == "sprsound":
        rows, stats = sprsound_index(root, level=level)
        name, mid = "SPRSound", f"M49_SPRSound_{level}"
    elif dataset == "hflung":
        rows, stats = hflung_index(root)
        name, mid, level = "HF_Lung_V1", "M49_HF_Lung_V1_cycle", "respiratory_cycle (I+E)"
    else:
        raise ValueError(dataset)
    if limit:
        rows = rows[:limit]
        print(f"  --limit {limit}: SMOKE RUN, not a reportable result")

    y = np.array([r["label"] for r in rows])
    got = predict(model, rows, cfg, batch_size=batch_size, want_features=probe)
    pred = got["logits"].argmax(1)

    groups = np.array([r["patient_id"] for r in rows])
    group_kind = "patient" if name == "SPRSound" else "recording_group"

    # ICBHI's own class balance. Measured from the training split when the gate carried it
    # over; otherwise the committed constant, and the JSON says which was used - a prior
    # correction against a guessed source prior would be a fabricated diagnostic.
    if icbhi_gate is not None and "train_prior" in icbhi_gate:
        source_prior, prior_src = icbhi_gate["train_prior"], "measured from the ICBHI train split"
        icbhi_logits, icbhi_y = icbhi_gate["logits"], icbhi_gate["y"]
    else:
        source_prior, prior_src = ICBHI_TRAIN_PRIOR, "committed constant (gate not carried over)"
        icbhi_logits = icbhi_y = None

    print("  analysis arms: baselines, strict/broad, prior correction, calibration ...")
    extra = analysis_arms(rows, y, got["logits"], groups, group_kind, source_prior,
                          icbhi_logits=icbhi_logits, icbhi_y=icbhi_y, n_boot=n_boot)
    extra["source_class_prior_provenance"] = prior_src
    if probe:
        print("  frozen-feature probe (grouped 5-fold CV) ...")
        extra["feature_probe"] = feature_probe(got["features"], y, groups)
    doc = make_results(mid, name, level, rows, y, pred, cfg, meta, stats, icbhi_ref,
                       got["seconds"], extra)
    if limit:
        doc["meta"]["notes"] = (f"SMOKE RUN on the first {limit} rows - NOT reportable. "
                                + doc["meta"]["notes"])

    out = os.path.join(out_dir, f"results_{mid}.json")
    json.dump(doc, open(out, "w"), indent=2)
    np.save(os.path.join(out_dir, f"preds_{mid}.npy"),
            {"y_true": y, "y_pred": pred, "logits": got["logits"],
             "group": np.array([r["patient_id"] for r in rows]),
             "stem": np.array([r["stem"] for r in rows])}, allow_pickle=True)
    plot_confusion(doc["best_metrics"]["confusion_matrix_raw"],
                   f"{name} ({level}) - zero-shot from ICBHI",
                   os.path.join(out_dir, f"confusion_{mid}.png"))
    if verbose:
        print_summary(doc)
        print(f"  wrote {out}")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", choices=["sprsound", "hflung"])
    ap.add_argument("--root")
    # M22_v2 (0.5602) is the model the paper reports as best. M45 P3 scores higher
    # (0.5764) but the paper declines it: the delta is 1.1x the seed noise range and
    # its patient-level interval spans zero. Do not swap this to P3 without changing
    # the paper too.
    ap.add_argument("--ckpt", default=os.path.abspath(
        os.path.join(HERE, "..", "Asif's", "M22_v2", "Results", "best_model.pth")))
    ap.add_argument("--level", default="event", choices=["event", "record"])
    ap.add_argument("--out_dir", default=HERE)
    ap.add_argument("--icbhi_audio")
    ap.add_argument("--icbhi_split", default=os.path.abspath(
        os.path.join(HERE, "..", "Asif's", "ICBHI_challenge_train_test.txt")))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.dataset and a.root):
        ap.error("--dataset and --root are required (or use --selftest)")
    run(a.dataset, a.root, a.ckpt, a.out_dir, a.level, a.icbhi_audio, a.icbhi_split,
        a.limit, a.batch_size, a.n_boot, a.probe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
