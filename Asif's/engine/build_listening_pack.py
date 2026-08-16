#!/usr/bin/env python3
"""
Build the clinician listening pack (Gate G0).

Produces a folder a non-technical physician can open on any laptop: numbered clips, a reference
set, an instruction sheet, and a spreadsheet. No install, no Python, no login.

    python3 build_listening_pack.py --data-root <ICBHI audio_and_txt_files> --out ./pack

WHY THE DESIGN IS WHAT IT IS
----------------------------
* **Blind.** Clips carry no DSP output and no informative filename. If the clinician sees our
  detector's answer first they anchor on it, and the "agreement" we measure is partly agreement
  with our own suggestion. See CLINICIAN_LABELING_PACK.md.
* **Hidden duplicates.** A fixed number of clips appear twice at distant positions. Comparing a
  rater against themselves is the cheapest quality signal available, and it is the first thing to
  check when the labels come back: if they disagree with themselves, nothing else in the file
  means much.
* **Stratified.** ICBHI is ~55% normal. An unstratified draw would spend the physician's time on
  easy negatives.
* **Loudness-normalised.** Otherwise they fight the volume knob and fatigue fast.
* **The key file stays behind.** `clip_key.csv` maps clip -> original cycle AND records ICBHI's own
  label. It must NOT go in the folder you send.

THE SECOND PURPOSE (added 2026-08-16, after M39)
------------------------------------------------
M39 found the DSP extractors barely beat chance against ICBHI labels (crackle AUROC 0.55).
There are two possible explanations and they need separating:

    (a) our extractors are bad, or
    (b) ICBHI's own cycle labels are noisy.

Because `clip_key.csv` records ICBHI's label for every clip, these labels answer that directly:
**clinician-vs-ICBHI agreement is a measurement we get for free here**, and it is diagnostic
regardless of how the extractors are performing. If a physician disagrees substantially with ICBHI
on the same audio, an AUROC of 0.55 against those labels means something quite different — and
that is a publishable finding about the benchmark, not just about us.

So this pack is worth building even though the extractors currently need work. The labels describe
the *audio*; they stay valid across every future extractor revision.
"""
import argparse
import csv
import glob
import os
import random
import shutil

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import librosa
except ImportError:
    librosa = None

# stratum -> how many unique clips
DEFAULT_STRATA = {"crackle_only": 40, "wheeze_only": 25, "both": 15, "normal": 40}
N_DUPLICATES = 12
TARGET_RMS = 0.06
MIN_CLIP_S = 1.5
PAD_S = 0.25          # context either side of the annotated cycle


def parse_annotations(txt):
    out = []
    for line in open(txt):
        p = line.split()
        if len(p) < 4:
            continue
        try:
            s, e, cr, wh = float(p[0]), float(p[1]), int(p[2]), int(p[3])
        except ValueError:
            continue
        if e > s:
            out.append((s, e, cr, wh))
    return out


def stratum_of(cr, wh):
    if cr and wh:
        return "both"
    if cr:
        return "crackle_only"
    if wh:
        return "wheeze_only"
    return "normal"


def collect_cycles(data_root):
    rows = []
    for wav in sorted(glob.glob(os.path.join(data_root, "*.wav"))):
        stem = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(data_root, stem + ".txt")
        if not os.path.exists(txt):
            continue
        for i, (s, e, cr, wh) in enumerate(parse_annotations(txt)):
            rows.append({"stem": stem, "cycle_idx": i, "wav": wav, "start": s, "end": e,
                         "icbhi_crackle": cr, "icbhi_wheeze": wh,
                         "stratum": stratum_of(cr, wh), "duration": e - s})
    return rows


def load_clip(row, sr):
    y, _ = librosa.load(row["wav"], sr=sr, mono=True,
                        offset=max(0.0, row["start"] - PAD_S),
                        duration=(row["end"] - row["start"]) + 2 * PAD_S)
    if len(y) < int(MIN_CLIP_S * sr):                 # pad with silence, never tile
        pad = int(MIN_CLIP_S * sr) - len(y)
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms > 1e-9:
        y = y * (TARGET_RMS / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:                                   # avoid clipping after normalisation
        y = y * (0.99 / peak)
    return y.astype(np.float32)


START_HERE = """# Respiratory Sound Listening Task

Thank you — this should take about **75 minutes**, and you can stop and resume any time.

## Setup (2 min)

1. Use **headphones or earbuds**, not laptop speakers. Crackles are quiet and very brief; laptop
   speakers cannot reproduce them.
2. Sit somewhere quiet. Set a comfortable volume using `reference_sounds/example_normal.wav`,
   then **do not change it** — every clip is at the same loudness.

## Calibrate (5 min)

Listen to all the files in `reference_sounds/` first. They show what we mean by each term so we
are using the words the same way. Go back to them any time you are unsure.

*(Note: these examples are drawn from the dataset's own labels, not curated by a clinician. If you
think one is mislabelled, please tell us — that is useful information in itself.)*

## Label (about 70 min)

**Open `labels.xlsx`** — every answer cell is a dropdown, so you pick rather than type. (There is
also a `labels.csv` if you prefer, but the xlsx has the dropdowns and is easier.)

*If we sent you a **Google Sheets** link instead, use that — same columns, same dropdowns, saves
automatically, and nothing to email back. Still download the clips folder and play the files
locally; a proper media player lets you scrub and replay far better than a browser preview, which
matters for sounds this brief.*

1. Play `clips/clip_001.wav` in any media player (double-click usually works).
2. Fill in that row using the dropdowns. Replay as often as you like.
3. Go to the next one. **Please go in order and do not skip.** If a clip is unjudgeable, mark
   `unsure` and move on.

### The columns

| Column | Options |
|---|---|
| `crackles` | `none` / `fine` / `coarse` / `both` / `unsure` |
| `wheeze` | `none` / `wheeze` / `rhonchi` / `both` / `unsure` |
| `confidence` | `low` / `medium` / `high` |
| `audio_quality` | `ok` / `noisy` / `unusable` |
| `notes` | anything you want to add (optional) |

**Please use `unsure` freely.** We would much rather have an honest "unsure" than a guess. A high
`unsure` rate is a genuine finding for us and will not mean your time was wasted.

**Please be honest with `confidence`.** Low-confidence answers are still useful — we weight by it.

## What we are NOT asking

You are not diagnosing anyone. There is no clinical history, no imaging, and nothing here affects
any patient's care. These are anonymous archived research recordings. The only question is:
**what sound do you hear in this clip?**

## When you are done

Email back `labels.xlsx` — or, if you used the Google Sheet, just tell us you have finished and we
will take it from there. That's it — thank you.
"""


# valid answers -- also used to build the spreadsheet dropdowns
OPTIONS = {
    "crackles": ["none", "fine", "coarse", "both", "unsure"],
    "wheeze": ["none", "wheeze", "rhonchi", "both", "unsure"],
    "confidence": ["low", "medium", "high"],
    "audio_quality": ["ok", "noisy", "unusable"],
}


def write_xlsx(path, label_rows):
    """Excel version with locked dropdowns. Falls back to CSV-only if openpyxl is absent.

    The dropdowns are not cosmetic: free-typed answers arrive as 'Fine', 'FINE', 'fine ',
    'f' and each variant silently becomes its own category at analysis time. Constraining the
    cells removes a whole class of cleanup, and a physician filling 132 rows should not have to
    remember exact spellings.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        print("  (openpyxl not installed -- labels.xlsx skipped, labels.csv is sufficient)")
        return None

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    headers = list(label_rows[0])
    ws.append(headers)

    head_fill = PatternFill("solid", fgColor="DDE5F0")
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=1, column=i)
        c.font = Font(bold=True)
        c.fill = head_fill
        c.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"

    for r in label_rows:
        ws.append([r[h] for h in headers])

    widths = {"clip_id": 12, "crackles": 14, "wheeze": 14, "confidence": 13,
              "audio_quality": 15, "notes": 46}
    for i, h in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 14)

    last = len(label_rows) + 1
    for col_name, opts in OPTIONS.items():
        if col_name not in headers:
            continue
        letter = get_column_letter(headers.index(col_name) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"',
                            allow_blank=True, showDropDown=False)
        dv.error = f"Please choose one of: {', '.join(opts)}"
        dv.errorTitle = "Not a valid answer"
        dv.prompt = f"Choose: {', '.join(opts)}"
        dv.promptTitle = col_name
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{last}")

    # A short instructions sheet, so the spreadsheet is self-explanatory if START_HERE is lost.
    ins = wb.create_sheet("instructions")
    for row in [
        ["Respiratory Sound Listening Task"], [],
        ["1.", "Use HEADPHONES, not laptop speakers. Crackles are quiet and very brief."],
        ["2.", "Listen to everything in reference_sounds/ first, to calibrate."],
        ["3.", "Play clips/clip_001.wav, fill row 2, then continue in order."],
        ["4.", "Replay as often as you like. Do not skip -- use 'unsure' instead."], [],
        ["crackles", " / ".join(OPTIONS["crackles"])],
        ["wheeze", " / ".join(OPTIONS["wheeze"])],
        ["confidence", " / ".join(OPTIONS["confidence"])],
        ["audio_quality", " / ".join(OPTIONS["audio_quality"])], [],
        ["Please use 'unsure' freely -- an honest 'unsure' is more useful to us than a guess."],
        ["You are NOT diagnosing anyone. These are anonymous archived research recordings."],
        ["When finished, email the file back. Thank you."],
    ]:
        ins.append(row)
    ins.column_dimensions["A"].width = 16
    ins.column_dimensions["B"].width = 78
    ins["A1"].font = Font(bold=True, size=13)

    wb.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="ICBHI audio_and_txt_files directory")
    ap.add_argument("--out", default="./ICBHI_listening_pack")
    ap.add_argument("--key-out", default="./clip_key.csv",
                    help="written OUTSIDE the pack -- never send this to the clinician")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if sf is None or librosa is None:
        raise SystemExit("Needs soundfile and librosa:  pip install soundfile librosa")

    rng = random.Random(a.seed)
    cycles = collect_cycles(a.data_root)
    if not cycles:
        raise SystemExit(f"No annotated cycles under {a.data_root}")
    print(f"Found {len(cycles)} annotated cycles.")

    # ---- stratified sample, prefer longer cycles (more listenable) ----
    chosen = []
    for stratum, n in DEFAULT_STRATA.items():
        pool = [c for c in cycles if c["stratum"] == stratum and c["duration"] >= 0.9]
        if len(pool) < n:
            print(f"  WARNING: only {len(pool)} '{stratum}' cycles available, wanted {n}")
            n = len(pool)
        pool.sort(key=lambda c: -c["duration"])
        top = pool[: max(n * 4, n)]              # longest 4n, then sample for patient variety
        rng.shuffle(top)
        seen_patients, picked = set(), []
        for c in top:                            # spread across patients where possible
            pid = c["stem"].split("_")[0]
            if pid in seen_patients and len(picked) < n:
                continue
            picked.append(c)
            seen_patients.add(pid)
            if len(picked) == n:
                break
        for c in top:                            # top up if patient-spreading fell short
            if len(picked) == n:
                break
            if c not in picked:
                picked.append(c)
        chosen += picked
        print(f"  {stratum:<14} {len(picked)} clips from {len(seen_patients)} patients")

    # ---- hidden duplicates, placed far from their twin ----
    # Random shuffling is not enough: it happily put two copies 2 clips apart in testing, which
    # the rater would simply recognise, destroying the intra-rater measurement it exists for.
    # Instead the first copy goes in the first half and the twin in the second half.
    MIN_SEP = 20
    playlist = list(chosen)
    rng.shuffle(playlist)
    half = len(playlist) // 2

    # Choose WHICH clips to duplicate only after shuffling, and only from early positions, so the
    # twin always has room to land >= MIN_SEP later. An earlier version instead swapped a
    # late-positioned original into the first half -- but that displaced whatever sat there, and
    # when the displaced item was another duplicate's original it landed beside its own twin.
    # 166 of 300 seeds violated the separation. Picking from the front removes the swap entirely,
    # and since every insertion happens AFTER an original, separations can only grow.
    dup_zone = max(1, min(half - MIN_SEP, len(playlist)))
    dups = rng.sample(playlist[:dup_zone], min(N_DUPLICATES, dup_zone))

    for d in dups:
        first_at = playlist.index(d)
        lo = min(first_at + MIN_SEP + 1, len(playlist))
        playlist.insert(rng.randrange(lo, len(playlist) + 1), d)

    # Verify the guarantee rather than trusting it -- an earlier arrangement violated it on more
    # than half of all seeds and produced a pack where the rater would simply recognise the repeat.
    seen_pos, seps = {}, []
    for i, c in enumerate(playlist):
        k = (c["stem"], c["cycle_idx"])
        if k in seen_pos:
            seps.append(i - seen_pos[k])
        else:
            seen_pos[k] = i
    assert seps, "no duplicate pairs were placed"
    assert min(seps) >= MIN_SEP, (
        f"duplicate pairs too close (min {min(seps)} < {MIN_SEP}) -- a rater would recognise the "
        f"repeat and the intra-rater check would be meaningless")

    print(f"\n{len(chosen)} unique + {len(dups)} hidden duplicates = {len(playlist)} clips")
    print(f"  duplicate separation: min {min(seps)}, median {sorted(seps)[len(seps) // 2]} "
          f"clips apart (>= {MIN_SEP} enforced)")

    out = a.out
    clips_dir = os.path.join(out, "clips")
    refs_dir = os.path.join(out, "reference_sounds")
    for d in (clips_dir, refs_dir):
        os.makedirs(d, exist_ok=True)

    # ---- write clips + key ----
    key_rows, label_rows = [], []
    for i, c in enumerate(playlist, start=1):
        cid = f"clip_{i:03d}"
        y = load_clip(c, a.sr)
        sf.write(os.path.join(clips_dir, cid + ".wav"), y, a.sr)
        key_rows.append({"clip_id": cid, "stem": c["stem"], "cycle_idx": c["cycle_idx"],
                         "start": round(c["start"], 4), "end": round(c["end"], 4),
                         "stratum": c["stratum"],
                         "icbhi_crackle": c["icbhi_crackle"], "icbhi_wheeze": c["icbhi_wheeze"],
                         "unit_id": f"{c['stem']}__{c['cycle_idx']}",
                         "is_duplicate_of": ""})
        label_rows.append({"clip_id": cid, "crackles": "", "wheeze": "",
                           "confidence": "", "audio_quality": "", "notes": ""})

    # mark duplicate pairs in the key (same unit_id appearing twice)
    seen = {}
    for r in key_rows:
        if r["unit_id"] in seen:
            r["is_duplicate_of"] = seen[r["unit_id"]]
        else:
            seen[r["unit_id"]] = r["clip_id"]

    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_rows[0]))
        w.writeheader()
        w.writerows(label_rows)

    write_xlsx(os.path.join(out, "labels.xlsx"), label_rows)

    with open(a.key_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)

    # ---- reference exemplars (longest clean example per category) ----
    refs = {"example_normal": "normal", "example_fine_crackle": "crackle_only",
            "example_coarse_crackle": "crackle_only", "example_wheeze": "wheeze_only",
            "example_rhonchi": "wheeze_only", "example_ambiguous": "both"}
    used = {c["unit_id"] if "unit_id" in c else (c["stem"], c["cycle_idx"]) for c in chosen}
    for name, stratum in refs.items():
        pool = [c for c in cycles
                if c["stratum"] == stratum and (c["stem"], c["cycle_idx"]) not in
                {(x["stem"], x["cycle_idx"]) for x in chosen}]
        if not pool:
            continue
        pool.sort(key=lambda c: -c["duration"])
        sf.write(os.path.join(refs_dir, name + ".wav"), load_clip(pool[0], a.sr), a.sr)

    with open(os.path.join(out, "START_HERE.md"), "w") as f:
        f.write(START_HERE)

    print(f"\nPack written to: {out}")
    print(f"  clips/            {len(playlist)} files")
    print(f"  reference_sounds/ {len(os.listdir(refs_dir))} files")
    if os.path.exists(os.path.join(out, "labels.xlsx")):
        print(f"  labels.xlsx       {len(label_rows)} rows, dropdowns  <-- the one to use")
    print(f"  labels.csv        same, plain-text fallback")
    print(f"  START_HERE.md")
    print(f"\nKEY (do NOT send): {a.key_out}")
    print(f"  records ICBHI's own label per clip -> lets you measure clinician-vs-ICBHI")
    print(f"  agreement, which is what separates 'our extractors are weak' from")
    print(f"  'the benchmark labels are noisy'.")
    print("\nBefore sending, check: no filename reveals a label, and clip_key.csv is NOT inside "
          f"{out}.")
    assert not os.path.exists(os.path.join(out, os.path.basename(a.key_out))), \
        "clip_key.csv ended up inside the pack -- remove it before sending."


if __name__ == "__main__":
    main()
