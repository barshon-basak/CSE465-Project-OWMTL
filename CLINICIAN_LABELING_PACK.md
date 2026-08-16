# Clinician Labeling Pack — how to actually run it

**For:** Asif, handing a listening task to a family physician
**Purpose:** produce a small, *validated* ground-truth set for the fine-grained acoustic concepts
(G0 in `OWMTL_Decision_Roadmap (v3).md`)

---

## ⚠️ First — a correction to my earlier estimate

In `OWMTL_Week0_Go_Memo.md` I estimated **~2 hours** based on "DSP pre-labels the clips, clinician
just corrects them." **That protocol is invalid for this purpose**, and I should have caught it
there.

The whole point of the exercise is to check whether our DSP extractors measure what their names
say. If the clinician sees the DSP's answer before giving theirs, they will anchor on it — and the
agreement we measure is partly *their agreement with a suggestion we made*, not independent
validation. It is circular.

**So: the clinician must be blind to the DSP output.** That makes each clip slower (~30–40 s
instead of ~10 s), so the honest trade is **fewer clips, labeled blind**:

| | Clips | Clinician time | Valid for G2? |
|---|---|---|---|
| ~~Pre-labeled review~~ | 300 | ~2 h | ❌ circular |
| **Blind labeling** ✅ | **120** | **~75 min** | ✅ |

**120 blind clips is worth more than 300 anchored ones.** 120 is enough: with ~40 crackle-positive
clips you can estimate agreement to roughly ±0.08, which is plenty to answer "does the detector
work at all."

*(DSP pre-labeling is still fine for any clips used only as bottleneck training input — just never
for the validation subset.)*

---

## Part 1 — The conversation (5 minutes, before any work)

Ask exactly these, in order. **Question 2 is the one that can end it.**

1. **"Can you give about 75 minutes, split into two sittings, over the next 2–3 weeks?"**

2. **"Are you comfortable judging crackle type — fine vs. coarse — from a recording, on
   headphones, without the patient in front of you?"**

   Ask it plainly and *accept a no*. Recorded auscultation is genuinely harder than bedside: no
   chest-wall context, no palpation, no ability to ask the patient to breathe deeper. Many
   clinicians will say they can call *crackle vs. wheeze* confidently but not *fine vs. coarse*
   from a clip.

   **A "no" here is a real result, not a failure.** It would mean the fine/coarse distinction is
   not reliably human-labelable from ICBHI audio *by anyone* — which tells us to report those
   concepts as unvalidated proxies permanently, and is itself worth a sentence in the paper.

3. **"May we name you in the acknowledgements as the validating clinician?"** — the phrase
   "validated against a clinical reference" means nothing if we can't say who.

4. *(If they're keen)* **"Could a second clinician colleague do 30 of the same clips?"** — lets us
   report inter-rater agreement, which reviewers ask for. Nice-to-have, not required.

---

## Part 2 — What you prepare (your side, ~1 day)

Deliver a **single folder** they can open on any laptop. No install, no login, no Python.

```
ICBHI_listening_pack/
├── START_HERE.pdf              <- 1-page instructions (Part 3 below)
├── reference_sounds/           <- 6 short exemplars, so they calibrate to OUR terms
│   ├── example_fine_crackle.wav
│   ├── example_coarse_crackle.wav
│   ├── example_wheeze.wav
│   ├── example_rhonchi.wav
│   ├── example_normal.wav
│   └── example_ambiguous.wav
├── clips/                      <- 132 clips, numbered, NOTHING else in the name
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
└── labels.xlsx                 <- one row per clip, pre-filled clip IDs, empty answers
```

### Rules for building `clips/`

| Rule | Why |
|---|---|
| **Filenames carry no information** — `clip_001.wav`, never `104_crackle.wav` | A filename hinting at the answer invalidates the label |
| **Randomize the order**, don't group by class | Prevents "I've had five crackles, this is probably another" |
| **Loudness-normalize every clip** (same RMS) | Otherwise they fight the volume knob and fatigue fast |
| **Pad very short cycles to ≥1.5 s** | Sub-second clips are unjudgeable |
| **Include 12 duplicate clips** (hidden, at different positions) | Gives intra-rater agreement for free — do they label the same clip the same way twice? This is the single cheapest quality signal you can get |
| **Stratify the sample**: ~40 crackle-only, ~25 wheeze-only, ~15 both, ~40 normal | ICBHI is ~55% normal; an unstratified draw wastes their time on easy negatives |
| **Keep a private `clip_key.csv`** mapping clip ID → original ICBHI file/cycle | You need it to join their labels back. **Do not put it in the folder you send.** |

The 132 = 120 unique + 12 duplicates.

---

## Part 3 — What they actually do (`START_HERE.pdf`)

Write it in this tone — short, concrete, no ML vocabulary:

> **Thank you — this should take about 75 minutes, and you can stop and resume any time.**
>
> **Setup**
> 1. Use **headphones or earbuds**, not laptop speakers. Crackles are quiet and brief; laptop
>    speakers cannot reproduce them.
> 2. Sit somewhere quiet. Set a comfortable volume on `reference_sounds/example_normal.wav` and
>    then **don't change it** — the clips are all at the same level.
>
> **Calibrate (5 min)**
> Listen to all six files in `reference_sounds/` first. They show what we mean by each term, so
> that we're using words the same way. Listen again any time you're unsure.
>
> **Label (about 70 min)**
> 1. Open `labels.xlsx`. Each row is one clip.
> 2. Play `clips/clip_001.wav` in any media player (double-click usually works).
> 3. Fill in that row. Replay as many times as you like.
> 4. Move to the next. **Please go in order and don't skip** — if a clip is unjudgeable, mark it
>    `unsure` and move on.
>
> **The columns**
>
> | Column | Options | Note |
> |---|---|---|
> | `crackles` | `none` / `fine` / `coarse` / `both` / `unsure` | The one we care most about |
> | `wheeze` | `none` / `wheeze` / `rhonchi` / `both` / `unsure` | |
> | `confidence` | `low` / `medium` / `high` | Please be honest — low-confidence answers are still useful, and we weight by this |
> | `audio_quality` | `ok` / `noisy` / `unusable` | Lets us drop bad recordings rather than blame the model for them |
> | `notes` | free text | Optional |
>
> **Please use `unsure` freely.** We would much rather have an honest "unsure" than a guess. A
> high `unsure` rate is itself a finding for us and will not have been wasted effort.
>
> **What we are *not* asking:** you are not diagnosing the patient. There is no clinical history
> and no chest X-ray, and nothing here affects anyone's care. This is only "what sound do you
> hear in this clip."
>
> When finished, email back `labels.xlsx`. That's it.

---

## Part 4 — What you do with it

1. **Intra-rater check first.** Compare the 12 duplicate pairs. If they disagree with themselves on
   more than ~3 of 12, the task is too hard from recordings — report that and treat the
   fine-grained concepts as permanently unvalidated proxies. **Check this before anything else**;
   it decides whether the rest of the labels mean anything.
2. **Drop `audio_quality = unusable`** clips before computing agreement.
3. **Compute agreement** between clinician labels and the DSP extractors:
   - `crackle_score` vs `crackles != none` → AUROC + CI
   - `wheeze_score` vs `wheeze != none` → AUROC + CI
   - `crackle_fine_ratio` vs `crackles == fine` → **this is the new one G0 buys**
   - `rhonchi_score` vs `wheeze == rhonchi`
   Use `Asif's/Statistics/owmtl_scores.py` — dump their labels as a score file and you get paired
   tests and CIs for free.
4. **Report `unsure` and low-confidence rates in the paper.** If the clinician was unsure on 30% of
   fine/coarse calls, that is a headline-worthy statement about the task, not an embarrassment.
5. **Promote the concepts that pass.** Anything with defensible agreement graduates from "proxy"
   to "validated" in `concept_extractors.py::CONCEPT_VALIDATION`, and may be called by its
   clinical name in the paper. Anything that fails stays a proxy — permanently and explicitly.

---

## What this changes if it works

Currently only **2 of 9** concepts (`crackle_score`, `wheeze_score`) can be validated at all,
against ICBHI's coarse labels. The other 7 are proxies with face validity only.

This exercise is the only thing that can promote `crackle_fine_ratio`, `rhonchi_score` and
`wheeze_pitch_hz` — and those are precisely the *clinically-named* concepts that make the
bottleneck interesting rather than a generic feature vector. That is why G0 is described as the
Q1-ceiling lever.

If the answer is no, everything still runs; the fine-grained concepts just stay honestly labeled as
proxies throughout.
