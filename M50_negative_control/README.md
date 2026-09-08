# M50 — CirCor DigiScope as a far-OOD negative control

**Created:** 2026-09-08 · **Read with:** `../M49_cross_dataset/README.md`,
`../Asif's/M29/` (the near-OOD open-set baselines), `../Model_Training_Protocol.md` §1.

**Review comment this belongs to (Dr. Khan):** *"Test your best trained model with another
dataset, i.e., SPRSound (2022, open access) or HF Lung or the CirCor DigiScope."*

He named CirCor alongside two lung corpora. **It is not the same kind of dataset**, and the
honest answer to that part of the comment is this folder rather than a row in the transfer
table.

---

## This is not a transfer test

CirCor is a **phonocardiogram** corpus. It holds heart sounds, labelled murmur present /
absent / unknown. A crackle/wheeze classifier has **zero label overlap** with that, so
scoring it with the ICBHI metric produces no generalization number at all. Cross-dataset
transfer lives in `../M49_cross_dataset/`, on SPRSound and HF_Lung_V1.

**No CirCor label is read here.** Not the murmur annotation, not the segmentation `.tsv`, not
the demographics. The corpus enters as one thing only: *audio that is definitionally not a
respiratory cycle.* That is what makes the experiment valid rather than a category error, and
it is why `circor_index` has no label argument to get wrong.

If this is ever written up as "we also tested on CirCor" without that distinction, a reviewer
who knows the corpus will read it as a category error, and they will be right.

---

## Why it is worth running anyway

Two of the project's own results turned this from optional into the cheapest useful thing
left.

**1. The open-set arm is crippled by sample size.** `final paper/main.tex` reports a post-hoc
energy score on the frozen backbone at **AUROC 0.6466** — but over **19 unknown patients**, so
every open-set result is written as "not shown to beat" rather than as an improvement. CirCor
carries **1,452 patients / 5,268 recordings**.

**2. M49 found the model wrong and confident.** On SPRSound: mean max-softmax **0.9435** at
accuracy **0.2479**, ECE 0.2987 → 0.6956. The obvious next question is what it claims about
audio from an organ it has never seen. This is the extreme point of that curve.

## The two questions

| | Question | How it is measured |
|---|---|---|
| **Rejection** | can the frozen model tell this is not its domain? | energy score `-logsumexp(z)` over ICBHI-known vs CirCor-unknown, AUROC with a **patient-level** bootstrap. MSP and entropy reported beside it |
| **Abstention** | when it is wrong, does it know? | mean/median max-softmax, fraction above 0.90 and 0.99, entropy, and the predicted-class histogram — against the same quantities on ICBHI |

The energy score is the same statistic as `score_energy` in
`../Asif's/M29/M29_openset_baselines.ipynb`, on the same frozen backbone, so the far-OOD
number here and the near-OOD number in the paper are directly comparable **as statistics** —
though not as difficulties. See below.

---

## The caveat that must travel with the number

CirCor is **far-OOD**: a different organ. The paper's existing open-set arm uses **near-OOD**
unknowns — held-out ICBHI disease classes, 19 patients. **Rejecting a heartbeat is an easier
task than rejecting an unseen lung pathology.**

So a CirCor AUROC **does not replace, upgrade or supersede the 0.6466.** It is a second,
easier point on a difficulty axis, and it earns its place by establishing that the detector
functions *at all* — which n=19 could not show. Both numbers get reported, each with its
unknown set named.

That sentence ships inside the results JSON as `openset.difficulty_note`, and the near-OOD
reference travels beside it as `openset.near_ood_reference`, so neither can be lost in
transcription.

### The 4 kHz confound, read against the result

CirCor is recorded at 4 kHz, so its Nyquist is **2,000 Hz — exactly the model's mel `fmax`**.
The top of the band arrives attenuated by the recorder's anti-aliasing, the same confound
already recorded for HF_Lung_V1.

This section was written before the run, and it predicted the confound would *inflate* the
AUROC: bandlimited audio should look more alien, so a high number would have been partly
bandwidth rather than rejection. **The run came back at AUROC 0.4942, with the interval
spanning 0.50.** The prediction was not wrong, but it now reads the other way round: a
confound that could only have helped the detector was present, and there was still no
separation at all. That makes the null stronger, not weaker.

`dataset_info.bandwidth_note` still labels the number an upper bound, which remains the correct
label — the true separation may be even lower.

---

### Why it may have failed: the normalisation removes the obvious cue

`M45.log_mel` ends with, when `minmax` is on — and it is on for this checkpoint:

```python
lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
```

Every spectrogram is mapped to exactly [0, 1] before it reaches the network. Absolute level,
recording gain and microphone sensitivity are removed **by construction, for every input,
whatever corpus it came from**. That is not a hypothesis about this result; it is what the line
does. An energy-based OOD score reads logits, and the logits are driven by a representation
that was handed scale-free input, so the score cannot use absolute level to notice that the
audio is alien — only shape within the band.

What that argument does *not* settle is how much of the remaining shape differs between a
heartbeat and a breath at 50--2000 Hz after per-spectrogram normalisation. Answering that needs
the audio, so it is stated in the paper as a limitation with a named mechanism rather than as a
measured cause. The honest form is: the pipeline discards the cue an energy detector would most
plausibly use, and we did not measure what is left.

## What the code does

| Decision | Why |
|---|---|
| Bootstrap unit is the **patient** (filename field before the first `_`) | one patient contributes up to five auscultation locations. Resampling recordings would treat five views of one child's chest as five independent observations — the inflation the paper's audit section is about |
| Each recording is cut into consecutive **8 s windows** | 8 s is the model's input length, so a window is the natural unit and needs no tiling |
| A trailing fragment shorter than **1 s** is dropped, not tiled | tiling a 0.3 s remainder repeats it ~27 times and manufactures a periodic transient — the experiment would then be measuring that artefact instead of the audio |
| The known side is the **ICBHI corrected-split test partition** | the same 2,636 cycles the gate verifies, so both sides come from one verified forward pass |
| Gate tolerance defaults to **0.005** | see below |

### The gate, and why its tolerance is 0.005

Before any open-set number is computed, the checkpoint is re-scored on the 2,636 ICBHI test
cycles and must reproduce its own stored score.

The gate reproduces **0.5588** against a stored **0.5602** — a gap of 0.0014, about four
cycles in 2,636.

**The cause is not identified, and two candidates have been ruled out.** The librosa version is
not it: SPRSound ran pinned at 0.10.2 and HF_Lung ran unpinned at 0.11.0, and both reproduced
the same value. Restoring the float16 quantisation that M22_v2's memmap cache applied during
training moved it from 0.5585 to 0.5588 — real, but a tenth of the gap. What remains is
unexplained and is written up that way rather than guessed at.

0.0014 is **10× below the paper's own three-seed noise floor of 0.0141**, and a structurally
wrong forward path — missing ImageNet normalisation, wrong channel expansion, wrong mel band —
misses by 0.05–0.30, not 0.0014. So 0.005 is still 10–60× tighter than any structural bug.

The reproduced value is written into the results JSON next to the stored one
(`source_model.icbhi_score_reproduced_here`), so the gap is **reported, not hidden**. Override
with `M50_GATE_TOL` if you need to.

---

## Running it

**On Kaggle** — GPU T4, Internet not needed, and Add Data:

| Dataset | Why |
|---|---|
| `bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2` | CirCor |
| `vbookshelf/respiratory-sound-database` | ICBHI — the known side *and* the gate |
| `Asif's/M22_v2/Results/best_model.pth` as a private dataset | the checkpoint |

Then Run All. ~15 min. `M50_results.zip` lands in the Output panel; unzip it here and commit.

Set `SMOKE = 200` in the environment cell for a two-minute wiring check.

CirCor is also on [PhysioNet](https://physionet.org/content/circor-heart-sound/1.0.3/)
(open, no request form) if you would rather download it directly.

**Locally:**

```
python m50_openset.py --selftest
python m50_openset.py --circor_root .../circor --icbhi_audio .../audio_and_txt_files
```

`--ckpt` defaults to `../Asif's/M22_v2/Results/best_model.pth`. `best_model_official.pth`
cannot be used by accident — `m49_xval.load_checkpoint` reads `split_method` out of the cfg
and raises on the published-verbatim partition.

---

## The self-test

`--selftest` builds a synthetic CirCor (two patients, four recordings) and pushes it through
the real index builder and the real scorers. It exists because every failure mode here is
**silent**:

- a patient id parsed from the wrong filename field pools five recordings of one child as five
  patients and narrows every interval
- a window loop that runs past the end of a recording indexes rows that decode empty
- **an AUROC taken with the score oriented the wrong way reports `1 − AUROC`** — which looks
  like a result rather than a bug, so the test asserts a deliberately reversed scorer comes
  out visibly wrong rather than plausibly low

It also runs the M49 self-test, since the forward path and the gate come from that module.
Both run before any real data is touched, and the notebook aborts if either fails.

---

## Files

| File | What |
|---|---|
| `m50_openset.py` | the module — CirCor index, open-set scorers, confidence block, self-test |
| `gen_M50_kaggle.py` | the generator; **regenerate, never hand-edit the .ipynb** |
| `M50_circor_kaggle.ipynb` | the notebook |
| `results_M50_circor.json` · `logits_M50_*.npy` · `M50_openset.png` | outputs, once run |

## Where the result goes in the paper

Its **own subsection**, framed as trustworthiness, next to the existing open-set arm — never a
row in the M49 transfer table. Two sentences carry it:

1. the far-OOD AUROC with its patient-level interval, beside the near-OOD 0.6466 at n=19, with
   the difficulty asymmetry stated;
2. what the model claims about a heartbeat — the confidence and the predicted-class histogram
   — beside the same numbers on SPRSound.

## What this folder does not do

No fine-tuning, no training, no CirCor label, and no ICBHI-metric score on heart sounds. The
stop-lists in `../DECISION_2026-08-16_PIVOT.md` §12 and `../RTK_requirements.md` §12 still hold.
