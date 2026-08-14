# Literature Review — AST: Audio Spectrogram Transformer

**Asif Mahbub** · CSE465 individual literature review · **due 1 Aug 2026**

> **This is a working outline, not a submission.** Everything marked `[verify]`
> is written from memory of the paper and **must be checked against the PDF**
> before it goes in front of Dr. Khan. He will catch a wrong number. Sections
> marked *(write after reading)* are deliberately empty — reading the paper is
> the assignment, and an outline that pre-writes your opinion defeats it.

---

## 0. Why this paper

Three reasons, in order of how much they matter:

1. **You run it on 8 August.** AST is your assigned model for the pretrained-model
   presentation. Reviewing it now means the review and the presentation reinforce
   each other instead of competing for the same fortnight.
2. **CORE A venue.** Interspeech satisfies Dr. Khan's venue rule without argument.
3. **It becomes Related Work.** The paragraph you write in §4 below is reusable
   almost verbatim in the manuscript's backbone-justification passage.

**Confirm the deliverable format before writing** — slides, a written review, or
both, and the length. It is not stated in any document in this repo.

---

## 1. Citation

> Gong, Y., Chung, Y.-A., & Glass, J. (2021). *AST: Audio Spectrogram
> Transformer.* In **Proc. Interspeech 2021**, pp. 571–575. `[verify pages]`
> arXiv:2104.01778. DOI: 10.21437/Interspeech.2021-698 `[verify]`

Venue rank: **CORE A** (Interspeech). `[verify against the current CORE portal —
Dr. Khan may ask for the source of the ranking]`

---

## 2. What the paper claims

*(Verify every number in this section against the PDF.)*

The first **convolution-free, purely attention-based** model for audio
classification. Prior spectrogram models put a CNN front-end under an attention
layer; AST removes the CNN entirely.

**Architecture**
- Input: 128-dim log Mel filterbank, 25 ms window / 10 ms shift → a
  1024 × 128 matrix for 10.24 s of audio. `[verify]`
- Split into **16 × 16 patches with stride 10** in both time and frequency, so
  patches overlap by 6. Yields **1212 patches** for a 1024-frame input. `[verify]`
- Each patch is linearly projected to a 768-d embedding, plus a trainable
  positional embedding and a `[CLS]` token.
- A 12-layer, 12-head **ViT-Base / DeiT** encoder. **~87 M parameters.** `[verify]`
- Mean of the `[CLS]` and distillation tokens → linear classifier. `[verify]`

**The transfer trick — this is the actual contribution**
ImageNet-pretrained ViT weights are adapted to audio by:
- averaging the three RGB channels of the patch-embedding filters into one;
- **cut-and-bilinear-interpolating** the 2-D positional embeddings from the ViT's
  square image grid onto the audio grid, which is long in time and short in
  frequency. `[verify the exact procedure]`

Without this, the paper reports a large drop — quantify it and put the number on
a slide. `[verify]`

**Headline results** `[verify all four]`
| Benchmark | AST | Prior SOTA |
|---|---|---|
| AudioSet (mAP, single model) | 0.459 | |
| AudioSet (mAP, ensemble) | 0.485 | |
| ESC-50 (accuracy) | 95.6% | |
| Speech Commands V2 | 98.11% | |

**Ablations reported:** ImageNet pretraining on/off, positional-embedding
adaptation strategy, patch overlap, model size, and (for AudioSet) balanced
sampling, mixup and SpecAugment. `[verify the list]`

---

## 3. Critical appraisal

*(Write after reading — this section is what separates a good review from a
summary. Below are the lines of attack worth checking; confirm each holds before
asserting it.)*

1. **Is it really convolution-free?** The patch-embedding layer is a strided
   linear projection over 16 × 16 patches — arithmetically a convolution. Does the
   paper acknowledge this? The claim may be about the *encoder*, not the model.
2. **How much of the result is ImageNet rather than the architecture?** The
   ablation is in the paper; report the delta honestly. If ImageNet pretraining
   carries most of the gain, "convolution-free architecture wins" is the wrong
   headline.
3. **Cost.** Self-attention is quadratic in sequence length and the sequence is
   1212 patches. Compare inference latency and parameter count against the CNN
   baselines it beats. **You will have your own measured numbers from the 8 Aug
   run** — using them here is a genuinely original contribution to the review, and
   it is the single highest-value thing you can add.
4. **Data scale.** AudioSet is ~2 M clips. ICBHI is 6,898 cycles from 126
   patients — roughly three orders of magnitude smaller. Nothing in the paper
   speaks to transformer behaviour at that scale. This is the gap your project
   sits in, and it is the honest framing: *AST is the strongest available audio
   encoder, but its evidence base does not cover small clinical corpora.*
5. **Closed-set only.** Every benchmark is closed-set: fixed label space, all test
   classes seen in training. Nothing evaluates behaviour on inputs from classes the
   model was never trained on. **This is the bridge to your project** — see §4.
6. **Positional embedding interpolation is a heuristic.** It works, but the paper
   offers no principled account of why bilinear interpolation is the right map from
   an image grid to a time–frequency grid.

---

## 4. Connection to the project *(reusable in Related Work)*

Draft, to be tightened after reading:

> AST established the transformer as the default encoder for spectrogram
> classification, and its ImageNet→AudioSet transfer recipe is what makes it
> viable on datasets far smaller than AudioSet — the regime ICBHI occupies. We
> adopt it as a backbone candidate for that reason. Its evaluation, however, is
> uniformly closed-set: performance is measured over a fixed label space in which
> every test class appears in training. It offers no evidence about what its
> representation does with inputs drawn from classes it has never seen, which is
> precisely the regime an open-world respiratory screening system operates in. We
> therefore evaluate the AST representation on a criterion the original work does
> not consider — how separable unseen disease classes are in its embedding space —
> and show that this criterion does not rank backbones the same way closed-set
> accuracy does.

That last clause is the M12 result. **Hedge it until you have the number** — if
the criteria turn out to agree, the sentence becomes *"and show that on ICBHI the
two criteria coincide, which has not previously been checked."* Both are
publishable; only one is true, and you don't know which yet.

---

## 5. Two alternatives, if you'd rather not review AST

Weigh these before committing — the choice is due with the review.

| Paper | Case for | Case against |
|---|---|---|
| **Cho & Lee (2025)**, *Enhancing Respiratory Sound Classification Based on Open-Set Semi-Supervised Learning*, CMC 84(2) | The closest prior art to the whole project; you'd have to read it eventually anyway | It is Member B's territory, not yours — reviewing it does not feed your 8 Aug run |
| **Karim, Mahmud & Khan (2024)**, *Advanced Vision Transformers and Open-Set Learning for Robust Mosquito Classification*, PLOS Comp Biol 20(12) | It is Dr. Khan's own paper; reviewing it signals engagement, and it is the OpenMax baseline the project must beat | He will catch every error. Only choose this if you will read it properly, twice |

**Recommendation: stay with AST.** It is the only one of the three that does
double duty with the 8 August deliverable, and you have eight days.

---

## 6. Schedule

| Date | Task |
|---|---|
| 25–26 Jul | Read the paper twice. Second pass: fill in every `[verify]` above |
| 27 Jul | Write §2 (summary) and §3 (appraisal) |
| 28 Jul | Skim the two or three papers AST builds on (DeiT; PSLA `[verify this is the right predecessor]`) for context |
| 29 Jul | Draft §4 and assemble |
| 30 Jul | Revise; confirm the required format and length |
| 31 Jul | Buffer |
| **1 Aug** | **Submit** |

Note that **3 August** — the internal deadline for shipping `splits/` to the
group — sits two days after this. Do not let the review consume the whole week;
the split artifacts block three other people.
