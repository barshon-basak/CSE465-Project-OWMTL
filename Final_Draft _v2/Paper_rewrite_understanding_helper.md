# Paper rewrite companion — understanding every part of `main.tex`

**Purpose.** This file explains the paper section by section, paragraph by paragraph, and
table by table, so that you can rewrite any part of it in your own words without breaking
an argument, losing a citation, or introducing a number that the repository cannot
support.

> **Revision note (v2).** The paper was revised after this guide was first written. Four
> changes matter for a rewrite, and every affected section below has been updated:
> (a) **model nicknames are gone from the body** — models are named by architecture, and the
> internal run identifiers now live only in Appendix A;
> (b) **the Audio Spectrogram Transformer is now reported** (Section III-D), including the fact
> that its challenge score is unrecoverable;
> (c) **four tables were cut or merged** as repetitive — the corpus table, the preprocessing
> chain table, the augmentation table and the failure catalogue; the partition and class
> tables merged, and so did the two human-comparison tables;
> (d) **ablation rows were relabelled** `REF`, `C1`–`C7`, `P1`–`P5`.
> The paper now carries **17 tables in the body, 1 in the appendix, and 4 figures**.

**How to read it.** Part 1 gives the project history — you need this, because several
choices in the paper only make sense once you know what the project originally was.
Part 2 is the argument in one page. Part 3 is the section-by-section walkthrough and is
the main body. Parts 4–6 are reference sheets for every table, figure and equation.
Part 7 tells you where each number lives on disk. Part 8 lists claims that must never
re-enter the paper. Parts 9–10 are the rules and the final checklist.

**Golden rule for every rewrite.** Change wording freely. Never change a number,
never change a hedge into a stronger claim, and never merge two rows that sit on
different data partitions.

---

## Ttiles : Title options

Audit-forward
1. Auditing the Score Before the Model: Corrected Evaluation Baselines and a Pre-Registered Negative Result for Respiratory Sound Classification (current)
2. What an ICBHI Score Measures: Five Evaluation Faults, Corrected Baselines, and a Retracted Ceiling
3. The Protocol Is the Model: How Five Measurement Faults Outweighed Every Architectural Choice on ICBHI 2017

Number-forward
4. From 0.7077 to 0.5602: Auditing Our Own Respiratory Sound Pipeline
5. Fourteen Points of Protocol: What an Unaudited Evaluation Is Worth on ICBHI 2017

Finding-forward
6. Machines Reproduce ICBHI's Labels Better Than a Physician Does: A Corrected Protocol and a Pre-Registered Retraction
7. A Gate That Could Fail, and Did: Pre-Registered Concept Validation and a Corrected Evaluation Protocol

Reproducibility-forward
8. Commit the Confusion Matrix: An Evaluation Audit of Respiratory Sound Classification on ICBHI 2017
9. Measuring the Measurement: An Ablation of the Evaluation Protocol for Respiratory Sound Classification

Conventional
10. Corrected Evaluation Baselines for Respiratory Sound Classification: Convolutional and Transformer Backbones, Ablation, and Explainability on ICBHI 2017

My picks: #3 reads as a real finding and states the paper's thesis; #8 is the most memorable and ties directly to the AST story you just added; #10 is the safest if the supervisor expects a conventional capstone title. 
---

# Part 1 — The story of the project

You cannot rewrite this paper well without knowing this. The paper is unusual: it does
not report a method that worked. It reports a measurement, and the measurement only
became the contribution because an earlier plan failed in a specific, honest way.

## 1.1 What the project was originally

The team (four members) set out to build **OWMTL** — an open-world multi-task learner for
respiratory sounds. The design had three parts:

1. A shared spectrogram backbone.
2. Two heads — one for the four sound-event classes (Normal / Crackle / Wheeze / Both),
   one for disease.
3. Between them, a **concept bottleneck**: a layer of physics-derived acoustic concepts
   (crackle score, wheeze score, rhonchi score, crackle rate, spectral flatness, and so
   on) computed by classical signal processing rather than learned. The idea was that a
   clinician could read the concept layer, see *why* the model decided what it decided,
   and correct a wrong concept value by hand.

This is the "concept bottleneck model" idea (Koh et al. 2020) applied to auscultation.
It is attractive here because crackles and wheezes have actual physical definitions, so
the concepts could be derived from acoustics rather than invented by a language model.

## 1.2 The gate, and why it existed

Before building the bottleneck, the team wrote down a **pre-registered gate**, called G2.
The rule, fixed in code before any test data was read:

> A concept is validated if its area under the ROC curve against the ICBHI label is
> **at least 0.65** *and* its 95 % DeLong confidence interval excludes 0.5.
> Budget: **two runs**. The test split may be read once per run.

Two details matter and both are in the paper.

- The **0.65 floor** is there because an earlier draft of the rule required only that the
  interval exclude chance. With about 2,600 test cycles that condition is met at an AUROC
  of 0.52 — statistically significant and practically useless. A gate that cannot fail is
  not a gate.
- The **two-run budget** is there so that the team could not keep revising the extractors
  until they passed. Every revision reads the test split; a third revision would have been
  fitting the test set, and any resulting pass would have meant nothing.

## 1.3 The gate failed, twice

| Concept | Run 1 | Run 2 | Verdict |
|---|---|---|---|
| crackle score | 0.5506 | 0.5580 | FAIL |
| wheeze score | 0.5729 | 0.5340 | FAIL |

Run 2 used extractors revised against five diagnosed failure modes from run 1. The
revisions worked *mechanically* — the crackle detector's firing rate dropped from ~10/s
to ~2.5/s, killing a saturation problem; the wheeze detector stopped being silent on half
of all wheeze cycles — but each fix traded one failure mode for its opposite (the wheeze
detector then fired on 99 % of everything). The team stopped, as the rule said.

**This is the decision record `DECISION_2026-08-16_PIVOT.md`.**

## 1.4 The pivot: looking for an explanation found five faults

Trying to explain the failure sent the team into its own evaluation code. Five faults
came out of a 39-run audit. They are Section III-A of the paper and they are the
contribution. Briefly:

1. The reported "ICBHI score" was a **macro variant**, not the challenge metric.
2. The split loader **silently fell back** to an 11-patient test set.
3. The published official split is **not patient-independent**.
4. One recording's filename does not match the released audio, so a join **drops it
   silently**.
5. The saved checkpoint was chosen by the **wrong rule**.

Correcting all of them moved the headline number from **0.7077 to 0.5602**.

## 1.5 The ceiling claim, and its retraction

After the gate failed, the team began to prefer a comfortable explanation: *ICBHI's cycle
labels are too unreliable for any detector to reach 0.65 — we found a ceiling.* Human
studies supported the mood: seven senior physicians reach an ICBHI score of 47.77 % on
this corpus (Tzeng et al. 2025), and twelve physicians agree at κ < 0.40 on fine
descriptions (Melbye et al. 2016).

Before writing that down, the team tested it — with the readings fixed in the script
**before** it ran (experiment N11). A logistic probe was fitted on the 79 training
patients and scored on the same 2,636 test cycles, in four representations:

| Representation | Crackle | Wheeze |
|---|---|---|
| our gate, one hand-built score | 0.5580 | 0.5340 |
| our 14 concepts as a vector | 0.6608 | 0.5815 |
| **frozen AudioSet transformer (never heard a lung sound)** | **0.7115** | **0.7621** |
| backbone trained on these labels | 0.7451 | 0.8673 |

The pre-registered "no ceiling" branch fired. **The ceiling claim was retracted.** The
labels support detection at 0.71–0.87; the binding constraint was the extractors, not the
reference standard.

**This is `DECISION_2026-08-30_CONSOLIDATION.md`.** Note what happened: the team wrote
down in advance the outcome that would refute them, got that outcome, and reported it.
That is the single most defensible thing in the paper and the reason the negative result
is worth publishing.

## 1.6 What replaced the ceiling claim, and why it is better

A frozen AudioSet model reproduces ICBHI's wheeze labels better than a self-consistent
physician does — measured on the *same* 132 clips, paired, +0.169, p = 0.046. That is a
sharper claim than "there is a ceiling", it uses evidence already on disk, and it does not
require a second clinician.

## 1.7 The course requirements pulled the paper back toward models

The course checklist (`paper_requirement.md`, transcribed from the supervisor's slides)
demands things the reliability story does not by itself produce: at least four transformer
models, one pretrained model per member, metrics and cost for every model, augmentation
results, XAI on the best model, an ablation study. So the team ran M40–M42 (three
transformers), M44 (XAI), M45 (twelve-row ablation), M48 (protocol ablation, seed band,
selection criterion, machine-vs-physician on the same clips).

**The paper you have is the join of those two demands.** The measurement story is the
spine; the model zoo hangs off it and is reported under the corrected protocol so it does
not contradict the spine.

---

# Part 2 — The spine of the argument, in one page

Read this before rewriting anything. Every section should serve one of these seven beats.

1. **ICBHI 2017 is the field's default benchmark and it is crowded** — well over a
   hundred published systems. A new architecture was never going to move it much.
2. **So we asked a prior question: does the reported score mean what people take it to
   mean?** We asked it of ourselves first.
3. **Five faults, each priced.** The two that matter compound to +0.1475. That is larger
   than pretraining, augmentation, and the spread between our best and worst backbone,
   combined. The unaudited pipeline said 0.7077; the corrected one says 0.5602.
4. **Under the corrected protocol we rebuild the model side honestly.** One CNN, three
   transformers, clean and augmented, identical everything else. Every transformer loses.
   The biggest one collapses to the majority class. We report it.
5. **We take the best model apart properly.** Twelve ablation rows, a measured noise
   floor that disqualifies three of them, a patient-level test that certifies only one,
   and attribution that contradicts the story the score tells.
6. **We return to the concept question and retract our own explanation.** The labels are
   learnable; our extractors were weak. What survives is sharper — machines reproduce
   this reference standard better than a physician does, on the same clips.
7. **The recommendation is procedural, not architectural.** Report the official metric
   with Se and Sp beside it, name the partition, commit the confusion matrix, state the
   selection rule, test over patients.

**The one-sentence version:** *On this benchmark, how you measure is worth more than what
you build, and we can prove it on our own pipeline.*

---

# Part 3 — Section-by-section walkthrough

Line numbers refer to `main.tex` as it stands. They will drift as you edit; use them as
a starting point, not an address.

---

## 3.0 Title, authors, abstract, keywords (lines 41–88)

### Title
> *Auditing the Score Before the Model: Corrected Evaluation Baselines and a
> Pre-Registered Negative Result for Respiratory Sound Classification on ICBHI 2017*

**Job.** Announce that this is a measurement paper, not a model paper, and name both
halves of the contribution (corrected baselines + a pre-registered negative result).

**If you rewrite it:** keep three things — the audit/evaluation idea, the word
*pre-registered*, and the dataset name. Do not promise a method. Do not use the word
"novel" — the paper argues its novelty by evidence, not by adjective.

### Author block (line 47)
Contains **`Farhana~[surname]` and `Sami~[surname]` placeholders you must fill in.**
The `\thanks{}` carries department, university, course code and supervisor. Keep the
supervisor line.

### Abstract (lines 56–81)

**Hard rules from the course rubric, all currently satisfied:**

| Rule | Status |
|---|---|
| 200–300 words | 293 |
| one paragraph | yes |
| order: purpose → methodology → results → conclusion | yes |
| mentions dataset, preprocessing, models, results, ablation, XAI | yes |
| **no citations** | none |
| **no abbreviations** | none — "ICBHI" never appears, it is "one public lung sound benchmark" |
| **no symbols** | no `$…$`, no `%` (the word "percent" is used) |
| **no equations** | none |

**Structure of the current abstract, sentence group by sentence group:**

1. *Setup* — the field ranks systems by one headline score.
2. *The turn* — "This project set out to build a better classifier and ended up asking a
   prior question instead." This sentence is the whole paper.
3. *The five faults*, each with its price: 9.5 points average / 22 worst; 11 patients
   worth almost 9 points; the leak worth four tenths of a point; the filename fault;
   7–10 points for the selection rule.
4. *What we then did* — cycles, log-mel, one CNN and three transformers, the full metric
   set, with and without masking.
5. *How the best model was examined* — ablation, seed floor, paired patient tests, XAI.
6. *The closing line* — "The contribution is a measurement, not a mechanism."

**If you rewrite it:** the abstract is written last, on purpose. Rewrite the body first,
then re-derive the abstract from it. Watch the word count (`verify.py` checks it) and do
not let an acronym slip in — "CNN", "ViT", "XAI", "AUROC" and "GPU" are all banned here.
Note the fourth fault has no number because it is a defect count, not a score; that is
deliberate, not an omission.

### Keywords (lines 83–88)
Twelve terms, **alphabetical** (checked by `verify.py`). If you add one, re-sort.

---

## 3.1 Introduction (lines 90–105)

**One paragraph, and its job is motivation** — the rubric asks specifically why *this*
capstone topic was chosen.

**The beats:**

1. Auscultation is the first thing a doctor does and one of the least reproducible.
2. What crackles and wheezes physically are, in plain words — short popping transients,
   longer musical tones — plus why they are hard: quiet, masked by heart sounds and room
   noise, described differently by different physicians.
3. ICBHI 2017 is the standard benchmark.
4. **The honest reason for the topic choice:** the benchmark is mature, a new
   architecture was never going to move it far, so it is a good place to ask a different
   question.
5. The question, and the sting in the tail — "we started by asking it of ourselves."

**Why it is written this way.** Most capstone introductions claim the problem is
important and the method is new. This one claims the problem is *saturated*, which is
what licenses the paper's actual contribution. Do not rewrite this into a
"respiratory disease is a major global burden" opener; that opener would make the rest of
the paper look like a failure instead of a choice.

**Safe to change:** all the wording. **Do not change:** the admission that the benchmark
is mature and that a new architecture was unlikely to help. That admission is load-bearing.

---

## 3.2 Literature Review (lines 106–165)

**Rubric requirement:** each of the four group members reviews at least one paper,
journal preferred, cited as `[1]`, `[2]`, … The reviews sit **inside** the Introduction,
not in a separate Related Works section (the supervisor's slide 8 says so explicitly).

**Structure: four themed blocks, each opening with a bold lead-in.** The themes are
ordered by when they became relevant to the project's own design, which is stated in the
one-line preamble.

### Block A — "Where the benchmark stands" (lines 111–125)
- `electronics2025review` — the survey. Cited for "well over a hundred published systems"
  and for the pattern that progress since ~2021 came from pretraining, not architecture.
- The two-tier picture: ImageNet-pretrained CNNs plateau near 0.56 (`respirenet2021`);
  AudioSet-pretrained transformers sit 5–9 points higher
  (`bae2023patchmix`, `bts2024`, `pafa2025`).
- `suma2025mtl` — the multi-task paper closest to the project's original design. It is
  cited **and criticised in the same breath**: it reports 74 % and 91 % without saying
  which partition. That criticism is the paper's first foreshadowing of its own theme.
- `adffnet2025`, `bspc2025fusion` — two 2025 fusion systems, cited for reporting gains of
  roughly one point, "which is the scale at which the measurement questions below start
  to matter."

**The rhetorical move:** the review is not a list, it sets up the argument. Note how the
last clause of the block ties the literature to the paper's thesis. Keep that move.

### Block B — "What pretraining is worth" (lines 127–138)
- `bae2023patchmix` at 0.6237 (fine-tuned AST) versus `frozenfm2025` at 0.5938
  (frozen model, linear probe only).
- The synthesis: most of the useful information is in the pretrained representation, not
  the fine-tuning, **provided the pretraining was on audio**.
- `scl2023` shows the same at a smaller parameter budget; `nguyen2022cotuning` warns that
  ImageNet→spectrogram transfer is fragile.

**Why this block exists.** It predicts the paper's own transformer result before it is
reported. When Section III shows all three ImageNet-pretrained transformers losing, the
reader has already been told why. Do not delete this block — it converts a disappointing
result into a confirmed prediction.

### Block C — "Interpretability and open-world behaviour" (lines 140–153)
- `koh2020cbm` — the concept bottleneck idea, and the sentence "this was our original
  plan" that connects the literature to the project history.
- `karim2024mosquito` — **the supervisor's own department's work**; vision transformers +
  OpenMax rejection, reimplemented as the project's open-set baseline. This citation does
  institutional work as well as scientific work; keep it.
- `pavel2024kd` — knowledge distillation with a teaching assistant; the compression arm.
- `yang2024ood` — the taxonomy separating OOD detection from open-set recognition.
- `kejriwal2024owl` — the staged open-world protocol.
- `cho2025openset` — the only open-set work on respiratory audio, and it reports a
  relative improvement rather than an absolute score (which is why it is later excluded
  from the comparison table).

### Block D — "How well humans do the same task" (lines 155–165)
Opens with "This was the most important group we read", which flags it.

- `tzeng2025jmir` — 7 senior physicians, 25 % of the ICBHI test set, blind. Score 47.77 %,
  sensitivity 23.23 %, confidence 2.88/5.
- `melbye2016kappa` — 12 physicians, 20 reference recordings. κ < 0.40 for detailed
  descriptions, 0.62 and 0.59 when collapsed to simple presence.
- `huang2024crackles` — a larger clinical archive; wheeze κ 0.948, crackle κ 0.516.
- The synthesis sentence: *the three disagree on the level and agree on the shape —
  crackle labels are far less reliable than wheeze labels, and reliability falls sharply
  as the description gets finer.*
- `scirep2025cnnrnn` — applies generic saliency to combined corpora without asking what
  the labels support; this is the gap the paper's XAI section is careful about.

**Citation trap.** This paper is cited project-wide in older documents as
"Aviles-Solis et al. 2016". **The first author is Melbye.** Aviles-Solis is a co-author on
other lung-sound papers from the same Tromsø group. The `.bib` entry is correct; do not
"fix" it back.

**If you rewrite this section:** keep the four-block structure and keep the synthesis
sentence at the end of Block D — the whole of Section III-G depends on it.

---

## 3.3 Gaps in the Related Work (lines 166–191)

**Rubric requirement:** a required paragraph stating the gaps, limitations and weaknesses
of the related work, at the end of the literature review.

**Four numbered gaps, each one paragraph, each phrased as an italicised claim followed by
its evidence:**

| # | Gap | What the paper does about it |
|---|---|---|
| 1 | The reported score is treated as a property of the model. No ICBHI paper ablates its evaluation protocol the way it ablates its architecture. | `tab:protocol` — the protocol ablation |
| 2 | The official split is assumed patient-independent and is not. | `tab:corpus`, the corrected partition, and row `E3` |
| 3 | The reference standard is never characterised. | `tab:human`, `tab:ceiling`, `tab:sameclips` |
| 4 | Negative results are missing. Nobody fixes a validity threshold in advance and reports the failure. | `tab:gate` |

**This mapping is the design of the paper.** Each gap has a table that closes it. If you
rewrite the gaps, check that each still points at its table.

**Tone note.** Gap 1 says "we could not find an ICBHI paper that…" rather than "no paper
does". That hedge is deliberate and correct — the team searched, it did not exhaustively
audit the literature. Keep the hedge.

---

## 3.4 Our Work (lines 192–238)

**Rubric requirement:** a brief description of your own work, 3–4 paragraphs,
second-to-last block of the Introduction; then a roadmap paragraph last.

**Five paragraphs:**

1. **The original plan and the gate.** Describes the bottleneck design, then the gate
   rule, then both failures with all four numbers, then "we stopped, as the gate said we
   would."
2. **The pivot.** The audit, the five faults, the 0.7077 → 0.5602 move, and the framing
   sentence: *"What we deliver is therefore not a mechanism but a measurement."*
3. **The model side rebuilt.** One CNN, three transformers, identical training,
   clean+augmented pairs; the best model taken apart four ways; every transformer lost and
   the largest collapsed; *"we report that rather than tuning until it stops being true."*
   Then one sentence pointing forward to the retraction.
4. **The novelty paragraph** — five numbered items, (i) to (v). This is the paragraph the
   supervisor will read to decide whether the paper is novel. Reproduced here in full
   because you must not weaken it:
   - (i) an ablation of the evaluation protocol itself, not seen on this benchmark;
   - (ii) the checkpoint selection rule identified as a first-class pipeline component,
     worth more than pretraining or augmentation;
   - (iii) a measured run-to-run noise floor used as an admissibility threshold for every
     ablation delta reported;
   - (iv) a pre-registered gate that failed, followed by a pre-registered ceiling estimate
     that refuted the team's own account of why it failed;
   - (v) an attribution analysis that contradicts the story the score tells, and which
     directly motivated one of the ablation rows.
5. **The roadmap** — what is in Section II, Section III, Section IV.

**Rewriting guidance.** Paragraph 4 is the most important paragraph in the Introduction.
Each of the five items is a *claim of novelty backed by a specific table*:

| Item | Table that proves it |
|---|---|
| (i) protocol ablation | `tab:protocol` |
| (ii) selection rule | `tab:selection` |
| (iii) noise floor as threshold | `tab:stats` upper block, and the "vs noise" column of `tab:ablation` |
| (iv) pre-registration and retraction | `tab:gate`, `tab:ceiling` |
| (v) contradictory attribution | `tab:xaiquant` |

If you reword an item, keep the pointer intact. If you add a sixth item, it needs a table.

---

## 3.5 Section II opening + flowchart (lines 240–258)

**Rubric requirement, and the trap:** *"Do not show any results in this section."* This is
an explicit rule. The opening sentence therefore says: *"Everything numeric that is a
result is held back for Section III; the counts here are properties of the data."*

That sentence is a shield. It lets the section contain 6,898, 2,636, 47 and so on without
breaking the rule, because those are dataset facts, not outcomes. **Do not put a score,
an accuracy, or a delta in Section II.**

**`fig:flowchart`** — required by the rubric ("include a flowchart of the complete
system"). It is generated by `make_diagrams.py`, which reads its constants from the
committed results files. The caption makes one substantive point beyond describing the
picture: the split loader, the metric definition and the checkpoint selection rule are
drawn as pipeline components with their own ablation rows *rather than as fixed
infrastructure*. That is the paper's thesis in a diagram.

---

## 3.6 Dataset (lines 259–351)

### Prose (lines 261–271)
Corpus description: 920 recordings, 126 patients, per-cycle annotation with start time,
end time, crackle flag, wheeze flag; the two flags combine into four classes; 6,898
annotated cycles; one diagnosis per patient; four stethoscopes; mixed sample rates
resampled to 16 kHz.

### The corpus table was **cut** as repetitive
An earlier version had a `tab:corpus` float carrying corpus properties and the diagnosis
distribution. Almost all of it repeated the surrounding prose, so it was removed and its two
load-bearing facts moved into a short paragraph:

- **COPD covers 64 of 126 patients and 83.3 % of all cycles** — which is why the disease head
  was dropped from the headline;
- **only 4 patients (112, 158, 218, 226) were recorded on more than one device** — so device
  identity is almost perfectly confounded with patient identity and no device-robustness
  study is possible on this corpus.

Keep both facts if you rewrite the paragraph. Everything else in that table (920 recordings,
126 patients, 6,898 cycles, 4 classes, 16 kHz, median 2.42 s, 8.0 s input) is already stated
in the prose or in `tab:partitions`.

### `tab:partitions` — partitions **and** class distribution, merged into one float
This float does the job of two. It carries, for each partition, the class counts, the totals,
the augmented total, the patient count and the recording count.

**The partitions, and why each is in the paper:**

| Partition | Policy | Test | Why it exists in the paper |
|---|---|---|---|
| Published, verbatim | none | 49 patients, 2,756 cycles | the only partition comparable to the literature |
| **Corrected (ours)** | reassign leaking patients to train | 47 patients, 2,636 cycles | **every new result** |
| Identifier fallback | — a fault | 11 patients, 492 cycles | audited, priced, never reported as a result |
| *(patient-disjoint alt.)* | drop leaking patients' train recordings | identical to the published test row | named in the note, not given a row |

The fourth policy has no row of its own because its test set is **byte-identical** to the
published one — that is the whole point of dropping from train rather than reassigning.

**This table is the fix for a real conflict.** Several repository documents put M2/M3
(patient-disjoint alternative) and M22-v2 (corrected) in one column. They are *different
test sets*: 2,756 versus 2,636 cycles. The policy column exists so nobody does that again.

**Footnote content:** COPD covers 64/126 patients and 83.3 % of cycles (which is why the
disease head was dropped from the headline); only 4 patients (112, 158, 218, 226) used
more than one device, so no device-robustness study is possible; the leak repair costs
12 of 381 test recordings.

**Trap.** Older documents say **3** device-spanning patients. It is **4**. Recomputed from
the annotation index.

**The rubric's slide-10 format is satisfied by this merged table:** class counts per
partition with an **augmented total** column. Corpus 3,642 / 1,864 / 886 / 506 = 6,898.

**Why the augmented total equals the clean total.** SpecAugment masks spectrograms in place;
it does not create new samples. Say this, or a reader will think the column is broken.

**The 11-cycle discrepancy, explained in the footnote.** The annotations hold 4,262
training cycles for the corrected partition; the pipeline indexes 4,251. The difference is
one recording named `226_1b1_Pl_sc_Meditron` in the split file and released as
`226_1b1_Pl_sc_LittC2SE`. It is on the training side, so no test partition is affected.
This is Fault 4, previewed here as a data fact and priced later as a fault.

### The accuracy warning (lines 350–355 area)
*Always predicting Normal scores 0.5918 accuracy on the corrected test partition, which is
higher than the accuracy of our best model.* (1,560 / 2,636 = 0.5918.)

This sentence sets up the ViT-collapse result in Section III. Keep it here, in the
methodology, so the later result lands as confirmation rather than as a surprise.

---

## 3.7 Dataset Preprocessing (lines 352–453)

### The preprocessing-chain table was **cut** as repetitive
An earlier version had a thirteen-row `tab:preproc`. The same information sits in the
equations and in the "change from the reference" column of `tab:ablation`, so the chain is now
a single prose sentence listing the stages in execution order, each pointing at its equation.

**Two things that paragraph must keep:**

1. **The stage list, in order**, with the settings: 16 kHz mono; optional zero-phase
   fourth-order Butterworth 50–2000 Hz; cycle segmentation from the annotations; 8.0 s by
   cyclic tiling (1); optional per-cycle peak normalisation; optional spectral gating at the
   25th percentile with over-subtraction factor two; log-mel with 128 bands, 1024-point FFT,
   hop 160, window 400 over 50–2000 Hz (2); min–max rescale (3); replicate to three channels;
   label encoding (4); weighted loss (5); spectrogram masking on the training split only (6).
2. **Three stages are deliberately OFF in the reference** — band-pass, denoising, amplitude
   normalisation — because turning any of them on would silently move the baseline that every
   other ablation delta is measured against, invalidating rows already run. This is why rows
   `P1`–`P3` are **ADD** rows and every other row is a REMOVE row. Their deltas read in
   opposite directions: a positive delta on an ADD row is a recommendation to adopt the stage.
   **If you rewrite the ablation discussion, do not lose this.**

### The equations

The rubric requires equations in an `equation` environment, numbered. There are nine.
Full reference in Part 6; here is what each one is *for* in the argument:

| Eq. | What | Argumentative job |
|---|---|---|
| (1) `eq:tile` | cyclic tiling to 8 s | sets up row `P4` and the XAI tiling result |
| (2) `eq:logmel` | log-mel with a **per-spectrogram max reference** | explains why `P5` is nearly free |
| (3) `eq:minmax` | min–max to [0,1] | the actual normalisation used |
| (4) `eq:label` | `y = c + 2w` | defines the four classes from two flags |
| (5) `eq:weights` | inverse-frequency weights **renormalised to unit mean** | makes ablation rows comparable |
| (6) `eq:specaug` | SpecAugment masking | the augmentation |
| (7) `eq:official` | the ICBHI challenge metric | Fault 1's correct definition |
| (8) `eq:macro` | the macro variant | Fault 1's wrong definition |
| (9) `eq:gate` | the pre-registered gate rule | the pre-registration |

**Three traps that were fixed in this version and must stay fixed:**

1. **(2) has `ref=max`, not a plain log.** The code is
   `librosa.power_to_db(m, ref=np.max)`. An earlier draft wrote a plain log with an
   epsilon floor. That is a different operation, and it matters: `ref=max` is already a
   normalisation, which is *why* the min–max stage that follows is worth only −0.0063 in
   the ablation. Get this wrong and the ablation stops making sense.
2. **(3) is min–max, not zero-mean/unit-variance.** The code is
   `(lm - lm.min()) / (lm.max() - lm.min() + 1e-8)`. The ablation row is literally named
   "− per-spectrogram min-max normalisation".
3. **(5) includes the renormalisation to unit mean.** The code is
   `w = c.sum()/(4*c); w = w / w.mean()`. Without the second line the loss scale moves with
   the class distribution, and ablation rows that change the training set stop being
   comparable.

**Rewriting guidance for the prose glue.** Each equation is introduced, stated, then given
one sentence of consequence. Keep that rhythm. The consequence sentences are where the
marks are — for instance, after (1): *"Cyclic tiling is common practice here, but it
repeats the same acoustic event two or three times inside one input and leaves a
discontinuity at each seam"*, which is what makes the later XAI finding interpretable.

---

## 3.8 Applied Models (lines 455–500)

### Prose (lines 457–475)
Two things:

1. **The design rule** — the four backbones share spectrograms, loss, partition, batch
   size, epoch cap and seed, "since anything else would turn the comparison into a
   comparison of tuning effort."
2. **The honest admission that the two families differ in optimiser settings**:
   - CNN: Adam, single learning rate **5e-4**, dropout 0.3, gradient clipping 5.0.
   - Transformers: AdamW, split rate **1e-4 backbone / 1e-3 head**.
   - Both: cosine annealing, weight decay 1e-4.
   - The ablation harness reuses the CNN settings unchanged.

**Why this admission is in the paper.** It would have been easy to write "identical
schedules" and move on. It would also have been false. The split-rate recipe is standard
for fine-tuning a pretrained transformer and is what their reference implementations
assume; using the CNN's single rate would have handicapped them. Saying so is better than
hiding it — and a supervisor who checks the JSONs will find it.

### `tab:models`
Eight rows in two blocks. **Columns:** backbone, family, pretraining, **owner**, protocol and
status. There is no identifier column any more — models are named by architecture.

- **Block 1, the corrected protocol:** MobileNetV2 (Asif Mahbub), ViT-B/16 (Barshon Basak),
  Swin-T (Farhana Rahman), DeiT-S (Sami Uddin).
- **Block 2:** AST (Asif Mahbub, Barshon Basak) marked *pre-audit; score NOT RECOVERABLE*, the
  2D-CNN and MobileNetV3-Small references, and the five-member ensemble (Sami Uddin).

**The owner column satisfies the rubric's "one pretrained model per member", and all four
transformers now exist** — ViT-B/16, Swin-T, DeiT-S and AST. That closes the four-transformer
requirement, but only on a technicality the paper states openly: the AST run predates the
audit and its challenge score cannot be recomputed. Do not quietly upgrade "four transformers
were run" into "four transformers are comparable" — they are not, and Section III-D exists to
say so.

---

## 3.9 Evaluation Metrics (lines 502–545)

**Job.** Define the metric before any result uses it, so Fault 1 in Section III has
something to point at.

Contains (7) the challenge metric, (8) the macro variant, and (9) the gate rule, plus:

- **Why the two metrics differ**, in one sentence that is the crux of Fault 1: *each
  per-class specificity in (8) counts true negatives drawn from the other three classes,
  so a rare class such as Both scores near 0.95 almost regardless of whether the model
  ever predicts it.* Understand this sentence; it is the mechanism behind a 22-point
  inflation.
- **The full metric list** reported for every model (accuracy, macro precision, macro
  recall, macro F1, parameters, size, training time) — this is the rubric's requirements 4
  and 5, promised here and delivered in `tab:main` and `tab:efficiency`.
- **The statistical machinery**: intervals resample **patients**, not cycles, 2,000 times;
  paired comparisons use a patient-level bootstrap plus McNemar where per-cycle
  predictions exist for both.
- **The gate rule and the justification of the 0.65 floor**, ending on the line *"A gate
  that cannot fail is not a gate."*

**Rewriting guidance.** This subsection is where you can save space if you need it, but do
not delete the explanation of *why* (8) inflates. Without it, `tab:metricaudit` is a table
of unexplained numbers.

---

## 3.10 Section III opening — experimental setup (lines 547–557)

**Rubric requirement: state where training was performed.** Two machines:

- **Tesla T4 (16 GB)** on Kaggle and Google Colab — the convolutional runs, the seed
  replicates, three preprocessing ablation rows.
- **NVIDIA GeForce RTX 4050 Laptop (6 GB)** — the transformer runs and seven ablation rows.
- PyTorch 2.10, CUDA 12.8.

Plus the evidence rule: every run writes one result file holding its configuration, its
raw 4×4 confusion matrix, its per-epoch history and its efficiency record, and every
number in the section was read from or recomputed from one of those files.

**Trap you must not reintroduce.** The two GPUs differ by roughly a factor of two on this
workload. **Never compare a T4 time against a 4050 time.** An earlier version said "the
transformer takes 2.4 times as long to train" — that compared 2,172.8 s on the laptop
against 892.0 s on the T4 and was meaningless. The efficiency table is split into two
device blocks for exactly this reason.

---

## 3.11 Five Faults (lines 558–708) — the core of the paper

Opening line: *"We audited 39 committed runs. Five faults came out. None is exotic, and
all five are the kind of thing that hides inside working code, which is why they are worth
reporting."*

**The framing discipline, and it is not optional.** Every fault is presented as *an error
we made and caught*, never as an accusation about other groups. This is both honest and
more persuasive: the implicit argument — if a team auditing itself this hard made all
five, the wider literature probably contains some of them — lands harder unstated. **Do
not make it stated.**

### Fault 1 — the metric (lines 565–614)

The pipeline computed (8) and called it the ICBHI score; the challenge metric is (7). Both
were recomputed from every run's stored confusion matrix.

**`tab:metricaudit`** — 20 runs, sorted by corrected score, with a **partition column** so
nobody ranks across partitions. Summary: mean +0.0948, max +0.2176, min +0.0025.

**The paragraph after the table is the important one.** The damaging property of the macro
variant is not that it is high but that it **hides pathologies**:

| Run | Reported under (8) | What was actually happening |
|---|---|---|
| M33 | 0.5506 | specificity **0.0000** — never once predicted Normal correctly |
| M21 | 0.5022 | sensitivity **0.0186** |
| M36 | 0.5137 | detected 9 % of abnormal cycles |

Under (8) all three look like slightly weak models. Under (7) all three are visibly broken.
That is the argument for the correction: it is a *diagnostic* argument, not just an
arithmetic one.

### Fault 2 — the split loader (lines 616–623)

The loader searched for the split file with the wrong filename casing using an exact path
test. On a case-sensitive filesystem it never matched and fell through to a documented
fallback: every patient with identifier ≤ 111 goes to test. Result: **11 test patients,
492 of 6,898 cycles — 7.1 %, not 40 %.**

**The line to keep:** the record written beside the run said
`patient_independent_official_60_40`, *"because the string was a literal and a literal
cannot disagree with the data."* That is the generalisable lesson and it is one sentence.

Four models carried this fault.

### Fault 3 — the split is not patient-independent (lines 625–627)

Patients 156 and 218 appear on both sides of the published file, which assigns
*recordings*. Three sentences only, ending: *"We report this one carefully, because it is
the fault that costs almost nothing."* That sets up row `E3`.

### Fault 4 — the filename mismatch (lines 629–633)

`226_1b1_Pl_sc_Meditron` in the split file is `226_1b1_Pl_sc_LittC2SE` on disk. A stem join
drops it silently. It is a training recording, so no score is affected — *"but a pipeline
that discards data without complaint can discard more."*

### Fault 5 — the checkpoint selection rule (lines 635–668)

**This is the paper's second-largest effect and its most original finding.**

The mechanism: the training loop keeps the "best" epoch, and which epoch is best depends
on the rule. Cross-entropy on a corpus that is 59 % Normal does not have its minimum where
balanced sensitivity and specificity have their maximum.

**`tab:selection`** — the same three runs scored both ways:

| Seed | By challenge score | By minimum loss | Δ | Epochs apart |
|---|---|---|---|---|
| 42 | 0.5540 | 0.4788 | +0.0752 | 15 |
| 1 | 0.5681 | 0.4666 | +0.1015 | 19 |
| 2 | 0.5657 | 0.4665 | +0.0992 | 17 |
| random init, frozen | 0.4979 | 0.4979 | 0.0000 | 0 |

**The last row is a control and you must keep it.** With a randomly initialised frozen
backbone there is nothing to select between, so the two rules coincide exactly. That rules
out the possibility that the effect is an artefact of the measurement.

**The caveat that must travel with this table, always.** Both criteria are applied to the
same monitoring partition, which in this harness *is the test partition*. That makes both
columns optimistic and **equally so**, which is why the *difference* is the reportable
quantity. It is not a clean train/validation/test result and must never be presented as
one. This caveat is in the table note and is repeated as the first item of
`subsec:limitations`.

**The comparison that makes the point:** selecting on loss costs 0.075–0.102. ImageNet
pretraining is worth 0.0603. SpecAugment is worth 0.0402. *A rule most papers do not state
is worth more than either of the two things they do.*

### "What the whole correction is worth" (lines 670–708)

**`tab:protocol` — the single most important table in the paper.**

The insight that generates it: the faults can be switched on and off one at a time, which
makes the evaluation protocol **ablatable in the same way a model is**. Nobody appears to
have done that on this benchmark.

| Row | Removed | Kind | Reported | Δ |
|---|---|---|---|---|
| E0 | nothing | baseline | 0.5602 | — |
| E1 | official metric → macro | rescore | 0.6140 | +0.0538 |
| E2 | official split → fallback | rerun | 0.6495 | +0.0893 |
| E3 | patient independence | rerun | 0.5641 | +0.0039 |
| E4 | device-safe join | count | 1 recording lost | — |
| E5 | patient-level unit | rescore | 7/10 significant vs 1/10 | — |
| **E1+E2** | **both — the unaudited pipeline** | rescore | **0.7077** | **+0.1475** |

**Two kinds of row, and the distinction is real.** *Rescore* rows apply a different rule to
fixed predictions — exactly one variable changes and there is no seed noise at all.
*Rerun* rows change the partition, which changes the test set, so the model is retrained
with everything else held fixed. Keep the "Kind" column.

**The two rows to emphasise in prose:**

- **E1+E2 = 0.7077.** Read against the literature that would have placed first, ahead of
  every published system — on 7.1 % of the corpus under a metric the team defined itself.
- **E3 = +0.0039.** The leak, the fault that sounds worst and gets papers rejected, is
  worth four tenths of a point — inside the measured noise floor.

**Why reporting E3 as a null is a strength, not a weakness.** Two reasons, both in the
text: (a) conceding that one of the four faults costs essentially nothing is what makes
the other two credible; (b) it separates two things usually conflated — a leak is a
**validity** problem, not necessarily an **inflation** problem, and it invalidates the
claim of patient independence regardless of what it does to the number.

**If you rewrite one paragraph in this paper with care, make it this one.**

---

## 3.11b The Fourth Transformer (Section III-D, `subsec:ast`) — **new in v2**

**Why this subsection exists.** The literature says the systems that do well on ICBHI are
pretrained on *audio*, and three of our four transformers are pretrained on *images*. The
obvious experiment is an Audio Spectrogram Transformer. We ran one — early in the project,
months before the evaluation audit.

**What the run recorded**, and these are the only four numbers that survive: accuracy 0.5528,
macro precision 0.5347, macro recall 0.4385, macro F1 0.4109. Plus: 86.4 M parameters,
329.5 MB, 86.74 ms per sample on a Tesla T4, and a macro-variant score of 0.6359.

**Four things stand between it and the main table, and the subsection names all four:**

1. **The partition** is the identifier-fallback one that Fault 2 produced — 492 cycles from
   11 patients.
2. **The metric** is the macro variant that Fault 1 describes.
3. **The preprocessing** is not the chain of Section II — it used a full-band 20–8000 Hz mel
   filterbank instead of 50–2000 Hz, and a 512-point FFT instead of 1024, both deliberate
   deviations meant to preserve the pretraining.
4. **It committed no raw confusion matrix.** Its numbers survive only as figures transcribed
   from a notebook's output into a later backbone-selection record.

**The fourth is the point of the subsection, and it is the strongest rhetorical moment in the
paper.** Every other score in the paper can be recomputed from a stored 4×4 matrix by a reader
with a calculator. This one cannot be recomputed by anybody, *including us*. We cannot convert
it to the challenge metric, give it a confidence interval, or place it in the metric-audit
table. The paper's own recommendation — commit the raw confusion matrix — is not a tidy
general principle; it is the rule the authors wish they had followed on the one model the
literature suggests should have won.

**Comparisons that ARE legitimate**, because they are internal to that study: on its own
partition under its own metric, AST placed **last** of the three backbones evaluated, at
roughly 24 times the parameters and 30 times the inference latency of the convolutional model
that beat it. Say that; it is a real result about the accuracy/compute trade-off.

**Rewriting traps.** Never give AST a challenge-metric score. Never put its 0.6359 in a table
of official-metric numbers except in the clearly-labelled fallback block of
`tab:comparison`. Never write "four comparable transformers". And do not soften "cannot be
recomputed by anybody, including us" — that admission is why the subsection is worth its
space.

---

## 3.12 Main Results

### `tab:main`
Eight model rows on the corrected partition — 2,636 test cycles, 47 patients — an
**always-Normal baseline row**, and then a clearly separated lower block holding the pre-audit
AST row with `NOT RECOVERABLE` in its $S_e$, $S_p$ and ICBHI columns.

Columns: model, backbone, augmented?, accuracy, macro precision, macro recall, macro F1,
Se, Sp, ICBHI with a 95 % patient-bootstrap interval.

**The always-Normal row is not decoration.** It is 0.5918 accuracy, 0.0000 Se, 1.0000 Sp,
0.5000 ICBHI. Comparing the two ViT rows against it *is* the ViT finding.

### The three discussion beats (lines 772–800 area)

1. **Every transformer lost.** Swin-T 0.5304 versus MobileNetV2 0.5602, at 12× the
   parameters; DeiT-S a further 1.5 points back. The explanation is already in the
   literature review: every system above 0.62 is AudioSet-pretrained and none of ours is.
   Then the discipline statement: *"We deliberately did not tune the transformers until
   one won, because that would have made the comparison a comparison of effort."*
2. **ViT-B/16 collapsed.** Sensitivity exactly 0.0000. A bootstrap interval of width 0.002
   *because there is no variation left to resample*. 2,634 of 2,636 predictions in one
   column. 85.8 M parameters against 4,262 training cycles. The augmented run collapsed
   identically. Reported rather than deleted — and flagged as possibly a schedule artefact
   because the warm-up re-run was never done, which is the **weaker** of the two available
   claims.
3. **Nothing here is close to the frontier.** 0.5602 corrected, 0.5641 published, against
   0.62–0.65 for audio-pretrained transformers. With the forward pointer: the
   decomposition of that gap in `subsec:comparison` is more interesting than the gap.

**Rewriting guidance.** Each beat is *result → mechanism → discipline statement*. The
discipline statements ("we did not tune until one won", "we report it rather than deleting
it", "the claim we make is the weaker one") are what turn three disappointing results into
three credible ones. Keep them.

### `tab:efficiency` (lines 771–812)
**Rubric requirement (slide 15):** epochs, time, parameters, size for *all* models, plus
the training hardware named.

**Structured as two device blocks.** Within-device comparison only.

**Two footnote points:**
- The **34,164-parameter discrepancy is explained**: the ablation harness reports
  2,263,160 where the notebook reports 2,228,996, and MobileNetV2 holds exactly 34,164
  batch-normalisation buffer elements (running mean and variance). The harness counts
  buffers; the notebook counts learnable parameters. Same network. *(This was an open,
  unexplained item in the earlier draft.)*
- The ten ablation rows appear in **both** device blocks because they were split across
  the two machines, and the faster laptop rows are the ones that shorten the input
  (64 mel bins, 4 s cycles) or freeze the backbone.

### The efficiency discussion (lines 814–819)
Two paragraphs, and the second one is a genuine finding:

- Parameters and size: ViT-B/16 is 38× the parameters and 37× the disk footprint of
  MobileNetV2 and scores below a constant predictor; Swin-T is 12× the parameters for
  three points less. *"At this data scale, parameter count buys nothing in accuracy."*
- **But it does not straightforwardly buy slowness either.** Within the laptop GPU,
  MobileNetV2 costs about 21–23 s/epoch while **DeiT-S costs 14.1** — faster per epoch
  despite ten times the parameters. Depthwise-separable convolutions are
  parameter-efficient but keep a GPU poorly occupied; a transformer's work is dense matrix
  multiplication. Only ViT-B/16, at 47.5 s/epoch, is slow in both senses.
- The conclusion: parameter count is a poor proxy for training cost here, which is one
  more reason to report time and size as separate columns.

**This paragraph is a small original observation and it costs nothing. Keep it.**

---

## 3.13 Data Augmentation

**Rubric requirement 7:** augmentation on training samples, with the same metric set reported
again. **That requirement is satisfied by `tab:main`,** which already carries both runs of
every backbone with the full metric set.

### The augmentation table was **cut** as repetitive
An earlier version had a `tab:augmentation` that re-presented numbers already in `tab:main`
with a Δ column. It is now a short subsection that reads the pairs off `tab:main` in prose.

**The paired design is what makes the deltas mean anything:** same seed, same everything, one
variable.

| Backbone | ICBHI Δ | Verdict |
|---|---|---|
| MobileNetV2 | **+0.0402** | 2.9× the noise floor |
| Swin-T | −0.0013 | below the floor |
| DeiT-S | −0.0168 | 1.2× |
| ViT-B/16 | +0.0006 | below; both runs collapsed |

### The discussion
SpecAugment helps the CNN and not the transformers. Then the interesting bit: on the two
transformers that did not collapse, **the composite is flat or down while macro F1 goes
up**. The mechanism: masking makes them spread predictions across the rare classes, which
improves per-class balance and costs sensitivity on the pooled abnormal category the
composite is built from.

*"Reporting both metrics is what makes that visible."* — this is the paper's theme applied
to its own results. Keep it.

**Trap.** Do not write "macro F1 rose in all three transformer pairs". ViT's F1 moved
0.1857 → 0.1859, which is meaningless because neither run left the majority class. The
text says "the two transformers that did not collapse" and adds a sentence saying ViT's
pair says nothing about augmentation.

**The ensemble rows** sit on a different partition with a calibration carve-out and commit
no confusion matrix, so they are reported on accuracy and macro F1 only and are explicitly
not rankable against the rows above. Keep that caveat in the note.

---

## 3.14 The Best Model in Detail (lines 866–931)

**Rubric requirement 6:** normalized confusion matrix, and loss/accuracy versus epoch, for
the **best** model only.

**The best-model rule is stated in advance and in the text:** *highest challenge score on
the corrected partition among runs with a committed confusion matrix.* That is
MobileNetV2 + SpecAugment at 0.5602. Stating the rule matters because ablation row `P3`
scores higher (0.5764) and is deliberately not promoted — see §3.19.

- **`fig:cm`** — row-normalised 4×4. Caption makes two observations: the diagonal falls
  monotonically with class frequency (0.7115 / 0.5186 / 0.2949 / 0.1163), and the heaviest
  off-diagonal mass is the entire abnormal column leaking into Normal.
- **`fig:curves`** — train/validation loss and accuracy against epoch with the selected
  epoch marked. The caption ties it back to Fault 5: loss keeps falling after the monitored
  score stops improving, which is the disagreement `tab:selection` prices.
- Both figures are regenerated from the same result file that produced `tab:main`, *"so
  neither can drift from it."*

### `tab:errors` — per-class plus error analysis (lines 896–928)

**Upper block:** per-class support, predicted count, precision, recall, F1, specificity.
Note Wheeze is under-predicted (224 predicted vs 373 true, −39.9 %) and Crackle
over-predicted (711 vs 617, +15.2 %).

**Lower block, and this is the best thing in the section:**

| Question | Answer |
|---|---|
| abnormal cycles called Normal | 479 of 1,076 (44.5 %) |
| abnormal cycles called *some* abnormal class | 597 of 1,076 (55.5 %) |
| abnormal cycles called the *right* abnormal class | 440 of 1,076 (40.9 %) |
| Normal cycles called abnormal | 450 of 1,560 (28.8 %) |
| **4-class task** | Se 0.4089, Sp 0.7115 → **0.5602** |
| **detect-only task** | Se 0.5548, Sp 0.7115 → **0.6332** |
| **cost of sub-typing** | **0.0730** |

**The argument.** The model flags 597 of 1,076 abnormal cycles as abnormal but types only
440 correctly. If the task were reduced to abnormal-versus-normal, the *same predictions*
would score 0.6332 instead of 0.5602. Roughly 40 % of the remaining error is not a failure
to hear anything — it is a failure to tell a crackle from a wheeze.

**And then the connection that earns the marks:** that maps directly onto the human
evidence. Melbye et al. report physician agreement of 0.62 and 0.59 for *presence* and
below 0.40 for finer description; Huang et al. report the same split between wheeze and
crackle. **The model fails where the annotators disagree.**

All of these numbers are recomputed from the same committed confusion matrix
`[[1110,333,81,36],[269,320,8,20],[178,39,110,46],[32,19,25,10]]`, so you can verify any
of them by hand.

---

## 3.15 Extension Experiments (lines 933–980)

**Rubric requirement 8:** novelties, reported with the same metric set.

### `tab:extensions`
Nine extensions, each with **its partition in its own column**. Most predate the split
correction and sit on a 70/30 partition the audit later invalidated for comparison
purposes, so none of them is rankable against `tab:main`.

**The framing sentence, which is the whole justification for including them:**
*"suppressing an experiment because its partition turned out to be wrong is a worse habit
than reporting it with a label."*

| Extension | ICBHI | Partition | Note |
|---|---|---|---|
| Physics-informed auxiliary loss (M35) | 0.6864 | 70/30 | highest score on any partition |
| Physics-informed, repeated (M35-v2) | 0.6719 | 70/30 | **best epoch is epoch 1** |
| LoRA adaptation (M37) | 0.6753 | 70/30 | 8,356 of 3.64 M parameters trained |
| Temporal transformer (M33-v2) | 0.5832 | 70/30 | first attempt had Sp = 0 |
| Curriculum learning (M34) | 0.5754 | 70/30 | |
| Dynamic multi-task weighting (M31) | 0.5535 | 70/30 | |
| Multistage distillation (M36) | 0.5052 | 70/30 | abnormal detection collapsed |
| Demographic fusion (M32) | 0.4733 | 70/30 | |
| Gated two-backbone fusion (M30) | 0.5975 | fallback | loses to either backbone alone |

**The footnote is doing the honest work:** *"The two highest numbers in this paper are here
and neither is claimable."* M35 sits on a partition whose test set is not ours, and its
repeat run selected epoch 1 — meaning it never really trained. LoRA is the one the team
would defend: 0.23 % of parameters trained for a score within 0.011 of the full fine-tune
on the same partition.

### The open-set paragraph
Reimplementing OpenMax after `karim2024mosquito` gave an unknown-detection AUROC of
**0.4516 — below chance**. A post-hoc energy score on the frozen backbone, requiring no
training at all, reached **0.6466**.

**The lesson, and it generalises:** several earlier claims in the project's own repository
compared a proposed mechanism against 0.4516 and reported large improvements; against
0.6466 those improvements disappear. *An open-set baseline scoring below chance is a broken
baseline, not an easy one.*

Plus the honesty rule: with only 19 unknown patients every interval in this arm spans
chance, so the paper writes **"not shown to beat"** rather than "beats" throughout. Keep
that phrasing discipline.

---

## 3.16 The Pre-Registered Concept Gate (lines 982–1019)

### `tab:gate`
Two concepts × two runs, with the run-2 confidence interval, AUPRC, prevalence, and the
verdict.

| Concept | Run 1 | Run 2 | CI (run 2) | AUPRC | Prevalence | Gate |
|---|---|---|---|---|---|---|
| crackle | 0.5506 | 0.5580 | [0.533, 0.583] | 0.308 | 0.267 | FAIL |
| wheeze | 0.5729 | 0.5340 | [0.505, 0.563] | 0.190 | 0.174 | FAIL |

**Read the AUPRC against the prevalence column** — 0.308 versus a base rate of 0.267 is
barely above chance, and that is more informative than the AUROC alone.

**The footnote makes two further points:**
- Both intervals *exclude chance* while both point estimates are far below 0.65 — exactly
  the case the AUROC floor was written to catch. This retroactively justifies the gate
  design.
- Of nine extracted concepts only these two have a ground truth in ICBHI at all; five are
  proxies with no reference standard on this corpus and two are purely descriptive. **That
  two-of-nine ratio is itself a finding** about what this benchmark can validate.

### The revision paragraph
The revisions between the runs worked mechanically and each traded one failure mode for
its opposite (crackle firing rate 10/s → 2.5/s killing saturation; the wheeze detector went
from silent on half of all wheeze cycles to firing on 99 % of everything). Then the
stopping statement: two runs was the budget, no third was attempted, *"because a third
revision aimed at a gate already read twice would be fitting the test set and any resulting
pass would be uninterpretable."*

**This paragraph is the methodological spine of the paper.** Rewrite it in your own words
if you like, but it must contain: the diagnostics (not just the AUROCs), the budget, and
the reason a third run was refused.

---

## 3.17 Is 0.65 Reachable at All? The Retraction (lines 1021–1062)

### The setup paragraph
States plainly that the team *"started to prefer an explanation we liked"* — that the
labels were too noisy for anyone — and then tested it with the readings fixed in the script
before it ran.

### `tab:ceiling`
Four representations, crackle and wheeze, with patient-bootstrap intervals. See the table
in §1.5 above.

**The "Saw ICBHI?" column is the argument.** The frozen AudioSet row is marked **never**:
that model was pretrained on general audio and has never heard a respiratory corpus or seen
an ICBHI label, so its 0.7115 / 0.7621 **cannot be dismissed as circular** the way the
last row (a backbone trained on these very labels) can.

### The retraction paragraph
*"The pre-registered 'no ceiling' branch fired, so we retract the ceiling claim."* The
labels support detection at 0.71–0.87; the binding constraint was the extractors. And the
reason for saying it plainly: *"conceding it is the whole point of having written the branch
down in advance."*

### The self-criticism paragraph
A second finding from the same table, and it criticises the team's own gate design: the
gate thresholded **one hand-built scalar**, but the same fourteen concepts taken **as a
vector** reach 0.6608 on crackle — above the threshold the scalar failed. So the concept
vocabulary carried more information than the scalar exposed, and a validity gate for a
concept *suite* should be posed over the suite, not over one summary score.

**Rewriting guidance.** Do not soften the retraction into "our results were mixed" or "the
picture is nuanced". The value of this subsection is that it is unambiguous. A reviewer who
sees a group retract its own preferred explanation, on pre-registered grounds, will trust
everything else in the paper more.

---

## 3.18 Machines Against Physicians (lines 1064–1139)

### `tab:human`
Four published rows plus three rows of the project's own clinician study.

**The project's study design:** one physician, blind, 132 clips, six calibration exemplars,
twelve hidden duplicates. The rater answered 116 of 132; the 16 blanks are exactly the
clips marked unusable, *"so they are an answer and not missing data."*

| | Crackle | Wheeze |
|---|---|---|
| agreement with ICBHI | κ **0.035** [−0.122, 0.192] | κ **0.266** [0.085, 0.445] |
| sensitivity | **23.1 %** [12.0, 35.3] | 29.7 % [15.4, 45.2] |
| intra-rater (12 duplicates) | 7/8 exact, 11/12 presence | 7/8 exact |

**The intra-rater row is what makes the rest interpretable**, and the footnote says so: the
listening pack's own rule was that more than three self-disagreements out of twelve would
mean the task is impossible from recordings. There was one. So the rater is self-consistent,
and his disagreement with ICBHI is therefore informative rather than noise.

**The replication paragraph.** Our crackle sensitivity of 23.1 % lands within a point of
the 23.23 % reported for seven senior physicians on this corpus — different rater,
different clip sample. Two independent measurements agreeing that closely say the low
agreement on crackles is a property of *the material and the label convention*, not of one
rater's skill.

### The same-clips block — the paper's sharpest result, **now merged into `tab:human`**

**Why it exists.** The upper block of `tab:human` compares machine and physician numbers
measured on *different sets of cycles*. The lower block removes that objection by putting both
on the same clips. The two used to be separate tables and are now one float with two blocks.

**Method:** frozen AudioSet embedding → logistic probe, fitted by **patient-grouped
five-fold cross-validation** so every clip is out of sample, then scored against each
reference standard in turn.

| | Crackle (n=108) | Wheeze (n=110) |
|---|---|---|
| probe vs ICBHI label | 0.6786 [0.578, 0.778] | 0.7519 [0.636, 0.857] |
| probe vs physician's label | 0.5097 [0.354, 0.665] | 0.5828 [0.412, 0.746] |
| **paired difference** | +0.169 [−0.025, +0.361], p = 0.087 | **+0.169 [+0.002, +0.344], p = 0.046** |
| physician–ICBHI raw agreement | 0.528 | 0.718 |

**Three things you must not lose when rewriting:**

1. **The effect size is identical for both concepts; only wheeze reaches significance.**
   Crackle is reported as *undecided*, not rounded up to a win. Do not write "both
   concepts show the effect" without the significance qualifier.
2. **The selection warning.** The 132 clips are not a random sample — the pack kept cycles
   of at least 0.9 s, sorted longest first, and over-sampled the abnormal strata. So the
   *absolute* AUROCs here are **not** comparable to `tab:ceiling`'s 2,636-cycle numbers.
3. **Why the contrast survives anyway:** the selection applies equally to both halves of
   the paired difference. *"The contrast is the result and the level is not."*

**Why a paired bootstrap and not two marginal intervals.** Two overlapping confidence
intervals are not a test. The estimator resamples the clips once per iteration and
recomputes *both* AUROCs on that same resample, which is what makes the difference
testable.

### The closing paragraph and its scope limits
*On the same clips, with the same scores, a network that has never heard a respiratory
corpus reproduces ICBHI's wheeze labels better than a self-consistent physician does.*
Immediately followed by the scope bound: it is a statement about what these labels encode,
**not** a claim that they are wrong and **not** a claim about respiratory machine learning
in general. Keep both halves. The claim without the bound is overreach.

---

## 3.19 Explainable Artificial Intelligence (lines 1141–1225)

**Rubric requirement 9:** XAI on the best model, *with interpretation, not just images.*

### `fig:xai`
Four-panel layout per example: log-mel spectrogram, gradient-based attribution, occlusion
map, predicted versus true label. Includes a **misclassified** example, which graders look
for.

### `tab:xaiquant` — the reason this section is not decoration
Every row is a measurement with an interval.

| Measure | Condition | Value |
|---|---|---|
| Band pointing | uniform baseline | 0.5547 |
| | wheeze absent (n=325) | 0.7674 |
| | wheeze present (n=75) | 0.7140 |
| | **difference** | **−0.0534** [−0.0785, −0.0256], p = 0.001 |
| Tiling consistency | mean r (n=361) | 0.098 |
| | median r | 0.180 |
| | fraction above 0.5 | 32.1 % |
| Padding attention | cyclic tiling, observed/uniform | 0.594 / 0.679 (ratio 0.87) |
| | zero padding, observed/uniform | 0.101 / 0.679 (ratio 0.15) |

### The two findings, and both cut against the model

1. **Band pointing is backwards.** The model does concentrate on the physiological
   100–1000 Hz band in general — both groups sit well above the 0.5547 uniform baseline —
   but when a wheeze is actually *present* it attends to that band **less**, and the
   interval on the difference is comfortably clear of zero. A wheeze detector should show
   the opposite contrast.

2. **Tiling consistency is the finding that changed the paper.** Every cycle shorter than
   8 s is repeated two or three times inside the input, so a model reading acoustics should
   attribute the same way to each copy. It does not: mean correlation 0.098, only a third
   of cycles above 0.5. **A substantial part of the attribution tracks position in the
   window rather than sound.**

   That observation is what made the team add ablation row `P4` (zero padding instead of
   cyclic tiling), and the padding-attention rows show the two models behaving completely
   differently in the padded region. `P4` costs 0.0311, so tiling *is* doing real work —
   but whether that work is acoustic or positional these measurements do not settle, and
   the paper says so.

**This is item (v) of the novelty paragraph: an XAI result that fed back into the
experimental design rather than just illustrating a finished model.** Most papers' XAI
sections are terminal. This one is causal. Make sure a rewrite keeps that causal link
visible.

### The resolution caveat (table footnote) — do not delete this
MobileNetV2 downsamples by 32, so its final convolutional block is **4 frequency cells by
26 time cells** — roughly 500 Hz and 0.31 s per cell. Every gradient map in the figure is
that grid interpolated up to 128 × 801.

Consequence, stated precisely: the **direction** of both measurements is trustworthy
because both intervals exclude their null; the **fine structure** visible in the figure is
interpolation, not measurement. And where the two methods disagree, prefer occlusion — its
patch grid is 16 mel bins by 80 frames and it needs no gradient through a downsampled
layer.

**Never write that the model "focuses on" some fine spectral structure.** The resolution
does not support it.

### The interpretation guard
*"We deliberately do not call any of this clinical validation."* The labels these
attributions are aligned to are reproduced by a physician at κ = 0.035 on crackles, so a
map aligned to labels of that reliability tells you **where a model looks**, not that it
reasons like a clinician.

### The dropped analysis, reported rather than faked
The team had planned a **pointing game** — does attribution mass land inside the annotated
adventitious-sound window? It is **not implementable**: ICBHI annotates *whether* a cycle
contains such a sound, not *where*, and the pipeline crops to exactly that window, so the
hit rate is 100 % by construction. Band pointing and tiling consistency are the
substitutes.

**Reporting the substitution is worth more than reporting a meaningless 100 %.** Keep it.

---

## 3.20 Statistical Analysis (lines 1227–1282)

**Job.** Answer two questions that must be settled before any ablation delta can be
believed: *how large is run-to-run noise*, and *what is the effective sample size*.

### `tab:stats`, upper block — the noise floor
The same configuration trained three times with nothing but the seed changed:

| Seed | 42 | 1 | 2 | Mean | SD | Range |
|---|---|---|---|---|---|---|
| ICBHI | 0.5540 | 0.5681 | 0.5657 | 0.5626 | **0.0075** | **0.0141** |

**Two uses:**
1. **0.0141 becomes an admissibility threshold.** Any ablation delta smaller than that
   range is not distinguishable from run-to-run noise, whatever its sign. This is why
   `tab:ablation` has a "vs noise" column.
2. **An unplanned reproducibility check.** The published reference value 0.5602 sits inside
   the band, and the replicates were produced by a *different harness* from the notebook
   that produced the reference — so two independent implementations of the same recipe
   agree to within run-to-run noise.

Reading the ablation against the range: three of eleven deltas do not clear it (removing
class weighting −0.0105, adding a band-pass filter −0.0089, removing the min–max rescale
−0.0063); two more clear it by less than 1.25× and are not safe alone; the remaining six
clear it by at least 2.2×.

### `tab:stats`, lower block — the unit of analysis
The same ten rows tested twice: McNemar over 2,636 cycles, and paired bootstrap over 47
patients.

**Result: 7 of 10 significant per cycle, 1 of 10 per patient, and six rows flip.**

The corrected test partition holds 2,636 cycles but only **47 patients**, and cycles from
one patient are not independent. A paper that runs its significance tests over cycles will
find effects a patient-level test cannot support.

### The synthesis paragraph — the most sophisticated argument in the paper
The two blocks look contradictory and are not:

> The seed band says run-to-run noise is too small to explain most of these deltas. The
> patient-level test says this test set is too small to certify them. **Most of the effects
> are probably real, and this benchmark cannot prove it.**

Both halves belong in the report. Do not drop one to make the story tidier — the pair is
worth more than either alone, and it is a statement about the benchmark, not about the
models.

---

## 3.21 Ablation Study (lines 1284–1398)

**Rubric requirement 10, and the rubric's own subsection 5.1.** Run on the best model, one
variable at a time.

### `tab:ablation` — twelve rows plus a thirteenth control

Seven rows change a model component (`A0`–`A6`), five change a preprocessing stage
(`P1`–`P5`), and `A24` is a two-variable control that earns its place.

**Row labels were renamed in v2** so they no longer echo internal run identifiers: the
reference is `REF`, model-component rows are `C1`–`C7`, preprocessing rows stay `P1`–`P5`.

| Row | Change | ICBHI | Δ | vs noise |
|---|---|---|---|---|
| REF | reference: MobileNetV2 + SpecAugment | 0.5602 | — | — |
| C1 | − SpecAugment | 0.5200 | −0.0402 | 2.9× |
| C2 | − ImageNet pretraining | 0.4999 | −0.0603 | 4.3× |
| C3 | − class-weighted loss | 0.5497 | −0.0105 | *below* |
| C4 | frozen backbone, head only | 0.4595 | −0.1007 | 7.1× |
| C5 | 64 mel bins instead of 128 | 0.5427 | −0.0175 | 1.2× |
| C6 | 4 s cycles instead of 8 s | 0.5224 | −0.0378 | 2.7× |
| C7 | random init **and** frozen | 0.4979 | −0.0623 | 4.4× |
| P1 | **+** band-pass filter | 0.5513 | −0.0089 | *below* |
| P2 | **+** spectral-gating denoising | 0.5248 | −0.0354 | 2.5× |
| P3 | **+** per-cycle amplitude norm. | **0.5764** | **+0.0162** | 1.1× |
| P4 | zero padding instead of tiling | 0.5291 | −0.0311 | 2.2× |
| P5 | − min–max rescale | 0.5539 | −0.0063 | *below* |

**Sign convention, and it is easy to get wrong.** `P1`–`P3` **add** a stage the reference
does not have. A positive delta there is a recommendation to adopt the stage. Every other
row removes or replaces something, so a negative delta means the thing helped.

**The two-implementation caveat in the table note.** `REF` and `C1` come from the training
notebooks; the other ten come from a separate harness reusing the same preprocessing, loss,
optimiser and seed. Run on the reference configuration at seed 42, that harness returns
**0.5540** against the notebook's **0.5602**. The difference sits inside the seed range,
which is what licenses reading the two as one table — and is also why any delta below that
range is treated as unresolved rather than as a small effect. **Keep this note.**

### The four discussion beats

1. **Fine-tuning matters far more than pretraining, and `C7` proves the decomposition.**
   Freezing the backbone costs 0.1007 (largest single effect, and the only row surviving the
   patient-level test). Random initialisation costs 0.0603. Both expected. `C7` does both at
   once and is not expected: freezing a **random** backbone scores 0.4979, while freezing the
   **ImageNet** backbone scores 0.4595. So **frozen ImageNet features are 0.038 worse than
   random projections on log-mel spectrograms** — 2.7× the noise range. Transfer from images
   to spectrograms only pays once the backbone can move, and until then it is actively
   harmful.

   That decomposes `C4`'s −0.1007 into two statements — *pretraining is worthless frozen* and
   *adaptation is what the 0.10 buys* — and **neither `C2` nor `C4` could produce it alone.**
   This is one of the paper's genuinely new results.

2. **The three preprocessing stages we never had were mostly not missing.** Band-pass
   filtering is neutral; denoising costs 0.0354; only amplitude normalisation helps
   (+0.0162). The denoising result has a mechanism worth stating: spectral gating removes
   stationary noise, and crackles are short transients whose detectability depends on
   exactly the low-energy structure the gate suppresses.

3. **The highest score in the table is not the model we report.** `P3` reaches 0.5764,
   above the reference, and is **not promoted**: its delta is 1.1× the noise range and its
   patient-level interval spans zero. *"Promoting it would be the same class of mistake this
   paper is about, made with our own evidence in front of us."*

   **This is a self-discipline moment and it is worth marks. Do not delete it to make the
   headline number bigger.**

4. **Class weighting barely matters** (−0.0105, below the floor), which is a useful negative:
   the standard answer to the imbalance is worth less here than the padding scheme.

### The cumulative-ladder paragraph
The table above is **leave-one-out** (full pipeline minus one). The team also assembled the
**cumulative** form (add one component at a time), which exposes an **ordering effect** a
single ordering would hide:

- added **after SpecAugment**, class weighting is credited **+0.0105**;
- added **after class weighting**, SpecAugment is credited **+0.0402**.

The two orderings agree only at the end point. The bottom three rungs of that ladder need
GPU runs that were not completed and are recorded as `NOT RUN` rather than filled with
plausible values.

**Why this matters conceptually:** a cumulative table silently claims its ordering is the
natural one. It is not — what a component is credited with depends on what preceded it.

### `tab:ablationreason` — the supervisor's slide-14 format

One row per metric, configuration A versus configuration B, a **Progress** column in
percent, and a **Possible reason** column.

| Metric | A: no SpecAugment | B: + SpecAugment | Progress | Possible reason |
|---|---|---|---|---|
| Accuracy | 0.5372 | 0.5880 | +9.46 % | fewer Normal cycles called abnormal |
| Precision | 0.3917 | 0.4322 | +10.34 % | masking suppresses over-firing on rare classes |
| Recall | 0.4091 | 0.4103 | +0.29 % | essentially unchanged; masking adds no coverage |
| F1 | 0.3985 | 0.4141 | +3.91 % | precision gain with recall flat |
| Sensitivity | 0.4266 | 0.4089 | **−4.15 %** | the model becomes more conservative on abnormal cycles |
| Specificity | 0.6135 | 0.7115 | **+15.97 %** | the whole gain sits here |
| **ICBHI** | 0.5200 | **0.5602** | **+7.73 %** | specificity gain outweighs the sensitivity loss |

**The "Possible reason" column is the part that earns marks** — the supervisor's slide says
so explicitly. Each reason must be a mechanism, not a restatement of the number.

**The paragraph after the table is the payoff.** SpecAugment does not make the model hear
more — sensitivity actually *falls*. It makes the model stop calling Normal cycles
abnormal. Because the challenge metric weights Se and Sp equally, a 16 % specificity gain
against a 4 % sensitivity loss reads as progress. **On a screening application where missed
abnormalities are the expensive error, the same change might be a regression.**

That last sentence connects a metric artefact to a clinical consequence. Keep it.

---

## 3.22 Failure Analysis — **now prose, not a table**

An earlier version had a `tab:failures`. Six of its eight rows appeared elsewhere in the
paper, so it is now two paragraphs. The unique content — *which automated check caught each
failure* — is preserved in full.

| Run | Symptom | Check | Consequence |
|---|---|---|---|
| M40 ViT-B/16 | Se = 0.0000, all-Normal | sensitivity floor | reported, not deleted |
| M33 temporal transformer | Sp = 0.0000 | specificity floor | superseded by M33-v2 |
| M36 distillation | Se = 0.0932 | sensitivity floor | excluded from claims |
| M21 curriculum | Se = 0.0186 | sensitivity floor | excluded from the main table |
| M35-v2 | best epoch is epoch 1 | selection sanity | flagged beside its score |
| M6 OpenMax | AUROC 0.4516, below chance | chance level | invalidates comparisons against it |
| M15 | score reported, no confusion matrix | verifiability | unverifiable, withdrawn |
| M11 | accuracy changes under temperature scaling | invariance | mathematically impossible |

**The eighth is the best check in the paper and gets its own paragraph:** one run reported
that temperature scaling improved its accuracy from 0.3573 to 0.6595. Temperature scaling is a
monotone transform of the logits, so it cannot change which class scores highest and therefore
cannot change accuracy at all. That run was reporting something which cannot happen, and a
three-line assertion catches it.

**Why this section exists.** The rubric asks for error/failure analysis. But it also does
argumentative work: it shows the audit is a *tool* that catches things automatically, not a
one-off act of introspection. That is what makes the "release the tooling" contribution
credible.

---

## 3.23 Limitations (lines 1434–1455)

**One dense paragraph, six limitations, ordered by seriousness.** Written as a paragraph
rather than a bulleted list to save space; you may split it into a list if you have room.

1. **The training loop monitors the test partition each epoch.** There is no separate
   validation split in the ablation harness, so "select the best epoch" means select on
   test. Every score carries that advantage, **equally**, and the deltas between rows are
   what survive it. `tab:selection` prices the selection rule at 0.075–0.102, which is a
   *lower bound* on what a clean protocol would cost. **This is the most serious limitation
   and it is stated first, deliberately.**
2. **M43 was not run** — and it is the one model that would test the pretraining-domain
   hypothesis directly.
3. **The ViT collapse may be a schedule artefact.** No warm-up re-run with a lower backbone
   learning rate was tried, which is the standard remedy. The paper makes the weaker claim.
4. **The split effect is measured on one model chain only.** Two earlier runs appear to show
   much larger split effects, but those pairs differ in *architecture* as well as partition,
   so their differences are confounded and are never quoted as split effects.
5. **One clinician**, so no inter-rater agreement is computable. Mitigated by the
   intra-rater duplicates and the near-exact replication of a published seven-physician
   sensitivity; not removed.
6. **One corpus.** The scope of every claim is ICBHI 2017.

**Rewriting guidance.** Do not soften item 1 and do not move it later. A supervisor who
finds it themselves after reading a confident results section will trust the paper less
than one who is told first.

---

## 3.24 Comparison with Existing Works (lines 1457–1532)

**Rubric requirement: the comparison table must be the very last thing in Results and
Discussion.** `verify.py` checks this.

### `tab:comparison`
The supervisor's slide-13 format — Author / Dataset / Network / … / This work — but with
two changes that are the paper's own argument applied to its own table:

1. **Three blocks by partition**, with an explicit statement that rows in different blocks
   are not rankable.
2. **Se and Sp columns kept beside the composite**, which is what makes the discussion
   below possible.

| Block | Contents |
|---|---|
| Published 60/40 — directly comparable | six published systems + the seven-physician row + **three of ours** |
| Corrected patient-independent | four of ours; **not comparable to the block above** |
| Identifier fallback | two rows, "shown only to price the fault" |

Published rows: RespireNet 0.5620, supervised contrastive 0.5755, frozen M2D probe 0.5938,
patch-mix AST 0.6237, BTS/CLAP 0.6354, PAFA/BEATs 0.6484, and seven senior physicians at
0.4777.

**Note the physician row sits inside the comparable block.** That is not a gimmick — it is
measured on the same partition and metric, and it is the most useful reference point in the
table.

**The exclusion footnote is the paper's argument turned on itself:** Suma et al. report
accuracy on an unstated partition and Cho & Lee report an improvement over a baseline
rather than an absolute score, so neither can be placed in this table. Both are cited
elsewhere in the paper.

### The three discussion paragraphs

1. **Where we land.** 0.5641 on the partition the literature uses — just above RespireNet,
   the strongest ImageNet-pretrained convolutional system, and below every audio-pretrained
   one, about 8 points short of the best row.

2. **The decomposition, and this is the payoff of the whole reporting discipline.**
   **Our sensitivity is not behind the field.** At 0.456 it beats RespireNet (0.401),
   supervised contrastive (0.392) and patch-mix (0.431), sits level with BTS (0.457), and
   trails only the frozen probe (0.466) and PAFA (0.476). **The entire deficit is
   specificity**: 0.672 against a field spanning 0.722–0.821.

   So the system is not uniformly weaker — it sits at a **different operating point**,
   finding abnormal cycles at roughly the field's rate while over-calling them on normal
   ones. That is the direction a class-weighted loss and SpecAugment push a model by design,
   and arguably the direction a screening tool should prefer, though the challenge metric
   gives no credit for it. *"Under the composite alone the distinction vanishes."*

   **This paragraph turns a mediocre headline into a substantive finding, and it is only
   possible because Se and Sp are reported separately. It is the concrete demonstration of
   the paper's own recommendation.**

3. **The last two rows.** The unaudited pipeline reported 0.7077, which would have placed
   *first* in this table, ahead of every published system — on 7.1 % of the corpus under a
   metric the team defined itself. *"It was not state of the art, it was an unaudited
   measurement."* Closing line: *"A 2.2 M-parameter MobileNetV2 landing level with a 2021
   ResNet34 is an ordinary outcome for a course project; finding out why the same code first
   said otherwise is not."*

---

## 3.25 Conclusions (lines 1534–end)

**Rubric requirement: final section, include two or three future works.**

**Five paragraphs:**

1. **The narrative recap.** "We set out to build … and ended up building an audit." The
   gate, the two failures, the five faults each with its price, and the 0.7077 → 0.5602 move.
2. **The model-side recap.** Best model and its two partition scores, the transformer
   losses, the four ablation findings, the noise floor disqualifying three deltas, the
   patient-level test certifying one, the XAI results, and the retraction with its numbers.
3. **The practical recommendation, five items:** report the challenge metric with Se and Sp
   beside it; name the partition and say whether it is patient-independent; commit the raw
   confusion matrix; state the checkpoint selection rule; test over patients rather than
   cycles. Closing: *"Four of our five faults are invisible without those habits, and all
   five sat in code that ran without error and produced plausible numbers."*
4. **Three future works, each italicised and each grounded in something unfinished:**
   - *A clean validation protocol, and the fourth transformer* — carve a validation split
     from the 79 training patients and re-run; the AST run belongs in the same batch.
   - *A reference standard built for the task* — a multi-rater re-annotation of a few
     hundred cycles with calibration exemplars and an explicit "fits no named category"
     option, which would separate model error from label noise. Supported by the listening
     pack's finding that roughly 7 % of clips contain a sound fitting none of the four named
     classes.
   - *Fine-tuning that preserves what pretraining learned* — grounded in `tab:ceiling` (the
     frozen probe reaches 0.71/0.76 with no task training) and in `A24` (our ImageNet
     backbones are useless until allowed to move). The probe supplies the success criterion.
5. **The closing two-sentence moral:** fix the threshold before running the experiment, and
   audit your own pipeline before you audit anyone else's.

**Rewriting trap.** Future work item 3 was originally grounded in an experiment (N1, the
LoRA concept-probing result) that is **not reported in this paper**. It was rewritten to
rest only on `tab:ceiling` and `A24`, which are. If you rewrite it, do not reintroduce a
claim about task adaptation degrading concept encoding — that result is real and lives in
the repository, but it is not in this paper, and a conclusion may not rest on unreported
evidence.

---

# Part 4 — Table-by-table reference sheet

For each table: the research question it answers, the source on disk, the one takeaway, and
the trap.

**17 tables in the body, 1 in the appendix.** Four were cut or merged in v2 as repetitive:
`tab:corpus` (repeated its prose), `tab:preproc` (repeated the equations and the ablation
table), `tab:augmentation` (re-presented `tab:main`), `tab:failures` (six of eight rows
appeared elsewhere). `tab:classdist` merged into `tab:partitions`, and `tab:sameclips` merged
into `tab:human`.

| # | Label | Question it answers | Source | Takeaway | Trap |
|---|---|---|---|---|---|
| I | `tab:partitions` | What partitions exist, and how imbalanced is each? | annotation index; `official_split.py`; each run's `dataset_info` | 86 Both cycles against 1,560 Normal; only one partition carries new results | Never merge the corrected (2,636) and published (2,756) test sets |
| II | `tab:models` | Which model does each member own, and what is each one's status? | run records + the course checklist | four transformers exist, but only three are comparable | Never call the four "comparable"; never give AST a challenge score |
| V | `tab:metricaudit` | What does the wrong metric cost, across the whole suite? | `icbhi_score_audit.py` over committed matrices | mean +0.0948, max +0.2176, min +0.0025 | Rows sit on four partitions — the partition column exists so nobody ranks across them |
| VI | `tab:selection` | How much does the checkpoint rule matter? | `M48_tier_A_table.json` | 0.075–0.102 across three seeds; more than pretraining | Both criteria are on one monitoring set; report the *difference*, never the levels |
| VII | `tab:protocol` | What is each protocol component worth? | `tier_e_protocol_ablation.py` | E1+E2 = 0.7077, +0.1475 over corrected | E3 is a **null** and is reported as one, on purpose |
| VIII | `tab:main` | How do all models compare under one protocol? | `results_M*.json` for each | every transformer loses; ViT = the constant predictor | The always-Normal row is the control, not filler; the AST block is not comparable to it |
| IX | `tab:efficiency` | What does every model cost? | `efficiency` block of each JSON | ViT is 38× the parameters for below-chance performance | Two device blocks — never compare across them |
| X | *(augmentation — now prose)* | What does augmentation buy, per backbone? | the paired rows of `tab:main` | +0.0402 on the CNN, nothing on the transformers | ViT's pair says nothing; both runs collapsed |
| XI | `tab:errors` | Where does the best model actually fail? | `results_M22_v2.json` confusion matrix | 0.0730 of the score is lost to sub-typing alone | Detect-only 0.6332 is *the same predictions*, not a different model |
| XII | `tab:extensions` | What did the novelty experiments give? | nine `results_M*.json` | the two highest numbers in the paper, neither claimable | Every row needs its partition label |
| XIII | `tab:gate` | Did the pre-registered gate pass? | `Asif's/M39/*` (9-concept runs) | No, twice | Do not mix in the later 14-concept engine numbers |
| XIV | `tab:ceiling` | Is 0.65 reachable at all against these labels? | `N11_supervised_ceiling.json` | Yes — 0.71/0.76 from a model that never heard a lung sound | The "Saw ICBHI?" column carries the argument |
| XV | `tab:human` (upper block) | How well do humans reproduce these labels? | three published papers + `CLINICIAN_RESULTS_v1.md` | our 23.1 % replicates a published 23.23 % | Melbye, not Aviles-Solis |
| XVI | `tab:human` (lower block) | Does the machine beat the physician **on the same clips**? | `results_M48_C4.json` | +0.169; significant for wheeze, undecided for crackle | Absolute AUROCs here are not corpus-comparable; only the contrast is |
| XVII | `tab:xaiquant` | Where does the model look, measurably? | `results_M44*.json` | it attends *less* to the wheeze band when a wheeze is present | 4×26 native resolution — direction is trustworthy, fine structure is not |
| XVIII | `tab:stats` | Which differences are real, and at what sample size? | `M48_tier_A_table.json`, `M45_paired_tests.json` | noise floor 0.0141; 1/10 significant per patient vs 7/10 per cycle | The two blocks are complementary, not contradictory |
| XIX | `tab:ablation` | What is each component worth, given everything else? | `M45_ablation_table.json` | fine-tuning ≫ pretraining; frozen ImageNet < random | Sign convention on `P1`–`P3`; two-implementation note |
| XX | `tab:ablationreason` | What did augmentation change, metric by metric, and why? | `results_M3_v2.json` vs `results_M22_v2.json` | all the gain is specificity; sensitivity falls | The "possible reason" column must give a mechanism |
| XXI | *(failure analysis — now prose)* | How did runs fail, and what caught them? | `PROJECT_AUDIT.md` + the matrices | eight failures, all caught automatically | Keep the temperature-scaling invariance paragraph |
| XXII | `tab:comparison` | How does this work compare to published systems? | `Papers/RESULT_COMPARISON.md` + our JSONs | our Se is mid-field; the whole deficit is Sp | Must be the **last** thing in Section III (the appendix table does not count) |
| A-I | `tab:runmap` | Which repository file is each model? | the repository | every number stays traceable | Update it if you rename a model |

---

# Part 5 — Figures

| Figure | File | Source | What it must show | Caption's substantive point |
|---|---|---|---|---|
| `fig:flowchart` | `fig_flowchart.pdf` | `make_diagrams.py`, reading committed JSONs | the complete system, raw audio → reported score | the split loader, metric and selection rule are drawn as **components with ablation rows**, not infrastructure |
| `fig:cm` | `fig_confusion_best.png` | `make_paper_figures.py` from `results_M22_v2.json` | row-normalised 4×4 for the **best model only** | diagonal falls monotonically with class frequency; abnormal column leaks into Normal |
| `fig:curves` | `fig_curves_best.png` | same generator | train/validation loss and accuracy vs epoch, best epoch marked | loss keeps falling after the monitored score stops improving — Fault 5 in a picture |
| `fig:xai` | `fig_xai_panels.png` | `m44_xai_best_model.py` | spectrogram · attribution · occlusion · predicted vs true, including a misclassified example | it is evidence about where the model looks, not clinical validation |

**Rule for all four:** every figure is regenerated from a committed results file by a named
script. **Never edit one in a vector editor** — edit the script and re-run, or the figure
will drift from its table, which is the exact fault this paper is about.

**Spare figures in `figures/` not currently used:** `fig_metric_inflation.png` (a scatter of
reported versus recomputed score — cut because `tab:metricaudit` carries all 20 rows and the
figure cost about 0.7 of a page), plus `fig_architecture.pdf`, `fig_modules.pdf`,
`fig_motivation.pdf`, `fig_metric_anatomy.pdf`, `auroc_forest_plot.png`,
`M48_selection_curves.png`. All are available if you have page budget.

---

# Part 6 — Equations

All nine are inside `equation` environments and numbered, as the rubric requires. Every one
is referenced from the body.

| # | Label | Statement | Implementation it must match |
|---|---|---|---|
| 1 | `eq:tile` | `x̃[n] = x[n mod L]`, n = 0…N−1 | `np.tile(a, ceil(N/len(a)))[:N]` |
| 2 | `eq:logmel` | log-mel with a **per-spectrogram max** in the denominator | `librosa.power_to_db(m, ref=np.max)` |
| 3 | `eq:minmax` | `(S − min S)/(max S − min S + ε)`, ε = 1e-8 | `lm = (lm - lm.min())/(lm.max() - lm.min() + 1e-8)` |
| 4 | `eq:label` | `y = c + 2w` | the two annotation flags |
| 5 | `eq:weights` | inverse-frequency weights **divided by their mean**, then weighted cross-entropy | `w = c.sum()/(4*c); w = w/w.mean()` |
| 6 | `eq:specaug` | two frequency masks ≤ 24 bins, two time masks ≤ 80 frames, zeroed | training split only |
| 7 | `eq:official` | Se over pooled abnormal, Sp over Normal, mean of the two | the ICBHI 2017 challenge metric |
| 8 | `eq:macro` | mean of macro recall and macro specificity | the variant the pipeline wrongly used |
| 9 | `eq:gate` | AUROC ≥ 0.65 **and** 95 % DeLong interval excludes 0.5 | fixed before the gate ran |

**Verification you can do by hand.** Take the best model's committed matrix
`[[1110,333,81,36],[269,320,8,20],[178,39,110,46],[32,19,25,10]]` and apply (7):
Sp = 1110/1560 = 0.7115; Se = (320+110+10)/1076 = 440/1076 = 0.4089; mean = 0.5602. That
reproduces the headline number in two lines of arithmetic, which is the point of committing
the matrix.

---

# Part 7 — Number provenance index

Every number in the paper traces to one of these. If you invent a number that is not here,
the paper breaks its own rule.

### Corpus and partitions
| Quantity | Value | Where it comes from |
|---|---|---|
| recordings / patients / cycles | 920 / 126 / 6,898 | ICBHI annotation files, recomputed |
| class counts, whole corpus | 3,642 / 1,864 / 886 / 506 | recomputed from `concepts_all.npz` |
| corrected train / test | 4,262 / 2,636 cycles, 79 / 47 patients | same, and every corrected-run `dataset_info` |
| corrected test class counts | 1,560 / 617 / 373 / 86 | matches every committed confusion matrix |
| crackle- / wheeze-positive test cycles | 703 / 459 | 617+86 and 373+86 |
| published-verbatim test | 2,756 cycles, 49 patients | `results_M2.json`, `results_M22_v2_official.json` |
| fallback test | 492 cycles, 11 patients | `results_M22.json` |
| pipeline indexes 4,251 not 4,262 | 11-cycle filename join loss | `226_1b1_Pl_sc_Meditron` vs `_LittC2SE` |
| device-spanning patients | 4 — 112, 158, 218, 226 | recomputed from the device field |
| COPD share | 64/126 patients, 83.3 % of cycles | recomputed |
| median cycle length | 2.42 s | annotation index |
| always-Normal accuracy | 0.5918 | 1,560 / 2,636 |

### Main models (corrected partition)
| Model | ICBHI | Se | Sp | Params | Size |
|---|---|---|---|---|---|
| M22-v2 MobileNetV2 + SpecAug | **0.5602** | 0.4089 | 0.7115 | 2,228,996 | 8.74 MB |
| M3-v2 MobileNetV2 clean | 0.5200 | 0.4266 | 0.6135 | 2,228,996 | 8.74 MB |
| M41 Swin-T | 0.5304 | 0.3429 | 0.7179 | 27,522,430 | 104.99 MB |
| M41-aug | 0.5291 | 0.3569 | 0.7013 | same | same |
| M42 DeiT-S | 0.5149 | 0.2946 | 0.7353 | 21,667,204 | 82.65 MB |
| M42-aug | 0.4981 | 0.2974 | 0.6987 | same | same |
| M40 ViT-B/16 | 0.4994 | 0.0000 | 0.9987 | 85,801,732 | 327.31 MB |
| M40-aug | 0.5000 | 0.0000 | 1.0000 | same | same |

### The Audio Spectrogram Transformer (pre-audit, unrecoverable)
| Quantity | Value |
|---|---|
| accuracy / macro precision / macro recall / macro F1 | 0.5528 / 0.5347 / 0.4385 / 0.4109 |
| macro-variant score | 0.6359 (**never** convert this to the challenge metric) |
| official ICBHI, $S_e$, $S_p$ | **not recoverable** — no confusion matrix committed |
| parameters / size / inference | 86,385,668 / 329.54 MB / 86.74 ms per sample (Tesla T4) |
| size and latency ratio vs the CNN that beat it | 23.8× and 29.5× |
| partition | identifier fallback, 492 cycles, 11 patients |
| preprocessing deviations | full-band 20–8000 Hz mel; 512-point FFT |

### Other partitions
| Model | ICBHI | Partition |
|---|---|---|
| M22-v2 official (published verbatim) | 0.5641 | published |
| M3 MobileNetV3-Small | 0.5132 | patient-disjoint alt. |
| M2 2D-CNN from scratch | 0.4720 | patient-disjoint alt. |
| M22 (fallback) | 0.6495 | identifier fallback |
| M22 same run, macro metric | 0.7077 | identifier fallback |

### Protocol, statistics, ablation
| Quantity | Value |
|---|---|
| metric inflation | mean +0.0948, max +0.2176, min +0.0025, over 20 runs |
| E1 / E2 / E3 / E1+E2 | +0.0538 / +0.0893 / +0.0039 / **+0.1475** |
| selection rule cost | +0.0752 / +0.1015 / +0.0992 (seeds 42/1/2); 15/19/17 epochs apart |
| seed band | 0.5540, 0.5681, 0.5657 → mean 0.5626, sd 0.0075, **range 0.0141** |
| unit of analysis | 7/10 significant per cycle, 1/10 per patient, 6 flip |
| ablation deltas | A1 −0.0402, A2 −0.0603, A3 −0.0105, A4 −0.1007, A5 −0.0175, A6 −0.0378, A24 −0.0623, P1 −0.0089, P2 −0.0354, P3 **+0.0162**, P4 −0.0311, P5 −0.0063 |
| ordering effect | class weighting +0.0105 after SpecAugment; SpecAugment +0.0402 after class weighting |
| harness vs notebook, same config seed 42 | 0.5540 vs 0.5602 |

### Concepts, ceiling, humans, XAI
| Quantity | Value |
|---|---|
| gate run 1 / run 2 | crackle 0.5506 / 0.5580; wheeze 0.5729 / 0.5340 |
| gate CIs (run 2) | crackle [0.533, 0.583]; wheeze [0.505, 0.563] |
| AUPRC vs prevalence | 0.308 vs 0.267; 0.190 vs 0.174 |
| ceiling: 14 concepts | 0.6608 [0.601, 0.713] / 0.5815 [0.492, 0.675] |
| ceiling: frozen AudioSet | **0.7115** [0.638, 0.771] / **0.7621** [0.702, 0.815] |
| ceiling: task-trained backbone | 0.7451 [0.694, 0.795] / 0.8673 [0.806, 0.918] |
| clinician vs ICBHI | κ 0.035 [−0.122, 0.192] / κ 0.266 [0.085, 0.445] |
| clinician sensitivity | 23.1 % [12.0, 35.3] / 29.7 % [15.4, 45.2] |
| intra-rater | 7/8, 7/8, 11/12 |
| same-clips paired difference | +0.169 [−0.025, 0.361] p = 0.087 / **+0.169 [0.002, 0.344] p = 0.046** |
| band pointing | 0.7140 present vs 0.7674 absent, **−0.0534** [−0.0785, −0.0256], p = 0.001; uniform baseline 0.5547 |
| tiling consistency | mean r 0.098, median 0.180, 32.1 % above 0.5, n = 361 |
| padding attention | 0.594/0.679 wrap; 0.101/0.679 zero |
| open-set | OpenMax 0.4516, energy 0.6466, 19 unknown patients |
| error analysis | 479/597/440 of 1,076; detect-only 0.6332; sub-typing cost 0.0730 |

### Published comparison rows
RespireNet 0.401/0.723/0.5620 · SCL 0.392/0.760/0.5755 · frozen M2D 0.466/0.722/0.5938 ·
patch-mix AST 0.431/0.817/0.6237 · BTS 0.457/0.814/0.6354 · PAFA 0.476/0.821/0.6484 ·
7 physicians 0.232/0.723/0.4777.

---

# Part 8 — Claims that must never appear in this paper

Every one of these appears somewhere in the project's history and every one is refuted by a
later, better-controlled run. If a rewrite reintroduces one, it is a regression.

| Dead claim | Why it is dead |
|---|---|
| "ICBHI score 0.7227" (M2) or 0.6984 (M3) | macro metric **and** the 11-patient split; corrected values are 0.4720 and 0.5132 |
| "Interpretability costs 0.2326 accuracy / 0.1432 F1" | single-seed artefact; with five seeds the ordering inverts and McNemar gives p = 1.0000 |
| "M15 beats M6 by 36.9 %" | a ratio against a **sub-chance** baseline (M6 AUROC 0.4516) |
| "Calibration improves accuracy 0.3573 → 0.6595" | temperature scaling is monotone; accuracy **cannot** change |
| "Conformal wrapper achieves 95.45 % coverage" | coverage is guaranteed by construction — circular. The informative number is the empty-set rate on unknowns, which is 0.0 |
| "ICBHI's labels bound concept detection" (the ceiling claim) | **retracted** by N11; a frozen AudioSet probe reaches 0.71/0.76 |
| "Physicians validate our crackle detector" | crackle κ vs ICBHI is +0.035 and its interval includes zero |
| "The concept bottleneck is interpretable and effective" | the gate failed twice |
| M4 (the original AST) as a committed baseline | no results JSON exists in the repo; it never entered the audit |
| Calling our concepts "label-free" | Label-free CBM is an LLM/CLIP-derived method; the collision would read as overclaiming. Always write **physics-derived** |
| Any accusation that other groups made these five faults | the framing is "errors we made and caught". The implication lands harder unstated |

---

# Part 9 — Rules for rewriting

## 9.1 Style
The paper is deliberately written the way a careful undergraduate writes, not the way a
language model writes. Preserve that:

- **Vary sentence length.** Short declaratives next to long ones. "Two runs was the budget."
- **Avoid these words entirely:** *furthermore, moreover, additionally, notably, it is
  important to note, in conclusion, crucial, pivotal, comprehensive, leverage, delve,
  underscore, paradigm, realm, cutting-edge*. None currently appears; keep it that way.
- **Use first person plural.** "We audited", "we stopped", "we do not promote it."
- **Do not polish every sentence to the same finish.** Some sentences are blunt on purpose:
  "It was not state of the art, it was an unaudited measurement."
- **Prefer the plain word.** "found" not "ascertained", "worse" not "suboptimal", "wrong"
  not "non-optimal".
- **Let the em-dash and the colon carry the argument** rather than a transition word.

## 9.2 Evidence discipline
- Every number must exist in a committed results file. Part 7 is the index.
- If a run does not exist, write `NOT RUN`. Never a plausible value.
- Never state "X beats Y" when the confidence interval spans chance. The paper writes
  "not shown to beat" throughout, and that is not timidity — at 19 unknown patients no
  interval can exclude chance.
- Never compare across partitions without saying so. Every table that could tempt a reader
  into it has a partition column.
- Never compare timings across the two GPUs.
- Any delta below 0.0141 is "not distinguishable from run-to-run noise", not "a small
  improvement".

## 9.3 Rubric constraints that a rewrite can silently break
| Rule | Where it bites |
|---|---|
| No results in the Proposed System section | Section II may contain counts, never scores |
| Comparison table last in Results and Discussion | do not append a subsection after `subsec:comparison` |
| Every equation numbered inside `equation` | do not switch to `\[ \]` or `align` without labels |
| Every table and figure referenced via `\ref` | `verify.py` catches orphans |
| Abstract 200–300 words, no citations/abbreviations/symbols/equations | `verify.py` checks all five |
| Keywords alphabetical | re-sort after adding one |
| Training hardware named | Section III opening and `tab:efficiency` |
| Minimum 6 pages, and your 18-page ceiling | `verify.py` estimates |

## 9.4 Page budget
Currently estimated **17.9 pages** against an 18-page limit — tight, and the estimate is
approximate (it over-charges the bibliography by roughly 0.4 of a page, so the true figure is
probably nearer 17.5). If a compile runs over, the cheapest levers in order:

1. Reduce `fig:flowchart` and `fig:xai` widths (currently 0.46 and 0.50 `\linewidth`).
2. Shorten the notes under `tab:partitions`, `tab:human` and `tab:comparison`.
3. Cut `tab:ablationreason` and put its reason column into the ablation prose — this is the
   last resort, because the supervisor's slide asks for that exact format.

If a compile comes in short, the cheapest additions in order: restore
`fig_metric_inflation.png`, restore the cumulative-ladder table, restore the failure
catalogue as a table, add the pediatric concept-fragility experiment (real, pre-registered,
currently omitted for space — see the conflicts file).

---

# Part 10 — Checklist before you submit

Run first: `python verify.py` (no LaTeX needed). Then, by hand:

- [ ] Compiled on Overleaf with **zero errors** (warnings are allowed).
- [ ] Appendix A still lists every model you renamed.
- [ ] No `??` anywhere in the PDF.
- [ ] Page count between 6 and 18.
- [ ] Abstract still 200–300 words after your edits, still with no citations, abbreviations,
      symbols or equations.
- [ ] Keywords still alphabetical.
- [ ] No score, accuracy or delta anywhere in Section II.
- [ ] `tab:comparison` is still the last thing in Section III.
- [ ] Every number you changed still matches its source in Part 7.
- [ ] No claim from Part 8 has crept back in.
- [ ] Every "not shown to beat" is still a "not shown to beat".
- [ ] The two-implementation note under `tab:ablation` survives.
- [ ] No model nickname has crept back into the body (the only `M`-token that belongs there is
      `M2D`, which is a published model's real name).
- [ ] The Audio Spectrogram Transformer is still marked NOT RECOVERABLE everywhere.
- [ ] The selection-criterion caveat survives, both under `tab:selection` and as the first
      limitation.
- [ ] The XAI resolution caveat survives.
- [ ] Overleaf link shared with **edit access** and submitted in Canvas.

---

## One last thing

The paper's credibility comes almost entirely from places where it argues against itself:
the null in `E3`, the refusal to promote `P3`, the retraction in `subsec:ceiling`, the
admission that the harness monitors the test partition, the `NOT RUN` on M43, the XAI result
that contradicts the score. Every one of those is a place where it would have been easier and
more flattering to write something else.

When you rewrite, those are the passages to protect. Anything else can be said in different
words.
