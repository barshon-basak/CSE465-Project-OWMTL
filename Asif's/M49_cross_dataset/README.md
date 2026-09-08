# M49 — external-dataset validation of the best model

**Review comment being answered (Dr. Khan):** *"Test your best trained model with another dataset,
i.e., SPRSound (2022, open access) or HF Lung or the CirCor DigiScope."*

**Status:** notebook built and statically verified. **Blocked on one prerequisite — see §1.**

---

## 0. Which of the three datasets, and why

He named three. They are not interchangeable.

| Dataset | Sound | Labels | Access | Verdict |
|---|---|---|---|---|
| **SPRSound 2022** | lung | event-level: Normal, Fine/Coarse Crackle, Wheeze, Rhonchi, Stridor, Wheeze+Crackle | public GitHub, no registration | **do this one** |
| HF Lung V1 | lung | inhalation / exhalation / CAS / DAS **segments** | request form + GitLab | possible later, poor value for the cost |
| CirCor DigiScope | **heart (PCG)** | murmur present / absent / unknown | PhysioNet, open | not a generalization test — see below |

**SPRSound is the only one of the three that can actually answer the question.** Its event
taxonomy maps onto our four ICBHI classes with one judgement call (rhonchi and stridor), and its
events are time-bounded segments, which is the same unit our model consumes. It is paediatric and
recorded on a different electronic stethoscope, so the domain shift is real rather than cosmetic.

**CirCor is a phonocardiogram corpus.** It contains heart sounds. A crackle/wheeze classifier has
zero label overlap with murmur annotations, so scoring it produces no generalization number at
all. It is still worth 30 minutes, but as a **negative control**: feed the model audio from an
organ it was never trained on and see whether it abstains or confidently predicts lung pathology.
That is a trustworthiness result, not a transfer result, and it belongs in a separate notebook
(`M50`) with that framing. If it is presented as "we also tested on CirCor" without the
distinction, a reviewer who knows the corpus will read it as a category error.

**HF Lung** would be a genuine second lung-sound test, but its labels are onset/offset segmentation
rather than cycle classes, so using it means inventing a cycle-construction rule that neither
corpus specifies. That rule would then be doing part of the work the result is attributed to. Skip
it unless he asks for it by name.

**Recommendation: run SPRSound. Add the CirCor negative control only if there is time.** One
external dataset done properly, with baselines and intervals, is worth more than three done
loosely — which is the same argument this project already makes about novelty items.

---

## 1. Prerequisite: the checkpoint is missing

The paper's best model is **MobileNetV2 + SpecAugment (M22_v2)**, ICBHI challenge score **0.5602**
on the corrected 60/40 patient-independent partition, best epoch 38.

**That checkpoint is not in the repository.** `Asif's/M22_v2/Results/` holds the results JSON,
per-cycle predictions and plots, but no `best_model.pth`. `.gitignore` excludes `checkpoints/`,
and the file was never committed.

Two checkpoints on disk look like it and are **not** it:

| File | What it actually is | Why it must not be used |
|---|---|---|
| `Asif's/M22/result_M22/best_model.pth` | the original M22 | trained under the silent `patient_id <= 111` fallback split — 11 test patients, 492 cycles. Score 0.6495. |
| `Archive_Files (v4)/M22_v2_duplicate_folders_20260830/m22-official-updated-raw/best_model.pth` | `M22_v2_official` | trained on the **published verbatim split**, which leaks patients 156 and 218. Score 0.5641. Its stored config records `official_60_40_published_verbatim_NOT_patient_independent`. |

Testing generalization with a checkpoint trained on a leaking split would undercut the paper's
central argument in the one section meant to defend it. Cell 4 of the notebook reads the config
stored inside the checkpoint and **raises** on either of these.

### Recovering it

Re-run the training notebook. Nothing needs to be uploaded or attached.

```
open   Asif's/M22_v2/m22-official-notebook.ipynb
check  cell 2:  VARIANT = "augmented"   and   SPLIT_MODE = "corrected"
run    Runtime → Run all
```

### The `SPLIT_MODE` trap, now defused

That notebook has **two** switches, and the second one used to default to the wrong value:

| `SPLIT_MODE` | Partition | Reportable? |
|---|---|---|
| `"corrected"` | 551/369 recs, 2,636 test cycles, 47 patients | **yes** — patient-independent, the paper's 0.5602 |
| `"official"` | 539/381 recs, 2,756 test cycles, 49 patients | no — the published file verbatim, patients 156 and 218 on both sides |

`"official"` does not mean *the correct one*. It means *the published file, leak included*. It has
one legitimate use — comparing against published ICBHI work, where every other paper is on the same
leaky partition, which is where the 0.5641 in the comparison table comes from. It is not the model
M49 can use.

The committed default was `SPLIT_MODE = "official"`, sitting twelve lines below `VARIANT`, so
setting `VARIANT = "augmented"` and running produced a checkpoint labelled `M22_v2_official` that
loads cleanly, scores sensibly, and is not reportable. That default is now `"corrected"`, the two
switches sit together with the trap spelled out, and the cell prints a verdict banner at the end of
its config dump saying whether the active partition is reportable.

M49 no longer trusts filenames either: it reads the `split_method` out of every candidate
checkpoint and skips the verbatim ones, so if both runs are on Drive it picks the right one on its
own.

Seed 42, ~40 epochs, roughly 2 hours on a T4, plus a one-off spectrogram cache build. The run is
deterministic in its configuration, so M49's sanity gate will confirm whether it landed back on
0.5602.

It saves itself to Drive now. `best_model.pth`, `latest.pth`, the results JSON, the predictions CSV
and every plot go to:

```
/content/drive/MyDrive/OWMTL/M22_v2/checkpoints/best_model.pth
/content/drive/MyDrive/OWMTL/M22_v2/results/
```

which is one of the paths M49 searches by default. `latest.pth` is rewritten every epoch, so a
Colab disconnect costs one epoch rather than the whole run — re-run the notebook and it resumes.

---

## 2. Run order

| Step | Notebook | GPU time | Repeatable? |
|---|---|---|---|
| 1 | `Asif's/M22_v2/m22-official-notebook.ipynb` (augmented + corrected) | ~2 h | resumable |
| 2 | `M49_SPRSound_transfer.ipynb` | ~25 min | yes, cheaply |

Step 2 breaks down as: SPRSound clone ~5 min, ICBHI sanity-gate inference ~5 min (cached after the
first run), SPRSound mel cache ~10 min, inference ~2 min, the rest seconds. Once the caches exist a
re-run takes about 3 minutes, so the analysis arms can be iterated on freely.

### Setup — two one-time things, then nothing

**Kaggle credentials.** Colab left sidebar → key icon (Secrets) → add `KAGGLE_USERNAME` and
`KAGGLE_KEY` (kaggle.com → Settings → API → Create New Token), and turn **Notebook access ON for
both**. The notebooks resolve credentials from Secrets, then environment variables, then
`~/.kaggle/kaggle.json`. The key is never written into a notebook — these files are git-tracked and
a Kaggle key is full account access.

**Drive.** Accept the mount prompt the first time. Declining means every checkpoint lives in
`/content` and vanishes on disconnect.

Nothing else is attached or uploaded. The ICBHI audio is downloaded automatically if it is not
already present, `ICBHI_challenge_train_test.txt` is carried inside the notebook as an embedded
3.5 KB gzip+base64 copy (verified byte-identical to `Asif's/ICBHI_challenge_train_test.txt`), and
SPRSound is cloned from GitHub with no credentials at all.

If the Kaggle download fails, the usual cause is that the account has not accepted the dataset's
terms — open
[the dataset page](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database) once
while signed in, then re-run the cell. The notebook says this in the error.

### Why these notebooks needed patching

`patch_data_setup.py` and `patch_drive_setup.py` made M2 / M3 / M22 self-sufficient on Colab. The
v2 re-runs and M49 were written afterwards and never got either patch, so they still shipped the
Kaggle-only `CELL 3 — LOCATE DATA`, which on Colab raises

```
FileNotFoundError: ICBHI audio not found. On Kaggle: Add Data -> 'vbookshelf/respiratory-sound-database'
```

`Asif's/engine/patch_colab_v2.py` applies both blocks to M22_v2, M3_v2 and M49; the cell bodies
live in `Asif's/engine/colab_v2_cells.py` so there is one source of truth. It is idempotent — safe
to re-run, and worth re-running if any of those notebooks is ever regenerated.

---

## 3. What the notebook measures

Nothing is trained. Every weight is the one that produced 0.5602.

1. **Sanity gate.** The checkpoint is re-scored on the ICBHI corrected test partition and must
   reproduce 0.5602 within tolerance. The cell rebuilds the corrected split from scratch, asserts
   the two leaking patients are the expected ones, and asserts the partition comes to 2,636 cycles.
   A transfer number from an unverified checkpoint is worthless, so this gate raises rather than
   warns.
2. **Zero-shot 4-class**, under two label mappings:
   - *strict* — only unambiguous event types; rhonchi and stridor dropped, because ICBHI's wheeze
     class was not annotated to cover a low-pitched continuous sound or an upper-airway inspiratory
     one.
   - *broad* — every continuous adventitious sound counts as wheeze, which is how several SPRSound
     papers collapse the taxonomy.

   Both are reported. Where they disagree, the disagreement is the finding.
3. **Zero-shot binary**, Normal against adventitious. This is the distinction the two corpora
   genuinely share, so it is the fairer test of whether the representation transferred at all. The
   paper already reports the binary collapse on ICBHI, so the two are directly comparable.
4. **Trivial baselines on SPRSound** — always-Normal, majority class, uniform random,
   prior-matched random. The transfer score cannot be read without them, for the same reason the
   paper puts the 0.5918 always-Normal accuracy next to the best model.
5. **Prior-shift decomposition.** Subtracting the source log-prior and adding the target one
   separates the part of the drop caused by class-balance shift from the part caused by the
   representation. It uses the target prior, so it is a diagnostic, and it is labelled as one in
   the JSON, in the figure and in the summary. It is not a zero-shot number.
6. **Calibration under shift** — ECE, mean confidence, entropy and overconfidence on SPRSound
   against the same quantities on ICBHI. A model that fails while staying confident is worse than
   one that fails loudly, and for a paper about trustworthy evaluation that distinction carries
   weight.
7. **Age stratification.** Gap7 found the physics-informed loss hurt paediatric generalization and
   read it as adult acoustic priors failing on children's higher resonant frequencies. SPRSound
   carries age, so that explanation becomes testable instead of asserted: if the reading is right,
   score should climb with age.

Every score is reported with a **95% bootstrap interval that resamples patients, not events**.
Resampling events would treat forty cycles from one child as forty independent observations, which
is the exact inflation the paper's audit section is about.

---

## 4. What to expect

**The score will drop, and that is the result.** M2 and M3 fell from ~0.61 to 0.4139 / 0.4490 when
moved from the fallback split onto a real patient-disjoint one. A frozen adult-trained model on
paediatric audio from a different stethoscope has further to fall than that.

Interpretation depends on where the interval lands, and only three readings are honest:

| Broad-arm 95% CI | What to write |
|---|---|
| lower bound above 0.5 | the representation carries something across corpora; report the gap and its size |
| interval contains 0.5 | on this corpus the frozen model is **not shown to beat chance**. Not "it fails", not "it works" |
| upper bound below 0.5 | worse than chance, which is itself a finding and needs the label mapping re-checked first |

The project has said "not shown to beat a coin flip yet" before, about the open-set detectors at
n=19. The same phrasing discipline applies here.

**Do not fine-tune on SPRSound.** The moment any SPRSound data touches the weights this stops
answering the question he asked and becomes a domain-adaptation experiment, which is a different
paper. If the zero-shot number is poor, the poor number is the contribution.

---

## 5. Where the result goes in the paper

The Results section currently has no external-validity claim, and the Comparison with Existing
Works table was removed. This gives the paper two things back:

- a **generalization subsection** in Results — one table (the arms with their intervals plus the
  SPRSound baselines) and one figure (`M49_transfer_scores.png`), written in the professor's
  table-anchored paragraph style;
- a concrete **Limitations** point, which all four of his own papers have and this draft currently
  does not.

It also strengthens the existing argument rather than sitting beside it. The paper's claim is that
evaluation protocol matters more than architecture. Showing that the best model's score moves
further when the corpus changes than when the backbone changes is that claim measured one more
time, on data we did not choose.

---

## 6. Files

```
M49_SPRSound_transfer.ipynb    21 code cells / 18 markdown, protocol-compliant
README.md                      this file
```

Notebook outputs, written to `results_M49/`:

```
results_M49.json               section 4 schema, incl. confusion_matrix_raw, CIs,
                               all arms, baselines, calibration, age strata
logits_M49_sprsound.npy        raw logits — every arm recomputable without re-inference
index_M49_sprsound.csv         the scored event index with patient ids
M49_confusion_matrices.png     ICBHI / SPRSound 4-class / SPRSound binary
M49_transfer_scores.png        all arms with 95% patient-bootstrap intervals
M49_confidence.png             confidence distribution, source vs transfer
M49_age_strata.png             score against age band
```

## 7. Verification already done

- every code cell parses, in all three patched notebooks
- `Asif's/engine/check_undefined_names.py` reports no undefined names across cells in execution
  order, for all three
- `Asif's/engine/check_cell_order.py` reports correct cell order for all three. This checker was
  written after a bug it would have caught: M49 globbed `/content/drive/MyDrive` in cell 7 while
  the Drive mount sat in cell 14, so the checkpoint search found nothing. Every name resolved and
  every cell compiled, so the undefined-name pass was clean — the *resource* was missing, not the
  name. The rule is now enforced statically: a cell that reads a Drive path may not precede the
  cell that mounts Drive, and the same for `DATA_ROOT` / `SPLIT_FILE` / `CFG`
- `_find_preds` in the paired-comparison cell was verified against a simulated Drive layout. The
  Drive redirect had broken it: it still searched the old ephemeral path, so the paired test would
  have reported SKIPPED after both two-hour runs had completed successfully
- the metric functions were unit-tested off-notebook: the official metric on 4-class and binary
  matrices, always-Normal scoring exactly 0.5, the patient bootstrap widening the interval 2.6× against
  a cycle-level bootstrap under within-patient correlation, the label maps covering all seven
  documented SPRSound event types, and the JSON writer rejecting NaN
- the **data-setup cell was executed** against a simulated 920-recording corpus, through all three
  branches: audio found, split file absent (embedded copy written, verified byte-identical to the
  committed file, patients 156 and 218 still straddling it), and no audio with no credentials
  (raises with the Secrets instructions rather than downloading nothing)
- the **output-setup cell was executed** and confirmed to redirect `CFG["ckpt_dir"]`,
  `CFG["results_dir"]` and `CFG["cache_dir"]` off `/content`

Not run end to end — that needs the GPU, the real corpora and the checkpoint.
