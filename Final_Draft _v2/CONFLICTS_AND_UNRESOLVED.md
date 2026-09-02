# `Final_Draft _v2` — conflicts found, how each was resolved, and what is still open

**Written:** 2026-09-02 · **Method:** the paper in `main.tex` was reconstructed from the
committed `results_M*.json` files, the ICBHI annotation index (`concepts_all.npz`), the
run scripts, and the decision records — **before** `DRAFT_PAPER_1/` or `Final_Draft/` were
opened. Those two drafts were read only afterwards, and diffed against the independent
version.

**Resolution priority used throughout:**
final/reproducible result → code and experiment output → latest validated documentation →
old draft. *Most authentic and scientifically valid*, never *most recent text*.

---

## 0. Headline

The independent reconstruction agreed with `Final_Draft/main.tex` on every major number.
The metric-audit table, the protocol ablation, the gate table, the supervised-ceiling
table, the transformer table, the twelve-row ablation, the seed band and the
interpretability numbers all reproduced exactly. That is a real cross-check and it
passed.

Nine conflicts were found. Two are new — they were open items in
`Final_Draft/CONFLICTS_AND_RESOLUTIONS.md` and are now closed (§1.1, §1.2). The rest are
conflicts *between repository documents*, resolved by going to the committed result file.

---

## 1. Conflicts resolved in this version that were open before

### 1.1 The 34,164-parameter discrepancy — **explained, no longer "unexplained"** ✅

| | |
|---|---|
| Conflict | The ablation harness reports `total_params = 2,263,160`; the training notebook reports `2,228,996` for the same MobileNetV2. The previous draft flagged the 34,164 gap as unexplained and asked the reader to accept it. |
| Evidence | `torchvision.models.mobilenet_v2(num_classes=4)` has exactly **2,228,996 parameters and 34,164 buffer elements** (batch-normalisation running mean and variance). 2,228,996 + 34,164 = 2,263,160 exactly. |
| Resolution | The harness counts buffers in its parameter total; the notebook counts learnable parameters only. **The two numbers describe the same network.** Stated in the note to `tab:efficiency`. |

### 1.2 Cross-device timing comparisons — **removed** ✅

| | |
|---|---|
| Conflict | It is tempting to write "the transformer takes 2.4 times as long to train" from 2,172.8~s (ViT) against 892.0~s (MobileNetV2). Those two runs are on **different GPUs** — RTX 4050 Laptop and Tesla T4 — and the paper's own table says times are not comparable across devices. |
| Evidence | `efficiency.gpu_name` in each results JSON. Ten ablation rows are split across both devices (7 on the laptop, 3 on the T4). |
| Resolution | Every cross-device timing claim was deleted. The efficiency table now has **two device blocks**, the ablation rows appear in both, and the one timing claim made in prose is within-device: on the laptop GPU MobileNetV2 costs 21–23~s/epoch while DeiT-S costs 14.1, i.e. the transformer is *faster* per epoch despite ten times the parameters. That is a better observation than the one it replaced. |

---

## 1b. Changes made in revision v2 (2026-09-02)

Four things changed after the first version of this paper, at the supervisor's request.

### 1b.1 The Audio Spectrogram Transformer is now reported ✅

**Previously:** the fourth transformer was written `NOT RUN` throughout, and requirement 3
(four transformer models) was recorded as unmet.

**What was found:** it *was* run, early in the project. Its numbers survive in the
backbone-selection record (`Asif's/M12/results_M12.json`, `decision.comparison_table.M4`),
transcribed from an executed notebook: accuracy 0.5528, macro precision 0.5347, macro recall
0.4385, macro F1 0.4109, macro-variant score 0.6359, 86,385,668 parameters, 329.54 MB,
86.74 ms per sample. Two notebooks exist (`Asif's/M4-AST-soundevent.ipynb`,
`Barshon's/M4/M4_ast_backbone.ipynb`), neither with stored outputs.

**How it is reported, and why that way.** It is in the lower block of `tab:main`, in
`tab:efficiency`, in the fallback block of `tab:comparison`, and it has its own subsection
(`subsec:ast`). Four caveats travel with it and all four are stated:

1. it sits on the **identifier-fallback partition** — the one Fault 2 produced;
2. its score is the **macro variant** — the one Fault 1 describes;
3. its **preprocessing differs** — full-band 20–8000 Hz mel and a 512-point FFT, both
   deliberate deviations to preserve the pretraining;
4. it **committed no raw confusion matrix**, so the challenge-metric score is
   `NOT RECOVERABLE` — by us or by anybody.

**No number was invented.** In particular, 0.6359 is never presented as a challenge score and
never enters `tab:metricaudit`, because that table's whole point is that every row was
recomputed from a stored matrix.

**Why this makes the paper stronger, not weaker.** The paper's central recommendation is
*commit the raw confusion matrix so anybody can recompute the score*. The one model the
literature suggests should have won is the one model that cannot be checked. That is the
paper's argument happening to the paper, and `subsec:ast` says so explicitly.

**What is still true:** four transformers were run, but only three are comparable. The paper
never claims otherwise.

### 1b.2 Model nicknames removed from the body ✅

Internal run identifiers (`M22-v2`, `M40`, `M4`, `M35`, and so on) were execution-sequence
labels, not names, and they are gone from the body. Models are named by architecture
throughout, and the ablation rows were relabelled `REF`, `C1`–`C7`, `P1`–`P5` so that they no
longer echo repository internals either.

Traceability is preserved by **Appendix A** (`tab:runmap`), which maps every identifier to the
name used in the paper and to the table it appears in. One `M`-token remains in the body by
design: **M2D**, which is the real published name of the model in Niizumi *et al.*

### 1b.3 Four tables cut or merged as repetitive ✅

The table count went from 22 to **17 in the body plus 1 in the appendix**.

| Table | Fate | Why |
|---|---|---|
| `tab:corpus` | **cut** | almost every cell repeated the surrounding prose; its two load-bearing facts (COPD 83.3 %, 4 device-spanning patients) moved into a paragraph |
| `tab:preproc` | **cut** | the same chain is in the equations and in the "change from the reference" column of `tab:ablation`; it is now one prose sentence with equation pointers |
| `tab:augmentation` | **cut** | it re-presented numbers already in `tab:main`, which carries both runs of every backbone; the deltas and the noise-floor verdicts are now prose |
| `tab:failures` | **cut** | six of its eight rows appeared elsewhere; the unique content (which automated check caught each failure) is now two paragraphs |
| `tab:classdist` | **merged** into `tab:partitions` | both described the partitions |
| `tab:sameclips` | **merged** into `tab:human` | both are the human-versus-machine evidence |

Nothing was cut because it was inconvenient. Every cut table's evidence still appears, either
in another table or in prose that names the same numbers.

### 1b.4 Author list finalised ✅

Barshon Basak, Asif Mahbub, Sami Uddin, Farhana Rahman. The bracketed surname placeholders are
gone.

---

## 2. Conflicts between repository documents

| # | Conflict | Sources | Resolution |
|---|---|---|---|
| C1 | **Which model is best.** `RTK_requirements.md` §6 says "M22 (0.6495) is the current best"; the consolidation record and the committed JSONs say M22-v2 at **0.5602**. | `RTK_requirements.md` vs `DECISION_2026-08-30_CONSOLIDATION.md` §3 | **0.5602.** §6 predates the corrected re-run. 0.6495 is an identifier-fallback number and appears only as an audit exhibit in `tab:metricaudit`, `tab:protocol` and the last block of `tab:comparison`. |
| C2 | **Two different "corrected" splits.** `CORRECTED_BASELINES.md` reports M2/M3 on `official_icbhi_60_40_patient_disjoint` (**77/49 patients, 2,756 test cycles**, policy `drop_from_train`); M22-v2, M3-v2, M40–M42, M45 and M48 are on `official_60_40_patient_independent_corrected` (**79/47 patients, 2,636 test cycles**, policy `reassign_to_train`). These are *not* the same test set, and several repository documents put them in one column. | `results_M2.json`, `results_M3.json` vs `results_M22_v2.json` | Both are named and separated in `tab:corpus` as distinct partitions with an **overlap-policy column**. M2 and M3 never appear in `tab:main`; they appear only in `tab:metricaudit` (labelled "disjoint alt.") and in the published-partition block of `tab:comparison`, where they are legitimate. |
| C3 | **M3's corrected score.** Consolidation record says 0.5135; the committed JSON says **0.5132**. | `DECISION_2026-08-30_CONSOLIDATION.md` vs `Asif's/M3/29 aug run/results_M3.json` | **0.5132**, from the JSON. 0.5135 is `M12_v2`'s independent re-inference and differs by a few confusion-matrix cells. |
| C4 | **Metric inflation: "+0.11 across 14 models" or "+0.0948 across 20 runs"?** | `PROGRESS_REPORT.md` §2.1 and the pivot record vs `ICBHI_SCORE_AUDIT.md` | **+0.0948 across 20 runs** — the later, larger audit, recomputed from confusion matrices. The paper reports mean, max (0.2176) and min (0.0025). |
| C5 | **Device-spanning patients: 3 or 4?** | `DECISION_2026-08-16_PIVOT.md` says 3/126; the committed device report says 4 | **4** — recomputed independently here from the annotation index: patients **112, 158, 218, 226**. Stated in the note to `tab:corpus`. |
| C6 | **Corpus is 6,898 or 6,887 cycles?** | annotation files vs every model run | **Both, and the difference is explained.** The annotations hold 6,898; the model pipeline indexes 6,887 because `226_1b1_Pl_sc_Meditron` in the split file is released as `..._LittC2SE`, so a stem join drops it and its 11 cycles. Verified: corrected train is 4,262 by annotation and 4,251 by pipeline; the test partition is 2,636 either way. The paper reports 6,898 for the corpus, 4,262/2,636 for the corrected partition, and explains the 11 in the note to `tab:classdist`. |
| C7 | **The human-benchmark citation.** Cited project-wide as "Aviles-Solis et al. 2016". | `Papers/HUMAN_BENCHMARKS.md` attribution correction of 2026-08-30 | **Melbye et al. 2016**, *BMJ Open Respiratory Research* 3(1):e000136. Aviles-Solis is a co-author on other papers from the same group. The `.bib` entry is correct. |
| C8 | **The label-reliability ceiling claim.** `DECISION_2026-08-16_PIVOT.md` and `PAPER_OUTLINE.md` argue that ICBHI's labels bound concept detection; `PROGRESS_REPORT.md` §0 still states it as a finding. | N11 (`N11_supervised_ceiling.json`), run 2026-08-30 with pre-registered readings | **Retracted, and the retraction is reported as a result** (`subsec:ceiling`). A frozen AudioSet probe reaches 0.7115/0.7621 on the same test cycles. `PROGRESS_REPORT.md` §0 is stale on this point. |
| C9 | **Two different G2 number pairs.** The 9-concept M39 gate gives 0.5506/0.5729 then 0.5580/0.5340; a later 14-concept engine gives 0.5556/0.5818. Earlier analyses conflated them. | `Asif's/M39/*` vs `owmtl_concept_engine/notebooks/01_*` | **Never mixed.** `tab:gate` cites only the pre-registered 9-concept runs. The 14-concept suite appears once, in `tab:ceiling`, explicitly labelled "our 14 concepts as a vector". |

---

## 3. Where the previous draft was right, and this version follows it

Recorded so these are not re-litigated:

- The 20-row metric audit — mean 0.0948, max 0.2176, min 0.0025. Reproduced exactly.
- The protocol ablation including `E1+E2 = 0.7077`, and the decision to report `E3`'s
  near-zero delta as a **null**. Reporting the leak as a validity fault worth $+0.0039$
  rather than as inflation is the single best judgement in the earlier draft, and it is
  kept.
- The refusal to promote ablation row `P3` (0.5764) over the reference despite it being
  the highest corrected score, because its patient-level interval spans zero.
- The exclusion of Suma *et al.* and Cho & Lee from the comparison table because neither
  reports a number that table can hold.
- The removal of the unverifiable `btspp2025` reference rather than citing it
  second-hand.
- Reporting the ViT collapse rather than deleting it, and calling it the weaker
  (small-data) claim because the warm-up re-run was never done.

---

## 4. Differences from the previous draft that are choices, not corrections

| | Previous draft | This version | Why |
|---|---|---|---|
| Length | 33 tables, 9 figures, ~2,370 source lines | **17 body tables + 1 appendix table, 4 figures**, estimated 17.9 pages | The 18-page limit. Tables were merged (per-class into error analysis; seed band into the statistics table; corpus and partitions into one float) or cut where another table carried the same evidence. |
| Metric-inflation figure | Included as a scatter | Cut | `tab:metricaudit` carries all 20 rows and is strictly more informative; the figure cost 0.7 of a page. |
| Cumulative ablation ladder | A table with 3 of 6 rungs marked pending | Two sentences in `subsec:ablation` | Half the table was `pending`; the measurable part is the ordering effect (+0.0402 vs +0.0105), which is one sentence. |
| Abstract | 303 words | **293 words** | Requirement is 200–300. |
| Pediatric / SPRSound arm (N8) | not in either draft's tables | omitted, and the `.bib` entry removed | It is a real pre-registered positive result about concept fragility, but it is a different corpus and a different task, and it did not fit the page budget. **It is available if the supervisor wants it back.** |

---

## 5. Unresolved — no number exists, and none was invented

| Item | State | Consequence for the paper |
|---|---|---|
| **A corrected AST re-run** — the audio-pretrained transformer under the current protocol | notebook written and self-tested; needs a 16 GB accelerator | The pre-audit AST run is now reported (§1b.1) but is not comparable to anything. A re-run is still the single most informative missing experiment, because it is the only one that tests the pretraining-domain hypothesis directly instead of by inference from other people's tables. |
| **The AST run's confusion matrix** | never committed; the checkpoint is not in the repository | Its challenge score is permanently unrecoverable. Reported as `NOT RECOVERABLE`, never estimated. |
| **Checkpoint selection reads the test partition** | known, priced, not fixed | The first item in `subsec:limitations`. Every ablation score carries the same selection advantage. Fixing it means carving a validation split from the 79 training patients and re-running the suite — the highest-value remaining experiment. |
| **Cumulative ladder rungs S0, S1, S2** | rows exist in `m48_gpu_rows.py`; ~55 min GPU | The ordering effect is reported without them; the ladder itself is not tabulated. |
| **M35 on the corrected partition** | exists only at 70/30 | The single number that could change the top of `tab:main`. Reported in `tab:extensions` with its partition, never ranked against the corrected rows. |
| **M40 ViT warm-up re-run** | not attempted | The collapse is reported as a small-data finding, which is the weaker of the two available claims. A warm-up schedule with a lower backbone learning rate is the standard remedy and was not tried. |
| **Split effect measured on one model chain** | M22 → M22-v2 only | M2's and M3's apparent split effects are confounded with architecture (different backbones, different widths, different learning rates), so they are never quoted as split effects. |
| **Single clinician** | no second rater | No inter-rater agreement is computable. Mitigated by the intra-rater duplicates and by the exact replication of the published seven-physician sensitivity (23.1 vs 23.23 percent), not removed. |
| **N5 permutation null at n = 5** | not raised to ≥ 20 | The null *mean* is quoted nowhere in this paper; the leakage result is not used. No action needed for this draft. |

---

## 6. Citation metadata still to confirm

| Entry | What to check |
|---|---|
| `nguyen2022cotuning` | Volume / number / pages came from the project's literature notes rather than the publisher record. Flagged in `references.bib`. |
| `electronics2025review` | Cited for "well over a hundred" published systems rather than an exact count, because the publisher blocks automated access. Correct at that precision. |
| `adffnet2025`, `bspc2025fusion` | Cited only for "gains of roughly one point". Author lists should be confirmed against the publisher record before submission. |

---

## 7. Two things a reader could still object to, and the honest answer

**"Your best model is worse than a 2021 paper."** Yes — 0.5641 on the published partition,
against 0.5620 for RespireNet and 0.6484 for the best audio-pretrained system. The paper
says so in `subsec:comparison`, and then decomposes it: our *sensitivity* (0.456) is
mid-field, and the entire deficit is specificity. The contribution is the measurement,
not the model.

**"You only audited yourselves."** True, and deliberate. Every fault is presented as one
we made and caught, with the corrected number beside it. The implication for the wider
literature is left unstated, which is both more honest and harder to argue with.

---

## 8. Reproducing the checks

```
cd "Final_Draft _v2"
python verify.py        # cross-refs, bibliography, braces, table columns,
                        # abstract/keyword rules, page estimate. No LaTeX needed.
```

`verify.py` cannot replace a real compile. **Compile once on Overleaf and confirm zero
errors, no `??` in the PDF, and the actual page count** before submitting. The page
estimate is approximate and currently reads 17.9 against the 18-page limit; if the real
compile runs over, the cheapest levers, in order, are: reduce `fig:flowchart` and
`fig:xai` widths (now 0.46 and 0.50), shorten the notes under `tab:partitions`, `tab:human`
and `tab:comparison`, or fold `tab:ablationreason` into the ablation prose.
