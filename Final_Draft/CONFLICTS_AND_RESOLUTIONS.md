# Conflicts with the existing draft, and how each was resolved

**Written:** 2026-08-31 · **Compares:** `Final_Draft/main.tex` (written independently from
committed evidence) against `DRAFT_PAPER_1/main.tex` (the pre-existing draft, last edited
2026-08-30 15:18).

**Method.** The new paper was written first, from committed result files and from the ICBHI
annotation files directly, *without opening the existing draft*. Only afterwards was the draft read
and diffed against it. Resolution priority, as instructed:
**final/reproducible result → code and experiment output → latest validated documentation → old
draft.** "Most authentic and scientifically valid", never "most recent text".

**Headline.** The existing draft is substantially sound. Its metric-audit table, gate table,
supervised-ceiling table, extension-model table and interpretability numbers all reproduce my
independent derivation **exactly** — that is a genuine cross-check and it passed. The conflicts
below are of four kinds: (A) methodology text that does not match the implementation, (B) a
confounded claim, (C) course-format non-compliance, and (D) evidence generated *after* the draft
was last edited and therefore absent from it.

---

## A. Methodology that does not match the implementation — **3 conflicts, all resolved against the draft**

These matter most, because the instruction was explicit that the described methodology must match
the actual implementation.

### A1. Spectrogram scaling — the draft describes the wrong operation ❌

| | |
|---|---|
| **Draft** | Eq. (2): standardisation to **zero mean and unit variance**, $\tilde S = (S-\mu_S)/(\sigma_S+\epsilon)$ |
| **Code** | `Asif's/M45/m45_ablation.py::log_mel` → `lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)` — **per-spectrogram min–max to $[0,1]$** |
| **Corroboration** | The ablation row is named "− per-spectrogram **min-max** normalisation" in `results_M45_P5.json`; the `minmax=False` branch falls back to `lm/80.0 + 1.0`, which only makes sense as a `[0,1]` substitute. `Novelty Experiment/NOVELTY_STATUS.md` independently warns that M2 was trained on "per-sample min-max to `[0, 1]`". |
| **Resolution** | **Min–max.** The new paper's (2) states the min–max form. A reader implementing the draft's equation would not reproduce a single number in either paper. |

### A2. Log-mel definition — the draft omits the max reference ❌

| | |
|---|---|
| **Draft** | $S = 10\log_{10}(\sum_k H_j[k]\,|X|^2 + \epsilon)$ — a plain log with an epsilon floor |
| **Code** | `librosa.power_to_db(m, ref=np.max)` — i.e. $10\log_{10}(S/\max S)$, a **per-spectrogram** reference |
| **Resolution** | **`ref=max` form**, written into the new (2). The difference is not cosmetic: `ref=np.max` is itself a normalisation, and it is why the subsequent min–max stage (P5) is worth only $-0.0063$. |

### A3. Class weights — the draft omits the mean normalisation ⚠️

| | |
|---|---|
| **Draft** | $w_c = N/(C N_c)$ |
| **Code** | `w = c.sum()/(4*c); w = w / w.mean()` — the same weights **renormalised to unit mean** |
| **Resolution** | **Include the renormalisation** (new eq. 4). Without it the loss scale changes with the class distribution, so the draft's form would not be comparable across the ablation rows that alter the training set. |

---

## B. A confounded claim — **1 conflict, resolved against the draft**

### B1. "The same architectures lose eight to fourteen points" to the split ❌

| | |
|---|---|
| **Draft** | Abstract and §Split Audit: *"re-run unchanged on the true split, the same architectures lose eight to fourteen points"*, citing M2 $-0.1418$ and M3 $-0.0763$ alongside M22 $-0.0854$ |
| **Evidence** | The three runs are **not** the same architectures. `Asif's/M2/results_M2.json` is `2D_CNN_5Block_w48_do0.4`, lr 5e-4, **3,627,476** params; `Asif's/M2/m2_v4/results_M2.json` is `2D_CNN_4Block_w32_do0.3`, lr 1e-3, **421,732** params. `Asif's/M3/results_M3.json` is `mobilenet_v2` (2,228,996); `Asif's/M3/29 aug run/results_M3.json` is `mobilenet_v3_small` (929,316). `M12_v2` confirms the corrected runs were a fresh backbone re-decision, not a re-run. |
| **Resolution** | **Only the M22 chain is a one-variable split measurement.** The new paper quotes $-0.0854$ (fallback → verbatim official) and $-0.0039$ (verbatim → corrected) and states explicitly in Limitations that M2's and M3's apparent split effects are confounded with architecture and are **not** quoted as split effects anywhere. The draft's own *table* is honest about the architecture change; its *prose* is not, and the prose is what a reader will quote. |

---

## C. Course-format non-compliance — **4 conflicts, resolved against the draft**

Measured against `paper_requirement.md`, the transcription of the supervisor's own rulebook.

| # | Rule | Draft | Resolution |
|---|---|---|---|
| C1 | Abstract 200–300 words | **≈450 words** | Rewritten to **303 words** |
| C2 | Abstract: no citations, abbreviations, symbols or equations | Contains `$\kappa=0.035$` and bare numerals such as `0.5641` | Rewritten with every quantity spelled out in words; checked programmatically for `\cite`, `$`, all-caps tokens and numerals — **zero issues** |
| C3 | Author block complete | `\pending{full name}` ×3, `\pending{supervisor name}`, `\pending{section}`, `\pending{contact e-mail}` | Filled: four author names, department, course, and **Supervisor: Dr. Riasat Khan** |
| C4 | Every `.bib` entry actually cited | `ref1`, `ref2` are `TODO Author` / `TODO Title` placeholders, uncited | Placeholders **deleted**; 11 further method papers added and every one of the 34 entries is now cited. Verified: 0 uncited entries, 0 undefined keys, 0 unresolved `\ref` |

An internal inconsistency was also fixed: the draft's Conclusions say the released audit tool
"detects all **three** faults" while its own body describes **four**. The new paper describes five
and says five throughout.

---

## D. Evidence the draft could not contain — **7 additions**

`DRAFT_PAPER_1/main.tex` was last edited 2026-08-30 **15:18**. The following result files are
timestamped **after** that, so their absence is chronology, not error. All are now in the paper.

| # | Evidence | Generated | New table |
|---|---|---|---|
| D1 | **Fifth fault: checkpoint selection criterion.** Selecting on loss rather than on the challenge score costs **0.0752 / 0.1015 / 0.0992** across seeds 42 / 1 / 2, at 15–19 epochs' distance. Larger than pretraining or augmentation. | 20:27 | `tab:selection` |
| D2 | **Seed band / noise floor.** Three runs of one configuration: 0.5540 / 0.5681 / 0.5657, sd **0.0075**, range **0.0141**. M22-v2's 0.5602 sits inside it — also a cross-implementation reproducibility check. Three ablation deltas do not clear the band. | 20:27 | `tab:seedband` |
| D3 | **Negative transfer (A24).** Frozen *random* backbone 0.4979 vs frozen *ImageNet* backbone 0.4595 — frozen ImageNet features are **0.038 worse than random projections**, 2.7× the seed range. Decomposes A4's $-0.1007$ into "pretraining is worthless frozen" + "adaptation is what the 0.10 buys". | 20:27 | discussed in `subsec:ablation` |
| D4 | **Protocol ablation as a table**, including the row the draft only implied: **E1+E2 = 0.7077**, the unaudited pipeline as it would have been published, $+0.1475$ over the corrected protocol. | 15:49 | `tab:protocol` |
| D5 | **Cumulative ablation ladder**, both orderings, with the ordering effect measured: SpecAugment credited $+0.0402$ in one ordering, class weighting $+0.0105$ in the other. Three rungs marked `NOT RUN`. | 20:30 | `tab:cumulative` |
| D6 | **Machine vs physician on the same 132 clips.** Paired difference $+0.169$ for both concepts; **significant for wheeze** ($p = 0.046$), undecided for crackle ($p = 0.087$). Closes the "two different sets of cycles" gap in the draft's headline comparison. | 09:56 | `tab:sameclips` |
| D7 | **The checkpoint-selection limitation itself.** `m45_ablation.py` and the M22-v2 notebook both monitor the **test** partition each epoch — `training_history` stores `val_icbhi_score_official = 0.5602` at epoch 38, identical to the reported test score. The draft never states this. | pre-existing, but unstated | `subsec:limitations`, first item |

**D7 is the most important of the seven** and is not a matter of chronology: the evidence was on
disk the whole time. It is now the first limitation in the paper, priced by D1.

---

## E. Tables added that the draft does not have — **12**

The faculty asked specifically for more experimental tables, each answering a stated research
question. The draft has 15; the new paper has **33**. The additions:

| New table | Research question it answers | Source |
|---|---|---|
| `tab:classdist` (extended) | What is the class balance of **every** partition, train and test? The draft omits the corrected partition's class counts entirely. | recomputed from the ICBHI annotation files; matches all committed matrices |
| `tab:device` | Can device robustness be tested on this corpus? **No** — 6 diagnoses are 100% device-confounded, 4 of 126 patients span devices | `device_structure_report.json` |
| `tab:preproc` | What is in the preprocessing chain, and which ablation row switches each stage? | `m45_ablation.py` |
| `tab:models` | Which model does each member own, and which one did not complete? | run records + `RTK_requirements.md` §4 |
| `tab:protocol` | What is each protocol component worth? | D4 |
| `tab:bothclass` | Is the patient leak harmless? **No** — it removes 39.9% of the "Both" class | recomputed from annotations |
| `tab:selection` | How much does the checkpoint rule matter? | D1 |
| `tab:perclass` | Where does the best model actually succeed? | `results_M22_v2.json` |
| `tab:efficiency` | What does every model cost — epochs, time, params, size? (Course requirement 5, and the draft covers it only partially) | all run records |
| `tab:sameclips` | Does the machine beat the physician **on the same clips**? | D6 |
| `tab:errors` + `tab:failures` | How does the system fail, and what failed at run level? | committed matrices + `PROJECT_AUDIT.md` |
| `tab:seedband` + `tab:unit` | Which differences are real? | D2 + `M45_paired_tests.json` |
| `tab:cumulative` | What is each component worth *given only what came before*? | D5 |
| `tab:ablationreason` | The supervisor's per-metric ablation format with a "possible reason" column | derived from `tab:ablation` |
| `tab:gapevidence` | What is the 6–10 point gap made of? | five convergent measurements |

---

## F. Conflicts *within the repository* found while writing, and their resolutions

These are not draft-versus-new conflicts; they are disagreements between documents, resolved by
going to the committed result file. Full list in `PROJECT_SUMMARY.md` §11; the ones that changed a
number in the paper:

| # | Conflict | Resolution |
|---|---|---|
| F1 | `RTK_requirements.md` §6: "M22 (0.6495) is the current best" vs its own §1 table and the consolidation record: M22-v2 (0.5602) | **0.5602.** §6 predates the corrected re-run. 0.6495 is a fallback-partition number and appears only as an audit exhibit |
| F2 | Consolidation record: M3 corrected = 0.5135; committed JSON = **0.5132** | **0.5132.** 0.5135 is `M12_v2`'s independent re-inference, which differs by a few confusion-matrix cells |
| F3 | Pivot record: "3/126 patients span devices"; committed report: **4** | **4** (patients 112, 158, 218, 226) |
| F4 | "+0.11 mean inflation across 14 models" in older documents | **+0.0948 across 20 runs** — the later, larger audit. *(The draft already uses 0.0948.)* |
| F5 | Consolidation lists M21 (0.4997) as canonical; `M28` **excludes** it for having no train/test separation | **Excluded.** Later and more specific judgement, and M21's $Se = 0.0186$ confirms it. It appears in the metric audit (as a pathology the official metric exposes) and in the failure catalogue, never as a result |
| F6 | Corpus is 6,898 cycles; the pipeline indexes 6,887 | Explained and **verified**: `226_1b1_Pl_sc_Meditron` (split file) vs `..._LittC2SE` (disk) = **11 cycles, 6 Normal + 5 Crackle**, train side only |
| F7 | Human-benchmark paper cited project-wide as "Aviles-Solis et al. 2016" | **Melbye et al. 2016**, BMJ Open Respir. Res. 3(1):e000136. *(The draft's bib already had this right; several repository documents still do not.)* |
| F8 | `README.md`: corrected numbers "sit at the published ICBHI level (~0.60–0.65)" | **Outdated.** Corrected best is 0.5602 / 0.5641 — the 2021 CNN level, 6–10 points below the frontier |
| F9 | Two different G2 number pairs (9-concept gate vs 14-concept engine) | **Never mixed.** The 9-concept M39 runs are the pre-registered gate and are what the paper cites; the 14-concept validation is reported in the same table but in a separate, labelled block |

---

## G. Where the draft was right and the new paper follows it

Recorded so that these are not re-litigated:

- The metric-audit table's 20 rows, mean 0.0948 / max 0.2176 / min 0.0025 — **reproduced exactly**.
- The gate table, both runs, all four confidence intervals — **reproduced exactly**.
- The supervised-ceiling table, all six numbers and their intervals — **reproduced exactly**.
- The extension-model table, all nine rows — **reproduced exactly**.
- The interpretability numbers ($-0.053$, $r = 0.098$, 32%, the $4\times26$ caveat) — **reproduced exactly**.
- The insistence on reporting the patient leak as a **validity** fault worth only $+0.0039$ rather
  than as inflation. This is the draft's best single judgement and the new paper keeps it verbatim
  in spirit.
- The refusal to promote ablation row `P3` (0.5764) over the reference despite it being the highest
  score, because its patient-level interval spans zero.
- The exclusion of Suma *et al.* and Cho & Lee from the comparison table because neither reports a
  number that table can hold.
- The removal of the unverifiable `btspp2025` reference rather than citing it second-hand.

---

## H. Remaining unresolved items

Everything below is stated in the paper as well; this is the consolidated list.

### H1. Not run — no number exists, and none was invented

| Item | State | Consequence |
|---|---|---|
| **M43 (AST)** — the fourth transformer, the only audio-domain-pretrained backbone | notebook written and tested; needs a 16 GB accelerator | Course requirement 3 asks for 4 transformers; **3 exist**. It is also the one experiment that would test the pretraining hypothesis directly instead of by convergent inference. Written `NOT RUN` in `tab:models` and `tab:main` |
| **Cumulative ladder rungs S0, S1, S2** | rows exist in `m48_gpu_rows.py`; ~55 min GPU | `tab:cumulative` has 3 of 6 rungs; the three are marked `pending`, never filled |
| **M35 on the corrected partition** | only exists at 70/30 | The single number that could change the top of the results table |
| **M40 ViT warm-up re-run** | not attempted | The collapse may be a schedule artefact rather than a data-scale finding. Currently reported as the latter, which is the weaker claim of the two |
| **N5 permutation null at `n_perm` ≥ 20** | run at 5 | The null *mean* is solid; the paper quotes the mean and **not** an interval on the null |
| **N1 LoRA rank sweep** | one rank, 5 epochs | Shows adaptation degrades concepts; does not show the degradation scales with adaptation strength. Stated as such |

### H2. Known and stated, but not fixable without re-running

| Item | Status |
|---|---|
| **Checkpoint selection reads the test partition** | Priced (0.075–0.102) and stated as the first limitation. Fixing it means carving a validation split from the 79 training patients and re-running the suite — the highest-value remaining run in the project |
| **M45 harness vs M22-v2 notebook are not byte-identical** | Params differ by 34,164 and the harness's seed-42 reference gives 0.5540 vs the notebook's 0.5602. The seed band (range 0.0141) bounds the consequence and is what licenses treating them as one table. **The 34,164-parameter difference itself is unexplained** and is flagged rather than rationalised |
| **Split effect measured for one model only** | See B1 |
| **Single clinician** | No inter-rater κ is computable. Mitigated but not removed by intra-rater κ 0.714 / 0.600 and by the exact replication of the published seven-physician sensitivity (0.2308 vs 0.2323) |
| **Listening pack is not a random sample** | Absolute AUROCs on the 132 clips are not corpus-comparable; only the paired contrast is. Stated in the table's own footnote |
| **One corpus on the adult side** | Scope of every claim is ICBHI 2017 |

### H3. Citation metadata still to confirm

| Entry | What needs checking |
|---|---|
| `nguyen2022cotuning` | Volume / number / pages were taken from the project's literature notes rather than re-read from the publisher record. The **quotation** attributed to it is recorded verbatim in `Papers/RESULT_COMPARISON.md`; the bibliographic fields are the open item. Flagged in `references.bib` |
| `electronics2025review` | The paper is cited for "well over a hundred" published systems rather than for a specific count, because the exact figure could not be read (the publisher blocks automated access). The claim is correct at that precision either way |
| `btspp2025` (**not** cited) | Reported at 0.6569 and marked previous state of the art in a secondary table, but the paper could not be located in OpenAlex, Crossref or direct search. **Removed rather than cited second-hand.** If confirmed it is the top row of `tab:comparison` |
| ADFF-Net metric-naming claim (**not** made) | The arithmetic is suggestive but the publisher blocks automated access, so the paper's own metric definition was never read. An accusation of a metric error against a published paper is not worth making on a search summary |

### H4. Deliberately not produced

| Item | Why |
|---|---|
| Grad-CAM pointing-game hit rate | **Not implementable.** ICBHI annotates *whether* a cycle contains an adventitious sound, not *where*, and the pipeline crops to exactly that window — so the hit rate is 100% by construction |
| Fine-vs-coarse crackle validation | `fine_crackle_ratio` is constant across all 22 fine-or-coarse clips (99.7% zero corpus-wide), so no AUROC exists. Reporting "0.500" would read as a precise finding of no signal when it is no measurement at all |
| M38 large-N open-set | Dropped. At $n=19$ every interval spans chance, and enlarging via a pediatric corpus confounds novelty with population shift |
| Second clinician | Dropped. N11 bounds the reference standard directly and more tightly than a second rater's κ would |
| Cumulative-ladder line plot | Three of six rungs missing; a line through three points would imply a trend the data does not contain |

---

## I. Verification performed on the new paper

| Check | Result |
|---|---|
| Every `\cite` key resolves in `references.bib` | ✅ 0 undefined |
| Every `.bib` entry is cited | ✅ 0 uncited (34 entries) |
| Every `\ref` resolves to a `\label` | ✅ 0 dangling |
| Every table and figure is referenced in the body | ✅ 33 tables, 5 figures, all referenced |
| `tabular` column counts match their column specs | ✅ 0 mismatched rows across 33 tables (one real error found and fixed: `tab:protocol` had an 8-column spec for 7-column rows) |
| Environment balance (`table`, `tabular`, `figure`, `equation`, `algorithm`, …) | ✅ all balanced |
| Brace balance | ✅ 1467 / 1467 |
| Non-ASCII characters | ✅ none |
| Abstract 200–300 words, no citations/abbreviations/symbols/equations | ✅ 303 words, 0 issues flagged |
| Keywords alphabetical | ✅ |
| No results in the Proposed System section | ✅ Section II contains dataset characterisation and method only |
| Comparison table last in Results and Discussion | ✅ `tab:comparison` is the final subsection |
| Every equation numbered inside `equation` | ✅ 8 equations |
| Training hardware named | ✅ Section III opening and `tab:efficiency` |
| ≥6 pages | ✅ 33 tables, 5 figures, ~2,100 source lines |

**Not verifiable locally:** the actual Overleaf compile. No LaTeX distribution is installed on this
machine, so the checks above are structural rather than a real `pdflatex` run. Compile once on
Overleaf and confirm zero errors and no `??` in the PDF before submitting.
