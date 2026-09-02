# Rewrite work split — three members

**Paper:** `main.tex` · **Read first:** `Paper_rewrite_understanding_helper.md`
**Measured totals to divide:** 7,885 prose words, 18 tables, 4 figures, 9 equations.

The split below is by **argument**, not by page count. Sections that share a claim are owned
by one person, because the fastest way to wreck this paper is to have two people restate the
same result in two slightly different ways.

---

## 0. Before anybody writes a word

Everyone reads three things, once, together if possible:

| Read | Why | Time |
|---|---|---|
| `Paper_rewrite_understanding_helper.md` **Parts 1 and 2** | the project history and the seven-beat argument. Nothing else makes sense without it | 20 min |
| The same file, **Part 8** | the eleven claims that must never re-enter the paper | 5 min |
| The same file, **Part 9.1–9.2** | the style rules and the evidence discipline | 5 min |

Then each member reads only their own sections in Part 3 and their own tables in Part 4.

---

## 1. The three blocks

### Member A — Framing and methodology
**Sections:** Abstract stub, Keywords, I (all), II (all)
**Prose:** ~3,050 words · **Tables:** 2 · **Figures:** 1 · **Equations:** all 9

| Subsection | Words | Owns |
|---|---:|---|
| Introduction | 162 | — |
| Literature Review | 547 | four themed blocks, ≥4 papers |
| Gaps in the Related Work | 225 | the four gaps |
| Our Work | 528 | **the five-item novelty paragraph** |
| Proposed System opening | 46 | `fig:flowchart` |
| Dataset | 189 | — |
| Partitions and class balance | 179 | `tab:partitions` |
| Dataset Preprocessing | 515 | equations (1)–(6) |
| Applied Models | 158 | `tab:models` |
| Evaluation Metrics | 208 | equations (7)–(9) |

**Why these go together.** Whoever writes the Introduction must write the roadmap and the
novelty paragraph, and whoever writes the preprocessing must own the equations that describe
it. This is also the block with no results in it at all, which is a hard rubric rule — if a
score appears anywhere in Section II, that is Member A's bug.

**Hardest part:** the novelty paragraph in *Our Work*. Its five items each point at a specific
table owned by B or C. Do not reword an item so that it no longer matches its table.

**Easiest part:** the Dataset and Partitions prose. Mostly description.

---

### Member B — The audit, the models, and the comparison
**Sections:** III opening, III-A (all five faults), III-B, III-C, III-D, III-E, III-M
**Prose:** ~2,120 words · **Tables:** 6 · **Figures:** 0

| Subsection | Words | Owns |
|---|---:|---|
| Results opening (hardware) | 109 | — |
| Five Faults, intro + Faults 1–5 + summary | 597 | `tab:metricaudit`, `tab:selection`, `tab:protocol` |
| Main Results | 286 | `tab:main` |
| The Fourth Transformer (AST) | 387 | — |
| Model Cost | 152 | `tab:efficiency` |
| Data Augmentation | 247 | — |
| Comparison with Existing Works | 293 | `tab:comparison` |

**Why these go together.** This is the spine. Faults 1 and 2 produce the two rows of
`tab:protocol` that produce `E1+E2 = 0.7077`, which is the number the comparison table's last
block prices, which is what the closing paragraph is about. One person, one voice, or the
argument fragments.

**Hardest part:** the AST subsection. It has to say four separate things are wrong with one
run, and then turn that into the paper's strongest rhetorical moment, without ever giving the
model a challenge-metric score.

**The sentence you must not soften:** *"this one cannot be recomputed by anybody, including
us."*

---

### Member C — Deep dives, ablation, and the closing
**Sections:** III-F, III-G through III-L, IV, Appendix A
**Prose:** ~2,715 words · **Tables:** 10 · **Figures:** 3

| Subsection | Words | Owns |
|---|---:|---|
| The Best Model in Detail | 160 | `tab:errors`, `fig:cm`, `fig:curves` |
| Extension Experiments | 166 | `tab:extensions` |
| The Pre-Registered Concept Gate | 123 | `tab:gate` |
| Is 0.65 Reachable? The Retraction | 188 | `tab:ceiling` |
| Machines Against Physicians | 167 | `tab:human` |
| Explainable AI | 335 | `tab:xaiquant`, `fig:xai` |
| Statistical Analysis | 225 | `tab:stats` |
| Ablation Study | 494 | `tab:ablation`, `tab:ablationreason` |
| Failure Analysis | 201 | — |
| Limitations | 274 | — |
| Appendix A | 53 | `tab:runmap` |

**Why these go together.** The gate, the retraction and the human comparison are one story told
in three steps. The seed band in `tab:stats` is what licenses the "vs noise" column in
`tab:ablation` — same person, or those two tables will disagree. And the XAI tiling result is
what motivated ablation row `P4`, so the same person should write both.

**Hardest part:** the Ablation Study, at 494 words and two tables, including the `C7`
negative-transfer decomposition.

**Most tables, fewest words.** Most of C's table work is rewriting captions and notes, not
building tables. Budget accordingly.

---

## 2. Phasing — nobody starts the Abstract

```
Phase 1  (parallel, ~4 days)   A, B, C rewrite their block bodies
Phase 2  (B, ~half a day)      Abstract + Conclusions, written FROM the finished body
Phase 3  (C, ~half a day)      consistency pass, verify.py, Overleaf compile
Phase 4  (all three, 1 hour)   read the whole paper end to end, once, out loud if possible
```

**Why B writes the Abstract and Conclusions.** Both are recaps of the spine, and B owns the
spine. They cannot be written until A and C are done, which is why they are Phase 2 and not
part of Phase 1. The Abstract currently in the file is a placeholder for this purpose — B
rewrites it last, from what the paper actually ends up saying.

**Why C does integration.** C has the fewest Phase 1 words and the most cross-references to
check.

---

## 3. Working on one file without fighting each other

Three people editing one `main.tex` on Overleaf will collide. Two options:

**Option 1 — time-slice (zero setup).** Agree that only one person edits at a time, in the
order A → B → C, each finishing their block before handing over. Slow but safe.

**Option 2 — split the file (10 minutes of setup, recommended).** Cut `main.tex` into:

```
main.tex          preamble, \maketitle, \input{...} lines, bibliography
sec_front.tex     abstract + keywords + Section I          (A, then B in Phase 2)
sec_system.tex    Section II                                (A)
sec_audit.tex     Section III opening through III-E + III-M (B)
sec_deep.tex      III-F through III-L                       (C)
sec_close.tex     Section IV + Appendix A                   (B writes IV, C writes App. A)
```

Then each person edits one file and Overleaf stops merging conflicts. **One caveat:**
`verify.py` reads `main.tex` only, so it would need its first line changed to concatenate the
inputs. Say the word and I will do the split and patch the checker in one go.

---

## 4. Shared rules — everyone, every section

### 4.1 Do not change these, whoever owns the section

| Thing | Why |
|---|---|
| Any number | Part 7 of the helper is the provenance index. If a number is not there, it does not go in |
| Any hedge | "not shown to beat" never becomes "beats". "undecided" never becomes "shows" |
| Partition labels on any row | rows on different partitions are never rankable |
| `NOT RUN` / `NOT RECOVERABLE` | never replaced with an estimate |
| The `P1`–`P3` sign convention | those rows *add* a stage; their deltas read backwards |
| The five self-critical passages | the `E3` null, the refusal to promote `P3`, the retraction, the test-monitoring admission, the AST unrecoverability. These are where the paper's credibility lives |

### 4.2 Naming conventions — agreed once, used by all three

| Write this | Not this |
|---|---|
| MobileNetV2, ViT-B/16, Swin-T, DeiT-S, AST | M22-v2, M40, M41, M42, M4 |
| `REF`, `C1`–`C7`, `P1`–`P5` | A0, A1, A24 |
| the corrected partition / the published partition / the identifier-fallback partition | "our split", "the official split" |
| the challenge metric | "the ICBHI score" (ambiguous — that is the whole problem) |
| the macro variant | "our metric" |
| physics-derived concepts | "label-free concepts" |
| Melbye *et al.* | Aviles-Solis *et al.* |
| spectrogram masking / SpecAugment | pick one per section, be consistent inside it |

### 4.3 Voice
First person plural. Varied sentence length. No *furthermore, moreover, additionally, it is
important to note, in conclusion, crucial, pivotal, comprehensive, leverage, delve*. Blunt
sentences are allowed and several are deliberate.

---

## 5. Cross-block dependencies — the six places two people must agree

These are the only real coupling points. Each has one owner and one consumer; the consumer
cites, the owner defines.

| # | Defined by | Used by | What must match |
|---|---|---|---|
| 1 | A — equations (7), (8) | B — Fault 1 | B cites (7) and (8) and never restates them. The "rare classes score near 0.95 on specificity" explanation lives in A's section |
| 2 | A — equation (9) | C — the gate | The 0.65 floor and the interval condition |
| 3 | C — `tab:stats` seed band, 0.0141 | B — augmentation deltas; C — ablation "vs noise" | The noise floor value, and the phrase used for a sub-floor delta |
| 4 | C — `tab:stats` unit-of-analysis block | B — `tab:protocol` row `E5` | 7 of 10 per cycle, 1 of 10 per patient |
| 5 | B — the AST subsection | C — Limitations item 2 | Why the image-vs-audio pretraining comparison cannot be made |
| 6 | B — `tab:main` best model 0.5602 | C — Best Model, Ablation `REF` | The best-model selection rule, stated identically in both |

**Rule:** if you need a number that another member owns, take it from Part 7 of the helper,
not from their draft. That way you cannot both drift.

---

## 6. Per-member handoff checklist

Tick before saying "done".

**Member A**
- [ ] No score, accuracy or delta anywhere in Section II
- [ ] All nine equations in `equation` environments, numbered, each referenced from prose
- [ ] Equations (2), (3) and (5) still match the code — `ref=max`, min–max, unit-mean renormalisation
- [ ] The novelty paragraph still has five items, each pointing at its table
- [ ] At least four journal papers reviewed, gaps paragraph present
- [ ] Keywords still alphabetical

**Member B**
- [ ] All five faults still named and priced
- [ ] `E3` still reported as a **null**, with the validity-versus-inflation distinction
- [ ] The selection-criterion caveat ("both criteria on one monitoring set") survives
- [ ] AST never gets a challenge-metric score; 0.6359 appears only as the macro variant
- [ ] No cross-device timing comparison anywhere
- [ ] `tab:comparison` is the **last** thing in Section III

**Member C**
- [ ] `P1`–`P3` sign convention intact
- [ ] The two-implementation note under `tab:ablation` survives (0.5540 vs 0.5602)
- [ ] `P3` still not promoted, with the reason given
- [ ] The XAI resolution caveat (4×26 cells) survives
- [ ] The retraction is still unambiguous
- [ ] Test-monitoring is still the **first** limitation
- [ ] Appendix A updated if any model was renamed

**Integration (C, Phase 3)**
- [ ] `python verify.py` → 0 failures
- [ ] Overleaf compiles with zero errors, no `??` in the PDF
- [ ] Page count between 6 and 18 — levers in helper §9.4 if over
- [ ] No run identifier left in the body except `M2D` (a real published model name)
- [ ] Terminology in §4.2 above used consistently across all three blocks

---

## 7. Rough effort estimate

| Member | Prose | Tables | Figures | Equations | Est. |
|---|---:|---:|---:|---:|---|
| A | 3,050 | 2 | 1 | 9 | ~10 h |
| B | 2,120 | 6 | 0 | 0 | ~10 h |
| C | 2,715 | 10 | 3 | 0 | ~11 h |
| B, Phase 2 | ~950 | — | — | — | ~3 h |
| C, Phase 3 | — | — | — | — | ~3 h |

A has the most words but the easiest ones. B has the fewest but the hardest argument. C has
the most tables but they are mostly caption work. It comes out roughly even.

---

## 8. If only two people end up doing it

Merge B and C's blocks and give A's Section II to whoever has more time — A's Section I and
the abstract stay with the same person, because framing and novelty must be one voice. Do not
split B's block; the audit is the one part of this paper that cannot survive two authors.
