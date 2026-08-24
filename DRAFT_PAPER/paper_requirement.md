# Paper requirements — CSE465 Final Report

Source: `4X5_Final_Report_261 (2).pptx` (16 slides, dated 24/04/2026), transcribed verbatim
where the wording matters. This is the rulebook for `main.tex`.

**Due:** May 12, Tuesday, 11:59 PM (slide 3).
**Deliverable:** the Overleaf link, submitted in Canvas **with edit access** — not a PDF.

Experiment-side status tracking lives in [`../RTK_requirements.md`](../RTK_requirements.md);
this file only covers what the *paper* must contain and look like.

---

## 1. Format — all mandatory

| Rule | Detail |
|---|---|
| Class | `\documentclass[journal,onecolumn]{IEEEtran}` — IEEE journal/conference **single column** |
| LaTeX | MUST. Overleaf MUST. |
| Length | **Minimum 6 pages** |
| Tables & figures | Cited automatically with `\label{}` / `\ref{}` — never hardcode "Table 2" |
| Equations | Every equation inside `\begin{equation}...\end{equation}` |
| References | `.bib` file + `\bibliographystyle{IEEEtran}` |
| Compile | **LaTeX Error = not allowed. LaTeX Warning = allowed.** |
| Submission | Overleaf link with **edit access** in Canvas |

---

## 2. Section structure (slide 5, in order)

```
Abstract
Keywords
Introduction
Proposed System
  4.1 Dataset
  4.2 Dataset Preprocessing
  4.3 Applied Models
Results and Discussion
  5.1 Ablation Study
Conclusions
References
```

> **Numbering note:** the deck labels the subsections 4.x and 5.x because its own list counts
> Abstract and Keywords as items 1 and 2. In IEEEtran, Abstract and Keywords are unnumbered, so
> Introduction becomes I, Proposed System II, Results III, Conclusions IV. Keep the *order* and
> the *content* exactly as above. Only add a separate "Related Works" section if the supervisor
> asks — slide 8 explicitly puts the literature review **inside** the Introduction.

---

## 3. Per-section rules

### Abstract (slides 6–7)
- 200–300 words, one concise paragraph.
- Covers, in order: **research purpose/objective → methodology → results → conclusion**.
- Must mention dataset, preprocessing, applied models, results, ablation study, XAI.
- **No citations. No abbreviations. No symbols. No equations.**
- It gets copied verbatim into the proceedings/journal webpage — write it last, write it clean.

### Keywords
- Main topics, **alphabetical order**.

### Introduction (slide 8)
1. **First paragraph:** motivation — why this capstone project was chosen.
2. **Literature review:** each group member reviews **at least one** paper individually — journal
   recommended over conference. Group is 4 → **at least 4 reviewed papers**, cited as `[1]`, `[2]`.
3. **End of the literature review:** state the existing **gaps, limitations and weaknesses** of the
   related works. This is a required paragraph, not optional.
4. **Second-last block:** brief description of YOUR work, 3–4 paragraphs.
5. **Last paragraph:** roadmap describing each section of the paper.

### Proposed System (slides 9–10)
- Dataset, preprocessing steps, applied models, flowcharts, tables and figures.
- **Cite the dataset.**
- Include the equations.
- Include a figure and/or algorithm block for any custom model or algorithm.
- Include a **flowchart of the complete system**.
- **Do not show any results in this section.** Methodology only.
- Dataset table format seen on slide 10 — class counts *before and after* augmentation:

  | Class | Open-source | Collected | Total | Augmented total |
  |---|---|---|---|---|

### Results and Discussion (slides 11–13, 15)
"Most critical section of research paper." Everything numeric goes here.
- Performance metrics: accuracy, precision, recall, F1 (IoU / mAP if applicable) for **all** models.
- Model metrics: **# of parameters, model size, training time per epoch or total** for **all** models.
  Slide 15 format:

  | Model | Epochs | Time (min) | Params | Size (MB) |
  |---|---|---|---|---|

  **State where the training was performed** (which GPU/runtime) — explicitly required.
- Figures, **for the BEST model only**: normalized confusion matrix; training & validation loss
  vs. epochs; training & validation accuracy vs. epochs.
- Augmented-training results, reported the same way as the clean results.
- Novelty results, reported the same way as the clean results.
- **XAI** on the best model — LIME, SHAP, Grad-CAM etc. — with interpretation, not just images.
- **Discuss** the results. Tables alone do not satisfy this section.
- **Comparison table, at the very end of this section** (slide 13). Required for software *and*
  hardware projects. Format from the slide:

  | Author | Dataset | Network | AUC | Accuracy | Specificity | Sensitivity |
  |---|---|---|---|---|---|---|
  | `[4]` | … | … | … | … | … | … |
  | **This work** | … | … | … | … | … | … |

  Prior-work rows are cited by reference number; the final row is always **This work**.

### Ablation Study (slide 14)
One row per metric, comparing the configurations, with a reason column. Slide format:

| Metric | Config A | Config B | Progress | Possible reason |
|---|---|---|---|---|
| Loss / Accuracy / Precision / Recall / F1-score / AUC | … | … | +2.21% | why the delta happened |

Run it on the **best** model. The "possible reason" column is the part that earns marks.

### Conclusions (slide 16)
- Final section; include **two or three future works**.

---

## 4. Experiment checklist (slides 3 and 12, identical lists)

Group size = 4, so "# of group members" = **4**.

- [ ] Apply **ALL** preprocessing techniques to the dataset
- [ ] Apply pre-trained DL models — **one model per member** (4)
- [ ] Apply **at least 4 transformer models** (ViT, Swin, DINO, PVT, CvT, …)
- [ ] Accuracy, precision, recall, F1 for **ALL** applied models
- [ ] Model size, # of parameters, training time per epoch or total for **ALL** applied models
- [ ] Normalized confusion matrix + loss/accuracy vs. epochs curves for the **BEST** model
- [ ] Data augmentation on training samples → report the same metric set again
- [ ] Novelties → report the same metric set again
- [ ] XAI on the **BEST** model
- [ ] Ablation study on the **BEST** model

---

## 5. Pre-submission check

- [ ] Compiles on Overleaf with **zero errors**
- [ ] ≥ 6 pages
- [ ] Every table and figure is referenced in the body via `\ref{}`, and every `\ref{}` resolves
      (no `??` in the PDF)
- [ ] Every equation is in an `equation` environment and numbered
- [ ] Bibliography renders through `IEEEtran` style; every `.bib` entry is actually cited
- [ ] Abstract is 200–300 words, with no citations/abbreviations/symbols/equations
- [ ] Keywords are alphabetical
- [ ] No results anywhere in the Proposed System section
- [ ] Comparison table is the last thing in Results and Discussion
- [ ] Training hardware named
- [ ] Overleaf link shared with **edit access**, submitted in Canvas before May 12, 11:59 PM
