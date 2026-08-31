# Figure specification — `Final_Draft/main.tex`

**Written:** 2026-08-31 · **Purpose:** everything needed to generate each figure later, without
re-deriving anything. Every figure below is either **already generated** (file present in
`figures/`, generator script named) or **to be generated** (data source, axes and exact numbers
given, so it can be drawn from the committed JSON without touching a GPU).

**Hard rule, same as the paper's:** a figure is regenerated *from the committed
`results_M*.json`*, never redrawn from a number typed into a slide. Two of the existing generators
already enforce this, which is why no figure in this paper can drift from its table.

---

## Status at a glance

| ID | `\label` | State | File | Generator |
|---|---|---|---|---|
| F1 | `fig:flowchart` | ✅ done | `figures/fig_flowchart.pdf` | `Final_Draft/make_diagrams.py flowchart` |
| F2 | `fig:inflation` | ✅ done | `figures/fig_metric_inflation.png` | `Asif's/audit/make_inflation_figure.py` |
| F3 | `fig:cm` | ✅ done | `figures/fig_confusion_best.png` | `Asif's/M22_v2/make_paper_figures.py` |
| F4 | `fig:curves` | ✅ done | `figures/fig_curves_best.png` | `Asif's/M22_v2/make_paper_figures.py` |
| F5 | `fig:xai` | ✅ done | `figures/fig_xai_panels.png` | `Asif's/M44/m44_xai_best_model.py` |
| F6 | *(optional)* | 🟡 asset exists | `figures/M48_selection_curves.png` | `M48_core_pipeline_ablation/m48_gpu_rows.py` |
| F7 | *(optional)* | 🟡 asset exists | `figures/auroc_forest_plot.png` | `Asif's/Statistics/plot_forest.py` |
| F8 | `fig:motivation` | ✅ done | `figures/fig_motivation.pdf` | `Final_Draft/make_diagrams.py motivation` |
| F9 | `fig:modules` | ✅ done | `figures/fig_modules.pdf` | `Final_Draft/make_diagrams.py modules` |
| F10 | `fig:architecture` | ✅ done | `figures/fig_architecture.pdf` | `Final_Draft/make_diagrams.py architecture` |
| F11 | `fig:metricanatomy` | ✅ done | `figures/fig_metric_anatomy.pdf` | `Final_Draft/make_diagrams.py metric` |

No figure in the paper is a placeholder any more. F6 and F7 are copied into `figures/` and are
ready to insert if page budget allows; the paper compiles without them.

**F1 and F8–F11 all come from one script**, `Final_Draft/make_diagrams.py` (drawing primitives in
`Final_Draft/diagram_lib.py`). It needs no GPU and no dataset — only `matplotlib` and `numpy` —
and it re-reads every quantity it prints from a committed results file:

| loaded | what it supplies |
|---|---|
| `Asif's/M22_v2/Results/results_M22_v2.json` | the best model's raw confusion matrix, official score, CI, patient/cycle counts, params, and the whole preprocessing config (sample rate, mel bins, FFT sizes, dropout) |
| `Asif's/audit/ICBHI_SCORE_AUDIT.json` | the fallback-partition and macro-variant scores of F8 |
| `M48_core_pipeline_ablation/M48_tier_A_table.json` | the select-on-loss score of F8 |
| `M40_M43_transformers/Results/results_M4{0,1,2}.json` | transformer parameter counts and scores in F10 |

The only literals in the script are the published scores of `tab:comparison` (other people's
numbers, cited in the caption) and shape/wording constants. **Nothing inside these figures may be
edited by hand in a vector editor** — edit the script and re-run, or the figure will drift from
its table, which is the fault this paper is about.

```bash
python Final_Draft/make_diagrams.py            # all five, PDF + PNG
python Final_Draft/make_diagrams.py flowchart  # one at a time
```

Both a `.pdf` (used by `main.tex`) and a `.png` (for previewing) are written for each.

> ⚠ **Do not put a figure number or a table number inside one of these images.** Cross-references
> renumber when a float is added. The images say "Section III", "Eq. 6" and "the attribution
> figure" for exactly this reason. Equation numbers *are* used, and they are stable: (1) tiling,
> (2) log-mel, (3) label, (4) class weights, (5) SpecAugment, (6) official, (7) macro, (8) gate.

---

## F1 — System flowchart

**Placement:** Section II, immediately after the opening paragraph, at
`\includegraphics[width=0.92\linewidth]{figures/fig_flowchart.pdf}`.

**Caption is in the .tex** — do not restate it inside the figure.

**Content, top to bottom** — this is what the script draws. Two blocks are visually heavier
because the caption calls them the contribution: the **split loader** and the **official scoring
module**.

```
[ ICBHI 2017 corpus ]              [ ICBHI_challenge_train_test.txt ]
  920 recordings / 126 patients            official split file
  cycle annotation .txt files                    |
            |                                    v
            |                   ╔══════════════════════════════════════╗
            |                   ║  AUDITED SPLIT LOADER   (heavy box)  ║
            |                   ║  • raise if file absent (no fallback)║
            |                   ║  • detect patients on both sides     ║
            |                   ║  • reassign 156, 218 -> train        ║
            |                   ║  • device-suffix-safe filename join  ║
            |                   ╚══════════════════════════════════════╝
            |                          |                       |
            v                          v                       v
   [ cycle segmentation ]      TRAIN 4,251 cycles       TEST 2,636 cycles
   start/end from .txt         79 patients              47 patients
            |                          |                       |
            v                          |                       |
  ┌──────────────────────────┐         |                       |
  │ PREPROCESSING            │<--------+-----------------------+
  │ 16 kHz mono              │
  │ cyclic tile to 8 s       │   (three stages OFF by default, each
  │ log-mel 128 x 801        │    an ADD row in the ablation:
  │   n_fft 1024, hop 160    │    band-pass P1, denoise P2, amp-norm P3)
  │   win 400, 50-2000 Hz    │
  │ per-spectrogram min-max  │
  │ 1 -> 3 channel + ImageNet│
  └──────────────────────────┘
            |                    \
            | train only          \  test only (no augmentation)
            v                      \
  [ SpecAugment: 2 freq <=24,       \
    2 time <=80 masks ]              \
            |                         |
            v                         |
  ┌──────────────────────────┐        |
  │ BACKBONE (one of four)   │        |
  │ MobileNetV2 / ViT-B/16   │        |
  │ Swin-T / DeiT-S          │        |
  │ + GAP + dropout 0.3      │        |
  │ + Linear(->4)            │        |
  └──────────────────────────┘        |
            |                         |
   inverse-frequency                  |
   class-weighted CE                  |
            |                         v
            +---------------> [ 4-class prediction per cycle ]
                                       |
                                       v
                    ╔══════════════════════════════════════════╗
                    ║  OFFICIAL SCORING MODULE   (heavy box)   ║
                    ║  • raw 4x4 confusion matrix (committed)  ║
                    ║  • ICBHI = (Se + Sp)/2, pooled abnormal  ║
                    ║  • macro variant kept, clearly labelled  ║
                    ║  • patient-level bootstrap CI, B = 1000  ║
                    ║  • paired McNemar / bootstrap            ║
                    ╚══════════════════════════════════════════╝
                                       |
                                       v
                    [ results_M*.json  ->  every table in Sec. III ]
```

**Style notes.** Greyscale-safe (the report may be printed): the four fills used across all five
diagrams are separated in luminance as well as hue. The two heavy boxes carry the same words the
caption uses ("audited split loader", "official scoring module") so the reader can match them.
The as-drawn figure is 6.7 × 6.6 in native, which lands at about 150 mm tall at
`0.92\linewidth` — taller than the 85 mm originally sketched here, because the eight stages do
not fit legibly at 7 pt in 85 mm. If the page budget forces a shorter figure, cut the *evaluation
lane* (the dashed bypass) rather than shrinking the type.

**Note on the partition counts.** The figure prints **patients and cycles only**, deliberately.
`results_M22_v2.json` records `train_recordings = 551` while `tab:splits` states 550 for the
corrected partition — the one-recording difference is the `226_1b1_Pl_sc_*` device-suffix join
described in that table's own footnote (counted in the split file, dropped by a stem join).
Printing a recording count in the figure would have forced a choice between the two without
resolving it. Patients (79 / 47) and cycles (4,251 / 2,636) agree everywhere and are what the
figure shows. **This discrepancy is still open and should be resolved in `tab:splits`.**

---

## F2 — Metric inflation scatter ✅ *(generated)*

**File:** `figures/fig_metric_inflation.png` · **Generator:** `Asif's/audit/make_inflation_figure.py`
· **Data:** `Asif's/audit/ICBHI_SCORE_AUDIT.json`

**What it shows:** reported macro score (x) against recomputed official score (y) for the 20
verifiable runs of Table `tab:metricaudit`, with the $y=x$ diagonal. Every point sits above the
diagonal; the vertical displacement is the fault, and its *variability* is the message.

**Regenerate with:** `python "Asif's/audit/make_inflation_figure.py"`
(then copy the PNG into `Final_Draft/figures/`).

**If it is redrawn from scratch,** the 20 (reported, official) pairs are exactly the two numeric
columns of `tab:metricaudit` in `main.tex`. Mark M33 (0.5506 → 0.3330) and M32 (0.6547 → 0.4733)
as the two extremes; they are the rows the discussion names.

---

## F3 — Normalised confusion matrix, best model ✅ *(generated)*

**File:** `figures/fig_confusion_best.png` · **Generator:** `Asif's/M22_v2/make_paper_figures.py`
· **Data:** `Asif's/M22_v2/Results/results_M22_v2.json` → `best_metrics.confusion_matrix_raw`

**Exact matrix** (rows = true Normal / Crackle / Wheeze / Both, columns = predicted):

```
[[1110,  333,   81,   36],
 [ 269,  320,    8,   20],
 [ 178,   39,  110,   46],
 [  32,   19,   25,   10]]
```

Row-normalised: `[[.7115 .2135 .0519 .0231], [.4360 .5186 .0130 .0324],
[.4772 .1046 .2949 .1233], [.3721 .2209 .2907 .1163]]`.

Annotate every cell, ≥150 DPI, sequential colormap, class order Normal/Crackle/Wheeze/Both. **Also
commit the raw-count version** beside it; the paper's own protocol requires the counts to be
recoverable from the figure.

---

## F4 — Training and validation curves, best model ✅ *(generated)*

**File:** `figures/fig_curves_best.png` · **Generator:** `Asif's/M22_v2/make_paper_figures.py`
· **Data:** `results_M22_v2.json` → `training_history`, a 38-element list with keys
`epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_f1_macro, val_f1_macro,
val_icbhi_score_official, lr, epoch_time_s`.

**Two panels side by side.** Left: `train_loss` and `val_loss` vs `epoch`. Right: `train_accuracy`
and `val_accuracy` vs `epoch`. Mark the **selected epoch 38** with a vertical line on both.

**Anchor values to sanity-check the plot after regeneration:**

| epoch | train_loss | val_loss | train_acc | val_acc | val ICBHI |
|---|---|---|---|---|---|
| 1  | 1.2685 | 1.4002 | 0.4451 | 0.4518 | 0.4606 |
| 38 | 0.1707 | 2.8055 | 0.9323 | 0.5880 | 0.5602 |

The divergence (train loss 1.27→0.17 while val loss 1.40→2.81) is the point of the figure and is
discussed in the text — do not smooth it away.

> ⚠ **Label the right-hand curve honestly.** In this run the "validation" series *is* the test
> partition (see `subsec:limitations`). If the axis label says "validation", add "(= test
> partition; see Limitations)" to the caption or the legend. The current caption in `main.tex`
> already says the paper reports this; keep the figure consistent with it.

---

## F5 — Interpretability panels ✅ *(generated)*

**File:** `figures/fig_xai_panels.png` (identical content also at `Asif's/M44/M44_xai_panels.png`)
· **Generator:** `Asif's/M44/m44_xai_best_model.py`
· **Data:** `Asif's/M44/results_M44.json`, checkpoint `best_model.pth` at epoch 38.

**Layout:** one row per class (Normal, Crackle, Wheeze, Both) plus **one misclassified example**.
Four columns per row: log-mel input · Grad-CAM overlay · occlusion-sensitivity map · predicted
vs true label.

**Do not crop out the misclassification row** — graders and reviewers both look for it, and the
paper's discussion depends on it.

**Resolution caveat that must appear in the caption** (already written into `main.tex`): MobileNetV2
downsamples 32×, so Grad-CAM is natively 4 frequency × 26 time cells (~500 Hz × 0.31 s per cell)
bilinearly upsampled. The occlusion maps use a 16-mel × 80-frame patch at stride 8×40 and are
better resolved; where the two disagree, prefer occlusion.

**File size warning:** the current PNG is 3.0 MB. Downsample to ≤600 DPI before uploading to
Overleaf or the compile will be slow.

---

## F6 — Selection-criterion curves 🟡 *(asset exists, not yet inserted)*

**File already copied:** `figures/M48_selection_curves.png`
· **Generator:** `M48_core_pipeline_ablation/m48_gpu_rows.py` (writes it as a side effect of the
Tier A run) · **Data:** `results_M48_A0_s42.json`, `_s1`, `_s2`, `_A24` → `training_history`.

**What it shows:** for each of the four Tier A rows, the official ICBHI score per epoch and the
validation loss per epoch on twin axes, with the epoch chosen by each criterion marked. It is the
picture behind Table `tab:selection` and makes the 15–19-epoch disagreement visible.

**To insert:** add after Table `tab:selection` in `subsec:selection`:

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.92\linewidth]{figures/M48_selection_curves.png}
\caption{Official ICBHI score and monitoring loss per epoch for the three seeds of
Table~\ref{tab:selection}. The epoch that minimises the loss and the epoch that
maximises the challenge score are 15 to 19 apart in every run, and the gap is worth
0.075 to 0.102 of official score.}
\label{fig:selection}
\end{figure}
```

Then reference it in the paragraph after the table. **Recommended** — it is the paper's
second-largest effect and currently has no picture.

---

## F7 — Open-set AUROC forest plot 🟡 *(asset exists, optional)*

**File already copied:** `figures/auroc_forest_plot.png`
· **Generator:** `Asif's/Statistics/plot_forest.py`
· **Data:** `Asif's/Statistics/significance_results.json`

**What it shows:** every open-set detector's AUROC with its 95% interval and a vertical line at
0.5. Every patient-level interval crosses the line — which is the whole argument of
`subsec:openset` in one image.

**Caveat to keep in the caption if it is used:** intervals are Hanley–McNeil (unpaired), because
the legacy runs did not commit per-patient raw scores. The unpaired form is conservative (wider),
so a *significant* result here would survive a paired test; a *non-significant* one might not.
M6's interval is the only one that excludes chance, and it does so in the **wrong** direction.

**Insert only if page budget allows.** The table already carries the argument.

---

## F8 — Motivation: the measurement spread ✅ *(generated)*

**File:** `figures/fig_motivation.pdf` · **Placement:** Section I, at the end of *This work*,
before the section roadmap · **Label:** `fig:motivation`

**What it shows.** Two panels on one shared x-axis (official ICBHI score). *Top:* the seven
published rows of `tab:comparison` that are on the official partition under the official metric,
as grey dots, with the literature's range shaded. *Bottom:* one MobileNetV2 — same corpus, same
seed — scored four ways, with the delta between consecutive rows annotated:

| row | value | source |
|---|---|---|
| checkpoint chosen on minimum loss | 0.4788 | `M48_tier_A_table.json → A7_selection_criterion.A0_s42.by_loss` |
| corrected protocol (what we report) | 0.5602 | `ICBHI_SCORE_AUDIT.json → M22_v2.official` |
| + identifier-fallback partition | 0.6495 | `ICBHI_SCORE_AUDIT.json → M22.official` |
| + macro-averaged metric | 0.7077 | `ICBHI_SCORE_AUDIT.json → M22.reported` |

The two upper steps are `tab:protocol` rows `E2` and `E1`+`E2`; the lower step is
`tab:selection`, seed 42. **The four values are not additive with the leave-one-out deltas and
must not be presented as if they were** — the figure draws only measured pairs, so the annotated
steps are exactly the differences between adjacent measured values (+0.0814, +0.0893, +0.0582).
The dashed vertical at 0.7077 crosses both panels because the whole point is that it lands above
every published row.

---

## F9 — The two contributed modules ✅ *(generated)*

**File:** `figures/fig_modules.pdf` · **Placement:** Section II, immediately after
Fig.~`fig:flowchart` · **Label:** `fig:modules`

**(a) Split loader**, as a decision flow: file present? → *no* aborts with `FileNotFoundError`
(never the identifier fallback); parse 920 recording assignments; device-suffix-safe join;
patient-set intersection empty? → *no* reassigns 156 and 218 to train; `assert` disjoint before
the training loop may start; TRAIN 79 patients / 4,251 cycles, TEST 47 / 2,636.

**(b) Scoring module**, as a fan-out: the raw 4 × 4 matrix is written first, then branches into
the official score (Se 0.4089, Sp 0.7115, ICBHI 0.5602), the macro variant under its own name
(0.6140), the patient-level bootstrap ([0.508, 0.614]) and the paired tests. A return edge marks
that the official score is also the checkpoint-selection criterion — that edge is the only place
in the paper where Fault 5 is drawn as a *component* rather than described.

All eight numbers are recomputed in the script from `confusion_matrix_raw`, so this figure and
Tables `tab:main`, `tab:perclass` and `tab:metricaudit` cannot disagree.

---

## F10 — Model architecture ✅ *(generated)*

**File:** `figures/fig_architecture.pdf` · **Placement:** Section II-C, after the classifier-head
paragraph · **Label:** `fig:architecture`

Three bands. **(a)** the shared front end left to right with the tensor shape under each stage:
cropped cycle → cyclic tiling (128,000 samples) → log-mel (1 × 128 × 801) → SpecAugment →
channel adapt (3 × 128 × 801) → backbone. **(b)** the four backbones as a small table — family,
parameter count, feature map entering the GAP, official score — with MobileNetV2 shaded, plus a
side panel expanding the MobileNetV2 path and marking the 4 × 26 Grad-CAM tap. **(c)** the shared
head (GAP → dropout 0.3 → Linear→4), the class-weighted objective, and the output.

Parameter counts and scores come from the four runs' own results records; the stage settings come
from `results_M22_v2.json → config`, so changing a preprocessing constant in the pipeline and
re-running the script updates the figure.

---

## F11 — Anatomy of the two metrics ✅ *(generated)*

**File:** `figures/fig_metric_anatomy.pdf` · **Placement:** Section III-A, immediately before
Fig.~`fig:inflation` · **Label:** `fig:metricanatomy`

**(a)** the best model's committed 4 × 4 matrix, row-normalised for colour but annotated with raw
counts, with the Normal row boxed (Sp = 1110/1560 = 0.7115) and the three abnormal rows boxed
together (Se = (320+110+10)/1076 = 0.4089). **(b)** per-class recall against per-class
specificity as paired bars, with support printed beside each class, and a callout on the Both
class: 10 of 86 cycles found, specificity reads 0.960.

This is the *mechanism* behind `tab:metricaudit`, which reports the effect. F2 shows the effect
across twenty runs; F11 shows why it happens on one. Both are computed from the same matrix.

---

## Figures deliberately NOT produced

| Not produced | Why |
|---|---|
| Grad-CAM "pointing game" hit-rate plot | Not implementable on ICBHI. The corpus annotates *whether* a cycle contains an adventitious sound, not *where* inside it, and the pipeline crops to exactly that window — so the annotated window **is** the model input and the hit rate is 100% by construction. Plotting it would plot an artefact of the crop. |
| Cumulative-ladder line plot | Three of six rungs (`S0`,`S1`,`S2`) have not been run. A line through three points with a gap would imply a trend the data does not contain. Re-draw once `M48_kaggle_tier_A.ipynb` completes those rows. |
| ROC curves for the 4-class task | The challenge metric is not threshold-swept; a per-class ROC would invite a comparison against published ICBHI scores that is not valid. |
| t-SNE / UMAP of the embedding | Decorative here. `figures/umap_ood.png` exists in the repo root from an earlier direction and is **not** on the corrected partition — do not reuse it. |

---

## Reproduction commands

```bash
# F1, F8, F9, F10, F11  block diagrams   (no GPU, no dataset; matplotlib + numpy)
python Final_Draft/make_diagrams.py
# writes fig_flowchart, fig_motivation, fig_modules, fig_architecture,
# fig_metric_anatomy -- each as .pdf and .png -- straight into Final_Draft/figures/
```

```bash
# F2  metric-inflation scatter        (no GPU, no dataset)
python "Asif's/audit/make_inflation_figure.py"

# F3 + F4  confusion matrix + curves  (no GPU, no dataset — reads the JSON only)
python "Asif's/M22_v2/make_paper_figures.py"

# F5  XAI panels                      (needs the checkpoint + ICBHI audio)
python "Asif's/M44/m44_xai_best_model.py"

# F7  forest plot                     (no GPU, no dataset)
python "Asif's/Statistics/plot_forest.py"
```

After running any of these, copy the output PNG into `Final_Draft/figures/` under the filename the
table above names — `main.tex` refers to the copies in `Final_Draft/figures/`, not to the
originals.
