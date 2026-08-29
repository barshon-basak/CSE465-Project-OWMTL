# Synthetic / fabricated data — remediation checklist

**Created:** 2026-08-29 · **Last updated:** 2026-08-29
**Scope:** every `.py` and `.ipynb` in the repo except `Archive_Files (v1–v3)` (docs only).
**Rule this enforces:** `Model_Training_Protocol.md` §1.2 — *real data only*. No `torch.randn`/`np.random`
placeholder Dataset, no hand-typed metric, no silently substituted sample.

**Read with:** `Asif's/audit/PROJECT_AUDIT.md` (protocol/validity audit — different failure modes),
`Novelty Experiment/NOVELTY_STATUS.md` §"Synthetic-data audit".

---

## Scoreboard

| # | Item | Severity | Effort | Status |
|---|---|---|---|---|
| 1 | M28 master merge — hardcoded benchmark table | 🔴 fabricated | no GPU | ✅ **done** |
| 2 | M15 v6 — `pid % 3` diagnosis-label fallback | 🔴 fabricated (dormant) | no GPU | ✅ **done** |
| 3 | Silent data substitution on load failure (44 sites, 4 shapes) | 🟡 silent substitution | no GPU | ✅ **done** |
| 5 | Stale docs still calling fixed models synthetic | 🟡 misleading | no GPU | ✅ **done** |
| 4 | M16 / M18 / M20 / M21 — no split, unprovable provenance | 🟡 unverifiable | **needs GPU re-run** | 🔧 **code fixed — awaiting your re-run** |

**What is left for you:** re-run M16, M18, M20, M21 on Kaggle/Colab and commit the notebooks
**with their outputs**. Everything else is finished. See §"Your re-run checklist" at the bottom.

---

## 1. ✅ M28 master merge — table no longer hand-typed

**Files changed:** `Barshon's/M28/M28_Master_Experiment_Merge.ipynb`, `m28-nb.ipynb`
**Regenerated:** `results_M28.json`, `master_results_summary.json`, `latex_table_1.tex`, Figures 1–4

### What was wrong

Section 2 correctly globbed every `results_M*.json` into `df_master`. Section 3 then discarded it
and hardcoded a list commented *"Explicit Hardcoded Gold Benchmark Data derived from verified runs"*.
Sections 4, 5 and 6 hardcoded their own arrays on top of that — Figure 2's `scores = [...]`,
Figure 3's `auroc_vals`, Figure 4's `model_sizes`/`params`, the LaTeX literal, and
`novelty_highlights`. Two distinct defects were mixed together:

**(a) Stale numbers pulled from the retired split.** M2's 0.7227, M3's 0.6984, M13's 0.7180 and
M30's 0.8213 are the **macro** `icbhi_score` from runs on the superseded
`patient_independent_60_40_patient_id_fallback` split (11 test patients / 492 cycles). The macro form
is not the ICBHI 2017 metric and runs 0.06–0.22 high; M2's official score on that same file is 0.6138.

**(b) Numbers no run produced at all.**

| Model | M28 claimed | Reality |
|---|---|---|
| M15 | 0.7210 / AUROC 0.5820 / AUPR 0.4910 | v6 `icbhi_score` **0.0000**; v4 AUROC 0.5782, AUPR **0.2202** |
| M17 | 0.7250 / AUROC 0.6120 / AUPR 0.5230 | M17 reports **no** ICBHI score; its real AUROC is **0.8561**, AUPR **0.8250** |
| M14 | 0.7250 / 0.6120 / 0.5230 | **byte-identical to M17's invented row**; M14 v2's real AUROC is 0.4809 |
| M16, M18, M20, M21 | 0.7080 / 0.7080 / 0.7250 / 0.7050 | none of these compute an ICBHI score at all |

And the two exports disagreed with each other for the same model: `latex_table_1.tex` gave M18 as
0.6950 / 1.66 M / 6.64 MB while `results_M28.json` gave 0.7080 / 1.49 M / 1.42 MB, against
`results_M18.json`'s actual 1.49 M / 5.7 MB.

### What was done

- Deleted the `benchmark_summary` literal and every hardcoded array in Sections 4–6.
  **No metric literal exists anywhere in the notebook now** — verified by reading it.
- Section 2 rebuilt: keeps `icbhi_official` and `icbhi_macro` in **separate columns** so the two can
  never merge again, and records `split_method`, `n_test`, `source_file` and `confusion_total` per row.
  Discovery deduplicates by `os.path.realpath` (overlapping search roots were triplicating files).
- Section 3 rebuilt: an explicit `CANONICAL` dict picks the file when a model has several runs, each
  with its reason; anything unlisted resolves to the newest `date_completed`, and an unresolved tie
  **raises rather than guessing**. An `EXCLUDED` dict withholds M16/M18/M20/M21 (item 4) and M22
  (retired split, superseded by M22_v2), each with a stated reason carried into the export.
- Missing metrics render as `--`. Runs reporting no headline metric at all (M7, M11, M39) are listed
  separately instead of appearing as rows of dashes.
- **The table now prints the split as a column** and ends with a loud "11 DIFFERENT SPLITS present —
  scores are NOT comparable across them" block. That confound is what made the old table look good.
- Figure 2 replaced: the invented "novelty ablation" is now a real **macro-vs-official metric
  inflation** chart computed from the JSONs.

### Verified

Executed Sections 1–6 locally. 46 result files → 34 model ids → 25 traceable rows. Every cell in
`results_M28.json` and `latex_table_1.tex` now carries the `source_file` it came from.

> ⚠️ **One decision left for you.** `CANONICAL['M2']` is set to `m2_v4` and `CANONICAL['M3']` to
> `29 aug run` (newest official-split runs), marked `# REVIEW` in the notebook. M2 has three
> official-split runs — `17aug_run_v2` (macro 0.6167), `29aug_run_v3` (0.5780), `m2_v4` (0.5480) —
> and the last two share the date 2026-08-28. Confirm which is the accepted one before the table is
> cited.

---

## 2. ✅ M15 v6 — fabricated-label fallback now raises

**File changed:** `Barshon's/M15/v6/notebookea9ebe614a.ipynb`

`load_diagnosis_map` ended in a fallback that invented labels — `if pid % 3 == 0: return 'URTI'`,
a coin flip wearing a label's name — for all 126 patients whenever the real diagnosis file was not
found. It **did not fire** in the committed run (`DEBUG: diag_map has 126 entries`), so v6's numbers
are genuine; it was a dormant landmine that would have fired silently on any machine with a
different path layout.

Replaced with `raise FileNotFoundError(...)` naming every path searched. The only remaining mention
of `pid % 3` in the file is the comment recording what was removed.

> Still open (separate, tracked in `PROJECT_AUDIT.md`): v6's committed `confusion_matrix_raw` sums to
> 0, so `icbhi_score = 0.0000` cannot be recomputed. Re-export with the matrix before citing it.

---

## 3. ✅ Silent data substitution on load failure — 44 sites, 4 shapes

The original audit found one shape. Fixing it surfaced three more; all four are the same bug —
**a fallback that substitutes data is the same defect as a synthetic Dataset.**

| Shape | Sites | What it did |
|---|---|---|
| `except: return np.zeros(...)` | 38 | failed wav → all-zero log-mel, trained on and scored as a real cycle |
| `if len(audio) == 0: return np.zeros(...)` | 39 (same files) | empty decode → same, reached the other way |
| `except: audio = np.zeros(...)` (assignment) | 5 | M16, M18, M20, M21, M6 — same, by assignment rather than return |
| `except: w = np.zeros(16000)` | 1 | **`Novelty Experiment/N1_fm_concept_probing.py`** — inside the LoRA fine-tuning `Dataset`, so a failed cycle became a 1-second silence that was fine-tuned on and written into `embeddings/ast_lora.npy`, which backs N1's headline result |

All now `raise RuntimeError(f"failed to load audio: {wav_path} [{start}, {end}]") from e`. A run that
cannot read its audio dies instead of quietly diluting the metrics.

> **Note for the Novelty folder:** `NOVELTY_STATUS.md` §"Synthetic-data audit" (2026-08-29) concluded
> "one substitution exists" and cleared everything else. It missed N1's. Worth adding to that file —
> the audit checked for synthetic *Datasets* and did not check for substitution *on error*.

**Not changed, deliberately:** `owmtl_concept_engine/build_notebooks_0304.py` sets `HAVE_FEATS = False`
and prints a note alongside its zeros — the degradation is declared, not silent.

**Verify (prints nothing when clean):**

```bash
python - <<'PY'
import json, glob, re
pats = {"except->return": re.compile(r"except[^\n:]*:\s*\n?\s*return (np|torch)\.zeros"),
        "except->assign": re.compile(r"except[^\n:]*:\s*\n\s*\w+\s*=\s*(np|torch)\.zeros"),
        "empty->return":  re.compile(r"if len\(\w+\) ?== ?0:\s*\n?\s*return (np|torch)\.zeros")}
for p in glob.glob("**/*.ipynb", recursive=True) + glob.glob("**/*.py", recursive=True):
    if ".git" in p or "build_notebooks_0304" in p: continue
    src = ("\n".join("".join(c.get("source", [])) for c in json.load(open(p, encoding="utf-8")).get("cells", [])
                     if c.get("cell_type") == "code") if p.endswith(".ipynb")
           else open(p, encoding="utf-8", errors="replace").read())
    for name, pat in pats.items():
        if pat.search(src): print(f"REMAINS [{name}] {p}")
PY
```

---

## 5. ✅ Stale docs corrected

`Asif's/audit/README.md` and `Asif's/CLAUDE.md` still carried the headline *"every model downstream of
M12 except M6 runs on `torch.randn` data"* and listed M11, M13, M15, M17, M19, M24 as
🔴 Synthetic. That has not been true for weeks — all six were re-run on real audio.

Re-ran `Asif's/audit/audit_project.py` (52 files: 19 CRITICAL / 86 WARNING / 43 INFO) — **zero
`synthetic_data_not_real_dataset` findings**. Both docs updated with the real cycle counts as
evidence, the ✅/🟡/🔴 tables corrected (🔴 is now empty), and the "Traps" entry in `CLAUDE.md`
rewritten to record all four shapes the bug took rather than just the Dataset one.

---

## 4. 🔧 M16 / M18 / M20 / M21 — code fixed, re-run needed

These four had no committed cell outputs, `results_M*.json` dated `2026-08-05`, and
`"data_source": "real_audio"` typed by hand into the exporter — so nothing proved which code produced
the numbers. Reviewing them turned up worse than a provenance gap.

### What was actually wrong

| | Defect |
|---|---|
| **all four** | `discover_icbhi_*_files` indexed **one row per `.wav`** — 851 whole recordings (920 for M21), tiled to 8 s, one label each. A 6898-cycle corpus reported as 851 rows. |
| **M16, M18, M20** | **No train/test split anywhere in the notebook.** M16 distilled over all 851 rows and reported accuracy on those same rows. M20 fitted the temperature and measured ECE on the same 851. The 0.9318 is a training-set score where COPD dominates the recordings. |
| **M21** | `break` after the **first** annotation line per recording, then labelled the whole 8 s clip with cycle 1's label. Reported accuracy 0.575 was last-epoch *training* accuracy. |
| **M20** | `else: all_logits = torch.randn(100, 3)` — if no audio loaded, it calibrated **100 random logit vectors** and exported the temperature and ECE as a result. A genuine synthetic result path. |
| **M18** | `load_base_model` swallowed every checkpoint error with `except Exception: pass`, so the whole sweep could run over a **randomly initialised network**. The committed sweep is consistent with exactly that: **0.0411 accuracy at pruning 0.0**, far below the 0.25 four-class chance line, next to a "best" of 0.9318 reached only once pruning collapsed the model onto the majority class. |
| **M16, M20** | `load_state_dict(..., strict=False)` inside a `try/except` that only printed a note — same failure, quieter. |
| **all four** | Label space never pinned: 3 `known_classes` declared, 4-output head instantiated, M17 checkpoint loaded loosely. |

### What was done

A shared, verified data block now replaces Section 3 in all four:

- **Corrected patient-independent official split**, asserted against the published counts
  (920 recordings / 539 train / 381 test / 126 patients / overlap `{156, 218}` → 551/369 after
  reassigning the straddling patients to train, per protocol §1). Raises if
  `ICBHI_challenge_train_test.txt` is absent — no invented split.
- **Cycle-level index**: every annotated cycle with its own `start`/`end`.
- Diagnosis map must have all 126 patients or the notebook raises.
- `assert not (set(df_train.patient_id) & set(df_test.patient_id))` — leakage cannot pass silently.
- Checkpoint loaders **raise** on missing files, on a head-width mismatch against
  `CFG['disease_classes']`, and on any missing parameter tensor.
- Fit on TRAIN, score on TEST, every epoch. M20 fits the temperature on a patient-disjoint 25%
  carve-out of TRAIN and reports only on TEST.
- **Selection by macro-F1, not accuracy** (M16, M18) — COPD dominance means accuracy rewards exactly
  the majority-class collapse these runs are prone to. `majority_class_rate` and
  `beats_majority_class` ship in every export.
- §4 schema export with `confusion_matrix_raw`, full provenance, and a `supersedes_note` naming the
  number being replaced. The loose `M*_metrics.json` is **deleted** rather than rewritten — two
  exports per model with different numbers is how the M28 table drifted.

### Verified

Split logic checked against the real committed `ICBHI_challenge_train_test.txt`: all asserts pass
(539/381 → 551/369, 79/47 patients, 0 leakage).

All four notebooks then executed **end-to-end on CPU** against a synthetic mini-ICBHI corpus
(920 real stems, 7360 cycles, short noise wavs — the `CLAUDE.md` "test before shipping" convention;
a wiring test, never a result). **All four PASS.** The new guards fire as intended:

- M18 emitted `WARNING: accuracy moves only 0.0000 across the whole sweep` and reported accuracy
  exactly equal to the majority-class rate — the collapse the old notebook hid.
- M20 reported TEST accuracy 0.1826 against a majority rate of 0.4523, i.e. visibly worse than
  guessing — previously invisible.
- M16 reported teacher macro-F1 0.0383, correctly identifying a near-random teacher.

### Correction — the guard fired, and it found the real bug

The first re-run of M16 stopped at the head-width guard:

> `ValueError: teacher head has 256 outputs but CFG["disease_classes"] declares 4`

That 256 is M17's `proto_embed_dim`, and it settles what `strict=False` had been hiding for months:
**M17 v2 is a prototypical network, and its checkpoint contains only
`proto_head_stage2.state_dict()` — the projection MLP (`embedding_dim → 512 → 256`).** There is no
backbone in that file and no classifier head anywhere; M17's four logits are negative squared
distances to four class prototypes computed from the support set at inference.

So M16/M18/M20 were not "mostly loading" a teacher. They were matching **zero** tensors and
operating on a fully randomly-initialised network — which is exactly what M18's committed sweep
showed (0.0411 accuracy at pruning 0.0, below the 0.25 four-class chance line).

**Fixed in all three.** Section 4 now assembles the real model from its three parts:

```
M2 backbone (frozen)  ->  M17 projection  ->  -||z - prototype||² / T
```

wrapped in a `PrototypicalTeacher` module that exposes ordinary 4-class logits, so distillation
(M16), compression (M18) and calibration (M20) all consume it unchanged. Prototypes are computed on
**TRAIN only**. In M18 they are **recomputed for every pruning ratio** — scoring a pruned encoder
against the dense model's centroids would measure the wrong thing.

Consequences for your re-run:

- **All three now need the M2 checkpoint attached as well as M17's.** `M2_CKPT_PATH` resolution was
  added to M16 and M18 (M20 already had it).
- `CFG` gained `m2_depth: 5`, `m2_base_width: 48`, `proto_embed_dim: 256`, `proto_temperature: 0.1`.
  These must match `results_M17.json → config` or the state dicts will not fit.
- A `.pth` written by torch ≥1.6 *is* a zip archive, so the loader only treats a file as a bundle
  when it actually contains a `.pth` member — otherwise it hands it straight to `torch.load`.
- The M2 checkpoint's own classifier head is dropped: only `get_embedding` is used.

Re-verified end-to-end on the mini-corpus with checkpoints matching the **real** shapes (full
backbone + projection-only head, `encoder.N.block.M.*` keys). All four PASS; the teacher assembles,
prototypes form, and retention is computed against the teacher's own TEST score.

### Your re-run checklist

For each of M16, M18, M20, M21:

1. Attach `ICBHI_challenge_train_test.txt` (repo copy: `Asif's/ICBHI_challenge_train_test.txt`) as a
   Kaggle dataset alongside the audio — the notebooks raise without it.
1b. **M16, M18, M20 also need the M2 checkpoint**, not just M17's — the teacher is
   M2 backbone + M17 projection. Attach both.
2. Check `CFG['disease_classes']` against your checkpoint. It is currently M17 v2's stage-2 space
   `['COPD', 'Healthy', 'URTI', 'Pneumonia']` and is marked `# REVIEW`. If your checkpoint is the
   3-class stage-0 head, shorten the list — Section 4 will raise on a mismatch rather than guess.
3. **If a loader raises, that is the finding, not a bug to work around.** This already happened once
   and produced the correction above. If it happens again, check `m2_base_width` / `m2_depth` /
   `proto_embed_dim` against `results_M17.json` before touching the loader.
4. Run all cells, then **commit the notebook with its outputs** — an outputless notebook proves nothing.
5. Delete the stale loose exports once the new `results_M*.json` exist:
   `Barshon's/M16/M16_metrics.json`, `M18/M18_metrics.json`, `M20/M20_metrics.json`,
   `M21/M21_metrics.json`, and `M19/M19_metrics.json`.
6. Re-run `python "Asif's/audit/audit_project.py"`, then re-run M28 to fold the four back in —
   remove them from `EXCLUDED` in Section 3 once they are audit-clean.

---

## ⚠️ Noted, not changed

`Barshon's/RESNET & EFFICIENTNET(Not Related to Project)/OWMTL (RESNET & EFFICIENTNET).ipynb` falls
back to a **random 80/20 patient split** when the official split file is absent. The folder is marked
not-related-to-project and produces no project result, so it was left alone. Fix it or delete the
folder before submission — a reviewer grepping for `np.random` will find it.

---

## ✅ Verified clean — do not "fix" these

| Usage | Where | Why it is fine |
|---|---|---|
| `random_control` probe arm | `Novelty Experiment/N1_fm_concept_probing.py` | a **negative control** — it shows the real probe is not a high-dimensional artefact |
| `torch.randn(...) * 0.01` | N1 | LoRA A-matrix initialisation. Not data. |
| `torch.randn(1, 1, n_mels, n_frames)` | M1, M2, M3, M22, M31–M37 | batch-1 inference-latency probes (protocol §3) |
| `nn.Parameter(torch.randn(...))` | M33, M33_v2 | positional embedding / CLS token init |
| `+= torch.randn_like(spec) * 0.05` | M23, M24 | SpecAugment noise applied **to real spectrograms** |
| `rng.permutation`, `rng.choice`, `np.random.RandomState` | N5–N8, `common.py`, `owmtl_scores.py`, `compute_significance.py`, `M30_vs_M2_Significance_Test.ipynb` | bootstrap CIs and permutation nulls **on real data** |
| shuffled-concept controls | N6, N7 | deliberate controls |
| `--dry-run`, `--selftest`, `test_nx.py` | N1, N9, tests | wiring checks that write **nothing** to `results/` |
| synthetic mini-corpora | `test_concept_extractors.py`, `test_icbhi.py`, `test_leakage_and_split.py`, `gen_M38.py`, `gen_M39.py` | unit tests with constructed ground truth; produce no reported result |
| `common.load_concepts()` | Novelty Experiment | already **raises** rather than substituting if `concepts_all.npz` is absent |

Confirmed absent repo-wide: SMOTE, mixup, GAN, `make_classification`, `make_blobs`,
`RandomOverSampler`, and any surviving `SyntheticICBHIDataset` / `torch.randn`-returning `__getitem__`.

The draft paper is clean: `DRAFT_PAPER/main.tex` includes only `fig_metric_inflation.png`,
`fig_confusion_best.png`, `fig_curves_best.png` — **none from M28**. The fabricated table never
reached the manuscript.

---

## Run log

| Date | Item | Note |
|---|---|---|
| 2026-08-29 | audit performed | 5 findings; 38 files with the zeros fallback |
| 2026-08-29 | item 1 done | M28 Sections 2–6 rebuilt from `df_master`; artifacts regenerated and verified |
| 2026-08-29 | item 2 done | M15 v6 label fallback → `FileNotFoundError` |
| 2026-08-29 | item 3 done | 44 substitution sites across 4 shapes → `raise`; N1 and M6 were new finds |
| 2026-08-29 | item 5 done | audit re-run (0 synthetic findings); `audit/README.md` + `CLAUDE.md` corrected |
| 2026-08-29 | item 4 correction | head-width guard fired on M16 → found M17's checkpoint is a projection MLP, not a classifier. Real prototypical teacher assembled in M16/M18/M20; re-verified, all 4 pass |
| 2026-08-29 | item 4 code fixed | M16/M18/M20/M21 rewritten; all 4 pass end-to-end on a synthetic mini-corpus. **Awaiting GPU re-run.** |
