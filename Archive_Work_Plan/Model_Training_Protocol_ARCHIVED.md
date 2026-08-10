# Model Training Protocol — Group 5 OWMTL Project

> **What this is:** A short, practical guide so all members' training results are compatible and mergeable. Follow the essentials below; everything else is up to you.
>
> **What this is NOT:** A rigid step-by-step script. You have freedom in how you structure your code, which libraries you use, and how you organize your workflow — as long as the outputs match.

---

## 1. The Essentials (Must-Do)

These are the only hard requirements. Everything else in this doc is guidance.

1. **Patient-independent splits.** No patient's cycles in both train and test. This is a research validity requirement, not a style choice.
2. **Save checkpoints every epoch** so Kaggle/Colab disconnects don't lose progress (see §11 for PyTorch 2.6+ checkpoint rules and Kaggle persistence protocol).
3. **Produce a structured results JSON** per model (schema in §4) — this is what makes the final merge work.
4. **Compute all metrics in §3** — accuracy, precision, recall, F1, confusion matrix, model size, params, training time.
5. **Generate the required plots** (§5) — loss curves, accuracy curves, confusion matrix.
6. **Use the shared preprocessing parameters** for audio (sample rate, mel bins, etc.) so everyone's spectrograms are comparable.

That's it. The rest of this document explains *how* to do these six things.

---

## 2. Shared Preprocessing Parameters

These audio parameters should stay consistent across members so spectrograms are comparable. If you have a strong reason to deviate for a specific model, document it in your results JSON.

```
Sample rate  : 16000 Hz
Duration     : 8.0 seconds
Mel bins     : 128
FFT size     : 1024
Hop length   : 160  (10 ms)
Window length: 400  (25 ms)
Freq range   : 50–2000 Hz
```

**Random seed:** Use `42` as the default for base experiments. If you need different seeds (e.g., ensemble members), just document what you used.

---

## 3. Required Metrics

Every model run must report these. Compute them on the **test/validation set** using the best checkpoint.

### Classification Metrics

| Metric | Scope |
|---|---|
| Accuracy | Overall |
| Precision | Macro + per-class |
| Recall (Sensitivity) | Macro + per-class |
| F1 Score | Macro + per-class |
| Confusion Matrix | Raw counts + normalized (row-wise) |

For sound-event models (M1–M4, M12, M21–M23), also compute:
- **Specificity** (macro + per-class)
- **ICBHI Score** = (Macro Sensitivity + Macro Specificity) / 2

For open-set models (M6, M15, M17), also compute:
- Unknown-detection precision, recall, AUROC, AUPR

### Efficiency Metrics

| Metric | How |
|---|---|
| Total parameters | `sum(p.numel() for p in model.parameters())` |
| Trainable parameters | `sum(p.numel() for p in model.parameters() if p.requires_grad)` |
| Model size (MB) | Save `state_dict()` to temp file, check file size |
| Training time per epoch | `time.time()` around each epoch |
| Total training time | Sum of epoch times |
| GPU type | `torch.cuda.get_device_name(0)` |

**Optional but recommended:** inference time (ms/sample) — useful for the paper's efficiency discussion.

### Quick reference for model size:

```python
import tempfile, os, torch

def get_model_size_mb(model):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        return round(os.path.getsize(tmp.name) / (1024 * 1024), 2)
```

---

## 4. Results JSON Format

Each completed model produces **one JSON file**: `results_<MODEL_ID>.json` (e.g., `results_M1.json`, `results_M23_aug.json`).

This is the format the merge script expects at M28. Stick to this structure so we don't have to reformat later.

```json
{
  "meta": {
    "model_id": "M1",
    "model_name": "Provisional CNN Backbone",
    "member": "A",
    "member_name": "Barshon",
    "date_completed": "2026-07-23",
    "is_augmented": false,
    "augmentation_method": "none",
    "notes": ""
  },

  "config": {
    "sample_rate": 16000,
    "n_mels": 128,
    "batch_size": 32,
    "num_epochs": 60,
    "lr": 0.001,
    "optimizer": "Adam",
    "scheduler": "StepLR",
    "architecture": "2D_CNN_4Block",
    "seed": 42
  },

  "environment": {
    "platform": "Kaggle",
    "gpu_name": "Tesla T4",
    "pytorch_version": "2.10.0",
    "python_version": "3.12.13"
  },

  "dataset_info": {
    "dataset": "ICBHI_2017",
    "train_samples": 6406,
    "test_samples": 492,
    "split_method": "patient_independent_60_40"
  },

  "efficiency": {
    "total_params": 421732,
    "trainable_params": 421732,
    "model_size_mb": 1.61,
    "training_time_total_s": 15660,
    "training_time_per_epoch_s_avg": 261,
    "gpu_name": "Tesla T4",
    "inference_time_ms_per_sample": null
  },

  "best_epoch": {
    "epoch": 47,
    "primary_metric": "icbhi_score",
    "primary_metric_value": 0.7144
  },

  "best_metrics": {
    "accuracy": 0.5200,
    "precision_macro": 0.4300,
    "recall_macro": 0.4800,
    "f1_macro": 0.4100,
    "specificity_macro": 0.8500,
    "icbhi_score": 0.6650,
    "per_class": {
      "Normal":  {"precision": 0.52, "recall": 0.90, "f1": 0.66, "support": 255},
      "Crackle": {"precision": 0.38, "recall": 0.23, "f1": 0.28, "support": 164},
      "Wheeze":  {"precision": 0.15, "recall": 0.21, "f1": 0.18, "support": 38},
      "Both":    {"precision": 0.29, "recall": 0.49, "f1": 0.36, "support": 35}
    },
    "confusion_matrix_raw": [[230, 15, 5, 5], [100, 38, 10, 16], [20, 5, 8, 5], [10, 5, 3, 17]],
    "confusion_matrix_normalized": []
  },

  "ablation": {
    "ablation_group": "backbone_architecture",
    "ablation_role": "variant",
    "baseline_model_id": null,
    "variable_changed": "backbone: 2D_CNN_4Block",
    "variables_held_constant": [
      "loss_function",
      "optimizer",
      "data_split",
      "augmentation: none",
      "seed: 42"
    ],
    "component_flags": {
      "has_sound_event_head": true,
      "has_disease_head": false,
      "has_cross_task_consistency": false,
      "has_cqkd_regularization": false,
      "has_openmax_rejection": false,
      "owl_stage": 0,
      "compression_clusters": null
    },
    "loss_weights": {
      "sound_event_weight": 1.0,
      "disease_weight": null,
      "consistency_weight": null
    }
  },

  "training_history": [
    {
      "epoch": 1,
      "train_loss": 1.23,
      "val_loss": 1.45,
      "train_accuracy": 0.45,
      "val_accuracy": 0.42,
      "train_f1_macro": 0.32,
      "val_f1_macro": 0.30,
      "lr": 0.001,
      "epoch_time_s": 261
    }
  ]
}
```

### What matters in this schema:
- **`meta`** — identifies who trained what. Keep `model_id` consistent with the Model Training Reference (M1, M2, ..., M28).
- **`best_metrics`** — the numbers that go in the paper. Must include all metrics from §3.
- **`training_history`** — one entry per epoch with at least loss, accuracy, and F1 for both train and val. This is what the plots are generated from.
- **`efficiency`** — params, model size, training time. Reviewers ask for these. `inference_time_ms_per_sample` is now included (set to `null` if not measured, but **strongly recommended** — it feeds directly into the efficiency columns of the ablation table).
- **`ablation`** — **NEW: required for every model.** This block self-documents each run's role in the ablation study so the final table can be assembled programmatically at M28 instead of manually. See §4.1 for the full explanation.

You can add extra fields if your model needs them (e.g., `auroc` for open-set models, `compression_ratio` for M18). Just don't remove or rename the fields above.

---

## 4.1 Ablation Metadata — How to Fill the `ablation` Block

Your supervisor wants an ablation study table in the final paper. An ablation table answers: *"What happens to performance when we add/remove/swap one component, holding everything else constant?"* The `ablation` block in the results JSON exists to make this table trivially assembable at M28 — **if you fill it in during training**, you won't have to reconstruct it from memory later.

### Fields explained:

| Field | What to put | Example |
|---|---|---|
| `ablation_group` | Which ablation table row-group does this model belong to? Use one of the canonical group names below. | `"backbone_architecture"` |
| `ablation_role` | Is this the full/baseline model (`"baseline"`) or a variant with something removed/changed (`"variant"`)? | `"variant"` |
| `baseline_model_id` | The model ID of the baseline this run is compared against. `null` if this *is* the baseline. | `"M4"` |
| `variable_changed` | Plain-English description of the one thing that differs from the baseline. | `"backbone: 2D_CNN_4Block"` |
| `variables_held_constant` | List of things deliberately kept the same so the comparison is fair. | `["loss_function", "optimizer", "seed: 42"]` |
| `component_flags` | Binary flags for which architectural components are active in this run. | See schema above |
| `loss_weights` | The actual loss-weighting values used. `null` if the head doesn't exist in this run. | `{"sound_event_weight": 1.0, ...}` |

### Canonical `ablation_group` names (use these exact strings):

| Group name | Proposal §8 ablation | Models involved |
|---|---|---|
| `backbone_architecture` | Backbone choice (CNN vs. transformer) | M2, M3, M4, M12 |
| `loss_weighting` | Loss weighting between heads | M13, M15 (varied weight runs) |
| `cross_task_consistency` | With/without cross-task consistency | M13 (without) vs. M15 (with) |
| `rejection_method` | Cross-task consistency vs. OpenMax/Weibull | M6 vs. M15 |
| `cqkd_regularization` | CQKD-regularized vs. unregularized | M17 (without) vs. M17+CQKD variant |
| `owl_stage_count` | OWL stage count (1 vs. 2 vs. 3) | M15 (Stage 1) vs. M17 (Stage 2) |
| `compression_level` | Compression sweep | M18 variants |
| `augmentation_effect` | Augmented vs. clean | M2→M21, M3→M22, M4→M23, M15→M24, M18→M25, M17→M26 |
| `ood_generalization` | Full model vs. ablated on Coswara/SPRSound | M19, variants |
| `uncertainty_method` | Ensemble vs. MC-Dropout vs. SNGP vs. Evidential | M7, M8, M9, M10 |
| `calibration_method` | Temperature vs. vector vs. focal | M11 variants |

### How `component_flags` works:

This is the key to building the ablation table automatically. Each flag answers a yes/no question about what's active in this particular run:

- `has_sound_event_head` — Is the 4-class sound-event head present?
- `has_disease_head` — Is the disease-diagnosis head present?
- `has_cross_task_consistency` — Is the cross-task disagreement mechanism active?
- `has_cqkd_regularization` — Is cluster-quantized distillation regularization applied?
- `has_openmax_rejection` — Is the OpenMax/Weibull rejection mechanism active (alternative to cross-task consistency)?
- `owl_stage` — Which OWL stage is this run evaluated at? (0, 1, or 2)
- `compression_clusters` — Number of clusters in CQKD compression. `null` if uncompressed.

These flags let the merge script at M28 auto-generate rows like:

| Variant | Sound Head | Disease Head | Cross-Task | CQKD | F1 | AUROC | Params | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| Full model | ✓ | ✓ | ✓ | ✓ | 0.xx | 0.xx | xxx | x.x |
| − Cross-task | ✓ | ✓ | ✗ | ✓ | 0.xx | 0.xx | xxx | x.x |
| − Disease head | ✓ | ✗ | ✗ | ✓ | 0.xx | — | xxx | x.x |

### `loss_weights` — why this matters for ablation:

The proposal lists "loss weighting between heads" as a full ablation. If you're training a multi-head model (M13, M15, M17), record the actual weight values you used:
- `sound_event_weight` — weight on the sound-event classification loss
- `disease_weight` — weight on the disease-diagnosis loss
- `consistency_weight` — weight on the cross-task consistency loss term (M15+)

If you run the same model with different weight ratios (e.g., 1:1:0.5 vs. 1:1:1.0), each run gets its own results JSON with the weights recorded. The ablation table then shows the effect of rebalancing.

### Quick example — filling this for M4 (AST backbone, clean):

```json
"ablation": {
  "ablation_group": "backbone_architecture",
  "ablation_role": "variant",
  "baseline_model_id": "M12",
  "variable_changed": "backbone: AST_pretrained",
  "variables_held_constant": [
    "loss_function: CrossEntropyLoss",
    "optimizer: Adam",
    "data_split: patient_independent_60_40",
    "augmentation: none",
    "seed: 42",
    "preprocessing: 128mel_16kHz_8s"
  ],
  "component_flags": {
    "has_sound_event_head": true,
    "has_disease_head": false,
    "has_cross_task_consistency": false,
    "has_cqkd_regularization": false,
    "has_openmax_rejection": false,
    "owl_stage": 0,
    "compression_clusters": null
  },
  "loss_weights": {
    "sound_event_weight": 1.0,
    "disease_weight": null,
    "consistency_weight": null
  }
}
```

### Quick example — filling this for M15 (cross-task consistency, OWL Stage 1):

```json
"ablation": {
  "ablation_group": "cross_task_consistency",
  "ablation_role": "baseline",
  "baseline_model_id": null,
  "variable_changed": "cross_task_consistency: enabled",
  "variables_held_constant": [
    "backbone: M12_final",
    "disease_head: M13_architecture",
    "data_split: patient_independent_LOPO",
    "augmentation: none",
    "seed: 42"
  ],
  "component_flags": {
    "has_sound_event_head": true,
    "has_disease_head": true,
    "has_cross_task_consistency": true,
    "has_cqkd_regularization": false,
    "has_openmax_rejection": false,
    "owl_stage": 1,
    "compression_clusters": null
  },
  "loss_weights": {
    "sound_event_weight": 1.0,
    "disease_weight": 1.0,
    "consistency_weight": 0.5
  }
}
```

> **Tip:** If a model participates in multiple ablation groups (e.g., M15 is the baseline for both `cross_task_consistency` and `rejection_method`), pick the *primary* group. The merge script at M28 can cross-reference by `model_id`. If you want to be thorough, you can add an optional `"secondary_ablation_groups": ["rejection_method"]` list.

---

## 5. Required Plots

Generate these for every model and save them alongside your results JSON.

| Plot | What it shows |
|---|---|
| **Loss curves** | Train loss + val loss vs. epoch (same axes) |
| **Accuracy curves** | Train accuracy + val accuracy vs. epoch |
| **F1 curves** | Train macro-F1 + val macro-F1 vs. epoch |
| **Confusion matrix** | Heatmap — at least the normalized version; raw counts version is a plus |

### Styling suggestions (not mandatory, but helps the paper look unified):
- 150 DPI or higher
- Mark the best epoch on the curves (dotted vertical line or similar)
- Use a consistent color for train (e.g., blue) and validation (e.g., orange/red)
- Include axis labels and a legend

---

## 6. Checkpoints & Disconnect Safety

Kaggle gives you ~9–12 hours; Colab free gives ~12 hours (less with GPU). Disconnects happen. Protect your training.

### The minimum you need:

1. **Save a checkpoint after every epoch** 

2. **Auto-resume at the start of the training loop:**

3. **Save `best_model.pth` separately** whenever the primary metric improves — this is the checkpoint you evaluate on.

4. **Clean up old per-epoch checkpoints** (keep the last 2–3) to avoid filling disk.

### Platform-specific tips:

**Kaggle:**
- `/kaggle/working/` is writable but wiped after session ends
- Use **Save Version** (File → Save Version → Save & Run All) to persist outputs
- Download checkpoints from the output panel periodically

**Colab:**
- Mount Google Drive and copy checkpoints there after each epoch:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # After saving checkpoint:
  !cp /content/checkpoints/best_model.pth /content/drive/MyDrive/OWMTL/M1/
  ```

---

## 7. Data Augmentation

Augmentation runs are **separate experiments** from the clean baseline (see M21–M27 in the Model Training Reference). The general approach:

- **Train with augmentation, validate/test without** — augmentation is training-time only.
- **Use the same seed** as the clean run so the only variable is the augmentation.
- **In your results JSON**, set `is_augmented: true` and note the method in `augmentation_method`.

### Augmentation by member:

| Member | Models | Augmentation type |
|---|---|---|
| A | M21–M23 | SpecAugment (time + frequency masking on spectrograms) |
| B | M24 | Class-balancing (noise injection, pitch shift, time stretch for minority classes) |
| C | M25–M26 | Domain-robustness (simulated channel/device noise) |
| D | M27 | Re-run calibration on A's best augmented backbone |

Implementation details are up to each member.

---

## 8. Hyperparameter Tuning

Not every model needs a full HP search. Use your judgment, but kkep this thing/task in consideration.

---

## 9. File Organization

Suggested (not mandatory) structure — the only hard requirement is that the results JSON and plots exist somewhere findable:

```
<MemberName>/
├── M1/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── latest.pth
│   ├── results/
│   │   ├── results_M1.json
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── f1_curve.png
│   │   └── confusion_matrix.png
│   └── notebook_M1.ipynb
├── M2/
│   └── ...
```

---

## 10. Before Calling a Model "Done"

Quick sanity check:

- [ ] Results JSON exists with all §3 metrics filled in
- [ ] Best model checkpoint exists
- [ ] Loss, accuracy, and F1 curve plots generated
- [ ] Confusion matrix plot generated
- [ ] Model size (MB) and parameter count recorded
- [ ] Training time recorded
- [ ] Patient-independent split was used
- [ ] For augmented runs: `is_augmented` and `augmentation_method` filled in
- [ ] **Ablation block filled in** — `ablation_group`, `ablation_role`, `baseline_model_id`, `variable_changed`, `component_flags`, and `loss_weights` are all populated (§4.1)
- [ ] **Inference time measured** (recommended) — `inference_time_ms_per_sample` in the `efficiency` block
- [ ] **PyTorch 2.6+ pickling safety verified** — explicit scalar casting (`int()`, `float()`) when saving and `weights_only=False` when loading (§11)
- [ ] **Kaggle persistence & eval_only fallback implemented** (§11)
- [ ] **Handoff download cell added at end of notebook** — clickable `FileLink`s or `.zip` bundle for `best_model.pth` and `results_M<ID>.json` (§11.D)

---

## 11. Checkpoint Safety, Disconnect Persistence & Kaggle Protocol

To prevent disconnects from losing progress, avoid PyTorch 2.6+ unpickling crashes, and allow committing notebooks via **"Save Version"** without re-training models from scratch, all members must follow this protocol in their training code:

### A. PyTorch 2.6+ Checkpoint Serialization Rules
PyTorch 2.6 strictly enforces `weights_only=True` by default when loading `.pth` files. To ensure checkpoints can be cleanly saved and loaded across sessions without unpickling errors:
1. **Cast scalars to native Python types when saving:** When constructing the state dictionary in `save_checkpoint`, explicitly cast NumPy floats or integer metrics to native Python types (`int()` and `float()`). Never save raw `np.float64`, `np.float32`, or sklearn metric objects directly into the checkpoint dictionary.
   ```python
   # Correct:
   state = {
       "epoch": int(epoch),
       "best_score": float(best_score),
       "model_state": model.state_dict(),
       ...
   }
   ```
2. **Use `weights_only=False` when loading trusted checkpoints:** Since team members generate their own checkpoint files, use `weights_only=False` in `torch.load` during auto-resume and evaluation to prevent PyTorch 2.6+ from rejecting custom metadata:
   ```python
   state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
   ```

### B. Kaggle Persistence Protocol (How to Never Lose Progress)
Interactive browser sessions on Kaggle time out after ~20–30 minutes of inactivity or network drops, which wipes the temporary `/kaggle/working/` directory.
1. **For Long Training Runs, Avoid Interactive Mode:**
   - Instead of running a multi-hour training loop interactively in your browser, click **Save Version** (top right) -> **Save & Run All (Commit)** -> **Save**.
   - This executes your notebook in an unattended background container (up to 12 hours) that never disconnects due to Wi-Fi drops, browser closing, or computer sleeping.
   - When finished, all checkpoints (`best_model.pth`, `epoch_*.pth`) and results JSONs are saved permanently in that version's **Output Files**.
2. **Auto-Resume Fallback (`/kaggle/input/` Discovery):**
   - Configure your checkpoint loader (`load_checkpoint`) so that if `/kaggle/working/checkpoints/` is empty, it automatically searches `/kaggle/input/` for existing checkpoint files from attached datasets or previous versions.
   - If found, copy them into `/kaggle/working/checkpoints/` and resume from the saved epoch.

### C. "Save Version" Without Re-Training Protocol
When you click **"Save & Run All (Commit)"**, Kaggle boots a clean, empty machine. To commit a final notebook version or generate reports without waiting hours for the model to re-train from scratch:
1. **Method 1 (Attach Previous Version Output):**
   - Open the right-hand sidebar in Kaggle -> **Add Data** -> **Your Work** -> **Output Files** -> click **Add** next to your previously trained notebook version.
   - When you click **Save & Run All**, your checkpoint loader will automatically discover the completed checkpoint (e.g., Epoch 60) in `/kaggle/input/`, see that `start_epoch = 61`, bypass the training loop in 1 second, and directly execute the evaluation cells.
2. **Method 2 (`eval_only` Mode):**
   - Include an `"eval_only": False` switch in your global configuration (`CFG`).
   - In your training loop, check `if CFG.get("eval_only", False):`. If `True`, skip the training loop (`start_epoch = CFG["num_epochs"] + 1`) and jump straight to final evaluation on whatever checkpoint is loaded.

### D. Cross-Member Handoff via Manual File Download / Upload
> [!NOTE]
> **Use this manual download/upload method whenever needed only.** If Kaggle's internal dataset linking across accounts works smoothly for your team, you may use direct linking instead. Use manual download/upload whenever direct sharing is inconvenient or across different platforms.

When different team members work across separate notebooks (e.g., Member A building M1, Member B building M6), sharing outputs via Kaggle's internal dataset linking can sometimes be inconvenient across accounts. Team members can seamlessly hand off models using **manual downloads**:
1. **Add a Handoff Download Cell at the End of Every Notebook:**
   At the very end of your notebook, include a dedicated code cell using `IPython.display.FileLink` to generate clickable download links for the exact output files needed by other members:
   - `best_model.pth`: Required by downstream models (e.g., M5/M6/M7 needing M1's backbone).
   - `results_M<ID>.json`: Required for final table reconciliation in M28.
   - Training curves / plots (`.png`).

   **Copy-paste this exact code block as your final notebook cell:**
   ```python
   # ============================================================
   # FINAL CELL — TEAM HANDOFF & ONE-CLICK FILE DOWNLOADS
   # ============================================================
   import os
   import shutil
   import glob
   from IPython.display import display, FileLink

   print("=" * 60)
   print("OFFICIAL PROTOCOL OUTPUTS READY FOR DOWNLOAD")
   print("=" * 60)

   # 1. Grab exactly the files specified in §9 of Model_Training_Protocol.md
   protocol_files = sorted(
       glob.glob(os.path.join(CFG["ckpt_dir"], "best_model.pth")) +
       glob.glob(os.path.join(CFG["results_dir"], "results_M*.json")) +
       glob.glob(os.path.join(CFG["results_dir"], "*.png"))
   )

   for fpath in protocol_files:
       if os.path.exists(fpath):
           size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
           print(f"Ready: {os.path.basename(fpath):<25} ({size_mb} MB)")
           display(FileLink(fpath))
       else:
           print(f"Missing: {os.path.basename(fpath)}")

   # 2. Create a clean ZIP archive of only the official protocol files
   bundle_dir = "/kaggle/working/protocol_bundle"
   if protocol_files:
       os.makedirs(bundle_dir, exist_ok=True)
       for fpath in protocol_files:
           if os.path.exists(fpath):
               shutil.copy2(fpath, os.path.join(bundle_dir, os.path.basename(fpath)))
       
       zip_path = shutil.make_archive("/kaggle/working/model_handoff_bundle", 'zip', bundle_dir)
       size_zip = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
       print(f"\nOr download all official files in a single ZIP bundle ({size_zip} MB):")
       display(FileLink("model_handoff_bundle.zip"))
   print("=" * 60)
   ```
2. **How to Download:**
   - When the notebook run finishes, run the final download cell. Click the generated links in your browser to download the individual files (or the combined `.zip` bundle) directly to your local computer.
3. **How to Import in the Next Notebook (e.g., M5, M6, M15):**
   - **On Kaggle:** In the destination notebook, click **Add Data** -> **New Dataset** -> upload the downloaded `best_model.pth` or results JSON (naming it something like `owmtl-m1-output`). The files will instantly be available at `/kaggle/input/owmtl-m1-output/`, ready for `torch.load()` or `json.load()`!
   - **On Colab / Local:** Place the downloaded files into your local checkpoint/results directory or mount Google Drive.

---

## 12. AI Model Notebook Generation Prompt (Instructions for LLMs)

> **SYSTEM INSTRUCTION FOR AI ASSISTANTS (e.g., Gemini, ChatGPT, Claude, GitHub Copilot):**
> When a user attaches or references this document (`Model_Training_Protocol.md`) and asks you to generate a Jupyter Notebook (`.ipynb`) for a specific model run (e.g., "Generate the notebook for M4" or "Create notebook for M15"), you must act as an expert AI medical imaging and audio signal processing engineer working on our research project: **"Cluster-Aware Open-World Multi-Task Learning for Respiratory Sound and Disease Diagnosis" (OWMTL)**, targeting Q1 journal publication (*Biomedical Signal Processing and Control*).
>
> Do not ask the user for a separate template or to fill in placeholders. Instead, autonomously retrieve the model specifications from the workspace (such as `Model_Training_Reference.md` or the user's prompt) and generate a complete, professional, clean, and fully executable Jupyter Notebook (`.ipynb`) in Python/PyTorch that strictly conforms to all rules in §§1–11 of this protocol and the instructions below.

---

### Step 1: Retrieve Target Model Specifications
Before generating code, identify the assigned model run (e.g., M1, M4, M15, M18, M23) and check its exact specifications from `Model_Training_Reference.md` or the user's prompt:
* **Model ID & Name:** Identify the assigned model ID and descriptive name.
* **Assigned Member Role:** Member A (Backbone/Augmentation), Member B (Disease/OWL), Member C (Compression/OOD), or Member D (Ensemble/Calibration).
* **Project Stage / Phase:** Identify the current research phase (e.g., Stage 1, Stage 2...).
* **Prerequisite / Dependent Checkpoints:** Check if training starts from scratch or loads a prior checkpoint (e.g., M1 checkpoint, M13 teacher, M12 winning backbone).
* **Input Representation & Preprocessing:** Strictly default to **§2** (Log-mel spectrograms, 128 bins, 8.0s @ 16kHz) unless explicitly overridden by the model specification.
* **Assigned Augmentation:** Check if clean baseline or augmented (§7: SpecAugment, class-balancing noise/pitch shift, or domain-robustness noise).
* **Architecture & Methodology Details:** Apply the exact architectural instructions (e.g., 4-block 2D CNN with BatchNorm/Dropout; AST fine-tuned from AudioSet; dual-head setup with cross-task consistency disagreement score).
* **Specific Outputs & Custom Metrics Required:** Automatically include the standard §3 metrics suite and §4.1 ablation block, plus any specialized requirements (unknown-detection AUROC/AUPR, inference latency ms/sample, CQKD compression ratio, forgetting curves).

---

### Step 2: Adhere to Project Scientific Context & Background
To ensure your implementation aligns with our core research methodology, keep the following foundational principles in mind:
1. **The Core Research Gap (Why we do this):** 
   Nearly all prior respiratory audio multi-task learning (MTL) operates under a *closed-world assumption*—assuming every disease seen at test time was present during training. This is clinically dangerous. Our project introduces an **Open-World Multi-Task Learning (OWMTL)** framework that uses **cross-task consistency disagreement** between a sound-event classification head and a disease-diagnosis head to flag unknown/unseen conditions.
2. **Dataset & Task Formulation**
3. **Efficiency & Deployability:** Track parameter counts, model size in MB, training time per epoch, and inference latency (ms/sample) to support our edge-deployment and clinical reliability claims.

---

### Step 3: Strict Adherence to Protocol Rules (§§1–11)
**IMPORTANT:** You must strictly abide by all rules, coding conventions, schema specifications, and evaluation criteria laid out in §§1–11 of this document. Do not invent custom reporting formats or violate shared constraints. Specifically, ensure that:
1. **Patient-Independent Data Splits (§1):** Never leak cycles from the same patient across train, validation, or test splits. Use a strict patient-independent split or Leave-One-Patient-Out (LOPO) as dictated by sample size.
2. **Shared Audio Preprocessing (§2):** Strictly use `sample_rate=16000`, `duration=8.0s`, `n_mels=128`, `n_fft=1024`, `hop_length=160` (10ms), `win_length=400` (25ms), `f_min=50`, `f_max=2000`, and default `seed=42` unless the model specification explicitly overrides them.
3. **Checkpoints & Disconnect Protection (§6 & §11):** Save model weights after *every single epoch* to disk/temp storage so progress is never lost during Kaggle/Colab session timeouts. Always track and save the best checkpoint based on validation primary metric (`icbhi_score` or validation loss). Strictly follow §11.A for PyTorch 2.6+ pickling safety (casting scalars to `int()`/`float()`) and §11.B/C for Kaggle persistence.
4. **Required Metric Suite (§3):** Automatically compute and print Overall Accuracy, Macro/Per-class Precision, Recall, F1, Macro Specificity, ICBHI Score `((Sensitivity + Specificity) / 2)`, and raw + normalized Confusion Matrices. Compute all efficiency metrics (Total/Trainable parameters, Model Size in MB via temp file, Epoch duration, GPU name, and Inference time ms/sample).
5. **Standardized Results JSON (§4 & §4.1):** The notebook must culminate in exporting a single, perfectly formatted JSON file named `results_<MODEL_ID>.json` matching the exact schema in §4 of the protocol (including `meta`, `config`, `environment`, `dataset_info`, `efficiency`, `best_epoch`, `best_metrics`, `ablation`, and `training_history`). Ensure the `ablation` block (§4.1) is fully populated.
6. **Required Visualization (§5):** Generate and display clean inline plots using matplotlib/seaborn: (a) Train/Val Loss vs. Epoch, (b) Train/Val Accuracy/Score vs. Epoch, and (c) Annotated Confusion Matrix (Raw and Normalized).
7. **Team Handoff Download Cell (§11.D):** Include the exact one-click team handoff file download cell at the very end of the notebook.

---

### Step 4: Follow Required Notebook Structure & Organization
Generate the `.ipynb` file cleanly organized into consecutive markdown and code cells following this exact 8-part structure:
* **### Title & Meta-Information:** Descriptive markdown header with Model ID, Name, Author/Member, and brief objective.
* **### Section 1: Environment Setup & Dependencies:** Device selection (`cuda` vs `cpu`), seed setting (`seed=42`), importing PyTorch, torchaudio, librosa, scikit-learn, seaborn, and helper libraries. Include Kaggle/Colab path autdetection if helpful.
* **### Section 2: Configuration & Hyperparameters:** A centralized dictionary or dataclass containing all audio parameters, learning rate, batch size, epoch count, paths, and model flags.
* **### Section 3: Dataset Loading & Patient-Independent Splitting:** Clean dataset class and splitting logic that guarantees zero patient leakage between train and test sets.
* **### Section 4: Model Architecture Definition:** Modular, well-documented PyTorch `nn.Module` classes for the backbone and task heads, including any assigned augmentation layers or custom loss functions.
* **### Section 5: Training & Validation Loop:** Robust training loop with `tqdm` progress bars, per-epoch metric logging, automatic best-model saving, checkpoint persistence to disk, and PyTorch 2.6+ scalar casting.
* **### Section 6: Comprehensive Evaluation & Visualization:** Loading the best checkpoint with `weights_only=False` and generating the full §3 metric suite and §5 plots (Loss curves, Accuracy curves, Confusion matrices).
* **### Section 7: Exporting Protocol-Compliant Results JSON & Handoff Cell:** Building the exact JSON data structure required by §4 and saving `results_<MODEL_ID>.json` to the working directory. Finally, include the exact one-click team handoff file download cell required by §11.D at the very end of the notebook.
* **### Section 8: Summary & Key Takeaways:** A concise markdown summary of final performance, efficiency numbers, and observations ready for project reporting.

---

### Step 5: Execute Coding Best Practices
* Write clean, idiomatic PyTorch code with clear variable names and inline docstrings.
* Handle potential division-by-zero or edge cases gracefully in custom metric calculations.
* Ensure the code is self-contained and runnable without manual intervention once dataset paths are set.
* Do not omit or truncate code blocks—provide the complete, functional implementation for every single cell.

---

## Quick Reference

```
SHARED AUDIO PARAMS:
  Sample rate: 16000 Hz | Duration: 8s | Mel bins: 128
  Hop: 160 | Window: 400 | Freq: 50–2000 Hz

DEFAULT SEED: 42

REQUIRED OUTPUTS PER MODEL:
  ✓ results_M<ID>.json     (metrics + config + efficiency + ablation)
  ✓ best_model.pth         (best checkpoint)
  ✓ loss_curve.png
  ✓ accuracy_curve.png
  ✓ f1_curve.png
  ✓ confusion_matrix.png

REQUIRED METRICS:
  Accuracy, Precision, Recall, F1 (macro + per-class)
  Confusion Matrix, Params, Model Size, Training Time

ABLATION BLOCK (in results JSON):
  ✓ ablation_group          (which table does this row go in?)
  ✓ ablation_role            (baseline or variant?)
  ✓ baseline_model_id        (compared against which model?)
  ✓ variable_changed         (what's different?)
  ✓ component_flags          (which heads/mechanisms are active?)
  ✓ loss_weights             (actual loss weights used)
  ○ inference_time_ms        (recommended for efficiency table)

AI NOTEBOOK GENERATION:
  See §12 for direct instructions for LLM assistants to generate protocol-compliant .ipynb files.
```

---
