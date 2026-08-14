"""
Metrics, results JSON, and the required plots.

Emits exactly the schema in Model_Training_Protocol.md §4 so the M28 merge script
can consume every member's runs without reformatting.

One correction to the protocol, applied here
--------------------------------------------
§3 defines "ICBHI Score = (Macro Sensitivity + Macro Specificity) / 2". That is
not the ICBHI 2017 challenge metric, and a number computed that way cannot be
compared to any published ICBHI result.

The challenge metric is a two-way normal/abnormal split:

    Sp    = correctly classified Normal cycles / all Normal cycles
    Se    = correctly classified abnormal cycles / all abnormal cycles
            (abnormal = Crackle + Wheeze + Both, pooled)
    Score = (Sp + Se) / 2

`compute_metrics` reports both. `icbhi_score` is the official one and is the
primary metric for model selection; `icbhi_score_macro` is the protocol's
definition, kept so nobody's existing runs become unreadable. Flag this to the
group — Barshon's M1 selected its best epoch on the macro variant.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np

NORMAL_CLASS_INDEX = 0  # Normal is index 0 in SOUND_EVENT_CLASSES


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    *,
    normal_index: int = NORMAL_CLASS_INDEX,
) -> Dict:
    """Full §3 metric suite for a closed-set classification run."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Per-class specificity = TN / (TN + FP), one-vs-rest.
    total = cm.sum()
    spec = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Official ICBHI 2017 challenge score.
    abnormal = [i for i in labels if i != normal_index]
    n_normal = cm[normal_index, :].sum()
    n_abnormal = cm[abnormal, :].sum()
    sp_official = cm[normal_index, normal_index] / n_normal if n_normal else 0.0
    se_official = sum(cm[i, i] for i in abnormal) / n_abnormal if n_abnormal else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "f1_macro": float(f1.mean()),
        "specificity_macro": float(spec.mean()),
        "icbhi_score": float((sp_official + se_official) / 2),
        "icbhi_sensitivity": float(se_official),
        "icbhi_specificity": float(sp_official),
        "icbhi_score_macro": float((rec.mean() + spec.mean()) / 2),
        "per_class": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "specificity": float(spec[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix_raw": cm.astype(int).tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 6).tolist(),
    }


def open_set_metrics(
    is_unknown: Sequence[int], unknown_score: Sequence[float], *, threshold: Optional[float] = None
) -> Dict:
    """Unknown-detection metrics for M6/M15/M17 and the separability harness."""
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(is_unknown, dtype=int)
    s = np.asarray(unknown_score, dtype=float)
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "n_known": int((y == 0).sum()),
        "n_unknown": int((y == 1).sum()),
        "threshold": threshold,
    }
    if threshold is not None:
        pred = (s >= threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        out.update(
            unknown_precision=float(p), unknown_recall=float(r), unknown_f1=float(f)
        )
    return out


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def model_size_mb(model) -> float:
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        return round(os.path.getsize(tmp.name) / (1024 * 1024), 3)


def count_params(model) -> Dict[str, int]:
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }


def measure_inference_ms(model, sample_input, *, n_warmup: int = 10, n_runs: int = 50) -> float:
    """Median per-sample latency at batch size 1, in milliseconds.

    `sample_input` must already carry a leading batch dimension of 1 — pass
    `dataset[0][0].unsqueeze(0)`. Inferring it from the rank is not safe here:
    AST consumes rank-2 (frames, mels) samples and CNN backbones consume rank-3
    (channel, mels, frames) ones, so no single rule covers both.
    """
    import time

    import torch

    model.eval()
    device = next(model.parameters()).device
    x = sample_input.to(device)
    if x.shape[0] != 1:
        raise ValueError(
            f"expected a batch dimension of 1, got shape {tuple(x.shape)}; "
            f"call .unsqueeze(0) on the sample first"
        )
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 4)


def environment_block() -> Dict:
    env = {
        "platform": _detect_platform(),
        "gpu_name": None,
        "pytorch_version": None,
        "python_version": platform.python_version(),
    }
    try:
        import torch

        env["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def _detect_platform() -> str:
    if os.path.isdir("/kaggle/input"):
        return "Kaggle"
    try:
        import google.colab  # noqa: F401

        return "Colab"
    except ImportError:
        return "Local"


# ---------------------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------------------


def build_results(
    *,
    model_id: str,
    model_name: str,
    member: str,
    member_name: str,
    config: Dict,
    dataset_info: Dict,
    efficiency: Dict,
    best_epoch: Dict,
    best_metrics: Dict,
    ablation: Dict,
    training_history: List[Dict],
    is_augmented: bool = False,
    augmentation_method: str = "none",
    notes: str = "",
    extra: Optional[Dict] = None,
) -> Dict:
    """Assemble a protocol-§4-compliant results dict."""
    from datetime import date

    payload = {
        "meta": {
            "model_id": model_id,
            "model_name": model_name,
            "member": member,
            "member_name": member_name,
            "date_completed": date.today().isoformat(),
            "is_augmented": is_augmented,
            "augmentation_method": augmentation_method,
            "notes": notes,
        },
        "config": config,
        "environment": environment_block(),
        "dataset_info": dataset_info,
        "efficiency": efficiency,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "ablation": ablation,
        "training_history": training_history,
    }
    if extra:
        payload.update(extra)
    return payload


REQUIRED_METRIC_KEYS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "specificity_macro",
    "icbhi_score",
    "per_class",
    "confusion_matrix_raw",
    "confusion_matrix_normalized",
)

REQUIRED_ABLATION_KEYS = (
    "ablation_group",
    "ablation_role",
    "baseline_model_id",
    "variable_changed",
    "variables_held_constant",
    "component_flags",
    "loss_weights",
)


def validate_results(payload: Dict) -> List[str]:
    """Check a results dict against the §10 'before calling a model done' list."""
    problems: List[str] = []
    for key in REQUIRED_METRIC_KEYS:
        if key not in payload.get("best_metrics", {}):
            problems.append(f"best_metrics missing '{key}'")
    for key in REQUIRED_ABLATION_KEYS:
        if key not in payload.get("ablation", {}):
            problems.append(f"ablation missing '{key}'")
    eff = payload.get("efficiency", {})
    for key in ("total_params", "trainable_params", "model_size_mb", "training_time_total_s"):
        if eff.get(key) in (None, 0):
            problems.append(f"efficiency['{key}'] is missing or zero")
    if not payload.get("training_history"):
        problems.append("training_history is empty — plots cannot be generated")
    split_method = payload.get("dataset_info", {}).get("split_method", "")
    if "patient" not in split_method.lower():
        problems.append(
            f"dataset_info['split_method']={split_method!r} does not look "
            f"patient-independent"
        )
    return problems


def save_results(payload: Dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{payload['meta']['model_id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_TRAIN_COLOR = "#2b6cb0"
_VAL_COLOR = "#dd6b20"


def _curve(ax, history, train_key, val_key, best_epoch, ylabel):
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h.get(train_key) for h in history], color=_TRAIN_COLOR, label="train")
    ax.plot(epochs, [h.get(val_key) for h in history], color=_VAL_COLOR, label="validation")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="grey", linestyle=":", linewidth=1.2,
                   label=f"best (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)


def plot_training_curves(payload: Dict, out_dir: str, *, dpi: int = 150) -> List[str]:
    """Loss / accuracy / macro-F1 curves — the three required by protocol §5."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    history = payload["training_history"]
    best = payload.get("best_epoch", {}).get("epoch")
    mid = payload["meta"]["model_id"]

    specs = [
        ("loss_curve.png", "train_loss", "val_loss", "loss"),
        ("accuracy_curve.png", "train_accuracy", "val_accuracy", "accuracy"),
        ("f1_curve.png", "train_f1_macro", "val_f1_macro", "macro F1"),
    ]
    written = []
    for fname, tk, vk, ylabel in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        _curve(ax, history, tk, vk, best, ylabel)
        ax.set_title(f"{mid} — {ylabel}")
        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written


def plot_confusion_matrix(
    payload: Dict, class_names: Sequence[str], out_dir: str, *, dpi: int = 150
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    raw = np.array(payload["best_metrics"]["confusion_matrix_raw"])
    norm = np.array(payload["best_metrics"]["confusion_matrix_normalized"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, mat, title, fmt in (
        (axes[0], raw, "counts", "d"),
        (axes[1], norm, "row-normalised", ".2f"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() else 1)
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        thresh = mat.max() / 2 if mat.max() else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{payload['meta']['model_id']} — confusion matrix")
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def finalise_run(payload: Dict, class_names: Sequence[str], out_dir: str) -> Dict:
    """Write the JSON, generate every required plot, and report what's missing."""
    problems = validate_results(payload)
    json_path = save_results(payload, out_dir)
    plots = plot_training_curves(payload, out_dir)
    plots.append(plot_confusion_matrix(payload, class_names, out_dir))
    return {"results_json": json_path, "plots": plots, "problems": problems}
