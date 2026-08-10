#!/usr/bin/env python3
"""
OWMTL project audit — protocol compliance + result-validity checks.

Scans every results file in the repo and reports two classes of problem:

  1. COMPLIANCE — does the file match the §4 schema the M28 merge expects?
  2. VALIDITY   — do the numbers actually mean anything, or is the run degenerate?

The validity checks exist because a run can be perfectly schema-compliant and still
be reporting nothing. Every check below was written against a failure that is
actually present in this repo, not a hypothetical one.

Usage:
    python3 "Asif's/audit/audit_project.py"                # writes PROJECT_AUDIT.md
    python3 "Asif's/audit/audit_project.py" --quiet        # summary line only

No dependencies beyond the standard library. No GPU, no dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field, asdict

# ── severity levels ──────────────────────────────────────────────────────────
CRITICAL = "CRITICAL"   # the result does not support the claim being made on it
WARNING = "WARNING"     # needs attention before the paper
INFO = "INFO"           # worth knowing, not blocking

SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}

# ── §4 schema requirements (Model_Training_Protocol.md) ──────────────────────
REQUIRED_BLOCKS = ["meta", "config", "environment", "dataset_info",
                   "efficiency", "best_epoch", "best_metrics", "ablation",
                   "training_history"]
REQUIRED_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
SOUND_EVENT_MODELS = {"M1", "M2", "M3", "M4", "M12", "M21", "M22", "M23"}
SOUND_EVENT_EXTRA = ["specificity_macro", "icbhi_score"]
OPEN_SET_MODELS = {"M6", "M15", "M17", "M29"}
OPEN_SET_EXTRA = ["auroc", "aupr"]      # matched loosely — any key containing these
REQUIRED_ABLATION = ["ablation_group", "ablation_role", "variable_changed",
                     "component_flags", "loss_weights"]


@dataclass
class Finding:
    severity: str
    model: str
    check: str
    message: str
    evidence: str = ""
    source: str = ""


@dataclass
class ModelAudit:
    model_id: str
    path: str
    kind: str                       # "protocol" | "legacy" | "sweep_csv"
    findings: list = field(default_factory=list)

    @property
    def worst(self):
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key=lambda s: SEVERITY_ORDER[s])


# ── helpers ──────────────────────────────────────────────────────────────────
def find_repo_root(start=None):
    d = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(8):
        if all(os.path.exists(os.path.join(d, m))
               for m in ("Model_Training_Reference.md", "Model_Training_Protocol.md")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


def model_id_from_path(path):
    """Recover 'M15' from .../M15/results_M15.json or .../M15_metrics.json."""
    base = os.path.basename(path)
    for token in base.replace(".", "_").split("_"):
        if token.startswith("M") and token[1:].isdigit():
            return token
    for part in reversed(os.path.dirname(path).split(os.sep)):
        if not part.strip():                # leading '/' yields an empty component
            continue
        cleaned = part.split()[0]           # handles 'M13 [UPDATED]'
        if cleaned.startswith("M") and cleaned[1:].isdigit():
            return cleaned
    return "?"


def flatten_numeric(obj, prefix=""):
    """Yield (dotted_key, float) for every numeric leaf in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def _is_cited_reference_field(key):
    """
    True when `key` is explicitly marked as citing another model's already-audited
    number rather than reporting this model's own result -- e.g. M29's
    `reference_M6_openmax.auroc`. That number is real and belongs in the file for
    context, but flagging it as a fresh CRITICAL against M29 would double-count a
    problem M6's own entry already reports, and would misleadingly suggest M29's
    OWN discrimination is broken when it is not.

    Deliberately keyed on the literal word "reference" so it only matches fields
    the author explicitly marked as such (a documented convention, not a guess) --
    it must NOT match M15's `M6_AUROC`, which names no such thing and genuinely is
    the unlabelled, uncritical comparison Attack 1 warns about.
    """
    return "reference" in key.lower()


# ── validity checks ──────────────────────────────────────────────────────────
def check_discrimination_at_chance(mid, payload, src, findings):
    """
    A discrimination metric (AUROC) at or below 0.5 means the score carries no
    usable signal -- 0.5 is a coin flip and below 0.5 is worse than one.
    """
    for key, val in flatten_numeric(payload):
        low = key.lower()
        if "auroc" not in low and "auc" not in low:
            continue
        if not (0.0 <= val <= 1.0):
            continue
        if _is_cited_reference_field(key):
            findings.append(Finding(
                INFO, mid, "cites_subchance_reference_value",
                f"`{key}` = {val:.4f} is cited here as context from another "
                f"model's already-audited result (it is at or below chance). This "
                f"is not a claim about {mid}'s own performance -- see that other "
                f"model's audit entry for the underlying problem -- but do not "
                f"treat it as a validity benchmark for {mid}.",
                f"{key} = {val:.4f}", src))
            continue
        if val <= 0.5:
            findings.append(Finding(
                CRITICAL, mid, "discrimination_at_or_below_chance",
                f"`{key}` = {val:.4f} is at or below chance (0.5). A random scorer "
                f"would do as well or better, so this value cannot support a "
                f"detection claim.",
                f"{key} = {val:.4f}", src))
        elif val < 0.65:
            findings.append(Finding(
                WARNING, mid, "weak_discrimination",
                f"`{key}` = {val:.4f} is weak. Conventionally <0.7 AUROC is "
                f"considered poor discrimination; this needs framing as a negative "
                f"or preliminary result, not a validated mechanism.",
                f"{key} = {val:.4f}", src))


def _looks_like_baseline_key(key, own_model_id):
    """
    True when `key` plausibly names a *comparison baseline* rather than just
    another measurement. Two AUROCs in one file are only a claim if one of them
    is being held up as the thing being beaten -- e.g. 'M6_AUROC' inside M15's
    file. Two dataset-level scores (Coswara vs SPRSound) are not a comparison,
    and flagging them as one would be a false positive.
    """
    low = key.lower()
    if "baseline" in low or "_vs_" in low or "prior" in low:
        return True
    for token in key.replace(".", "_").split("_"):
        if (token.startswith("M") and token[1:].isdigit()
                and token != own_model_id):
            return True
    return False


def check_ratio_against_subchance_baseline(mid, payload, src, findings):
    """
    Catches 'we beat the baseline by N%' where the baseline is itself below chance.
    A coin flip also beats a sub-chance baseline, so the ratio is meaningless.

    Fields explicitly marked as a cited reference (see _is_cited_reference_field)
    are excluded from the "baseline" side: citing a known-broken number for context
    is not the same claim as computing an uncritical improvement ratio against it,
    and the field is already surfaced by check_discrimination_at_chance.
    """
    aurocs = {k: v for k, v in flatten_numeric(payload)
              if ("auroc" in k.lower() or "auc" in k.lower()) and 0.0 <= v <= 1.0}
    if len(aurocs) < 2:
        return
    subchance = {k: v for k, v in aurocs.items()
                 if v <= 0.5 and _looks_like_baseline_key(k, mid)
                 and not _is_cited_reference_field(k)}
    above = {k: v for k, v in aurocs.items() if v > 0.5}
    if subchance and above:
        b_key, b_val = min(subchance.items(), key=lambda kv: kv[1])
        a_key, a_val = max(above.items(), key=lambda kv: kv[1])
        findings.append(Finding(
            CRITICAL, mid, "comparison_against_subchance_baseline",
            f"This file compares `{a_key}` ({a_val:.4f}) against `{b_key}` "
            f"({b_val:.4f}), but the baseline is BELOW chance. Any relative "
            f"improvement quoted here (e.g. "
            f"'{(a_val / b_val - 1) * 100:.1f}% better') is not a real result -- "
            f"a coin flip would also 'beat' it. The baseline itself is broken and "
            f"must be fixed before the comparison means anything.",
            f"{a_key}={a_val:.4f} vs {b_key}={b_val:.4f}", src))


def check_perfect_metrics(mid, payload, src, findings):
    """Perfect scores on a hard task are a leak or train-set evaluation, not a win."""
    perfect = [k for k, v in flatten_numeric(payload)
               if v == 1.0 and any(t in k.lower() for t in
                                   ("accuracy", "precision", "recall", "f1", "auroc"))]
    if len(perfect) >= 2:
        findings.append(Finding(
            CRITICAL, mid, "perfect_metrics_implausible",
            f"{len(perfect)} classification metrics are exactly 1.0. On ICBHI this is "
            f"not achievable -- published state of the art is ~0.60-0.65 ICBHI score. "
            f"The near-certain causes are evaluating on the training split or a label "
            f"leaking into the features. Treat this run as invalid until the "
            f"evaluation split is verified.",
            ", ".join(perfect[:6]), src))


def check_single_class_collapse(mid, payload, src, findings):
    """A model predicting one class for everything scores non-zero but learned nothing."""
    bm = payload.get("best_metrics", {})
    per_class = bm.get("per_class", {})
    if not isinstance(per_class, dict) or len(per_class) < 2:
        return
    recalls = {c: v.get("recall") for c, v in per_class.items()
               if isinstance(v, dict) and v.get("recall") is not None}
    if len(recalls) < 2:
        return
    at_one = [c for c, r in recalls.items() if r >= 0.999]
    at_zero = [c for c, r in recalls.items() if r <= 0.001]
    if at_one and len(at_zero) == len(recalls) - len(at_one):
        findings.append(Finding(
            CRITICAL, mid, "single_class_collapse",
            f"The model predicts `{', '.join(at_one)}` for essentially every sample "
            f"(recall 1.00) and never predicts {', '.join(at_zero)} (recall 0.00). "
            f"This is majority-class collapse -- the reported accuracy reflects the "
            f"class prior, not learning.",
            f"recalls: {recalls}", src))


def check_near_majority_class(mid, payload, src, findings):
    """Accuracy no better than always predicting the largest class."""
    bm = payload.get("best_metrics", {})
    per_class = bm.get("per_class", {})
    acc = bm.get("accuracy")
    if not isinstance(per_class, dict) or acc is None:
        return
    supports = {c: v.get("support") for c, v in per_class.items()
                if isinstance(v, dict) and isinstance(v.get("support"), (int, float))}
    total = sum(supports.values()) if supports else 0
    if total <= 0 or len(supports) < 2:
        return
    majority = max(supports.values()) / total
    if acc <= majority + 0.02:
        findings.append(Finding(
            WARNING, mid, "no_better_than_majority_class",
            f"Accuracy {acc:.4f} is not meaningfully above the majority-class rate "
            f"{majority:.4f} (always predicting the largest class). The model may be "
            f"learning little; report macro-F1 rather than accuracy and check the "
            f"per-class breakdown.",
            f"accuracy={acc:.4f}, majority prior={majority:.4f}", src))


def check_frozen_validation_curve(mid, payload, src, findings):
    """Validation metric identical every epoch => the model never changed its outputs."""
    hist = payload.get("training_history", [])
    if not isinstance(hist, list) or len(hist) < 4:
        return
    for key in ("val_accuracy", "val_f1_macro", "val_icbhi_score"):
        vals = [h.get(key) for h in hist if isinstance(h, dict) and h.get(key) is not None]
        if len(vals) >= 4 and len(set(round(v, 6) for v in vals)) == 1:
            findings.append(Finding(
                CRITICAL, mid, "frozen_validation_metric",
                f"`{key}` is identical ({vals[0]}) across all {len(vals)} epochs. The "
                f"model's validation predictions never changed, which means it is "
                f"emitting a constant output regardless of input.",
                f"{key} constant at {vals[0]} for {len(vals)} epochs", src))


def check_early_best_epoch(mid, payload, src, findings):
    """Best epoch in the first 10% of the budget suggests it never really trained."""
    be = payload.get("best_epoch", {})
    hist = payload.get("training_history", [])
    epoch = be.get("epoch") if isinstance(be, dict) else None
    total = payload.get("config", {}).get("num_epochs") or (len(hist) or None)
    if not epoch or not total or total < 10:
        return
    if epoch == 1:
        findings.append(Finding(
            CRITICAL, mid, "best_epoch_is_first",
            f"The best epoch is epoch 1 of {total}. Training made the model worse from "
            f"the very first update, which usually indicates a broken loss, label "
            f"mismatch, or a learning rate far too high.",
            f"best_epoch=1, num_epochs={total}", src))
    elif epoch / total <= 0.15:
        findings.append(Finding(
            WARNING, mid, "best_epoch_very_early",
            f"Best epoch {epoch} of {total} ({epoch / total:.0%} through the budget). "
            f"The remaining {total - epoch} epochs only overfit. Worth reporting, and "
            f"worth checking the model was given a fair chance to converge.",
            f"best_epoch={epoch}/{total}", src))


def check_patient_independence(mid, payload, src, findings):
    """Protocol §1 is the project's one non-negotiable validity requirement."""
    di = payload.get("dataset_info", {})
    split = str(di.get("split_method", "")).lower()
    verified = di.get("patient_leakage_verified")
    if verified is True:
        return
    if "patient" not in split and "lopo" not in split:
        findings.append(Finding(
            WARNING, mid, "patient_independence_unclear",
            f"`dataset_info.split_method` = '{di.get('split_method')}' does not state "
            f"a patient-independent split, and `patient_leakage_verified` is absent. "
            f"Protocol §1 calls this a research-validity requirement, not a style "
            f"choice -- it should be asserted in code, not assumed.", "", src))
    else:
        findings.append(Finding(
            INFO, mid, "patient_independence_not_asserted",
            f"Split is declared patient-independent ('{di.get('split_method')}') but "
            f"`patient_leakage_verified` is not set, so nothing checked it at runtime.",
            "", src))


def check_sample_counts(mid, payload, src, findings):
    """Implausibly small evaluation sets make every metric high-variance."""
    di = payload.get("dataset_info", {})
    test_n = di.get("test_samples")
    if isinstance(test_n, (int, float)) and 0 < test_n < 150:
        findings.append(Finding(
            WARNING, mid, "small_evaluation_set",
            f"Only {int(test_n)} test samples. Every metric derived from this has wide "
            f"confidence intervals; a single sample moves accuracy by "
            f"{1 / test_n:.1%}. Report intervals or use LOPO.",
            f"test_samples={int(test_n)}", src))


def check_open_set_metrics_present(mid, payload, src, findings):
    """M6/M15/M17 exist to produce unknown-detection metrics."""
    if mid not in OPEN_SET_MODELS:
        return
    keys = " ".join(k.lower() for k, _ in flatten_numeric(payload))
    if not any(t in keys for t in OPEN_SET_EXTRA):
        findings.append(Finding(
            CRITICAL, mid, "missing_open_set_metrics",
            f"{mid} is an open-set model but reports no AUROC/AUPR anywhere in its "
            f"results file. Model_Training_Reference.md requires unknown-detection "
            f"precision/recall + AUROC/AUPR for {mid} -- these are the metrics it "
            f"exists to provide, and the paper's headline comparison needs them.",
            "", src))


# ── schema compliance ────────────────────────────────────────────────────────
def check_schema(mid, payload, src, findings):
    missing_blocks = [b for b in REQUIRED_BLOCKS if b not in payload]
    if missing_blocks:
        findings.append(Finding(
            WARNING, mid, "schema_missing_blocks",
            f"Missing §4 block(s): {', '.join(missing_blocks)}. The M28 merge script "
            f"expects every block; absent ones must be reconstructed by hand.",
            ", ".join(missing_blocks), src))

    bm = payload.get("best_metrics", {})
    needed = list(REQUIRED_METRICS)
    if mid in SOUND_EVENT_MODELS:
        needed += SOUND_EVENT_EXTRA
    missing_metrics = [m for m in needed if m not in bm]
    if missing_metrics:
        findings.append(Finding(
            WARNING, mid, "schema_missing_metrics",
            f"Missing §3 metric(s) in `best_metrics`: {', '.join(missing_metrics)}.",
            ", ".join(missing_metrics), src))

    ab = payload.get("ablation", {})
    if not ab:
        findings.append(Finding(
            WARNING, mid, "schema_missing_ablation",
            "No §4.1 `ablation` block. Without it this run cannot be placed in the "
            "ablation table automatically at M28.", "", src))
    else:
        missing_ab = [f for f in REQUIRED_ABLATION if f not in ab]
        if missing_ab:
            findings.append(Finding(
                INFO, mid, "schema_incomplete_ablation",
                f"`ablation` block missing: {', '.join(missing_ab)}.",
                ", ".join(missing_ab), src))

    eff = payload.get("efficiency", {})
    if eff and eff.get("inference_time_ms_per_sample") in (None, 0):
        findings.append(Finding(
            INFO, mid, "no_inference_latency",
            "`inference_time_ms_per_sample` not measured. Recommended by §4 and it "
            "feeds the efficiency columns of the ablation table.", "", src))


# ── file-type auditors ───────────────────────────────────────────────────────
def audit_protocol_json(path, repo_root):
    mid = model_id_from_path(path)
    src = os.path.relpath(path, repo_root)
    audit = ModelAudit(mid, src, "protocol")
    try:
        payload = json.load(open(path))
    except Exception as e:
        audit.findings.append(Finding(CRITICAL, mid, "unparseable",
                                      f"Could not parse JSON: {e}", "", src))
        return audit

    for fn in (check_schema, check_discrimination_at_chance,
               check_ratio_against_subchance_baseline, check_perfect_metrics,
               check_single_class_collapse, check_near_majority_class,
               check_frozen_validation_curve, check_early_best_epoch,
               check_patient_independence, check_sample_counts,
               check_open_set_metrics_present):
        try:
            fn(mid, payload, src, audit.findings)
        except Exception as e:                       # a check must never kill the audit
            audit.findings.append(Finding(INFO, mid, "check_error",
                                          f"Check {fn.__name__} errored: {e}", "", src))
    return audit


def audit_legacy_metrics(path, repo_root):
    """
    Non-protocol metrics files (M15_metrics.json etc). These are invisible to the
    M28 merge, so they are audited separately and flagged as non-compliant.
    """
    mid = model_id_from_path(path)
    src = os.path.relpath(path, repo_root)
    audit = ModelAudit(mid, src, "legacy")
    try:
        payload = json.load(open(path))
    except Exception as e:
        audit.findings.append(Finding(CRITICAL, mid, "unparseable",
                                      f"Could not parse JSON: {e}", "", src))
        return audit

    audit.findings.append(Finding(
        CRITICAL, mid, "not_protocol_compliant",
        f"`{os.path.basename(path)}` is not a §4 `results_{mid}.json`. It holds "
        f"{len(payload)} loose field(s) instead of the required schema "
        f"(meta/config/efficiency/best_metrics/ablation/training_history). The M28 "
        f"merge cannot consume this, and it carries none of the efficiency, "
        f"per-class or confusion-matrix data the paper needs.",
        f"fields: {', '.join(list(payload)[:8])}", src))

    for fn in (check_discrimination_at_chance, check_ratio_against_subchance_baseline,
               check_perfect_metrics):
        try:
            fn(mid, payload, src, audit.findings)
        except Exception as e:
            audit.findings.append(Finding(INFO, mid, "check_error",
                                          f"Check {fn.__name__} errored: {e}", "", src))
    return audit


def audit_notebook_data_source(path, repo_root):
    """
    The most important check in this file.

    A notebook can be fully protocol-compliant, produce plausible curves, and still
    be training on data it invented. The signature is a torch.utils.data.Dataset
    whose __getitem__ *manufactures* its input tensor (torch.randn / np.random)
    instead of loading audio. Such a run measures nothing about ICBHI no matter how
    the metrics look.

    Deliberately narrow to avoid false positives: random tensors used for shape
    checks, latency probes or weight init are normal and are NOT flagged. Only a
    Dataset that fabricates the sample it returns counts.
    """
    mid = model_id_from_path(path)
    src_rel = os.path.relpath(path, repo_root)
    audit = ModelAudit(mid, src_rel, "notebook")
    try:
        nb = json.load(open(path))
    except Exception:
        return audit

    cells = ["".join(c.get("source", [])) for c in nb.get("cells", [])
             if c.get("cell_type") == "code"]
    whole = "\n".join(cells)

    REAL_AUDIO = ("librosa.load", "soundfile", "sf.read", "torchaudio.load",
                  "audio_and_txt_files", ".wav", "wavfile")
    has_real_audio = any(t in whole for t in REAL_AUDIO)

    # Signal 1 -- a Dataset subclass whose __getitem__ fabricates its return tensor.
    fabricating = []
    for cell in cells:
        if "Dataset" not in cell or "__getitem__" not in cell:
            continue
        # crude but effective: take the __getitem__ body to the end of the cell
        body = cell.split("__getitem__", 1)[1]
        makes_random = any(t in body for t in
                           ("torch.randn", "torch.rand(", "np.random.randn",
                            "np.random.rand(", "np.random.RandomState",
                            "rng.randn", "rng.normal", "rng.rand"))
        loads_file = any(t in body for t in REAL_AUDIO)
        if makes_random and not loads_file:
            for line in cell.split("\n"):
                if "class " in line and "Dataset" in line:
                    fabricating.append(line.strip().rstrip(":"))
                    break
            else:
                fabricating.append("<unnamed Dataset>")

    # Signal 2 -- the author named it what it is. Nobody calls a genuine ICBHI
    # loader "SyntheticICBHIDataset" or "generate_simulated_logits", so a naming
    # match is a strong, low-false-positive signal that catches generators which
    # never subclass Dataset at all (e.g. functions returning logits directly).
    SYNTH_NAMES = ("synthetic", "simulated", "dummy", "fake", "toy", "mock")
    for cell in cells:
        for line in cell.split("\n"):
            stripped = line.strip()
            if not (stripped.startswith("class ") or stripped.startswith("def ")):
                continue
            name = stripped.split()[1].split("(")[0].split(":")[0]
            if any(t in name.lower() for t in SYNTH_NAMES):
                decl = stripped.rstrip(":")
                if decl not in fabricating:
                    fabricating.append(decl)

    if fabricating and not has_real_audio:
        audit.findings.append(Finding(
            CRITICAL, mid, "synthetic_data_not_real_dataset",
            f"This notebook trains and evaluates on **synthetically generated data, not "
            f"ICBHI**. {', '.join(fabricating)} builds its input tensor with "
            f"`torch.randn`/`np.random` inside `__getitem__` and never loads an audio "
            f"file. Every metric produced by this notebook describes random noise, so "
            f"none of it can appear in the paper. This is pipeline scaffolding that was "
            f"committed as a result -- the experiment still needs to be run on the real "
            f"corpus.",
            "; ".join(fabricating), src_rel))

    if "Using random initialization" in whole or "random initialization for testing" in whole:
        audit.findings.append(Finding(
            CRITICAL, mid, "model_may_be_randomly_initialised",
            "The notebook contains a fallback that proceeds with a randomly-initialised "
            "model when the upstream checkpoint is missing. If that branch fired, the "
            "reported metrics describe an untrained network. Make the missing checkpoint "
            "a hard failure rather than a warning.",
            "", src_rel))

    return audit


def audit_sweep_csv(path, repo_root):
    """
    Sweep tables (e.g. compression). The signature failure is a metric that does
    not move at all across the swept variable.
    """
    mid = model_id_from_path(path)
    src = os.path.relpath(path, repo_root)
    audit = ModelAudit(mid, src, "sweep_csv")
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception as e:
        audit.findings.append(Finding(CRITICAL, mid, "unparseable",
                                      f"Could not parse CSV: {e}", "", src))
        return audit
    if len(rows) < 2:
        return audit

    for col in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (TypeError, ValueError):
                vals = []
                break
        if not vals or len(set(vals)) != 1:
            continue
        if any(t in col.lower() for t in ("acc", "f1", "auroc", "retention", "score")):
            spread = ""
            for pcol in rows[0]:
                if "param" in pcol.lower():
                    try:
                        ps = [float(r[pcol]) for r in rows]
                        spread = (f" while `{pcol}` changes "
                                  f"{max(ps) / max(min(ps), 1):.0f}x")
                    except (TypeError, ValueError):
                        pass
            audit.findings.append(Finding(
                CRITICAL, mid, "metric_constant_across_sweep",
                f"`{col}` is identical ({vals[0]}) at every point of the sweep"
                f"{spread}. A model whose accuracy does not move under that much "
                f"capacity change is emitting a constant prediction; the sweep is "
                f"measuring nothing.",
                f"{col} = {vals[0]} across {len(vals)} sweep points", src))
    return audit


# ── discovery ────────────────────────────────────────────────────────────────
def discover(repo_root):
    protocol, legacy, sweeps, notebooks = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__",
                                                        "protocol_bundle", "venv_m2")]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if fn.startswith("results_M") and fn.endswith(".json"):
                protocol.append(full)
            elif fn.endswith("_metrics.json") and fn.startswith("M"):
                legacy.append(full)
            elif fn.endswith(".csv") and "sweep" in fn.lower():
                sweeps.append(full)
            elif fn.endswith(".ipynb") and "checkpoint" not in fn.lower():
                notebooks.append(full)
    return sorted(protocol), sorted(legacy), sorted(sweeps), sorted(notebooks)


# ── report ───────────────────────────────────────────────────────────────────
def build_report(audits, repo_root):
    import datetime
    all_findings = [f for a in audits for f in a.findings]
    crit = [f for f in all_findings if f.severity == CRITICAL]
    warn = [f for f in all_findings if f.severity == WARNING]
    info = [f for f in all_findings if f.severity == INFO]

    models_with_crit = sorted({f.model for f in crit})
    clean = sorted({a.model_id for a in audits if not a.findings})

    L = []
    A = L.append
    A("# OWMTL Project Audit")
    A("")
    A(f"**Generated:** {datetime.date.today().isoformat()} by "
      f"`Asif's/audit/audit_project.py` · **Files scanned:** {len(audits)}")
    A("")
    A("Automated protocol-compliance and result-validity audit across every results "
      "file in the repository. Each check corresponds to a failure mode actually "
      "present in this repo, not a hypothetical one.")
    A("")
    A("---")
    A("")
    A("## Summary")
    A("")
    A(f"| Severity | Count | Meaning |")
    A(f"|---|---|---|")
    A(f"| 🔴 CRITICAL | {len(crit)} | The result does not support the claim made on it |")
    A(f"| 🟡 WARNING | {len(warn)} | Needs resolving before submission |")
    A(f"| ⚪ INFO | {len(info)} | Worth knowing, not blocking |")
    A("")
    if models_with_crit:
        A(f"**Models with critical findings:** {', '.join(models_with_crit)}")
        A("")
    if clean:
        A(f"**Files with no findings:** {', '.join(clean)}")
        A("")
    A("---")
    A("")

    # ── critical section first, grouped by model ──
    for severity, emoji, title in ((CRITICAL, "🔴", "Critical findings"),
                                   (WARNING, "🟡", "Warnings"),
                                   (INFO, "⚪", "Informational")):
        subset = [f for f in all_findings if f.severity == severity]
        if not subset:
            continue
        A(f"## {emoji} {title}")
        A("")
        by_model = {}
        for f in subset:
            by_model.setdefault(f.model, []).append(f)
        for mid in sorted(by_model, key=lambda m: (len(m), m)):
            A(f"### {mid}")
            A("")
            for f in by_model[mid]:
                A(f"**`{f.check}`** — {f.message}")
                if f.evidence:
                    A("")
                    A(f"> `{f.evidence}`")
                A("")
                A(f"<sub>source: `{f.source}`</sub>")
                A("")
        A("---")
        A("")

    A("## What the checks look for")
    A("")
    A("| Check | Catches |")
    A("|---|---|")
    A("| `synthetic_data_not_real_dataset` | a Dataset that fabricates its input instead of loading audio |")
    A("| `model_may_be_randomly_initialised` | a fallback that proceeds with an untrained model |")
    A("| `discrimination_at_or_below_chance` | AUROC ≤ 0.5 — a coin flip does as well |")
    A("| `comparison_against_subchance_baseline` | \"beats baseline by N%\" where the baseline is below chance |")
    A("| `perfect_metrics_implausible` | metrics of exactly 1.0 — leak or train-set evaluation |")
    A("| `metric_constant_across_sweep` | accuracy that does not move as capacity changes |")
    A("| `single_class_collapse` | one class at recall 1.0, the rest at 0.0 |")
    A("| `frozen_validation_metric` | validation score identical every epoch |")
    A("| `no_better_than_majority_class` | accuracy at the class prior |")
    A("| `best_epoch_is_first` / `_very_early` | the model never really trained |")
    A("| `missing_open_set_metrics` | M6/M15/M17 with no AUROC/AUPR |")
    A("| `not_protocol_compliant` | metrics files the M28 merge cannot read |")
    A("| `schema_*` | §4 / §4.1 blocks the merge expects |")
    A("| `patient_independence_*` | protocol §1, the one non-negotiable requirement |")
    A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="print the summary line only")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()

    repo_root = find_repo_root()
    protocol, legacy, sweeps, notebooks = discover(repo_root)

    audits = ([audit_protocol_json(p, repo_root) for p in protocol]
              + [audit_legacy_metrics(p, repo_root) for p in legacy]
              + [audit_sweep_csv(p, repo_root) for p in sweeps]
              + [a for a in (audit_notebook_data_source(p, repo_root) for p in notebooks)
                 if a.findings])

    report = build_report(audits, repo_root)
    out_md = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "PROJECT_AUDIT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)

    out_json = os.path.splitext(out_md)[0] + ".json"
    with open(out_json, "w") as f:
        json.dump([{"model_id": a.model_id, "path": a.path, "kind": a.kind,
                    "findings": [asdict(x) for x in a.findings]} for a in audits],
                  f, indent=2)

    all_findings = [f for a in audits for f in a.findings]
    n_crit = sum(1 for f in all_findings if f.severity == CRITICAL)
    n_warn = sum(1 for f in all_findings if f.severity == WARNING)
    n_info = sum(1 for f in all_findings if f.severity == INFO)

    if not args.quiet:
        print("=" * 78)
        print("OWMTL PROJECT AUDIT")
        print("=" * 78)
        print(f"Repo      : {repo_root}")
        print(f"Scanned   : {len(protocol)} protocol JSON, {len(legacy)} legacy metrics, "
              f"{len(sweeps)} sweep CSV, {len(notebooks)} notebooks")
        print(f"Findings  : {n_crit} CRITICAL, {n_warn} WARNING, {n_info} INFO")
        print("-" * 78)
        for a in sorted(audits, key=lambda x: (SEVERITY_ORDER.get(x.worst, 3),
                                               len(x.model_id), x.model_id)):
            mark = {CRITICAL: "CRIT", WARNING: "WARN", INFO: "info", None: " OK "}[a.worst]
            print(f"  [{mark}] {a.model_id:<5} {a.path}")
            for f in a.findings:
                if f.severity == CRITICAL:
                    print(f"          -> {f.check}")
        print("-" * 78)
        print(f"Report    : {out_md}")
        print(f"Machine   : {out_json}")
        print("=" * 78)
    else:
        print(f"{n_crit} CRITICAL, {n_warn} WARNING, {n_info} INFO -> {out_md}")

    return 1 if n_crit else 0


if __name__ == "__main__":
    sys.exit(main())
