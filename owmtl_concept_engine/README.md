# OWMTL — Critical-Fix Engine (concept bottleneck foundation)

This package implements **only the Critical Fixes** from `OWMTL_Fix_Triage.md`, aligned to the
current direction (physics-grounded, label-free **acoustic concept bottleneck** + faithfulness audit).
It is the shared engine both Path A and Path B stand on. Everything here has been unit-tested on
synthetic signals and toy data in this environment; the two notebooks run on real ICBHI once you set
three paths.

## What this delivers (maps to the triage)

| Critical fix | File(s) | Gate |
|---|---|---|
| **M35 → physics concept extractors** (clinically-named, label-free DSP) | `owmtl/concept_extractors.py`, `notebooks/01_*.ipynb` | **G2** |
| **M13 → strict concept bottleneck head** + independent/sequential/leaky/opaque controls | `owmtl/bottleneck.py`, `notebooks/02_*.ipynb` | **G3** |
| **CC1 — raw per-patient score dumping + CIs + paired tests** | `owmtl/eval_utils.py` | Protocol #10 |
| **CC2 — device-structure feasibility check** | `owmtl/device_check.py`, `scripts/check_device_structure.py` | **G4** |

## The concept vocabulary (why it's defensible for Q1)

14 **clinically-named, physics-computed** concepts per cycle — *not* generic MFCC/spectral features
(that design is already published; your novelty is the clinical naming + strict bottleneck + faithfulness):

`crackle_presence, crackle_rate_hz, fine_crackle_ratio, coarse_crackle_ratio, wheeze_presence,
wheeze_dominant_freq_hz, wheeze_duration_ratio, rhonchi_presence, spectral_flatness, papr_db,
inspiratory_energy_fraction, transient_timing_centroid, dominant_freq_hz, low_high_freq_ratio`

The synthetic test (`tests/test_concepts_synthetic.py`) proves the DSP discriminates crackle vs wheeze
vs rhonchi vs normal on the right physics **before** touching ICBHI. On ICBHI, notebook 01 validates
`crackle_presence`/`wheeze_presence` against the real crackle/wheeze labels (Gate G2).

## How to run

1. **Local sanity (no dataset):**
   ```bash
   python3 tests/test_concepts_synthetic.py          # concept physics -> ALL PASS
   python3 scripts/check_device_structure.py         # CC2 self-test
   ```
2. **On Kaggle/Colab:** upload the `owmtl/` folder (as a dataset or beside the notebook), open
   `notebooks/01_...ipynb`, set `AUDIO_DIR / SPLIT_FILE / DIAG_FILE` to the ICBHI paths, Run All.
   It writes `concepts_all.npz`, `concept_validation_report.json` (the **G2** evidence), and the device report.
3. Then run `notebooks/02_...ipynb` — plug in frozen **M2 features** (`get_M2_features`), Run All. It trains
   the four bottleneck variants and prints the **G3 inputs** (interpretability cost, leakage proxy, COPD-collapse
   check), dumping per-patient scores and `results_M13cbm_*.json` (§4 schema, with confusion matrices + CIs).

> `independent`-CBM runs on concepts alone; `sequential/leaky/opaque` need M2 features. Easiest path: in
> notebook 01, also push each cycle's mel-spectrogram through frozen M2 and save `M2_features.npy` aligned to
> the cycle order.

## Protocol compliance (baked in)

- **Official 60/40 split** enforced via `owmtl.icbhi_data` (one source of truth).
- **`confusion_matrix_raw` committed** for every trained variant.
- **Bootstrap 95% CIs + McNemar** available in `eval_utils`; **raw per-patient scores dumped** (CC1) so a
  proper paired (DeLong) test becomes possible — the exact gap `SIGNIFICANCE_REPORT.md` flagged.
- **`concept_metrics` + new `component_flags`** written per the updated `Model_Training_Protocol.md`.

## Dependencies
`numpy scipy scikit-learn` (extractors + eval), `torch` (bottleneck), `librosa` (audio loading in notebooks),
`nbformat` (only to regenerate notebooks). See `requirements.txt`.

## What this is NOT
Downstream steps (concept-space OOD, full information-theoretic leakage, intervention API, FM probing,
covariate-shift eval) are **not** here — they are the Weeks 6–7 fixes, deliberately out of scope for the
Critical-Fix stage. Do G2 → G3 first.
