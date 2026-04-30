# Model Card: XGBoost Drag Surrogate

## Intended Use

This model is a fast directional surrogate for early vehicle shape exploration in the AI Studio Feasibility Copilot demo.

It is intended to help compare nearby DrivAerML-style geometry variants and surface candidate designs that may merit further engineering review.

## Non-Intended Use

This model must not be described as a CFD replacement. It is not an OEM package tool, a legal compliance check, or a production aerodynamic validation method.

## Dataset

- Source: DrivAerML (`neashton/drivaerml`)
- MVP files: aggregate geometry/reference/force CSVs, with optional STL files for selected runs
- Excluded files: OpenFOAM volume fields, boundary fields, and large slice datasets

## Target

Primary target: drag coefficient (`cd`) from the processed design table. The ingestion step aliases source drag coefficient columns to `cd` after preserving their original source-prefixed columns.

## Features

Features include numeric geometry parameters plus derived ratios and proxy package features documented in `reports/feature_dictionary.md`.

## Train/Test Split

The training script uses `train_test_split(test_size=0.2, random_state=42)`. Optional cross-validation can be enabled with `--cv` when enough rows are available.

## Baselines

The model is evaluated against:

- Mean predictor
- Ridge regression
- XGBoost regressor

XGBoost should clearly outperform the trivial and linear baselines before its predictions are used in the Streamlit demo.

## Metrics

Saved in `models/metrics.json`:

- MAE
- RMSE
- R2

Saved figures:

- `reports/figures/predicted_vs_actual.png`
- `reports/figures/residuals.png`
- `reports/figures/feature_importance.png`
- `reports/figures/shap_summary.png` when optional SHAP execution is available

## Known Limitations

- Trained on open DrivAerML geometry variations, not proprietary Ford studio data.
- Candidate search is local perturbation, not a validated aerodynamic optimization workflow.
- Feasibility checks are proxy package rules, not official hardpoint validation.
- Prediction uncertainty is approximate unless the stretch OOD workflow is enabled.

## Optional Explainability and OOD Checks

`src/copilot/explain.py` provides SHAP summary and local contribution helpers. `src/copilot/ood.py` stores a training feature profile and computes min/max plus z-distance warnings for selected baselines or generated candidates.

These checks explain and flag model behavior; they do not make the surrogate production-valid.
