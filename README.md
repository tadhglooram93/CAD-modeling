# AI Studio Feasibility Copilot

This project is a portfolio demonstration of how AI/ML can support automotive studio engineering workflows. It combines open automotive CFD geometry data, an XGBoost aerodynamic surrogate model, simplified package-aware feasibility checks, and a lightweight interactive UI for vehicle shape exploration.

The goal is not to replace CFD, CATIA/3DX package validation, regulatory checks, or production studio engineering review. The goal is to show a traceable AI-assisted workflow that moves from open vehicle geometry data to fast drag prediction, transparent feasibility warnings, and nearby candidate alternatives.

## Why It Matters

Generic AI-generated 3D shapes are rarely engineering-usable by themselves. Studio engineering workflows need data pedigree, reproducible models, package-style constraints, clear assumptions, and exportable artifacts. This project demonstrates that bridge in a compact form.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

bash scripts/download_drivaerml.sh --aggregate
python -m copilot.data_ingest --raw-dir data/raw/drivaerml --out data/processed/design_table.parquet
python -m copilot.train_xgboost --data data/processed/design_table.parquet --target cd
python -m copilot.search --baseline-run-id 42 --num-candidates 1000
streamlit run app/streamlit_app.py
```

## Dataset

The MVP uses DrivAerML from Hugging Face (`neashton/drivaerml`). The default download path fetches the aggregate CSV files needed for tabular surrogate modeling. Per-run STL downloads are optional and should be limited to a small subset for visualization.

Large volume and boundary files are intentionally excluded from the MVP.

## Modeling

The training pipeline compares three models:

- Mean predictor
- Ridge regression
- XGBoost regressor

The XGBoost model is only used in the demo when it improves on the trivial and linear baselines. Metrics and figures are saved under `models/` and `reports/figures/`.

## Feasibility Rules

The feasibility engine applies educational proxy package rules for length, width, height, frontal area drift, wheelbase ratio, hood/roof envelope, and normalized parameter deltas.

These are educational proxy rules used to demonstrate package-aware AI workflows. They are not OEM hardpoint checks, legal compliance checks, or production vehicle package validation.

## Demo

The Streamlit app lets a reviewer:

- Select a baseline vehicle run
- View source lineage and geometry parameters
- Predict drag with the trained surrogate
- Inspect feasibility warnings
- Compare top candidate alternatives
- Download candidate metadata

Screenshots and demo media belong in `artifacts/screenshots/` and `reports/figures/demo.gif`.

## Repository Structure

```text
src/copilot/
  data_ingest.py      # DrivAerML CSV scanning and manifest creation
  features.py         # Derived geometry and proxy package features
  train_xgboost.py    # Mean, ridge, and XGBoost model training
  evaluate.py         # Prediction and evaluation helpers
  feasibility.py      # Transparent proxy package rule engine
  search.py           # Local candidate perturbation search
  geometry.py         # STL preview and simplified scaffold export helpers
  lineage.py          # Artifact and candidate provenance schemas
app/
  streamlit_app.py    # Interactive copilot demo
reports/
  model_card.md
  feasibility_rules.md
```

## Local Workflow

1. Download a small tabular subset:

```bash
bash scripts/download_drivaerml.sh --aggregate
```

2. Build the processed design table:

```bash
python -m copilot.data_ingest --raw-dir data/raw/drivaerml --out data/processed/design_table.parquet
```

3. Train the surrogate:

```bash
python -m copilot.train_xgboost --data data/processed/design_table.parquet --target cd
```

4. Generate candidate alternatives:

```bash
python -m copilot.search --baseline-run-id 42 --num-candidates 1000
```

5. Launch the demo:

```bash
streamlit run app/streamlit_app.py
```

### Geometry preview (optional STL mesh)

Tabular training uses aggregate CSVs only. To show the 3D mesh preview for a specific run, download that run’s STL (large files—fetch only what you need):

```bash
# Example: run ID 42 only
bash scripts/download_drivaerml.sh --per-run --with-stl --run-start 42 --run-end 42
```

Files are written to `data/raw/drivaerml/run_<id>/drivaer_<id>.stl`. The app looks there even if you ingested from aggregate CSVs; re-running ingestion is optional.

## Limitations

- The feasibility checks are simplified, configurable proxy rules.
- The XGBoost model is a fast directional surrogate, not a CFD solver.
- Candidate search perturbs tabular geometry parameters; it does not produce production A-surfaces.
- CadQuery scaffold export is a simplified envelope artifact for review, not native OEM CAD automation.

## Ford Studio Engineering Relevance

My background is ML/AI engineering, so I built a miniature AI-augmented studio engineering workflow. It ingests open automotive CFD geometry data, trains an XGBoost surrogate model for fast drag prediction, applies simplified package-aware feasibility checks, proposes candidate design alternatives, and preserves data pedigree for generated artifacts.

The project is intentionally honest about its limits: it is not a production CATIA/3DX package tool, but it demonstrates how I would bridge AI workflows with real studio engineering constraints.

Application talking points:

- Built a traceable ML pipeline from open automotive CFD geometry data to drag prediction.
- Added package-aware feasibility checks so the AI workflow speaks engineering constraints, not just generative visuals.
- Preserved model, feature, rule-set, and source-file provenance for every generated candidate.
- Shipped an interactive demo that a hiring manager can understand quickly while retaining technical depth for reviewers.

## Future Work

- Add SHAP explanations for local feature contributions
- Add stronger out-of-distribution warnings
- Export simplified CadQuery STEP/STL envelope scaffolds
- Evaluate DrivAerNet++ after the MVP is polished
