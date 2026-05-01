"""Evaluation and prediction helpers for the XGBoost drag surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from copilot.config import SETTINGS
from copilot.features import FeatureSchema, add_derived_features
from copilot.train_xgboost import regression_metrics


def load_model(model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json") -> Any:
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(model_path)
        return model
    except Exception:
        return joblib.load(model_path)


def prepare_features(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    engineered = add_derived_features(frame)
    missing = [column for column in schema.feature_columns if column not in engineered.columns]
    for column in missing:
        engineered[column] = 0.0
    features = engineered[schema.feature_columns].apply(pd.to_numeric, errors="coerce")
    return features.fillna(features.median(numeric_only=True)).fillna(0.0)


def load_holdout_surrogate_rmse(metrics_path: Path = SETTINGS.models_dir / "metrics.json") -> float | None:
    """Approximate prediction uncertainty: hold-out RMSE (or MAE fallback) for the trained surrogate."""
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = data.get("models") or {}
    surrogate = models.get("xgboost")
    if not isinstance(surrogate, dict):
        return None
    if "rmse" in surrogate:
        return float(surrogate["rmse"])
    if "mae" in surrogate:
        return float(surrogate["mae"])
    return None


def load_model_and_predict(
    frame: pd.DataFrame,
    model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json",
    schema_path: Path = SETTINGS.models_dir / "feature_schema.json",
) -> pd.Series:
    schema = FeatureSchema.load(schema_path)
    model = load_model(model_path)
    predictions = model.predict(prepare_features(frame, schema))
    return pd.Series(predictions, index=frame.index, name="predicted_cd")


def evaluate_dataset(
    data_path: Path,
    model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json",
    schema_path: Path = SETTINGS.models_dir / "feature_schema.json",
) -> dict[str, object]:
    frame = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    schema = FeatureSchema.load(schema_path)
    predictions = load_model_and_predict(frame, model_path, schema_path)
    output = frame.copy()
    output["predicted_cd"] = predictions
    out_path = SETTINGS.reports_dir / "evaluation_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    metrics = {"prediction_file": str(out_path)}
    if schema.target_column in output.columns:
        valid = output[schema.target_column].notna()
        metrics["metrics"] = regression_metrics(output.loc[valid, schema.target_column], predictions.loc[valid])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SETTINGS.models_dir / "xgboost_cd_model.json")
    parser.add_argument("--data", type=Path, default=SETTINGS.data_processed / "design_table.parquet")
    parser.add_argument("--schema", type=Path, default=SETTINGS.models_dir / "feature_schema.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_dataset(args.data, args.model, args.schema)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
