"""Train baseline and XGBoost drag surrogate models for studio exploration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from copilot.config import SETTINGS
from copilot.features import build_feature_matrix, write_feature_dictionary
from copilot.ood import OODProfile

LOGGER = logging.getLogger(__name__)


def make_xgb(random_state: int = SETTINGS.random_state) -> Any:
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        LOGGER.warning(
            "XGBoost is unavailable in this Python environment. "
            "Falling back to GradientBoostingRegressor for smoke tests."
        )
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            random_state=random_state,
        )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_predicted_vs_actual(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("Actual Cd")
    plt.ylabel("Predicted Cd")
    plt.title("Predicted vs Actual Drag Coefficient")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_residuals(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true.to_numpy() - y_pred
    plt.figure(figsize=(7, 4))
    plt.scatter(y_pred, residuals, alpha=0.75)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Cd")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _save_feature_importance(model: Any, feature_columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = np.zeros(len(feature_columns))
    importances = pd.Series(values, index=feature_columns).sort_values().tail(20)
    plt.figure(figsize=(8, max(4, len(importances) * 0.25)))
    importances.plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def train(
    data_path: Path,
    target: str = "cd",
    model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json",
    metrics_path: Path = SETTINGS.models_dir / "metrics.json",
    run_cv: bool = False,
) -> dict[str, object]:
    SETTINGS.ensure_directories()
    data = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found. Available columns: {list(data.columns)}")

    features, y, schema = build_feature_matrix(data, target_column=target)
    if y is None:
        raise ValueError(f"Target column '{target}' could not be built.")
    valid = y.notna()
    features = features.loc[valid]
    y = y.loc[valid]
    if len(features) < 4:
        raise ValueError("Need at least 4 labeled rows to train and evaluate a surrogate model.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=SETTINGS.random_state
    )

    models = {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(),
        "xgboost": make_xgb(),
    }
    metrics: dict[str, object] = {
        "target": target,
        "row_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "feature_columns": list(features.columns),
        "models": {},
    }

    fitted: dict[str, object] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics["models"][name] = regression_metrics(y_test, preds)
        fitted[name] = model

    if run_cv and len(features) >= 10:
        cv = KFold(n_splits=min(5, len(features)), shuffle=True, random_state=SETTINGS.random_state)
        scores = cross_val_score(make_xgb(), features, y, scoring="neg_mean_absolute_error", cv=cv)
        metrics["xgboost_cv_mae"] = [float(-score) for score in scores]

    xgb = fitted["xgboost"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(xgb, "save_model"):
        xgb.save_model(model_path)
    else:
        joblib.dump(xgb, model_path)
    joblib.dump({"mean": fitted["mean"], "ridge": fitted["ridge"]}, SETTINGS.models_dir / "baselines.joblib")
    schema.save(SETTINGS.models_dir / "feature_schema.json")
    write_feature_dictionary()

    xgb_preds = xgb.predict(x_test)
    _save_predicted_vs_actual(y_test, xgb_preds, SETTINGS.figures_dir / "predicted_vs_actual.png")
    _save_residuals(y_test, xgb_preds, SETTINGS.figures_dir / "residuals.png")
    _save_feature_importance(xgb, list(features.columns), SETTINGS.figures_dir / "feature_importance.png")
    OODProfile.fit(features).save(SETTINGS.models_dir / "ood_profile.json")
    try:
        from copilot.explain import save_shap_summary

        save_shap_summary(data)
    except Exception:
        LOGGER.info("SHAP summary skipped because optional SHAP execution failed.", exc_info=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    LOGGER.info("Saved model to %s", model_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SETTINGS.data_processed / "design_table.parquet")
    parser.add_argument("--target", default="cd")
    parser.add_argument("--model-out", type=Path, default=SETTINGS.models_dir / "xgboost_cd_model.json")
    parser.add_argument("--metrics-out", type=Path, default=SETTINGS.models_dir / "metrics.json")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    metrics = train(args.data, args.target, args.model_out, args.metrics_out, args.cv)
    LOGGER.info("Metrics: %s", json.dumps(metrics["models"], indent=2))


if __name__ == "__main__":
    main()
