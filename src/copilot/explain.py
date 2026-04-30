"""Optional SHAP explanations for XGBoost surrogate predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from copilot.config import SETTINGS
from copilot.evaluate import load_model, prepare_features
from copilot.features import FeatureSchema


def shap_values(
    frame: pd.DataFrame,
    model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json",
    schema_path: Path = SETTINGS.models_dir / "feature_schema.json",
) -> tuple[pd.DataFrame, object]:
    import shap

    schema = FeatureSchema.load(schema_path)
    model = load_model(model_path)
    features = prepare_features(frame, schema)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)
    return pd.DataFrame(values, columns=schema.feature_columns, index=frame.index), explainer


def top_local_contributions(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    values, _ = shap_values(frame)
    row = values.iloc[0].sort_values(key=lambda series: series.abs(), ascending=False).head(top_n)
    return pd.DataFrame({"feature": row.index, "shap_value": row.values})


def save_shap_summary(
    frame: pd.DataFrame,
    out_path: Path = SETTINGS.figures_dir / "shap_summary.png",
) -> Path | None:
    try:
        import shap

        schema = FeatureSchema.load()
        model = load_model()
        features = prepare_features(frame, schema)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(features)
        shap.summary_plot(values, features, show=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path
    except Exception:
        return None
