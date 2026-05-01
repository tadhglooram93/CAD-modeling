from pathlib import Path

import numpy as np
import pandas as pd

from copilot.features import (
    FeatureSchema,
    add_derived_features,
    build_feature_matrix,
    select_feature_columns,
)
from copilot.train_xgboost import train


def sample_frame(n: int = 20) -> pd.DataFrame:
    length = np.linspace(4.2, 4.8, n)
    width = np.linspace(1.75, 1.9, n)
    height = np.linspace(1.35, 1.5, n)
    wheelbase = np.linspace(2.55, 2.85, n)
    cd = 0.22 + 0.03 * (height / width) + 0.01 * (length / width)
    return pd.DataFrame(
        {
            "run_id": range(1, n + 1),
            "geo_param_length": length,
            "geo_param_width": width,
            "geo_param_height": height,
            "geo_param_wheelbase": wheelbase,
            "geo_ref_reference_area": width * height,
            "cd": cd,
        }
    )


def test_add_derived_features_hand_computed() -> None:
    frame = sample_frame(3)
    result = add_derived_features(frame, baseline_run_id=1)

    assert result.loc[0, "length_to_width"] == frame.loc[0, "geo_param_length"] / frame.loc[0, "geo_param_width"]
    assert result.loc[0, "height_to_width"] == frame.loc[0, "geo_param_height"] / frame.loc[0, "geo_param_width"]
    assert result.loc[0, "wheelbase_to_length"] == frame.loc[0, "geo_param_wheelbase"] / frame.loc[0, "geo_param_length"]
    assert "param_delta_norm_geo_param_length" in result.columns


def test_feature_schema_round_trip(tmp_path: Path) -> None:
    features, target, schema = build_feature_matrix(sample_frame(), target_column="cd")
    path = tmp_path / "schema.json"
    schema.save(path)
    loaded = FeatureSchema.load(path)

    assert target is not None
    assert features.shape[0] == 20
    assert loaded.feature_columns == schema.feature_columns
    assert "length_to_width" in loaded.feature_columns


def test_cl_cs_not_used_as_cd_covariates() -> None:
    """Lift/side coeffs are simulation outputs; must not enter the drag surrogate."""
    frame = sample_frame(10)
    frame["cl"] = np.linspace(0.1, 0.2, 10)
    frame["cs"] = np.linspace(0.05, 0.08, 10)
    engineered = add_derived_features(frame, baseline_run_id=1)
    assert "param_delta_norm_cl" not in engineered.columns
    assert "param_delta_norm_cs" not in engineered.columns
    assert "param_delta_norm_force_cl" not in engineered.columns

    _, _, schema = build_feature_matrix(frame, target_column="cd")
    cols = set(schema.feature_columns)
    assert "cl" not in cols and "cs" not in cols
    eng_cols = select_feature_columns(engineered, target_column="cd")
    assert "cl" not in eng_cols and "cs" not in eng_cols


def test_train_pipeline_beats_mean_on_synthetic_data(tmp_path: Path) -> None:
    data_path = tmp_path / "design_table.parquet"
    sample_frame(40).to_parquet(data_path, index=False)
    metrics = train(
        data_path,
        target="cd",
        model_path=tmp_path / "xgb.json",
        metrics_path=tmp_path / "metrics.json",
    )

    mean_mae = metrics["models"]["mean"]["mae"]
    xgb_mae = metrics["models"]["xgboost"]["mae"]
    assert xgb_mae < mean_mae
