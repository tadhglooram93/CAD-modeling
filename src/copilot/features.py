"""Feature engineering for the DrivAerML tabular aerodynamic surrogate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from copilot.config import SETTINGS


class FeatureSchema(BaseModel):
    version: str = SETTINGS.feature_schema_version
    feature_columns: list[str]
    target_column: str = "cd"
    numeric_source_columns: list[str] = Field(default_factory=list)
    train_feature_min: dict[str, float] = Field(default_factory=dict)
    train_feature_max: dict[str, float] = Field(default_factory=dict)

    def save(self, path: Path = SETTINGS.models_dir / "feature_schema.json") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS.models_dir / "feature_schema.json") -> FeatureSchema:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def find_column(frame: pd.DataFrame, *tokens: str) -> str | None:
    lowered = [token.lower() for token in tokens]
    for column in frame.columns:
        name = column.lower()
        if all(token in name for token in lowered):
            return column
    return None


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_derived_features(frame: pd.DataFrame, baseline_run_id: int | None = None) -> pd.DataFrame:
    result = frame.copy()

    length_col = find_column(result, "length")
    width_col = find_column(result, "width")
    height_col = find_column(result, "height")
    wheelbase_col = find_column(result, "wheelbase")
    hood_col = find_column(result, "hood", "height")
    roof_col = find_column(result, "roof", "height")
    rear_slope_col = find_column(result, "rear", "slope")
    frontal_area_col = find_column(result, "frontal", "area") or find_column(result, "reference", "area")

    if length_col and width_col:
        result["length_to_width"] = _safe_ratio(_numeric(result, length_col), _numeric(result, width_col))
    if height_col and width_col:
        result["height_to_width"] = _safe_ratio(_numeric(result, height_col), _numeric(result, width_col))
    if wheelbase_col and length_col:
        result["wheelbase_to_length"] = _safe_ratio(
            _numeric(result, wheelbase_col), _numeric(result, length_col)
        )
    if frontal_area_col:
        result["frontal_area_proxy"] = _numeric(result, frontal_area_col)
    elif width_col and height_col:
        result["frontal_area_proxy"] = _numeric(result, width_col) * _numeric(result, height_col)
    if roof_col:
        result["roof_height_proxy"] = _numeric(result, roof_col)
    if hood_col:
        result["hood_height_proxy"] = _numeric(result, hood_col)
    if rear_slope_col:
        result["rear_slope_proxy"] = _numeric(result, rear_slope_col)

    if length_col and wheelbase_col:
        remaining = _numeric(result, length_col) - _numeric(result, wheelbase_col)
        result["overhang_total_proxy"] = remaining
        result["overhang_front_proxy"] = remaining / 2.0
        result["overhang_rear_proxy"] = remaining / 2.0

    numeric_sources = [
        column
        for column in result.select_dtypes(include="number").columns
        if column != "run_id" and not column.startswith("param_delta_norm_")
    ]
    if baseline_run_id is not None and "run_id" in result.columns:
        matches = result[result["run_id"] == baseline_run_id]
        if not matches.empty:
            baseline = matches.iloc[0]
            stds = result[numeric_sources].std(numeric_only=True).replace(0, np.nan)
            for column in numeric_sources:
                if column in baseline.index:
                    result[f"param_delta_norm_{column}"] = (
                        _numeric(result, column) - float(baseline[column])
                    ) / float(stds[column])

    return result


def select_feature_columns(frame: pd.DataFrame, target_column: str = "cd") -> list[str]:
    excluded = {
        "run_id",
        target_column,
        "source_file_count",
    }
    excluded_prefixes = ("force_", "force_constref_", "source_", "dataset_", "ingested_", "stl_path")
    candidates: list[str] = []
    for column in frame.select_dtypes(include="number").columns:
        if column in excluded:
            continue
        if column.startswith(excluded_prefixes):
            continue
        candidates.append(column)
    return candidates


def build_feature_matrix(
    frame: pd.DataFrame,
    target_column: str = "cd",
    baseline_run_id: int | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, FeatureSchema]:
    engineered = add_derived_features(frame, baseline_run_id=baseline_run_id)
    feature_columns = select_feature_columns(engineered, target_column=target_column)
    features = engineered[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    target = None
    if target_column in engineered.columns:
        target = pd.to_numeric(engineered[target_column], errors="coerce")
    schema = FeatureSchema(
        feature_columns=feature_columns,
        target_column=target_column,
        numeric_source_columns=list(frame.select_dtypes(include="number").columns),
        train_feature_min={column: float(features[column].min()) for column in feature_columns},
        train_feature_max={column: float(features[column].max()) for column in feature_columns},
    )
    return features, target, schema


def write_feature_dictionary(path: Path = SETTINGS.reports_dir / "feature_dictionary.md") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = """# Feature Dictionary

This file documents the derived proxy features used by the XGBoost drag surrogate.

| Feature | Meaning |
| --- | --- |
| `length_to_width` | Overall length divided by width. |
| `height_to_width` | Overall height divided by width. |
| `wheelbase_to_length` | Wheelbase divided by length. |
| `frontal_area_proxy` | Reference/frontal area if available, otherwise width times height. |
| `roof_height_proxy` | Roof height parameter when available. |
| `hood_height_proxy` | Hood height parameter when available. |
| `rear_slope_proxy` | Rear slope parameter when available. |
| `overhang_total_proxy` | Length minus wheelbase. |
| `overhang_front_proxy` | Half of total overhang proxy. |
| `overhang_rear_proxy` | Half of total overhang proxy. |
| `param_delta_norm_*` | Candidate or run delta from a selected baseline, scaled by train-set standard deviation. |

These features are simplified engineering proxies and should not be read as official package
or homologation metrics.
"""
    path.write_text(content, encoding="utf-8")


def schema_to_json(schema: FeatureSchema) -> str:
    return json.dumps(schema.model_dump(), indent=2)
