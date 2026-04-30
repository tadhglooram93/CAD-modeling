"""Out-of-distribution checks for generated or selected vehicle designs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from copilot.config import SETTINGS
from copilot.features import FeatureSchema, add_derived_features


class OODProfile(BaseModel):
    feature_columns: list[str]
    mean: dict[str, float]
    std: dict[str, float]
    train_feature_min: dict[str, float]
    train_feature_max: dict[str, float]
    mahalanobis_warning_threshold: float

    @classmethod
    def fit(cls, features: pd.DataFrame) -> OODProfile:
        std = features.std(numeric_only=True).replace(0, 1.0).fillna(1.0)
        z = (features - features.mean(numeric_only=True)) / std
        distances = np.sqrt((z**2).sum(axis=1))
        threshold = float(np.quantile(distances, 0.95)) if len(distances) else 0.0
        return cls(
            feature_columns=list(features.columns),
            mean={column: float(features[column].mean()) for column in features.columns},
            std={column: float(std[column]) for column in features.columns},
            train_feature_min={column: float(features[column].min()) for column in features.columns},
            train_feature_max={column: float(features[column].max()) for column in features.columns},
            mahalanobis_warning_threshold=threshold,
        )

    def save(self, path: Path = SETTINGS.models_dir / "ood_profile.json") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS.models_dir / "ood_profile.json") -> OODProfile:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_ood(
    frame: pd.DataFrame,
    schema_path: Path = SETTINGS.models_dir / "feature_schema.json",
    profile_path: Path = SETTINGS.models_dir / "ood_profile.json",
) -> dict[str, object]:
    schema = FeatureSchema.load(schema_path)
    profile = OODProfile.load(profile_path)
    engineered = add_derived_features(frame)
    for column in schema.feature_columns:
        if column not in engineered.columns:
            engineered[column] = 0.0
    features = engineered[schema.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    row = features.iloc[0]
    minmax_violations = []
    for column in profile.feature_columns:
        value = float(row[column])
        low = profile.train_feature_min[column]
        high = profile.train_feature_max[column]
        if value < low or value > high:
            minmax_violations.append(
                {"feature": column, "value": value, "train_min": low, "train_max": high}
            )
        mahalanobis_distance = float(
        np.sqrt(
            sum(
                ((float(row[column]) - profile.mean[column]) / profile.std[column]) ** 2
                for column in profile.feature_columns
            )
        )
    )
    return {
        "status": "warning"
        if minmax_violations or mahalanobis_distance > profile.mahalanobis_warning_threshold
        else "in_distribution",
        "mahalanobis_distance": mahalanobis_distance,
        "threshold": profile.mahalanobis_warning_threshold,
        "minmax_violations": minmax_violations,
    }
