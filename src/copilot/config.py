"""Shared path and run configuration for the studio-feasibility workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    candidates_dir: Path = PROJECT_ROOT / "artifacts" / "candidates"
    geometry_exports_dir: Path = PROJECT_ROOT / "artifacts" / "geometry_exports"
    rule_config_path: Path = PROJECT_ROOT / "configs" / "feasibility_rules.yaml"
    dataset_name: str = "DrivAerML"
    dataset_version: str = "neashton/drivaerml-main"
    feature_schema_version: str = "feature-schema-v1"
    rule_set_version: str = "proxy-rules-v1"
    random_state: int = 42

    def ensure_directories(self) -> None:
        for path in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.models_dir,
            self.figures_dir,
            self.artifacts_dir,
            self.candidates_dir,
            self.geometry_exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
