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
    rule_config_path: Path = PROJECT_ROOT / "configs" / "feasibility_rules.yaml"
    dataset_name: str = "DrivAerML"
    dataset_version: str = "neashton/drivaerml-main"
    feature_schema_version: str = "feature-schema-v1"
    rule_set_version: str = "proxy-rules-v1"
    random_state: int = 42
    hf_stl_assets_repo: str = "Kusya1/ai-studio-feasibility-copilot-asset"
    demo_run_ids: tuple[int, ...] = (1, 42, 100, 150, 200, 250, 300, 350, 400, 450)
    demo_vehicle_preview_dir: Path = PROJECT_ROOT / "assets" / "demo_vehicle_previews"

    @property
    def xgboost_model_path(self) -> Path:
        """Saved XGBoost booster (UBJSON binary via ``save_model``)."""
        return self.models_dir / "xgboost_cd_model.ubj"

    def ensure_directories(self) -> None:
        for path in (
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.models_dir,
            self.figures_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
