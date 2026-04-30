"""Build a traceable tabular design table from selected DrivAerML files."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from copilot.config import SETTINGS
from copilot.lineage import ArtifactRef, sha256_file, utc_now_iso

LOGGER = logging.getLogger(__name__)


def normalize_name(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or "unnamed"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts: dict[str, int] = {}
    names: list[str] = []
    for column in result.columns:
        base = normalize_name(column)
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    result.columns = names
    return result


def detect_run_column(frame: pd.DataFrame) -> str | None:
    candidates = ("run_id", "run", "case", "case_id", "design", "design_id", "id")
    for column in frame.columns:
        if column in candidates:
            return column
    for column in frame.columns:
        if "run" in column or "design" in column or column.endswith("_id"):
            return column
    return None


def coerce_run_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def read_csv_normalized(path: Path) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(path))


def flatten_per_run_csv(path: Path) -> dict[str, object]:
    frame = read_csv_normalized(path)
    if frame.empty:
        return {}

    if frame.shape[1] >= 2:
        first, second = frame.columns[:2]
        first_values = frame[first].astype(str).map(str.strip)
        if first_values.is_unique and frame.shape[0] > 1:
            return {normalize_name(key): value for key, value in zip(frame[first], frame[second], strict=False)}

    row = frame.iloc[0].to_dict()
    return {normalize_name(key): value for key, value in row.items()}


def _prefix_columns(frame: pd.DataFrame, prefix: str, keep: Iterable[str]) -> pd.DataFrame:
    keep_set = set(keep)
    renamed = {
        column: column if column in keep_set or column.startswith(prefix) else f"{prefix}{column}"
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def _read_aggregate(raw_dir: Path) -> pd.DataFrame | None:
    aggregate_files = {
        "geo_param_": raw_dir / "geo_parameters_all.csv",
        "geo_ref_": raw_dir / "geo_ref_all.csv",
        "force_": raw_dir / "force_mom_all.csv",
        "force_constref_": raw_dir / "force_mom_constref_all.csv",
    }
    existing = {prefix: path for prefix, path in aggregate_files.items() if path.exists()}
    if not existing:
        return None

    merged: pd.DataFrame | None = None
    source_refs: list[ArtifactRef] = []
    for prefix, path in existing.items():
        frame = read_csv_normalized(path)
        run_column = detect_run_column(frame)
        if run_column is None:
            frame.insert(0, "run_id", range(1, len(frame) + 1))
        else:
            frame["run_id"] = frame[run_column].map(coerce_run_id)
        frame = frame.dropna(subset=["run_id"]).copy()
        frame["run_id"] = frame["run_id"].astype(int)
        frame = _prefix_columns(frame, prefix, keep=("run_id",))
        source_refs.append(ArtifactRef.from_path(path, SETTINGS.project_root))
        merged = frame if merged is None else merged.merge(frame, on="run_id", how="outer")

    if merged is None:
        return None

    merged["source_dataset"] = SETTINGS.dataset_name
    merged["dataset_version"] = SETTINGS.dataset_version
    merged["ingested_at"] = utc_now_iso()
    merged["source_file_count"] = len(source_refs)
    merged["source_files_json"] = json.dumps([ref.model_dump() for ref in source_refs])
    return merged.sort_values("run_id").reset_index(drop=True)


def scan_runs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("run_*") if path.is_dir())


def _read_run_folder(run_dir: Path) -> dict[str, object] | None:
    run_id = coerce_run_id(run_dir.name)
    if run_id is None:
        LOGGER.warning("Skipping run folder without numeric id: %s", run_dir)
        return None

    row: dict[str, object] = {
        "run_id": run_id,
        "source_dataset": SETTINGS.dataset_name,
        "dataset_version": SETTINGS.dataset_version,
        "ingested_at": utc_now_iso(),
    }
    source_refs: list[ArtifactRef] = []
    patterns = {
        "geo_param_": f"geo_parameters_{run_id}.csv",
        "geo_ref_": f"geo_ref_{run_id}.csv",
        "force_": f"force_mom_{run_id}.csv",
        "force_constref_": f"force_mom_constref_{run_id}.csv",
    }
    for prefix, file_name in patterns.items():
        path = run_dir / file_name
        if not path.exists():
            LOGGER.warning("Missing optional DrivAerML file: %s", path)
            continue
        for key, value in flatten_per_run_csv(path).items():
            if key in {"run_id", "run"}:
                continue
            row[f"{prefix}{key}"] = value
        source_refs.append(ArtifactRef.from_path(path, SETTINGS.project_root))

    stl_path = run_dir / f"drivaer_{run_id}.stl"
    if stl_path.exists():
        row["stl_path"] = str(stl_path.relative_to(SETTINGS.project_root))
        source_refs.append(ArtifactRef.from_path(stl_path, SETTINGS.project_root, hash_file=False))

    if len(source_refs) == 0:
        LOGGER.warning("Skipping run %s because no expected files were found.", run_id)
        return None

    row["source_file_count"] = len(source_refs)
    row["source_files_json"] = json.dumps([ref.model_dump() for ref in source_refs])
    return row


def build_design_table(raw_dir: Path) -> pd.DataFrame:
    aggregate = _read_aggregate(raw_dir)
    if aggregate is not None:
        return aggregate

    rows = [row for run_dir in scan_runs(raw_dir) if (row := _read_run_folder(run_dir)) is not None]
    if not rows:
        raise FileNotFoundError(f"No aggregate CSVs or run_* folders found under {raw_dir}")
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def _target_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in list(result.columns):
        short = column.split("_")[-1]
        if short in {"cd", "cl", "cs", "cml", "cmm", "cmn"} and short not in result.columns:
            result[short] = result[column]
    return result


def validate_training_target(table: pd.DataFrame, raw_dir: Path) -> None:
    if "cd" in table.columns:
        return
    force_columns = [column for column in table.columns if column.startswith(("force_", "force_constref_"))]
    if force_columns:
        return
    raise ValueError(
        "No drag target column was found after ingestion. "
        "The processed table needs force/moment data such as "
        "`force_mom_all.csv`, `force_mom_constref_all.csv`, or per-run "
        "`force_mom_<id>.csv` files. "
        f"Current raw directory only produced columns from: {raw_dir}"
    )


def write_manifest(table: pd.DataFrame, out_path: Path, raw_dir: Path) -> Path:
    manifest_path = out_path.with_name("data_manifest.json")
    source_paths = sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix != ".stl")
    manifest = {
        "dataset": SETTINGS.dataset_name,
        "dataset_version": SETTINGS.dataset_version,
        "created_at": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "row_count": int(len(table)),
        "columns": list(table.columns),
        "source_files": [
            {
                "path": str(path.relative_to(SETTINGS.project_root))
                if path.is_relative_to(SETTINGS.project_root)
                else str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in source_paths
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def ingest(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    SETTINGS.ensure_directories()
    table = _target_aliases(build_design_table(raw_dir))
    validate_training_target(table, raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    table.to_csv(out_path.with_suffix(".csv"), index=False)
    write_manifest(table, out_path, raw_dir)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=SETTINGS.data_raw / "drivaerml")
    parser.add_argument("--out", type=Path, default=SETTINGS.data_processed / "design_table.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    table = ingest(args.raw_dir, args.out)
    LOGGER.info("Wrote %s rows and %s columns to %s", len(table), len(table.columns), args.out)


if __name__ == "__main__":
    main()
