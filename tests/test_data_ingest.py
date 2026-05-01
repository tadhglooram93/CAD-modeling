from pathlib import Path

import pandas as pd

from copilot.data_ingest import build_design_table, ingest

FIXTURE = Path(__file__).parent / "fixtures" / "fake_drivaerml"


def test_build_design_table_from_run_folder() -> None:
    table = build_design_table(FIXTURE)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["run_id"] == 1
    assert row["geo_param_length"] == 4.6
    assert row["geo_ref_reference_area"] == 2.11
    assert row["force_constref_cd"] == 0.299
    assert "source_files_json" in table.columns


def test_ingest_writes_outputs(tmp_path: Path) -> None:
    out = tmp_path / "processed" / "design_table.parquet"
    table = ingest(FIXTURE, out)

    assert out.exists()
    assert out.with_suffix(".csv").exists()
    assert out.with_name("data_manifest.json").exists()
    assert pd.read_parquet(out).shape == table.shape
    assert "cd" in table.columns
