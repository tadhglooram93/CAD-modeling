"""Geometry helpers for STL visualization and simplified CAD scaffold export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from copilot.config import SETTINGS


def resolve_stl_path(row: pd.Series | dict[str, object]) -> Path | None:
    value = dict(row).get("stl_path")
    if not value or pd.isna(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = SETTINGS.project_root / path
    return path if path.exists() else None


def load_mesh(path: Path):
    import trimesh

    return trimesh.load_mesh(path)


def render_mesh_preview(path: Path, out_path: Path | None = None) -> Path | None:
    try:
        mesh = load_mesh(path)
        scene = mesh.scene()
        png = scene.save_image(resolution=(900, 600))
    except Exception:
        return None
    if png is None:
        return None
    out = out_path or (SETTINGS.artifacts_dir / "screenshots" / f"{path.stem}_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return out


def build_envelope_scaffold(parameters: dict[str, float], candidate_id: str) -> dict[str, str]:
    """Create a simplified CadQuery vehicle envelope and export STEP/STL when available."""
    out_base = SETTINGS.geometry_exports_dir / candidate_id
    out_base.parent.mkdir(parents=True, exist_ok=True)
    length = float(parameters.get("geo_param_length", parameters.get("length", 4.5)))
    width = float(parameters.get("geo_param_width", parameters.get("width", 1.8)))
    height = float(parameters.get("geo_param_height", parameters.get("height", 1.4)))
    try:
        import cadquery as cq

        body = cq.Workplane("XY").box(length, width, height)
        step_path = out_base.with_suffix(".step")
        stl_path = out_base.with_suffix(".stl")
        cq.exporters.export(body, str(step_path))
        cq.exporters.export(body, str(stl_path))
        return {"step": str(step_path), "stl": str(stl_path)}
    except Exception:
        metadata_path = out_base.with_suffix(".json")
        metadata_path.write_text(
            (
                "{\n"
                f'  "candidate_id": "{candidate_id}",\n'
                f'  "length": {length},\n'
                f'  "width": {width},\n'
                f'  "height": {height},\n'
                '  "note": "CadQuery export unavailable; this metadata describes the envelope scaffold."\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        return {"metadata": str(metadata_path)}
