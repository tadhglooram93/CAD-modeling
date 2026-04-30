"""Geometry helpers for STL visualization and simplified CAD scaffold export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from copilot.config import SETTINGS


def resolve_stl_path(row: pd.Series | dict[str, object]) -> Path | None:
    """Resolve STL path from ingested metadata, or conventional per-run download layout."""
    data = dict(row)
    value = data.get("stl_path")
    if value and not pd.isna(value):
        path = Path(str(value))
        if not path.is_absolute():
            path = SETTINGS.project_root / path
        if path.exists():
            return path

    run_key = data.get("run_id")
    if run_key is None or pd.isna(run_key):
        return None
    run_id = int(run_key)
    conventional = SETTINGS.data_raw / "drivaerml" / f"run_{run_id}" / f"drivaer_{run_id}.stl"
    return conventional if conventional.exists() else None


def load_mesh(path: Path):
    import trimesh

    return trimesh.load_mesh(path)


def render_mesh_preview(path: Path, out_path: Path | None = None) -> Path | None:
    """Raster preview for Streamlit. Trimesh uses pyglet when available; otherwise VTK via PyVista."""
    out = out_path or (SETTINGS.artifacts_dir / "screenshots" / f"{path.stem}_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = load_mesh(path)
        scene = mesh.scene()
        png = scene.save_image(resolution=(900, 600))
        if png:
            out.write_bytes(png)
            return out
    except Exception:
        pass

    try:
        import pyvista as pv

        pv_mesh = pv.read(str(path))
        plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        plotter.set_background("white")
        plotter.add_mesh(pv_mesh, color="#c0c0c0", smooth_shading=True)
        plotter.reset_camera()
        plotter.camera.zoom(1.05)
        plotter.screenshot(str(out))
        plotter.close()
        return out if out.exists() and out.stat().st_size > 0 else None
    except Exception:
        return None


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
