"""Geometry helpers for STL visualization."""

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


def render_mesh_preview(
    path: Path,
    out_path: Path | None = None,
    *,
    resolution: tuple[int, int] = (900, 600),
) -> Path | None:
    """Raster preview for Streamlit.

    Trimesh uses pyglet when available; otherwise VTK via PyVista. No Matplotlib fallback.

    PyVista uses ``pv.wrap(trimesh_mesh)`` so STL does not depend on PyVista's optional meshio reader.
    """
    out = out_path or (SETTINGS.artifacts_dir / "screenshots" / f"{path.stem}_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        mesh = load_mesh(path)
    except Exception:
        return None

    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=resolution)
        if png:
            out.write_bytes(png)
            return out
    except Exception:
        pass

    try:
        import pyvista as pv

        pv_mesh = pv.wrap(mesh)
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.set_background("white")
        plotter.add_mesh(pv_mesh, color="#c0c0c0", smooth_shading=True)
        plotter.reset_camera()
        plotter.camera.zoom(1.05)
        plotter.screenshot(str(out))
        plotter.close()
        return out if out.exists() and out.stat().st_size > 0 else None
    except Exception:
        return None
