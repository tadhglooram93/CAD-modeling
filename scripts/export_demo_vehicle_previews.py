#!/usr/bin/env python3
"""Pre-render demo vehicle meshes to PNG for Streamlit Cloud (same pipeline as local STL preview).

Downloads each demo STL from Hugging Face, runs ``copilot.geometry.render_mesh_preview`` — **Trimesh
scene (pyglet)**, then **PyVista** if needed. No Matplotlib.

Commit the PNGs under ``assets/demo_vehicle_previews/`` so deployment serves static images only.

Requires: network for HF download; optional ``HF_TOKEN`` for gated datasets. Run from a normal
terminal on your machine (not a restricted sandbox): VTK/Pyglet need a working GL stack.

Usage (from repo root):

  python scripts/export_demo_vehicle_previews.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from huggingface_hub import hf_hub_download

from copilot.config import SETTINGS
from copilot.geometry import render_mesh_preview

# Bundled previews: slightly sharper than default Streamlit preview resolution.
_EXPORT_RESOLUTION = (1200, 800)


def main() -> int:
    out_dir = SETTINGS.demo_vehicle_preview_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for run_id in SETTINGS.demo_run_ids:
        dest_png = out_dir / f"run_{run_id}.png"
        try:
            stl_path = hf_hub_download(
                repo_id=SETTINGS.hf_stl_assets_repo,
                repo_type="dataset",
                filename=f"run_{run_id}/drivaer_{run_id}.stl",
            )
        except Exception as exc:
            print(f"[skip] run_{run_id}: HF download failed: {exc}")
            continue
        rendered = render_mesh_preview(
            Path(stl_path), dest_png, resolution=_EXPORT_RESOLUTION
        )
        if rendered and rendered.exists():
            print(f"[ ok ] {rendered}")
            ok += 1
        else:
            print(
                f"[fail] run_{run_id}: render_mesh_preview failed "
                "(try outside sandbox; ensure trimesh + pyvista/VTK work)"
            )
    total = len(SETTINGS.demo_run_ids)
    print(f"Done. Wrote {ok}/{total} previews under {out_dir}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
