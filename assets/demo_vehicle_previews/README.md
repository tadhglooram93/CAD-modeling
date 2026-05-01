# Bundled vehicle previews (Streamlit)

The deployed demo loads **pre-rendered PNGs** from this folder instead of downloading STL files or running VTK/OpenGL at runtime.

**Expected files** (one per curated demo `run_id` in `SETTINGS.demo_run_ids`):

- `run_1.png`, `run_42.png`, `run_100.png`, …

Generate them locally (needs Hugging Face network access once), then **commit the PNGs**. The script uses ``copilot.geometry.render_mesh_preview`` (**Trimesh** then **PyVista**; no Matplotlib). Run from your normal shell (not a sandboxed environment); GPU/OpenGL stacks may abort if restricted.

```bash
python scripts/export_demo_vehicle_previews.py
```

After that, Streamlit Community Cloud serves images straight from the repo with no mesh rendering pipeline.
