"""Streamlit demo for the AI Studio Feasibility Copilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copilot.config import SETTINGS
from copilot.evaluate import load_model_and_predict
from copilot.feasibility import FeasibilityConfig, evaluate_rules, feasibility_status
from copilot.features import FeatureSchema
from copilot.geometry import build_envelope_scaffold, render_mesh_preview, resolve_stl_path
from copilot.ood import evaluate_ood
from copilot.search import search_candidates
from copilot.visualization import parameter_delta_chart, prediction_comparison, rule_table

EDITABLE_PARAMETERS = [
    {"column": "geo_param_vehicle_length", "label": "Vehicle length", "safe_pct": 0.05},
    {"column": "geo_param_vehicle_width", "label": "Vehicle width", "safe_pct": 0.03},
    {"column": "geo_param_vehicle_height", "label": "Vehicle height", "safe_pct": 0.04},
    {"column": "geo_param_front_overhang", "label": "Front overhang", "safe_pct": 0.05},
    {"column": "geo_param_rear_overhang", "label": "Rear overhang", "safe_pct": 0.05},
    {"column": "geo_param_vehicle_ride_height", "label": "Ride height", "safe_pct": 0.08},
    {"column": "geo_param_vehicle_pitch", "label": "Vehicle pitch", "safe_pct": 0.15},
    {"column": "geo_param_hood_angle", "label": "Hood angle", "safe_pct": 0.08},
]
MODEL_ONLY_COLUMNS = {"geo_param_run", "cd", "cl", "cs", "source_file_count", "run_id"}
STATUS_LABELS = {"pass": "PASS", "warning": "WARNING", "fail": "FAIL"}


@st.cache_data
def load_design_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@st.cache_data
def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_baseline(table: pd.DataFrame, run_id: int) -> pd.Series:
    return table[table["run_id"] == run_id].iloc[0].copy()


def reset_edited_design(baseline: pd.Series) -> None:
    st.session_state.edited_row = baseline.copy()
    st.session_state.last_result = None


def init_design_state(baseline: pd.Series) -> None:
    run_id = int(baseline["run_id"])
    if st.session_state.get("active_run_id") != run_id:
        st.session_state.active_run_id = run_id
        st.session_state.baseline_row = baseline.copy()
        reset_edited_design(baseline)


def safe_prediction(row: pd.Series) -> float | None:
    try:
        return float(load_model_and_predict(pd.DataFrame([row])).iloc[0])
    except Exception as exc:
        st.info(f"Prediction unavailable: {exc}")
        return None


def run_ood(row: pd.Series) -> dict[str, object] | None:
    if not (SETTINGS.models_dir / "ood_profile.json").exists():
        return None
    try:
        return evaluate_ood(pd.DataFrame([row]))
    except Exception as exc:
        return {"status": "unavailable", "message": str(exc)}


def run_shap(row: pd.Series, top_n: int = 3) -> pd.DataFrame | None:
    try:
        from copilot.explain import top_local_contributions

        return top_local_contributions(pd.DataFrame([row]), top_n=top_n)
    except Exception:
        return None


def evaluate_design(baseline: pd.Series, edited: pd.Series) -> dict[str, Any]:
    baseline_pred = safe_prediction(baseline)
    edited_pred = safe_prediction(edited)
    rules = evaluate_rules(edited, baseline, FeasibilityConfig.load())
    status = feasibility_status(rules)
    delta = None if baseline_pred is None or edited_pred is None else baseline_pred - edited_pred
    pct_delta = None if not baseline_pred or delta is None else (delta / baseline_pred) * 100.0
    result = {
        "baseline": baseline.copy(),
        "edited": edited.copy(),
        "baseline_predicted_cd": baseline_pred,
        "edited_predicted_cd": edited_pred,
        "cd_delta": delta,
        "pct_delta": pct_delta,
        "rule_results": rules,
        "feasibility_status": status,
        "ood": run_ood(edited),
        "shap": run_shap(edited, top_n=3),
    }
    st.session_state.last_result = result
    return result


def status_text(status: str) -> str:
    return STATUS_LABELS.get(status, status.upper())


def status_badge(status: str) -> None:
    st.markdown(f"**Status:** `{status_text(status)}`")


def parameter_bounds(
    baseline: pd.Series,
    column: str,
    safe_pct: float,
    schema: FeatureSchema | None,
    stay_safe: bool,
) -> tuple[float, float]:
    value = float(baseline[column])
    if stay_safe:
        if column in {"geo_param_vehicle_length", "geo_param_vehicle_width", "geo_param_vehicle_height"}:
            low = value * (1.0 - safe_pct)
            high = value * (1.0 + safe_pct)
            return min(low, high), max(low, high)
        if schema and column in schema.train_feature_min and column in schema.train_feature_max:
            return schema.train_feature_min[column], schema.train_feature_max[column]
    span = abs(value) * 0.25 if value else 1.0
    return value - span, value + span


def parameter_step(value: float) -> float:
    return max(abs(value) * 0.005, 0.001)


def parameter_delta_table(baseline: pd.Series, edited: pd.Series, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in baseline or column not in edited:
            continue
        base = float(baseline[column])
        new = float(edited[column])
        pct = None if base == 0 else ((new - base) / base) * 100.0
        rows.append(
            {
                "parameter": column.replace("geo_param_", "").replace("_", " "),
                "baseline": base,
                "edited": new,
                "delta": new - base,
                "delta_pct": pct,
            }
        )
    return pd.DataFrame(rows)


def display_source_files(row: pd.Series) -> None:
    if "source_files_json" in row and pd.notna(row["source_files_json"]):
        st.json(json.loads(row["source_files_json"]))
    else:
        st.write("No source file metadata available.")


def display_key_parameters(row: pd.Series, columns: list[str]) -> None:
    available = [column for column in columns if column in row.index]
    if available:
        st.dataframe(pd.DataFrame([row[available]]), width="stretch")
    else:
        st.info("No curated baseline parameters were found in this design table.")


def sorted_rule_frame(rules) -> pd.DataFrame:
    frame = rule_table(rules)
    order = {"fail": 0, "warning": 1, "info": 2}
    return frame.assign(_order=frame["severity"].map(order)).sort_values("_order").drop(columns="_order")


def human_result_sentence(result: dict[str, Any]) -> str:
    pct = result.get("pct_delta")
    status = result.get("feasibility_status", "warning")
    ood = result.get("ood") or {}
    ood_status = ood.get("status", "unknown")
    if pct is None:
        drag = "Drag change is unavailable."
    elif pct >= 0:
        drag = f"Predicted drag improved by {pct:.2f}%."
    else:
        drag = f"Predicted drag worsened by {abs(pct):.2f}%."
    return f"{drag} Feasibility is {status_text(status)}. OOD status: {ood_status}."


@st.cache_data(show_spinner="Rendering geometry preview…")
def cached_mesh_preview(stl_path_str: str, st_mtime_ns: int) -> bytes | None:
    """Cache raster preview by path and modification time (large meshes are slow to rasterize)."""
    rendered = render_mesh_preview(Path(stl_path_str))
    return rendered.read_bytes() if rendered else None


def main() -> None:
    st.set_page_config(page_title="AI Studio Feasibility Copilot", layout="wide")
    st.title("AI Studio Feasibility Copilot")
    st.caption(
        "Educational vehicle-shape exploration with an XGBoost drag surrogate and proxy package checks."
    )
    st.warning(
        "These are educational proxy rules. They are not OEM hardpoint checks, legal compliance checks, "
        "or production vehicle package validation."
    )

    data_path = st.sidebar.text_input(
        "Design table",
        value=str(SETTINGS.data_processed / "design_table.parquet"),
    )
    path = Path(data_path)
    if not path.exists():
        st.info("Run ingestion and training first, then reload this app.")
        st.code(
            "bash scripts/download_drivaerml.sh --aggregate\n"
            "python -m copilot.data_ingest --raw-dir data/raw/drivaerml --out data/processed/design_table.parquet\n"
            "python -m copilot.train_xgboost --data data/processed/design_table.parquet --target cd"
        )
        return

    table = load_design_table(path)
    schema = FeatureSchema.load() if (SETTINGS.models_dir / "feature_schema.json").exists() else None
    tabs = st.tabs(
        [
            "Choose a Baseline",
            "Edit Design",
            "Results",
            "Explore Alternatives",
            "Model & Data Details",
        ]
    )

    with tabs[0]:
        st.subheader("Choose a Baseline")
        run_id = st.selectbox("Run ID", table["run_id"].sort_values().tolist())
        baseline = get_baseline(table, int(run_id))
        init_design_state(baseline)
        edited = st.session_state.edited_row
        baseline_pred = safe_prediction(baseline)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Run ID", int(run_id))
        if "cd" in baseline:
            metric_cols[1].metric("Actual Cd", f"{float(baseline['cd']):.5f}")
        else:
            metric_cols[1].metric("Actual Cd", "N/A")
        metric_cols[2].metric(
            "Predicted Cd",
            "N/A" if baseline_pred is None else f"{baseline_pred:.5f}",
        )
        metric_cols[3].metric("Dataset", str(baseline.get("source_dataset", SETTINGS.dataset_name)))

        curated_cols = [param["column"] for param in EDITABLE_PARAMETERS]
        st.markdown("#### Key baseline parameters")
        display_key_parameters(baseline, curated_cols)

        st.markdown("#### Geometry preview")
        stl_path = resolve_stl_path(baseline)
        if stl_path:
            stat = stl_path.stat()
            png_bytes = cached_mesh_preview(str(stl_path.resolve()), int(stat.st_mtime_ns))
            if png_bytes:
                st.image(png_bytes, caption=str(stl_path))
            else:
                st.warning(
                    "STL was found but preview rendering failed (missing VTK/OpenGL or display). "
                    "Try `pip install pyglet` for the Trimesh renderer, or run Streamlit on a machine "
                    "with working OpenGL for PyVista."
                )
                st.caption(f"Path: `{stl_path}`")
        else:
            rid = int(run_id)
            st.info("No local STL found for this run.")
            with st.expander("How to download STL for geometry preview", expanded=False):
                st.markdown(
                    "Aggregate CSV ingestion does not bundle meshes. Download the mesh for this run ID from "
                    "Hugging Face into the repo’s raw data folder, then refresh the app (re-ingest is optional)."
                )
                st.code(
                    f"cd {ROOT}\n"
                    f"bash scripts/download_drivaerml.sh --per-run --with-stl "
                    f"--run-start {rid} --run-end {rid}",
                    language="bash",
                )
                st.caption(
                    f"Expected file: `data/raw/drivaerml/run_{rid}/drivaer_{rid}.stl` "
                    "(large; only fetch runs you need)."
                )

        with st.expander("Source files and metadata"):
            display_source_files(baseline)

    baseline = st.session_state.get("baseline_row")
    edited = st.session_state.get("edited_row")
    if baseline is None or edited is None:
        st.stop()

    with tabs[1]:
        st.subheader("Edit Design")
        stay_safe = st.toggle("Stay within safe range", value=True)
        st.caption("Edit a small set of understandable shape parameters, then evaluate the design.")

        active_columns: list[str] = []
        for param in EDITABLE_PARAMETERS:
            column = param["column"]
            if column not in edited.index or pd.isna(edited[column]):
                continue
            active_columns.append(column)
            label = param["label"]
            base_value = float(baseline[column])
            current = float(st.session_state.edited_row[column])
            low, high = parameter_bounds(baseline, column, param["safe_pct"], schema, stay_safe)
            step = parameter_step(current)
            st.markdown(f"**{label}**")
            # Slider is the single source of truth: a separate number_input retains its own Streamlit widget
            # state and was overwriting slider moves with stale values on Evaluate.
            cols = st.columns([3, 1, 1, 1])
            slider_value = cols[0].slider(
                f"{label} slider",
                min_value=float(low),
                max_value=float(high),
                value=float(min(max(current, low), high)),
                step=float(step),
                label_visibility="collapsed",
            )
            cols[1].markdown(f"`{slider_value:.6g}`")
            if cols[2].button("-1%", key=f"minus_{column}"):
                new_v = max(low, current * 0.99) if stay_safe else current * 0.99
                st.session_state.edited_row[column] = new_v
                st.rerun()
            if cols[3].button("+1%", key=f"plus_{column}"):
                new_v = min(high, current * 1.01) if stay_safe else current * 1.01
                st.session_state.edited_row[column] = new_v
                st.rerun()
            st.session_state.edited_row[column] = slider_value
            delta_pct = None if base_value == 0 else ((slider_value - base_value) / base_value) * 100.0
            st.caption(f"Baseline: {base_value:.4g} | Edited: {slider_value:.4g} | Delta: {delta_pct or 0:.2f}%")

        with st.expander("Advanced parameters"):
            advanced_cols = [
                column
                for column in table.select_dtypes(include="number").columns
                if column.startswith("geo_param_")
                and column not in active_columns
                and column not in MODEL_ONLY_COLUMNS
            ]
            for column in advanced_cols:
                if column not in edited.index or pd.isna(edited[column]):
                    continue
                value = float(st.session_state.edited_row[column])
                st.session_state.edited_row[column] = st.number_input(
                    column.replace("geo_param_", "").replace("_", " ").title(),
                    value=value,
                    step=parameter_step(value),
                    key=f"advanced_{column}",
                )

        st.markdown("#### Edited vs baseline")
        st.dataframe(parameter_delta_table(baseline, st.session_state.edited_row, active_columns), width="stretch")

        action_cols = st.columns([1, 1, 3])
        if action_cols[0].button("Reset to baseline"):
            reset_edited_design(baseline)
            st.rerun()
        if action_cols[1].button("Evaluate design", type="primary"):
            evaluate_design(baseline, st.session_state.edited_row)
            st.success("Design evaluated. Summary below; open **Results** for charts and full feasibility detail.")

        snapshot = st.session_state.get("last_result")
        if snapshot is not None:
            st.markdown("#### Latest evaluation (this session)")
            snap_cols = st.columns(4)
            e_cd = snapshot.get("edited_predicted_cd")
            b_cd = snapshot.get("baseline_predicted_cd")
            snap_cols[0].metric("Predicted Cd (edited)", "N/A" if e_cd is None else f"{e_cd:.5f}")
            snap_cols[1].metric("Predicted Cd (baseline)", "N/A" if b_cd is None else f"{b_cd:.5f}")
            snap_cols[2].metric("Cd delta", "N/A" if snapshot.get("cd_delta") is None else f"{snapshot['cd_delta']:.5f}")
            snap_cols[3].metric("Feasibility", status_text(snapshot.get("feasibility_status", "warning")))
            st.caption(human_result_sentence(snapshot))

    with tabs[2]:
        st.subheader("Results")
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Edit a design and click Evaluate design first.")
        else:
            top_cols = st.columns(5)
            edited_cd = result["edited_predicted_cd"]
            baseline_cd = result["baseline_predicted_cd"]
            top_cols[0].metric("Edited predicted Cd", "N/A" if edited_cd is None else f"{edited_cd:.5f}")
            top_cols[1].metric("Baseline predicted Cd", "N/A" if baseline_cd is None else f"{baseline_cd:.5f}")
            top_cols[2].metric("Cd delta", "N/A" if result["cd_delta"] is None else f"{result['cd_delta']:.5f}")
            top_cols[3].metric(
                "% improvement",
                "N/A" if result["pct_delta"] is None else f"{result['pct_delta']:.2f}%",
            )
            top_cols[4].metric("Feasibility", status_text(result["feasibility_status"]))
            st.info(human_result_sentence(result))

            st.markdown("### A. Performance")
            if baseline_cd is not None and edited_cd is not None:
                st.plotly_chart(prediction_comparison(baseline_cd, edited_cd), width="stretch")
            if "cd" in baseline:
                st.caption(f"Actual baseline Cd: {float(baseline['cd']):.5f}")

            st.markdown("### B. Feasibility")
            rules = result["rule_results"]
            counts = pd.Series([rule.severity for rule in rules]).value_counts()
            st.write(
                f"{counts.get('info', 0)} rules pass, "
                f"{counts.get('warning', 0)} warnings, "
                f"{counts.get('fail', 0)} failures."
            )
            st.dataframe(sorted_rule_frame(rules), width="stretch")

            ood = result.get("ood")
            if ood:
                st.markdown("### Training data range")
                st.write(ood)

            st.markdown("### C. Why the model thinks this")
            shap_frame = result.get("shap")
            if isinstance(shap_frame, pd.DataFrame) and not shap_frame.empty:
                st.caption("Top local SHAP signals. These are model explanation signals, not physical causality.")
                st.dataframe(shap_frame, width="stretch")
            else:
                st.info("SHAP explanation is unavailable in this environment.")

    with tabs[3]:
        st.subheader("Explore Alternatives")
        result = st.session_state.get("last_result")
        seed_row = result["edited"] if result else st.session_state.edited_row
        st.caption("Generate copilot suggestions near the edited design.")
        mode = st.radio(
            "Mode",
            ["Suggest nearby alternatives", "Optimize within bounds"],
            horizontal=True,
        )
        default_count = 300 if mode == "Suggest nearby alternatives" else 1000
        num_candidates = st.slider("Candidates to sample", 50, 2000, default_count, step=50)
        top_n = st.slider("Number of suggestions", 3, 5, 3)
        perturb_sigma = 0.02 if mode == "Suggest nearby alternatives" else 0.04
        if st.button("Suggest better nearby options", type="primary"):
            try:
                top = search_candidates(
                    table,
                    int(baseline["run_id"]),
                    num_candidates=num_candidates,
                    perturb_sigma=perturb_sigma,
                    top_n=top_n,
                    seed_row=seed_row,
                )
                for candidate in top:
                    st.markdown(f"### {candidate.candidate_id}")
                    status_badge(candidate.feasibility_status)
                    st.write(
                        {
                            "predicted_cd": candidate.predicted_cd,
                            "improvement_vs_current_design": candidate.estimated_cd_improvement,
                            "accepted": candidate.accepted,
                            "rejection_reasons": candidate.rejection_reasons,
                        }
                    )
                    st.plotly_chart(parameter_delta_chart(candidate.parameter_deltas), width="stretch")
                    st.download_button(
                        "Download candidate JSON",
                        candidate.model_dump_json(indent=2),
                        file_name=f"{candidate.candidate_id}.json",
                        mime="application/json",
                    )
                    if st.button(f"Export scaffold for {candidate.candidate_id}", key=f"scaffold_{candidate.candidate_id}"):
                        st.write(build_envelope_scaffold(candidate.parameters, candidate.candidate_id))
            except Exception as exc:
                st.error(f"Search unavailable: {exc}")

    with tabs[4]:
        st.subheader("Model & Data Details")
        st.markdown("#### Data pedigree")
        st.write(
            {
                "dataset_name": baseline.get("source_dataset", SETTINGS.dataset_name),
                "run_id": int(baseline["run_id"]),
                "model_version": str(SETTINGS.models_dir / "xgboost_cd_model.json"),
                "feature_schema_version": SETTINGS.feature_schema_version,
                "rule_set_version": SETTINGS.rule_set_version,
            }
        )
        with st.expander("Source files"):
            display_source_files(baseline)

        st.markdown("#### Training metrics")
        st.json(load_json_file(SETTINGS.models_dir / "metrics.json"))

        with st.expander("Feature schema"):
            st.json(load_json_file(SETTINGS.models_dir / "feature_schema.json"))

        with st.expander("OOD methodology"):
            st.write(
                "OOD status compares the edited design against the training feature envelope "
                "and a z-score distance threshold saved in `models/ood_profile.json`."
            )
            st.json(load_json_file(SETTINGS.models_dir / "ood_profile.json"))

        with st.expander("Feasibility rule definitions"):
            rule_path = SETTINGS.rule_config_path
            if rule_path.exists():
                st.code(rule_path.read_text(encoding="utf-8"), language="yaml")
            doc_path = SETTINGS.reports_dir / "feasibility_rules.md"
            if doc_path.exists():
                st.markdown(doc_path.read_text(encoding="utf-8"))

        with st.expander("Full SHAP details"):
            shap_frame = run_shap(st.session_state.edited_row, top_n=10)
            if isinstance(shap_frame, pd.DataFrame) and not shap_frame.empty:
                st.dataframe(shap_frame, width="stretch")
            else:
                st.info("SHAP details unavailable. Install optional SHAP dependencies and retrain if needed.")


if __name__ == "__main__":
    main()
