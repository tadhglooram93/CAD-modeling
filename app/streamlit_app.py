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
from copilot.evaluate import load_holdout_surrogate_rmse, prepare_features
from copilot.feasibility import FeasibilityConfig, evaluate_rules, feasibility_status
from copilot.features import FeatureSchema
from copilot.visualization import rule_table

EDITABLE_PARAMETERS = [
    {"column": "geo_param_vehicle_length", "label": "Vehicle length", "safe_pct": 0.05, "unit": "mm"},
    {"column": "geo_param_vehicle_width", "label": "Vehicle width", "safe_pct": 0.03, "unit": "mm"},
    {"column": "geo_param_vehicle_height", "label": "Vehicle height", "safe_pct": 0.04, "unit": "mm"},
    {"column": "geo_param_front_overhang", "label": "Front overhang", "safe_pct": 0.05, "unit": "mm"},
    {"column": "geo_param_rear_overhang", "label": "Rear overhang", "safe_pct": 0.05, "unit": "mm"},
    {"column": "geo_param_vehicle_ride_height", "label": "Ride height", "safe_pct": 0.08, "unit": "mm"},
    {"column": "geo_param_vehicle_pitch", "label": "Vehicle pitch", "safe_pct": 0.15, "unit": "deg"},
    {"column": "geo_param_hood_angle", "label": "Hood angle", "safe_pct": 0.08, "unit": "deg"},
]
MODEL_ONLY_COLUMNS = {"geo_param_run", "cd", "cl", "cs", "source_file_count", "run_id"}
STATUS_LABELS = {"pass": "PASS", "warning": "WARNING", "fail": "FAIL"}


def resolved_surrogate_model_path() -> Path:
    """Prefer ``.ubj``, fall back to legacy ``.json`` (same logic as ``evaluate.load_model``)."""
    primary = SETTINGS.xgboost_model_path
    if primary.exists():
        return primary.resolve()
    legacy = primary.with_suffix(".json")
    if legacy.exists():
        return legacy.resolve()
    return primary.resolve()


@st.cache_resource(show_spinner="Loading surrogate model…")
def cached_surrogate_model(model_path_str: str, mtime_ns: int) -> Any:
    """Single in-memory booster; cache key invalidates when the file changes."""
    from copilot.evaluate import load_model

    return load_model(Path(model_path_str))


@st.cache_data(show_spinner="Loading feature schema…")
def cached_feature_schema(schema_path_str: str, mtime_ns: int) -> FeatureSchema:
    return FeatureSchema.load(Path(schema_path_str))


@st.cache_data(show_spinner="Loading design table…")
def load_design_table(path_str: str, mtime_ns: int) -> pd.DataFrame:
    path = Path(path_str)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@st.cache_data
def cached_holdout_surrogate_rmse(metrics_path_str: str, mtime_ns: int) -> float | None:
    return load_holdout_surrogate_rmse(Path(metrics_path_str))


@st.cache_data
def load_json_file(path_str: str, mtime_ns: int) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_snapshot(path: Path) -> dict[str, Any]:
    """Thin wrapper so call sites stay readable; caching lives in ``load_json_file``."""
    return load_json_file(str(path.resolve()), int(path.stat().st_mtime_ns) if path.exists() else 0)


@st.cache_data
def load_bundled_vehicle_preview(path_str: str, mtime_ns: int) -> bytes:
    return Path(path_str).read_bytes()


def get_baseline(table: pd.DataFrame, run_id: int) -> pd.Series:
    return table[table["run_id"] == run_id].iloc[0].copy()


def reset_edited_design(baseline: pd.Series) -> None:
    st.session_state.edited_row = baseline.copy()
    st.session_state.last_result = None
    st.session_state.edit_widget_generation = st.session_state.get("edit_widget_generation", 0) + 1


def init_design_state(baseline: pd.Series) -> None:
    run_id = int(baseline["run_id"])
    if st.session_state.get("active_run_id") != run_id:
        st.session_state.active_run_id = run_id
        st.session_state.baseline_row = baseline.copy()
        reset_edited_design(baseline)


def safe_prediction(row: pd.Series) -> float | None:
    try:
        mp = resolved_surrogate_model_path()
        if not mp.exists():
            return None
        schema_path = SETTINGS.models_dir / "feature_schema.json"
        if not schema_path.exists():
            return None
        model = cached_surrogate_model(str(mp), int(mp.stat().st_mtime_ns))
        schema = cached_feature_schema(str(schema_path.resolve()), int(schema_path.stat().st_mtime_ns))
        features = prepare_features(pd.DataFrame([row]), schema)
        pred = model.predict(features)
        return float(pd.Series(pred).iloc[0])
    except Exception as exc:
        st.info(f"Prediction unavailable: {exc}")
        return None


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
    }
    st.session_state.last_result = result
    return result


def status_text(status: str) -> str:
    return STATUS_LABELS.get(status, status.upper())


def format_cd_band(value: float | None, band: float | None) -> str:
    if value is None:
        return "N/A"
    if band is None:
        return f"{value:.5f}"
    return f"{value:.5f} ± {band:.5f}"


def format_cd_point(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.5f}"


def holdout_error_caption(band: float | None) -> str:
    if band is None:
        return "Hold-out error (RMSE/MAE) not found in `models/metrics.json`."
    return (
        f"Approx. ± {band:.5f} from hold-out RMSE (or MAE if RMSE missing). "
        "Not a calibrated predictive interval."
    )


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
    return max(round(abs(value) * 0.005, 4), 0.01)


def geo_column_unit(column: str) -> str:
    """Linear geometry uses mm; pitch and hood angle use degrees in the UI."""
    if column in {"geo_param_vehicle_pitch", "geo_param_hood_angle"}:
        return "deg"
    return "mm"


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
                "unit": geo_column_unit(column),
                "baseline": round(base, 2),
                "edited": round(new, 2),
                "delta": round(new - base, 2),
                "delta_pct": None if pct is None else round(pct, 2),
            }
        )
    return pd.DataFrame(rows)


def display_source_files(row: pd.Series) -> None:
    if "source_files_json" in row and pd.notna(row["source_files_json"]):
        st.json(json.loads(row["source_files_json"]))
    else:
        st.write("No source file metadata available.")


def sorted_rule_frame(rules) -> pd.DataFrame:
    frame = rule_table(rules)
    order = {"fail": 0, "warning": 1, "info": 2}
    return frame.assign(_order=frame["severity"].map(order)).sort_values("_order").drop(columns="_order")


def human_result_sentence(result: dict[str, Any]) -> str:
    pct = result.get("pct_delta")
    status = result.get("feasibility_status", "warning")
    if pct is None:
        drag = "Drag change is unavailable."
    elif pct >= 0:
        drag = f"Predicted drag improved by {pct:.2f}%."
    else:
        drag = f"Predicted drag worsened by {abs(pct):.2f}%."
    return f"{drag} Feasibility is {status_text(status)}."


def render_feasibility_verdict(status: str) -> None:
    label = status_text(status)
    if status == "pass":
        st.success(f"**Feasibility:** {label}")
    elif status == "fail":
        st.error(f"**Feasibility:** {label}")
    else:
        st.warning(f"**Feasibility:** {label}")


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

    path = SETTINGS.data_processed / "design_table.parquet"
    st.sidebar.info(
        "10 representative DrivAerML vehicle-geometry variants were curated for this demo. "
        "These are generic, open automotive aerodynamics geometries."
    )
    if not path.exists():
        st.info("Run ingestion and training first, then reload this app.")
        st.code(
            "bash scripts/download_drivaerml.sh --aggregate\n"
            "python -m copilot.data_ingest --raw-dir data/raw/drivaerml --out data/processed/design_table.parquet\n"
            "python -m copilot.train_xgboost --data data/processed/design_table.parquet --target cd"
        )
        return

    table = load_design_table(str(path.resolve()), int(path.stat().st_mtime_ns))
    allowed = set(SETTINGS.demo_run_ids)
    present_ids = {int(x) for x in table["run_id"].unique().tolist()}
    run_options = sorted(present_ids & allowed)
    if not run_options:
        st.error(
            "No demo run IDs are present in this design table. "
            f"Ingest data that includes at least one of: {sorted(allowed)}."
        )
        return
    missing_demo = sorted(allowed - present_ids)
    if missing_demo:
        st.sidebar.caption(f"Runs not in table (skipped): {', '.join(str(x) for x in missing_demo)}")
    run_id = st.sidebar.selectbox("Run ID", run_options, format_func=lambda x: str(int(x)))
    baseline = get_baseline(table, int(run_id))
    init_design_state(baseline)
    baseline = st.session_state.baseline_row
    metrics_path = SETTINGS.models_dir / "metrics.json"
    metrics_mtime = int(metrics_path.stat().st_mtime_ns) if metrics_path.exists() else 0
    holdout_rmse = cached_holdout_surrogate_rmse(str(metrics_path.resolve()), metrics_mtime)

    schema_path = SETTINGS.models_dir / "feature_schema.json"
    schema = None
    if schema_path.exists():
        schema = cached_feature_schema(str(schema_path.resolve()), int(schema_path.stat().st_mtime_ns))

    tabs = st.tabs(["Dashboard", "Model & Data Details"])

    with tabs[0]:
        st.subheader("Baseline summary")
        baseline_pred = safe_prediction(baseline)
        geo_col, summary_col = st.columns([1, 1])
        with geo_col:
            rid = int(baseline["run_id"])
            preview_png = SETTINGS.demo_vehicle_preview_dir / f"run_{rid}.png"
            if preview_png.exists():
                mtime = int(preview_png.stat().st_mtime_ns)
                st.image(
                    load_bundled_vehicle_preview(str(preview_png.resolve()), mtime),
                    caption="DrivAerML-style geometry (pre-rendered for this demo)",
                )
            else:
                st.info(
                    f"No bundled preview image for run **{rid}**. From the repo root, run "
                    f"`python scripts/export_demo_vehicle_previews.py` (generates `assets/demo_vehicle_previews/run_{rid}.png` "
                    "from the HF STLs) and commit the PNGs for Streamlit Cloud."
                )

        with summary_col:
            actual_cd = float(baseline["cd"]) if "cd" in baseline and pd.notna(baseline.get("cd")) else None
            summary_row = {
                "run_id": int(baseline["run_id"]),
                "actual_cd": actual_cd if actual_cd is not None else "N/A",
                "predicted_cd": format_cd_band(baseline_pred, holdout_rmse),
                "dataset": str(baseline.get("source_dataset", SETTINGS.dataset_name)),
            }
            st.dataframe(pd.DataFrame([summary_row]), width="stretch", hide_index=True)
            st.caption(
                "The ± value is an approximate band from hold-out error (RMSE or MAE) on the surrogate; "
                "it is not a calibrated predictive interval."
            )

        st.divider()
        st.subheader("Edit design")
        st.caption("Adjust sliders; the comparison table updates as you move them.")

        stay_safe = st.toggle("Stay within safe range", value=True)

        edit_left, edit_right = st.columns([1.1, 1])
        active_columns: list[str] = []
        widget_gen = int(st.session_state.get("edit_widget_generation", 0))
        run_key = int(st.session_state.active_run_id)

        def render_param_slider(param: dict[str, Any]) -> None:
            column = param["column"]
            er = st.session_state.edited_row
            if column not in er.index or pd.isna(er[column]):
                return
            active_columns.append(column)
            label = param["label"]
            unit = param.get("unit") or geo_column_unit(column)
            base_value = float(baseline[column])
            current = round(float(er[column]), 2)
            low, high = parameter_bounds(baseline, column, param["safe_pct"], schema, stay_safe)
            low, high = round(float(low), 2), round(float(high), 2)
            step = parameter_step(current)
            st.markdown(f"**{label} Delta from Baseline ({unit}) :**")
            s_col, v_col = st.columns([4, 1])
            with s_col:
                slider_value = st.slider(
                    f"{label} slider",
                    min_value=float(low),
                    max_value=float(high),
                    value=float(min(max(current, low), high)),
                    step=float(step),
                    label_visibility="collapsed",
                    key=f"sl_{column}_{run_key}_{widget_gen}",
                )
            slider_value = round(float(slider_value), 2)
            with v_col:
                st.markdown(f"`{slider_value:.2f}`")
            st.session_state.edited_row[column] = slider_value
            delta_mm = round(slider_value - base_value, 2)
            delta_pct = None if base_value == 0 else round(((slider_value - base_value) / base_value) * 100.0, 2)
            rel_txt = f"{delta_pct:.2f}%" if delta_pct is not None else "N/A"
            st.caption(
                f"Baseline {base_value:.2f} {unit} · absolute Δ {delta_mm:+.2f} {unit} · relative Δ {rel_txt}"
            )

        with edit_left:
            inner_l, inner_r = st.columns(2)
            with inner_l:
                for param in EDITABLE_PARAMETERS[:4]:
                    render_param_slider(param)
            with inner_r:
                for param in EDITABLE_PARAMETERS[4:]:
                    render_param_slider(param)
            with st.expander("Advanced parameters", expanded=False):
                advanced_cols = [
                    column
                    for column in table.select_dtypes(include="number").columns
                    if column.startswith("geo_param_")
                    and column not in active_columns
                    and column not in MODEL_ONLY_COLUMNS
                ]
                for column in advanced_cols:
                    if column not in st.session_state.edited_row.index or pd.isna(st.session_state.edited_row[column]):
                        continue
                    value = round(float(st.session_state.edited_row[column]), 2)
                    u = geo_column_unit(column)
                    nice = column.replace("geo_param_", "").replace("_", " ").title()
                    st.markdown(f"**{nice} Delta from Baseline ({u}) :**")
                    st.session_state.edited_row[column] = round(
                        float(
                            st.number_input(
                                f"{nice} value",
                                value=value,
                                step=parameter_step(value),
                                format="%.2f",
                                label_visibility="collapsed",
                                key=f"advanced_{column}_{run_key}",
                            )
                        ),
                        2,
                    )

        with edit_right:
            st.markdown("#### Edited vs baseline")
            st.dataframe(
                parameter_delta_table(baseline, st.session_state.edited_row, active_columns),
                width="stretch",
            )
            _, action_mid, _ = st.columns([1, 3, 1])
            with action_mid:
                r_btn, e_btn = st.columns(2)
                with r_btn:
                    if st.button("Reset to baseline", use_container_width=True, key="reset_baseline_btn"):
                        reset_edited_design(baseline)
                        st.rerun()
                with e_btn:
                    if st.button("Evaluate design", type="primary", use_container_width=True, key="evaluate_design_btn"):
                        evaluate_design(baseline, st.session_state.edited_row)
                        st.success("Design evaluated. Results appear in the section below.")

        st.divider()
        st.subheader("Results")
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Run **Evaluate design** to see feasibility and predicted drag for your edits.")
        else:
            render_feasibility_verdict(result.get("feasibility_status", "warning"))
            top_cols = st.columns(5)
            edited_cd = result["edited_predicted_cd"]
            baseline_cd = result["baseline_predicted_cd"]
            with top_cols[0]:
                st.metric("Edited predicted Cd", format_cd_point(edited_cd))
                st.caption(holdout_error_caption(holdout_rmse))
            with top_cols[1]:
                st.metric("Baseline predicted Cd", format_cd_point(baseline_cd))
                st.caption(holdout_error_caption(holdout_rmse))
            top_cols[2].metric("Cd delta", "N/A" if result["cd_delta"] is None else f"{result['cd_delta']:.5f}")
            top_cols[3].metric(
                "% improvement",
                "N/A" if result["pct_delta"] is None else f"{result['pct_delta']:.2f}%",
            )
            top_cols[4].metric("Feasibility", status_text(result["feasibility_status"]))
            st.caption(human_result_sentence(result))

            st.markdown("### Feasibility")
            rules = result["rule_results"]
            counts = pd.Series([rule.severity for rule in rules]).value_counts()
            st.write(
                f"{counts.get('info', 0)} rules pass, "
                f"{counts.get('warning', 0)} warnings, "
                f"{counts.get('fail', 0)} failures."
            )
            st.dataframe(sorted_rule_frame(rules), width="stretch")

    with tabs[1]:
        st.subheader("Model & Data Details")
        st.markdown("#### Data pedigree")
        st.write(
            {
                "dataset_name": baseline.get("source_dataset", SETTINGS.dataset_name),
                "run_id": int(baseline["run_id"]),
                "model_version": str(SETTINGS.xgboost_model_path),
                "feature_schema_version": SETTINGS.feature_schema_version,
                "rule_set_version": SETTINGS.rule_set_version,
            }
        )
        with st.expander("Source files"):
            display_source_files(baseline)

        st.markdown("#### Training metrics")
        st.json(json_snapshot(SETTINGS.models_dir / "metrics.json"))

        with st.expander("Feature schema"):
            st.json(json_snapshot(SETTINGS.models_dir / "feature_schema.json"))

        with st.expander("OOD methodology"):
            st.write(
                "OOD status compares the edited design against the training feature envelope "
                "and a z-score distance threshold saved in `models/ood_profile.json`."
            )
            st.json(json_snapshot(SETTINGS.models_dir / "ood_profile.json"))

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
