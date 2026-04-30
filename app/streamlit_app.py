"""Streamlit demo for the AI Studio Feasibility Copilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from copilot.config import SETTINGS

from copilot.evaluate import load_model_and_predict
from copilot.feasibility import FeasibilityConfig, evaluate_rules
from copilot.features import FeatureSchema
from copilot.geometry import build_envelope_scaffold, render_mesh_preview, resolve_stl_path
from copilot.ood import evaluate_ood
from copilot.search import search_candidates
from copilot.visualization import parameter_delta_chart, prediction_comparison, rule_table


@st.cache_data
def load_design_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


@st.cache_data
def cached_predictions(frame: pd.DataFrame) -> pd.Series:
    return load_model_and_predict(frame)


def ood_warnings(row: pd.Series, schema: FeatureSchema) -> list[str]:
    warnings = []
    for column in schema.feature_columns:
        if column not in row:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        low = schema.train_feature_min.get(column)
        high = schema.train_feature_max.get(column)
        if low is not None and high is not None and (value < low or value > high):
            warnings.append(f"{column}: {value:.4g} outside train range {low:.4g} to {high:.4g}")
    return warnings


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
    run_id = st.sidebar.selectbox("Baseline run id", table["run_id"].sort_values().tolist())
    baseline = table[table["run_id"] == run_id].iloc[0]
    tabs = st.tabs(
        [
            "Dataset / Baseline",
            "Surrogate Prediction",
            "Feasibility Checks",
            "Design Alternatives",
            "Data Pedigree",
        ]
    )

    with tabs[0]:
        st.subheader(f"Baseline run {run_id}")
        left, right = st.columns([2, 1])
        with left:
            display_cols = [
                column
                for column in table.columns
                if column.startswith(("geo_param_", "geo_ref_")) or column in {"run_id", "cd"}
            ]
            st.dataframe(pd.DataFrame([baseline[display_cols]]), use_container_width=True)
        with right:
            st.write("Source files")
            if "source_files_json" in baseline and pd.notna(baseline["source_files_json"]):
                st.json(json.loads(baseline["source_files_json"]))
            else:
                st.write("No source file metadata available.")
        stl_path = resolve_stl_path(baseline)
        if stl_path:
            preview = render_mesh_preview(stl_path)
            if preview:
                st.image(str(preview), caption=str(stl_path))
            else:
                st.write(f"STL available at `{stl_path}`, but no preview renderer was available.")
        else:
            st.write("No local STL found for this run.")

    with tabs[1]:
        st.subheader("Surrogate prediction")
        try:
            prediction = float(load_model_and_predict(pd.DataFrame([baseline])).iloc[0])
            st.metric("Predicted Cd", f"{prediction:.5f}")
            if "cd" in baseline:
                st.metric("Actual Cd", f"{float(baseline['cd']):.5f}")
            schema = FeatureSchema.load()
            warnings = ood_warnings(baseline, schema)
            if warnings:
                st.warning("Candidate is outside the training feature envelope.")
                st.write(warnings)
            else:
                st.success("No simple min/max out-of-distribution warnings.")
            if (SETTINGS.models_dir / "ood_profile.json").exists():
                ood = evaluate_ood(pd.DataFrame([baseline]))
                st.write("OOD profile", ood)
            with st.expander("SHAP local explanation"):
                try:
                    from copilot.explain import top_local_contributions

                    st.dataframe(top_local_contributions(pd.DataFrame([baseline])), use_container_width=True)
                except Exception as shap_exc:
                    st.info(f"SHAP explanation unavailable: {shap_exc}")
        except Exception as exc:
            st.error(f"Prediction unavailable: {exc}")

    with tabs[2]:
        st.subheader("Feasibility checks")
        rules = evaluate_rules(baseline, baseline, FeasibilityConfig.load())
        st.dataframe(rule_table(rules), use_container_width=True)

    with tabs[3]:
        st.subheader("Design alternatives")
        num_candidates = st.slider("Candidates to sample", 50, 2000, 300, step=50)
        if st.button("Generate alternatives"):
            try:
                top = search_candidates(table, int(run_id), num_candidates=num_candidates, top_n=3)
                for candidate in top:
                    st.markdown(f"### {candidate.candidate_id}")
                    st.plotly_chart(
                        prediction_comparison(candidate.baseline_predicted_cd, candidate.predicted_cd),
                        use_container_width=True,
                    )
                    st.plotly_chart(parameter_delta_chart(candidate.parameter_deltas), use_container_width=True)
                    st.write(
                        {
                            "predicted_cd": candidate.predicted_cd,
                            "estimated_cd_improvement": candidate.estimated_cd_improvement,
                            "feasibility_status": candidate.feasibility_status,
                            "accepted": candidate.accepted,
                            "rejection_reasons": candidate.rejection_reasons,
                        }
                    )
                    st.download_button(
                        "Download candidate JSON",
                        candidate.model_dump_json(indent=2),
                        file_name=f"{candidate.candidate_id}.json",
                        mime="application/json",
                    )
                    if st.button(f"Export scaffold for {candidate.candidate_id}"):
                        st.write(build_envelope_scaffold(candidate.parameters, candidate.candidate_id))
            except Exception as exc:
                st.error(f"Search unavailable: {exc}")

    with tabs[4]:
        st.subheader("Data pedigree")
        st.write(
            {
                "dataset_name": baseline.get("source_dataset", SETTINGS.dataset_name),
                "run_id": int(run_id),
                "model_version": str(SETTINGS.models_dir / "xgboost_cd_model.json"),
                "feature_schema_version": SETTINGS.feature_schema_version,
                "rule_set_version": SETTINGS.rule_set_version,
            }
        )
        if "source_files_json" in baseline and pd.notna(baseline["source_files_json"]):
            st.json(json.loads(baseline["source_files_json"]))


if __name__ == "__main__":
    main()
