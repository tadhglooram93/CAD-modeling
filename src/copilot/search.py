"""Random local candidate search around a selected DrivAerML baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from copilot.config import SETTINGS
from copilot.evaluate import load_model, prepare_features
from copilot.feasibility import (
    RuleResult,
    evaluate_rules,
    feasibility_penalty,
    feasibility_status,
)
from copilot.features import FeatureSchema, add_derived_features
from copilot.lineage import CandidateLineage, RunLineage


class Candidate(BaseModel):
    candidate_id: str
    baseline_run_id: int
    predicted_cd: float
    baseline_predicted_cd: float
    estimated_cd_improvement: float
    score: float
    feasibility_status: str
    parameters: dict[str, float]
    parameter_deltas: dict[str, float]
    rule_results: list[RuleResult]
    lineage: CandidateLineage
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)


def _search_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.select_dtypes(include="number").columns:
        if column == "run_id" or column == "cd" or column.startswith(("force_", "force_constref_")):
            continue
        if any(token in column for token in ("length", "width", "height", "wheelbase", "hood", "roof")):
            columns.append(column)
    return columns


def _predict_row(row: pd.Series, model, schema: FeatureSchema) -> float:
    features = prepare_features(pd.DataFrame([row.to_dict()]), schema)
    return float(model.predict(features)[0])


def _candidate_path(candidate_id: str) -> Path:
    return SETTINGS.candidates_dir / f"{candidate_id}.json"


def search_candidates(
    design_table: pd.DataFrame,
    baseline_run_id: int,
    num_candidates: int = 1000,
    perturb_sigma: float = 0.02,
    top_n: int = 3,
    model_path: Path = SETTINGS.models_dir / "xgboost_cd_model.json",
    schema_path: Path = SETTINGS.models_dir / "feature_schema.json",
) -> list[Candidate]:
    SETTINGS.ensure_directories()
    if "run_id" not in design_table.columns:
        raise ValueError("design_table must contain run_id")
    matches = design_table[design_table["run_id"] == baseline_run_id]
    if matches.empty:
        raise ValueError(f"Baseline run_id {baseline_run_id} not found.")

    schema = FeatureSchema.load(schema_path)
    model = load_model(model_path)
    rng = np.random.default_rng(SETTINGS.random_state)
    baseline = add_derived_features(pd.DataFrame([matches.iloc[0].to_dict()])).iloc[0]
    baseline_pred = _predict_row(baseline, model, schema)
    columns = _search_columns(design_table)
    if not columns:
        raise ValueError("No numeric geometry columns were found for candidate perturbation.")

    candidates: list[Candidate] = []
    source_run = RunLineage(
        run_id=baseline_run_id,
        source_dataset=str(baseline.get("source_dataset", SETTINGS.dataset_name)),
        dataset_version=str(baseline.get("dataset_version", SETTINGS.dataset_version)),
    )
    for _ in range(num_candidates):
        row = baseline.copy()
        deltas: dict[str, float] = {}
        for column in columns:
            value = float(row[column])
            scale = abs(value) * perturb_sigma if value else perturb_sigma
            delta = float(rng.normal(0.0, scale))
            row[column] = value + delta
            deltas[column] = delta
        engineered_pair = add_derived_features(pd.DataFrame([baseline.to_dict(), row.to_dict()]), baseline_run_id=None)
        candidate_row = engineered_pair.iloc[1]
        predicted = _predict_row(candidate_row, model, schema)
        rules = evaluate_rules(candidate_row, baseline)
        status = feasibility_status(rules)
        distance = float(np.sqrt(sum(delta * delta for delta in deltas.values())))
        score = predicted + feasibility_penalty(rules) + 0.01 * distance
        candidate_id = f"cand_{baseline_run_id}_{uuid4().hex[:10]}"
        failures = [f"{rule.rule_id}: {rule.explanation}" for rule in rules if rule.severity == "fail"]
        candidate = Candidate(
            candidate_id=candidate_id,
            baseline_run_id=baseline_run_id,
            predicted_cd=predicted,
            baseline_predicted_cd=baseline_pred,
            estimated_cd_improvement=baseline_pred - predicted,
            score=score,
            feasibility_status=status,
            parameters={column: float(candidate_row[column]) for column in columns},
            parameter_deltas=deltas,
            rule_results=rules,
            accepted=status != "fail",
            rejection_reasons=failures,
            lineage=CandidateLineage(
                candidate_id=candidate_id,
                baseline_run_id=baseline_run_id,
                model_version=str(model_path),
                feature_schema_version=schema.version,
                rule_set_version=SETTINGS.rule_set_version,
                source_run=source_run,
            ),
        )
        _candidate_path(candidate_id).write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
        candidates.append(candidate)

    ranked = sorted(candidates, key=lambda item: item.score)
    top = ranked[:top_n]
    top_path = SETTINGS.candidates_dir / f"top_{baseline_run_id}.json"
    top_path.write_text(json.dumps([candidate.model_dump() for candidate in top], indent=2), encoding="utf-8")
    report = SETTINGS.reports_dir / f"candidate_report_{baseline_run_id}.md"
    report.write_text(_candidate_report(top), encoding="utf-8")
    return top


def _candidate_report(candidates: list[Candidate]) -> str:
    lines = ["# Candidate Search Report", ""]
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate.candidate_id}",
                "",
                f"- Predicted Cd: {candidate.predicted_cd:.5f}",
                f"- Baseline predicted Cd: {candidate.baseline_predicted_cd:.5f}",
                f"- Estimated improvement: {candidate.estimated_cd_improvement:.5f}",
                f"- Feasibility status: {candidate.feasibility_status}",
                f"- Score: {candidate.score:.5f}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SETTINGS.data_processed / "design_table.parquet")
    parser.add_argument("--baseline-run-id", type=int, required=True)
    parser.add_argument("--num-candidates", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--perturb-sigma", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(args.data) if args.data.suffix == ".parquet" else pd.read_csv(args.data)
    top = search_candidates(
        table,
        args.baseline_run_id,
        args.num_candidates,
        args.perturb_sigma,
        args.top_n,
    )
    print(json.dumps([candidate.model_dump() for candidate in top], indent=2))


if __name__ == "__main__":
    main()
