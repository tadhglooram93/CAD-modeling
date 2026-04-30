"""Small visualization helpers shared by notebooks and the Streamlit demo."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from copilot.feasibility import RuleResult


def parameter_delta_chart(deltas: dict[str, float]) -> go.Figure:
    frame = pd.DataFrame({"parameter": list(deltas.keys()), "delta": list(deltas.values())})
    return px.bar(frame, x="delta", y="parameter", orientation="h", title="Candidate parameter deltas")


def rule_table(results: list[RuleResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": result.rule_id,
                "severity": result.severity,
                "metric_value": result.metric_value,
                "allowed_range": f"{result.allowed_range[0]:.4g} to {result.allowed_range[1]:.4g}",
                "explanation": result.explanation,
                "recommendation": result.recommendation,
            }
            for result in results
        ]
    )


def prediction_comparison(baseline_cd: float, candidate_cd: float) -> go.Figure:
    frame = pd.DataFrame(
        {
            "design": ["baseline", "candidate"],
            "predicted_cd": [baseline_cd, candidate_cd],
        }
    )
    return px.bar(frame, x="design", y="predicted_cd", title="Predicted drag comparison")
