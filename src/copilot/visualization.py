"""Visualization helpers for feasibility rule tables (Streamlit and reports)."""

from __future__ import annotations

import pandas as pd

from copilot.feasibility import RuleResult


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
