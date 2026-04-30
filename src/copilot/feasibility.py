"""Transparent proxy package-feasibility checks for candidate vehicle designs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel

from copilot.config import SETTINGS
from copilot.features import add_derived_features, find_column

Severity = Literal["info", "warning", "fail"]


class FeasibilityConfig(BaseModel):
    length_pct: float = 0.05
    width_pct: float = 0.03
    height_pct: float = 0.04
    frontal_area_increase_pct: float = 0.03
    wheelbase_to_length_min: float = 0.52
    wheelbase_to_length_max: float = 0.68
    hood_roof_drop_pct: float = 0.03
    max_normalized_parameter_delta: float = 2.5
    warning_fraction: float = 0.8

    @classmethod
    def load(cls, path: Path = SETTINGS.rule_config_path) -> FeasibilityConfig:
        if not path.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


class RuleResult(BaseModel):
    rule_id: str
    severity: Severity
    metric_value: float
    allowed_range: tuple[float, float]
    explanation: str
    recommendation: str


def _get(row: pd.Series, *tokens: str) -> float | None:
    frame = pd.DataFrame([row.to_dict()])
    column = find_column(frame, *tokens)
    if column is None or pd.isna(row[column]):
        return None
    return float(row[column])


def _severity(value: float, low: float, high: float, warning_fraction: float) -> Severity:
    if value < low or value > high:
        return "fail"
    if low in (float("-inf"), float("inf")) or high in (float("-inf"), float("inf")):
        return "info"
    if low == 0:
        return "warning" if value >= high * warning_fraction else "info"
    center = (low + high) / 2.0
    half_span = (high - low) / 2.0
    if half_span and abs(value - center) >= half_span * warning_fraction:
        return "warning"
    return "info"


def _bounded_rule(
    rule_id: str,
    value: float | None,
    low: float,
    high: float,
    explanation: str,
    recommendation: str,
    warning_fraction: float,
) -> RuleResult:
    if value is None:
        return RuleResult(
            rule_id=rule_id,
            severity="warning",
            metric_value=float("nan"),
            allowed_range=(low, high),
            explanation=f"{explanation} Metric unavailable.",
            recommendation="Confirm the source column exists in the processed design table.",
        )
    return RuleResult(
        rule_id=rule_id,
        severity=_severity(value, low, high, warning_fraction),
        metric_value=value,
        allowed_range=(low, high),
        explanation=explanation,
        recommendation=recommendation,
    )


def _pct_rule(
    rule_id: str,
    metric: str,
    candidate: pd.Series,
    baseline: pd.Series,
    pct: float,
    recommendation: str,
    warning_fraction: float,
) -> RuleResult:
    cand = _get(candidate, metric)
    base = _get(baseline, metric)
    if cand is None or base is None:
        return _bounded_rule(
            rule_id,
            None,
            -pct,
            pct,
            f"{metric} must stay within +/-{pct:.0%} of baseline.",
            recommendation,
            warning_fraction,
        )
    delta = (cand - base) / base if base else 0.0
    return _bounded_rule(
        rule_id,
        delta,
        -pct,
        pct,
        f"{metric} delta must stay within +/-{pct:.0%} of baseline.",
        recommendation,
        warning_fraction,
    )


def evaluate_rules(
    candidate: pd.Series | dict[str, object],
    baseline: pd.Series | dict[str, object],
    config: FeasibilityConfig | None = None,
) -> list[RuleResult]:
    cfg = config or FeasibilityConfig.load()
    candidate_row = pd.Series(candidate)
    baseline_row = pd.Series(baseline)
    engineered = add_derived_features(pd.DataFrame([baseline_row.to_dict(), candidate_row.to_dict()]))
    baseline_row = engineered.iloc[0]
    candidate_row = engineered.iloc[1]

    results = [
        _pct_rule(
            "PKG_001",
            "length",
            candidate_row,
            baseline_row,
            cfg.length_pct,
            "Reduce length change or revisit overhang targets.",
            cfg.warning_fraction,
        ),
        _pct_rule(
            "PKG_002",
            "width",
            candidate_row,
            baseline_row,
            cfg.width_pct,
            "Reduce width change to preserve package and track assumptions.",
            cfg.warning_fraction,
        ),
        _pct_rule(
            "PKG_003",
            "height",
            candidate_row,
            baseline_row,
            cfg.height_pct,
            "Adjust roof or body height back toward baseline envelope.",
            cfg.warning_fraction,
        ),
    ]

    cand_area = _get(candidate_row, "frontal", "area") or _get(candidate_row, "frontal_area_proxy")
    base_area = _get(baseline_row, "frontal", "area") or _get(baseline_row, "frontal_area_proxy")
    area_delta = ((cand_area - base_area) / base_area) if cand_area is not None and base_area else None
    results.append(
        _bounded_rule(
            "PKG_004",
            area_delta,
            float("-inf"),
            cfg.frontal_area_increase_pct,
            "Estimated frontal area must not increase beyond the configured limit.",
            "Reduce width or height growth, or accept the drag/package tradeoff explicitly.",
            cfg.warning_fraction,
        )
    )

    wbl = _get(candidate_row, "wheelbase_to_length")
    results.append(
        _bounded_rule(
            "PKG_005",
            wbl,
            cfg.wheelbase_to_length_min,
            cfg.wheelbase_to_length_max,
            "Wheelbase-to-length ratio must stay in a plausible package range.",
            "Adjust wheelbase or overall length to return to the configured package envelope.",
            cfg.warning_fraction,
        )
    )

    roof_candidate = _get(candidate_row, "roof", "height")
    roof_baseline = _get(baseline_row, "roof", "height")
    hood_candidate = _get(candidate_row, "hood", "height")
    hood_baseline = _get(baseline_row, "hood", "height")
    min_roof_ratio = 1.0 - cfg.hood_roof_drop_pct
    roof_ratio = roof_candidate / roof_baseline if roof_candidate is not None and roof_baseline else None
    hood_ratio = hood_candidate / hood_baseline if hood_candidate is not None and hood_baseline else None
    envelope_ratio = min(value for value in (roof_ratio, hood_ratio) if value is not None) if any(
        value is not None for value in (roof_ratio, hood_ratio)
    ) else None
    results.append(
        _bounded_rule(
            "PKG_006",
            envelope_ratio,
            min_roof_ratio,
            float("inf"),
            "Hood/roof height proxy must not fall below the baseline envelope tolerance.",
            "Raise hood or roof guide surfaces, or document why the envelope change is acceptable.",
            cfg.warning_fraction,
        )
    )

    delta_columns = [column for column in candidate_row.index if column.startswith("param_delta_norm_")]
    max_delta = (
        float(pd.to_numeric(candidate_row[delta_columns], errors="coerce").abs().max())
        if delta_columns
        else 0.0
    )
    results.append(
        _bounded_rule(
            "PKG_007",
            max_delta,
            0.0,
            cfg.max_normalized_parameter_delta,
            "Candidate must not exceed the maximum normalized parameter delta.",
            "Reduce perturbation size or search closer to the selected baseline.",
            cfg.warning_fraction,
        )
    )
    return results


def feasibility_status(results: list[RuleResult]) -> str:
    severities = {result.severity for result in results}
    if "fail" in severities:
        return "fail"
    if "warning" in severities:
        return "warning"
    return "pass"


def feasibility_penalty(results: list[RuleResult]) -> float:
    weights = {"info": 0.0, "warning": 0.02, "fail": 0.2}
    return sum(weights[result.severity] for result in results)
