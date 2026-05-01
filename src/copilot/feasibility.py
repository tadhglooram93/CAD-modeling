"""Transparent proxy package-feasibility checks for edited vehicle designs."""

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
    wheelbase_delta_pct: float = 0.05
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
    # Upper bound only (low unbounded): warn when close to failing high.
    if low == float("-inf") and high != float("inf"):
        band = high - max(abs(high), 1e-9) * (1.0 - warning_fraction)
        return "warning" if value >= band else "info"
    # Lower bound only (high unbounded): fail below `low`; warn in a thin band just above the floor.
    if high == float("inf") and low != float("-inf"):
        margin = (
            abs(low) * (1.0 - warning_fraction)
            if low < 0
            else max(1.0 - low, low * 0.05, 1e-9) * (1.0 - warning_fraction)
        )
        upper = low + margin
        return "warning" if low <= value <= upper else "info"
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

    wbl_c = _get(candidate_row, "wheelbase_to_length")
    wbl_b = _get(baseline_row, "wheelbase_to_length")
    if (
        wbl_c is None
        or pd.isna(wbl_c)
        or wbl_b is None
        or pd.isna(wbl_b)
        or wbl_b == 0
    ):
        results.append(
            RuleResult(
                rule_id="PKG_005",
                severity="info",
                metric_value=float("nan"),
                allowed_range=(-cfg.wheelbase_delta_pct, cfg.wheelbase_delta_pct),
                explanation=(
                    "Wheelbase-to-length delta skipped: metric unavailable for candidate or baseline "
                    "(missing wheelbase / unusable overhang proxy)."
                ),
                recommendation=(
                    "Supply wheelbase or consistent length/overhang geometry, or rely on other package rules."
                ),
            )
        )
    else:
        delta_wbl = (wbl_c - wbl_b) / abs(wbl_b)
        results.append(
            _bounded_rule(
                "PKG_005",
                delta_wbl,
                -cfg.wheelbase_delta_pct,
                cfg.wheelbase_delta_pct,
                "Wheelbase-to-length delta vs baseline must stay within the configured band.",
                "Adjust wheelbase or overall length toward the baseline proportion.",
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
            "Reduce perturbation size or stay closer to the selected baseline.",
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
