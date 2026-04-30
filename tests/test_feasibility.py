import pandas as pd

from copilot.feasibility import FeasibilityConfig, evaluate_rules, feasibility_status


def base_row() -> pd.Series:
    return pd.Series(
        {
            "geo_param_length": 4.6,
            "geo_param_width": 1.8,
            "geo_param_height": 1.4,
            "geo_param_wheelbase": 2.7,
            "geo_ref_reference_area": 2.5,
            "geo_param_hood_height": 0.8,
            "geo_param_roof_height": 1.3,
            "param_delta_norm_geo_param_length": 0.0,
        }
    )


def candidate(**updates: float) -> pd.Series:
    row = base_row().copy()
    for key, value in updates.items():
        row[key] = value
    return row


def result_for(rule_id: str, **updates: float):
    results = evaluate_rules(candidate(**updates), base_row(), FeasibilityConfig())
    return next(result for result in results if result.rule_id == rule_id)


def test_rule_status_pass_warning_fail_for_length() -> None:
    assert result_for("PKG_001", geo_param_length=4.61).severity == "info"
    assert result_for("PKG_001", geo_param_length=4.79).severity == "warning"
    assert result_for("PKG_001", geo_param_length=4.9).severity == "fail"


def test_width_height_and_area_rules_fail_when_outside_limits() -> None:
    assert result_for("PKG_002", geo_param_width=1.9).severity == "fail"
    assert result_for("PKG_003", geo_param_height=1.5).severity == "fail"
    assert result_for("PKG_004", geo_ref_reference_area=2.7).severity == "fail"


def test_wheelbase_roof_and_delta_rules() -> None:
    assert result_for("PKG_005", geo_param_wheelbase=2.0).severity == "fail"
    assert result_for("PKG_006", geo_param_roof_height=1.0).severity == "fail"
    assert result_for("PKG_007", param_delta_norm_geo_param_length=3.0).severity == "fail"


def test_rule_result_ranges_and_overall_status() -> None:
    results = evaluate_rules(base_row(), base_row(), FeasibilityConfig())

    assert feasibility_status(results) == "pass"
    for result in results:
        low, high = result.allowed_range
        assert low <= high
