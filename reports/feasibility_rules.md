# Feasibility Rules

These are educational proxy rules used to demonstrate package-aware AI workflows. They are not OEM hardpoint checks, legal compliance checks, or production vehicle package validation.

| Rule | Description |
| --- | --- |
| `PKG_001` | Overall length must stay within +/-5% of baseline by default. |
| `PKG_002` | Overall width must stay within +/-3% of baseline by default. |
| `PKG_003` | Overall height must stay within +/-4% of baseline by default. |
| `PKG_004` | Estimated frontal area must not increase by more than 3% by default. |
| `PKG_005` | Change in **wheelbase ÷ length** vs the selected baseline, within `wheelbase_delta_pct` (like length/width/height rules). Skipped (info) when the ratio is missing for either side. |
| `PKG_007` | Candidate must not exceed the maximum normalized parameter delta. |

Thresholds are configured in `configs/feasibility_rules.yaml` so the rule set is reviewable and tunable without changing code.
