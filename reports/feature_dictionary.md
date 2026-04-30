# Feature Dictionary

This file documents the derived proxy features used by the XGBoost drag surrogate.

| Feature | Meaning |
| --- | --- |
| `length_to_width` | Overall length divided by width. |
| `height_to_width` | Overall height divided by width. |
| `wheelbase_to_length` | Wheelbase divided by length. |
| `frontal_area_proxy` | Reference/frontal area if available, otherwise width times height. |
| `roof_height_proxy` | Roof height parameter when available. |
| `hood_height_proxy` | Hood height parameter when available. |
| `rear_slope_proxy` | Rear slope parameter when available. |
| `overhang_total_proxy` | Length minus wheelbase. |
| `overhang_front_proxy` | Half of total overhang proxy. |
| `overhang_rear_proxy` | Half of total overhang proxy. |
| `param_delta_norm_*` | Candidate or run delta from a selected baseline, scaled by train-set standard deviation. |

These features are simplified engineering proxies and should not be read as official package
or homologation metrics.
