"""Illustrative threshold-trend RUL baseline for CapLife AI.

This is not a validated reliability or maintenance model. It uses invented
teaching thresholds shared with the CapLife AI simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EOL_THRESHOLDS, SEQ_LEN
from .data import validate_measurements
from .simulator import end_of_life_causes

# A negative direction means a falling indicator approaches its lower limit.
# A positive direction means a rising indicator approaches its upper limit.
_INDICATORS = {
    "C_ratio": ("C_ratio_min", -1),
    "ESR_ratio": ("ESR_ratio_max", 1),
    "Leakage_ratio": ("Leakage_ratio_max", 1),
    "Case_Strain_Index": ("Case_Strain_Index_max", 1),
}


def assess_trend(cap_df: pd.DataFrame, window_size: int = SEQ_LEN) -> dict:
    """Extrapolate the latest observed indicator trends to demo EOL thresholds.

    Takes only measurements already available at the prediction hour.
    Does not use RUL_hours, EOL_cause, or future measurements.
    Returns None for RUL if no indicator is trending toward its threshold.
    """
    if not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2.")

    data = validate_measurements(cap_df, seq_len=window_size)
    if data["Cap_ID"].nunique() != 1:
        raise ValueError("Select exactly one capacitor before trend prediction.")

    recent = data.sort_values("Hours").tail(window_size)
    latest = recent.iloc[-1].to_dict()
    reached = end_of_life_causes(latest)

    # As with the GRU interface, an observed demo threshold sets RUL to zero.
    if reached:
        return {
            "rul_hours": 0.0,
            "limiting_indicator": None,
            "indicator_rul_hours": {key: None for key in _INDICATORS},
            "window_size": window_size,
            "eol_causes": reached,
            "message": "A demo EOL threshold is already reached; zero RUL is rule-based.",
        }

    times = recent["Hours"].to_numpy(dtype=float)
    centered_t = times - times.mean()
    sum_sq_t = float(np.dot(centered_t, centered_t))
    if sum_sq_t <= 0:
        raise ValueError("Measurement times must be distinct.")

    indicator_rul = {}
    for feature, (threshold_key, direction) in _INDICATORS.items():
        values = recent[feature].to_numpy(dtype=float)
        centered_y = values - values.mean()
        slope = float(np.dot(centered_t, centered_y) / sum_sq_t)
        current = float(values[-1])
        limit = float(EOL_THRESHOLDS[threshold_key])

        # A non-degrading or nearly flat trend cannot estimate a crossing.
        approaching = slope < -1e-12 if direction < 0 else slope > 1e-12
        if approaching:
            remaining = (limit - current) / slope
            indicator_rul[feature] = (
                float(max(0.0, remaining)) if np.isfinite(remaining) else None
            )
        else:
            indicator_rul[feature] = None

    finite = {
        feature: hours for feature, hours in indicator_rul.items()
        if hours is not None
    }
    if not finite:
        return {
            "rul_hours": None,
            "limiting_indicator": None,
            "indicator_rul_hours": indicator_rul,
            "window_size": window_size,
            "eol_causes": [],
            "message": "No approaching demo threshold could be estimated from these trends.",
        }

    limiting = min(finite, key=finite.get)
    return {
        "rul_hours": finite[limiting],
        "limiting_indicator": limiting,
        "indicator_rul_hours": indicator_rul,
        "window_size": window_size,
        "eol_causes": [],
        "message": "Linear extrapolation of recent indicator trends; not validated for real AECs.",
    }
