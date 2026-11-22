"""CapLife AI v0.9: descriptive interval rates for *directly observed* AEC indicators.

Only raw measured values enter this module. No imputation, smoothing, trend
extrapolation, physics-derived H2/pressure, RUL, failure threshold, or health score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Each channel is kept in its original units, never pooled into a unitless score.
CHANNELS = (
    ("Capacitance_uF", "Capacitance", "µF"),
    ("ESR_ohm", "ESR", "Ω"),
    ("Leakage_uA", "Leakage current", "µA"),
    ("Case_Expansion_um", "Case expansion", "µm"),
    ("H2_Generated_mL_STP", "H₂ generated (directly measured)", "mL STP"),
    ("Pressure_Abs_kPa", "Internal absolute pressure (directly measured)", "kPa absolute"),
)

SUMMARY_COLUMNS = [
    "Cap_ID", "Indicator", "Unit", "Available_readings", "Valid_intervals",
    "Valid_accelerations", "Time_weighted_rate_per_hour", "Latest_rate_per_hour",
    "Latest_acceleration_per_hour2", "Latest_observation_hour", "Status",
]
INTERVAL_COLUMNS = [
    "Cap_ID", "Indicator", "Unit", "Start_Hour", "End_Hour", "Midpoint_Hour",
    "Start_Value", "End_Value", "Change", "Rate_per_hour", "Source",
]
ACCELERATION_COLUMNS = [
    "Cap_ID", "Indicator", "Unit", "At_Hours", "First_interval_end_h",
    "Second_interval_end_h", "First_rate_per_hour", "Second_rate_per_hour",
    "Acceleration_per_hour2", "Source",
]


RATE_CHANGE_RELATIVE_TOLERANCE = 1e-9


def _clean_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def _esr_conditions_compatible(group: pd.DataFrame) -> tuple[bool, bool]:
    """A full-history conservative gate matching v0.8's ESR ratio policy.

    Returns (comparable, documented). If either condition varies or is only
    partially supplied, withhold ESR rate estimates. Entirely absent test
    conditions allow a *provisional* rate, clearly labeled.
    """
    present = []
    for name in ("ESR_Test_Hz", "ESR_Test_Temp_C"):
        if name not in group or group[name].isna().all():
            present.append(False)
            continue
        series = _clean_numeric(group[name])
        if series.isna().any() or series.nunique(dropna=True) > 1:
            return False, False
        present.append(True)
    return True, all(present)


def calculate_degradation_rates(measurements: pd.DataFrame) -> dict:
    """Calculate observed first differences and changes in first differences.

    Interval rate = (value at right endpoint - value at left endpoint) / Δhours.
    Acceleration = (rate of interval 2 - rate of interval 1) /
                   (midpoint of interval 2 - midpoint of interval 1).

    Intervals are formed ONLY from adjacent chronological *CSV rows*. If a
    sensor value is missing at a row, no interval bridges across that gap.
    Acceleration needs two adjacent valid intervals (three consecutive rows).
    Time-weighted rate averages only observed valid intervals, never assuming
    values between non-adjacent observations. The line between two real
    measurements is a secant used for arithmetic, not an interpolated sample.
    """
    if not isinstance(measurements, pd.DataFrame) or not {"Cap_ID", "Hours"}.issubset(measurements.columns):
        raise ValueError("The rate calculator requires Cap_ID and Hours columns.")
    if measurements.empty:
        raise ValueError("Cannot calculate rates from an empty measurement table.")

    readings = measurements.copy()
    readings["Hours"] = _clean_numeric(readings["Hours"])
    if readings["Hours"].isna().any() or not np.isfinite(readings["Hours"]).all() or (readings["Hours"] < 0).any():
        raise ValueError("Hours must be finite, nonnegative numbers.")
    if readings["Cap_ID"].isna().any() or (readings["Cap_ID"].astype(str).str.strip() == "").any():
        raise ValueError("Cap_ID must not be missing.")
    if readings.duplicated(["Cap_ID", "Hours"]).any():
        raise ValueError("Duplicate Cap_ID/Hours entries are not supported.")
    readings = readings.sort_values(["Cap_ID", "Hours"], kind="mergesort")

    summaries, intervals, accelerations, warnings = [], [], [], []
    for cap_id, group in readings.groupby("Cap_ID", sort=True):
        group = group.reset_index(drop=True)
        times = group["Hours"].to_numpy(dtype=float)
        if len(times) > 1 and (np.diff(times) <= 0).any():
            raise ValueError(f"{cap_id}: Hours must strictly increase.")

        for column, label, unit in CHANNELS:
            y = _clean_numeric(group[column]).to_numpy(dtype=float) if column in group else np.full(len(group), np.nan)
            if np.isinf(y).any():
                raise ValueError(f"{cap_id}: {column} contains an infinite value.")
            available = int(np.isfinite(y).sum())
            gate = True
            documented = True
            if column == "ESR_ohm" and available > 1:
                observed = group.loc[np.isfinite(y)]
                gate, documented = _esr_conditions_compatible(observed)
                if not gate:
                    warnings.append(f"{cap_id}: ESR rate withheld because recorded ESR test conditions vary or are partially missing.")
                elif not documented:
                    warnings.append(f"{cap_id}: ESR test conditions are undocumented or incomplete; ESR rates are provisional.")

            current_intervals = []
            if gate:
                for j in range(1, len(group)):
                    if not (np.isfinite(y[j - 1]) and np.isfinite(y[j])):
                        continue
                    t0, t1 = float(times[j - 1]), float(times[j])
                    delta = float(y[j] - y[j - 1])
                    rate = delta / (t1 - t0)
                    record = {
                        "Cap_ID": cap_id, "Indicator": label, "Unit": unit,
                        "Start_Hour": t0, "End_Hour": t1, "Midpoint_Hour": (t0 + t1) / 2,
                        "Start_Value": float(y[j - 1]), "End_Value": float(y[j]),
                        "Change": delta, "Rate_per_hour": rate,
                        "Source": "DIRECT MEASUREMENT (ESR provisional)" if column == "ESR_ohm" and not documented else "DIRECT MEASUREMENT",
                    }
                    intervals.append(record)
                    current_intervals.append((j, record))
            current_accels = []
            for (_, prev), (next_index, nxt) in zip(current_intervals[:-1], current_intervals[1:]):
                # Consecutive valid intervals MUST share one measured endpoint.
                # Never calculate across an intervening missing sensor reading.
                if abs(prev["End_Hour"] - nxt["Start_Hour"]) > 1e-9:
                    continue
                mid_difference = nxt["Midpoint_Hour"] - prev["Midpoint_Hour"]
                rate_change = nxt["Rate_per_hour"] - prev["Rate_per_hour"]
                # Equal rates can differ by floating-point rounding alone (~1e-16
                # relative). A relative change below 1e-9 is far beyond any sensor
                # resolution, so report it as zero rather than as a spurious value.
                if abs(rate_change) <= RATE_CHANGE_RELATIVE_TOLERANCE * max(
                        abs(nxt["Rate_per_hour"]), abs(prev["Rate_per_hour"])):
                    rate_change = 0.0
                acceleration = rate_change / mid_difference
                record = {
                    "Cap_ID": cap_id, "Indicator": label, "Unit": unit,
                    "At_Hours": (nxt["Midpoint_Hour"] + prev["Midpoint_Hour"]) / 2,
                    "First_interval_end_h": prev["End_Hour"],
                    "Second_interval_end_h": nxt["End_Hour"],
                    "First_rate_per_hour": prev["Rate_per_hour"],
                    "Second_rate_per_hour": nxt["Rate_per_hour"],
                    "Acceleration_per_hour2": float(acceleration),
                    "Source": nxt["Source"],
                }
                accelerations.append(record)
                current_accels.append(record)

            if available < 2:
                status = "Insufficient measured readings"
            elif not gate:
                status = "ESR test conditions incompatible; rates withheld"
            elif not current_intervals:
                status = "No adjacent observed readings; gaps not bridged"
            elif not documented:
                status = "Provisional rates; ESR test settings not fully documented"
            elif len(current_accels) == 0:
                status = "Rate only; at least 3 consecutive readings needed for acceleration"
            else:
                status = "Descriptive rate and acceleration available"

            latest_interval = current_intervals[-1][1] if current_intervals and current_intervals[-1][0] == len(group) - 1 else None
            latest_accel = current_accels[-1] if current_accels and current_accels[-1]["Second_interval_end_h"] == float(times[-1]) else None
            weighted = (
                sum(row["Change"] for _, row in current_intervals)
                / sum(row["End_Hour"] - row["Start_Hour"] for _, row in current_intervals)
                if current_intervals else np.nan
            )
            if column == "H2_Generated_mL_STP" and any(row["Change"] < -1e-12 for _, row in current_intervals):
                warnings.append(f"{cap_id}: cumulative generated-H₂ readings decrease at an interval; verify whether these are truly cumulative production measurements.")
            summaries.append({
                "Cap_ID": cap_id, "Indicator": label, "Unit": unit,
                "Available_readings": available, "Valid_intervals": len(current_intervals),
                "Valid_accelerations": len(current_accels),
                "Time_weighted_rate_per_hour": weighted,
                "Latest_rate_per_hour": latest_interval["Rate_per_hour"] if latest_interval else np.nan,
                "Latest_acceleration_per_hour2": latest_accel["Acceleration_per_hour2"] if latest_accel else np.nan,
                "Latest_observation_hour": float(times[np.flatnonzero(np.isfinite(y))[-1]]) if available else np.nan,
                "Status": status,
            })
    return {
        "summary": pd.DataFrame(summaries, columns=SUMMARY_COLUMNS),
        "intervals": pd.DataFrame(intervals, columns=INTERVAL_COLUMNS),
        "accelerations": pd.DataFrame(accelerations, columns=ACCELERATION_COLUMNS),
        "warnings": list(dict.fromkeys(warnings)),
    }
