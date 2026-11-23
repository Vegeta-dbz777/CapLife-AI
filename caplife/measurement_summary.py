"""One-row-per-capacitor summary of uploaded measurements (Analyze measurements, first tab)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from caplife.six_rul import INDICATOR_LABELS, current_limit_used
from caplife.six_sim import SIGNALS

CHANGE_COLUMNS = {
    "Capacitance_uF": "Capacitance change (%)",
    "ESR_ohm": "ESR change (%)",
    "Leakage_uA": "Leakage change (%)",
    "Case_Expansion_um": "Case expansion (µm)",
    "H2_Generated_mL_STP": "H₂ generated (mL)",
    "Pressure_Abs_kPa": "Pressure rise (kPa)",
}


def capacitor_summary(readings: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Readings, time span, parameters measured, change of each parameter from its first to its latest
    reading, the parameter closest to its default end-of-life limit, and the number of data notes."""
    rows = []
    for device, group in readings.sort_values("Hours").groupby("Cap_ID", sort=True):
        measured = [signal for signal in SIGNALS if signal in group and group[signal].notna().sum() >= 2]
        row = {"Capacitor": device, "Readings": len(group),
               "Time span (h)": f"{group['Hours'].min():,.0f} – {group['Hours'].max():,.0f}",
               "Parameters": f"{len(measured)} of 6"}
        used = {}
        for signal, label in CHANGE_COLUMNS.items():
            values = group[signal].dropna() if signal in group else pd.Series(dtype=float)
            if signal not in measured:
                row[label] = np.nan
                continue
            first, last = float(values.iloc[0]), float(values.iloc[-1])
            if signal in ("Capacitance_uF", "ESR_ohm", "Leakage_uA"):
                row[label] = 100 * (last / first - 1) if first > 0 else np.nan
            else:  # expansion, H2 and pressure: the change in their own units
                row[label] = last - first
            pair = pd.DataFrame({signal: [first, last]})
            used[signal] = float(current_limit_used(pair)[signal].iloc[1])
        if used:
            closest = max(used, key=used.get)
            row["Closest to its default limit"] = f"{INDICATOR_LABELS[closest]} ({max(used[closest], 0):.0%} of the way)"
        else:
            row["Closest to its default limit"] = "—"
        row["Data notes"] = sum(str(w).startswith(f"{device}:") for w in warnings)
        rows.append(row)
    return pd.DataFrame(rows)

