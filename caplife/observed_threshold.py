"""Summarize observed capacitance-loss crossing windows."""

import numpy as np
import pandas as pd


def observed_capacitance_threshold(history: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """Bracket the first observed crossing and track later confirmation or reversal."""
    rows = []
    for cap_id, group in history.groupby("Cap_ID", sort=True):
        readings = group.dropna(subset=["Capacitance_loss_pct"]).sort_values("Age_h")
        losses = readings["Capacitance_loss_pct"].to_numpy()
        ages = readings["Age_h"].to_numpy()
        above = losses >= threshold_pct - 1e-10
        hits = np.flatnonzero(above)
        last_below = first_above = confirmation = return_below = None
        if not len(readings):
            status = "No comparable capacitance readings"
        elif len(readings) == 1:
            status = "Need a baseline and a later reading"
        elif hits.size:
            first = hits[0]
            first_above = float(ages[first])
            last_below = float(ages[first - 1]) if first else None
            repeats = np.flatnonzero(above[:-1] & above[1:]) + 1
            later_below = np.flatnonzero(~above[first + 1:]) + first + 1
            if repeats.size:
                confirmation = float(ages[repeats[0]])
            if later_below.size:
                return_below = float(ages[later_below[0]])
            status = "Threshold observed"
        else:
            last_below = float(ages[-1])
            status = "Below threshold through last reading"
        rows.append({
            "Capacitor": cap_id,
            "Latest measured loss (%)": float(losses[-1]) if losses.size else None,
            "Last below before crossing (h)": last_below,
            "First at or above threshold (h)": first_above,
            "First confirmed by two readings (h)": confirmation,
            "First later reading below threshold (h)": return_below,
            "Status": status,
        })
    return pd.DataFrame(rows)
