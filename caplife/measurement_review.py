"""Read and summarize user-supplied experimental measurements."""

import csv
from io import BytesIO, TextIOWrapper

import numpy as np
import pandas as pd

from caplife.real_data import RealDataValidationError


def read_experimental_csv(payload: bytes, reference_age: bool = False) -> pd.DataFrame:
    """Preserve literal IDs and reject duplicate or misaligned measurement columns."""
    with TextIOWrapper(BytesIO(payload), encoding="utf-8-sig", newline="") as source:
        try:
            rows = csv.reader(source, strict=True)
            header = next((row for row in rows if len(row) > 1 or (row and row[0].strip())), [])
        except csv.Error as error:
            raise RealDataValidationError(f"Invalid CSV header. {error}") from error
    if len(header) != len(set(header)):
        raise RealDataValidationError("Duplicate CSV column headers are not supported.")
    frame = pd.read_csv(
        BytesIO(payload), low_memory=False, converters={"Cap_ID": str},
        keep_default_na=not reference_age,
        float_precision="round_trip" if reference_age else None,
    )
    if not isinstance(frame.index, pd.RangeIndex):
        raise RealDataValidationError("CSV rows must match the header column count.")
    return frame


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


def reference_age_observations(history: pd.DataFrame, early_fraction: float = 0.10) -> dict:
    """Describe sampling gaps and early capacitance change without interpreting them.

    ``history`` has one row per capacitor and distinct age, as returned by
    ``assess_reference_age_measurements``. The result reports the longest
    interval between consecutive readings and how much of each capacitor's
    latest capacitance loss had already appeared within the first
    ``early_fraction`` of its recorded age span.
    """
    largest_gap = None
    early_shares, early_cutoffs, first_steps, first_step_ages = [], [], [], []
    for cap_id, group in history.groupby("Cap_ID", sort=True):
        readings = group.dropna(subset=["Capacitance_loss_pct"]).sort_values("Age_h")
        ages = readings["Age_h"].to_numpy(dtype=float)
        losses = readings["Capacitance_loss_pct"].to_numpy(dtype=float)
        if ages.size < 2:
            continue
        gaps = np.diff(ages)
        index = int(np.argmax(gaps))
        if largest_gap is None or gaps[index] > largest_gap["hours"]:
            largest_gap = {"Cap_ID": cap_id, "start_h": float(ages[index]),
                           "end_h": float(ages[index + 1]), "hours": float(gaps[index])}
        first_steps.append(float(losses[1]))
        first_step_ages.append(float(ages[1]))
        span = ages[-1] - ages[0]
        cutoff = ages[0] + early_fraction * span
        early_cutoffs.append(float(cutoff))
        if ages.size >= 3 and losses[-1] > 0:
            early_loss = losses[ages <= cutoff][-1]
            early_shares.append(float(early_loss / losses[-1]))
    span_total = float(history["Age_h"].max() - history["Age_h"].min()) if len(history) else 0.0
    return {
        "largest_gap": largest_gap,
        "largest_gap_fraction_of_span": (largest_gap["hours"] / span_total
                                         if largest_gap and span_total > 0 else None),
        "early_fraction": early_fraction,
        "early_cutoff_h": float(np.median(early_cutoffs)) if early_cutoffs else None,
        "median_early_share": float(np.median(early_shares)) if early_shares else None,
        "first_step_loss_range": (min(first_steps), max(first_steps)) if first_steps else None,
        "first_step_age_range": (min(first_step_ages), max(first_step_ages)) if first_step_ages else None,
    }
