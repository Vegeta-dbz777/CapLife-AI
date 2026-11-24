"""Operating conditions on the RUL page: what was recorded, how stressful it is, and planned changes.

The RUL networks read the conditions recorded with each reading (see ``caplife.six_rul``) and take
those of the newest reading to hold from then on. A planned change is therefore entered by replacing
the newest reading's conditions; nothing else in the history changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from caplife.six_rul import CONDITION_LABELS, condition_values, conditions_outside_training
from caplife.six_sim import CONDITION_COLUMNS, RATED_TEMP_C, frequency_multiplier

# Inputs for planned conditions: column, label, unit, step.
PLANNED_INPUTS = (
    ("Ambient_C", "Ambient temperature", "°C", 1.0),
    ("Voltage_V", "Applied voltage", "V", 1.0),
    ("Rated_V", "Rated voltage", "V", 1.0),
    ("Ripple_A", "Ripple current", "A", 0.05),
    ("Rated_Ripple_A", "Rated ripple current", "A", 0.05),
    ("Ripple_Hz", "Operating frequency", "Hz", 100.0),
)
HEATING_AT_RATED_RIPPLE_K = 12.0  # the simulator's rule; used only to update a recorded core temperature


def recorded_columns(readings: pd.DataFrame) -> list[str]:
    """Condition columns with at least one value."""
    return [column for column in CONDITION_COLUMNS if column in readings and readings[column].notna().any()]


def latest(readings: pd.DataFrame) -> pd.Series:
    """Each condition column's latest recorded value (NaN when never recorded)."""
    return pd.Series({column: (pd.to_numeric(readings[column], errors="coerce").dropna().iloc[-1]
                               if column in recorded_columns(readings) else np.nan)
                      for column in CONDITION_COLUMNS})


def _heating(ripple_a: float, rated_a: float, hz: float) -> float:
    if not all(np.isfinite(value) for value in (ripple_a, rated_a, hz)) or rated_a <= 0:
        return np.nan
    return float(HEATING_AT_RATED_RIPPLE_K * (ripple_a / rated_a / frequency_multiplier(hz)) ** 2)


def planned_core(recorded: pd.Series, planned: dict) -> float:
    """Core temperature under planned conditions, from the recorded one (NaN if none was recorded).

    The recorded core temperature moves with the ambient temperature and with the change in ripple
    heating (about 12 K at the rated ripple current, less at higher frequencies).
    """
    core = recorded.get("Core_C", np.nan)
    if not np.isfinite(core):
        return np.nan
    value = lambda column: planned.get(column, recorded.get(column, np.nan))
    ambient_change = value("Ambient_C") - recorded.get("Ambient_C", np.nan)
    core = core + (ambient_change if np.isfinite(ambient_change) else 0.0)
    before = _heating(recorded.get("Ripple_A", np.nan), recorded.get("Rated_Ripple_A", np.nan),
                      recorded.get("Ripple_Hz", np.nan))
    after = _heating(value("Ripple_A"), value("Rated_Ripple_A"), value("Ripple_Hz"))
    if np.isfinite(before) and np.isfinite(after):
        core += after - before
    return float(core)


def with_planned(readings: pd.DataFrame, planned: dict) -> pd.DataFrame:
    """The readings with the newest reading's conditions replaced by the planned ones."""
    changed = readings.copy()
    recorded = latest(readings)
    values = {column: value for column, value in planned.items() if value is not None and np.isfinite(value)}
    core = planned_core(recorded, values)
    if np.isfinite(core):
        values["Core_C"] = core
    for column, value in values.items():
        if column not in changed:
            changed[column] = np.nan
        changed[column] = changed[column].astype(float)
        changed.loc[changed.index[-1], column] = float(value)
    return changed


def stress_notes(values: pd.Series) -> list[tuple[str, str]]:
    """(level, message) judgements of one set of conditions in user units: 'error', 'warning' or 'info'."""
    derived = condition_values(pd.DataFrame([values]))
    ratio = lambda name: float(derived[name].iloc[0])
    notes = []
    voltage = ratio("voltage_ratio")
    if np.isfinite(voltage):
        if voltage > 1.0:
            notes.append(("error", f"Applied voltage is {voltage:.0%} of the rated voltage: above rating, which "
                                   "shortens life sharply and risks failure."))
        elif voltage > 0.9:
            notes.append(("warning", f"Applied voltage is {voltage:.0%} of rated, with little derating. Running at "
                                     "80% of rated or less is a common design rule."))
        else:
            notes.append(("info", f"Applied voltage is {voltage:.0%} of rated."))
    ripple = ratio("ripple_ratio")
    if np.isfinite(ripple):
        hz = float(values.get("Ripple_Hz", np.nan))
        allowed = float(frequency_multiplier(hz)) if np.isfinite(hz) else 1.0
        share = ripple / allowed
        at = f" at {hz:,.0f} Hz" if np.isfinite(hz) else ""
        if share > 1.0:
            notes.append(("error", f"Ripple current is {ripple:.0%} of the rated ripple current{at}, above what the "
                                   "capacitor is rated for; ripple heating speeds up aging."))
        elif share > 0.8:
            notes.append(("warning", f"Ripple current is {ripple:.0%} of the rated ripple current{at}, close to "
                                     "the rating."))
        else:
            notes.append(("info", f"Ripple current is {ripple:.0%} of the rated ripple current{at}."))
    core = float(values.get("Core_C", np.nan))
    hot = core if np.isfinite(core) else float(values.get("Ambient_C", np.nan))
    which = "Core" if np.isfinite(core) else "Ambient"
    if np.isfinite(hot):
        if hot > RATED_TEMP_C:
            notes.append(("error", f"{which} temperature {hot:.0f} °C is above the {RATED_TEMP_C:.0f} °C rating of "
                                   "a typical long-life series; life halves for about every 10 °C of extra heat."))
        elif hot > RATED_TEMP_C - 10:
            notes.append(("warning", f"{which} temperature {hot:.0f} °C is within 10 °C of a {RATED_TEMP_C:.0f} °C "
                                     "rating; every 10 °C cooler roughly doubles life."))
        else:
            notes.append(("info", f"{which} temperature is {hot:.0f} °C."))
    outside = conditions_outside_training(derived.iloc[0])
    if outside:
        notes.append(("warning", "Outside the conditions the model was trained on, so its prediction is an "
                                 f"extrapolation: {', '.join(outside)}."))
    return notes


def summary_columns(readings: pd.DataFrame) -> dict:
    """Latest core (or ambient) temperature, and voltage and ripple as % of rated, for the fleet table."""
    derived = condition_values(readings).iloc[-1] if len(readings) else pd.Series(dtype=float)
    core = derived.get("core_temperature", np.nan)
    return {"Temperature (°C)": core if np.isfinite(core) else derived.get("ambient_temperature", np.nan),
            "Voltage / rated": 100 * derived.get("voltage_ratio", np.nan),
            "Ripple / rated": 100 * derived.get("ripple_ratio", np.nan)}


def describe(values: pd.Series) -> str:
    """One line listing recorded conditions in user units."""
    parts = []
    units = {"Ambient_C": "°C ambient", "Core_C": "°C core", "Voltage_V": "V applied", "Rated_V": "V rated",
             "Ripple_A": "A ripple", "Rated_Ripple_A": "A rated ripple", "Ripple_Hz": "Hz"}
    for column, unit in units.items():
        value = values.get(column, np.nan)
        if np.isfinite(value):
            parts.append(f"{value:,.4g} {unit}")
    return ", ".join(parts)


__all__ = ["CONDITION_LABELS", "PLANNED_INPUTS", "describe", "latest", "planned_core", "recorded_columns",
           "stress_notes", "summary_columns", "with_planned"]
