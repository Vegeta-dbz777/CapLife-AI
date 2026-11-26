"""Match the columns of any uploaded table to the columns CapLife reads.

Each column in the user's file gets a target ("Use as"), guessed from its name and changeable by the user.
A file whose columns the readers already recognise is passed on untouched. Otherwise it is rewritten as a CSV
with CapLife's column names, values converted to CapLife's units where the user chose another unit, and
unused columns left out, so every reader, the cache and the saved copy see the same, standard file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from caplife.file_reader import _DECIMAL_COMMA, _LOOKUP, _TIME_LOOKUP, MISSING_MARKERS, _key, read_table
from caplife.real_data import RealDataValidationError

NOT_USED = "Not used"
MEASUREMENTS, OUTCOMES = "measurements", "outcomes"
_TIME_COLUMNS = ("Hours", "Days", "Weeks", "Months")


@dataclass(frozen=True)
class Target:
    label: str  # shown in "Use as"
    column: str  # the column name CapLife reads
    factor: float = 1.0  # multiplies values into CapLife's unit
    unit_note: str = ""  # e.g. "nF to µF"


MEASUREMENT_TARGETS = (
    Target("Capacitor ID", "Cap_ID"),
    Target("Time (hours)", "Hours"),
    Target("Time (days)", "Days"),
    Target("Time (weeks)", "Weeks"),
    Target("Time (months)", "Months"),
    Target("Capacitance (µF)", "Capacitance_uF"),
    Target("Capacitance (nF)", "Capacitance_uF", 1e-3, "nF to µF"),
    Target("Capacitance (mF)", "Capacitance_uF", 1e3, "mF to µF"),
    Target("ESR (Ω)", "ESR_ohm"),
    Target("ESR (mΩ)", "ESR_ohm", 1e-3, "mΩ to Ω"),
    Target("Leakage current (µA)", "Leakage_uA"),
    Target("Leakage current (nA)", "Leakage_uA", 1e-3, "nA to µA"),
    Target("Leakage current (mA)", "Leakage_uA", 1e3, "mA to µA"),
    Target("Case expansion (µm)", "Case_Expansion_um"),
    Target("Case expansion (mm)", "Case_Expansion_um", 1e3, "mm to µm"),
    Target("H₂ generated (mL)", "H2_Generated_mL_STP"),
    Target("Internal pressure (kPa)", "Pressure_Abs_kPa"),
    Target("Internal pressure (bar)", "Pressure_Abs_kPa", 100.0, "bar to kPa"),
    Target("Internal pressure (psi)", "Pressure_Abs_kPa", 6.894757, "psi to kPa"),
    Target("Ambient temperature (°C)", "Ambient_C"),
    Target("Core temperature (°C)", "Core_C"),
    Target("Applied voltage (V)", "Voltage_V"),
    Target("Rated voltage (V)", "Rated_V"),
    Target("Ripple current (A)", "Ripple_A"),
    Target("Ripple current (mA)", "Ripple_A", 1e-3, "mA to A"),
    Target("Rated ripple current (A)", "Rated_Ripple_A"),
    Target("Rated ripple current (mA)", "Rated_Ripple_A", 1e-3, "mA to A"),
    Target("Operating frequency (Hz)", "Ripple_Hz"),
    Target("Operating frequency (kHz)", "Ripple_Hz", 1e3, "kHz to Hz"),
    Target("ESR test frequency (Hz)", "ESR_Test_Hz"),
    Target("ESR test temperature (°C)", "ESR_Test_Temp_C"),
)
OUTCOME_TARGETS = (
    Target("Capacitor ID", "Cap_ID"),
    Target("Experiment ID", "Experiment_ID"),
    Target("Endpoint (what counted as failure)", "Endpoint"),
    Target("Last seen working (h)", "Last_alive_h"),
    Target("Failed by (h)", "Failure_by_h"),
    Target("Failure mode", "Failure_mode"),
)
TARGETS = {MEASUREMENTS: MEASUREMENT_TARGETS, OUTCOMES: OUTCOME_TARGETS}
# Columns a file must have; for measurements any of the time targets counts as the time.
REQUIRED = {MEASUREMENTS: (("Capacitor ID",), ("Time (hours)", "Time (days)", "Time (weeks)", "Time (months)")),
            OUTCOMES: (("Capacitor ID",), ("Experiment ID",), ("Endpoint (what counted as failure)",),
                       ("Last seen working (h)",), ("Failed by (h)",))}
_BY_LABEL = {kind: {target.label: target for target in targets} for kind, targets in TARGETS.items()}
_TIME_LABELS = {"hours": "Time (hours)", "days": "Time (days)", "weeks": "Time (weeks)", "months": "Time (months)"}
_PLAIN = {kind: {target.column: target.label for target in targets if target.factor == 1.0}
          for kind, targets in TARGETS.items()}

# Name stems for columns the readers do not recognise exactly, checked in order (more specific first).
_STEMS = (
    ("serial", "Capacitor ID"), ("capacitorno", "Capacitor ID"), ("capacitornumber", "Capacitor ID"),
    ("capno", "Capacitor ID"), ("sampleno", "Capacitor ID"), ("capacitorid", "Capacitor ID"),
    ("ratedripple", "Rated ripple current"), ("ripplehz", "Operating frequency"), ("ripplefreq", "Operating frequency"),
    ("frequency", "Operating frequency"), ("freq", "Operating frequency"), ("ripple", "Ripple current"),
    ("ratedvolt", "Rated voltage"), ("vrated", "Rated voltage"), ("ratedv", "Rated voltage"),
    ("volt", "Applied voltage"), ("capacit", "Capacitance"), ("esr", "ESR"), ("leak", "Leakage current"),
    ("dcl", "Leakage current"), ("expan", "Case expansion"), ("bulge", "Case expansion"),
    ("hydrogen", "H₂ generated"), ("h2", "H₂ generated"), ("pressure", "Internal pressure"),
    ("ambient", "Ambient temperature"), ("core", "Core temperature"), ("casetemp", "Core temperature"),
    ("cap", "Capacitance"),
)
_UNIT_SUFFIXES = {"Capacitance": ("nf", "mf", "uf"), "ESR": ("mohm", "ohm"), "Leakage current": ("na", "ma", "ua"),
                  "Case expansion": ("mm", "um"), "Internal pressure": ("bar", "psi", "kpa"),
                  "Ripple current": ("ma", "a"), "Rated ripple current": ("ma", "a"),
                  "Operating frequency": ("khz", "hz")}
_UNIT_NAMES = {"nf": "nF", "mf": "mF", "uf": "µF", "mohm": "mΩ", "ohm": "Ω", "na": "nA", "ma": "mA", "ua": "µA",
               "mm": "mm", "um": "µm", "bar": "bar", "psi": "psi", "kpa": "kPa", "a": "A", "khz": "kHz", "hz": "Hz"}
_OUTCOME_WORDS = (
    (("capid", "capacitorid", "capacitor", "cap", "id", "device", "deviceid", "sample", "sampleid", "serial",
      "serialnumber", "unit", "unitid"), "Capacitor ID"),
    (("experimentid", "experiment", "test", "testid", "batch", "lot", "run", "group"), "Experiment ID"),
    (("endpoint", "failurecriterion", "criterion", "eolcriterion", "endoflife"), "Endpoint (what counted as failure)"),
    (("lastaliveh", "lastalive", "lastok", "lastworking", "lastseen", "lastseenworking", "lastgood", "lastpass"),
     "Last seen working (h)"),
    (("failurebyh", "failureby", "failedby", "failedat", "failuretime", "timeoffailure", "failureh", "failure"),
     "Failed by (h)"),
    (("failuremode", "mode", "failuretype"), "Failure mode"),
)


def _measurement_guess(name: str) -> str:
    key = _key(name)
    if key in _LOOKUP:
        return _PLAIN[MEASUREMENTS].get(_LOOKUP[key], NOT_USED)
    if key in _TIME_LOOKUP:
        return _TIME_LABELS[_TIME_LOOKUP[key]]
    for stem, quantity in _STEMS:
        if stem in key:
            # The unit is whatever the name ends with, e.g. "ESR (mOhm)" -> "esrmohm".
            for suffix in _UNIT_SUFFIXES.get(quantity, ()):
                if key.endswith(suffix):
                    label = f"{quantity} ({_UNIT_NAMES[suffix]})"
                    if label in _BY_LABEL[MEASUREMENTS]:
                        return label
            return next(target.label for target in MEASUREMENT_TARGETS
                        if target.label == quantity or target.label.startswith(quantity + " ("))
    if any(word in key for word in ("time", "age", "hour", "hrs", "day", "week", "month", "elapsed")):
        unit = next((unit for unit in ("month", "week", "day") if unit in key), "hour")
        return _TIME_LABELS[unit + "s"]
    return NOT_USED


def _outcome_guess(name: str) -> str:
    key = _key(name)
    keys = {key, re.sub(r"(hours|hrs|h)$", "", key), re.sub(r"(hours|hrs)$", "h", key)}
    for words, label in _OUTCOME_WORDS:
        if keys & set(words):
            return label
    return NOT_USED


def guess(columns: list[str], kind: str) -> list[str]:
    """A "Use as" target for each column, guessed from its name; a target is given to one column at most."""
    guesser = _measurement_guess if kind == MEASUREMENTS else _outcome_guess
    taken, guesses = set(), []
    for column in columns:
        label = guesser(column)
        column_name = _BY_LABEL[kind][label].column if label != NOT_USED else None
        if column_name in taken or (column_name in _TIME_COLUMNS and taken & set(_TIME_COLUMNS)):
            label = NOT_USED
        if label != NOT_USED:
            taken.add(column_name)
        guesses.append(label)
    return guesses


def problems(choices: list[str], kind: str) -> list[str]:
    """Why these choices cannot be used: a required column not chosen, or one target chosen twice."""
    found = []
    for options in REQUIRED[kind]:
        if not any(option in choices for option in options):
            found.append(f"Choose the column to use as {options[0].split(' (')[0]}.")
    columns = [_BY_LABEL[kind][label].column for label in choices if label != NOT_USED]
    times = [column for column in columns if column in _TIME_COLUMNS]
    if len(times) > 1:
        found.append("More than one column is set as Time. Keep one.")
    for column in dict.fromkeys(columns):
        if column not in _TIME_COLUMNS and columns.count(column) > 1:
            label = next(t.label for t in TARGETS[kind] if t.column == column).split(" (")[0]
            found.append(f"More than one column is set as {label}. Keep one.")
    return found


def _number(text: str) -> float | None:
    text = str(text).strip()
    if text.lower() in MISSING_MARKERS:
        return None
    if _DECIMAL_COMMA.match(text):
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _untouched(columns: list[str], choices: list[str], kind: str) -> bool:
    """True when the reader would read the file exactly as chosen, so it can be passed on unchanged."""
    if kind == OUTCOMES:
        return all(choice == NOT_USED or _BY_LABEL[kind][choice].column == str(column).strip()
                   for column, choice in zip(columns, choices)) and all(
            choice != NOT_USED or str(column).strip() not in {t.column for t in OUTCOME_TARGETS}
            for column, choice in zip(columns, choices))
    for column, choice in zip(columns, choices):
        key = _key(column)
        if key in _LOOKUP:
            automatic = _PLAIN[kind].get(_LOOKUP[key], NOT_USED)
        elif key in _TIME_LOOKUP:
            automatic = _TIME_LABELS[_TIME_LOOKUP[key]]
        else:
            automatic = NOT_USED
        if choice != automatic:
            return False
    return True


def apply(payload: bytes, filename: str | None, choices: list[str], kind: str) -> bytes:
    """The file with the chosen columns under CapLife's names and units (unchanged when nothing changes)."""
    table, _ = read_table(payload, filename)
    columns = list(table.columns)
    if len(choices) != len(columns):
        raise RealDataValidationError("The column choices do not match the file. Upload it again.")
    if _untouched(columns, choices, kind):
        return payload
    out = {}
    for column, choice in zip(columns, choices):
        if choice == NOT_USED:
            continue
        target = _BY_LABEL[kind][choice]
        values = table[column].astype(str)
        if target.factor != 1.0:
            # A cell that is not a number is kept as written, so the reader still reports it.
            values = values.map(lambda text: text if (number := _number(text)) is None
                                else f"{number * target.factor:.10g}")
        out[target.column] = values
    return pd.DataFrame(out).to_csv(index=False).encode("utf-8")


def preview(payload: bytes, filename: str | None, kind: str) -> pd.DataFrame:
    """One row per column of the file: its name, a few example values and the guessed target."""
    table, _ = read_table(payload, filename)
    columns = [str(column) for column in table.columns]
    examples = []
    for column in table.columns:
        values = [value for value in table[column].astype(str).str.strip() if value][:3]
        # Long numbers are shortened (157.20416666666668 -> 157.204); text is shown as written.
        values = [value if (number := _number(value)) is None else f"{number:.6g}" for value in values]
        examples.append(", ".join(values)[:60])
    # "Use as" comes second so it stays in view on a phone, where the table scrolls sideways.
    return pd.DataFrame({"Column in your file": columns, "Use as": guess(columns, kind),
                         "Example values": examples})
