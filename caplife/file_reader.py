"""Read uploaded measurement tables the way labs actually export them.

Accepts CSV (comma, semicolon or tab separated, with decimal points or decimal
commas as European Excel writes them, UTF-8 or Windows encodings) and Excel
.xlsx files. Recognises common variations of the column names, treats markers
such as "n/a" or "-" as missing, converts time in days, weeks or months to
hours, and detects gauge pressure. Rows that cannot be used (no capacitor ID,
no time, negative time, a repeated reading) are left out rather than rejecting
the whole file. Every adjustment is reported in plain language, and no
measurement is ever invented.
"""
from __future__ import annotations

import csv
import re
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

from caplife.real_data import OPTIONAL_COLUMNS, RealDataValidationError

ATMOSPHERE_KPA = 101.325
# A first pressure reading below this is far under atmospheric pressure, so it is almost surely gauge pressure.
GAUGE_PRESSURE_BELOW_KPA = 80.0
TIME_UNITS = {"hours": 1.0, "days": 24.0, "weeks": 168.0, "months": 730.0}
MISSING_MARKERS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "—", "–", "?", "#n/a", "#value!", "missing"}
_DECIMAL_COMMA = re.compile(r"^[+-]?\d+,\d+([eE][+-]?\d+)?$")


def _key(name: str) -> str:
    """Column-name key that ignores case, spaces, punctuation and unit symbols."""
    text = str(name).strip().lower()
    for old, new in (("µ", "u"), ("μ", "u"), ("ω", "ohm"), ("Ω".lower(), "ohm"), ("₂", "2"), ("°", "")):
        text = text.replace(old, new)
    return re.sub(r"[^a-z0-9]", "", text)


# Canonical column -> accepted spellings (compared by _key). Time columns in other units are listed
# separately so they can be converted to hours.
ALIASES = {
    "Cap_ID": ("Cap_ID", "Capacitor ID", "Capacitor", "Cap", "ID", "Device", "Device ID", "Sample", "Sample ID",
               "Serial", "Serial number", "Unit", "Unit ID"),
    "Hours": ("Hours", "Hour", "Hrs", "Time (h)", "Time h", "Time_h", "Time hours", "Age (h)", "Age_h", "Age",
              "Test time (h)", "Time"),
    "Capacitance_uF": ("Capacitance_uF", "Capacitance (uF)", "Capacitance uF", "Capacitance", "C (uF)", "C_uF",
                       "Cap (uF)"),
    "ESR_ohm": ("ESR_ohm", "ESR (ohm)", "ESR (Ω)", "ESR ohm", "ESR", "ESR_ohms", "ESR (ohms)"),
    "Leakage_uA": ("Leakage_uA", "Leakage (uA)", "Leakage current (uA)", "Leakage current", "Leakage",
                   "Leakage_current_uA", "I_leak (uA)", "DCL (uA)", "DCL"),
    "Case_Expansion_um": ("Case_Expansion_um", "Case expansion (um)", "Case expansion", "Expansion (um)",
                          "Expansion", "Bulge (um)", "Case_bulge_um", "Bulge"),
    "H2_Generated_mL_STP": ("H2_Generated_mL_STP", "H2 generated (mL)", "H2 generated", "H2 (mL)", "H2_mL",
                            "H2", "Hydrogen (mL)", "Hydrogen", "H₂ generated (mL)"),
    "Pressure_Abs_kPa": ("Pressure_Abs_kPa", "Pressure (kPa)", "Internal pressure (kPa)", "Internal pressure",
                         "Pressure", "Pressure_kPa", "Absolute pressure (kPa)", "Pressure abs (kPa)"),
    "Ambient_C": ("Ambient_C", "Ambient (C)", "Ambient temperature (C)", "Ambient temperature", "Ambient temp",
                  "T_ambient", "Ta (C)"),
    "Core_C": ("Core_C", "Core (C)", "Core temperature (C)", "Core temperature", "Core temp", "Case temperature (C)",
               "Case temperature", "T_core", "Tc (C)"),
    "Voltage_V": ("Voltage_V", "Voltage (V)", "Applied voltage (V)", "Applied voltage", "Operating voltage (V)",
                  "Operating voltage", "Bias voltage (V)", "DC voltage (V)", "Voltage"),
    "Rated_V": ("Rated_V", "Rated voltage (V)", "Rated voltage", "V_rated", "Rated V"),
    "Ripple_A": ("Ripple_A", "Ripple current (A)", "Ripple current", "Ripple (A)", "I_ripple (A)", "Ripple"),
    "Rated_Ripple_A": ("Rated_Ripple_A", "Rated ripple current (A)", "Rated ripple current", "Rated ripple (A)",
                       "Rated ripple", "I_ripple_rated (A)"),
    "Ripple_Hz": ("Ripple_Hz", "Ripple frequency (Hz)", "Ripple frequency", "Operating frequency (Hz)",
                  "Operating frequency", "Switching frequency (Hz)", "Switching frequency", "Frequency (Hz)",
                  "Ripple_frequency_Hz"),
}
TIME_ALIASES = {
    "days": ("Days", "Day", "Time (d)", "Time (days)", "Time_days", "Age (days)", "Age_d", "Age_days"),
    "weeks": ("Weeks", "Week", "Time (weeks)", "Time_weeks", "Age (weeks)"),
    "months": ("Months", "Month", "Time (months)", "Time_months", "Age (months)"),
}
for _column in OPTIONAL_COLUMNS:
    ALIASES.setdefault(_column, (_column,))
_LOOKUP = {_key(alias): column for column, aliases in ALIASES.items() for alias in aliases}
_TIME_LOOKUP = {_key(alias): unit for unit, aliases in TIME_ALIASES.items() for alias in aliases}
NUMERIC_COLUMNS = tuple(column for column in ALIASES if column != "Cap_ID")


def _decode(payload: bytes) -> tuple[str, str | None]:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding), (None if encoding == "utf-8-sig" else
                                              "Read the file as Windows (cp1252) text because it is not UTF-8.")
        except UnicodeDecodeError:
            continue
    return payload.decode("latin-1"), "Read the file as Latin-1 text because it is not UTF-8."


def _delimiter(text: str) -> str:
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {sep: header.count(sep) for sep in (",", ";", "\t")}
    return max(counts, key=counts.get) if max(counts.values()) else ","


def read_table(payload: bytes, filename: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Any supported file as a table of text cells, plus notes on how it was read."""
    notes: list[str] = []
    # Excel files are ZIP archives; the name is not trusted, since a file whose columns were matched on the
    # page is passed on as CSV under its original name.
    if payload[:4] == b"PK\x03\x04":
        try:
            frame = pd.read_excel(BytesIO(payload), sheet_name=0, dtype=str, keep_default_na=False)
        except Exception as error:  # openpyxl raises many types for damaged files
            raise RealDataValidationError(f"The Excel file could not be read: {error}") from error
        frame = frame.fillna("").astype(str)
        frame = frame.loc[frame.apply(lambda row: any(cell.strip() for cell in row), axis=1)].reset_index(drop=True)
        frame.columns = [str(column) for column in frame.columns]
        notes.append("Read the first sheet of the Excel file.")
        return frame, notes
    text, note = _decode(payload)
    if note:
        notes.append(note)
    if not text.strip():
        raise RealDataValidationError("The file is empty.")
    sep = _delimiter(text)
    if sep != ",":
        notes.append("Read a " + ("semicolon" if sep == ";" else "tab") + "-separated file.")
    rows = [row for row in csv.reader(StringIO(text, newline=""), delimiter=sep)]
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise RealDataValidationError("The file is empty.")
    header = [cell.strip() for cell in rows[0]]
    if len(set(header)) != len(header):
        raise RealDataValidationError("Duplicate CSV column headers are not supported.")
    # A trailing empty column (a separator at the end of every line) is harmless; otherwise a row
    # with a different number of fields could shift values into the wrong columns, so it is refused.
    while header and not header[-1].strip() and all(len(row) == len(header) and not row[-1].strip() for row in rows):
        header, rows = header[:-1], [row[:-1] for row in rows]
    bad = [index + 1 for index, row in enumerate(rows) if len(row) != len(header)]
    if bad:
        raise RealDataValidationError(
            f"Line {bad[0]} does not match the header column count ({len(rows[bad[0] - 1])} values for "
            f"{len(header)} columns), so values could land in the wrong column. Check for an extra or missing "
            "separator on that line.")
    return pd.DataFrame(rows[1:], columns=header), notes


def _rename(frame: pd.DataFrame, notes: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Canonical column names; returns the unit of a non-hour time column, if one was used."""
    mapping, time_unit, renamed = {}, None, []
    for column in frame.columns:
        key = _key(column)
        target = _LOOKUP.get(key)
        if target is None and key in _TIME_LOOKUP and "Hours" not in mapping.values():
            target, time_unit = "Hours", _TIME_LOOKUP[key]
        if target is None:
            continue
        if target in mapping.values():
            raise RealDataValidationError(
                f"Duplicate CSV column headers: two columns both look like {target}: '{next(k for k, v in mapping.items() if v == target)}' and "
                f"'{column}'. Rename or remove one.")
        mapping[column] = target
        if str(column) != target:
            renamed.append(f"'{column}' as {target}")
    if renamed:
        notes.append("Recognised " + ", ".join(renamed) + ".")
    return frame.rename(columns=mapping), time_unit


def read_measurement_file(payload: bytes, filename: str | None = None, time_unit: str = "hours",
                          pressure: str = "auto") -> tuple[pd.DataFrame, list[str]]:
    """A measurement table ready for validation, and plain-language notes on every adjustment.

    ``time_unit`` is the unit of the time column when its name does not say (hours, days, weeks or
    months). ``pressure`` is "absolute", "gauge" or "auto" (gauge when a capacitor's first pressure
    reading is far below atmospheric).
    """
    table, notes = read_table(payload, filename)
    frame, named_unit = _rename(table, notes)
    missing = [column for column in ("Cap_ID", "Hours") if column not in frame]
    if missing:
        found = ", ".join(f"'{c}'" for c in table.columns[:8])
        raise RealDataValidationError(
            f"Could not find a {' or '.join(missing)} column. Columns found: {found}. Name the capacitor column "
            "Cap_ID and the time column Hours (or Days, Weeks).")
    frame = frame[[c for c in frame.columns if c in ALIASES]].copy()
    frame["Cap_ID"] = frame["Cap_ID"].astype(str).str.strip()

    # Numbers: missing markers become blank; decimal commas become decimal points.
    markers, commas = 0, 0
    for column in [c for c in NUMERIC_COLUMNS if c in frame]:
        text = frame[column].astype(str).str.strip()
        blank = text.str.lower().isin(MISSING_MARKERS)
        markers += int((blank & text.ne("")).sum())
        comma = text.str.match(_DECIMAL_COMMA)
        commas += int(comma.sum())
        text = text.where(~comma, text.str.replace(",", ".", regex=False))
        frame[column] = text.where(~blank, None)
    if markers:
        notes.append(f"Treated {markers} cell(s) such as 'n/a' or '-' as missing readings.")
    if commas:
        notes.append("Read decimal commas (for example 0,085) as decimal points.")
    for column in [c for c in NUMERIC_COLUMNS if c in frame]:
        parsed = pd.to_numeric(frame[column], errors="coerce")
        bad = frame[column].notna() & (parsed.isna() | ~np.isfinite(parsed.astype(float)))
        if bad.any():
            row = int(bad[bad].index[0]) + 2
            raise RealDataValidationError(
                f"{column}: '{frame.loc[bad[bad].index[0], column]}' on line {row} is not a number.")
        frame[column] = parsed.astype(float)

    # Rows that cannot be placed on a capacitor's timeline are left out, not the whole file.
    no_id = frame["Cap_ID"].eq("") | frame["Cap_ID"].str.lower().isin({"nan", "none"})
    if no_id.any():
        notes.append(f"Left out {int(no_id.sum())} row(s) without a capacitor ID.")
    no_time = frame["Hours"].isna() | (frame["Hours"] < 0)
    if (no_time & ~no_id).any():
        notes.append(f"Left out {int((no_time & ~no_id).sum())} row(s) with a missing or negative time.")
    frame = frame.loc[~no_id & ~no_time].reset_index(drop=True)

    unit = named_unit or time_unit
    if unit not in TIME_UNITS:
        raise RealDataValidationError(f"Unknown time unit '{unit}'.")
    if unit != "hours":
        frame["Hours"] = frame["Hours"] * TIME_UNITS[unit]
        notes.append(f"Converted time from {unit} to hours (1 {unit[:-1]} = {TIME_UNITS[unit]:g} h).")

    duplicated = frame.duplicated(["Cap_ID", "Hours"], keep="first")
    if duplicated.any():
        examples = frame.loc[duplicated, ["Cap_ID", "Hours"]].head(3)
        shown = ", ".join(f"{row.Cap_ID} at {row.Hours:g} h" for row in examples.itertuples())
        notes.append(f"Kept only the first of {int(duplicated.sum())} repeated reading(s) for the same capacitor and "
                     f"time ({shown}{' …' if duplicated.sum() > 3 else ''}).")
        frame = frame.loc[~duplicated].reset_index(drop=True)

    if "Pressure_Abs_kPa" in frame and frame["Pressure_Abs_kPa"].notna().any():
        first = frame.dropna(subset=["Pressure_Abs_kPa"]).sort_values("Hours").groupby("Cap_ID")["Pressure_Abs_kPa"].first()
        gauge = list(first.index) if pressure == "gauge" else (
            list(first.index[first < GAUGE_PRESSURE_BELOW_KPA]) if pressure == "auto" else [])
        if gauge:
            rows = frame["Cap_ID"].isin(gauge)
            frame.loc[rows, "Pressure_Abs_kPa"] += ATMOSPHERE_KPA
            if pressure == "gauge":
                notes.append("Converted gauge pressure to absolute pressure by adding 101.3 kPa.")
            else:
                shown = ", ".join(gauge[:5]) + (" …" if len(gauge) > 5 else "")
                notes.append(f"Pressure for {len(gauge)} capacitor(s) starts far below atmospheric (101.3 kPa), so it "
                             f"looks like gauge pressure; added 101.3 kPa to make it absolute ({shown}).")
    if frame.empty:
        raise RealDataValidationError("No usable rows: every row lacks a capacitor ID or a valid time.")
    return frame, notes
