"""Descriptive assessment of uploaded AEC measurements (CapLife AI v0.8).

Does not call GRU inference, compute RUL, apply EOL thresholds, or fabricate data.
Compatible with pandas and numpy versions in the existing CapLife AI project.
"""
from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Cap_ID", "Hours")
REFERENCE_AGE_COLUMNS = ("Cap_ID", "Age_h", "Capacitance_uF", "Resistance_Ohm")
MEASUREMENTS = {
    "Capacitance_uF": ("Capacitance", "µF"),
    "ESR_ohm": ("ESR", "Ω"),
    "Leakage_uA": ("Leakage current", "µA"),
    "Case_Expansion_um": ("Case expansion", "µm"),
    "H2_Generated_mL_STP": ("H₂ generated (measured)", "mL at 0°C, 1 atm"),
    "Pressure_Abs_kPa": ("Internal pressure (measured)", "kPa absolute"),
}
OPTIONAL_COLUMNS = (
    "Capacitance_uF", "ESR_ohm", "Leakage_uA", "Ambient_C",
    "Core_C", "Voltage_V", "Rated_V", "Ripple_A", "Rated_Ripple_A", "Ripple_Hz",
    "ESR_Test_Hz", "ESR_Test_Temp_C", "Case_Expansion_um",
    "H2_Generated_mL_STP", "Pressure_Abs_kPa",
)
TEMPLATE_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
RATIO_COLUMNS = {
    "Capacitance_uF": "C_ratio",
    "ESR_ohm": "ESR_ratio",
    "Leakage_uA": "Leakage_ratio",
}


class RealDataValidationError(ValueError):
    """A raw measurement file cannot be assessed safely."""


def blank_template_csv() -> str:
    """A header-only file: no synthetic rows disguised as real observations."""
    return pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False)


def _valid_numeric_series(series: pd.Series, column: str) -> pd.Series:
    """Allow blanks, but reject nonnumeric nonblank values and infinities."""
    stripped = series.astype("string").str.strip()
    blank = series.isna() | stripped.eq("").fillna(False)
    parsed = pd.to_numeric(series.where(~blank, np.nan), errors="coerce")
    invalid = (~blank) & (parsed.isna() | ~np.isfinite(parsed))
    if invalid.any():
        raise RealDataValidationError(
            f"{column}: invalid numeric entry at CSV row {int(invalid[invalid].index[0]) + 2}."
        )
    return parsed.astype(float)


def _validate_capacitor_ids(series: pd.Series) -> pd.Series:
    ids = series.astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise RealDataValidationError("Cap_ID must not be blank.")
    # Protect downloaded CSV from spreadsheet formula injection through IDs.
    if ids.str.startswith(("=", "+", "-", "@"), na=False).any():
        raise RealDataValidationError("Cap_ID may not begin with =, +, -, or @.")
    return ids.astype(str)


def assess_reference_age_measurements(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize reference-age readings using the median at each recorded age."""
    if raw.empty:
        raise RealDataValidationError("The CSV contains no measurement rows.")
    if len(raw) > 100_000:
        raise RealDataValidationError("Maximum 100,000 rows per upload in this prototype.")
    if not isinstance(raw.index, pd.RangeIndex):
        raise RealDataValidationError("CSV rows must match the header column count.")
    if raw.columns.duplicated().any():
        raise RealDataValidationError("Duplicate CSV column headers are not supported.")
    missing = [column for column in REFERENCE_AGE_COLUMNS if column not in raw]
    if missing:
        raise RealDataValidationError("Missing required column(s): " + ", ".join(missing))

    signals = ["Capacitance_uF", "Resistance_Ohm"]
    measurements = raw.loc[:, list(REFERENCE_AGE_COLUMNS)].copy()
    measurements["Cap_ID"] = _validate_capacitor_ids(measurements["Cap_ID"])
    for column in ("Age_h", *signals):
        measurements[column] = _valid_numeric_series(measurements[column], column)
        if measurements[column].isna().any():
            raise RealDataValidationError(f"{column} must be present for every row.")
    if measurements["Age_h"].lt(0).any():
        raise RealDataValidationError("Age_h must be nonnegative.")
    if measurements[signals].le(0).any().any():
        raise RealDataValidationError("Capacitance and resistance must be positive.")

    history = measurements.groupby(["Cap_ID", "Age_h"], sort=True, as_index=False)[signals].median()
    by_capacitor = history.groupby("Cap_ID", sort=True)
    if by_capacitor.size().lt(2).any():
        raise RealDataValidationError("Each capacitor needs readings at two or more distinct ages.")

    ratios = history[signals] / by_capacitor[signals].transform("first")
    history["Capacitance_loss_pct"] = 100 * (1 - ratios["Capacitance_uF"])
    history["Resistance_change_pct"] = 100 * (ratios["Resistance_Ohm"] - 1)
    metrics = ["Capacitance_loss_pct", "Resistance_change_pct"]
    summary = history.groupby("Cap_ID", sort=True)[["Age_h", *metrics]].last()
    summary = summary.rename(columns={"Age_h": "Latest_age_h"})
    summary.insert(0, "Baseline_age_h", by_capacitor["Age_h"].first())
    summary["RUL_status"] = "No validated estimate"
    return summary.reset_index(), history


def validate_real_measurements(raw: pd.DataFrame) -> pd.DataFrame:
    """Check identity, timeline, units' numeric formats and physical sign limits.

    Missing non-key sensor observations remain NaN. Does not assume conditions
    from nearby rows or accept precomputed ratios as source-of-truth.
    """
    if not isinstance(raw, pd.DataFrame):
        raise RealDataValidationError("Expected a CSV table.")
    if raw.empty:
        raise RealDataValidationError("The CSV contains no measurement rows.")
    if len(raw) > 100_000:
        raise RealDataValidationError("Maximum 100,000 rows per upload in this prototype.")
    if raw.columns.duplicated().any():
        raise RealDataValidationError("Duplicate CSV column headers are not supported.")
    missing = [col for col in REQUIRED_COLUMNS if col not in raw]
    if missing:
        raise RealDataValidationError("Missing required column(s): " + ", ".join(missing))
    if not any(col in raw for col in MEASUREMENTS):
        raise RealDataValidationError("Supply at least one measured degradation indicator (C, ESR, leakage, expansion, H2, or pressure).")

    # Drop unrecognized and user-supplied derived columns to avoid exporting
    # untrusted spreadsheet formulas or relying on precomputed ratios.
    df = raw[[c for c in TEMPLATE_COLUMNS if c in raw]].copy()
    df["Cap_ID"] = _validate_capacitor_ids(df["Cap_ID"])
    present = ["Hours", *(c for c in OPTIONAL_COLUMNS if c in df)]
    for col in present:
        df[col] = _valid_numeric_series(df[col], col)

    if df["Hours"].isna().any() or (df["Hours"] < 0).any():
        raise RealDataValidationError("Hours must be present and nonnegative for every row.")

    positive = ("Capacitance_uF", "ESR_ohm", "Rated_V", "Rated_Ripple_A", "ESR_Test_Hz", "Ripple_Hz",
                "Pressure_Abs_kPa")
    nonnegative = ("Leakage_uA", "Ripple_A", "Voltage_V", "H2_Generated_mL_STP")
    for col in positive:
        if col in df and df[col].notna().any() and (df[col].dropna() <= 0).any():
            raise RealDataValidationError(f"{col} must be positive when provided.")
    for col in nonnegative:
        if col in df and df[col].notna().any() and (df[col].dropna() < 0).any():
            raise RealDataValidationError(f"{col} must be nonnegative when provided.")

    df = df.sort_values(["Cap_ID", "Hours"], kind="mergesort").reset_index(drop=True)
    if df.duplicated(["Cap_ID", "Hours"]).any():
        raise RealDataValidationError("Duplicate capacitor/time entries: each Cap_ID and Hours pair must be unique.")

    for cap_id, group in df.groupby("Cap_ID", sort=True):
        if len(group) < 2:
            raise RealDataValidationError(f"{cap_id}: at least two chronological readings are required.")
        if not any(col in group and group[col].notna().sum() >= 2 for col in MEASUREMENTS):
            raise RealDataValidationError(
                f"{cap_id}: at least two available readings of one degradation indicator are required."
            )
    return df


ESR_STATE_WORDING = {"incomplete documentation": "only partly documented", "undocumented": "not documented"}


def _esr_comparability(group: pd.DataFrame) -> str:
    """Return documented, undocumented, or inconsistent conditions.

    Only comparisons at the same recorded frequency and temperature are
    considered documented; if neither is documented, flag as provisional.
    """
    if "ESR_ohm" not in group:
        return "not measured"
    observed = group[group["ESR_ohm"].notna()]
    if observed.empty:
        return "not measured"
    supplied = []
    for col in ("ESR_Test_Hz", "ESR_Test_Temp_C"):
        if col not in group or observed[col].isna().all():
            supplied.append(False)
            continue
        supplied.append(True)
        if observed[col].isna().any() or observed[col].nunique(dropna=True) > 1:
            return "inconsistent"
    if all(supplied):
        return "documented"
    if any(supplied):
        return "incomplete documentation"
    return "undocumented"


def assess_real_measurements(raw: pd.DataFrame) -> dict:
    """Return an upload-wide quality/observed-change report, no forecasting."""
    df = validate_real_measurements(raw)
    # Ignore unverified user-supplied ratios (if any). Compute afresh.
    for name in RATIO_COLUMNS.values():
        df[name] = np.nan
    # Preserve missing/supplied column distinctions separately for quality.
    quality_rows, change_rows, warnings = [], [], []
    for cap_id, indices in df.groupby("Cap_ID", sort=True).groups.items():
        group = df.loc[indices].sort_values("Hours")
        n = len(group)
        hour_zero = group[group["Hours"] == 0]
        esr_state = _esr_comparability(group)
        if esr_state == "inconsistent":
            warnings.append(
                f"{cap_id}: ESR test frequency or test temperature is inconsistent/missing across readings. "
                "ESR ratios and ESR percentage changes were withheld."
            )
        elif esr_state != "documented" and esr_state != "not measured":
            warnings.append(
                f"{cap_id}: ESR test conditions are {ESR_STATE_WORDING.get(esr_state, esr_state)}; "
                "ESR comparisons are provisional, "
                "not confirmed as degradation."
            )

        for col, (display_name, unit) in MEASUREMENTS.items():
            column_was_supplied = col in raw
            available = int(group[col].notna().sum()) if col in group else 0
            missing = n - available
            status = (
                "Not supplied" if not column_was_supplied else
                "No measurements" if available == 0 else
                "Insufficient for change" if available < 2 else
                "Partial history" if missing else "Complete history"
            )
            quality_rows.append({
                "Cap_ID": cap_id, "Indicator": col, "Available": available,
                "Missing": missing, "Completeness_percent": round(100.0 * available / n, 2),
                "Status": status,
            })

            change = {
                "Cap_ID": cap_id, "Indicator": display_name,
                "Unit": unit, "First_Hour": np.nan, "Last_Hour": np.nan,
                "First_Value": np.nan, "Last_Value": np.nan,
                "Observed_Change_percent": np.nan, "Observed_Change_um": np.nan,
                "Observed_Change_mL_STP": np.nan, "Observed_Change_kPa": np.nan,
                "Status": "Not available",
            }
            if available < 2:
                change["Status"] = "Not measured" if available == 0 else "Insufficient readings"
                change_rows.append(change)
                continue

            measurements = group.dropna(subset=[col])
            first, last = measurements.iloc[0], measurements.iloc[-1]
            change.update({
                "First_Hour": float(first["Hours"]), "Last_Hour": float(last["Hours"]),
                "First_Value": float(first[col]), "Last_Value": float(last[col]),
            })

            baseline = hour_zero.iloc[0][col] if not hour_zero.empty and col in hour_zero else np.nan
            if pd.isna(baseline):
                change["Status"] = "No measured hour-zero baseline; change withheld"
                warnings.append(f"{cap_id}: {col} has no measured hour-zero baseline.")
                change_rows.append(change)
                continue

            if col == "Case_Expansion_um":
                change["Observed_Change_um"] = float(last[col] - baseline)
                change["Status"] = "Observed expansion change only (not EOL)"
            elif col == "H2_Generated_mL_STP":
                change["Observed_Change_mL_STP"] = float(last[col] - baseline)
                change["Status"] = "Measured H₂ change only (not EOL)"
            elif col == "Pressure_Abs_kPa":
                change["Observed_Change_kPa"] = float(last[col] - baseline)
                change["Status"] = "Measured absolute pressure change only (not EOL)"
            elif col == "ESR_ohm" and esr_state == "inconsistent":
                change["Status"] = "Incomparable ESR test conditions; change withheld"
            elif baseline <= 0:
                change["Status"] = "Nonpositive baseline; ratio undefined"
                warnings.append(f"{cap_id}: {col} has a zero baseline, so its ratio is undefined.")
            else:
                ratio_col = RATIO_COLUMNS[col]
                df.loc[measurements.index, ratio_col] = measurements[col] / baseline
                change["Observed_Change_percent"] = 100.0 * (float(last[col]) / float(baseline) - 1.0)
                change["Status"] = (
                    "Provisional (ESR test conditions unverified)"
                    if col == "ESR_ohm" and esr_state != "documented" else
                    "Observed change only (not EOL)"
                )
            change_rows.append(change)

    # Measured gas values are never substituted with physics-derived estimates.
    if "H2_Generated_mL_STP" in df and df["H2_Generated_mL_STP"].notna().any():
        warnings.append("H₂ input is interpreted as cumulative generated hydrogen in mL at 0°C and 1 atm; verify measurement method and gas reference conditions.")
    if "Pressure_Abs_kPa" in df and df["Pressure_Abs_kPa"].notna().any():
        warnings.append("Pressure input is interpreted as absolute kPa (not gauge); verify reference pressure and sensor calibration.")
    return {
        "measurements": df,
        "quality": pd.DataFrame(quality_rows),
        "changes": pd.DataFrame(change_rows),
        "warnings": list(dict.fromkeys(warnings)),
    }


def assessment_zip_bytes(result: dict) -> bytes:
    """Build export in memory without keeping user's upload on disk."""
    readme = (
        "CAPLIFE AI — EXPERIMENTAL MEASUREMENT SUMMARY\n\n"
        "This report describes observed measurements only.\n"
        "It does NOT provide GRU inference, RUL prediction, calibrated uncertainty,\n"
        "health scoring, EOL decisions, or maintenance advice.\n"
        "Missing values were not interpolated or filled.\n"
        "Ratios require measured hour-zero baselines.\n"
        "ESR ratios may be provisional when test frequency or temperature is undocumented.\n"
        "Review sensor calibration, units, test conditions and data provenance separately.\n"
    )
    stream = BytesIO()
    with ZipFile(stream, mode="w", compression=ZIP_DEFLATED) as archive:
        for filename, key in (
            ("processed_measurements.csv", "measurements"),
            ("measurement_quality.csv", "quality"),
            ("observed_degradation.csv", "changes"),
        ):
            archive.writestr(filename, result[key].to_csv(index=False))
        archive.writestr("README.txt", readme)
        if result["warnings"]:
            archive.writestr("measurement_warnings.txt", "\n".join(result["warnings"]) + "\n")
    return stream.getvalue()
