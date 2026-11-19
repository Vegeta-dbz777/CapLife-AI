"""Descriptive assessment of uploaded AEC measurements (CapLife AI v0.7).

Does not call GRU inference, compute RUL, apply EOL thresholds, or fabricate data.
Compatible with pandas and numpy versions in the existing CapLife AI project.
"""
from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Cap_ID", "Hours")
MEASUREMENTS = {
    "Capacitance_uF": ("Capacitance", "µF"),
    "ESR_ohm": ("ESR", "Ω"),
    "Leakage_uA": ("Leakage current", "µA"),
    "Case_Expansion_um": ("Case expansion", "µm"),
}
OPTIONAL_COLUMNS = (
    "Capacitance_uF", "ESR_ohm", "Leakage_uA", "Ambient_C",
    "Core_C", "Voltage_V", "Rated_V", "Ripple_A", "Ripple_Hz",
    "ESR_Test_Hz", "ESR_Test_Temp_C", "Case_Expansion_um",
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
        raise RealDataValidationError("Supply at least one measured degradation indicator (C, ESR, leakage, or expansion).")

    # Drop unrecognized and user-supplied derived columns to avoid exporting
    # untrusted spreadsheet formulas or relying on precomputed ratios.
    df = raw[[c for c in TEMPLATE_COLUMNS if c in raw]].copy()
    ids = df["Cap_ID"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise RealDataValidationError("Cap_ID must not be blank.")
    # Protect downloaded CSV from spreadsheet formula injection through IDs.
    if ids.str.startswith(("=", "+", "-", "@"), na=False).any():
        raise RealDataValidationError("Cap_ID may not begin with =, +, -, or @.")
    df["Cap_ID"] = ids.astype(str)
    present = ["Hours", *(c for c in OPTIONAL_COLUMNS if c in df)]
    for col in present:
        df[col] = _valid_numeric_series(df[col], col)

    if df["Hours"].isna().any() or (df["Hours"] < 0).any():
        raise RealDataValidationError("Hours must be present and nonnegative for every row.")

    positive = ("Capacitance_uF", "ESR_ohm", "Rated_V", "ESR_Test_Hz", "Ripple_Hz")
    nonnegative = ("Leakage_uA", "Ripple_A", "Voltage_V")
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
                f"{cap_id}: ESR test conditions are {esr_state}; ESR comparisons are provisional, "
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

    return {
        "measurements": df,
        "quality": pd.DataFrame(quality_rows),
        "changes": pd.DataFrame(change_rows),
        "warnings": list(dict.fromkeys(warnings)),
    }


def assessment_zip_bytes(result: dict) -> bytes:
    """Build export in memory without keeping user's upload on disk."""
    readme = (
        "CAPLIFE AI v0.6 — EXPERIMENTAL MEASUREMENT SUMMARY\n\n"
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
            archive.writestr(filename, result[key].to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("README.txt", readme)
        if result["warnings"]:
            archive.writestr("measurement_warnings.txt", "\n".join(result["warnings"]) + "\n")
    return stream.getvalue()
