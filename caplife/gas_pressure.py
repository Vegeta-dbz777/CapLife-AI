"""CapLife AI v0.8-preview: assumed hydrogen/pressure evolution scenarios.

This module is a *physics illustration*, not measurement reconstruction,
vent-opening prediction, or validated capacitor lifetime estimation.

Assumptions:
- Measured leakage current is only a proxy for Faradaic H2-producing current.
- A user-selected fraction of it is assumed to generate hydrogen.
- A fixed fraction of generated H2 is assumed to remain as gas.
- The gas headspace is fixed and the gas is ideal.
- The initial headspace gas amount is constant. Temperature scaling is included.
- Vapor, diffusion kinetics, dissolution, electrolyte chemistry, mechanical
  deformation, and actual vent behavior are NOT modeled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FARADAY_C_PER_MOL = 96485.33212
R_J_PER_MOL_K = 8.314462618
STP_ML_PER_MOL = 22414.0  # 0 C, 1 atm (ideal gas, rounded)


class GasPressureInputError(ValueError):
    """The requested scenario cannot be calculated without assumptions or data."""


def calculate_h2_pressure_history(
    measurements: pd.DataFrame,
    *,
    headspace_ml: float,
    initial_pressure_abs_kpa: float,
    h2_current_fraction: float,
    retained_fraction: float,
) -> pd.DataFrame:
    """Integrate an illustrative H2 generation history and ideal-gas pressure.

    Returns calculations at existing observation hours, without making a
    genuine continuous-time prediction between or beyond measured samples.
    All model-based output columns end with ``_est``.

    Initial pressure is **absolute**, not gauge. The first reading must be
    at 0 h, because P0 and initial gas moles are anchored there.
    """
    if not isinstance(measurements, pd.DataFrame):
        raise GasPressureInputError("Expected a pandas measurement table.")
    required = ("Hours", "Leakage_uA", "Core_C")
    missing = [col for col in required if col not in measurements.columns]
    if missing:
        raise GasPressureInputError("Required for scenario: " + ", ".join(missing))
    if not len(measurements):
        raise GasPressureInputError("At least two observations are required.")
    if "Cap_ID" in measurements and measurements["Cap_ID"].nunique(dropna=False) != 1:
        raise GasPressureInputError("Run the scenario for one capacitor at a time.")

    for name, val in (
        ("headspace_ml", headspace_ml),
        ("initial_pressure_abs_kpa", initial_pressure_abs_kpa),
        ("h2_current_fraction", h2_current_fraction),
        ("retained_fraction", retained_fraction),
    ):
        if not isinstance(val, (int, float, np.number)) or not np.isfinite(val):
            raise GasPressureInputError(f"{name} must be a finite number.")
    if headspace_ml <= 0 or initial_pressure_abs_kpa <= 0:
        raise GasPressureInputError("Headspace and initial absolute pressure must be positive.")
    if not (0 <= h2_current_fraction <= 1 and 0 <= retained_fraction <= 1):
        raise GasPressureInputError("Hydrogen-producing and retention fractions must be between 0 and 1.")

    frame = measurements.sort_values("Hours", kind="mergesort").copy().reset_index(drop=True)
    if len(frame) < 2:
        raise GasPressureInputError("At least two chronological observations are required.")
    for field in required:
        try:
            frame[field] = pd.to_numeric(frame[field], errors="raise")
        except (ValueError, TypeError) as exc:
            raise GasPressureInputError(f"{field} must contain numerical readings.") from exc
        if frame[field].isna().any() or not np.isfinite(frame[field].to_numpy(dtype=float)).all():
            raise GasPressureInputError(
                f"{field} has missing or nonfinite observations. "
                "This preview does not fill measurement gaps."
            )

    hours = frame["Hours"].to_numpy(dtype=float)
    leakage = frame["Leakage_uA"].to_numpy(dtype=float)
    core_c = frame["Core_C"].to_numpy(dtype=float)
    if hours[0] != 0 or np.any(np.diff(hours) <= 0):
        raise GasPressureInputError("Need a 0-hour reading and strictly increasing operating hours.")
    if np.any(leakage < 0):
        raise GasPressureInputError("Leakage_uA must not be negative.")
    if np.any(core_c <= -273.15):
        raise GasPressureInputError("Core_C must be above absolute zero.")

    # Trapezoidal integration of sampled current: assumes linear change between samples.
    delta_hours = np.diff(hours)
    q_intervals_c = (leakage[:-1] + leakage[1:]) / 2.0 * 1e-6 * (delta_hours * 3600.0)
    q_total_c = np.r_[0.0, np.cumsum(q_intervals_c)]

    # 2 mol electrons per 1 mol H2: an assumed electrochemical conversion.
    mol_h2_generated = q_total_c * h2_current_fraction / (2.0 * FARADAY_C_PER_MOL)
    mol_h2_retained = mol_h2_generated * retained_fraction
    temperature_k = core_c + 273.15
    headspace_m3 = headspace_ml * 1e-6

    # Constant amount of initial non-H2 gas; temperature effects on it included.
    baseline_gas_kpa = initial_pressure_abs_kpa * temperature_k / temperature_k[0]
    h2_partial_kpa = mol_h2_retained * R_J_PER_MOL_K * temperature_k / headspace_m3 / 1000.0
    pressure_abs_kpa = baseline_gas_kpa + h2_partial_kpa

    result = pd.DataFrame({
        "Hours": hours,
        "Leakage_uA": leakage,
        "Core_C": core_c,
        "Integrated_Charge_C_est": q_total_c,
        "H2_Generated_mmol_est": mol_h2_generated * 1000.0,
        "H2_Generated_mL_STP_est": mol_h2_generated * STP_ML_PER_MOL,
        "H2_Retained_mmol_est": mol_h2_retained * 1000.0,
        "H2_PartialPressure_kPa_est": h2_partial_kpa,
        "Pressure_Abs_kPa_est": pressure_abs_kpa,
        "Pressure_Change_From_Initial_kPa_est": pressure_abs_kpa - initial_pressure_abs_kpa,
    })
    if "Cap_ID" in frame:
        result.insert(0, "Cap_ID", frame["Cap_ID"].to_numpy())
    return result
