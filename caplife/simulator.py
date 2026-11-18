"""Synthetic trajectories for software prototyping, not calibrated capacitor physics.

Degradation laws and cutoff values below are deliberately invented. Do not use
this simulator to support scientific performance claims or maintenance decisions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import EOL_THRESHOLDS


def end_of_life_causes(row: dict) -> list[str]:
    t = EOL_THRESHOLDS
    return [
        label for label, triggered in [
            ("capacitance", row["C_ratio"] <= t["C_ratio_min"]),
            ("ESR", row["ESR_ratio"] >= t["ESR_ratio_max"]),
            ("leakage", row["Leakage_ratio"] >= t["Leakage_ratio_max"]),
            ("casing strain", row["Case_Strain_Index"] >= t["Case_Strain_Index_max"]),
        ] if triggered
    ]


def simulate_capacitors(n_caps: int = 160, seed: int = 42,
                        max_hours: int = 15000, step_hours: int = 100) -> pd.DataFrame:
    """Produce chronological, multi-mode synthetic data with known/unknown EOL.

    Each capacitor receives its own stress environment and failure susceptibilities.
    Future EOL labels are populated *only when an EOL event was observed*.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cap_id in range(1, n_caps + 1):
        ambient_base = rng.uniform(12, 50)
        ripple_base = rng.uniform(0.8, 3.3)
        voltage_base = rng.uniform(0.72, 1.03)
        frequency = int(rng.choice([100, 300, 1000, 10000]))
        rates = rng.lognormal(0, 0.46, 4)
        # Four partially distinct degradation coordinates; units are arbitrary.
        dc = de = dl = ds = 0.0
        history = []
        observed_eol = None
        failure_cause = "unobserved"
        phase = rng.uniform(0, 2 * np.pi)
        for hour in range(0, max_hours + 1, step_hours):
            # The environmental conditions vary with time; long-term time-series
            # history can thus carry information beyond the final measurement.
            ambient = ambient_base + 4.0 * np.sin(hour / 1350 + phase)
            ripple = max(0.15, ripple_base * (1 + 0.09 * np.sin(hour / 890 + phase)))
            voltage = max(0.4, voltage_base + 0.015 * np.sin(hour / 1100 + phase))
            c_ratio = max(0.0, 1.0 - 0.20 * dc)
            esr_ratio = 1.0 + de
            leakage_ratio = 1.0 + 4.0 * dl
            strain = ds
            # Teaching-level Joule-heating proxy; NOT a validated thermal model.
            fmult = 1 + 0.08 * np.log10(frequency / 100)
            core = ambient + 8 * (ripple ** 2) * 0.10 * esr_ratio * fmult
            record = {
                "Cap_ID": f"AEC_{cap_id:03d}",
                "Hours": int(hour),
                "Ambient_C": round(float(ambient), 3),
                "Core_C": round(float(core), 3),
                "Voltage_ratio": round(float(voltage), 5),
                "Ripple_A": round(float(ripple), 5),
                "Ripple_Hz": frequency,
                "C_ratio": round(float(c_ratio), 6),
                "ESR_ratio": round(float(esr_ratio), 6),
                "Leakage_ratio": round(float(leakage_ratio), 6),
                "Case_Strain_Index": round(float(strain), 6),
            }
            history.append(record)
            causes = end_of_life_causes(record)
            if causes:
                observed_eol = hour
                failure_cause = ", ".join(causes)
                break
            # Varying stress and independent susceptibilities yield different
            # observed reasons for EOL, but this is still invented toy physics.
            stress = np.clip(
                np.exp((core - 55) / 34) * (voltage / 0.85) ** 1.5,
                0.12, 3.8,
            )
            dt = step_hours
            dc += dt / 4700 * stress * rates[0]
            de += dt / 4700 * stress * (0.22 * rates[0] + 0.78 * rates[1])
            dl += dt / 4900 * stress * rates[2]
            ds += dt / 5200 * stress * rates[3]
        for rec in history:
            rec["RUL_hours"] = (
                float(observed_eol - rec["Hours"])
                if observed_eol is not None else np.nan
            )
            rec["EOL_observed"] = observed_eol is not None
            rec["EOL_cause"] = failure_cause
            rows.append(rec)
    return pd.DataFrame(rows)
