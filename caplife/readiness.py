"""CapLife AI v0.8: transparent, non-predictive research-data readiness audit.

Runs only after v0.6 real-measurement validation. Never produces EOL/RUL,
health scores, confidence intervals, fabricated readings, or AI predictions.
"""
from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

DEGRADATION = (
    ("Capacitance_uF", "Capacitance"),
    ("ESR_ohm", "ESR"),
    ("Leakage_uA", "Leakage current"),
    ("Case_Expansion_um", "Case expansion"),
    ("H2_Generated_mL_STP", "H₂ generation (measured)"),
    ("Pressure_Abs_kPa", "Internal pressure (measured)"),
)
CONTEXT = (
    ("Ambient_C", "Ambient temperature", "°C"),
    ("Core_C", "Core temperature", "°C"),
    ("Voltage_V", "Applied voltage", "V"),
    ("Rated_V", "Rated voltage", "V"),
    ("Ripple_A", "Ripple current", "A"),
    ("Ripple_Hz", "Ripple frequency", "Hz"),
    ("ESR_Test_Hz", "ESR measurement frequency", "Hz"),
    ("ESR_Test_Temp_C", "ESR measurement temperature", "°C"),
)


def _append_action(actions, cap_id, category, recommendation, priority="Review"):
    actions.append({
        "Cap_ID": cap_id,
        "Category": category,
        "Priority": priority,
        "Recommendation": recommendation,
    })


def audit_measurement_readiness(report: dict) -> dict:
    """Audit what can be *described*, not whether a capacitor is healthy.

    Input is assess_real_measurements(...) output from caplife.real_data.
    The audit avoids universal sensor/aging cutoffs. Five observations is a
    tutorial-oriented flag for limited history, not an equipment standard.
    """
    df = report["measurements"]
    changes = report["changes"]
    actions, summaries, context_rows = [], [], []

    for cap_id, g in df.groupby("Cap_ID", sort=True):
        g = g.sort_values("Hours")
        n = len(g)
        c = changes.loc[changes["Cap_ID"] == cap_id]
        compared = int(c[["Observed_Change_percent", "Observed_Change_um", "Observed_Change_mL_STP", "Observed_Change_kPa"]].notna().any(axis=1).sum())
        supplied = [col for col, _ in DEGRADATION if col in g and g[col].notna().any()]
        missing_among_recorded = sum(int(g[col].isna().sum()) for col in supplied)
        possible_among_recorded = n * len(supplied)
        coverage = round(100 * (1 - missing_among_recorded / possible_among_recorded), 2) if possible_among_recorded else 0.0
        baseline = bool((g["Hours"] == 0).any())

        if not baseline:
            _append_action(actions, cap_id, "Baseline", "A measured 0-hour baseline is absent. Ratios and percent changes cannot be established without a valid reference.")
        for col, label in DEGRADATION:
            if col not in g or g[col].notna().sum() == 0:
                continue  # Channels never measured are not counted as failed sensors.
            if not baseline or g.loc[g["Hours"] == 0, col].isna().all():
                _append_action(actions, cap_id, "Baseline", f"{label}: record a genuine reference reading, if available; do not invent it.")
            missing = int(g[col].isna().sum())
            if missing:
                _append_action(actions, cap_id, "Missing readings", f"{label}: {missing} of {n} time points have no observation. Leave them blank and document why.")

        if n < 5:
            _append_action(actions, cap_id, "History length", f"Only {n} observations are recorded. This permits basic comparisons but provides limited evidence for a stable degradation pattern.")

        hours = g["Hours"].to_numpy(dtype=float)
        spacings = np.diff(hours)
        if len(spacings) >= 3 and np.median(spacings) > 0 and float(np.max(spacings)) > 3 * float(np.median(spacings)):
            _append_action(actions, cap_id, "Sampling gaps", "A sampling interval exceeds three times the median interval. Check for missed measurements or intentional schedule changes.")

        esr = c[c["Indicator"] == "ESR"]
        if not esr.empty and g.get("ESR_ohm", pd.Series(dtype=float)).notna().sum() >= 2:
            esr_status = str(esr.iloc[0]["Status"])
            if "Incomparable" in esr_status:
                _append_action(actions, cap_id, "ESR comparability", "ESR test settings differ or are incompletely recorded. Do not interpret the ESR change as aging until measurements are made comparable.", "Important")
            elif "Provisional" in esr_status:
                _append_action(actions, cap_id, "ESR comparability", "Record ESR measurement frequency AND temperature at each test point. Current ESR change is provisional.")

        if "H2_Generated_mL_STP" in supplied:
            _append_action(actions, cap_id, "Gas measurement provenance", "Document whether H₂ is cumulative generated gas or gas remaining, plus reference temperature, pressure and collection method.", "Documentation")
        if "Pressure_Abs_kPa" in supplied:
            _append_action(actions, cap_id, "Pressure measurement provenance", "Document absolute-vs-gauge pressure, ambient reference, sensor calibration, internal probe geometry and operating temperature.", "Documentation")
        supplied_context = [col for col, _, _ in CONTEXT if col in g and g[col].notna().any()]
        if not any(col in supplied_context for col in ("Ambient_C", "Core_C", "Voltage_V", "Ripple_A", "Ripple_Hz")):
            _append_action(actions, cap_id, "Operating conditions", "Document operating temperature, voltage and ripple conditions alongside measurements, where available.")

        # Capacitance and leakage comparability depend on their test protocols as well.
        if any(col in supplied for col in ("Capacitance_uF", "Leakage_uA")):
            _append_action(actions, cap_id, "Measurement protocol", "Keep the capacitance/leakage test method (frequency, temperature, voltage and settling time as applicable) in your laboratory record.", "Documentation")

        for col, label, unit in CONTEXT:
            observed = g[col].dropna() if col in g else pd.Series(dtype=float)
            context_rows.append({
                "Cap_ID": cap_id, "Condition": label, "Unit": unit,
                "Available": int(len(observed)),
                "Min_observed": float(observed.min()) if len(observed) else np.nan,
                "Max_observed": float(observed.max()) if len(observed) else np.nan,
            })

        if compared == 0:
            state = "No baseline-based comparison available"
        elif n < 5 or missing_among_recorded > 0 or not baseline:
            state = "Limited descriptive comparison"
        else:
            state = "Descriptive comparison available"
        summaries.append({
            "Cap_ID": cap_id,
            "Measurements": n,
            "First_hour": float(hours[0]),
            "Last_hour": float(hours[-1]),
            "Recorded_indicators": len(supplied),
            "Observed_coverage_percent": coverage,
            "Hour_zero_recorded": baseline,
            "Baseline_based_comparisons": compared,
            "Data_readiness": state,
        })

    return {
        "summary": pd.DataFrame(summaries),
        "actions": pd.DataFrame(actions, columns=["Cap_ID", "Category", "Priority", "Recommendation"]),
        "conditions": pd.DataFrame(context_rows),
    }


def assessment_zip_with_readiness_bytes(report: dict, audit: dict) -> bytes:
    """Extend the v0.6 export with an audit, retaining every existing file."""
    from .real_data import assessment_zip_bytes

    old_bytes = assessment_zip_bytes(report)
    dest = BytesIO()
    with ZipFile(BytesIO(old_bytes), "r") as old, ZipFile(dest, "w", ZIP_DEFLATED) as new:
        for name in old.namelist():
            new.writestr(name, old.read(name))
        for filename, key in (
            ("research_data_readiness.csv", "summary"),
            ("measurement_review_actions.csv", "actions"),
            ("recorded_operating_conditions.csv", "conditions"),
        ):
            new.writestr(filename, audit[key].to_csv(index=False))
        new.writestr(
            "READINESS_NOTE.txt",
            "CAPLIFE AI v0.8 RESEARCH-DATA AUDIT\n\n"
            "Readiness describes documentation and data availability ONLY.\n"
            "It is not a health score, reliability estimate, EOL finding, RUL forecast,\n"
            "calibrated uncertainty interval, or recommendation for maintenance.\n"
            "Incomplete laboratory records and test-condition changes can undermine\n"
            "even apparently complete datasets. No values have been fabricated.\n",
        )
    return dest.getvalue()
