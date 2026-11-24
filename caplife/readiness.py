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
    ("Rated_Ripple_A", "Rated ripple current", "A"),
    ("Ripple_Hz", "Operating (ripple) frequency", "Hz"),
    ("ESR_Test_Hz", "ESR measurement frequency", "Hz"),
    ("ESR_Test_Temp_C", "ESR measurement temperature", "°C"),
)


FILE_SCOPE, CAP_SCOPE = "File", "Capacitor"
PRIORITY_ORDER = {"Important": 0, "Review": 1, "Documentation": 2}
SHOWN_IDS = 3  # longer capacitor lists are shortened to "A, B, C + 18 more"; exports keep every row


def _append_action(actions, cap_id, category, recommendation, priority="Review", scope=CAP_SCOPE):
    actions.append({
        "Cap_ID": cap_id,
        "Category": category,
        "Priority": priority,
        "Recommendation": recommendation,
        "Scope": scope,
    })


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _h(hours: float) -> str:
    return f"{hours:,.0f} h"


def _hour_list(hours, limit: int = 3) -> str:
    shown = ", ".join(f"{value:,.0f}" for value in list(hours)[:limit])
    return f"{shown}{', …' if len(hours) > limit else ''} h"


def audit_measurement_readiness(report: dict) -> dict:
    """Audit what can be *described*, not whether a capacitor is healthy.

    Input is assess_real_measurements(...) output from caplife.real_data.
    The audit avoids universal sensor/aging cutoffs. Five observations is a
    tutorial-oriented flag for limited history, not an equipment standard.

    Capacitor-specific findings name their own numbers (which parameter, how many readings, which
    hours). Documentation advice that holds for every capacitor measuring a parameter has scope
    "File"; ``grouped_actions`` shows each such item once.
    """
    df = report["measurements"]
    changes = report["changes"]
    actions, summaries, context_rows = [], [], []

    for cap_id, g in df.groupby("Cap_ID", sort=True):
        g = g.sort_values("Hours")
        n = len(g)
        hours = g["Hours"].to_numpy(dtype=float)
        c = changes.loc[changes["Cap_ID"] == cap_id]
        compared = int(c[["Observed_Change_percent", "Observed_Change_um", "Observed_Change_mL_STP", "Observed_Change_kPa"]].notna().any(axis=1).sum())
        supplied = [col for col, _ in DEGRADATION if col in g and g[col].notna().any()]
        missing_among_recorded = sum(int(g[col].isna().sum()) for col in supplied)
        possible_among_recorded = n * len(supplied)
        coverage = round(100 * (1 - missing_among_recorded / possible_among_recorded), 2) if possible_among_recorded else 0.0
        baseline = bool((g["Hours"] == 0).any())
        limits = []

        if not baseline:
            _append_action(actions, cap_id, "Baseline",
                           f"No reading at 0 h (the first is at {_h(hours[0])}), so changes from new can't be "
                           "calculated. Add the 0 h reading if you have it.")
            limits.append(f"no 0 h reading (first at {_h(hours[0])})")
        for col, label in DEGRADATION:
            if col not in g or g[col].notna().sum() == 0:
                continue  # Channels never measured are not counted as failed sensors.
            if baseline and g.loc[g["Hours"] == 0, col].isna().all():
                first = float(g.loc[g[col].notna(), "Hours"].iloc[0])
                _append_action(actions, cap_id, "Baseline",
                               f"{label} has no 0 h reading (first at {_h(first)}). Add it if you have it; don't estimate it.")
                limits.append(f"{label}: no 0 h reading")
            gaps = g.loc[g[col].isna(), "Hours"]
            if len(gaps):
                _append_action(actions, cap_id, "Missing readings",
                               f"{label} is missing at {len(gaps)} of {n} readings ({_hour_list(gaps)}). "
                               "Leave them blank and note why.")
                limits.append(f"{label}: {len(gaps)} of {n} missing")

        if n < 5:
            _append_action(actions, cap_id, "History length",
                           f"Only {n} readings ({_h(hours[0])} to {_h(hours[-1])}). More readings would show the trend more clearly.")
            limits.append(f"only {n} readings")

        spacings = np.diff(hours)
        if len(spacings) >= 3 and np.median(spacings) > 0 and float(np.max(spacings)) > 3 * float(np.median(spacings)):
            at = int(np.argmax(spacings))
            _append_action(actions, cap_id, "Sampling gaps",
                           f"{_h(spacings[at])} gap between {_h(hours[at])} and {_h(hours[at + 1])} "
                           f"(usual spacing {_h(np.median(spacings))}). Check for missed readings.")
            limits.append(f"{_h(spacings[at])} gap after {_h(hours[at])}")

        esr = c[c["Indicator"] == "ESR"]
        if not esr.empty and g.get("ESR_ohm", pd.Series(dtype=float)).notna().sum() >= 2:
            esr_status = str(esr.iloc[0]["Status"])
            if "Incomparable" in esr_status:
                _append_action(actions, cap_id, "ESR comparability",
                               "ESR test frequency or temperature changes between readings or is partly missing, "
                               "so the ESR change may not be aging.", "Important")
                limits.append("ESR test settings vary")
            elif "Provisional" in esr_status:
                _append_action(actions, cap_id, "ESR comparability",
                               "Record the ESR test frequency and temperature at each reading; until then the ESR change is provisional.",
                               scope=FILE_SCOPE)

        if "H2_Generated_mL_STP" in supplied:
            _append_action(actions, cap_id, "Gas measurement provenance",
                           "Note whether H₂ is total generated gas or gas remaining, and the reference temperature, "
                           "pressure and collection method.", "Documentation", FILE_SCOPE)
        if "Pressure_Abs_kPa" in supplied:
            _append_action(actions, cap_id, "Pressure measurement provenance",
                           "Note whether pressure is absolute or gauge, the ambient reference, sensor calibration, "
                           "probe position and operating temperature.", "Documentation", FILE_SCOPE)
        supplied_context = [col for col, _, _ in CONTEXT if col in g and g[col].notna().any()]
        if not any(col in supplied_context for col in ("Ambient_C", "Core_C", "Voltage_V", "Ripple_A", "Ripple_Hz")):
            _append_action(actions, cap_id, "Operating conditions",
                           "Record operating temperature, voltage and ripple with the measurements.", scope=FILE_SCOPE)

        # Capacitance and leakage comparability depend on their test protocols as well.
        if any(col in supplied for col in ("Capacitance_uF", "Leakage_uA")):
            _append_action(actions, cap_id, "Measurement protocol",
                           "Record how capacitance and leakage were measured (frequency, temperature, voltage, settling time).",
                           "Documentation", FILE_SCOPE)

        for col, label, unit in CONTEXT:
            observed = g[col].dropna() if col in g else pd.Series(dtype=float)
            context_rows.append({
                "Cap_ID": cap_id, "Condition": label, "Unit": unit,
                "Available": int(len(observed)),
                "Min_observed": float(observed.min()) if len(observed) else np.nan,
                "Max_observed": float(observed.max()) if len(observed) else np.nan,
            })

        if compared == 0:
            state = "Not comparable"
        elif limits:  # each limit is spelled out in the Limits column
            state = "Limited"
        else:
            state = "Ready"
        summaries.append({
            "Cap_ID": cap_id,
            "Data_readiness": state,
            "Limits": _sentence_case("; ".join(limits)) if limits else "—",
            "Measurements": n,
            "First_hour": float(hours[0]),
            "Last_hour": float(hours[-1]),
            "Recorded_indicators": len(supplied),
            "Observed_coverage_percent": coverage,
            "Hour_zero_recorded": baseline,
            "Baseline_based_comparisons": compared,
        })

    return {
        "summary": pd.DataFrame(summaries),
        "actions": pd.DataFrame(actions, columns=["Cap_ID", "Category", "Priority", "Recommendation", "Scope"]),
        "conditions": pd.DataFrame(context_rows),
    }


def grouped_actions(actions: pd.DataFrame, all_caps) -> pd.DataFrame:
    """Each distinct recommendation once, with the capacitors it applies to; most urgent first."""
    columns = ["Applies_to", "Recommendation", "Priority", "Category"]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    total = len(set(all_caps))
    rows = []
    for (category, priority, text), group in actions.groupby(["Category", "Priority", "Recommendation"], sort=False):
        caps = list(dict.fromkeys(group["Cap_ID"].astype(str)))
        if len(caps) == total and total > 1:
            applies = f"All {total} capacitors"
        elif len(caps) > SHOWN_IDS + 1:
            applies = f"{', '.join(caps[:SHOWN_IDS])} + {len(caps) - SHOWN_IDS} more"
        else:
            applies = ", ".join(caps)
        rows.append({"Applies_to": applies, "Category": category, "Priority": priority, "Recommendation": text,
                     "_order": (PRIORITY_ORDER.get(priority, 1), group["Scope"].iloc[0] == FILE_SCOPE, -len(caps))})
    frame = pd.DataFrame(rows)
    frame = frame.iloc[sorted(range(len(frame)), key=lambda i: frame["_order"].iloc[i])]
    return frame[columns].reset_index(drop=True)


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
            "CAPLIFE AI RESEARCH-DATA AUDIT\n\n"
            "Readiness describes documentation and data availability ONLY.\n"
            "It is not a health score, reliability estimate, EOL finding, RUL forecast,\n"
            "calibrated uncertainty interval, or recommendation for maintenance.\n"
            "Incomplete laboratory records and test-condition changes can undermine\n"
            "even apparently complete datasets. No values have been fabricated.\n",
        )
    return dest.getvalue()
