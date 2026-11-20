"""CapLife AI — v0.8 six-indicator descriptive audit + preserved synthetic GRU demo."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from caplife.config import EOL_THRESHOLDS, FEATURES, SEQ_LEN
from caplife.data import DataValidationError, validate_measurements
from caplife.predict import assess_capacitor, load_engine
from caplife.trend import assess_trend


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "sample_data" / "demo_capacitors.csv"
TEMPLATE_DATA = ROOT / "sample_data" / "upload_template.csv"
ARTIFACTS = ROOT / "artifacts"
UNCERTAINTY_JSON = ARTIFACTS / "trend_uncertainty_synthetic_v0_5.json"

st.set_page_config(
    page_title="CapLife AI — AEC RUL Prototype",
    page_icon="⚡",
    layout="wide",
)
st.title("⚡ CapLife AI")
st.caption("Aluminum Electrolytic Capacitor Health Monitoring & Remaining Useful Life")
st.warning(
    "**RESEARCH PROTOTYPE — NOT VALIDATED FOR REAL CAPACITORS.** "
    "The synthetic RUL mode uses a GRU trained on invented histories. "
    "The experimental-measurement mode provides descriptive checks and observed changes only. "
    "Neither mode provides validated maintenance or lifetime advice."
)


# v0.6 introduces a separate mode. The original GRU RUL page below is unchanged
# for users who select the synthetic demonstrator. Real-data uploads NEVER enter it.
with st.sidebar:
    st.subheader("Workflow")
    workflow = st.radio(
        "Choose an assessment mode",
        ("Experimental measurements (v0.8)", "Synthetic RUL demonstration (v0.5)"),
        key="caplife_workflow_v07",
    )
    st.divider()

if workflow == "Experimental measurements (v0.8)":
    from io import BytesIO
    from caplife.real_data import (
        RealDataValidationError,
        assess_real_measurements,
        blank_template_csv,
    )
    from caplife.readiness import (
        audit_measurement_readiness,
        assessment_zip_with_readiness_bytes,
    )

    st.subheader("1 · Upload experimental measurements")
    st.write(
        "Upload one CSV containing one or multiple capacitor histories. "
        "Only **Cap_ID** and **Hours** and at least one measured degradation "
        "indicator are required. Keep unavailable measurements blank."
    )
    st.download_button(
        "Download real-data CSV template",
        data=blank_template_csv(),
        file_name="CapLife_Real_Data_Template_v0_8.csv",
        mime="text/csv",
    )
    st.caption(
        "Measured columns: Capacitance_uF, ESR_ohm, Leakage_uA, "
        "Case_Expansion_um, H2_Generated_mL_STP, Pressure_Abs_kPa. "
        "The final two channels are optional DIRECT measurements: H₂ must be cumulative volume at 0°C/1 atm; pressure must be absolute kPa, not gauge. "
        "Optional: environmental, electrical, and ESR test conditions. "
        "Units must match the column names; do not upload normalized GRU features as raw measurements."
    )
    upload_real = st.file_uploader(
        "Upload real-data CSV (8 MB maximum)", type=["csv"], key="caplife_real_csv_v06"
    )
    if upload_real is None:
        st.info(
            "Choose a CSV to view measurement completeness, recorded changes, and graphs. "
            "The synthetic GRU is **not** run in this mode."
        )
        with st.expander("How to prepare measurement data"):
            st.write(
                "A row is one measurement for one capacitor at a stated operating hour. "
                "Include actual measured capacitance, ESR, leakage, or case expansion in the "
                "documented units. Supply a measured hour-zero baseline when possible. "
                "For meaningful ESR comparison, document the same ESR_Test_Hz and "
                "ESR_Test_Temp_C for the readings being compared. "
                "Do not fill unknown measurements with artificial values."
            )
        st.stop()

    if upload_real.size > 8 * 1024 * 1024:
        st.error("This prototype accepts at most 8 MB per uploaded file.")
        st.stop()
    try:
        uploaded_frame = pd.read_csv(BytesIO(upload_real.getvalue()), low_memory=False, dtype={"Cap_ID": str})
        real_report = assess_real_measurements(uploaded_frame)
    except (RealDataValidationError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        st.error(f"Upload could not be assessed: {exc}")
        st.info("Correct the source CSV and upload it again. No values were generated to replace missing readings.")
        st.stop()

    readings = real_report["measurements"]
    quality = real_report["quality"]
    changes = real_report["changes"]
    warnings = real_report["warnings"]
    n_caps = readings["Cap_ID"].nunique()
    metrics = st.columns(3)
    metrics[0].metric("Capacitors", str(n_caps))
    metrics[1].metric("Measurement rows", f"{len(readings):,}")
    metrics[2].metric("Measured channels", str(sum(
        c in uploaded_frame and uploaded_frame[c].notna().any()
        for c in ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um", "H2_Generated_mL_STP", "Pressure_Abs_kPa")
    )))
    st.success("Basic data checks passed. This does not establish measurement validity or real-world reliability.")
    for warning in warnings:
        st.warning(warning)

    st.subheader("2 · Measurement-quality report")
    st.caption("Zero recorded readings means 'not supplied' or 'no measurements' — not a zero sensor reading.")
    st.dataframe(quality, hide_index=True, use_container_width=True)

    st.subheader("2A · Research-data readiness (NOT a health score)")
    audit = audit_measurement_readiness(real_report)
    st.caption(
        "This audit flags missing baselines, sparse histories, unrecorded conditions "
        "and measurement gaps. It does not certify sensor accuracy or predict RUL."
    )
    st.dataframe(audit["summary"], hide_index=True, use_container_width=True)
    with st.expander("Recommended measurement and documentation improvements"):
        st.dataframe(audit["actions"], hide_index=True, use_container_width=True)
    with st.expander("Recorded operating and test conditions"):
        st.dataframe(audit["conditions"], hide_index=True, use_container_width=True)

    st.subheader("3 · Observed degradation (no forecasting)")
    st.dataframe(changes.round(3), hide_index=True, use_container_width=True)
    st.caption(
        "Changes are calculated only with an available hour-zero baseline; percentages "
        "refer to the observed raw measurement, not to remaining life. "
        "Case expansion is expressed as an absolute µm difference. "
        "Inconsistent ESR test conditions suppress the ESR ratio."
    )

    st.subheader("4 · Inspect a capacitor's measured history")
    cap_choice = st.selectbox("Capacitor ID", sorted(readings["Cap_ID"].unique()))
    cap_rows = readings.loc[readings["Cap_ID"] == cap_choice].sort_values("Hours")
    plot_groups = (
        ("C_ratio", "Capacitance ratio"),
        ("ESR_ratio", "ESR ratio"),
        ("Leakage_ratio", "Leakage current ratio"),
        ("Case_Expansion_um", "Case expansion (µm)"),
    )
    left, right = st.columns(2)
    for idx, (column, label) in enumerate(plot_groups):
        place = left if idx % 2 == 0 else right
        with place:
            st.markdown(f"**{label}**")
            if column not in cap_rows or cap_rows[column].notna().sum() == 0:
                st.caption("No comparable measured values are available.")
            else:
                # Scatter only: connecting points could conceal missing measurements.
                actual_points = cap_rows[["Hours", column]].dropna()
                st.scatter_chart(actual_points, x="Hours", y=column, height=240)
                missing_hours = cap_rows.loc[cap_rows[column].isna(), "Hours"].tolist()
                if missing_hours:
                    preview = ", ".join(f"{x:g}" for x in missing_hours[:12])
                    st.caption(
                        f"Missing / unavailable at {preview} h"
                        + (" (more omitted)" if len(missing_hours) > 12 else "")
                    )
    # Two additional measured channels. Missing channels remain missing;
    # no assumed physics output is passed off as sensor data.
    present_measured_gas = [
        ("H2_Generated_mL_STP", "Measured H₂ generation (mL at 0°C, 1 atm)"),
        ("Pressure_Abs_kPa", "Measured internal absolute pressure (kPa)"),
    ]
    if any(col in cap_rows and cap_rows[col].notna().any() for col, _ in present_measured_gas):
        st.markdown("**Additional directly measured indicators (if recorded)**")
        mleft, mright = st.columns(2)
        for place, (column, label) in zip((mleft, mright), present_measured_gas):
            with place:
                st.markdown(f"**{label}**")
                if column in cap_rows and cap_rows[column].notna().any():
                    st.scatter_chart(cap_rows[["Hours", column]].dropna(), x="Hours", y=column, height=240)
                else:
                    st.caption("No direct measurements supplied for this channel.")
    with st.expander("View processed measurements"):
        st.dataframe(cap_rows, hide_index=True, use_container_width=True)
    gas_df = None
    scenario_params = None
    # v0.8 PREVIEW: optional assumption-based calculation; never treated as a measurement.
    st.subheader("4A · H₂ generation and internal pressure (v0.8 physics preview)")
    st.caption(
        "Explore how **assumed** hydrogen production from a fraction of measured leakage "
        "current could change gas quantity and internal pressure over the recorded history. "
        "This is **not** a pressure measurement, vent-opening model, EOL assessment, or RUL prediction."
    )
    show_gas_preview = st.checkbox(
        "Enable assumption-based H₂ / pressure scenario",
        value=False,
        key="caplife_gas_scenario_v08",
    )
    if show_gas_preview:
        from caplife.gas_pressure import (
            GasPressureInputError, calculate_h2_pressure_history,
        )
        with st.expander("Physics assumptions — please review before interpreting", expanded=True):
            st.write(
                "**Leakage current is not equivalent to hydrogen-generation current.** "
                "This preview uses a user-selected assumed fraction of leakage current; "
                "it then assumes a fixed fraction of generated hydrogen stays in the gas space. "
                "Core_C is used as an assumed gas-temperature proxy. The gas is ideal and the gas space is fixed; the model does not describe "
                "electrolyte vapor, diffusion, dissolution, venting, swelling, or actual gas chemistry. "
                "Values shown are a *scenario*, not experimentally validated quantities."
            )
            scenario_left, scenario_right = st.columns(2)
            with scenario_left:
                assumed_headspace = st.number_input(
                    "Assumed internal free-gas volume (mL)",
                    min_value=0.01, value=5.0, step=0.5,
                    key="caplife_gas_space_ml_v08",
                )
                assumed_h2_fraction = st.slider(
                    "Assumed fraction of leakage current producing H₂",
                    0.0, 1.0, 0.10, 0.05,
                    key="caplife_gas_h2_fraction_v08",
                )
            with scenario_right:
                assumed_initial_p = st.number_input(
                    "Assumed initial absolute pressure (kPa; NOT gauge)",
                    min_value=0.01, value=101.325, step=1.0,
                    key="caplife_gas_initial_p_v08",
                )
                assumed_retained = st.slider(
                    "Assumed fraction of formed H₂ retained in gas space",
                    0.0, 1.0, 0.50, 0.05,
                    key="caplife_gas_retention_v08",
                )
        try:
            gas_df = calculate_h2_pressure_history(
                cap_rows,
                headspace_ml=assumed_headspace,
                initial_pressure_abs_kpa=assumed_initial_p,
                h2_current_fraction=assumed_h2_fraction,
                retained_fraction=assumed_retained,
            )
        except GasPressureInputError as exc:
            st.info(f"Gas/pressure scenario unavailable for this capacitor: {exc}")
            st.caption(
                "For this first preview, the selected capacitor must have a 0-hour reading "
                "and complete Leakage_uA and Core_C readings at each recorded time. "
                "No missing values are invented or interpolated."
            )
        else:
            scenario_params = {
                "headspace_ml": float(assumed_headspace),
                "initial_pressure_abs_kpa": float(assumed_initial_p),
                "h2_current_fraction": float(assumed_h2_fraction),
                "retained_fraction": float(assumed_retained),
                "reference_H2_volume": "0 C, 1 atm",
                "input_provenance": "Measured leakage-current history; estimated hydrogen and pressure",
                "model_status": "UNVALIDATED hypothetical physics scenario; NOT directly measured",
            }
            last = gas_df.iloc[-1]
            indicators = st.columns(3)
            indicators[0].metric(
                "Calculated H₂ formed (STP equivalent)",
                f"{last['H2_Generated_mL_STP_est']:.3g} mL",
            )
            indicators[1].metric(
                "Calculated pressure (absolute)",
                f"{last['Pressure_Abs_kPa_est']:.1f} kPa",
            )
            indicators[2].metric(
                "Calculated pressure change from initial",
                f"{last['Pressure_Change_From_Initial_kPa_est']:+.1f} kPa",
            )
            st.caption(
                "H₂ volume is expressed at 0°C and 1 atm. The pressure model uses the "
                "assumed physical headspace, not the STP gas volume. Thermal pressure "
                "changes of the initial gas are included. Positive pressure change is "
                "not necessarily only caused by H₂."
            )
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**Calculated H₂ formed and retained over time (mmol)**")
                st.line_chart(
                    gas_df.set_index("Hours")[[
                        "H2_Generated_mmol_est", "H2_Retained_mmol_est"
                    ]], height=260,
                )
            with chart_right:
                st.markdown("**Calculated internal pressure over time (absolute kPa)**")
                st.line_chart(
                    gas_df.set_index("Hours")["Pressure_Abs_kPa_est"],
                    height=260,
                )
            with st.expander("Inspect calculation table and equations"):
                st.dataframe(gas_df, hide_index=True, use_container_width=True)
                st.latex(r"Q(t) = \int_0^t I_{\rm leakage}(\tau) \, d\tau")
                st.latex(r"n_{H_2,\rm generated}(t) = \eta \frac{Q(t)}{2F}")
                st.latex(r"P_{\rm abs}(t) = P_0\frac{T(t)}{T_0} + \frac{n_{H_2,\rm retained}(t) R T(t)}{V_{\rm headspace}}")
                st.caption(
                    "The integral uses trapezoidal approximation between observed leakage "
                    "readings. η and retention are assumptions, not inferred physical constants."
                )
            st.info(
                "This is an **assumption-based estimate**, not measured H₂/pressure. "
                "When you download a report, the scenario will be included in separately "
                "labeled columns or sheets/pages, never merged into raw measurements."
            )

    # v0.9: descriptive rates from six direct measurement channels only.
    # This module NEVER consumes hypothetical H2 / pressure scenario outputs.
    from caplife.degradation_rates import calculate_degradation_rates
    st.subheader("4B · Six-indicator degradation rates (v0.9 preview)")
    st.caption(
        "Rate = observed change divided by elapsed hours. Acceleration = change in "
        "successive interval rates divided by elapsed midpoint time. Both use "
        "measured points only: missing readings are never filled or bridged. "
        "No health score, failure threshold, RUL, or forecasting is calculated."
    )
    rate_report = calculate_degradation_rates(readings)
    rate_summary = rate_report["summary"].loc[
        lambda df: df["Cap_ID"] == cap_choice
    ]
    # Display very small accelerations in scientific notation without
    # incorrectly rounding a nonzero value to zero.
    rate_summary_display = rate_summary.copy()
    for numeric_column in (
        "Time_weighted_rate_per_hour", "Latest_rate_per_hour",
        "Latest_acceleration_per_hour2",
    ):
        rate_summary_display[numeric_column] = rate_summary_display[numeric_column].map(
            lambda value: "—" if pd.isna(value) else f"{value:+.5g}"
        )
    st.dataframe(rate_summary_display, hide_index=True, use_container_width=True)
    for warning in rate_report["warnings"]:
        if str(cap_choice) in warning:
            st.warning(warning)
    with st.expander("View interval-by-interval rates for all six indicators", expanded=False):
        rate_rows = rate_report["intervals"].loc[
            lambda df: df["Cap_ID"] == cap_choice
        ]
        two_cols = st.columns(2)
        for chart_idx, (_, display_name, unit) in enumerate((
            ("Capacitance_uF", "Capacitance", "µF"),
            ("ESR_ohm", "ESR", "Ω"),
            ("Leakage_uA", "Leakage current", "µA"),
            ("Case_Expansion_um", "Case expansion", "µm"),
            ("H2_Generated_mL_STP", "H₂ generated (directly measured)", "mL STP"),
            ("Pressure_Abs_kPa", "Internal absolute pressure (directly measured)", "kPa absolute"),
        )):
            with two_cols[chart_idx % 2]:
                st.markdown(f"**{display_name}: {unit}/h**")
                present = rate_rows.loc[rate_rows["Indicator"] == display_name]
                if present.empty:
                    st.caption("Insufficient adjacent comparable measurements.")
                else:
                    st.scatter_chart(
                        present[["Midpoint_Hour", "Rate_per_hour"]],
                        x="Midpoint_Hour", y="Rate_per_hour", height=220,
                    )
        st.caption(
            "Dots represent rates between adjacent recorded timestamps, plotted at "
            "interval midpoints. They are NOT continuously measured rates. "
            "Each graph uses its own units and scale."
        )
        st.dataframe(rate_rows.round(10), hide_index=True, use_container_width=True)
    with st.expander("Inspect rate acceleration calculations", expanded=False):
        accel_rows = rate_report["accelerations"].loc[
            lambda df: df["Cap_ID"] == cap_choice
        ]
        st.caption(
            "Acceleration here means a change in *measurement rate*, not mechanical "
            "acceleration. It requires three consecutive nonmissing readings "
            "and has units of indicator per hour squared. Negative capacitance "
            "acceleration can indicate faster loss, but neither sign is a health verdict."
        )
        if accel_rows.empty:
            st.info("No comparable three-point sequences are available for this capacitor.")
        else:
            chosen_indicator = st.selectbox(
                "Indicator for rate-acceleration chart",
                sorted(accel_rows["Indicator"].unique()),
                key="caplife_v09_accel_indicator",
            )
            accel_subset = accel_rows.loc[accel_rows["Indicator"] == chosen_indicator]
            st.scatter_chart(
                accel_subset[["At_Hours", "Acceleration_per_hour2"]],
                x="At_Hours", y="Acceleration_per_hour2", height=250,
            )
            st.dataframe(accel_rows.round(12), hide_index=True, use_container_width=True)
    st.download_button(
        "Download observed interval-rate table (CSV)",
        data=rate_report["intervals"].to_csv(index=False).encode("utf-8-sig"),
        file_name="CapLife_v0_9_Observed_Interval_Rates.csv", mime="text/csv",
        key="caplife_v09_rate_csv",
    )
    st.caption(
        "Rates and changes may reflect ambient/core temperature, voltage, load, "
        "measurement methods, or test conditions—not only aging. A slope sign "
        "alone is not a failure or health classification."
    )
    st.caption(
        "This v0.9 preview adds descriptive rates to the Excel/PDF downloads. "
        "CSV measured histories remain unchanged; the separate rate CSV is above. "
        "Rates have not been validated for health inference."
    )

    st.subheader("5 · Download your assessment")
    from caplife.report_exports import csv_report_bytes, csv_measurement_history_bytes, xlsx_report_bytes, pdf_report_bytes

    chosen_format = st.radio(
        "Choose one file format",
        ("CSV (.csv)", "Excel (.xlsx)", "PDF (.pdf)"),
        horizontal=True,
        index=1,
        key="caplife_report_format_v07",
    )
    if chosen_format == "CSV (.csv)":
        csv_content = st.radio(
            "What should the CSV contain?",
            ("Complete measured history (all capacitors)", "Per-capacitor assessment summary"),
            horizontal=True,
            key="caplife_csv_content_v08",
        )
        if csv_content.startswith("Complete measured history"):
            st.caption(
                "Every recorded measurement row, including all available six indicators "
                "and their baseline-normalized ratios. Physics scenario estimates are "
                "never mixed into the measured history."
            )
            output_bytes = csv_measurement_history_bytes(real_report)
            export_basename = "CapLife_AI_v0_8_Measured_History"
        else:
            st.caption(
                "One row per capacitor with observed changes. If enabled, assumption-based "
                "gas/pressure results appear in clearly named estimate columns."
            )
            output_bytes = csv_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params)
            export_basename = "CapLife_AI_v0_8_Assessment_Summary"
        extension = "csv"
        mime_type = "text/csv"
    elif chosen_format == "Excel (.xlsx)":
        st.caption("Includes measured histories, six-indicator rate/acceleration tables and separate charts; optional physics estimates remain separately labeled.")
        output_bytes = xlsx_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params, rate_report=rate_report)
        export_basename = "CapLife_AI_v0_9_Assessment"
        extension = "xlsx"
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        st.caption("Includes descriptive rate and acceleration appendices for the selected capacitor, plus optional separately labeled physics scenario.")
        output_bytes = pdf_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params, rate_report=rate_report)
        export_basename = "CapLife_AI_v0_9_Assessment"
        extension = "pdf"
        mime_type = "application/pdf"
    st.download_button(
        f"Download {chosen_format} assessment",
        data=output_bytes,
        file_name=f"{export_basename}.{extension}",
        mime=mime_type,
        type="primary",
    )
    with st.expander("Advanced: download every assessment table as separate CSV files (ZIP)"):
        original_zip = assessment_zip_with_readiness_bytes(real_report, audit)
        if gas_df is not None:
            # Add explicitly named assumption-based files without altering the
            # existing measured and audit CSVs or inventing readings.
            from zipfile import ZIP_DEFLATED, ZipFile
            with BytesIO() as new_zip_buf:
                with ZipFile(BytesIO(original_zip), "r") as original, ZipFile(new_zip_buf, "w", ZIP_DEFLATED) as archive:
                    for filename in original.namelist():
                        archive.writestr(filename, original.read(filename))
                    archive.writestr("hypothetical_H2_pressure_scenario_estimates.csv", gas_df.to_csv(index=False))
                    archive.writestr("hypothetical_H2_pressure_scenario_assumptions.json", json.dumps(scenario_params, indent=2))
                original_zip = new_zip_buf.getvalue()
        st.download_button(
            "Download complete CSV package (ZIP)",
            data=original_zip,
            file_name="CapLife_v0_8_Measurement_and_Physics.zip",
            mime="application/zip",
        )
    st.info(
        "**Limitations:** This mode does not determine health, EOL, RUL, uncertainty "
        "intervals, or maintenance actions. A complete dataset can still have inaccurate "
        "sensors, changing operating conditions, or incomparable test settings. "
        "The uploaded file is processed by the web-app server; check hosting/privacy "
        "policies before submitting confidential experiments."
    )
    st.caption("CapLife AI v0.8 · six-indicator observation plus optional hypothetical physics · no real-data neural-network prediction")
    st.stop()

# Previous synthetic RUL page is intentionally preserved below this line.


@st.cache_resource(show_spinner="Loading the demonstration neural network…")
def get_engine():
    return load_engine(ARTIFACTS)


try:
    net, metadata = get_engine()
except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
    st.error("Model files are missing or incompatible. See README.md for training instructions.")
    st.code("python -m caplife.train --root .", language="bash")
    st.stop()

@st.cache_data(show_spinner=False)
def load_synthetic_calibration():
    """Read and validate the saved *synthetic-only* interval calibration.

    The JSON does NOT give calibrated intervals for real uploaded AEC data.
    """
    try:
        artifact = json.loads(UNCERTAINTY_JSON.read_text(encoding="utf-8"))
        if artifact["version"] != "v0.5-synthetic-demo":
            raise ValueError("Unexpected calibration artifact version")
        if artifact["method"] != "group-specific split conformal":
            raise ValueError("Unexpected calibration method")
        if artifact["prediction_engine"] != "multi-indicator threshold trend":
            raise ValueError("Calibration was not made for the trend engine")
        if artifact["real_world_validated"] is not False:
            raise ValueError("Invalid synthetic-only calibration provenance")
        expected_groups = {"Short": [0, 1500], "Medium": [1500, 3000], "Long": [3000, None]}
        if artifact["predicted_rul_groups_hours"] != expected_groups:
            raise ValueError("Predicted-RUL grouping differs from this app")
        for name, expected_threshold in EOL_THRESHOLDS.items():
            actual_threshold = float(artifact["eol_thresholds"][name])
            if not np.isclose(actual_threshold, expected_threshold, atol=0, rtol=1e-10):
                raise ValueError("Current illustrative EOL thresholds differ from calibration")
        for group in expected_groups:
            margin = float(artifact["error_margins_hours"][group])
            if not np.isfinite(margin) or margin < 0:
                raise ValueError("Invalid uncertainty margin")
        if not np.isclose(float(artifact["target_coverage"]), 0.90):
            raise ValueError("Unexpected calibration coverage target")
        return artifact, None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)


synthetic_calibration, calibration_issue = load_synthetic_calibration()


def synthetic_interval_for_trend(trend_rul, *, bundled_demo, eol_causes, training_warnings):
    """Create an illustrative interval only in the matching demo context."""
    if (not bundled_demo or synthetic_calibration is None or eol_causes
            or training_warnings or trend_rul is None):
        return None
    prediction = float(trend_rul)
    if not np.isfinite(prediction) or prediction < 0:
        return None
    if prediction <= 1500:
        group = "Short"
    elif prediction <= 3000:
        group = "Medium"
    else:
        group = "Long"
    margin = float(synthetic_calibration["error_margins_hours"][group])
    return {
        "lower": max(0.0, prediction - margin),
        "upper": prediction + margin,
        "margin": margin,
        "group": group,
    }


with st.sidebar:
    st.header("About this prototype")
    st.write("**AI engine:** GRU recurrent neural network")
    st.write(f"**Historical window:** {SEQ_LEN} measurements")
    st.write("**Source:** simulated AEC aging only")
    st.caption("v0.5: synthetic-only trend intervals; not valid for real AEC measurements")
    st.divider()
    st.subheader("Upload data specification")
    st.write(
        f"Each capacitor needs at least **{SEQ_LEN} chronological measurements**. "
        "One row represents one measurement."
    )
    st.write("Required headers:")
    st.code("Cap_ID, " + ", ".join(FEATURES), language=None)
    st.download_button(
        "Download CSV template",
        data=TEMPLATE_DATA.read_bytes(),
        file_name="aec_measurements_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.expander("Measurement units"):
        for key, description in metadata["unit_conventions"].items():
            st.markdown(f"**{key}** — {description}")
    st.divider()
    st.caption(
        "Uploaded data are processed by the web-app server; review hosting/privacy "
        "policies before uploading confidential experiments."
    )

st.subheader("1 · Choose a dataset")
file = st.file_uploader(
    "Upload capacitor measurement history",
    type=["csv"],
    help="CSV file; maximum 8 MB in this prototype.",
)
if file is not None:
    if file.size > 8 * 1024 * 1024:
        st.error("The uploaded CSV exceeds this prototype's 8 MB limit.")
        st.stop()
    # Include content, not only name and size, when recognizing a changed upload.
    current_tag = hashlib.sha256(file.getvalue()).hexdigest()
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    st.caption(f"Using uploaded dataset: **{file.name}**")
else:
    current_tag = "bundled-demo"
    st.info(
        "No file uploaded. Using the bundled **synthetic example dataset** "
        "so you can explore the full workflow."
    )
    df = pd.read_csv(DEMO_DATA)

try:
    data = validate_measurements(df)
except DataValidationError as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.info(
        "Use the CSV template in the sidebar. Units and measurement methods "
        "must also match the model's documented schema."
    )
    st.stop()

available_caps = data["Cap_ID"].unique().tolist()
col1, col2, col3 = st.columns(3)
col1.metric("Capacitors", f"{len(available_caps):,}")
col2.metric("Measurement rows", f"{len(data):,}")
col3.metric("AI input features", str(len(FEATURES)))

st.subheader("2 · Select a capacitor and prediction time")
selected = st.selectbox("Capacitor ID", available_caps)
cap_df = data.loc[data["Cap_ID"] == selected].copy().sort_values("Hours")
st.caption(
    f"This capacitor has {len(cap_df)} measurements, "
    f"from hour {cap_df['Hours'].min():.0f} to {cap_df['Hours'].max():.0f}."
)

# Earliest choice leaves enough readings for one complete GRU input sequence.
time_options = cap_df["Hours"].iloc[SEQ_LEN - 1:].tolist()
if len(time_options) == 1:
    prediction_hour = time_options[0]
    st.info(
        f"Only {SEQ_LEN} readings are available; prediction time is fixed "
        f"at {prediction_hour:,.0f} hours."
    )
else:
    prediction_hour = st.select_slider(
        "Select prediction time (operating hours)",
        options=time_options,
        value=time_options[-1],
        format_func=lambda hour: f"{hour:,.0f} hours",
    )

# This historical slice is used BOTH for inference and for the visible charts.
# No measurements after the selected time are provided to the neural network.
history_to_show = cap_df.loc[cap_df["Hours"] <= prediction_hour].copy()
st.caption(
    f"Prediction will use {len(history_to_show)} readings up to "
    f"{prediction_hour:,.0f} hours. Future measurements are excluded."
)

if st.button("Analyze capacitor", type="primary", use_container_width=True):
    try:
        assessment = assess_capacitor(history_to_show, net, metadata)
        trend_assessment = assess_trend(history_to_show, window_size=SEQ_LEN)
    except (DataValidationError, ValueError) as exc:
        st.error(f"Could not assess this capacitor: {exc}")
        st.stop()

    st.session_state["last_assessment"] = assessment
    st.session_state["last_trend_assessment"] = trend_assessment
    st.session_state["last_assessed_cap"] = selected
    st.session_state["last_data_tag"] = current_tag
    st.session_state["last_prediction_hour"] = prediction_hour

result = st.session_state.get("last_assessment")
trend_result = st.session_state.get("last_trend_assessment")
result_is_current = (
    result is not None
    and isinstance(trend_result, dict)
    and st.session_state.get("last_assessed_cap") == selected
    and st.session_state.get("last_data_tag") == current_tag
    and st.session_state.get("last_prediction_hour") == prediction_hour
)

st.subheader("3 · RUL assessment")
if result_is_current:
    m1, m2, m3 = st.columns(3)
    m1.metric("GRU remaining life (DEMO)", f"{result['rul_hours']:,.0f} h")
    m2.metric("Age at prediction", f"{result['hours_observed']:,.0f} h")
    m3.metric(
        "Age + estimated RUL (DEMO)",
        f"{result['hours_observed'] + result['rul_hours']:,.0f} h",
    )
    st.write("**Threshold status:** " + result["status"])
    if result["eol_causes"]:
        st.error("Illustrative EOL trigger(s): " + ", ".join(result["eol_causes"]))
        st.caption(
            "A demo threshold has been observed at this measurement. Zero RUL comes "
            "from a hard-coded rule, **not** from the GRU. The displayed age is not "
            "necessarily the exact physical failure time."
        )
    for warning in result["warnings"]:
        st.error("Outside training range — do not rely on prediction: " + warning)
    st.caption(
        "The GRU estimate is **uncalibrated**. A separate trend interval is available "
        "only for bundled synthetic demonstration data; no validated real-AEC "
        "uncertainty interval or health score is available."
    )

    # The mathematical baseline and the GRU receive the same available history.
    st.markdown("**Compare the two prediction methods**")
    trend_rul = trend_result["rul_hours"] if trend_result is not None else None
    gru_rul = float(result["rul_hours"])
    absolute_disagreement = None
    relative_disagreement = None
    disagreement_label = "Not estimable"

    if trend_rul is not None:
        trend_rul = float(trend_rul)
        absolute_disagreement = abs(gru_rul - trend_rul)
        relative_disagreement = (
            absolute_disagreement / max(500.0, (gru_rul + trend_rul) / 2)
        )
        # Experimental thresholds; not failure probabilities or confidence levels.
        if relative_disagreement <= 0.10:
            disagreement_label = "Low"
        elif relative_disagreement <= 0.30:
            disagreement_label = "Moderate"
        else:
            disagreement_label = "High"

    c_gru, c_trend, c_disagreement = st.columns(3)
    c_gru.metric("GRU RUL (DEMO)", f"{gru_rul:,.0f} h")
    c_trend.metric(
        "Threshold-trend RUL (DEMO)",
        f"{trend_rul:,.0f} h" if trend_rul is not None else "Not estimable",
    )
    c_disagreement.metric(
        "Model disagreement (DEMO)",
        f"{relative_disagreement * 100:,.1f}%" if relative_disagreement is not None else "N/A",
    )

    if result["eol_causes"]:
        st.info("Both displayed zero-hour estimates come from the illustrative EOL rule, not from forecasting.")
    elif trend_rul is None:
        st.info(
            "The recent indicator trends do not produce an approaching threshold "
            "crossing. An unestimated trend RUL does not mean infinite or safe lifetime."
        )
    else:
        limiting = trend_result["limiting_indicator"]
        st.write(f"**Predicted limiting indicator (trend):** `{limiting}`")
        st.write(f"**Absolute model disagreement:** {absolute_disagreement:,.0f} hours")
        if disagreement_label == "High":
            st.warning("High model disagreement (experimental warning); the two estimates conflict substantially.")
        else:
            st.caption(f"Experimental disagreement band: **{disagreement_label}**")

    # Step 99: synthetic-only interval. Disabled for any uploaded dataset.
    trend_interval = synthetic_interval_for_trend(
        trend_rul,
        bundled_demo=(file is None),
        eol_causes=result["eol_causes"],
        training_warnings=result["warnings"],
    )
    st.markdown("**Experimental uncertainty for the threshold-trend estimate**")
    if file is not None:
        st.info(
            "No calibrated interval is provided for uploaded datasets. "
            "The saved uncertainty margins were obtained from invented synthetic histories "
            "and must not be applied to real capacitor measurements."
        )
    elif calibration_issue:
        st.warning(
            "Synthetic interval unavailable because the calibration artifact is missing "
            "or incompatible. The GRU and trend estimates remain available."
        )
    elif result["eol_causes"]:
        st.caption("No forecasting interval at an already-triggered illustrative EOL threshold.")
    elif result["warnings"]:
        st.warning(
            "Synthetic interval withheld because inputs fall outside the model's "
            "documented training ranges."
        )
    elif trend_interval is None:
        st.caption("The trend RUL is not estimable, so no interval can be displayed.")
    else:
        lower_col, upper_col = st.columns(2)
        lower_col.metric("Trend interval lower bound (DEMO)", f"{trend_interval['lower']:,.0f} h")
        upper_col.metric("Trend interval upper bound (DEMO)", f"{trend_interval['upper']:,.0f} h")
        st.caption(
            f"Predicted-life group: **{trend_interval['group']}** · "
            f"synthetic error margin: ±{trend_interval['margin']:,.0f} h · "
            "nominal calibration target: 90%."
        )
        st.warning(
            "**Synthetic-only interval; NOT a validated confidence statement for this "
            "individual capacitor.** The 90% target was calibrated from one randomly "
            "sampled pre-EOL observation per simulated capacitor. It does not establish "
            "90% coverage for a manually selected hour, multiple time points from one "
            "capacitor, changed operating conditions, or any real AEC dataset."
        )
        observed_coverage = synthetic_calibration.get("independent_synthetic_test", {}).get("overall_coverage_percent")
        if observed_coverage is not None:
            st.caption(
                f"Prior independent-simulator test: {float(observed_coverage):.2f}% overall "
                "coverage across synthetic sampled observations; group coverage differed."
            )

    with st.expander("Show each indicator's mathematical threshold-crossing estimate"):
        indicator_labels = {
            "C_ratio": "Capacitance ratio",
            "ESR_ratio": "ESR ratio",
            "Leakage_ratio": "Leakage ratio",
            "Case_Strain_Index": "Case strain index (hypothetical)",
        }
        detail_rows = [
            {
                "Degradation indicator": indicator_labels.get(name, name),
                "Time to illustrative EOL threshold": (
                    f"{hours:,.0f} h" if hours is not None else "Not estimable"
                ),
            }
            for name, hours in trend_result["indicator_rul_hours"].items()
        ]
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)
        st.caption(trend_result["message"])

    st.caption(
        "The trend linearly extrapolates the latest six measurements toward the "
        "simulator's four **illustrative** EOL thresholds; the GRU is trained on "
        "synthetic data. Neither method is validated on real capacitors. "
        "Disagreement is NOT calibrated uncertainty or a health probability."
    )

    report = pd.DataFrame([{
        "Cap_ID": result["cap_id"],
        "Prediction_hour": prediction_hour,
        "Age_hours": result["hours_observed"],
        "Demo_predicted_RUL_hours": round(result["rul_hours"], 2),
        "GRU_RUL_hours": round(gru_rul, 2),
        "Threshold_trend_RUL_hours": (round(trend_rul, 2) if trend_rul is not None else None),
        "Trend_limiting_indicator": trend_result["limiting_indicator"],
        "Model_disagreement_hours": (
            round(absolute_disagreement, 2) if absolute_disagreement is not None else None
        ),
        "Relative_disagreement_percent": (
            round(100 * relative_disagreement, 2) if relative_disagreement is not None else None
        ),
        "Trend_synthetic_interval_lower_hours": (
            round(trend_interval["lower"], 2) if trend_interval else None
        ),
        "Trend_synthetic_interval_upper_hours": (
            round(trend_interval["upper"], 2) if trend_interval else None
        ),
        "Trend_synthetic_interval_group": (
            trend_interval["group"] if trend_interval else None
        ),
        "Trend_synthetic_interval_scope": (
            "Demonstration-only; calibrated on one random observation per synthetic capacitor"
            if trend_interval else "Not applicable / withheld"
        ),
        "Threshold_status": result["status"],
        "Triggered_demo_thresholds": "; ".join(result["eol_causes"]),
        "Out_of_training_range": "; ".join(result["warnings"]),
        "Model": result["model_id"],
        "Important_warning": "Synthetic-trained model; NOT validated for real AECs",
    }])
    st.download_button(
        "Download assessment CSV",
        data=report.to_csv(index=False),
        file_name=f"{selected}_demo_assessment_{prediction_hour:.0f}h.csv",
        mime="text/csv",
    )
else:
    st.info("Choose a prediction time and click **Analyze capacitor** to see an assessment.")

st.subheader("4 · Visualize degradation up to the prediction time")
st.caption(
    f"These graphs stop at {prediction_hour:,.0f} hours. "
    "Measurements taken afterward are not shown or used by the model."
)
plot_df = history_to_show.set_index("Hours")
left, right = st.columns(2)
with left:
    st.write("**Capacitance ratio** (1.0 = initial capacitance)")
    st.line_chart(plot_df[["C_ratio"]])
    st.write("**ESR ratio** (1.0 = initial ESR)")
    st.line_chart(plot_df[["ESR_ratio"]])
with right:
    st.write("**Leakage current ratio**")
    st.line_chart(plot_df[["Leakage_ratio"]])
    st.write("**Case strain index** (hypothetical)")
    st.line_chart(plot_df[["Case_Strain_Index"]])

with st.expander("View measurements used at the selected prediction time"):
    st.dataframe(history_to_show, use_container_width=True, hide_index=True)
    st.download_button(
        "Download measurements up to prediction time",
        data=history_to_show.to_csv(index=False),
        file_name=f"{selected}_up_to_{prediction_hour:.0f}h.csv",
        mime="text/csv",
    )

with st.expander("View full capacitor history (reference only — includes future readings)"):
    st.caption(
        "For retrospective research only. These later readings are NOT passed to "
        "the GRU when predicting at an earlier time."
    )
    st.dataframe(cap_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download full selected history",
        data=cap_df.to_csv(index=False),
        file_name=f"{selected}_full_measurements.csv",
        mime="text/csv",
    )

with st.expander("Model testing — synthetic capacitors only"):
    st.json(metadata["metrics"])
    st.write(
        "Train/validation/test splits contain different capacitor IDs. "
        "Scores describe invented test trajectories and cannot be generalized to real AECs."
    )


# The retrospective dashboard runs separately from the single-time assessment.
# It never passes future measurements or ground-truth labels to the GRU.
def make_prediction_trajectory(cap_history, through_hour, model, model_metadata, show_reference):
    chronological = cap_history.loc[
        cap_history["Hours"] <= through_hour
    ].sort_values("Hours").reset_index(drop=True)
    sequence_length = int(model_metadata["sequence_length"])
    if len(chronological) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} observations.")

    # Cap work at 400 evaluation points for unusually long uploaded histories.
    possible_ends = np.arange(sequence_length - 1, len(chronological))
    if len(possible_ends) > 400:
        picked = np.linspace(0, len(possible_ends) - 1, 400, dtype=int)
        possible_ends = possible_ends[np.unique(picked)]

    rows = []
    for end_idx in possible_ends:
        # Only the latest six feature readings, up through this hour, are provided
        # to assess_capacitor. RUL_hours and EOL_observed are never model inputs.
        recent = chronological.iloc[
            end_idx - sequence_length + 1:end_idx + 1
        ][["Cap_ID", *FEATURES]]
        assessment = assess_capacitor(recent, model, model_metadata)
        trend_assessment = assess_trend(recent, window_size=sequence_length)
        actual_rul = np.nan
        if show_reference and "RUL_hours" in chronological.columns:
            reference_value = chronological.iloc[end_idx]["RUL_hours"]
            if pd.notna(reference_value):
                actual_rul = float(reference_value)

        predicted_rul = float(assessment["rul_hours"])
        trend_rul = trend_assessment["rul_hours"]
        trend_rul = float(trend_rul) if trend_rul is not None else np.nan
        from_gru = not bool(assessment["eol_causes"])
        abs_disagreement = (
            abs(predicted_rul - trend_rul) if np.isfinite(trend_rul) else np.nan
        )
        relative_disagreement = (
            100 * abs_disagreement / max(500.0, (predicted_rul + trend_rul) / 2)
            if np.isfinite(abs_disagreement) else np.nan
        )
        point_interval = synthetic_interval_for_trend(
            trend_rul if np.isfinite(trend_rul) else None,
            bundled_demo=show_reference,
            eol_causes=assessment["eol_causes"],
            training_warnings=assessment["warnings"],
        )
        rows.append({
            "Prediction_hour": float(assessment["hours_observed"]),
            "Trend_synthetic_lower_RUL_hours": (
                point_interval["lower"] if point_interval else np.nan
            ),
            "Trend_synthetic_upper_RUL_hours": (
                point_interval["upper"] if point_interval else np.nan
            ),
            "Trend_synthetic_interval_group": (
                point_interval["group"] if point_interval else None
            ),
            "Predicted_RUL_hours": predicted_rul,
            "Trend_RUL_hours": trend_rul,
            "Trend_limiting_indicator": trend_assessment["limiting_indicator"],
            "Absolute_disagreement_hours": abs_disagreement,
            "Relative_disagreement_percent": relative_disagreement,
            "Reference_RUL_hours": actual_rul,
            "Absolute_error_hours": (
                abs(predicted_rul - actual_rul) if np.isfinite(actual_rul) else np.nan
            ),
            "Trend_absolute_error_hours": (
                abs(trend_rul - actual_rul)
                if np.isfinite(trend_rul) and np.isfinite(actual_rul)
                else np.nan
            ),
            "Prediction_source": "GRU" if from_gru else "Toy EOL rule (0 h)",
            "Outside_demo_training_range": bool(assessment["warnings"]),
        })
    return pd.DataFrame(rows)


st.subheader("5 · Retrospective RUL prediction dashboard")
st.caption(
    "Evaluate predictions at successive observation times up to your selected hour. "
    "Each estimate uses only the latest six measurements available at that time. "
    "This is a retrospective teaching exercise, not an independent model-validation study."
)

reference_is_known = (
    file is None
    and "EOL_observed" in cap_df.columns
    and "RUL_hours" in cap_df.columns
    and bool(cap_df["EOL_observed"].all())
    and bool(cap_df["RUL_hours"].notna().all())
    and bool(np.isclose(cap_df["RUL_hours"].iloc[-1], 0))
)

if reference_is_known:
    st.info(
        "A recorded simulated EOL event is available for this example. The reference "
        "RUL is computed from the **full future history for scoring only**; it is never "
        "supplied to the GRU. Some example capacitors may have been used in training, "
        "so this graph is not necessarily held-out test performance."
    )
else:
    st.info(
        "No verified synthetic run-to-EOL reference is available for this selection. "
        "The dashboard can plot demo model estimates, but **cannot calculate actual "
        "RUL or prediction error**. Uploaded data are not automatically treated as "
        "validated lifetime ground truth."
    )

if st.button("Analyze RUL across the available history", use_container_width=True):
    try:
        with st.spinner("Calculating retrospective predictions…"):
            trajectory = make_prediction_trajectory(
                cap_df, prediction_hour, net, metadata, reference_is_known
            )
    except (DataValidationError, ValueError) as exc:
        st.error(f"Could not calculate the trajectory: {exc}")
    else:
        st.session_state["rul_trajectory"] = trajectory
        st.session_state["trajectory_cap"] = selected
        st.session_state["trajectory_tag"] = current_tag
        st.session_state["trajectory_hour"] = prediction_hour

trajectory = st.session_state.get("rul_trajectory")
trajectory_is_current = (
    isinstance(trajectory, pd.DataFrame)
    and not trajectory.empty
    and st.session_state.get("trajectory_cap") == selected
    and st.session_state.get("trajectory_tag") == current_tag
    and st.session_state.get("trajectory_hour") == prediction_hour
)

if trajectory_is_current:
    st.caption(
        f"{len(trajectory):,} prediction points through {prediction_hour:,.0f} hours. "
        "The last eligible measurement is always included."
    )
    if reference_is_known:
        comparison = trajectory.set_index("Prediction_hour")[[
            "Reference_RUL_hours", "Predicted_RUL_hours", "Trend_RUL_hours"
        ]].rename(columns={
            "Predicted_RUL_hours": "GRU RUL (DEMO)",
            "Trend_RUL_hours": "Threshold-trend RUL (DEMO)",
            "Reference_RUL_hours": "Synthetic reference RUL",
        })
        st.line_chart(comparison)
        st.caption(
            "Reference RUL comes from the simulated eventual EOL and is used ONLY "
            "for evaluation. GRU and trend predictions use data available at each prediction hour."
        )
        # Do not let automatic zero at the toy EOL artificially improve GRU metrics.
        scored = trajectory.loc[
            (trajectory["Prediction_source"] == "GRU")
            & trajectory["Reference_RUL_hours"].notna()
            & (trajectory["Reference_RUL_hours"] > 0)
        ]
        if not scored.empty:
            errors = (
                scored["Predicted_RUL_hours"] - scored["Reference_RUL_hours"]
            ).to_numpy()
            mae = float(np.abs(errors).mean())
            rmse = float(np.sqrt(np.square(errors).mean()))
            a, b, c = st.columns(3)
            a.metric("Mean absolute error (DEMO)", f"{mae:,.0f} h")
            b.metric("RMSE (DEMO)", f"{rmse:,.0f} h")
            c.metric("GRU evaluation points", f"{len(scored):,}")
            trend_scored = scored.loc[scored["Trend_RUL_hours"].notna()]
            if not trend_scored.empty:
                trend_mae = float(trend_scored["Trend_absolute_error_hours"].mean())
                st.metric("Threshold-trend MAE (DEMO)", f"{trend_mae:,.0f} h")
            st.caption(
                "Errors summarize correlated time points from ONE simulated "
                "capacitor, not independent test cases. EOL rule-generated zero "
                "predictions and known-zero reference points are excluded. "
                "Trend and GRU MAEs may cover different counts if some trend estimates are unavailable."
            )
        else:
            st.warning("No pre-EOL GRU estimates are available for error scoring.")
    else:
        st.line_chart(
            trajectory.set_index("Prediction_hour")[["Predicted_RUL_hours", "Trend_RUL_hours"]]
        )
        st.caption("Only demonstration predictions can be shown; no error metrics are available.")

    if trajectory["Outside_demo_training_range"].any():
        st.warning(
            "One or more evaluation times fall outside the synthetic training "
            "ranges. Do not interpret these estimates as reliable."
        )
    if file is None and synthetic_calibration is not None:
        st.caption(
            "The downloadable trajectory contains synthetic-only illustrative trend "
            "interval bounds where eligible. **These are not calibrated to provide "
            "90% simultaneous coverage across a capacitor's history.**"
        )
    with st.expander("View time-by-time predictions"):
        st.dataframe(trajectory, use_container_width=True, hide_index=True)
    st.download_button(
        "Download retrospective RUL results (CSV)",
        data=trajectory.to_csv(index=False),
        file_name=f"{selected}_demo_RUL_trajectory_to_{prediction_hour:.0f}h.csv",
        mime="text/csv",
    )
else:
    st.info("Click **Analyze RUL across the available history** to build the dashboard.")

st.divider()
st.caption(
    "CapLife AI v0.5 · GRU + illustrative multi-indicator trend + "
    "synthetic-only trend intervals · No calibrated uncertainty for real capacitors"
)
