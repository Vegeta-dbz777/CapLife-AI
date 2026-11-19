"""CapLife AI — v0.6 experimental measurements + preserved synthetic RUL demo."""
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
        ("Experimental measurements (v0.6)", "Synthetic RUL demonstration (v0.5)"),
        key="caplife_workflow_v06",
    )
    st.divider()

if workflow == "Experimental measurements (v0.6)":
    from io import BytesIO
    from caplife.real_data import (
        RealDataValidationError,
        assess_real_measurements,
        assessment_zip_bytes,
        blank_template_csv,
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
        file_name="CapLife_Real_Data_Template_v0_6.csv",
        mime="text/csv",
    )
    st.caption(
        "Measured columns: Capacitance_uF, ESR_ohm, Leakage_uA, "
        "Case_Expansion_um. Optional: environmental, electrical, and ESR test conditions. "
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
    except (RealDataValidationError, ValueError, pd.errors.ParserError, UnicodeError) as exc:
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
        for c in ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um")
    )))
    st.success("Basic data checks passed. This does not establish measurement validity or real-world reliability.")
    for warning in warnings:
        st.warning(warning)

    st.subheader("2 · Measurement-quality report")
    st.caption("Zero recorded readings means 'not supplied' or 'no measurements' — not a zero sensor reading.")
    st.dataframe(quality, hide_index=True, use_container_width=True)

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
    with st.expander("View processed measurements"):
        st.dataframe(cap_rows, hide_index=True, use_container_width=True)
    st.subheader("5 · Download assessment package")
    st.download_button(
        "Download all-capacitor assessment (ZIP)",
        data=assessment_zip_bytes(real_report),
        file_name="CapLife_v0_6_Measurement_Assessment.zip",
        mime="application/zip",
    )
    st.info(
        "**Limitations:** This mode does not determine health, EOL, RUL, uncertainty "
        "intervals, or maintenance actions. A complete dataset can still have inaccurate "
        "sensors, changing operating conditions, or incomparable test settings. "
        "The uploaded file is processed by the web-app server; check hosting/privacy "
        "policies before submitting confidential experiments."
    )
    st.caption("CapLife AI v0.6 · descriptive measurements only · no neural-network prediction")
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
