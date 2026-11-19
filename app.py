"""CapLife AI — synthetic-data research demonstrator, not a validated reliability tool."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from caplife.config import FEATURES, SEQ_LEN
from caplife.data import DataValidationError, validate_measurements
from caplife.predict import assess_capacitor, load_engine
from caplife.trend import assess_trend


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "sample_data" / "demo_capacitors.csv"
TEMPLATE_DATA = ROOT / "sample_data" / "upload_template.csv"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="CapLife AI — AEC RUL Prototype",
    page_icon="⚡",
    layout="wide",
)
st.title("⚡ CapLife AI")
st.caption("Aluminum Electrolytic Capacitor Health Monitoring & Remaining Useful Life")
st.warning(
    "**RESEARCH DEMONSTRATOR — NOT VALIDATED FOR REAL CAPACITORS.** "
    "This neural network was trained on invented synthetic aging histories. "
    "Its RUL predictions are examples, not trustworthy lifetime or maintenance advice."
)


@st.cache_resource(show_spinner="Loading the demonstration neural network…")
def get_engine():
    return load_engine(ARTIFACTS)


try:
    net, metadata = get_engine()
except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
    st.error("Model files are missing or incompatible. See README.md for training instructions.")
    st.code("python -m caplife.train --root .", language="bash")
    st.stop()

with st.sidebar:
    st.header("About this prototype")
    st.write("**AI engine:** GRU recurrent neural network")
    st.write(f"**Historical window:** {SEQ_LEN} measurements")
    st.write("**Source:** simulated AEC aging only")
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
        "**These are uncalibrated estimates.** No confidence interval or validated "
        "health score is available. The model is trained only on synthetic histories."
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
        rows.append({
            "Prediction_hour": float(assessment["hours_observed"]),
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
    "CapLife AI v0.4 · GRU + illustrative multi-indicator trend · "
    "Synthetic-only demonstration; no real-world validation or calibrated uncertainty"
)
