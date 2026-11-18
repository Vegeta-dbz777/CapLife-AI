"""CapLife AI — synthetic-data research demonstrator, not a validated reliability tool."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from caplife.config import FEATURES, SEQ_LEN
from caplife.data import DataValidationError, validate_measurements
from caplife.predict import assess_capacitor, load_engine


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
    except (DataValidationError, ValueError) as exc:
        st.error(f"Could not assess this capacitor: {exc}")
        st.stop()

    st.session_state["last_assessment"] = assessment
    st.session_state["last_assessed_cap"] = selected
    st.session_state["last_data_tag"] = current_tag
    st.session_state["last_prediction_hour"] = prediction_hour

result = st.session_state.get("last_assessment")
result_is_current = (
    result is not None
    and st.session_state.get("last_assessed_cap") == selected
    and st.session_state.get("last_data_tag") == current_tag
    and st.session_state.get("last_prediction_hour") == prediction_hour
)

st.subheader("3 · RUL assessment")
if result_is_current:
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated remaining life (DEMO)", f"{result['rul_hours']:,.0f} h")
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
    report = pd.DataFrame([{
        "Cap_ID": result["cap_id"],
        "Prediction_hour": prediction_hour,
        "Age_hours": result["hours_observed"],
        "Demo_predicted_RUL_hours": round(result["rul_hours"], 2),
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

st.divider()
st.caption(
    "CapLife AI v0.2 · GRU sequence regressor · "
    "No real-world validation or calibrated uncertainty"
)
