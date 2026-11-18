"""CapLife AI — shareable Streamlit demonstrator, not a clinical/reliability tool."""
from __future__ import annotations
from pathlib import Path
import io
import json
import pandas as pd
import streamlit as st
from caplife.config import FEATURES, SEQ_LEN
from caplife.data import DataValidationError, validate_measurements
from caplife.predict import load_engine, assess_capacitor

ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "sample_data" / "demo_capacitors.csv"
TEMPLATE_DATA = ROOT / "sample_data" / "upload_template.csv"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="CapLife AI — AEC RUL Prototype", page_icon="⚡", layout="wide")
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
except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as e:
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
    st.write("Each capacitor needs at least **six chronological measurements**. One row represents one measurement.")
    st.write("Required headers:")
    st.code("Cap_ID, " + ", ".join(FEATURES), language=None)
    st.download_button(
        "Download CSV template",
        data=TEMPLATE_DATA.read_bytes(),
        file_name="aec_measurements_template.csv",
        mime="text/csv", use_container_width=True,
    )
    with st.expander("Measurement units"):
        for key, description in metadata["unit_conventions"].items():
            st.markdown(f"**{key}** — {description}")
    st.divider()
    st.caption("Uploaded data are processed by the web-app server; review hosting/privacy policies before uploading confidential experiments.")

st.subheader("1 · Choose a dataset")
file = st.file_uploader("Upload capacitor measurement history", type=["csv"], help="CSV file; max 8 MB")
if file is not None:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    st.caption(f"Using uploaded dataset: **{file.name}**")
else:
    st.info("No file uploaded. Using the bundled **synthetic example dataset** so you can explore the full workflow.")
    df = pd.read_csv(DEMO_DATA)

try:
    data = validate_measurements(df)
except DataValidationError as e:
    st.error(f"Dataset validation failed: {e}")
    st.info("Use the CSV template in the sidebar. Units and measurement methods must also match the model's documented schema.")
    st.stop()

available_caps = data["Cap_ID"].unique().tolist()
col1, col2, col3 = st.columns(3)
col1.metric("Capacitors", f"{len(available_caps):,}")
col2.metric("Measurement rows", f"{len(data):,}")
col3.metric("AI input features", str(len(FEATURES)))

st.subheader("2 · Select a capacitor")
selected = st.selectbox("Capacitor ID", available_caps)
cap_df = data.loc[data["Cap_ID"] == selected].copy().sort_values("Hours")
st.caption(f"This capacitor has {len(cap_df)} measurements, from hour {cap_df['Hours'].min():.0f} to {cap_df['Hours'].max():.0f}.")

if st.button("Analyze capacitor", type="primary", use_container_width=True):
    result = assess_capacitor(cap_df, net, metadata)
    st.session_state["last_assessment"] = result
    st.session_state["last_assessed_cap"] = selected
    st.session_state["last_data_tag"] = f"{file.name}:{file.size}" if file else "bundled-demo"

current_tag = f"{file.name}:{file.size}" if file else "bundled-demo"
result = st.session_state.get("last_assessment")
if (result and st.session_state.get("last_assessed_cap") == selected
        and st.session_state.get("last_data_tag") == current_tag):
    st.subheader("3 · RUL assessment")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated remaining life (DEMO)", f"{result['rul_hours']:,.0f} h")
    m2.metric("Measured operating age", f"{result['hours_observed']:,.0f} h")
    m3.metric("Estimated total lifetime (DEMO)", f"{result['hours_observed'] + result['rul_hours']:,.0f} h")
    st.write("**Threshold status:** " + result["status"])
    if result["eol_causes"]:
        st.error("Illustrative EOL trigger(s): " + ", ".join(result["eol_causes"]))
    for warning in result["warnings"]:
        st.error("Outside training range — do not rely on prediction: " + warning)
    st.caption(
        "**These are uncalibrated estimates.** No confidence interval or validated health score "
        "is available. At a demo EOL threshold, RUL is set to zero by a rule, not predicted by the GRU."
    )
    report = pd.DataFrame([{
        "Cap_ID": result["cap_id"],
        "Age_hours": result["hours_observed"],
        "Demo_predicted_RUL_hours": round(result["rul_hours"], 2),
        "Threshold_status": result["status"],
        "Triggered_demo_thresholds": "; ".join(result["eol_causes"]),
        "Out_of_training_range": "; ".join(result["warnings"]),
        "Model": result["model_id"],
        "Important_warning": "Synthetic-trained model; NOT validated for real AECs",
    }])
    st.download_button("Download assessment CSV", data=report.to_csv(index=False),
                       file_name="caplife_demo_assessment.csv", mime="text/csv")

st.subheader("4 · Visualize measured degradation")
left, right = st.columns(2)
with left:
    st.write("**Capacitance ratio** (1.0 = initial capacitance)")
    st.line_chart(cap_df.set_index("Hours")[["C_ratio"]])
    st.write("**ESR ratio** (1.0 = initial ESR)")
    st.line_chart(cap_df.set_index("Hours")[["ESR_ratio"]])
with right:
    st.write("**Leakage current ratio**")
    st.line_chart(cap_df.set_index("Hours")[["Leakage_ratio"]])
    st.write("**Case strain index** (hypothetical)")
    st.line_chart(cap_df.set_index("Hours")[["Case_Strain_Index"]])
with st.expander("View full measurement history"):
    st.dataframe(cap_df, use_container_width=True, hide_index=True)
    st.download_button("Download selected history", data=cap_df.to_csv(index=False),
                       file_name=f"{selected}_measurements.csv", mime="text/csv")
with st.expander("Model testing — synthetic capacitors only"):
    st.json(metadata["metrics"])
    st.write("The train/validation/test splits contain different capacitor IDs. "
             "The scores describe invented test trajectories and cannot be generalized to real AECs.")
st.divider()
st.caption("CapLife AI v0.1 · GRU sequence regressor · No real-world validation or calibrated uncertainty")
