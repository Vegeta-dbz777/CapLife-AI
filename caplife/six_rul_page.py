"""Streamlit page: six-parameter health monitoring and neural RUL prediction (demonstration model)."""
from __future__ import annotations

from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from caplife.measurement_review import read_experimental_csv
from caplife.real_data import RealDataValidationError, validate_real_measurements
from caplife.six_rul import INDICATOR_LABELS, MIN_READINGS, NOMINAL_LIMITS, SixRULModel, example_fleet
from caplife.six_sim import ATMOSPHERE_KPA, EOL_LIMITS, SIGNALS

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"
UPLOAD, EXAMPLE = "Upload my data", "Use simulated example capacitors"
UNITS = {"Capacitance_uF": "µF", "ESR_ohm": "Ω", "Leakage_uA": "µA", "Case_Expansion_um": "µm",
         "H2_Generated_mL_STP": "mL", "Pressure_Abs_kPa": "kPa"}
DIGITS = {"Capacitance_uF": 1, "ESR_ohm": 4, "Leakage_uA": 2, "Case_Expansion_um": 1,
          "H2_Generated_mL_STP": 2, "Pressure_Abs_kPa": 1}
LIMIT_TEXT = {
    "Capacitance_uF": f"{EOL_LIMITS['capacitance_loss_pct']:.0f}% loss",
    "ESR_ohm": f"{EOL_LIMITS['esr_ratio']:g}× initial",
    "Leakage_uA": f"{EOL_LIMITS['leakage_ratio']:g}× initial",
    "Case_Expansion_um": f"{NOMINAL_LIMITS['Case_Expansion_um']:g} µm (nominal)",
    "H2_Generated_mL_STP": f"{NOMINAL_LIMITS['H2_Generated_mL_STP']:g} mL (nominal)",
    "Pressure_Abs_kPa": f"{EOL_LIMITS['pressure_abs_kpa']:g} kPa vent",
}
TEAL, ORANGE, GREY = "#2f9e8f", "#d9822b", "#8a9a9a"
INPUT_ERRORS = (RealDataValidationError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeError)


@st.cache_resource(show_spinner=False)
def _model() -> SixRULModel:
    return SixRULModel(ARTIFACT_DIR)


@st.cache_data(show_spinner=False)
def _example() -> tuple[pd.DataFrame, dict]:
    return example_fleet()


def _status(health: float, reached: bool) -> str:
    if reached:
        return "End of life"
    return "Good" if health >= 50 else "Watch" if health >= 20 else "Critical"


def _limit_value(signal: str, baseline: pd.Series) -> float:
    return {
        "Capacitance_uF": baseline["Capacitance_uF"] * (1 - EOL_LIMITS["capacitance_loss_pct"] / 100),
        "ESR_ohm": baseline["ESR_ohm"] * EOL_LIMITS["esr_ratio"],
        "Leakage_uA": baseline["Leakage_uA"] * EOL_LIMITS["leakage_ratio"],
        "Case_Expansion_um": baseline["Case_Expansion_um"] + NOMINAL_LIMITS["Case_Expansion_um"],
        "H2_Generated_mL_STP": baseline["H2_Generated_mL_STP"] + NOMINAL_LIMITS["H2_Generated_mL_STP"],
        "Pressure_Abs_kPa": EOL_LIMITS["pressure_abs_kpa"],
    }[signal]


def _hours(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f} h"


def _short(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 10_000 else f"{value:,.0f}"


def _load() -> tuple[pd.DataFrame, dict | None, str] | None:
    source = st.radio("Data", (UPLOAD, EXAMPLE), horizontal=True, key="caplife_six_source")
    template = pd.DataFrame(columns=["Cap_ID", "Hours", *SIGNALS]).to_csv(index=False)
    if source == EXAMPLE:
        measurements, truth = _example()
        st.caption("Eight simulated capacitors, two for each way a capacitor can reach end of life, observed part "
                   "of the way there. Because they are simulated, their true end of life is known and shown for "
                   "comparison.")
        st.download_button("Download example CSV", measurements.to_csv(index=False),
                           "caplife_six_parameter_example.csv", mime="text/csv")
        return measurements, truth, "Simulated example"
    upload = st.file_uploader("Measurement CSV", type=["csv"], key="caplife_six_upload")
    left, right = st.columns([1, 2])
    left.download_button("Download CSV template", template, "caplife_six_parameter_template.csv", mime="text/csv")
    with right.expander("Columns and units"):
        st.markdown(
            "One row per reading: `Cap_ID`, `Hours`, `Capacitance_uF`, `ESR_ohm`, `Leakage_uA`, "
            "`Case_Expansion_um`, `H2_Generated_mL_STP` (cumulative, at 0 °C and 1 atm) and `Pressure_Abs_kPa` "
            "(absolute). The first reading of each capacitor is its baseline. Remaining life needs at least "
            f"{MIN_READINGS} complete readings."
        )
    st.caption("Uploads are processed on the app server. Share data you have permission to use.")
    if upload is None:
        return None
    if upload.size > 8 * 1024 * 1024:
        st.error("The CSV exceeds the 8 MB limit.")
        return None
    try:
        raw = read_experimental_csv(upload.getvalue())
        missing = [INDICATOR_LABELS[s] for s in SIGNALS if s not in raw.columns]
        if missing:
            raise RealDataValidationError("Missing indicator column(s): " + ", ".join(missing))
        readings = validate_real_measurements(raw)
    except INPUT_ERRORS as error:
        st.error(f"The CSV could not be used: {error}")
        return None
    complete = readings.dropna(subset=list(SIGNALS))
    if len(complete) < len(readings):
        st.warning(f"{len(readings) - len(complete)} reading(s) with a missing indicator were left out. "
                   "No values were filled in.")
    if complete.empty:
        st.error("No reading has all six indicators.")
        return None
    return complete.loc[:, ["Cap_ID", "Hours", *SIGNALS]], None, upload.name


def _assess(measurements: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    model = _model()
    return {device: (readings.sort_values("Hours").reset_index(drop=True), model.assess(readings))
            for device, readings in measurements.groupby("Cap_ID", sort=True)}


def _fleet_table(results: dict, truth: dict | None) -> pd.DataFrame:
    rows = []
    for device, (readings, assessment) in results.items():
        last = assessment.iloc[-1]
        has_rul = "RUL_median_h" in assessment and pd.notna(last.get("RUL_median_h"))
        row = {
            "Capacitor": device,
            "Latest reading (h)": last["Hours"],
            "Overall health (%)": round(last["Overall_health_pct"]),
            "Limiting indicator": last["Limiting_indicator"],
            "Status": _status(last["Overall_health_pct"], bool(last["End_of_life_reached"])),
            "Remaining life, median (h)": last["RUL_median_h"] if has_rul else np.nan,
            "Remaining life, 80% range (h)": (f"{last['RUL_low_h']:,.0f} – {last['RUL_high_h']:,.0f}"
                                              if has_rul else f"Needs {MIN_READINGS} readings"),
            "Predicted end-of-life age (h)": last["Hours"] + last["RUL_median_h"] if has_rul else np.nan,
        }
        if truth:
            row["Simulated true end of life (h)"] = truth[device]["eol_h"]
        rows.append(row)
    return pd.DataFrame(rows)


def _health_cards(readings: pd.DataFrame, assessment: pd.DataFrame) -> None:
    baseline, latest, last = readings.iloc[0], readings.iloc[-1], assessment.iloc[-1]
    cards = []
    for signal in SIGNALS:
        health = last[f"{signal}_health_pct"]
        status = _status(health, health <= 0 and signal not in NOMINAL_LIMITS)
        value = latest[signal]
        if signal == "Pressure_Abs_kPa":
            change = f"{value - ATMOSPHERE_KPA:+,.1f} kPa above atmospheric"
        elif signal in NOMINAL_LIMITS:
            change = f"{value - baseline[signal]:+,.{DIGITS[signal]}f} {UNITS[signal]} since baseline"
        else:
            change = f"{100 * (value / baseline[signal] - 1):+.1f}% since baseline"
        cards.append(
            f'<div class="caplife-health-card" data-status="{escape(status.lower().replace(" ", "-"))}">'
            f'<div class="caplife-health-top"><span>{escape(INDICATOR_LABELS[signal])}</span>'
            f'<span class="caplife-health-chip">{escape(status)}</span></div>'
            f'<div class="caplife-health-value">{value:,.{DIGITS[signal]}f} <small>{UNITS[signal]}</small></div>'
            f'<div class="caplife-health-change">{escape(change)}</div>'
            f'<div class="caplife-health-bar" role="img" aria-label="{health:.0f}% health remaining">'
            f'<span style="width:{health:.1f}%"></span></div>'
            f'<div class="caplife-health-foot"><span>{health:.0f}% health</span>'
            f'<span>Limit: {escape(LIMIT_TEXT[signal])}</span></div></div>'
        )
    st.html('<div class="caplife-health-grid">' + "".join(cards) + "</div>")


def _signal_charts(readings: pd.DataFrame) -> None:
    baseline = readings.iloc[0]
    columns = st.columns(3)
    for index, signal in enumerate(SIGNALS):
        data = readings[["Hours", signal]].rename(columns={signal: "value"})
        limit = _limit_value(signal, baseline)
        y = alt.Y("value:Q", title=UNITS[signal], scale=alt.Scale(zero=False))
        line = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=28, filled=True), color=TEAL).encode(
            x=alt.X("Hours:Q", title="Hours"), y=y,
            tooltip=[alt.Tooltip("Hours:Q", format=",.0f"), alt.Tooltip("value:Q", title=INDICATOR_LABELS[signal],
                                                                       format=f",.{DIGITS[signal]}f")])
        rule = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=ORANGE, strokeDash=[5, 4]).encode(
            y="limit:Q", tooltip=[alt.Tooltip("limit:Q", title="Limit", format=f",.{DIGITS[signal]}f")])
        with columns[index % 3]:
            st.caption(f"**{INDICATOR_LABELS[signal]}** · dashed line: limit, {LIMIT_TEXT[signal]}")
            st.altair_chart((line + rule).properties(height=170), use_container_width=True)


def _rul_chart(assessment: pd.DataFrame, true_eol: float | None) -> alt.Chart:
    data = assessment.dropna(subset=["RUL_median_h"])
    band = alt.Chart(data).mark_area(opacity=.25, color=TEAL).encode(
        x=alt.X("Hours:Q", title="Capacitor age when the prediction was made (h)"),
        y=alt.Y("RUL_low_h:Q", title="Remaining useful life (h)"), y2="RUL_high_h:Q")
    line = alt.Chart(data).mark_line(point=True, color=TEAL).encode(
        x="Hours:Q", y="RUL_median_h:Q",
        tooltip=[alt.Tooltip("Hours:Q", title="Age (h)", format=",.0f"),
                 alt.Tooltip("RUL_median_h:Q", title="Predicted RUL (h)", format=",.0f"),
                 alt.Tooltip("RUL_low_h:Q", title="80% range from", format=",.0f"),
                 alt.Tooltip("RUL_high_h:Q", title="80% range to", format=",.0f")])
    layers = [band, line]
    if true_eol is not None:
        truth = pd.DataFrame({"Hours": data["Hours"], "true": np.maximum(true_eol - data["Hours"], 0)})
        layers.append(alt.Chart(truth).mark_line(color=ORANGE, strokeDash=[6, 4]).encode(
            x="Hours:Q", y="true:Q", tooltip=[alt.Tooltip("true:Q", title="True RUL (h)", format=",.0f")]))
    return alt.layer(*layers).properties(height=280)


def _model_card() -> None:
    meta = _model().metadata
    metrics = meta["test_metrics"]
    neural, trend = metrics["neural_calibrated"], metrics["trend_baseline"]
    fleets = meta["fleets"]
    st.markdown(
        f"- **Network:** a GRU (recurrent neural network) reads up to the last {meta['window']} readings. Each "
        "indicator enters as the fraction of its limit already used, with the capacitor's age and the time "
        "between readings. It outputs the 10th, 50th and 90th percentiles of remaining life.\n"
        f"- **Training data:** {fleets['train']['capacitors']:,} simulated capacitors "
        f"({fleets['train']['predictions']:,} prediction points) from a physics-inspired simulator: "
        "temperature-accelerated electrolyte loss lowers capacitance and raises ESR; leakage current generates "
        "hydrogen by Faraday's law; retained hydrogen raises internal pressure and bulges the case. "
        "Coefficients are illustrative, not fitted to a real capacitor series.\n"
        f"- **End of life:** capacitance loss ≥ {EOL_LIMITS['capacitance_loss_pct']:.0f}%, ESR ≥ "
        f"{EOL_LIMITS['esr_ratio']:g}× initial, leakage ≥ {EOL_LIMITS['leakage_ratio']:g}× initial, or internal "
        f"pressure ≥ {EOL_LIMITS['pressure_abs_kpa']:g} kPa (vent). Expansion and H₂ use nominal limits that "
        "correspond to the vent pressure for a typical simulated capacitor.\n"
        f"- **80% range:** calibrated on {fleets['calibration']['capacitors']} further simulated capacitors "
        "(conformalized quantile regression)."
    )
    st.markdown(f"**Test on {fleets['test']['capacitors']} simulated capacitors the model never saw** "
                f"({neural['predictions']:,} predictions)")
    st.dataframe(pd.DataFrame([
        {"Method": "Neural network (GRU)", "Typical error (h)": neural["median_absolute_error_h"],
         "Typical relative error": neural["median_relative_error"],
         "True RUL inside 80% range": neural["interval_coverage"],
         "Typical 80% range width (h)": neural["median_interval_width_h"]},
        {"Method": "Trend line to nearest limit", "Typical error (h)": trend["median_absolute_error_h"],
         "Typical relative error": None, "True RUL inside 80% range": None, "Typical 80% range width (h)": None},
    ]), hide_index=True, use_container_width=True, column_config={
        "Typical error (h)": st.column_config.NumberColumn(format="%.0f"),
        "Typical relative error": st.column_config.NumberColumn(format="percent"),
        "True RUL inside 80% range": st.column_config.NumberColumn(format="percent"),
        "Typical 80% range width (h)": st.column_config.NumberColumn(format="%.0f"),
    })
    st.caption("Typical error is the median absolute difference between predicted and true remaining life. "
               "These results hold for the simulator. Real capacitors may age differently, so accuracy on real "
               "data is unknown until the model is tested on measured failures. Retrain with "
               "`python -m caplife.six_rul`.")


def render() -> None:
    st.header("Six-parameter health and remaining-life prediction")
    st.write("Upload measurement histories of capacitance, ESR, leakage current, case expansion, generated "
             "hydrogen and internal pressure. CapLife scores the health of each parameter against its limit and a "
             "neural network predicts remaining useful life with an 80% range.")
    st.info("**Demonstration model.** The neural network learned from simulated capacitors, not laboratory "
            "measurements. It shows how six-parameter RUL prediction works; its accuracy on real capacitors "
            "has not been tested.")
    loaded = _load()
    if loaded is None:
        return
    measurements, truth, label = loaded
    results = _assess(measurements)
    fleet = _fleet_table(results, truth)

    st.subheader("Fleet overview")
    st.caption(label)
    cards = st.columns(3)
    cards[0].metric("Capacitors", len(fleet))
    cards[1].metric("Need attention", int(fleet["Status"].isin(["Watch", "Critical", "End of life"]).sum()))
    shortest = fleet["Remaining life, median (h)"].min()
    cards[2].metric("Shortest predicted remaining life", _hours(shortest))
    config = {"Remaining life, median (h)": st.column_config.NumberColumn(format="%.0f"),
              "Predicted end-of-life age (h)": st.column_config.NumberColumn(format="%.0f"),
              "Latest reading (h)": st.column_config.NumberColumn(format="%.0f"),
              "Overall health (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")}
    if truth:
        config["Simulated true end of life (h)"] = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(fleet, hide_index=True, use_container_width=True, column_config=config)

    st.subheader("Capacitor detail")
    device = st.selectbox("Capacitor", list(results), key="caplife_six_device")
    readings, assessment = results[device]
    last = assessment.iloc[-1]
    if bool(last["End_of_life_reached"]):
        st.error(f"An end-of-life limit is already met at {last['Hours']:,.0f} h ({last['Limiting_indicator']}).")
    elif "RUL_median_h" not in assessment or pd.isna(last.get("RUL_median_h")):
        st.warning(f"Remaining life needs at least {MIN_READINGS} readings; this capacitor has {len(readings)}.")
    else:
        top = st.columns(4)
        top[0].metric("Remaining useful life", _hours(last["RUL_median_h"]))
        top[1].metric("80% range", f"{_short(last['RUL_low_h'])}–{_short(last['RUL_high_h'])} h")
        top[2].metric("Predicted end-of-life age", _hours(last["Hours"] + last["RUL_median_h"]))
        top[3].metric("Overall health", f"{last['Overall_health_pct']:.0f}%")
        note = f"Based on {len(readings)} readings up to {last['Hours']:,.0f} h. Limiting indicator: " \
               f"{last['Limiting_indicator']}."
        if truth:
            true_rul = truth[device]["eol_h"] - last["Hours"]
            note += f" Simulated true remaining life: {true_rul:,.0f} h ({truth[device]['eol_cause']})."
        st.caption(note)
        if bool(last.get("Outside_training_range")):
            st.warning("Some readings go beyond what the simulated training capacitors showed, so this prediction "
                       "is an extrapolation.")
    _health_cards(readings, assessment)

    st.markdown("##### Six-parameter history")
    _signal_charts(readings)
    if assessment["RUL_median_h"].notna().any() if "RUL_median_h" in assessment else False:
        st.markdown("##### Remaining-life prediction over time")
        true_eol = truth[device]["eol_h"] if truth else None
        st.altair_chart(_rul_chart(assessment, true_eol), use_container_width=True)
        st.caption("Each point uses only the readings available at that age. Shaded band: 80% range."
                   + (" Dashed orange line: the simulated true remaining life." if truth else ""))

    with st.expander("About the model and how well it did on simulated data"):
        _model_card()

    st.subheader("Downloads")
    left, right = st.columns(2)
    left.download_button("Download fleet summary (CSV)", fleet.to_csv(index=False), "caplife_rul_fleet.csv",
                         mime="text/csv", type="primary")
    right.download_button(f"Download {device} assessment (CSV)",
                          readings.merge(assessment, on="Hours").to_csv(index=False),
                          f"caplife_rul_{device}.csv", mime="text/csv")
    st.caption("Demonstration model trained on simulated capacitors. Not a maintenance or safety tool.")
