"""Streamlit page: six-parameter health monitoring and neural RUL prediction (demonstration model)."""
from __future__ import annotations

from base64 import b64encode
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from caplife.accounts import save_controls, saved_picker, saving_available
from caplife.measurement_review import read_experimental_csv
from caplife.real_data import RealDataValidationError, validate_real_measurements
from caplife.six_rul import INDICATOR_LABELS, MIN_READINGS, NOMINAL_LIMITS, SixRULModel, example_fleet
from caplife.six_sim import ATMOSPHERE_KPA, EOL_LIMITS, SIGNALS
from caplife.versioning import code_version

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"
UPLOAD, EXAMPLE, SAVED = "Upload my data", "Use simulated example capacitors", "My saved datasets"
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


# Each cache takes code_version() so an app update (new model, simulator or code) refreshes it.
@st.cache_resource(show_spinner=False)
def _load_model(version: str) -> SixRULModel:
    return SixRULModel(ARTIFACT_DIR)


def _model() -> SixRULModel:
    return _load_model(code_version())


@st.cache_data(show_spinner=False)
def _load_example(version: str) -> tuple[pd.DataFrame, dict]:
    return example_fleet()


def _example() -> tuple[pd.DataFrame, dict]:
    return _load_example(code_version())


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
    sources = (UPLOAD, EXAMPLE, SAVED) if saving_available() else (UPLOAD, EXAMPLE)
    source = st.radio("Data", sources, horizontal=True, key="caplife_six_source")
    template = pd.DataFrame(columns=["Cap_ID", "Hours", *SIGNALS]).to_csv(index=False)
    if source == EXAMPLE:
        measurements, truth = _example()
        st.caption("Eight simulated capacitors, two for each way a capacitor can reach end of life, observed part "
                   "of the way there. Because they are simulated, their true end of life is known and shown for "
                   "comparison.")
        st.download_button("Download example CSV", measurements.to_csv(index=False),
                           "caplife_six_parameter_example.csv", mime="text/csv")
        return measurements, truth, "Simulated example"
    if source == SAVED:
        picked = saved_picker("six_parameter", "caplife_six")
        if picked is None:
            return None
        files, name = picked
        return _parse(files["measurements"], name, save=False)
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
    st.caption("Uploads are processed on the app server and are not stored unless you sign in and save them.")
    if upload is None:
        return None
    if upload.size > 8 * 1024 * 1024:
        st.error("The CSV exceeds the 8 MB limit.")
        return None
    return _parse(upload.getvalue(), upload.name, save=True)


def _parse(payload: bytes, label: str, save: bool) -> tuple[pd.DataFrame, dict | None, str] | None:
    try:
        raw = read_experimental_csv(payload)
        missing = [INDICATOR_LABELS[s] for s in SIGNALS if s not in raw.columns]
        if missing:
            raise RealDataValidationError("Missing indicator column(s): " + ", ".join(missing))
        readings = validate_real_measurements(raw)
    except INPUT_ERRORS as error:
        st.error(f"The CSV could not be used: {error}")
        return None
    if save:
        save_controls("six_parameter", {"measurements": payload}, label.removesuffix(".csv"), "caplife_six")
    complete = readings.dropna(subset=list(SIGNALS))
    if len(complete) < len(readings):
        st.warning(f"{len(readings) - len(complete)} reading(s) with a missing indicator were left out. "
                   "No values were filled in.")
    if complete.empty:
        st.error("No reading has all six indicators.")
        return None
    # Capacitance, ESR and leakage are scored as ratios to the first reading, so it must be positive.
    first = complete.sort_values("Hours").groupby("Cap_ID").first()
    unusable = first.index[(first[["Capacitance_uF", "ESR_ohm", "Leakage_uA"]] <= 0).any(axis=1)].tolist()
    if unusable:
        shown = ", ".join(unusable[:5]) + (" …" if len(unusable) > 5 else "")
        st.warning(f"Left out {len(unusable)} capacitor(s) whose first reading has zero leakage current, which "
                   f"cannot serve as a baseline: {shown}.")
        complete = complete.loc[~complete["Cap_ID"].isin(unusable)]
        if complete.empty:
            return None
    return complete.loc[:, ["Cap_ID", "Hours", *SIGNALS]], None, label


@st.cache_data(show_spinner="Assessing capacitors…", max_entries=4)
def _assess_cached(measurements: pd.DataFrame, version: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    model = _model()
    return {device: (readings.sort_values("Hours").reset_index(drop=True), model.assess(readings))
            for device, readings in measurements.groupby("Cap_ID", sort=True)}


def _assess(measurements: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return _assess_cached(measurements, code_version())


def _fleet_table(results: dict, truth: dict | None) -> pd.DataFrame:
    rows = []
    for device, (_, assessment) in results.items():
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


# Mid-tone status colours that read on both the light and the dark theme.
SPARK_COLORS = {"good": "#2a9d8f", "watch": "#d4881a", "critical": "#d9544d", "end-of-life": "#d9544d"}


def _sparkline(hours: pd.Series, values: pd.Series, label: str, status: str = "good") -> str:
    """A small trend of one parameter, scaled to its own range, as an embedded SVG image.

    st.html strips inline <svg>, so the drawing is passed as an <img> data URI with its colour baked in.
    """
    x, y = hours.to_numpy(dtype=float), values.to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(y).all():
        return ""
    width, height, pad = 240.0, 40.0, 4.0
    x_span = (x.max() - x.min()) or 1.0
    y_span = y.max() - y.min()
    xs = pad + (x - x.min()) / x_span * (width - 2 * pad)
    # A flat series is drawn through the middle rather than stretched from noise.
    ys = (np.full_like(y, height / 2) if y_span == 0
          else height - pad - (y - y.min()) / y_span * (height - 2 * pad))
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    area = f"{xs[0]:.1f},{height:.1f} {line} {xs[-1]:.1f},{height:.1f}"
    color = SPARK_COLORS.get(status, SPARK_COLORS["good"])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}">'
           f'<polygon points="{area}" fill="{color}" fill-opacity="0.14"/>'
           f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
           f'stroke-linejoin="round" stroke-linecap="round"/>'
           f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.8" fill="{color}"/></svg>')
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img class="caplife-spark" src="data:image/svg+xml;base64,{encoded}" '
            f'alt="{escape(label)} trend over {len(x)} readings">')


def _health_cards(readings: pd.DataFrame, assessment: pd.DataFrame) -> None:
    baseline, latest, last = readings.iloc[0], readings.iloc[-1], assessment.iloc[-1]
    cards = []
    for signal in SIGNALS:
        health = last[f"{signal}_health_pct"]
        status = _status(health, health <= 0 and signal not in NOMINAL_LIMITS)
        status_key = status.lower().replace(" ", "-")
        value = latest[signal]
        if signal == "Pressure_Abs_kPa":
            change = f"{value - ATMOSPHERE_KPA:+,.1f} kPa above atmospheric"
        elif signal in NOMINAL_LIMITS:
            change = f"{value - baseline[signal]:+,.{DIGITS[signal]}f} {UNITS[signal]} since baseline"
        else:
            change = f"{100 * (value / baseline[signal] - 1):+.1f}% since baseline"
        cards.append(
            f'<div class="caplife-health-card" data-status="{escape(status_key)}">'
            f'<div class="caplife-health-top"><span>{escape(INDICATOR_LABELS[signal])}</span>'
            f'<span class="caplife-health-chip">{escape(status)}</span></div>'
            f'<div class="caplife-health-value">{value:,.{DIGITS[signal]}f} <small>{UNITS[signal]}</small></div>'
            f'<div class="caplife-health-change">{escape(change)}</div>'
            f'{_sparkline(readings["Hours"], readings[signal], INDICATOR_LABELS[signal], status_key)}'
            f'<div class="caplife-health-bar" role="img" aria-label="{health:.0f}% health remaining">'
            f'<span style="width:{health:.1f}%"></span></div>'
            f'<div class="caplife-health-foot"><span>{health:.0f}% health</span>'
            f'<span>Limit: {escape(LIMIT_TEXT[signal])}</span></div></div>'
        )
    st.html('<div class="caplife-health-grid">' + "".join(cards) + "</div>")


def _chart_domain(values: pd.Series, limit: float, digits: int) -> tuple[float, float, str | None]:
    """Y-axis range that shows the data's own trend, and where the limit sits.

    The limit is drawn when it lies within 1.5 data spans of the readings; otherwise the axis
    fits the data and the returned direction ("above" or "below") says where the limit is.
    """
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 0.05 * max(abs(high), abs(low)), 10.0 ** -digits)
    pad = 0.15 * span
    if limit > high and limit - high > 1.5 * span:
        return low - pad, high + pad, "above"
    if limit < low and low - limit > 1.5 * span:
        return low - pad, high + pad, "below"
    return min(low, limit) - pad, max(high, limit) + pad, None


def _signal_charts(readings: pd.DataFrame) -> None:
    baseline = readings.iloc[0]
    columns = st.columns(3)
    for index, signal in enumerate(SIGNALS):
        data = readings[["Hours", signal]].rename(columns={signal: "value"})
        limit = _limit_value(signal, baseline)
        bottom, top, off_chart = _chart_domain(data["value"], limit, DIGITS[signal])
        y = alt.Y("value:Q", title=UNITS[signal], scale=alt.Scale(domain=[bottom, top], zero=False, nice=False))
        line = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=28, filled=True), color=TEAL).encode(
            x=alt.X("Hours:Q", title="Hours"), y=y,
            tooltip=[alt.Tooltip("Hours:Q", format=",.0f"), alt.Tooltip("value:Q", title=INDICATOR_LABELS[signal],
                                                                       format=f",.{DIGITS[signal]}f")])
        if off_chart is None:
            marker = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=ORANGE, strokeDash=[5, 4]).encode(
                y=alt.Y("limit:Q", scale=alt.Scale(domain=[bottom, top], nice=False)),
                tooltip=[alt.Tooltip("limit:Q", title="Limit", format=f",.{DIGITS[signal]}f")])
        else:
            # The limit is far outside the readings: say where it is instead of flattening the trend.
            label = f"Limit {limit:,.{DIGITS[signal]}f} {UNITS[signal]} {'↑' if off_chart == 'above' else '↓'}"
            # Placed on the left: rising curves start low there and falling ones start high,
            # so the note never covers the latest readings at the right.
            marker = alt.Chart(pd.DataFrame({"Hours": [float(data["Hours"].min())],
                                             "value": [top if off_chart == "above" else bottom], "label": [label]})
                               ).mark_text(align="left", dx=4, baseline="top" if off_chart == "above" else "bottom",
                                           dy=4 if off_chart == "above" else -4, color=ORANGE, fontSize=11).encode(
                x="Hours:Q", y=alt.Y("value:Q", scale=alt.Scale(domain=[bottom, top], nice=False)), text="label:N")
        with columns[index % 3]:
            st.caption(f"**{INDICATOR_LABELS[signal]}** · limit: {LIMIT_TEXT[signal]}")
            st.altair_chart((line + marker).properties(height=170), width="stretch")


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
    ]), hide_index=True, width="stretch", column_config={
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
    st.dataframe(fleet, hide_index=True, width="stretch", column_config=config)

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
        st.altair_chart(_rul_chart(assessment, true_eol), width="stretch")
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
