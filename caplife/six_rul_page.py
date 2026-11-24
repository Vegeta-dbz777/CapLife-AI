"""Streamlit page "Monitor and Predict": six-parameter health monitoring and neural RUL prediction."""
from __future__ import annotations

from base64 import b64encode
from html import escape
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from caplife.tables import table
from caplife.chart_hover import NO_TOOLTIP, hoverable as _hoverable, hoverable_rule as _hoverable_rule
from caplife.monitor_report import MonitorReport, pdf_bytes, xlsx_bytes

from caplife.accounts import save_controls, saved_picker
from caplife.file_reader import read_measurement_file
from caplife.upload_ui import UPLOAD_TYPES, show_notes
from caplife.real_data import RealDataValidationError, validate_real_measurements
from caplife.six_rul import (
    COMMON_SUBSET, INDICATOR_LABELS, LIMIT_KEYS, MIN_READINGS, SixRULModel, current_limit_used,
    example_fleet, full_limits, limit_used, subset_key, value_at_fraction,
)
from caplife import operating_conditions as conditions
from caplife.six_sim import ATMOSPHERE_KPA, CONDITION_COLUMNS, EOL_LIMITS, LIMIT_RANGES, SIGNALS
from caplife.versioning import code_version

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"
UPLOAD, EXAMPLE, SAVED = "Upload my data", "Use simulated example capacitors", "My saved datasets"
UNITS = {"Capacitance_uF": "µF", "ESR_ohm": "Ω", "Leakage_uA": "µA", "Case_Expansion_um": "µm",
         "H2_Generated_mL_STP": "mL", "Pressure_Abs_kPa": "kPa"}
DIGITS = {"Capacitance_uF": 1, "ESR_ohm": 4, "Leakage_uA": 2, "Case_Expansion_um": 1,
          "H2_Generated_mL_STP": 2, "Pressure_Abs_kPa": 1}
# Hover values: two digits after the decimal point; ESR keeps four (its values are around 0.1 ohm).
HOVER_FORMAT = {signal: ",.4f" if signal == "ESR_ohm" else ",.2f" for signal in DIGITS}
# The six end-of-life limit inputs: label, the limit key, how the box shows it, and its step.
# ESR and leakage limits are entered as a rise in percent (100% rise = 2x the first reading).
LIMIT_INPUTS = {
    "Capacitance_uF": ("Capacitance loss (%)", 0.5),
    "ESR_ohm": ("ESR rise (%)", 5.0),
    "Leakage_uA": ("Leakage current rise (%)", 10.0),
    "Case_Expansion_um": ("Case expansion (µm)", 1.0),
    "H2_Generated_mL_STP": ("H₂ generated (mL)", 0.1),
    "Pressure_Abs_kPa": ("Internal pressure (kPa, absolute)", 5.0),
}
LIMITS_STATE = "caplife_six_limits"
CONDITIONS_STATE = "caplife_six_conditions"  # per capacitor: operating conditions the user changed
BADGE_COLORS = {"error": "red", "warning": "orange", "info": "green"}


def _shown_limit(signal: str, value: float) -> float:
    """A limit as its input box shows it."""
    return round((value - 1) * 100, 6) if signal in ("ESR_ohm", "Leakage_uA") else value


def _stored_limit(signal: str, shown: float) -> float:
    return 1 + shown / 100 if signal in ("ESR_ohm", "Leakage_uA") else float(shown)


def _limit_text(signal: str, limits: dict | None = None) -> str:
    value = full_limits(limits)[LIMIT_KEYS[signal]]
    if signal == "Capacitance_uF":
        return f"{value:g}% loss"
    if signal in ("ESR_ohm", "Leakage_uA"):
        return f"+{(value - 1) * 100:g}% ({value:g}× initial)"
    if signal == "Case_Expansion_um":
        return f"{value:g} µm bulge"
    if signal == "H2_Generated_mL_STP":
        return f"{value:g} mL"
    return f"{value:g} kPa vent"
TEAL, ORANGE, GREY = "#2f9e8f", "#d9822b", "#8a9a9a"
DIVIDER = "#9fb0bd"  # the latest reading: measured to the left, forecast to the right
PURPLE = "#8e6bd8"
GOLD = "#d9a514"  # changed operating conditions (opposite purple, which is the recorded-conditions forecast)  # forecasts: distinct from measured readings (teal) and limits (orange) in both themes
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


# Health is the share of a parameter's allowance to its limit that is still unused. "Good" needs
# under 30% of it used (for ESR, a rise of under 30% towards the 2x limit); past 60% is "Critical".
GOOD_HEALTH_PCT, WATCH_HEALTH_PCT = 70, 40


def _status(health: float, reached: bool) -> str:
    if reached:
        return "End of life"
    return "Good" if health >= GOOD_HEALTH_PCT else "Watch" if health >= WATCH_HEALTH_PCT else "Critical"


def _value_at(signal: str, baseline: pd.Series, used: float, limits: dict | None = None) -> float:
    """The parameter value at which the given fraction of its limit is used."""
    return float(value_at_fraction(signal, baseline, used, limits))


def _limit_value(signal: str, baseline: pd.Series, limits: dict | None = None) -> float:
    """The parameter's end-of-life limit in its own units."""
    return _value_at(signal, baseline, 1.0, limits)


def _hours(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.0f} h"


def _short(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 10_000 else f"{value:,.0f}"


def _load() -> tuple[pd.DataFrame, dict | None, str] | None:
    source = st.radio("Data", (UPLOAD, EXAMPLE, SAVED), horizontal=True, key="caplife_six_source", persist_state="session")
    template = pd.DataFrame(columns=["Cap_ID", "Hours", *SIGNALS, *CONDITION_COLUMNS]).to_csv(index=False)
    if source == EXAMPLE:
        measurements, truth = _example()
        st.caption("Twelve simulated capacitors, two for each of the six ways a capacitor can reach end of life, "
                   "observed part of the way there. Because they are simulated, their true end of life with the "
                   "default limits is known and shown for comparison.")
        if st.checkbox("Use only capacitance, ESR and leakage current", key="caplife_six_three_only", persist_state="session",
                       help="Shows how the prediction works when the other three parameters are not measured."):
            measurements = measurements.loc[:, ["Cap_ID", "Hours", *COMMON_SUBSET,
                                                *conditions.recorded_columns(measurements)]]
        st.download_button("Download example CSV", measurements.to_csv(index=False),
                           "caplife_six_parameter_example.csv", mime="text/csv")
        return measurements, truth, "Simulated example"
    if source == SAVED:
        picked = saved_picker("six_parameter", "caplife_six")
        if picked is None:
            return None
        files, name = picked
        return _parse(files["measurements"], name, save=False)
    upload = st.file_uploader("Measurement file (CSV or Excel)", type=UPLOAD_TYPES, key="caplife_six_upload")
    st.download_button("Download CSV template", template, "caplife_six_parameter_template.csv", mime="text/csv")
    st.caption("Uploads are processed on the app server and are not stored unless you sign in and save them.")
    if upload is None:
        return None
    if upload.size > 8 * 1024 * 1024:
        st.error("The CSV exceeds the 8 MB limit.")
        return None
    return _parse(upload.getvalue(), upload.name, save=True)


def _parse(payload: bytes, label: str, save: bool,
           settings: tuple[str, str] = ("hours", "auto")) -> tuple[pd.DataFrame, dict | None, str] | None:
    try:
        raw, notes = read_measurement_file(payload, label, *settings)
        readings = validate_real_measurements(raw)
    except INPUT_ERRORS as error:
        st.error(f"The file could not be used: {error}")
        return None
    if save:
        save_controls("six_parameter", {"measurements": payload}, label.rsplit(".", 1)[0], "caplife_six")
    present = [signal for signal in SIGNALS if signal in readings and readings[signal].notna().any()]
    if not present:
        st.error("The file has none of the six parameters.")
        return None
    kept, dropped_readings, unusable, empty = [], {}, [], []
    for device, group in readings.sort_values("Hours").groupby("Cap_ID", sort=True):
        # A capacitor's parameters are the ones measured in at least half of its readings; readings
        # missing one of them are left out, and nothing is filled in.
        own = [signal for signal in present if group[signal].notna().mean() >= 0.5]
        if not own:
            empty.append(str(device))
            continue
        complete = group.dropna(subset=own)
        if len(complete) < len(group):
            dropped_readings[str(device)] = len(group) - len(complete)
        # Capacitance, ESR and leakage are scored as ratios to the first reading, so it must be positive.
        ratio_based = [signal for signal in ("Capacitance_uF", "ESR_ohm", "Leakage_uA") if signal in own]
        if complete.empty or (complete.iloc[0][ratio_based] <= 0).any():
            unusable.append(str(device))
            continue
        complete = complete.copy()
        complete[[signal for signal in present if signal not in own]] = np.nan
        kept.append(complete)
    shown = lambda names: ", ".join(names[:5]) + (" …" if len(names) > 5 else "")
    if dropped_readings:
        notes.append("Left out readings missing one of their capacitor's parameters (no values were filled in): "
                     + shown([f"{device} ({count})" for device, count in dropped_readings.items()]) + ".")
    if empty:
        notes.append(f"Left out {len(empty)} capacitor(s) with no parameter in at least half of their readings: "
                     f"{shown(empty)}.")
    if unusable:
        notes.append(f"Left out {len(unusable)} capacitor(s) whose first reading has zero capacitance, ESR or "
                     f"leakage current, which cannot serve as a baseline: {shown(unusable)}.")
    show_notes(notes)
    if not kept:
        st.error("No capacitor has enough complete readings to assess.")
        return None
    combined = pd.concat(kept, ignore_index=True)
    recorded = conditions.recorded_columns(combined)  # operating conditions go to the model as well
    return combined.loc[:, ["Cap_ID", "Hours", *present, *recorded]], None, label


@st.cache_data(show_spinner="Assessing capacitors…", max_entries=8)
def _assess_cached(measurements: pd.DataFrame, limits_json: str, planned_json: str,
                   version: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Each capacitor's readings as the model sees them and its assessment.

    Operating conditions a user changed replace the newest reading's conditions, so the predictions
    from that reading on assume them; earlier predictions are unchanged.
    """
    model = _model()
    limits, planned = json.loads(limits_json), json.loads(planned_json)
    results = {}
    for device, readings in measurements.groupby("Cap_ID", sort=True):
        readings = readings.sort_values("Hours").reset_index(drop=True)
        if planned.get(str(device)):
            readings = conditions.with_planned(readings, planned[str(device)])
        results[device] = (readings, model.assess(readings, _measured(readings), limits.get(str(device))))
    return results


@st.cache_data(show_spinner=False, max_entries=16)
def _forecast_cached(readings: pd.DataFrame, remaining_h: float, limits_json: str, version: str) -> pd.DataFrame:
    return _model().forecast(readings, remaining_h, _measured(readings), json.loads(limits_json))


@st.cache_data(show_spinner="Predicting with your conditions…", max_entries=16)
def _changed_history_cached(readings: pd.DataFrame, planned_json: str, limits_json: str,
                            version: str) -> pd.DataFrame:
    """At every reading from the third, the remaining life predicted if the capacitor ran at the planned
    conditions from that reading on (each from the readings available then)."""
    model, planned, limits = _model(), json.loads(planned_json), json.loads(limits_json)
    measured, rows = _measured(readings), []
    for end in range(MIN_READINGS - 1, len(readings)):
        part = conditions.with_planned(readings.iloc[:end + 1].reset_index(drop=True), planned)
        rows.append(model.assess(part, measured, limits).iloc[-1][["Hours", "RUL_median_h", "RUL_low_h", "RUL_high_h"]])
    return pd.DataFrame(rows).reset_index(drop=True)


def _measured(readings: pd.DataFrame) -> tuple[str, ...]:
    """Parameters with a value in every reading (for one capacitor: the ones it was assessed with)."""
    return tuple(signal for signal in SIGNALS if signal in readings and readings[signal].notna().all())


def _assess(measurements: pd.DataFrame, limits: dict, planned: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return _assess_cached(measurements, json.dumps(limits, sort_keys=True), json.dumps(planned, sort_keys=True),
                          code_version())


def _device_limits(devices) -> dict:
    """Each capacitor's end-of-life limits: the defaults until a user changes them."""
    stored = st.session_state.setdefault(LIMITS_STATE, {})
    return {str(device): full_limits(stored.get(str(device))) for device in devices}


def _limit_inputs(device: str, measured: tuple[str, ...]) -> dict:
    """Six limit boxes for the selected capacitor; returns its limits."""
    stored = st.session_state.setdefault(LIMITS_STATE, {})
    current = full_limits(stored.get(device))
    widget = lambda signal: f"caplife_six_limit_{device}_{signal}"

    def changed(signal):
        stored.setdefault(device, {})[LIMIT_KEYS[signal]] = _stored_limit(signal, st.session_state[widget(signal)])

    def reset():
        stored.pop(device, None)
        for signal in SIGNALS:
            st.session_state[widget(signal)] = _shown_limit(signal, EOL_LIMITS[LIMIT_KEYS[signal]])

    st.markdown(f"**End-of-life limits for {device}**",
                help="The failure boundary of each parameter for this capacitor's series, for example from its "
                     "datasheet tolerance. Health, end of life and every prediction on this page use these limits. "
                     "Capacitance, ESR and leakage are measured from the capacitor's first reading.")
    for index, signal in enumerate(SIGNALS):
        if index % 3 == 0:  # one row of three at a time, so a phone stacks them in reading order
            columns = st.columns(3)
        label, step = LIMIT_INPUTS[signal]
        low, high = LIMIT_RANGES[LIMIT_KEYS[signal]]
        kwargs = {} if widget(signal) in st.session_state else {"value": _shown_limit(signal, current[LIMIT_KEYS[signal]])}
        columns[index % 3].number_input(
            label, min_value=_shown_limit(signal, low), max_value=_shown_limit(signal, high), step=step,
            key=widget(signal), persist_state="session", on_change=changed, args=(signal,), disabled=signal not in measured,
            help=("Not measured for this capacitor." if signal not in measured else
                  f"Allowed: {_shown_limit(signal, low):g} to {_shown_limit(signal, high):g}, the range the model "
                  "was trained on."), **kwargs)
    st.button("Reset to defaults", on_click=reset, key=f"caplife_six_reset_{device}")
    return full_limits(stored.get(device))


def _fleet_table(results: dict, truth: dict | None, limits: dict | None = None) -> pd.DataFrame:
    rows = []
    for device, (_, assessment) in results.items():
        last = assessment.iloc[-1]
        has_rul = "RUL_median_h" in assessment and pd.notna(last.get("RUL_median_h"))
        row = {
            "Capacitor": device,
            "Parameters": f"{len(_measured(results[device][0]))} of 6",
            "Latest reading (h)": last["Hours"],
            "Overall health (%)": round(last["Overall_health_pct"]),
            "Limiting indicator": last["Limiting_indicator"],
            "Status": _status(last["Overall_health_pct"], bool(last["End_of_life_reached"])),
            "Remaining life, median (h)": last["RUL_median_h"] if has_rul else np.nan,
            "Remaining life, 80% range (h)": (f"{last['RUL_low_h']:,.0f} – {last['RUL_high_h']:,.0f}"
                                              if has_rul else f"Needs {MIN_READINGS} readings"),
            "Predicted end-of-life age (h)": last["Hours"] + last["RUL_median_h"] if has_rul else np.nan,
        }
        row.update(conditions.summary_columns(results[device][0]))
        if truth:
            # The simulated truth uses the default limits; it does not apply once a user changes them.
            default = limits is None or limits.get(str(device)) == full_limits()
            row["Simulated true end of life (h)"] = truth[device]["eol_h"] if default else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    # Condition columns only when some capacitor has them.
    return frame.drop(columns=[column for column in ("Temperature (°C)", "Voltage / rated", "Ripple / rated")
                               if frame[column].isna().all()])


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


def _health_cards(readings: pd.DataFrame, assessment: pd.DataFrame, limits: dict | None = None) -> None:
    baseline, latest, last = readings.iloc[0], readings.iloc[-1], assessment.iloc[-1]
    cards = []
    for signal in _measured(readings):
        health = last[f"{signal}_health_pct"]
        status = _status(health, health <= 0)
        status_key = status.lower().replace(" ", "-")
        value = latest[signal]
        if signal == "Pressure_Abs_kPa":
            change = f"{value - ATMOSPHERE_KPA:+,.1f} kPa above atmospheric"
        elif signal in ("Case_Expansion_um", "H2_Generated_mL_STP"):
            change = f"{value - baseline[signal]:+,.{DIGITS[signal]}f} {UNITS[signal]} since baseline"
        else:
            change = f"{100 * (value / baseline[signal] - 1):+.1f}% since baseline"
        # Health follows the worst level held so far; say so when the latest reading is better.
        held_used = float(limit_used(readings, limits)[signal].iloc[-1])
        if held_used > float(current_limit_used(readings, limits)[signal].iloc[-1]) + 0.05:
            worst = _value_at(signal, baseline, held_used, limits)
            change += f" · worst held {worst:,.{DIGITS[signal]}f} {UNITS[signal]}"
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
            f'<span>Limit: {escape(_limit_text(signal, limits))}</span></div></div>'
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


def signal_charts(readings: pd.DataFrame, limits: dict | None = None, reference: bool = False,
                  alternative: pd.DataFrame | None = None) -> None:
    """Each parameter's readings joined in time order, two charts per row, with its end-of-life limit.

    Parameters with fewer than two readings are skipped, and blank readings are left out of the
    line. ``reference`` labels the limit as the default reference limit (Analyze measurements).
    ``alternative`` is the estimated history had the capacitor always run at other conditions,
    drawn gold and dashed next to the measured readings.
    """
    readings = readings.sort_values("Hours")
    baseline = readings.bfill().iloc[0]  # each parameter's first available reading
    signals = [signal for signal in SIGNALS if signal in readings and readings[signal].notna().sum() >= 2]
    columns = st.columns(2)
    for index, signal in enumerate(signals):
        data = readings[["Hours", signal]].dropna().rename(columns={signal: "value"}).reset_index(drop=True)
        other = None
        if alternative is not None and signal in alternative:
            other = alternative[["Hours", signal]].dropna().rename(columns={signal: "value"})
        limit = _limit_value(signal, baseline, limits)
        bottom, top, off_chart = _chart_domain(pd.concat([data["value"], *([other["value"]] if other is not None
                                                                           else [])]), limit, DIGITS[signal])
        scale = alt.Scale(domain=[bottom, top], zero=False, nice=False)
        measured_tip = [alt.Tooltip("Hours:Q", format=",.0f"),
                        alt.Tooltip("value:Q", title=INDICATOR_LABELS[signal], format=HOVER_FORMAT[signal])]
        line = alt.Chart(data).mark_line(color=TEAL, strokeWidth=1.8, point=alt.OverlayMarkDef(
            color=TEAL, size=30, filled=True)).encode(
            x=alt.X("Hours:Q", title="Hours"), y=alt.Y("value:Q", title=UNITS[signal], scale=scale),
            tooltip=measured_tip)
        if off_chart is None:
            marker = alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=ORANGE, strokeDash=[5, 4]).encode(
                y=alt.Y("limit:Q", scale=alt.Scale(domain=[bottom, top], nice=False)),
                tooltip=[alt.Tooltip("limit:Q", title="Limit", format=HOVER_FORMAT[signal])])
        else:
            # The limit is far outside the readings: say where it is instead of flattening the trend.
            label = f"Limit {limit:,.{DIGITS[signal]}f} {UNITS[signal]} {'↑' if off_chart == 'above' else '↓'}"
            # Placed on the left: rising curves start low there and falling ones start high,
            # so the note never covers the latest readings at the right.
            marker = alt.Chart(pd.DataFrame({"Hours": [float(data["Hours"].min())],
                                             "value": [top if off_chart == "above" else bottom], "label": [label]})
                               ).mark_text(align="left", dx=4, baseline="top" if off_chart == "above" else "bottom",
                                           dy=4 if off_chart == "above" else -4, color=ORANGE, fontSize=11).encode(
                x="Hours:Q", y=alt.Y("value:Q", scale=alt.Scale(domain=[bottom, top], nice=False)), text="label:N",
                tooltip=NO_TOOLTIP)
        with columns[index % 2]:
            kind = "default reference limit" if reference else "limit"
            st.caption(f"**{INDICATOR_LABELS[signal]}** · {kind}: {_limit_text(signal, limits)}")
            layers = [line, marker]
            hovers = _hoverable(data, "Hours:Q", alt.Y("value:Q", scale=scale), measured_tip, TEAL)
            if other is not None:
                other_tip = [alt.Tooltip("Hours:Q", format=",.0f"),
                             alt.Tooltip("value:Q", title="Had it always run at your conditions",
                                         format=HOVER_FORMAT[signal])]
                layers.append(alt.Chart(other).mark_line(
                    color=GOLD, strokeWidth=2.2, strokeDash=[6, 3],
                    point=alt.OverlayMarkDef(color=GOLD, size=22, filled=True)).encode(
                    x="Hours:Q", y=alt.Y("value:Q", scale=scale), tooltip=other_tip))
                hovers += _hoverable(other, "Hours:Q", alt.Y("value:Q", scale=scale), other_tip, GOLD)
            st.altair_chart(alt.layer(*layers, *hovers).properties(height=260), width="stretch")


_signal_charts = signal_charts


def _forecast_charts(readings: pd.DataFrame, forecast: pd.DataFrame, last: pd.Series,
                     limits: dict | None = None, zoom: bool = False,
                     reference: tuple[pd.DataFrame, pd.Series] | None = None) -> None:
    """Each measured parameter from the first reading to the predicted end of life: the measured
    readings (solid) followed by the forecast path from the latest reading (dashed, with its range).

    ``reference`` is the forecast and assessment with the recorded operating conditions, drawn in grey
    on the same time axis when the user changed the conditions, so the effect on each parameter shows.
    """
    baseline = readings.iloc[0]
    first, latest = float(readings["Hours"].min()), float(last["Hours"])
    end = latest + float(last["RUL_median_h"])
    eol_range = pd.DataFrame({"from": [latest + float(last["RUL_low_h"])], "to": [latest + float(last["RUL_high_h"])]})
    right = latest + float(last["RUL_high_h"])
    if reference is not None:
        right = max(right, latest + float(reference[1]["RUL_high_h"]))
    # Zoomed in, the chart shows the latest readings (a fifth of the width) and the forecast.
    left = max(first, latest - 0.25 * (right - latest)) if zoom else first
    hours_scale = alt.Scale(domain=[left, right], nice=False)
    x = alt.X("Hours:Q", title="Hours", scale=hours_scale)
    columns = st.columns(2)
    for index, signal in enumerate(_measured(readings)):
        data = forecast.loc[forecast["signal"].eq(signal)].reset_index(drop=True)
        measured = readings[["Hours", signal]].rename(columns={signal: "value"}).reset_index(drop=True)
        if zoom:  # keep one reading before the window so the line enters from the edge
            inside = measured["Hours"] >= left
            measured = measured.loc[inside | inside.shift(-1, fill_value=False)]
        limit = _limit_value(signal, baseline, limits)
        values = pd.concat([measured.loc[measured["Hours"] >= left, "value"], data["low"], data["high"]]
                           + ([reference[0].loc[reference[0]["signal"].eq(signal), "median"]] if reference is not None
                              else []))
        bottom, top, off_chart = _chart_domain(values, limit, DIGITS[signal])
        if reference is not None:  # room along the bottom for the two end-of-life labels
            bottom -= 0.12 * (top - bottom)
        scale = alt.Scale(domain=[bottom, top], zero=False, nice=False)
        number = HOVER_FORMAT[signal]
        # Purple is the forecast with the recorded conditions; with changed conditions, theirs is gold.
        mine = GOLD if reference is not None else PURPLE
        eol_color = GOLD if reference is not None else GREY
        window = alt.Chart(eol_range).mark_rect(color=eol_color, opacity=.14 if reference is not None else .16).encode(
            x=alt.X("from:Q", scale=hours_scale), x2="to:Q",
            tooltip=[alt.Tooltip("from:Q", title="End of life, 80% range from (h)", format=",.0f"),
                     alt.Tooltip("to:Q", title="to (h)", format=",.0f")])
        now = alt.Chart(pd.DataFrame({"Hours": [latest]})).mark_rule(color=DIVIDER, strokeWidth=3).encode(
            x=alt.X("Hours:Q", scale=hours_scale),
            tooltip=[alt.Tooltip("Hours:Q", title="Latest reading (h)", format=",.0f")])
        now_label = alt.Chart(pd.DataFrame({"Hours": [latest], "value": [top], "label": ["Latest reading"]})).mark_text(
            align="right", baseline="top", dx=-5, dy=4, color=DIVIDER, fontSize=11, fontWeight="bold").encode(
            x=alt.X("Hours:Q", scale=hours_scale), y=alt.Y("value:Q", scale=scale), text="label:N",
            tooltip=NO_TOOLTIP)
        measured_tip = [alt.Tooltip("Hours:Q", title="Age (h)", format=",.0f"),
                        alt.Tooltip("value:Q", title=f"Measured {INDICATOR_LABELS[signal]}", format=number)]
        history = alt.Chart(measured).mark_line(color=TEAL, strokeWidth=1.8, clip=True, point=alt.OverlayMarkDef(
            color=TEAL, size=26, filled=True)).encode(
            x=x, y=alt.Y("value:Q", title=UNITS[signal], scale=scale), tooltip=measured_tip)
        y_value, y_median = alt.Y("value:Q", scale=scale), alt.Y("median:Q", scale=scale)
        ahead_tip = [alt.Tooltip("Hours:Q", title="Age (h)", format=",.0f"),
                     alt.Tooltip("median:Q", title=f"Predicted {INDICATOR_LABELS[signal]}", format=number),
                     alt.Tooltip("low:Q", title="80% range from", format=number),
                     alt.Tooltip("high:Q", title="80% range to", format=number)]
        band = alt.Chart(data).mark_area(color=mine, opacity=.28).encode(
            x=x, y=alt.Y("low:Q", scale=scale), y2="high:Q", tooltip=NO_TOOLTIP)
        age = alt.Tooltip("Hours:Q", title="Age (h)", format=",.0f")
        predicted = alt.Tooltip("median:Q", title=f"Predicted {INDICATOR_LABELS[signal]}", format=number)
        start, ahead, final = data.iloc[[0]], data.iloc[1:], data.iloc[[-1]]
        edges = pd.concat([final.assign(value=final["high"], edge="Top of the 80% range at end of life"),
                           final.assign(value=final["low"], edge="Bottom of the 80% range at end of life")])
        edge_tip = [alt.Tooltip("edge:N", title="Point"), age,
                    alt.Tooltip("value:Q", title=INDICATOR_LABELS[signal], format=number)]
        path = alt.layer(
            alt.Chart(data).mark_line(color=mine, strokeWidth=3, strokeDash=[7, 3]).encode(
                x=x, y=alt.Y("median:Q", scale=scale), tooltip=NO_TOOLTIP),
            # The first point is the latest reading itself: its value only, no range.
            alt.Chart(start).mark_point(color=mine, size=34, filled=True, opacity=1).encode(
                x=x, y=alt.Y("median:Q", scale=scale), tooltip=[age, predicted]),
            alt.Chart(ahead).mark_point(color=mine, size=34, filled=True, opacity=1).encode(
                x=x, y=alt.Y("median:Q", scale=scale), tooltip=ahead_tip),
            # The ends of the range on the end-of-life line, with their values.
            alt.Chart(edges).mark_point(color=mine, size=30, filled=False, strokeWidth=2, opacity=1).encode(
                x=x, y=alt.Y("value:Q", scale=scale), tooltip=edge_tip),
        )
        hovers = [*_hoverable(measured.loc[measured["Hours"] >= left], x, y_value, measured_tip, TEAL),
                  *_hoverable(start, x, y_median, [age, predicted], mine),
                  *_hoverable(ahead, x, y_median, ahead_tip, mine),
                  *_hoverable(edges, x, y_value, edge_tip, mine)]
        x_rule = alt.X("Hours:Q", scale=hours_scale)
        eol_tip = [alt.Tooltip("Hours:Q", title="EOL for Changed Conditions (h)" if reference is not None
                               else "Predicted end of life (h)", format=",.0f")]
        latest_tip = [alt.Tooltip("Hours:Q", title="Latest reading (h)", format=",.0f")]
        eol = alt.Chart(pd.DataFrame({"Hours": [end]})).mark_rule(color=eol_color, strokeWidth=2.5).encode(
            x=x_rule, tooltip=eol_tip)
        hovers += [*_hoverable_rule(pd.DataFrame({"Hours": [latest]}), x_rule, latest_tip, DIVIDER),
                   *_hoverable_rule(pd.DataFrame({"Hours": [end]}), x_rule, eol_tip, eol_color)]
        layers = [window, now, now_label, band, history, path, eol]
        if reference is not None:
            before = reference[0].loc[reference[0]["signal"].eq(signal)]
            recorded_tip = [alt.Tooltip("Hours:Q", title="Age (h)", format=",.0f"),
                            alt.Tooltip("median:Q", title=f"{INDICATOR_LABELS[signal]}, recorded conditions",
                                        format=number)]
            hovers += _hoverable(before.iloc[1:], x, y_median, recorded_tip, PURPLE)
            recorded_eol_tip = [alt.Tooltip("Hours:Q", title="EOL for Recorded Conditions (h)", format=",.0f")]
            hovers += _hoverable_rule(pd.DataFrame({"Hours": [latest + float(reference[1]["RUL_median_h"])]}),
                                      x_rule, recorded_eol_tip, PURPLE)
            recorded = reference[1]
            recorded_end = latest + float(recorded["RUL_median_h"])
            height = top - bottom
            ranges = {"Changed": (latest + float(last["RUL_low_h"]), end, latest + float(last["RUL_high_h"])),
                      "Recorded": (latest + float(recorded["RUL_low_h"]), recorded_end,
                                   latest + float(recorded["RUL_high_h"]))}
            layers = [
                # Each end of life's 80% range shaded in its colour (they blend where they overlap).
                *(alt.Chart(pd.DataFrame({"from": [low], "to": [high]})).mark_rect(
                    color=GOLD if name == "Changed" else PURPLE, opacity=.17 if name == "Changed" else .2).encode(
                    x=alt.X("from:Q", scale=hours_scale), x2="to:Q", tooltip=NO_TOOLTIP)
                  for name, (low, _, high) in ranges.items()),
                now, now_label, band, history,
                *_range_hover(ranges, hours_scale),  # above the band, below the lines and their points
                path, eol,
                # Points make the thin dotted line easy to hover.
                alt.Chart(before).mark_line(color=PURPLE, strokeWidth=2.2, strokeDash=[2, 3], clip=True,
                                            point=alt.OverlayMarkDef(color=PURPLE, size=26, filled=True)).encode(
                    x=x, y=alt.Y("median:Q", scale=scale), tooltip=recorded_tip),
                alt.Chart(pd.DataFrame({"Hours": [recorded_end]})).mark_rule(
                    color=PURPLE, strokeWidth=2.5, strokeDash=[4, 3]).encode(x=x_rule, tooltip=recorded_eol_tip),
                # Labels at the median lines, each on the outer side of its line and at its own height.
                *(alt.Chart(pd.DataFrame({"Hours": [when], "value": [bottom + lift * height], "label": [name]}))
                  .mark_text(align="left" if outer else "right", dx=4 if outer else -4, baseline="bottom",
                             fontSize=11, fontWeight="bold", color=colour)
                  .encode(x=alt.X("Hours:Q", scale=hours_scale), y=alt.Y("value:Q", scale=scale), text="label:N",
                          tooltip=NO_TOOLTIP)
                  for when, lift, name, colour, outer in (
                      (end, 0.02, "Changed", GOLD, end >= recorded_end),
                      (recorded_end, 0.08, "Recorded", PURPLE, recorded_end > end))),
            ]
        if off_chart is None:
            layers.append(alt.Chart(pd.DataFrame({"limit": [limit]})).mark_rule(color=ORANGE, strokeDash=[5, 4])
                          .encode(y=alt.Y("limit:Q", scale=scale),
                                  tooltip=[alt.Tooltip("limit:Q", title="Your limit", format=number)]))
        else:
            label = f"Limit {limit:{number}} {UNITS[signal]} {'↑' if off_chart == 'above' else '↓'}"
            # At the right edge: the left one holds the "Latest reading" label when zoomed in.
            layers.append(alt.Chart(pd.DataFrame({"Hours": [right], "value": [top if off_chart == "above" else bottom],
                                                  "label": [label]}))
                          .mark_text(align="right", dx=-4, baseline="top" if off_chart == "above" else "bottom",
                                     dy=4 if off_chart == "above" else -4, color=ORANGE, fontSize=11)
                          .encode(x=alt.X("Hours:Q", scale=hours_scale), y=alt.Y("value:Q", scale=scale),
                                  text="label:N", tooltip=NO_TOOLTIP))
        with columns[index % 2]:
            st.caption(f"**{INDICATOR_LABELS[signal]}** · limit: {_limit_text(signal, limits)}")
            st.altair_chart(alt.layer(*layers, *hovers).properties(height=260), width="stretch")


def _range_hover(ranges: dict, hours_scale: alt.Scale) -> list:
    """Invisible hover areas over two end-of-life ranges: where only one covers the time, its details;
    where both do, both. ``ranges`` maps "Changed"/"Recorded" to (from, median, to) in hours."""
    names = {"Changed": "EOL for Changed Conditions", "Recorded": "EOL for Recorded Conditions"}
    (c_low, c_mid, c_high), (r_low, r_mid, r_high) = ranges["Changed"], ranges["Recorded"]
    edges = sorted({c_low, c_high, r_low, r_high})
    single, both = [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        middle = (start + stop) / 2
        inside = [name for name, (low, _, high) in ranges.items() if low <= middle <= high]
        if len(inside) == 2:
            both.append({"from": start, "to": stop})
        elif inside:
            low, mid, high = ranges[inside[0]]
            single.append({"from": start, "to": stop, "which": names[inside[0]], "median": mid,
                           "range": f"{low:,.0f} – {high:,.0f}"})
    hour = lambda field, title: alt.Tooltip(f"{field}:Q", title=title, format=",.0f")
    layers = []
    if single:
        layers.append(alt.Chart(pd.DataFrame(single)).mark_rect(opacity=0.001).encode(
            x=alt.X("from:Q", scale=hours_scale), x2="to:Q",
            tooltip=[alt.Tooltip("which:N", title="Point"), hour("median", "Predicted EOL (h)"),
                     alt.Tooltip("range:N", title="80% range (h)")]))
    if both:
        frame = pd.DataFrame(both).assign(
            note="Both EOL 80% ranges cover this time", c_mid=c_mid, c_range=f"{c_low:,.0f} – {c_high:,.0f}",
            r_mid=r_mid, r_range=f"{r_low:,.0f} – {r_high:,.0f}")
        layers.append(alt.Chart(frame).mark_rect(opacity=0.001).encode(
            x=alt.X("from:Q", scale=hours_scale), x2="to:Q",
            tooltip=[alt.Tooltip("note:N", title="Point"),
                     hour("c_mid", "EOL for Changed Conditions (h)"), alt.Tooltip("c_range:N", title="80% range (h)"),
                     hour("r_mid", "EOL for Recorded Conditions (h)"),
                     alt.Tooltip("r_range:N", title="80% range (h) ")]))
    return layers


def _first_to_limit(forecast: pd.DataFrame) -> str:
    """The parameter the forecast expects to reach its limit first (the highest share of its limit at the end)."""
    final = forecast.loc[forecast["Hours"].eq(forecast["Hours"].max())]
    return INDICATOR_LABELS[final.loc[final["fraction_median"].idxmax(), "signal"]]


def _end_of_life_message(assessment: pd.DataFrame) -> str:
    """Say when, and on which indicator, a limit was first met, not only at the latest reading."""
    last = assessment.iloc[-1]
    first = assessment.loc[assessment["End_of_life_reached"].astype(bool)].iloc[0]
    latest = f"{last['Hours']:,.0f} h ({last['Limiting_indicator']})"
    if float(first["Hours"]) < float(last["Hours"]):
        return (f"End-of-life limit first met at {first['Hours']:,.0f} h ({first['Limiting_indicator']}) "
                f"and still met at the latest reading, {latest}.")
    return f"End-of-life limit met at the latest reading, {latest}."


def _always_at_end_of_life(assessment: pd.DataFrame) -> bool:
    """True when every remaining-life prediction is 0 h, so a trend chart would be a flat line."""
    predicted = assessment.dropna(subset=["RUL_median_h"])
    return bool(len(predicted)) and bool((predicted["RUL_high_h"] == 0).all())


def _countdown(now: pd.Series) -> pd.DataFrame:
    """From the latest reading, remaining life falls one hour per hour: median and 80% range until zero."""
    latest, mid, low, high = (float(now[k]) for k in ("Hours", "RUL_median_h", "RUL_low_h", "RUL_high_h"))
    hours = np.linspace(latest, latest + high, 40)
    gone = hours - latest
    return pd.DataFrame({"Hours": hours, "median": np.clip(mid - gone, 0, None), "low": np.clip(low - gone, 0, None),
                         "high": np.clip(high - gone, 0, None)}).loc[lambda f: f["Hours"] <= latest + high]


def _with_range(frame: pd.DataFrame, low: str = "RUL_low_h", high: str = "RUL_high_h") -> pd.DataFrame:
    """Adds a readable "range" column ("1,416 – 2,530") for tooltips."""
    return frame.assign(range=[f"{max(a, 0):,.0f} – {max(b, 0):,.0f}" for a, b in zip(frame[low], frame[high])])


def _rul_chart(assessment: pd.DataFrame, true_eol: float | None, changed_now: pd.Series | None = None,
               changed_history: pd.DataFrame | None = None) -> alt.Chart:
    """Remaining life predicted at every reading (purple, from the readings available then), and from the
    latest reading a countdown to end of life with its 80% range and end-of-life range, like the parameter
    forecasts. ``assessment`` uses the recorded conditions; with changed ones, ``changed_history`` is the
    prediction at each reading had the capacitor run at them from then on, and ``changed_now`` the one at
    the latest reading (gold line, countdown and end of life)."""
    data = assessment.dropna(subset=["RUL_median_h"])
    recorded_now = data.iloc[-1]
    latest = float(recorded_now["Hours"])
    nows = {"Recorded": recorded_now, **({"Changed": changed_now} if changed_now is not None else {})}
    right = max(latest + float(now["RUL_high_h"]) for now in nows.values())
    if true_eol is not None:
        right = max(right, true_eol)
    first = float(data["Hours"].min())
    hours_scale = alt.Scale(domain=[first, right * 1.02], nice=False)
    highs = [float(data["RUL_high_h"].max()), *(float(now["RUL_high_h"]) for now in nows.values())]
    if changed_history is not None and changed_history["RUL_high_h"].notna().any():
        highs.append(float(changed_history["RUL_high_h"].max()))
    top = max(highs) * 1.08
    # A little room below zero, so lines ending at zero are not drawn on the chart's edge.
    scale = alt.Scale(domain=[-0.05 * top, top], nice=False)
    x = alt.X("Hours:Q", title="Capacitor age (h)", scale=hours_scale)
    y = alt.Y("RUL_median_h:Q", title="Remaining useful life (h)", scale=scale,
              axis=alt.Axis(labelExpr="datum.value < 0 ? '' : format(datum.value, ',.0f')"))
    hour = lambda field, title: alt.Tooltip(f"{field}:Q", title=title, format=",.0f")
    changed = changed_now is not None
    colours = {"Recorded": PURPLE, "Changed": GOLD}

    layers, hovers = [], []
    ranges = {name: (latest + float(now["RUL_low_h"]), latest + float(now["RUL_median_h"]),
                     latest + float(now["RUL_high_h"])) for name, now in nows.items()}
    # End-of-life 80% ranges, shaded (grey when there is only the one prediction).
    for name, (low, _, high) in ranges.items():
        layers.append(alt.Chart(pd.DataFrame({"from": [low], "to": [high]})).mark_rect(
            color=colours[name] if changed else GREY, opacity=.18 if changed else .16).encode(
            x=alt.X("from:Q", scale=hours_scale), x2="to:Q",
            tooltip=NO_TOOLTIP if changed else [hour("from", "End of life, 80% range from (h)"),
                                                hour("to", "to (h)")]))
    if changed:
        layers += _range_hover(ranges, hours_scale)
    # Predictions made at past readings.
    data = _with_range(data)
    # Every tooltip on this chart: what the point is, then age, remaining life and its 80% range.
    past_tip = [alt.Tooltip("point:N", title="Point"), hour("Hours", "Age (h)"),
                hour("RUL_median_h", "Remaining life (h)"), alt.Tooltip("range:N", title="80% range (h)")]
    data = data.assign(point="Prediction at a reading · Recorded Conditions")
    layers += [
        alt.Chart(data).mark_area(color=PURPLE, opacity=.25).encode(
            x=x, y=alt.Y("RUL_low_h:Q", scale=scale), y2="RUL_high_h:Q", tooltip=NO_TOOLTIP),
        alt.Chart(data).mark_line(color=PURPLE, strokeWidth=2.2, point=alt.OverlayMarkDef(
            color=PURPLE, size=30, filled=True)).encode(x=x, y=y, tooltip=past_tip),
        alt.Chart(pd.DataFrame({"Hours": [latest]})).mark_rule(color=DIVIDER, strokeWidth=3).encode(
            x=alt.X("Hours:Q", scale=hours_scale), tooltip=[hour("Hours", "Latest reading (h)")]),
        alt.Chart(pd.DataFrame({"Hours": [latest], "value": [top], "label": ["Latest reading"]})).mark_text(
            align="right", baseline="top", dx=-5, dy=4, color=DIVIDER, fontSize=11, fontWeight="bold").encode(
            x=alt.X("Hours:Q", scale=hours_scale), y=alt.Y("value:Q", scale=scale), text="label:N",
            tooltip=NO_TOOLTIP),
    ]
    hovers += _hoverable(data, x, y, past_tip, PURPLE)
    hovers += _hoverable_rule(pd.DataFrame({"Hours": [latest]}), alt.X("Hours:Q", scale=hours_scale),
                              [hour("Hours", "Latest reading (h)")], DIVIDER)
    if changed_history is not None and len(changed_history):
        gold = _with_range(changed_history.dropna(subset=["RUL_median_h"])).assign(
            point="Prediction at a reading · Changed Conditions from then on")
        gold_tip = past_tip
        layers += [
            alt.Chart(gold).mark_area(color=GOLD, opacity=.2).encode(
                x=x, y=alt.Y("RUL_low_h:Q", scale=scale), y2="RUL_high_h:Q", tooltip=NO_TOOLTIP),
            alt.Chart(gold).mark_line(color=GOLD, strokeWidth=2.4, strokeDash=[6, 3], point=alt.OverlayMarkDef(
                color=GOLD, size=30, filled=True)).encode(x=x, y=y, tooltip=gold_tip),
        ]
        hovers += _hoverable(gold, x, y, gold_tip, GOLD)
    # Countdowns from the latest reading to end of life.
    for name, now in nows.items():
        colour = colours[name]
        path = _countdown(now)
        recorded_line = name == "Recorded" and changed
        layers += [
            alt.Chart(path).mark_area(color=colour, opacity=.16 if recorded_line else .24).encode(
                x=x, y=alt.Y("low:Q", scale=scale), y2="high:Q", tooltip=NO_TOOLTIP),
            # The median line stops at its end of life instead of running along zero.
            alt.Chart(path.loc[path["Hours"] <= latest + float(now["RUL_median_h"])]
                      .pipe(lambda f: pd.concat([f, pd.DataFrame([{"Hours": latest + float(now["RUL_median_h"]),
                                                                   "median": 0.0}])])))
            .mark_line(color=colour, strokeWidth=2.2 if recorded_line else 3,
                                      strokeDash=[2, 3] if recorded_line else [7, 3]).encode(
                x=x, y=alt.Y("median:Q", scale=scale), tooltip=NO_TOOLTIP),
        ]
        end = ranges[name][1]
        layers.append(alt.Chart(pd.DataFrame({"Hours": [end]})).mark_rule(
            color=colour if changed else GREY, strokeWidth=2.5, strokeDash=[4, 3] if recorded_line else [1, 0]).encode(
            x=alt.X("Hours:Q", scale=hours_scale),
            tooltip=[hour("Hours", f"EOL for {name} Conditions (h)")]))
        hovers += _hoverable_rule(pd.DataFrame({"Hours": [end]}), alt.X("Hours:Q", scale=hours_scale),
                                  [hour("Hours", f"EOL for {name} Conditions (h)")], colour if changed else GREY)
        # Points along the countdown, at 0%, 10%, ..., 100% of the predicted remaining life.
        mid, low, high = (float(now[k]) for k in ("RUL_median_h", "RUL_low_h", "RUL_high_h"))
        steps = mid * np.linspace(0, 1, 11)
        points = _with_range(pd.DataFrame({"Hours": latest + steps, "median": mid - steps, "low": low - steps,
                                           "high": high - steps}), "low", "high")
        conditions_name = f"{name} Conditions"
        points["point"] = [f"Latest reading · {conditions_name}"] + [f"Countdown · {conditions_name}"] * 10
        point_tip = [alt.Tooltip("point:N", title="Point"), hour("Hours", "Age (h)"),
                     hour("median", "Remaining life (h)"), alt.Tooltip("range:N", title="80% range (h)")]
        # The last point is the predicted end of life: its hour and the end of life's own 80% range.
        eol_point = pd.DataFrame([{"Hours": ranges[name][1], "median": 0.0, "point": f"EOL for {conditions_name}",
                                   "range": f"{ranges[name][0]:,.0f} – {ranges[name][2]:,.0f}"}])
        eol_tip = [alt.Tooltip("point:N", title="Point"), hour("Hours", "Predicted EOL (h)"),
                   alt.Tooltip("range:N", title="80% range (h)")]
        points = points.iloc[:-1]
        y_median = alt.Y("median:Q", scale=scale)
        layers += [alt.Chart(points).mark_point(color=colour, size=34, filled=True, opacity=1).encode(
                       x=x, y=y_median, tooltip=point_tip),
                   alt.Chart(eol_point).mark_point(color=colour, size=60, filled=True, opacity=1).encode(
                       x=x, y=y_median, tooltip=eol_tip)]
        hovers += _hoverable(points, x, y_median, point_tip, colour)
        hovers += _hoverable(eol_point, x, y_median, eol_tip, colour)
    if changed:  # labels near the top of the end-of-life lines, on their outer sides and at two heights
        recorded_end, changed_end = ranges["Recorded"][1], ranges["Changed"][1]
        for when, lift, name, outer in ((changed_end, 0.99, "Changed", changed_end >= recorded_end),
                                        (recorded_end, 0.92, "Recorded", recorded_end > changed_end)):
            layers.append(alt.Chart(pd.DataFrame({"Hours": [when], "value": [lift * top], "label": [name]})).mark_text(
                align="left" if outer else "right", dx=4 if outer else -4, baseline="top", fontSize=11,
                fontWeight="bold", color=colours[name]).encode(
                x=alt.X("Hours:Q", scale=hours_scale), y=alt.Y("value:Q", scale=scale), text="label:N",
                tooltip=NO_TOOLTIP))
    if true_eol is not None:
        truth = pd.DataFrame({"Hours": np.linspace(first, true_eol, 60)})
        truth["true"] = np.maximum(true_eol - truth["Hours"], 0)
        tip = [hour("Hours", "Age (h)"), hour("true", "True remaining life (h)")]
        layers.append(alt.Chart(truth).mark_line(color=ORANGE, strokeDash=[6, 4], strokeWidth=2).encode(
            x=x, y=alt.Y("true:Q", scale=scale), tooltip=tip))
        hovers += _hoverable(truth.iloc[::4], x, alt.Y("true:Q", scale=scale), tip, ORANGE)
    return alt.layer(*layers, *hovers).properties(height=320)


# Labels for use mid-sentence: lower case except abbreviations (ESR, H₂).
INLINE_LABELS = {"Capacitance_uF": "capacitance", "ESR_ohm": "ESR", "Leakage_uA": "leakage current",
                 "Case_Expansion_um": "case expansion", "H2_Generated_mL_STP": "H₂ generated",
                 "Pressure_Abs_kPa": "internal pressure"}


def _names(signals) -> str:
    """'capacitance, ESR and leakage current' style list for use inside a sentence."""
    labels = [INLINE_LABELS[signal] for signal in signals]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _model_card(measured: tuple[str, ...]) -> None:
    meta = _model().metadata
    key = subset_key(measured)
    neural = meta["test_metrics_by_subset"][key]
    trend = meta["trend_baseline_by_subset"].get(key)
    full = meta["test_metrics_by_subset"][subset_key(SIGNALS)]
    fleets = meta["fleets"]
    st.markdown(
        f"- **Network:** a GRU (recurrent neural network) reads up to the last {meta['window']} readings. Each "
        "indicator enters as the fraction of its limit already used, with a flag saying whether it was measured "
        "and the limit it is measured against, plus the capacitor's age and the time between readings, and the "
        "operating conditions recorded with each reading: core and ambient temperature, applied over rated "
        "voltage, ripple over rated ripple current and operating frequency (the latest ones are taken to "
        "continue). During training, indicators and conditions were hidden at random, so the same network "
        "predicts from any combination of them. It outputs the 10th, 50th and 90th percentiles of remaining "
        "life.\n"
        f"- **Training data:** {fleets['train']['capacitors']:,} simulated capacitors "
        f"({fleets['train']['predictions']:,} prediction points) from a physics-inspired simulator: "
        "electrolyte loss, which speeds up about 2× for every 10 °C of core temperature and with the cube of "
        "applied over rated voltage, lowers capacitance and raises ESR; ripple current heats the core (less at "
        "higher operating frequencies, where ESR is lower), and 40% of the capacitors change operating conditions "
        "part-way through life; leakage current generates "
        "hydrogen by Faraday's law; retained hydrogen raises internal pressure and bulges the case. "
        "Coefficients are illustrative, not fitted to a real capacitor series.\n"
        "- **End of life:** the first parameter to reach the limit you set. Defaults: capacitance loss "
        f"{EOL_LIMITS['capacitance_loss_pct']:.0f}%, ESR {EOL_LIMITS['esr_ratio']:g}× initial, leakage "
        f"{EOL_LIMITS['leakage_ratio']:g}× initial, case expansion {EOL_LIMITS['expansion_um']:g} µm, H₂ generated "
        f"{EOL_LIMITS['h2_ml']:g} mL, internal pressure {EOL_LIMITS['pressure_abs_kpa']:g} kPa (vent). The networks "
        "were trained on simulated capacitors with limits drawn at random from the ranges the limit boxes allow "
        "(30% kept the defaults), so they predict the time to the limits you set. The default expansion and H₂ "
        "limits are the levels at "
        "which a typical simulated capacitor reaches the vent pressure.\n"
        f"- **80% range:** calibrated on {fleets['calibration']['capacitors']} further simulated capacitors "
        "(conformalized quantile regression), separately for every combination of measured parameters, so it "
        "widens when fewer parameters are available.\n"
        "- **Parameter trends:** a second GRU reads the same inputs and forecasts each parameter at 10%, 20%, ..., "
        "100% of the predicted remaining life, as a median path that never improves (aging damage does not heal) "
        "and an 80% range calibrated on the same capacitors. It learned from the simulated capacitors' true "
        "paths to end of life."
    )
    st.markdown(f"**Test on {fleets['test']['capacitors']} simulated capacitors the model never saw**, with the "
                f"default limits ({neural['predictions']:,} predictions), using {_names(measured)}")

    def row(method, metrics):
        return {"Method": method, "Typical error (h)": metrics["median_absolute_error_h"],
                "Typical relative error": metrics.get("median_relative_error"),
                "True RUL inside 80% range": metrics.get("interval_coverage"),
                "Typical 80% range width (h)": metrics.get("median_interval_width_h")}

    rows = [row("Neural network (GRU), these parameters, with operating conditions", neural)]
    bare = meta.get("test_metrics_by_subset_without_conditions", {}).get(key)
    if bare:
        rows.append(row("Neural network (GRU), these parameters, without operating conditions", bare))
    custom = meta.get("test_metrics_by_subset_custom_limits", {}).get(key)
    if custom:
        rows.append(row("Neural network (GRU), these parameters, random limits", custom))
    if key != subset_key(SIGNALS):
        rows.append(row("Neural network (GRU), all six parameters", full))
    if trend:
        rows.append(row("Trend line to nearest limit, these parameters", trend))
    table(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Typical error (h)": st.column_config.NumberColumn(format="%.0f"),
        "Typical relative error": st.column_config.NumberColumn(format="percent"),
        "True RUL inside 80% range": st.column_config.NumberColumn(format="percent"),
        "Typical 80% range width (h)": st.column_config.NumberColumn(format="%.0f"),
    })
    st.caption("Typical error is the median absolute difference between predicted and true remaining life. "
               f"The random-limits row is a further {fleets.get('test_custom', {}).get('capacitors', 0)} test "
               "capacitors, each with its own limits drawn from the allowed ranges. "
               "These results hold for the simulator. Real capacitors may age differently, so accuracy on real "
               "data is unknown until the model is tested on measured failures. Retrain with "
               "`python -m caplife.six_rul`.")


def _parameter_note(measured: tuple[str, ...]) -> None:
    accuracy = _model().metrics_for(measured)
    missing = [signal for signal in SIGNALS if signal not in measured]
    st.caption(f"Using {len(measured)} of 6 parameters: {_names(measured)} "
               f"(not in the data: {_names(missing)}). Typical error with these on simulated capacitors: "
               f"{accuracy['median_absolute_error_h']:,.0f} h ({accuracy['median_relative_error']:.0%}).")


def _report(label, fleet, device, readings, assessment, measured, limits, all_limits, recorded, changed, forecast,
            measurements) -> MonitorReport:
    """Everything the downloads need; the data check is the readiness audit of the uploaded readings."""
    check = None
    try:
        from caplife.readiness import audit_measurement_readiness
        from caplife.real_data import assess_real_measurements
        check = audit_measurement_readiness(assess_real_measurements(measurements))["summary"]
    except (ValueError, KeyError):  # the data check is extra; the report is still useful without it
        pass
    return MonitorReport(source=label, fleet=fleet, device=device, readings=readings, assessment=assessment,
                         measured=measured, limits=limits, all_limits=all_limits, recorded_conditions=recorded,
                         changed_conditions=changed, forecast=forecast, data_check=check, measurements=measurements,
                         accuracy=_model().metrics_for(measured))


def _device_conditions(devices) -> dict:
    """Operating conditions the user changed, per capacitor (only the changed values)."""
    stored = st.session_state.setdefault(CONDITIONS_STATE, {})
    return {str(device): dict(stored[str(device)]) for device in devices if stored.get(str(device))}


def _condition_inputs(device: str, recorded: pd.Series, effective: pd.Series) -> None:
    """Six operating-condition boxes for the selected capacitor, filled with its latest recorded values."""
    stored = st.session_state.setdefault(CONDITIONS_STATE, {})
    widget = lambda column: f"caplife_six_condition_{device}_{column}"

    def changed(column):
        value = st.session_state[widget(column)]
        own = stored.setdefault(device, {})
        if value is None or np.isclose(value, recorded.get(column, np.nan)):
            own.pop(column, None)
        else:
            own[column] = float(value)
        if not own:
            stored.pop(device, None)

    def reset():
        stored.pop(device, None)
        for column, *_ in conditions.PLANNED_INPUTS:
            # Set, not delete: a keyed widget keeps its last value when its state is only deleted.
            value = recorded.get(column, np.nan)
            st.session_state[widget(column)] = float(value) if np.isfinite(value) else None

    st.markdown(f"**Operating conditions for {device}**",
                help="Filled in with the latest conditions in the data. The model reads the conditions recorded "
                     "with every reading and assumes the latest ones continue. Change a value to see how remaining "
                     "life, end of life and the parameter forecasts change if the capacitor runs at those conditions "
                     "from its latest reading on; the measured history stays as it is. The core temperature moves "
                     "with the ambient temperature and ripple heating (about 12 K at the rated ripple current, less "
                     "at higher frequencies).")
    own = stored.get(device, {})
    for index, (column, label, unit, step) in enumerate(conditions.PLANNED_INPUTS):
        if index % 3 == 0:  # one row of three at a time, so a phone stacks them in reading order
            columns = st.columns(3)
        value = own.get(column, recorded.get(column, np.nan))
        kwargs = {} if widget(column) in st.session_state else {"value": float(value) if np.isfinite(value) else None}
        seen = recorded.get(column, np.nan)
        columns[index % 3].number_input(
            f"{label} ({unit})", min_value=0.0, step=step, key=widget(column), persist_state="session", on_change=changed, args=(column,),
            placeholder="Not recorded",
            help=(f"Latest recorded: {seen:,.4g} {unit}." if np.isfinite(seen) else "Not recorded in the data."),
            **kwargs)
    row = st.container(horizontal=True, vertical_alignment="center")
    row.button("Reset to recorded conditions", on_click=reset, key=f"caplife_six_condition_reset_{device}",
               disabled=not own)
    if own:
        row.badge("Changed conditions in use", icon=":material/edit:", color="violet",
                  help="Every prediction on this page for this capacitor uses the conditions above.")
    elif not recorded.notna().any():
        row.badge("No conditions recorded", color="gray",
                  help="The model predicts from the parameters alone. Enter conditions above, or add Ambient_C, "
                       "Core_C, Voltage_V, Rated_V, Ripple_A, Rated_Ripple_A and Ripple_Hz columns to the file.")
    for level, short, message in conditions.judgements(effective):
        row.badge(short, color=BADGE_COLORS[level], help=message)


def render() -> None:
    st.header("Monitor and Predict",
              help="Upload measurement histories of capacitance, ESR, leakage current, case expansion, generated "
                   "hydrogen and internal pressure (any of them), optionally with operating conditions. CapLife "
                   "scores each parameter's health against its limit, and a neural network predicts remaining useful "
                   "life with an 80% range. Demonstration model: trained on simulated capacitors; its accuracy on "
                   "real capacitors has not been tested.")
    loaded = _load()
    if loaded is None:
        return
    measurements, truth, label = loaded
    devices = sorted(measurements["Cap_ID"].astype(str).unique())
    all_limits = _device_limits(devices)
    planned = _device_conditions(devices)
    results = _assess(measurements, all_limits, planned)
    fleet = _fleet_table(results, truth, all_limits)

    custom = [device for device in devices if all_limits[device] != full_limits()]
    st.subheader("Fleet overview",
                 help=f"{label}. Each capacitor is assessed against its own end-of-life limits and operating "
                      "conditions, set under Capacitor detail.")
    cards = st.columns(3)
    cards[0].metric("Capacitors", len(fleet))
    cards[1].metric("Need attention", int(fleet["Status"].isin(["Watch", "Critical", "End of life"]).sum()))
    shortest = fleet["Remaining life, median (h)"].min()
    cards[2].metric("Shortest predicted remaining life", _hours(shortest))
    config = {"Capacitor": st.column_config.TextColumn(pinned=True),  # stays visible when swiping sideways on a phone
              "Remaining life, median (h)": st.column_config.NumberColumn(format="%.0f"),
              "Predicted end-of-life age (h)": st.column_config.NumberColumn(format="%.0f"),
              "Latest reading (h)": st.column_config.NumberColumn(format="%.0f"),
              "Overall health (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
              "Temperature (°C)": st.column_config.NumberColumn(format="%.0f", help="Latest core temperature, or "
                                                                "ambient where core was not recorded."),
              "Voltage / rated": st.column_config.NumberColumn(format="%.0f%%"),
              "Ripple / rated": st.column_config.NumberColumn(format="%.0f%%")}
    if truth:
        config["Simulated true end of life (h)"] = st.column_config.NumberColumn(format="%.0f")
    table(fleet, hide_index=True, width="stretch", column_config=config)
    changed_notes = ([f"Custom limits: {', '.join(custom)}"] if custom else []) + \
        ([f"Changed operating conditions: {', '.join(planned)}"] if planned else [])
    if changed_notes:
        st.caption(". ".join(changed_notes) + ".")

    st.subheader("Capacitor detail")
    device = st.selectbox("Capacitor", list(results), key="caplife_six_device", persist_state="session")
    readings, assessment = results[device]
    original = measurements.loc[measurements["Cap_ID"].astype(str).eq(str(device))].sort_values("Hours")
    measured = _measured(readings)
    if len(measured) < len(SIGNALS):
        _parameter_note(measured)
    with st.container(border=True):
        limits = _limit_inputs(str(device), measured)
    with st.container(border=True):
        _condition_inputs(str(device), conditions.latest(original), conditions.latest(readings))
    last = assessment.iloc[-1]
    default_limits = limits == full_limits()
    original = original.reset_index(drop=True)
    recorded_assessment = _model().assess(original, measured, limits) if planned.get(str(device)) else None
    recorded_last = recorded_assessment.iloc[-1] if recorded_assessment is not None else None
    if recorded_last is not None and pd.isna(recorded_last.get("RUL_median_h")):
        recorded_last = None
    if bool(last["End_of_life_reached"]):
        st.error(_end_of_life_message(assessment))
    elif "RUL_median_h" not in assessment or pd.isna(last.get("RUL_median_h")):
        st.warning(f"Remaining life needs at least {MIN_READINGS} readings; this capacitor has {len(readings)}.")
    else:
        detail = f"Based on {len(readings)} readings up to {last['Hours']:,.0f} h."
        delta = None
        if recorded_last is not None:
            delta = f"{last['RUL_median_h'] - recorded_last['RUL_median_h']:+,.0f} h vs recorded conditions"
            detail += f" With the recorded conditions: {_hours(recorded_last['RUL_median_h'])}."
            factor = conditions.aging_rule_factor(conditions.latest(original), planned[str(device)])
            if factor is not None:
                detail += (f" The standard aging rule (2× per 10 °C of core temperature, cube of the voltage ratio) "
                           f"would give {_hours(recorded_last['RUL_median_h'] * factor)}; on simulated capacitors "
                           "the network catches about two thirds of such a change.")
        if truth and default_limits and not planned.get(str(device)):
            detail += (f" Simulated true remaining life: {truth[device]['eol_h'] - last['Hours']:,.0f} h "
                       f"({truth[device]['eol_cause']}).")
        top = st.columns(4)
        top[0].metric("Remaining useful life", _hours(last["RUL_median_h"]), delta, delta_color="normal",
                      help=detail)
        top[1].metric("80% range", f"{_short(last['RUL_low_h'])}–{_short(last['RUL_high_h'])} h",
                      help="The true remaining life falls inside this range about 80% of the time on simulated "
                           "test capacitors.")
        top[2].metric("Predicted end-of-life age", _hours(last["Hours"] + last["RUL_median_h"]))
        top[3].metric("Overall health", f"{last['Overall_health_pct']:.0f}%",
                      help=f"The lowest parameter health. Limiting indicator: {last['Limiting_indicator']}.")
        used_now = limit_used(readings, limits)[list(measured)].iloc[-1]
        if used_now.max() >= 0.95:
            st.info(f"**{INDICATOR_LABELS[used_now.idxmax()]} is within 5% of your limit**, so end of life is "
                    "close and the 80% range is wider.")
        if bool(last.get("Outside_training_range")):
            st.warning("Some readings go beyond the simulated training capacitors, so this prediction is an "
                       "extrapolation.")
    _health_cards(readings, assessment, limits)

    st.markdown("##### Parameter history" if len(measured) < len(SIGNALS) else "##### Six-parameter history",
                help="Each dot is one measured reading, joined in time order. Orange dashed line: your limit. "
                     "With changed operating conditions, the gold dashed line estimates the history had the "
                     "capacitor always run at them: each measured state moves to the hour it would have been "
                     "reached by the standard aging rules (electrolyte aging 2× per 10 °C of core temperature and "
                     "with the cube of the voltage ratio; gas from leakage current, 2× per 20 °C), and gas, "
                     "pressure and expansion gains scale with how much faster gas forms per unit of aging. Real "
                     "capacitors follow these rules only approximately.")
    alternative = conditions.alternative_history(original, planned[str(device)]) if planned.get(str(device)) else None
    if planned.get(str(device)):
        st.caption("Teal: measured. Gold dashed: estimated history had it always run at your conditions."
                   if alternative is not None else
                   "An estimated history at your conditions needs a recorded core or ambient temperature.")
    _signal_charts(readings, limits, alternative=alternative)
    has_rul = "RUL_median_h" in assessment and pd.notna(last.get("RUL_median_h"))
    forecast = None
    if has_rul and not bool(last["End_of_life_reached"]):
        forecast = _forecast_cached(readings, float(last["RUL_median_h"]), json.dumps(limits, sort_keys=True),
                                    code_version())
        st.markdown(
            "##### Predicted parameter trends to end of life",
            help=f"Teal: the {len(readings)} measured readings up to {last['Hours']:,.0f} h (the thick line marked "
                 "Latest reading). Purple: the trend network's median forecast from there (dashed) and its 80% range "
                 f"(band). Grey line: the predicted end of life at {last['Hours'] + last['RUL_median_h']:,.0f} h, "
                 f"with its 80% range shaded grey. Orange dashed line: your limit. {_first_to_limit(forecast)} is "
                 "expected to reach its limit first.")
        zoom = st.toggle("Zoom in on the forecast", key="caplife_six_forecast_zoom", persist_state="session",
                         help="Show the latest readings and the forecast instead of the whole history.")
        reference = None
        if recorded_last is not None:
            reference = (_forecast_cached(original, float(recorded_last["RUL_median_h"]),
                                          json.dumps(limits, sort_keys=True), code_version()), recorded_last)
            st.caption("Gold: forecast and end of life (shaded 80% range) with your changed operating "
                       "conditions. Purple dotted: with the recorded conditions, and its shaded range. Hover a shaded "
                       "range for its hours.")
        _forecast_charts(readings, forecast, last, limits, zoom, reference)
    if assessment["RUL_median_h"].notna().any() if "RUL_median_h" in assessment else False:
        true_eol = truth[device]["eol_h"] if truth and default_limits and not planned.get(str(device)) else None
        changed = recorded_assessment is not None
        st.markdown("##### Remaining-life prediction over time",
                    help="Left of the Latest reading line: the remaining life predicted at each reading, from the "
                         "readings available then (purple, with its 80% range). Right of it: from the latest "
                         "reading, remaining life counts down one hour per hour to end of life; the shaded area at "
                         "the end is the end of life's 80% range."
                         + (" Purple: with the recorded conditions. Gold: with your changed conditions; left of the "
                            "line, each point is the prediction at that reading had the capacitor run at your "
                            "conditions from then on." if changed else "")
                         + (" Orange dashed: the simulated true remaining life." if true_eol else ""))
        if changed:
            st.caption("Purple: with the recorded conditions. Gold: with your changed conditions. Hover a shaded "
                       "range for its hours.")
        if _always_at_end_of_life(assessment):
            st.info("Every prediction is 0 h because an end-of-life limit was already met when the first "
                    "prediction could be made, so there is no remaining-life trend to plot.")
        else:
            history = None
            if changed:
                history = _changed_history_cached(original, json.dumps(planned[str(device)], sort_keys=True),
                                                  json.dumps(limits, sort_keys=True), code_version())
            st.altair_chart(_rul_chart(recorded_assessment if changed else assessment, true_eol,
                                       last if changed else None, history), width="stretch")

    with st.expander("About the model and how well it did on simulated data"):
        _model_card(measured)

    st.subheader("Downloads", help="The PDF and Excel reports cover the fleet, the selected capacitor (limits, "
                                   "operating conditions, health, forecasts, remaining life) and a data check. They "
                                   "are built when you click.")
    report = lambda: _report(label, fleet, str(device), readings, assessment, measured, limits, all_limits,
                             conditions.latest(original), planned.get(str(device), {}), forecast, measurements)
    stem = f"caplife_{str(device).replace(' ', '_')}"
    row = st.container(horizontal=True)
    row.download_button("Report (PDF)", lambda: pdf_bytes(report()), f"{stem}_report.pdf", mime="application/pdf",
                        type="primary", icon=":material/picture_as_pdf:", key="caplife_six_report_pdf")
    row.download_button("Report (Excel)", lambda: xlsx_bytes(report()), f"{stem}_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        icon=":material/table_view:", key="caplife_six_report_xlsx")
    row.download_button("Fleet summary (CSV)", fleet.to_csv(index=False), "caplife_fleet.csv", mime="text/csv",
                        icon=":material/download:", key="caplife_six_fleet_csv")
    row.download_button(f"{device} assessment (CSV)", readings.merge(assessment, on="Hours").to_csv(index=False),
                        f"{stem}_assessment.csv", mime="text/csv", icon=":material/download:",
                        key="caplife_six_device_csv")
