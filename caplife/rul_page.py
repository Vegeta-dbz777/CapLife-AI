"""Streamlit page: train and test a neural remaining-life model on the user's own data."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

import altair as alt

from caplife.chart_hover import hoverable
from caplife.six_rul_page import ZOOM_HELP, _rul_chart, _show_zoomable, _zoom_hours
from caplife.train_report import TrainView, pdf_bytes, xlsx_bytes
import pandas as pd
import streamlit as st

from caplife.tables import table

from caplife.accounts import save_controls, save_model_controls, saved_picker
from caplife.measurement_review import read_experimental_csv
from caplife.upload_ui import kept_file_uploader, show_notes
from caplife.real_data import RealDataValidationError
from caplife.rul_example import example_dataset
from caplife.versioning import code_version
from caplife.rul_workflow import (
    MAX_CENSOR_CONTRADICTIONS, MIN_CALIBRATION_DEVICES, MIN_TEST_COVERAGE, MIN_TEST_DEVICES,
    MIN_TEST_FAILURES, MIN_TRAINING_DEVICES, SPLIT_BY_DEVICE, SPLIT_BY_EXPERIMENT, SPLIT_ROTATE,
    TRAINING_RANGE_MARGIN, measurement_template_csv, outcome_template_csv, read_outcomes,
    run_rul_workflow,
)

UPLOAD, EXAMPLE, SAVED = "Upload my data", "Use simulated example capacitors", "My saved datasets"
REPORT_VIEW = "caplife_rul_report_view"  # the results as last shown, for the reports (and tests)
SUDDEN = "Chance of sudden failure (%)"
LIMITS_KEY = "caplife_rul_limit_"
INPUT_ERRORS = (RealDataValidationError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeError)


@st.cache_data(show_spinner=False)
def _example_csvs_cached(version: str) -> tuple[bytes, bytes]:
    measurements, outcomes = example_dataset()
    return measurements.to_csv(index=False).encode(), outcomes.to_csv(index=False).encode()


def _example_csvs() -> tuple[bytes, bytes]:
    return _example_csvs_cached(code_version())


# The last few training results, shared by every visitor (the same files and settings give the same result).
# Kept here rather than in st.cache_data so the page can show training progress while a result is made.
_RESULTS: OrderedDict = OrderedDict()
_RESULTS_LOCK = threading.Lock()
MAX_RESULTS = 8


def _run(measurements: bytes, outcomes: bytes, method: str, calibration: str | None, test: str | None,
         limits: dict | None = None, settings: tuple[str, str] = ("hours", "auto"), progress=None) -> dict:
    # Keyed by code_version() so results from an older model version are never reused after an update.
    key = hashlib.sha256(json.dumps([hashlib.sha256(measurements).hexdigest(), hashlib.sha256(outcomes).hexdigest(),
                                     method, calibration, test, limits or {}, list(settings), code_version()],
                                    sort_keys=True).encode()).hexdigest()
    with _RESULTS_LOCK:
        if key in _RESULTS:
            _RESULTS.move_to_end(key)
            return _RESULTS[key]
    time_unit, pressure = settings
    result = run_rul_workflow(read_experimental_csv(measurements, time_unit=time_unit, pressure=pressure),
                              read_outcomes(outcomes, time_unit), method, calibration, test, limits, progress)
    with _RESULTS_LOCK:
        _RESULTS[key] = result
        while len(_RESULTS) > MAX_RESULTS:
            _RESULTS.popitem(last=False)
    return result


def _hours(value) -> str:
    return "—" if pd.isna(value) else f"{value:,.0f}"


def _forecast_display(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Lead with remaining life; keep every number the CSV download has."""
    return pd.DataFrame({
        "Capacitor": forecasts["Capacitor"],
        "Remaining life (h)": forecasts["Remaining from latest, median (h)"].map(_hours),
        "80% range (h)": [
            "—" if pd.isna(low) else f"{low:,.0f} – {high:,.0f}" for low, high in zip(
                forecasts["Remaining from latest, 80% range from (h)"],
                forecasts["Remaining from latest, 80% range to (h)"])],
        "Latest reading (h)": forecasts["Latest reading (h)"].map(_hours),
        "Failure age (h)": forecasts["Predicted failure age (h)"].map(_hours),
        "Failure 80% range (h)": [f"{low:,.0f} – {high:,.0f}" for low, high in zip(
            forecasts["80% range from (h)"], forecasts["80% range to (h)"])],
        **({"Sudden failure (%)": forecasts[SUDDEN].map(lambda value: f"{value:.0f}")} if SUDDEN in forecasts else {}),
    })


FORECAST_COLUMNS = {
    "Capacitor": st.column_config.TextColumn(pinned=True),
    "Remaining life (h)": st.column_config.Column(help="Predicted remaining life from the latest reading (median)."),
    "80% range (h)": st.column_config.Column(help="The remaining life falls in this range 4 times in 5."),
    "Latest reading (h)": st.column_config.Column(help="Age at the capacitor's latest reading; remaining life "
                                                       "counts from here."),
    "Failure age (h)": st.column_config.Column(help="Predicted age at the failure endpoint: latest reading plus "
                                                    "remaining life."),
    "Failure 80% range (h)": st.column_config.Column(help="The age at failure falls in this range 4 times in 5."),
    "Sudden failure (%)": st.column_config.Column(help="The chance that a sudden failure (open circuit, short, vent), "
                                                       "rather than a parameter reaching its limit, ends this "
                                                       "capacitor's life. Your data included sudden failures, so the "
                                                       "trained model learned how likely they are."),
}


def _forecast_summary(forecasts: pd.DataFrame, device: str) -> str:
    row = forecasts.loc[forecasts["Capacitor"].eq(device)].iloc[0]
    return (f"**{device}** · remaining life {_hours(row['Remaining from latest, median (h)'])} h "
            f"(80% range {_hours(row['Remaining from latest, 80% range from (h)'])} – "
            f"{_hours(row['Remaining from latest, 80% range to (h)'])} h) · predicted failure at "
            f"{_hours(row['Predicted failure age (h)'])} h"
            + (f" · chance of sudden failure {row[SUDDEN]:.0f}%" if SUDDEN in row else ""))


def _model_name(source: str) -> str:
    """A suggested name for a model trained on this data: "Lab run 3 model", "batch_A model" (from batch_A_measurements.csv)."""
    import re
    stem = source.split(" and ")[0].rsplit(".", 1)[0] if source.endswith((".csv", ".xlsx")) or " and " in source \
        else source
    stem = re.sub(r"[\s_\-]*(measurements?|readings?|history)$", "", stem, flags=re.IGNORECASE).strip(" _-")
    return f"{stem or 'My'} model"


def _model_info(result: dict, context: dict) -> dict:
    """What a saved model was trained on and how it did on its test, for My Data (and Monitor and Predict)."""
    from caplife.real_data import MEASUREMENTS
    from caplife.six_rul import CONDITION_LABELS
    score = lambda name: {"miss_h": result["report"]["models"][name]["mean_distance_to_observed_window_h"],
                          "inside": result["report"]["models"][name]["definite_80pct_band_coverage_observed_events"]}
    return {
        "trained_on": context["source"], "split": context["split"], "endpoint": result["audit"]["endpoint"],
        "limits": result["limits"], "parameters": list(result["parameters"]),
        "parameter_names": [MEASUREMENTS[signal][0].split(" (")[0] for signal in result["parameters"]],
        "conditions": [CONDITION_LABELS[name] for name in result["operating_conditions"]],
        "scores": {"trained": score("trained"), "built_in": score("built_in"), "average": score("intercept_only")},
        "failure_modes": result["failure_modes"], "sudden": result["model"].sudden is not None,
        "tested": [list(row) for row in result["tested"]], "warnings": list(result.get("warnings", [])),
    }


def _reason_list(reasons: dict) -> str:
    return "\n".join(f"- **{device}**: {reason}" for device, reason in reasons.items())


def _test_chart(rows: pd.DataFrame, hours: tuple[float, float] | None = None, brush: bool = False) -> alt.LayerChart:
    """Each test capacitor on its own row: the neural network's predicted failure age and 80% range (purple),
    what was observed (teal: the failure window, or a triangle where it was still working when testing
    stopped) and when the forecast was made (grey tick). Styled and hovered like Monitor and Predict's charts;
    ``hours`` zooms it and ``brush`` lets a drag select hours."""
    from caplife.six_rul_page import DIVIDER, PURPLE, TEAL, _clipped, _zoom_layers
    failed = rows["Found failed by (h)"].notna()
    data = pd.DataFrame({
        "Capacitor": rows["Capacitor"], "Hours": rows["Predicted failure age (h)"],
        "low": rows["80% range from (h)"], "high": rows["80% range to (h)"], "made": rows["Forecast made at (h)"],
        "lower": rows["Last seen working (h)"], "upper": rows["Found failed by (h)"], "result": rows["Result"],
    })
    data["range"] = [f"{low:,.0f} – {high:,.0f}" for low, high in zip(data["low"], data["high"])]
    y = alt.Y("Capacitor:N", sort=None, title=None)
    scale = alt.Scale(domain=list(hours), nice=False) if hours else alt.Undefined
    x = alt.X("Hours:Q", title="Capacitor age (h)", scale=scale)
    hour = lambda field, title: alt.Tooltip(f"{field}:Q", title=title, format=",.0f")
    point = alt.Tooltip("point:N", title="Point")
    predicted = data.assign(point="Predicted failure")
    predicted_tip = [point, alt.Tooltip("Capacitor:N"), hour("Hours", "Predicted failure age (h)"),
                     alt.Tooltip("range:N", title="80% range (h)"), alt.Tooltip("result:N", title="Result")]
    observed = data.loc[failed.to_numpy()].assign(point="Observed failure")
    observed_tip = [point, alt.Tooltip("Capacitor:N"), hour("lower", "Last seen working (h)"),
                    hour("upper", "Found failed by (h)")]
    running = data.loc[~failed.to_numpy()].assign(point="Still working when testing stopped", Hours=lambda f: f["lower"])
    running_tip = [point, alt.Tooltip("Capacitor:N"), hour("Hours", "Last check (h)")]
    made = data.assign(point="Forecast made", Hours=data["made"])
    made_tip = [point, alt.Tooltip("Capacitor:N"), hour("Hours", "Age (h)")]
    layers = [
        alt.Chart(predicted).mark_bar(height=14, opacity=.32, color=PURPLE, cornerRadius=3).encode(
            y=y, x=alt.X("low:Q", title="Capacitor age (h)", scale=scale), x2="high:Q", tooltip=predicted_tip),
        alt.Chart(observed).mark_bar(height=6, color=TEAL, cornerRadius=2).encode(
            y=y, x=alt.X("lower:Q", scale=scale), x2="upper:Q", tooltip=observed_tip),
        alt.Chart(made).mark_tick(thickness=2, size=18, color=DIVIDER).encode(y=y, x=x, tooltip=made_tip),
        alt.Chart(running).mark_point(shape="triangle-right", filled=True, size=80, color=TEAL, opacity=1).encode(
            y=y, x=x, tooltip=running_tip),
        alt.Chart(predicted).mark_point(filled=True, size=60, color=PURPLE, opacity=1).encode(
            y=y, x=x, tooltip=predicted_tip),
    ]
    hovers = [*hoverable(predicted, x, y, predicted_tip, PURPLE), *hoverable(running, x, y, running_tip, TEAL),
              *hoverable(made, x, y, made_tip, DIVIDER)]
    if brush:  # at the bottom: a double-click anywhere on the chart resets the zoom
        ages = pd.concat([data["low"], data["high"], data["lower"], data["upper"], data["made"]]).dropna()
        layers[:0] = _zoom_layers(*(hours or (float(ages.min()), float(ages.max()))), exact=bool(hours))
    # Enough room per row that every capacitor label is drawn.
    return _clipped(alt.layer(*layers, *hovers).properties(height=max(160, 44 * len(data))), bool(hours))


TESTED_NOTE = ("Answers about conditions, limits or parameters outside these come from what the AI knew "
               "before training on your data.")
TEST_LEGEND = ("**Purple**: predicted failure age and its 80% range · **Teal**: when it actually failed (▶ still "
               "working when testing stopped) · **Grey tick**: when the forecast was made")


def _inputs() -> tuple[bytes, bytes, str] | None:
    source = st.radio("Data", (UPLOAD, EXAMPLE, SAVED), horizontal=True, key="caplife_rul_source",
                      persist_state="session",
                      help="The simulated example has 37 capacitors in four experiments, five of them still running. It "
                           "shows how the page works; its results say nothing about real capacitors.")
    if source == SAVED:
        picked = saved_picker("train_rul", "caplife_rul")
        return None if picked is None else (picked[0]["measurements"], picked[0]["outcomes"], picked[1])
    if source == EXAMPLE:
        measurements, outcomes = _example_csvs()
        row = st.container(horizontal=True)
        row.download_button("Download example measurements", measurements, "caplife_rul_example_measurements.csv",
                            mime="text/csv")
        row.download_button("Download example outcomes", outcomes, "caplife_rul_example_outcomes.csv",
                            mime="text/csv")
        return measurements, outcomes, "Simulated example"

    left, right = st.columns(2)
    with left:
        measurements = kept_file_uploader("Measurement history (CSV or Excel)", key="caplife_rul_measurements")
        st.download_button("Download measurement template", measurement_template_csv(),
                           "caplife_rul_measurements_template.csv", mime="text/csv")
    with right:
        outcomes = kept_file_uploader("Outcomes (CSV or Excel)", key="caplife_rul_outcomes")
        st.download_button("Download outcome template", outcome_template_csv(),
                           "caplife_rul_outcomes_template.csv", mime="text/csv")
    st.caption("Uploads are processed on the app server and are not stored unless you sign in and save them.")
    if measurements is None or outcomes is None:
        return None
    for upload in (measurements, outcomes):
        if upload.size > 8 * 1024 * 1024:
            st.error(f"{upload.name} exceeds the 8 MB limit.")
            return None
    save_controls("train_rul", {"measurements": measurements.getvalue(), "outcomes": outcomes.getvalue()},
                  measurements.name.rsplit(".", 1)[0], "caplife_rul")
    return measurements.getvalue(), outcomes.getvalue(), f"{measurements.name} and {outcomes.name}"


def _limit_inputs(measured: tuple[str, ...]) -> dict:
    """The end-of-life limits that defined failure in the user's test: six boxes like Monitor and Predict's."""
    from caplife.six_rul import LIMIT_KEYS
    from caplife.six_rul_page import LIMIT_INPUTS, _shown_limit, _stored_limit
    from caplife.six_sim import EOL_LIMITS, LIMIT_RANGES, SIGNALS
    key = lambda signal: LIMITS_KEY + signal

    def reset():
        for signal in SIGNALS:
            st.session_state[key(signal)] = _shown_limit(signal, EOL_LIMITS[LIMIT_KEYS[signal]])

    with st.container(border=True):
        st.markdown("**End-of-life limits used in your test**",
                    help="The limits that defined failure for your capacitors: a capacitor reached end of life when "
                         "a parameter reached its limit. A failure that came while every parameter was still inside "
                         "its limits counts as a sudden failure (open circuit, short, vent). The trained model "
                         "predicts the time to these limits, and to any others you set later. Capacitance, ESR and "
                         "leakage are measured from each capacitor's first reading.")
        limits = {}
        for index, signal in enumerate(SIGNALS):
            if index % 3 == 0:  # one row of three at a time, so a phone stacks them in reading order
                columns = st.columns(3)
            label, step = LIMIT_INPUTS[signal]
            low, high = LIMIT_RANGES[LIMIT_KEYS[signal]]
            kwargs = {} if key(signal) in st.session_state else {
                "value": _shown_limit(signal, EOL_LIMITS[LIMIT_KEYS[signal]])}
            shown = columns[index % 3].number_input(
                label, min_value=_shown_limit(signal, low), max_value=_shown_limit(signal, high), step=step,
                key=key(signal), persist_state="session", disabled=signal not in measured,
                help=("Not measured in your data." if signal not in measured else
                      f"Allowed: {_shown_limit(signal, low):g} to {_shown_limit(signal, high):g}."), **kwargs)
            limits[LIMIT_KEYS[signal]] = _stored_limit(signal, shown)
        st.button("Reset to defaults", on_click=reset, key="caplife_rul_limits_reset")
    return limits


ROTATE, ONE_TEST, BY_CAPACITOR = "Every experiment in turn (recommended)", "One test experiment", "By capacitor"


def _split_settings(experiments: list[str]) -> tuple[str, str | None, str | None]:
    options = {ROTATE: SPLIT_ROTATE, ONE_TEST: SPLIT_BY_EXPERIMENT, BY_CAPACITOR: SPLIT_BY_DEVICE}
    if len(experiments) >= 3:
        choice = st.radio("Choose the capacitors held out for testing", tuple(options), horizontal=True,
                          key="caplife_rul_split_v2", persist_state="session",
                          help="Every experiment in turn: one experiment calibrates the 80% range, and each other "
                               "experiment is the test once, with the model trained on the rest. The checks then "
                               "cover every one of those capacitors, so one unlucky experiment cannot decide the "
                               "verdict; forecasts come from a model trained on every experiment except the "
                               "calibration one. Takes about a minute. One test experiment: one experiment "
                               "calibrates, another tests and the rest train, quicker but decided by fewer "
                               "capacitors. By capacitor: about 20% of eligible capacitors go to calibration (at "
                               f"least {MIN_CALIBRATION_DEVICES}), 20% to testing (at least {MIN_TEST_DEVICES}) and the "
                               f"rest to training (at least {MIN_TRAINING_DEVICES}), fixed by each capacitor ID, not "
                               "by its outcome.")
    else:
        choice = BY_CAPACITOR
        st.caption("Splitting by experiment needs at least three experiments, so capacitors are split "
                   "individually. Test capacitors then share test conditions with training capacitors, "
                   "which makes the test easier than forecasting a new experiment.")
    method = options[choice]
    if method == SPLIT_BY_DEVICE:
        return method, None, None
    if method == SPLIT_ROTATE:
        left, _ = st.columns(2)
        calibration = left.selectbox("Calibration experiment", experiments, index=0, key="caplife_rul_calibration",
                                     persist_state="session",
                                     help="Sets the width of the 80% range; every other experiment is tested in turn.")
        return method, calibration, None
    left, right = st.columns(2)
    calibration = left.selectbox("Calibration experiment", experiments, index=0, key="caplife_rul_calibration", persist_state="session")
    test = right.selectbox("Test experiment", [e for e in experiments if e != calibration], index=0,
                           key="caplife_rul_test", persist_state="session",
                           help="All other experiments train the model.")
    return method, calibration, test


def _show_results(result: dict, attempts: int, context: dict) -> None:
    roles = result["roles"]
    by_role = result["experiments_by_role"]
    from caplife.real_data import MEASUREMENTS
    used = [MEASUREMENTS[signal][0].split(" (")[0] for signal in result["parameters"]]
    from caplife.six_rul import CONDITION_LABELS
    operating = [CONDITION_LABELS[name].lower() for name in result.get("operating_conditions", ())]
    split = (f" Training experiments: {', '.join(by_role['training'])}. Calibration: "
             f"{', '.join(by_role['calibration'])}. Test: {', '.join(by_role['test'])}."
             if by_role.get("test") else "")
    data_note = (f"Endpoint: {result['audit']['endpoint']}.{split} The training capacitors gave "
                 f"{result['training_landmarks']:,} readings to learn from. Parameters used (the ones every capacitor "
                 f"with an outcome has): {', '.join(used)}. "
                 + (f"Operating conditions used: {', '.join(operating)}." if operating else
                    "No operating condition is recorded for every capacitor with an outcome, so none is used."))
    st.subheader("1 · Data used", help=f"Endpoint: {result['audit']['endpoint']}.{split} The training capacitors "
                                       f"gave {result['training_landmarks']:,} readings to learn from.")
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    tags = []
    for name in ("training", "calibration", "test"):
        colour = {"training": "blue", "calibration": "violet", "test": "orange"}[name]
        rotated = name == "test" and result.get("turns") is not None
        row.badge(f"{name.title()} {roles[name]}", color=colour,
                  help=f"{roles[name]} capacitors" + (f" from {', '.join(by_role[name])}" if by_role.get(name) else "")
                       + (", each tested by a model that never saw its experiment" if rotated else "")
                       + (". The final model trains on all of them" if name == "training" and result.get("turns")
                          is not None else "") + ".")
        tags.append((colour, f"{name.title()} {roles[name]}"))
    row.badge(f"Parameters {len(used)} of 6", icon=":material/monitoring:", color="gray",
              help=f"The ones every capacitor with an outcome has: {', '.join(used)}.")
    tags.append(("gray", f"Parameters {len(used)} of 6"))
    conditions_tag = f"Operating conditions {len(operating)}" if operating else "No operating conditions"
    row.badge(conditions_tag, icon=":material/thermostat:", color="gray",
              help=(f"Used: {', '.join(operating)}." if operating else
                    "No operating condition is recorded for every capacitor with an outcome, so none is used."))
    tags.append(("gray", conditions_tag))
    excluded = result["audit"]["devices_excluded"]
    if excluded:
        with st.expander(f"{len(excluded)} capacitor(s) with outcomes could not be used"):
            st.markdown(_reason_list(excluded))
    tested = pd.DataFrame(result["tested"], columns=["What", "Your data"])
    with st.expander("What your data tested", icon=":material/fact_check:"):
        st.markdown("\n".join(f"- **{what}:** {value}" for what, value in result["tested"]))
        st.caption(TESTED_NOTE)

    st.subheader("2 · Test on capacitors the model never saw",
                 help="Average miss is how far the predicted failure age fell outside each test capacitor's observed "
                      "failure window (zero when inside it). The built-in model, as it is before training on your "
                      "data, and the average-lifetime guess show whether your trained model adds anything. In the "
                      "chart, purple is your trained model's predicted failure age and its 80% range; teal is when "
                      "each test capacitor actually failed (between last seen working and found failed), or a "
                      "triangle where it was still working when testing stopped; the grey tick is when the forecast "
                      "was made." + ZOOM_HELP)
    table(
        result["model_table"], hide_index=True, width="stretch",
        column_config={
            "Model": st.column_config.TextColumn(pinned=True),
            "Failures inside 80% range": st.column_config.NumberColumn(format="percent"),
        },
    )
    turns = result.get("turns")
    if turns is not None:
        st.markdown("##### Each experiment as the test",
                    help="Each row is one turn: the model trained on the other experiments, calibrated on the "
                         "calibration experiment and tested on this one. The verdict below uses all turns together; "
                         "\"On its own\" says whether that turn alone would pass the same checks.")
        table(turns, hide_index=True, width="stretch", column_config={
            "Test experiment": st.column_config.TextColumn(pinned=True),
            "Failures inside 80% range": st.column_config.NumberColumn(format="percent")})
    test_rows = result["test_table"]
    test_key = "caplife_rul_zoom_test_" + hashlib.sha256(",".join(test_rows["Capacitor"]).encode()).hexdigest()[:12]
    test_charts = _show_zoomable([(test_key, lambda hours: (TEST_LEGEND, _test_chart(test_rows, hours, brush=True)))],
                                 columns=1)
    with st.expander("Each test capacitor"):
        table(result["test_table"], hide_index=True, width="stretch",
                     column_config={"Capacitor": st.column_config.TextColumn(pinned=True)})

    checks = (f"Checks: at least {MIN_TEST_FAILURES} observed test failures; the 80% range contains at least "
              f"{MIN_TEST_COVERAGE:.0%} of them; the average miss is no worse than predicting the average lifetime; "
              f"at most {MAX_CENSOR_CONTRADICTIONS:.0%} of still-working test capacitors get a range that ends before "
              "their last check. Passing shows the model worked on these test capacitors under these test "
              "conditions, not under different ones.")
    st.subheader("3 · Verdict", help=checks)
    verdict = ("**Passed.** Your trained model met every check on the test capacitors, so its forecasts are shown "
               "below." if result["passed"] else "**Forecasts withheld.** Your trained model did not pass:")
    if result["passed"]:
        st.success(verdict)
    else:
        st.warning(verdict + "\n\n" + "\n".join(f"- {reason}" for reason in result["reasons"]))
    warnings = list(result.get("warnings", []))
    if attempts > 1:
        warnings.append(f"You have trained {attempts} different settings (limits or split) on this data. Choosing "
                        "settings after seeing results weakens the test; report every one you tried.")
    for warning in warnings:
        st.warning(warning)
    if result["passed"]:
        save_model_controls(result["model"], _model_info(result, context), context["default_name"],
                            "caplife_rul_model", worse_than_built_in=bool(result.get("warnings")))

    st.subheader("4 · Remaining-life forecasts",
                 help="Each forecast uses all of a capacitor's readings up to its latest one; remaining life counts "
                      "from that reading.")
    forecasts, withheld = result["forecasts"], result["forecasts_withheld"]
    forecast_message, rul_charts, shown_forecasts = None, [], None
    history = result.get("forecast_history")
    if not result["forecast_devices"]:
        forecast_message = ("Every capacitor has an outcome row. To forecast a capacitor, include its readings "
                            "without an outcome row.")
        st.info(forecast_message)
    elif not result["passed"]:
        forecast_message = (f"{len(result['forecast_devices'])} capacitor(s) have no outcome row. Their forecasts "
                            "are not shown because the test did not pass.")
        st.write(forecast_message)
    else:
        if forecasts is not None and len(forecasts):
            shown_forecasts = _forecast_display(forecasts)
            table(shown_forecasts, hide_index=True, width="stretch", column_config=FORECAST_COLUMNS)
            if history is not None and len(history):
                st.markdown("##### Remaining-life prediction over time",
                            help="Each point is the forecast the trained model would have made at that reading, "
                                 "using only the readings up to it, with its calibrated 80% range. From the latest "
                                 "reading, remaining life counts down to the predicted end of life, whose 80% range "
                                 "is shaded." + ZOOM_HELP)
                device = st.selectbox("Capacitor", list(forecasts["Capacitor"]), key="caplife_rul_forecast_device",
                                      persist_state="session")
                summary = _forecast_summary(forecasts, device)
                device_history = history.loc[history["Capacitor"].eq(device)]
                _show_zoomable([(f"caplife_rul_zoom_rul_{device}", lambda hours: (
                    summary, _rul_chart(device_history, None, hours=hours, brush=True)))], columns=1)
                # The report has every forecast capacitor's chart (zoomed as left on the page).
                rul_charts = [(_forecast_summary(forecasts, name), _rul_chart(
                    history.loc[history["Capacitor"].eq(name)], None,
                    hours=_zoom_hours(f"caplife_rul_zoom_rul_{name}"))) for name in forecasts["Capacitor"]]
        if withheld:
            with st.expander(f"Forecast withheld for {len(withheld)} capacitor(s)", icon=":material/block:"):
                st.markdown(_reason_list(withheld))
                st.caption(f"A forecast is withheld when early readings go more than {TRAINING_RANGE_MARGIN:.0%} of "
                           "the training range beyond what the training capacitors showed.")

    withheld_note = (f"A forecast is withheld when early readings go more than {TRAINING_RANGE_MARGIN:.0%} of the "
                     "training range beyond what the training capacitors showed.")
    view = TrainView(
        source=context["source"], data_metrics=context["metrics"], split=context["split"],
        tags=tags, data_note=data_note, excluded=excluded, tested=tested, tested_note=TESTED_NOTE,
        model_table=result["model_table"],
        test_legend=TEST_LEGEND, test_chart=test_charts[0][1], test_table=result["test_table"].round(1),
        passed=result["passed"], verdict=verdict, reasons=[] if result["passed"] else list(result["reasons"]),
        checks=checks, warnings=warnings, forecast_message=forecast_message, forecasts=shown_forecasts,
        rul_charts=rul_charts, withheld=withheld, withheld_note=withheld_note,
        turns=turns,
        tables={"Model comparison": result["model_table"],
                **({"Each experiment as the test": turns} if turns is not None else {}),
                "Test results": result["test_table"],
                "Forecasts": forecasts, "Forecast history": None if history is None else history.rename(columns={
                    "Hours": "Age (h)", "RUL_median_h": "Remaining life, median (h)",
                    "RUL_low_h": "Remaining life, 80% range from (h)",
                    "RUL_high_h": "Remaining life, 80% range to (h)"}),
                "What your data tested": tested,
                "Not used": pd.DataFrame({"Capacitor": list(excluded), "Reason": list(excluded.values())}),
                "Forecast withheld": pd.DataFrame({"Capacitor": list(withheld), "Reason": list(withheld.values())})})
    st.session_state[REPORT_VIEW] = view

    st.subheader("Downloads", help="The PDF report is the page as shown, with every forecast capacitor's "
                                   "remaining-life chart; the Excel workbook has every table in full. Both are built "
                                   "when you click.")
    report = {**result["report"], "splits_tried_on_this_data_in_session": attempts}
    row = st.container(horizontal=True)
    row.download_button("Report (PDF)", lambda: pdf_bytes(view), "caplife_train_report.pdf", mime="application/pdf",
                        type="primary", icon=":material/picture_as_pdf:", key="caplife_rul_report_pdf")
    row.download_button("Report (Excel)", lambda: xlsx_bytes(view), "caplife_train_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        icon=":material/table_view:", key="caplife_rul_report_xlsx")
    row.download_button("Full report (JSON)", json.dumps(report, indent=2, allow_nan=False),
                        "caplife_rul_report.json", mime="application/json", icon=":material/data_object:")
    row.download_button("Test results (CSV)", result["test_table"].to_csv(index=False),
                        "caplife_rul_test_results.csv", mime="text/csv", icon=":material/download:")
    if forecasts is not None and len(forecasts):
        row.download_button("Forecasts (CSV)", forecasts.to_csv(index=False), "caplife_rul_forecasts.csv",
                            mime="text/csv", icon=":material/download:")


def render() -> None:
    st.header("Train The AI",
              help="Upload measurement histories (any of the six indicators) and the recorded outcome of each tested "
                   "capacitor. CapLife trains the built-in AI further on some of your capacitors, tests it on "
                   "capacitors it never saw, and forecasts remaining life for your other capacitors only if that "
                   "test passes. "
                   "Forecasts describe the uploaded capacitors and endpoint only; accuracy under other operating "
                   "conditions, product series or endpoints is untested. A research prototype, not a maintenance or "
                   "safety tool.")
    with st.expander("How it works"):
        st.markdown(
            "CapLife starts from the built-in AI, the same one Monitor and Predict uses, and trains it further on "
            "your capacitors, so it keeps everything it can do: any of the six parameters, any end-of-life "
            "limits, operating conditions and each parameter's path to end of life.\n\n"
            "1. **Training** capacitors teach it how your capacitors age: how fast each parameter moves towards "
            "the limits you set, and, if some failed suddenly while every parameter was still inside its limits, "
            "how likely such a sudden failure is. Capacitors still working at the end of testing count too: "
            "they show the failure came later.\n"
            "2. **Calibration** capacitors set the width of the 80% range, separately early and late in life: "
            "once a parameter has used half its limit, the readings show how fast a capacitor is wearing and the "
            "range can be much tighter.\n"
            "3. **Test** capacitors check the predictions against what actually happened. The built-in AI and an "
            "average-lifetime guess are scored alongside for comparison. With **every experiment in turn**, each "
            "experiment except the calibration one is the test once, with the model trained on the others, so the "
            "checks cover far more capacitors; the forecasts then come from a model trained on all of them.\n"
            "4. If your trained model passes, it forecasts each capacitor without an outcome from its latest "
            "reading, using the same model and range that were tested.\n\n"
            "Every reading from a capacitor's third onward is a training example. Calibration and testing use one "
            "reading per capacitor, chosen from its ID and readings, never from its outcome, so the checks stay "
            "independent."
        )

    payload = _inputs()
    if payload is None:
        return
    measurements, outcomes, source = payload
    settings = ("hours", "auto")  # hours unless a column says Days or Weeks; gauge pressure is detected
    try:
        outcome_frame = read_outcomes(outcomes, settings[0])
        measurement_frame = read_experimental_csv(measurements, time_unit=settings[0], pressure=settings[1])
    except INPUT_ERRORS as error:
        st.error(f"The files could not be read: {error}")
        return
    show_notes(list(dict.fromkeys(measurement_frame.attrs.get("notes", []) + outcome_frame.attrs.get("notes", []))))
    experiments = sorted(outcome_frame["Experiment_ID"].astype(str).str.strip().unique()) \
        if "Experiment_ID" in outcome_frame else []
    measured = measurement_frame["Cap_ID"].astype(str).str.strip().nunique() if "Cap_ID" in measurement_frame else 0
    metrics = [("Capacitors measured", f"{measured}"), ("With an outcome", f"{len(outcome_frame)}"),
               ("Experiments", f"{len(experiments)}")]
    for card, (name, value) in zip(st.columns(3), metrics):
        card.metric(name, value)

    from caplife.six_sim import SIGNALS
    present = tuple(signal for signal in SIGNALS
                    if signal in measurement_frame and measurement_frame[signal].notna().any())
    limits = _limit_inputs(present)
    method, calibration, test = _split_settings(experiments)
    data_key = hashlib.sha256(measurements + b"\0" + outcomes).hexdigest()
    run_key = (data_key, json.dumps(limits, sort_keys=True), method, calibration, test)
    if st.button("Train and test the model", type="primary",
                 help="Decide the limits and the split before looking at results: trying settings until one passes "
                      "weakens the test. Training takes up to about a minute."):
        tried = st.session_state.setdefault("caplife_rul_tried", {})
        tried.setdefault(data_key, set()).add(run_key[1:])
        st.session_state["caplife_rul_run"] = run_key
    if st.session_state.get("caplife_rul_run") != run_key:
        return
    bar = st.empty()
    try:
        result = _run(measurements, outcomes, method, calibration, test, limits, settings,
                      progress=lambda share, text: bar.progress(min(max(share, 0.0), 1.0), text=text))
    except INPUT_ERRORS as error:
        bar.empty()
        st.error(f"The model could not be trained: {error}")
        return
    bar.empty()
    attempts = len(st.session_state.get("caplife_rul_tried", {}).get(data_key, ()))
    split = {SPLIT_ROTATE: f"Every experiment in turn: {calibration} calibrates the 80% range, and each other "
                           "experiment is the test once, with the model trained on the rest. Forecasts come from a "
                           f"model trained on every experiment except {calibration}.",
             SPLIT_BY_EXPERIMENT: f"One test experiment: {calibration} calibrates the 80% range, {test} tests the "
                                  "model, and the other experiments train it.",
             SPLIT_BY_DEVICE: "Split by capacitor: about 20% of eligible capacitors calibrate the 80% range, 20% test "
                              "the model and the rest train it."}[method]
    default_name = _model_name(source)
    _show_results(result, attempts, {"source": source, "metrics": metrics, "split": split,
                                     "default_name": default_name})

