"""Streamlit page: train and test a neural remaining-life model on the user's own data."""
from __future__ import annotations

import hashlib
import json

import altair as alt

from caplife.chart_hover import hoverable
from caplife.six_rul_page import ZOOM_HELP, _rul_chart, _show_zoomable
import pandas as pd
import streamlit as st

from caplife.tables import table

from caplife.accounts import save_controls, saved_picker
from caplife.measurement_review import read_experimental_csv
from caplife.upload_ui import kept_file_uploader, show_notes
from caplife.real_data import RealDataValidationError
from caplife.rul_example import example_dataset
from caplife.versioning import code_version
from caplife.rul_workflow import (
    MAX_CENSOR_CONTRADICTIONS, MIN_CALIBRATION_DEVICES, MIN_TEST_COVERAGE, MIN_TEST_DEVICES,
    MIN_TEST_FAILURES, MIN_TRAINING_DEVICES, SPLIT_BY_DEVICE, SPLIT_BY_EXPERIMENT,
    TRAINING_RANGE_MARGIN, measurement_template_csv, outcome_template_csv, read_outcomes,
    run_rul_workflow,
)

UPLOAD, EXAMPLE, SAVED = "Upload my data", "Use the synthetic example", "My saved datasets"
INPUT_ERRORS = (RealDataValidationError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeError)


@st.cache_data(show_spinner=False)
def _example_csvs_cached(version: str) -> tuple[bytes, bytes]:
    measurements, outcomes = example_dataset()
    return measurements.to_csv(index=False).encode(), outcomes.to_csv(index=False).encode()


def _example_csvs() -> tuple[bytes, bytes]:
    return _example_csvs_cached(code_version())


def _run(measurements: bytes, outcomes: bytes, method: str, calibration: str | None, test: str | None,
         settings: tuple[str, str] = ("hours", "auto")) -> dict:
    return _run_cached(measurements, outcomes, method, calibration, test, settings, code_version())


# Keyed by code_version() so results from an older model version are never reused after an update.
@st.cache_data(show_spinner=False, max_entries=8)
def _run_cached(measurements: bytes, outcomes: bytes, method: str, calibration: str | None, test: str | None,
                settings: tuple[str, str], version: str) -> dict:
    time_unit, pressure = settings
    return run_rul_workflow(read_experimental_csv(measurements, time_unit=time_unit, pressure=pressure),
                            read_outcomes(outcomes, time_unit), method, calibration, test)


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
        "Status": forecasts["Status"],
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


TEST_LEGEND = ("**Purple**: predicted failure age and its 80% range · **Teal**: when it actually failed (▶ still "
               "working when testing stopped) · **Grey tick**: when the forecast was made")


def _inputs() -> tuple[bytes, bytes] | None:
    source = st.radio("Data", (UPLOAD, EXAMPLE, SAVED), horizontal=True, key="caplife_rul_source",
                      persist_state="session",
                      help="The synthetic example has 37 generated capacitors in four experiments, five of them still "
                           "running. It shows how the page works; its results say nothing about real capacitors.")
    if source == SAVED:
        picked = saved_picker("train_rul", "caplife_rul")
        return None if picked is None else (picked[0]["measurements"], picked[0]["outcomes"])
    if source == EXAMPLE:
        measurements, outcomes = _example_csvs()
        left, right = st.columns(2)
        left.download_button("Download example measurements", measurements, "caplife_rul_example_measurements.csv",
                             mime="text/csv")
        right.download_button("Download example outcomes", outcomes, "caplife_rul_example_outcomes.csv",
                              mime="text/csv")
        return measurements, outcomes

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
    return measurements.getvalue(), outcomes.getvalue()


def _split_settings(experiments: list[str]) -> tuple[str, str | None, str | None]:
    options = {"By experiment (recommended)": SPLIT_BY_EXPERIMENT, "By capacitor": SPLIT_BY_DEVICE}
    if len(experiments) >= 3:
        choice = st.radio("Choose the capacitors held out for testing", tuple(options), horizontal=True,
                          key="caplife_rul_split", persist_state="session",
                          help="By experiment: one experiment calibrates the 80% range, another tests the model, "
                               "and the rest train it, so the test is a new experiment. By capacitor: about 20% of "
                               f"eligible capacitors go to calibration (at least {MIN_CALIBRATION_DEVICES}), 20% to "
                               f"testing (at least {MIN_TEST_DEVICES}) and the rest to training (at least "
                               f"{MIN_TRAINING_DEVICES}), fixed by each capacitor ID, not by its outcome.")
    else:
        choice = "By capacitor"
        st.caption("Splitting by experiment needs at least three experiments, so capacitors are split "
                   "individually. Test capacitors then share test conditions with training capacitors, "
                   "which makes the test easier than forecasting a new experiment.")
    method = options[choice]
    if method == SPLIT_BY_DEVICE:
        return method, None, None
    left, right = st.columns(2)
    calibration = left.selectbox("Calibration experiment", experiments, index=0, key="caplife_rul_calibration", persist_state="session")
    test = right.selectbox("Test experiment", [e for e in experiments if e != calibration], index=0,
                           key="caplife_rul_test", persist_state="session",
                           help="All other experiments train the model.")
    return method, calibration, test


def _show_results(result: dict, attempts: int) -> None:
    roles = result["roles"]
    by_role = result["experiments_by_role"]
    from caplife.real_data import MEASUREMENTS
    used = [MEASUREMENTS[signal][0].split(" (")[0] for signal in result["parameters"]]
    from caplife.six_rul import CONDITION_LABELS
    operating = [CONDITION_LABELS[name].lower() for name in result.get("operating_conditions", ())]
    split = (f" Training experiments: {', '.join(by_role['training'])}. Calibration: "
             f"{', '.join(by_role['calibration'])}. Test: {', '.join(by_role['test'])}."
             if by_role.get("test") else "")
    st.subheader("1 · Data used", help=f"Endpoint: {result['audit']['endpoint']}.{split} The training capacitors "
                                       f"gave {result['training_landmarks']:,} readings to learn from.")
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    for name in ("training", "calibration", "test"):
        row.badge(f"{name.title()} {roles[name]}", color={"training": "blue", "calibration": "violet",
                                                         "test": "orange"}[name],
                  help=f"{roles[name]} capacitors" + (f" from {', '.join(by_role[name])}" if by_role.get(name) else "")
                       + ".")
    row.badge(f"Parameters {len(used)} of 6", icon=":material/monitoring:", color="gray",
              help=f"The ones every capacitor with an outcome has: {', '.join(used)}.")
    row.badge(f"Operating conditions {len(operating)}" if operating else "No operating conditions",
              icon=":material/thermostat:", color="gray",
              help=(f"Used: {', '.join(operating)}." if operating else
                    "No operating condition is recorded for every capacitor with an outcome, so none is used."))
    excluded = result["audit"]["devices_excluded"]
    if excluded:
        with st.expander(f"{len(excluded)} capacitor(s) with outcomes could not be used"):
            st.markdown(_reason_list(excluded))

    st.subheader("2 · Test on capacitors the model never saw",
                 help="Average miss is how far the predicted failure age fell outside each test capacitor's observed "
                      "failure window (zero when inside it). The two simpler models show whether the neural network "
                      "adds anything. In the chart, teal bars are the neural network's 80% range with the dot at its "
                      "predicted failure age; orange is when each test capacitor actually failed (between last seen "
                      "working and found failed), or a triangle where it was still working when testing stopped."
                      + ZOOM_HELP)
    table(
        result["model_table"], hide_index=True, width="stretch",
        column_config={
            "Model": st.column_config.TextColumn(pinned=True),
            "Average miss (h)": st.column_config.NumberColumn(format="%.0f"),
            "Failures inside 80% range": st.column_config.NumberColumn(format="percent"),
            "Typical 80% range width (h)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    test_rows = result["test_table"]
    test_key = "caplife_rul_zoom_test_" + hashlib.sha256(",".join(test_rows["Capacitor"]).encode()).hexdigest()[:12]
    _show_zoomable([(test_key, lambda hours: (TEST_LEGEND, _test_chart(test_rows, hours, brush=True)))], columns=1)
    with st.expander("Each test capacitor"):
        table(result["test_table"].round(1), hide_index=True, width="stretch",
                     column_config={"Capacitor": st.column_config.TextColumn(pinned=True)})

    st.subheader("3 · Verdict",
                 help=f"Checks: at least {MIN_TEST_FAILURES} observed test failures; the 80% range contains at least "
                      f"{MIN_TEST_COVERAGE:.0%} of them; the average miss is no worse than predicting the average "
                      f"lifetime; at most {MAX_CENSOR_CONTRADICTIONS:.0%} of still-working test capacitors get a range "
                      "that ends before their last check. Passing shows the model worked on these test capacitors "
                      "under these test conditions, not under different ones.")
    if result["passed"]:
        st.success("**Passed.** The neural network met every check on the test capacitors, "
                   "so its forecasts are shown below.")
    else:
        st.warning("**Forecasts withheld.** The neural network did not pass:\n\n"
                   + "\n".join(f"- {reason}" for reason in result["reasons"]))
    if attempts > 1:
        st.warning(f"You have trained {attempts} different splits of this data. Choosing the split after seeing "
                   "results weakens the test; report every split you tried.")

    st.subheader("4 · Remaining-life forecasts",
                 help="Each forecast uses all of a capacitor's readings up to its latest one; remaining life counts "
                      "from that reading.")
    forecasts, withheld = result["forecasts"], result["forecasts_withheld"]
    if not result["forecast_devices"]:
        st.info("Every capacitor has an outcome row. To forecast a capacitor, include its readings "
                "without an outcome row.")
    elif not result["passed"]:
        st.write(f"{len(result['forecast_devices'])} capacitor(s) have no outcome row. Their forecasts are not "
                 "shown because the test did not pass.")
    else:
        if forecasts is not None and len(forecasts):
            table(_forecast_display(forecasts), hide_index=True, width="stretch", column_config=FORECAST_COLUMNS)
            history = result.get("forecast_history")
            if history is not None and len(history):
                st.markdown("##### Remaining-life prediction over time",
                            help="Each point is the forecast the trained model would have made at that reading, "
                                 "using only the readings up to it, with its calibrated 80% range. From the latest "
                                 "reading, remaining life counts down to the predicted end of life, whose 80% range "
                                 "is shaded." + ZOOM_HELP)
                device = st.selectbox("Capacitor", list(forecasts["Capacitor"]), key="caplife_rul_forecast_device",
                                      persist_state="session")
                row = forecasts.loc[forecasts["Capacitor"].eq(device)].iloc[0]
                summary = (f"**{device}** · remaining life {_hours(row['Remaining from latest, median (h)'])} h "
                           f"(80% range {_hours(row['Remaining from latest, 80% range from (h)'])} – "
                           f"{_hours(row['Remaining from latest, 80% range to (h)'])} h) · predicted failure at "
                           f"{_hours(row['Predicted failure age (h)'])} h")
                device_history = history.loc[history["Capacitor"].eq(device)]
                _show_zoomable([(f"caplife_rul_zoom_rul_{device}", lambda hours: (
                    summary, _rul_chart(device_history, None, hours=hours, brush=True)))], columns=1)
        if withheld:
            with st.expander(f"Forecast withheld for {len(withheld)} capacitor(s)", icon=":material/block:"):
                st.markdown(_reason_list(withheld))
                st.caption(f"A forecast is withheld when early readings go more than {TRAINING_RANGE_MARGIN:.0%} of "
                           "the training range beyond what the training capacitors showed.")

    st.subheader("Downloads")
    report = {**result["report"], "splits_tried_on_this_data_in_session": attempts}
    cols = st.columns(3)
    cols[0].download_button("Download full report (JSON)", json.dumps(report, indent=2, allow_nan=False),
                            "caplife_rul_report.json", mime="application/json", type="primary")
    cols[1].download_button("Download test results (CSV)", result["test_table"].to_csv(index=False),
                            "caplife_rul_test_results.csv", mime="text/csv")
    if forecasts is not None and len(forecasts):
        cols[2].download_button("Download forecasts (CSV)", forecasts.to_csv(index=False),
                                "caplife_rul_forecasts.csv", mime="text/csv")


def render() -> None:
    st.header("Train The AI",
              help="Upload measurement histories (any of the six indicators) and the recorded outcome of each tested "
                   "capacitor. CapLife trains a neural network on some of your capacitors, tests it on capacitors it "
                   "never saw, and forecasts remaining life for your other capacitors only if that test passes. "
                   "Forecasts describe the uploaded capacitors and endpoint only; accuracy under other operating "
                   "conditions, product series or endpoints is untested. A research prototype, not a maintenance or "
                   "safety tool.")
    with st.expander("How it works"):
        st.markdown(
            "1. **Training** capacitors teach the neural network how early changes in the six indicators relate "
            "to when a capacitor reaches your failure endpoint. Capacitors still working at the end of testing "
            "count too: they show the failure came later.\n"
            "2. **Calibration** capacitors set the width of the 80% range.\n"
            "3. **Test** capacitors check the predictions against what actually happened. A simple log-linear "
            "model and an average-lifetime guess are scored alongside for comparison.\n"
            "4. If the neural network passes, it forecasts each capacitor without an outcome from its latest "
            "reading, using the same trained model and range that were tested.\n\n"
            "The model learns from every reading from a capacitor's third onward: the change in each parameter "
            "since the first reading, its recent rate and the time since the first reading, against the time "
            "that was left until the endpoint. Each capacitor counts equally, however many readings it has. "
            "Calibration and testing use one reading per capacitor, chosen from its ID and readings, never from "
            "its outcome, so the checks stay independent."
        )

    payload = _inputs()
    if payload is None:
        return
    measurements, outcomes = payload
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
    cards = st.columns(3)
    cards[0].metric("Capacitors measured", measured)
    cards[1].metric("With an outcome", len(outcome_frame))
    cards[2].metric("Experiments", len(experiments))

    method, calibration, test = _split_settings(experiments)
    data_key = hashlib.sha256(measurements + b"\0" + outcomes).hexdigest()
    run_key = (data_key, method, calibration, test)
    if st.button("Train and test the model", type="primary",
                 help="Decide the split before looking at results: trying splits until one passes weakens the test. "
                      "Small datasets train in a few seconds."):
        tried = st.session_state.setdefault("caplife_rul_tried", {})
        tried.setdefault(data_key, set()).add(run_key[1:])
        st.session_state["caplife_rul_run"] = run_key
    if st.session_state.get("caplife_rul_run") != run_key:
        return
    try:
        with st.spinner("Training and testing…"):
            result = _run(measurements, outcomes, method, calibration, test, settings)
    except INPUT_ERRORS as error:
        st.error(f"The model could not be trained: {error}")
        return
    attempts = len(st.session_state.get("caplife_rul_tried", {}).get(data_key, ()))
    _show_results(result, attempts)

