"""Streamlit page: train and test a neural remaining-life model on the user's own data."""
from __future__ import annotations

import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from caplife.measurement_review import read_experimental_csv
from caplife.real_data import RealDataValidationError
from caplife.rul_example import example_dataset
from caplife.rul_workflow import (
    MAX_CENSOR_CONTRADICTIONS, MIN_CALIBRATION_DEVICES, MIN_TEST_COVERAGE, MIN_TEST_DEVICES,
    MIN_TEST_FAILURES, MIN_TRAINING_DEVICES, SPLIT_BY_DEVICE, SPLIT_BY_EXPERIMENT,
    TRAINING_RANGE_MARGIN, measurement_template_csv, outcome_template_csv, read_outcomes,
    run_rul_workflow,
)

UPLOAD, EXAMPLE = "Upload my data", "Use the synthetic example"
RANGE_COLOR, POINT_COLOR, OBSERVED_COLOR = "#3aa597", "#0f6f63", "#d9822b"
INPUT_ERRORS = (RealDataValidationError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeError)


@st.cache_data(show_spinner=False)
def _example_csvs() -> tuple[bytes, bytes]:
    measurements, outcomes = example_dataset()
    return measurements.to_csv(index=False).encode(), outcomes.to_csv(index=False).encode()


@st.cache_data(show_spinner=False, max_entries=8)
def _run(measurements: bytes, outcomes: bytes, method: str, calibration: str | None, test: str | None) -> dict:
    return run_rul_workflow(read_experimental_csv(measurements), read_outcomes(outcomes),
                            method, calibration, test)


def _hours(value) -> str:
    return "—" if pd.isna(value) else f"{value:,.0f}"


def _forecast_display(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Lead with remaining life; keep every number the CSV download has."""
    return pd.DataFrame({
        "Capacitor": forecasts["Capacitor"],
        "Remaining life, median (h)": forecasts["Remaining from latest, median (h)"].map(_hours),
        "Remaining life, 80% range (h)": [
            "—" if pd.isna(low) else f"{low:,.0f} – {high:,.0f}" for low, high in zip(
                forecasts["Remaining from latest, 80% range from (h)"],
                forecasts["Remaining from latest, 80% range to (h)"])],
        "Latest reading (h)": forecasts["Latest reading (h)"].map(_hours),
        "Predicted failure age (h)": forecasts["Predicted failure age (h)"].map(_hours),
        "Failure age, 80% range (h)": [f"{low:,.0f} – {high:,.0f}" for low, high in zip(
            forecasts["80% range from (h)"], forecasts["80% range to (h)"])],
        "Status": forecasts["Status"],
    })


def _reason_list(reasons: dict) -> str:
    return "\n".join(f"- **{device}**: {reason}" for device, reason in reasons.items())


def _interval_chart(table: pd.DataFrame, observed: bool) -> alt.Chart:
    data = pd.DataFrame({
        "capacitor": table["Capacitor"], "low": table["80% range from (h)"],
        "mid": table["Predicted failure age (h)"], "high": table["80% range to (h)"],
    })
    y = alt.Y("capacitor:N", sort=None, title=None)
    layers = [
        alt.Chart(data).mark_bar(height=10, opacity=.45, color=RANGE_COLOR, cornerRadius=3).encode(
            y=y, x=alt.X("low:Q", title="Capacitor age (h)"), x2="high:Q",
            tooltip=[alt.Tooltip("capacitor:N", title="Capacitor"),
                     alt.Tooltip("low:Q", title="80% range from (h)", format=",.0f"),
                     alt.Tooltip("high:Q", title="80% range to (h)", format=",.0f")]),
        alt.Chart(data).mark_point(filled=True, size=55, color=POINT_COLOR).encode(
            y=y, x="mid:Q", tooltip=[alt.Tooltip("mid:Q", title="Predicted failure age (h)", format=",.0f")]),
    ]
    if observed:
        seen = table.assign(capacitor=table["Capacitor"], lower=table["Last seen working (h)"],
                            upper=table["Found failed by (h)"])
        failed = seen.dropna(subset=["upper"])
        layers.append(alt.Chart(failed).mark_rule(strokeWidth=3, color=OBSERVED_COLOR).encode(
            y=y, x="lower:Q", x2="upper:Q",
            tooltip=[alt.Tooltip("lower:Q", title="Last seen working (h)", format=",.0f"),
                     alt.Tooltip("upper:Q", title="Found failed by (h)", format=",.0f")]))
        layers.append(alt.Chart(failed).mark_tick(thickness=3, size=14, color=OBSERVED_COLOR).encode(
            y=y, x="upper:Q"))
        running = seen.loc[seen["upper"].isna()]
        if len(running):
            layers.append(alt.Chart(running).mark_point(shape="triangle-right", filled=True, size=70,
                                                        color=OBSERVED_COLOR).encode(
                y=y, x="lower:Q", tooltip=[alt.Tooltip("lower:Q", title="Still working at (h)", format=",.0f")]))
    else:
        latest = table.assign(capacitor=table["Capacitor"], latest=table["Latest reading (h)"])
        layers.append(alt.Chart(latest).mark_tick(thickness=2, size=18, color=OBSERVED_COLOR).encode(
            y=y, x="latest:Q", tooltip=[alt.Tooltip("latest:Q", title="Latest reading (h)", format=",.0f")]))
    # Enough room per row that every capacitor label is drawn.
    return alt.layer(*layers).properties(height=max(150, 42 * len(data)))


def _inputs() -> tuple[bytes, bytes] | None:
    source = st.radio("Data", (UPLOAD, EXAMPLE), horizontal=True, key="caplife_rul_source")
    if source == EXAMPLE:
        measurements, outcomes = _example_csvs()
        st.info("The synthetic example has 37 generated capacitors in four experiments, five of them still "
                "running. It shows how the page works; its results say nothing about real capacitors.")
        left, right = st.columns(2)
        left.download_button("Download example measurements", measurements, "caplife_rul_example_measurements.csv",
                             mime="text/csv")
        right.download_button("Download example outcomes", outcomes, "caplife_rul_example_outcomes.csv",
                              mime="text/csv")
        return measurements, outcomes

    left, right = st.columns(2)
    with left:
        measurements = st.file_uploader("Measurement history CSV", type=["csv"], key="caplife_rul_measurements")
        st.download_button("Download measurement template", measurement_template_csv(),
                           "caplife_rul_measurements_template.csv", mime="text/csv")
    with right:
        outcomes = st.file_uploader("Outcome CSV", type=["csv"], key="caplife_rul_outcomes")
        st.download_button("Download outcome template", outcome_template_csv(),
                           "caplife_rul_outcomes_template.csv", mime="text/csv")
    with st.expander("What goes in each file"):
        st.markdown(
            "**Measurement history** — one row per reading: `Cap_ID`, `Hours` and all six indicators "
            "(capacitance µF, ESR Ω, leakage µA, case expansion µm, cumulative H₂ mL at 0 °C and 1 atm, "
            "absolute internal pressure kPa). Each capacitor needs a complete reading at 0 h and two more "
            "complete readings. Add `ESR_Test_Hz` and `ESR_Test_Temp_C` when known.\n\n"
            "**Outcomes** — one row per capacitor that finished testing: `Cap_ID`, `Experiment_ID`, "
            "`Endpoint` (the same failure definition for every row), `Last_alive_h` (last age it was seen "
            "working) and `Failure_by_h` (age by which it was found failed). Leave `Failure_by_h` blank if "
            "the capacitor was still working when testing stopped.\n\n"
            "**Capacitors to forecast** — include their readings in the measurement file but give them no "
            "outcome row."
        )
    st.caption("Uploads are processed on the app server. Share data you have permission to use.")
    if measurements is None or outcomes is None:
        return None
    for upload in (measurements, outcomes):
        if upload.size > 8 * 1024 * 1024:
            st.error(f"{upload.name} exceeds the 8 MB limit.")
            return None
    return measurements.getvalue(), outcomes.getvalue()


def _split_settings(experiments: list[str]) -> tuple[str, str | None, str | None]:
    options = {"By experiment (recommended)": SPLIT_BY_EXPERIMENT, "By capacitor": SPLIT_BY_DEVICE}
    if len(experiments) >= 3:
        choice = st.radio("Choose the capacitors held out for testing", tuple(options), horizontal=True,
                          key="caplife_rul_split")
    else:
        choice = "By capacitor"
        st.caption("Splitting by experiment needs at least three experiments, so capacitors are split "
                   "individually. Test capacitors then share test conditions with training capacitors, "
                   "which makes the test easier than forecasting a new experiment.")
    method = options[choice]
    if method == SPLIT_BY_DEVICE:
        st.caption(f"About 20% of eligible capacitors go to calibration (at least {MIN_CALIBRATION_DEVICES}), "
                   f"20% to testing (at least {MIN_TEST_DEVICES}) and the rest to training "
                   f"(at least {MIN_TRAINING_DEVICES}). The assignment is fixed by each capacitor ID, not by its outcome.")
        return method, None, None
    left, right = st.columns(2)
    calibration = left.selectbox("Calibration experiment", experiments, index=0, key="caplife_rul_calibration")
    test = right.selectbox("Test experiment", [e for e in experiments if e != calibration], index=0,
                           key="caplife_rul_test")
    st.caption("All other experiments train the model.")
    return method, calibration, test


def _show_results(result: dict, attempts: int) -> None:
    roles = result["roles"]
    st.subheader("1 · Data used")
    cards = st.columns(4)
    cards[0].metric("Eligible capacitors", sum(roles.values()))
    cards[1].metric("Training", roles["training"])
    cards[2].metric("Calibration", roles["calibration"])
    cards[3].metric("Test", roles["test"])
    by_role = result["experiments_by_role"]
    st.caption(f"Endpoint: {result['audit']['endpoint']}. Training experiments: "
               f"{', '.join(by_role['training'])}. Calibration: {', '.join(by_role['calibration'])}. "
               f"Test: {', '.join(by_role['test'])}.")
    excluded = result["audit"]["devices_excluded"]
    if excluded:
        with st.expander(f"{len(excluded)} capacitor(s) with outcomes could not be used"):
            st.markdown(_reason_list(excluded))

    st.subheader("2 · Test on capacitors the model never saw")
    st.dataframe(
        result["model_table"], hide_index=True, use_container_width=True,
        column_config={
            "Average miss (h)": st.column_config.NumberColumn(format="%.0f"),
            "Failures inside 80% range": st.column_config.NumberColumn(format="percent"),
            "Typical 80% range width (h)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption("Average miss is how far the predicted failure age fell outside each test capacitor's observed "
               "failure window (zero when inside it). The two simpler models show whether the neural network "
               "adds anything.")
    st.altair_chart(_interval_chart(result["test_table"], observed=True), use_container_width=True)
    st.caption("Teal bars: the neural network's 80% range, with the dot at its predicted failure age. "
               "Orange: when each test capacitor actually failed (between last seen working and found failed), "
               "or a triangle where it was still working when testing stopped.")
    with st.expander("Each test capacitor"):
        st.dataframe(result["test_table"].round(1), hide_index=True, use_container_width=True)

    st.subheader("3 · Verdict")
    if result["passed"]:
        st.success("**Passed.** The neural network met every check on the test capacitors, "
                   "so its forecasts are shown below.")
    else:
        st.warning("**Forecasts withheld.** The neural network did not pass:\n\n"
                   + "\n".join(f"- {reason}" for reason in result["reasons"]))
    st.caption(f"Checks: at least {MIN_TEST_FAILURES} observed test failures; the 80% range contains at least "
               f"{MIN_TEST_COVERAGE:.0%} of them; the average miss is no worse than predicting the average "
               f"lifetime; at most {MAX_CENSOR_CONTRADICTIONS:.0%} of still-working test capacitors get a range "
               "that ends before their last check. Passing shows the model worked on these test capacitors "
               "under these test conditions, not under different ones.")
    if attempts > 1:
        st.warning(f"You have trained {attempts} different splits of this data. Choosing the split after seeing "
                   "results weakens the test; report every split you tried.")

    st.subheader("4 · Remaining-life forecasts")
    forecasts, withheld = result["forecasts"], result["forecasts_withheld"]
    if not result["forecast_devices"]:
        st.info("Every capacitor has an outcome row. To forecast a capacitor, include its readings "
                "without an outcome row.")
    elif not result["passed"]:
        st.write(f"{len(result['forecast_devices'])} capacitor(s) have no outcome row. Their forecasts are not "
                 "shown because the test did not pass.")
    else:
        if forecasts is not None and len(forecasts):
            st.altair_chart(_interval_chart(forecasts, observed=False), use_container_width=True)
            st.caption("Teal bars: 80% range for the age at failure, with the dot at the predicted failure age. "
                       "Orange tick: each capacitor's latest reading.")
            st.dataframe(_forecast_display(forecasts), hide_index=True, use_container_width=True)
            made_at = forecasts["Forecast made at (h)"].unique()
            when = f"{made_at[0]:,.0f} h" if len(made_at) == 1 else "each capacitor's third complete reading"
            st.caption(f"Each forecast uses readings up to {when}, the same point used in testing. "
                       "Remaining life counts from the latest reading.")
        if withheld:
            st.write(f"Withheld for {len(withheld)} capacitor(s):")
            st.markdown(_reason_list(withheld))
            st.caption(f"A forecast is withheld when early readings go more than {TRAINING_RANGE_MARGIN:.0%} of the "
                       "training range beyond what the training capacitors showed.")

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
    st.header("Train a remaining-life model on your data")
    st.write("Upload six-indicator measurement histories and the recorded outcome of each tested capacitor. "
             "CapLife trains a neural network on some of your capacitors, tests it on capacitors it never saw, "
             "and forecasts remaining life for your other capacitors only if that test passes.")
    with st.expander("How it works"):
        st.markdown(
            "1. **Training** capacitors teach the neural network how early changes in the six indicators relate "
            "to when a capacitor reaches your failure endpoint. Capacitors still working at the end of testing "
            "count too: they show the failure came later.\n"
            "2. **Calibration** capacitors set the width of the 80% range.\n"
            "3. **Test** capacitors check the predictions against what actually happened. A simple log-linear "
            "model and an average-lifetime guess are scored alongside for comparison.\n"
            "4. If the neural network passes, it forecasts each capacitor without an outcome, using the same "
            "trained model and range that were tested.\n\n"
            "Each capacitor contributes one forecast, made at its third complete reading after a complete 0 h "
            "baseline, from the changes and rates of change up to that point."
        )

    payload = _inputs()
    if payload is None:
        return
    measurements, outcomes = payload
    try:
        outcome_frame = read_outcomes(outcomes)
        measurement_frame = read_experimental_csv(measurements)
    except INPUT_ERRORS as error:
        st.error(f"The files could not be read: {error}")
        return
    experiments = sorted(outcome_frame["Experiment_ID"].astype(str).str.strip().unique()) \
        if "Experiment_ID" in outcome_frame else []
    measured = measurement_frame["Cap_ID"].astype(str).str.strip().nunique() if "Cap_ID" in measurement_frame else 0
    cards = st.columns(3)
    cards[0].metric("Capacitors measured", measured)
    cards[1].metric("With an outcome", len(outcome_frame))
    cards[2].metric("Experiments", len(experiments))

    method, calibration, test = _split_settings(experiments)
    st.caption("Decide the split before looking at results. Trying splits until one passes weakens the test.")
    data_key = hashlib.sha256(measurements + b"\0" + outcomes).hexdigest()
    run_key = (data_key, method, calibration, test)
    if st.button("Train and test the model", type="primary"):
        tried = st.session_state.setdefault("caplife_rul_tried", {})
        tried.setdefault(data_key, set()).add(run_key[1:])
        st.session_state["caplife_rul_run"] = run_key
    if st.session_state.get("caplife_rul_run") != run_key:
        st.caption("Results appear here after training. Small datasets train in a few seconds.")
        return
    try:
        with st.spinner("Training and testing…"):
            result = _run(measurements, outcomes, method, calibration, test)
    except INPUT_ERRORS as error:
        st.error(f"The model could not be trained: {error}")
        return
    attempts = len(st.session_state.get("caplife_rul_tried", {}).get(data_key, ()))
    _show_results(result, attempts)
    st.caption("Forecasts describe the uploaded capacitors and endpoint only. Accuracy under other operating "
               "conditions, product series or endpoints is untested. This is a research prototype, not a "
               "maintenance or safety tool.")
