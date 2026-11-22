"""CapLife AI measurement assessments and forecast research."""
from __future__ import annotations

from base64 import b64encode
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


def _forget_outdated_caplife_modules() -> None:
    """Make the next imports load the current ``caplife`` code after a repository update.

    Streamlit Community Cloud pulls an update into the running server and reruns this
    script, but modules imported by an earlier run stay in ``sys.modules`` with their old
    code, so a page could keep showing behaviour that has already been replaced.
    """
    package = sys.modules.get("caplife")
    if package is None:
        return
    fingerprint = getattr(package, "source_fingerprint", None)
    if fingerprint is not None and fingerprint() == getattr(package, "LOADED_FINGERPRINT", None):
        return
    for name in [name for name in sys.modules if name == "caplife" or name.startswith("caplife.")]:
        del sys.modules[name]


_forget_outdated_caplife_modules()

from caplife.config import EOL_THRESHOLDS, FEATURES, SEQ_LEN
from caplife.data import DataValidationError, validate_measurements
from caplife.accounts import current_user, render_my_data, sidebar_account
from caplife.predict import assess_capacitor, default_prediction_hour, load_engine
from caplife.trend import assess_trend
from caplife.versioning import code_version


ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / "sample_data" / "demo_capacitors.csv"
TEMPLATE_DATA = ROOT / "sample_data" / "upload_template.csv"
PUBLIC_SAMPLE = ROOT / "sample_data" / "nasa_es14_reference.csv"
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
UNCERTAINTY_JSON = ARTIFACTS / "trend_uncertainty_synthetic_v0_5.json"
PUBLIC_APP_URL = "https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/"
PROJECT_URL = "https://github.com/Vegeta-dbz777/CapLife-AI"
MEASUREMENT_WORKFLOW = "Experimental measurements (v0.8)"
RUL_WORKFLOW = "Train an RUL model on uploaded outcomes"
SIX_RUL_WORKFLOW = "Six-parameter RUL prediction (demonstration model)"
MY_DATA_WORKFLOW = "My saved datasets"
MAIN_WORKFLOWS = {
    MEASUREMENT_WORKFLOW: "Analyze measurements",
    SIX_RUL_WORKFLOW: "RUL prediction",
    RUL_WORKFLOW: "Train RUL model",
}
RESEARCH_WORKFLOWS = {
    "Public-data forecast benchmark": "Forecast benchmarks",
    "Public-data neural experiment": "Neural research",
    "Synthetic RUL demonstration (v0.5)": "Earlier GRU demo",
}
WORKFLOWS = {**MAIN_WORKFLOWS, **RESEARCH_WORKFLOWS}
MAIN_KEY, RESEARCH_KEY = "caplife_workflow_v07", "caplife_research_page"

st.set_page_config(
    page_title="CapLife AI | Capacitor research",
    page_icon="⚡",
    layout="wide",
)
st.html(ASSETS / "interface.css")

if st.query_params.get("page") == "privacy":
    from caplife.privacy import render as render_privacy
    render_privacy(PUBLIC_APP_URL)
    st.stop()

with st.sidebar:
    st.html(
        '<div class="caplife-brand"><span class="caplife-brand-mark" aria-hidden="true">CL</span>'
        '<span>CapLife AI</span></div>'
    )
    st.html(ASSETS / "theme-switch.html", unsafe_allow_javascript=True)
    main_pages = {**MAIN_WORKFLOWS, **({MY_DATA_WORKFLOW: "My data"} if current_user() else {})}
    # Two lists, one selection: choosing a page in one list clears the other.
    st.session_state.setdefault(MAIN_KEY, MEASUREMENT_WORKFLOW)
    st.session_state.setdefault(RESEARCH_KEY, None)
    if st.session_state[MAIN_KEY] not in (None, *main_pages):  # e.g. My data after signing out
        st.session_state[MAIN_KEY] = MEASUREMENT_WORKFLOW

    def _choose(chosen_key: str, other_key: str) -> None:
        if st.session_state[chosen_key] is not None:
            st.session_state[other_key] = None

    st.radio("Workspace", tuple(main_pages), format_func=main_pages.__getitem__, key=MAIN_KEY,
             on_change=_choose, args=(MAIN_KEY, RESEARCH_KEY))
    with st.expander("Research", expanded=st.session_state[RESEARCH_KEY] is not None):
        st.radio("Research pages", tuple(RESEARCH_WORKFLOWS), format_func=RESEARCH_WORKFLOWS.__getitem__,
                 key=RESEARCH_KEY, label_visibility="collapsed", on_change=_choose, args=(RESEARCH_KEY, MAIN_KEY))
    workflow = st.session_state[MAIN_KEY] or st.session_state[RESEARCH_KEY] or MEASUREMENT_WORKFLOW
    st.divider()
    sidebar_account()
    st.markdown(f"[Project and methods ↗]({PROJECT_URL})")
    st.markdown(f"[Privacy]({PUBLIC_APP_URL}?page=privacy)")

if workflow == MEASUREMENT_WORKFLOW:
    hero = b64encode((ASSETS / "capacitor-hero.webp").read_bytes()).decode("ascii")
    st.html(
        '<section class="caplife-hero">'
        f'<img src="data:image/webp;base64,{hero}" alt="">'
        '<div><div class="caplife-overline">Aluminum electrolytic capacitors</div>'
        '<h1>CapLife AI</h1>'
        '<p>Explore measured degradation, inspect thresholds, and export a report.</p></div>'
        '</section>'
    )

if workflow == MY_DATA_WORKFLOW:
    render_my_data()
    st.stop()

if workflow == SIX_RUL_WORKFLOW:
    from caplife.six_rul_page import render as render_six_rul_page
    render_six_rul_page()
    st.stop()

if workflow == RUL_WORKFLOW:
    from caplife.rul_page import render as render_rul_page
    render_rul_page()
    st.stop()

if workflow == "Public-data neural experiment":
    benchmark = json.loads((ARTIFACTS / "neural_benchmark.json").read_text(encoding="utf-8"))
    horizon = benchmark["horizon_h"]
    threshold = benchmark["threshold_pct"]
    st.header("Public-data neural experiment")
    st.write(
        f"Could a small neural network predict whether a capacitor would reach {threshold}% "
        f"capacitance loss within {horizon} hours? Each model learns from one study and is "
        f"tested on capacitors from the other."
    )
    st.info("The simple trend scored better than the neural network in both original EOS-2 and PoFINN transfers.")
    study = st.selectbox("Held-out study", list(benchmark["results"]))
    result = benchmark["results"][study]
    predictions = pd.DataFrame(result["predictions"])
    summary = pd.DataFrame(result["summary"])
    resolved = summary.iloc[0]
    st.caption(
        f"Training study · {result['train_study']} ({resolved['training_devices']} devices) | "
        f"{resolved['test_devices']} test devices · {resolved['resolved']} resolved forecasts · "
        f"{resolved['unresolved']} unresolved"
    )
    st.dataframe(summary[["method", "brier_score", "true_alerts", "false_alerts", "missed_crossings"]]
                 .rename(columns={"method": "Method", "brier_score": "Brier score",
                                  "true_alerts": "Correct alerts", "false_alerts": "False alerts",
                                  "missed_crossings": "Missed crossings"}).round(3), hide_index=True)
    st.caption(
        "Lower Brier scores mean better forecasts. Each capacitor has equal weight. "
        "An alert is any forecast at or above 50%. Alert counts use only resolved outcomes."
    )

    device = st.selectbox("Capacitor", predictions["device"].unique())
    device_predictions = predictions.loc[predictions["device"] == device].set_index("origin_h")
    origin = st.select_slider("Issue the retrospective forecast at this study age (hours)",
                              options=device_predictions.index.tolist(),
                              value=device_predictions.index[len(device_predictions) // 2])
    forecast = device_predictions.loc[origin]
    measured = pd.DataFrame(result["traces"])
    history = measured.loc[(measured["device"] == device) & (measured["age_h"] <= origin)]
    st.line_chart(history.rename(columns={"age_h": "Study age (h)", "loss_pct": "Measured loss (%)"}),
                  x="Study age (h)", y="Measured loss (%)", height=230)

    score, trend, loss = st.columns(3)
    score.metric("Neural crossing score", f"{forecast['neural']:.0%}")
    trend.metric("Trend alert", "Yes" if forecast["trend"] else "No")
    loss.metric("Measured capacitance loss", f"{forecast['loss_pct']:.1f}%")
    if pd.isna(forecast["outcome"]):
        outcome = "Unresolved from inspection times"
    else:
        outcome = "Crossed by the horizon" if forecast["outcome"] else "Stayed below through the horizon"
    st.write(f"**Observed result** · {outcome}")
    seed_low, seed_high = f"{forecast['neural_seed_low']:.0%}", f"{forecast['neural_seed_high']:.0%}"
    if seed_low == seed_high:
        seed_note = f"All three training starts gave {seed_low} at this forecast. "
    else:
        seed_note = (f"Scores from three training starts ranged from {seed_low} to {seed_high}. "
                     "This spread shows sensitivity to initialization. ")
    st.caption(seed_note + "The estimated crossing probabilities still need calibration on independent data.")
    if not forecast["within_training_feature_ranges"]:
        st.warning("At least one input is outside the range observed in the training study.")
    st.download_button("Download neural benchmark results", predictions.to_csv(index=False),
                       "caplife_neural_benchmark.csv", mime="text/csv")

    es10 = benchmark.get("es10_stress_check")
    if es10 is not None:
        with st.expander("Additional NASA ES10 check"):
            es10_scores = pd.DataFrame(es10["summary"])
            st.write(
                "Here the models train on both primary studies and are tested on NASA ES10. "
                "The ES10 capacitance readings near 100 Hz stayed below the 20% mark. This check can reveal "
                "false alerts. Estimating missed crossings requires a cohort with observed crossings."
            )
            st.dataframe(es10_scores[["method", "resolved", "unresolved", "false_alerts", "brier_score"]]
                         .rename(columns={"method": "Method", "resolved": "Resolved",
                                          "unresolved": "Unresolved", "false_alerts": "False alerts",
                                          "brier_score": "Brier score"}).round(3), hide_index=True)
            st.caption(f"{es10['outside_training_ranges']} ES10 forecasts had inputs outside the training ranges.")
            st.markdown(f"[NASA ES10 source data]({es10['source']})")
    with st.expander("Evidence and methods"):
        st.write(
            "The network and logistic model use capacitance loss and the two most recent loss rates. "
            "Only readings available at the forecast age enter the model. If a crossing window "
            "straddles the horizon or follow-up ends early, the outcome stays unresolved. "
            "The model uses capacitance only because the public studies lack compatible measurements "
            "of other indicators and operating conditions."
        )
        for name, url in benchmark["sources"].items():
            st.markdown(f"- [{name}]({url})")
    with st.expander("Extended public-data validation"):
        validation_bytes = (ARTIFACTS / "release_validation.json").read_bytes()
        validation = json.loads(validation_bytes)
        outcomes = validation["scenarios"]["primary"]["held_out"]
        methods = validation["methods"]
        rows = [
            {"Held-out experiment": experiment, "Capacitors in experiment": scores["devices_in_experiment"],
             "Capacitors with forecasts": scores["capacitors"],
             "Resolved windows": scores["resolved"],
             **{methods[method]: entry["brier"] for method, entry in scores["by_method"].items()}}
            for experiment, scores in outcomes.items()
        ]
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        st.caption(
            "The NASA ES10, ES12 and ES14 cohorts were held out together. Lower Brier scores "
            "mean better forecasts. Each capacitor contributes equal weight to its experiment's score."
        )
        st.download_button(
            "Download extended validation results", validation_bytes,
            "caplife_release_validation.json", mime="application/json",
        )
    with st.expander("Threshold warnings and physical failure research"):
        failure_bytes = (ARTIFACTS / "physical_failure_validation.json").read_bytes()
        failure = json.loads(failure_bytes)
        rows = []
        for case, result in failure["folds_by_minimum_observations"]["3"]["held_out_cases"].items():
            if result["metrics"] is None:
                continue
            for name, model in result["metrics"]["models"].items():
                scores = model["first_origin_per_device"]
                rows.append({
                    "Held-out condition": f"Case {case}",
                    "Model": {"intercept_only": "Intercept only", "log_linear": "Log-linear",
                              "neural": "Neural"}[name],
                    "Capacitors": result["devices_with_eligible_history"],
                    "Mean distance to failure window (h)": scores["mean_distance_to_observed_window_h"],
                    "Possible coverage of nominal 80% band":
                        scores["possible_80pct_band_coverage_observed_events"],
                })
        st.write(
            "The physical failure study reports catastrophic open or short circuits. "
            "The exploratory test used capacitance and ESR read from the published figures, "
            "with one forecast per eligible capacitor. The nominal 80% neural bands "
            "failed the held-out condition checks."
        )
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        st.caption(
            "Possible coverage counts any overlap with the reported failure window. "
            "The third condition had insufficient pre-failure readings for a forecast."
        )
        st.markdown(f"[Aalborg study and test conditions]({failure['source']})")
        for label, filename, contents in (
            ("Download physical failure validation", "physical_failure_validation.json", failure_bytes),
            ("Download time-to-threshold validation", "remaining_time_validation.json", None),
            ("Download constrained warning benchmark", "constrained_validation.json", None),
        ):
            st.download_button(label, contents if contents is not None else (ARTIFACTS / filename).read_bytes(),
                               f"caplife_{filename}", mime="application/json")
    st.stop()

if workflow == "Public-data forecast benchmark":
    benchmark = json.loads((ARTIFACTS / "public_benchmark.json").read_text(encoding="utf-8"))
    origin = benchmark["reference_age_h"]
    threshold = benchmark["capacitance_loss_threshold_pct"]
    mass = benchmark["nominal_band_mass"]
    st.header("Public-data forecast benchmark")
    st.write(
        f"At {origin} aging hours, each method forecasts the time until capacitance loss "
        f"first reaches {threshold}%. Each study takes a turn as the held-out test set."
    )
    st.info(
        "These results describe the two public studies. Physical failure dates and reliable "
        "prediction ranges for new devices still require validation."
    )

    summary = pd.DataFrame(benchmark["summary"])
    predictions = pd.DataFrame(benchmark["predictions"])
    test_study = st.selectbox("Held-out study", summary["test_study"].unique())
    results = summary.loc[summary["test_study"] == test_study].copy()
    observed = results["observed_crossings"].iloc[0]
    censored = results["right_censored"].iloc[0]
    st.caption(
        f"Training study · {results['train_study'].iloc[0]}  |  "
        f"{observed} crossings observed  |  {censored} tests ended below the threshold"
    )
    st.dataframe(results[[
        "model", "mean_gap_h", "point_hits", "band_overlaps", "band_contains", "median_band_width_h",
    ]].rename(columns={
        "model": "Method",
        "mean_gap_h": "Mean miss (h)",
        "point_hits": "Point forecasts inside window",
        "band_overlaps": "Bands overlapping window",
        "band_contains": "Bands covering entire window",
        "median_band_width_h": "Median band width (h)",
    }).round(1), hide_index=True)
    st.caption(
        f"Mean miss is the distance to the nearest edge of the observed crossing window, with zero "
        f"for forecasts inside it. Range counts use the {observed} observed crossings. "
        f"The fitted models' {mass:.0%} bands are model estimates whose transfer was tested here."
    )

    method = st.selectbox("Method for device results", results["model"])
    detail = predictions.loc[
        (predictions["test_study"] == test_study) & (predictions["model"] == method)
    ]
    st.dataframe(detail[[
        "device", "loss_at_origin_pct", "rul_lower_h", "rul_upper_h", "forecast_h",
        "band_lower_h", "band_upper_h",
    ]].rename(columns={
        "device": "Capacitor",
        "loss_at_origin_pct": f"Loss at {origin} h (%)",
        "rul_lower_h": "Last below threshold (h)",
        "rul_upper_h": "First at or above threshold (h)",
        "forecast_h": "Forecast (h)",
        "band_lower_h": "Band start (h)",
        "band_upper_h": "Band end (h)",
    }).round(1), hide_index=True)
    st.caption(
        f"Device times are measured from {origin} aging hours. A blank first crossing means the device "
        "stayed below the threshold through its last observation. The recent trend produces point forecasts only."
    )
    st.download_button("Download all device forecasts", predictions.to_csv(index=False),
                       "caplife_public_benchmark.csv", mime="text/csv")
    with st.expander("How results change with forecast age"):
        sensitivity = pd.DataFrame(benchmark["inspection_age_sensitivity"])
        study_ages = sensitivity.loc[sensitivity["test_study"] == test_study]
        age_columns = ["eos2_age_h", "pofinn_age_h", "observed_crossings"]
        st.write(
            "Mean forecast miss in hours at six pairs of actual inspection ages. Each row uses "
            "devices that were still below the threshold at their forecast inspection."
        )
        st.dataframe(study_ages.pivot(index=age_columns, columns="model", values="mean_gap_h")
                     .reset_index().rename(columns={
                         "eos2_age_h": "EOS-2 age (h)",
                         "pofinn_age_h": "PoFINN age (h)",
                         "observed_crossings": "Observed crossings",
                     }).round(1), hide_index=True)
        st.caption(
            "The same devices contribute at multiple ages. Later rows can contain fewer devices "
            "because earlier crossings have already occurred."
        )
        st.download_button("Download age sensitivity results", sensitivity.to_csv(index=False),
                           "caplife_benchmark_by_forecast_age.csv", mime="text/csv")
    with st.expander("When should a forecast be withheld?"):
        screening = pd.DataFrame(benchmark["forecast_screening_summary"])
        decisions = pd.DataFrame(benchmark["forecast_screening_details"])
        study_screening = screening.loc[screening["test_study"] == test_study]
        st.write(
            "This experiment compares showing every calibrated trend forecast with screening it first. "
            "The history check requires increasing loss across the last three readings and a forecast "
            "range that ends within the span of those readings. The training-range check also requires "
            "the latest loss to fall within the range seen in the training study at the paired inspection."
        )
        st.dataframe(study_screening[[
            "policy", "eligible_origins", "shown", "devices_shown", "scored_crossings",
            "mean_gap_h", "band_overlaps", "band_contains",
        ]].rename(columns={
            "policy": "Policy",
            "eligible_origins": "Forecast opportunities",
            "shown": "Forecasts shown",
            "devices_shown": "Capacitors with forecasts",
            "scored_crossings": "Observed crossings scored",
            "mean_gap_h": "Mean miss (h)",
            "band_overlaps": "Ranges overlapping window",
            "band_contains": "Ranges covering entire window",
        }).round(1), hide_index=True)
        st.caption(
            "The six inspection ages can reuse devices. An overlap means the range could contain the "
            "crossing. Covering the entire window gives stronger evidence. A blank error means no "
            "forecast had an observed crossing to score under that policy."
        )
        stricter = screening.loc[screening["policy"] == "Screen and check training range"]
        shown, scored, overlap, covered = stricter[[
            "shown", "scored_crossings", "band_overlaps", "band_contains",
        ]].sum()
        st.info(
            f"Across both studies, {shown} forecasts passed both checks, with {scored} observed "
            f"crossings to score. {overlap} ranges overlapped and {covered} covered an entire window."
        )
        st.download_button("Download screening decisions", decisions.to_csv(index=False),
                           "caplife_forecast_screening.csv", mime="text/csv")
    with st.expander("Methods and sources"):
        st.write(
            "Age only uses the training study's crossing windows. Recent trend extrapolates the last "
            "three measured capacitance losses in each device. The calibrated trend scales those "
            "estimates using the training study. Both fitted models use the last observation for "
            "devices that finish below the threshold."
        )
        st.write(
            "NASA EOS-2 uses electrical stress. PoFINN uses storage conditions. The studies lack "
            "comparable device-level records of temperature, voltage, and ripple, so this benchmark "
            "cannot estimate their individual effects."
        )
        for study, url in benchmark["sources"].items():
            st.markdown(f"- [{study}]({url})")
    st.stop()

if workflow == MEASUREMENT_WORKFLOW:
    from io import BytesIO
    from caplife.measurement_review import (
        group_capacitor_warnings, nasa_reference_to_caplife, observed_capacitance_threshold,
        read_experimental_csv, reference_age_observations,
    )
    from caplife.real_data import (
        RealDataValidationError,
        assess_real_measurements,
        blank_template_csv,
    )
    from caplife.readiness import (
        audit_measurement_readiness,
        assessment_zip_with_readiness_bytes,
    )

    def clear_public_sample() -> None:
        st.session_state.pop("caplife_public_sample_active", None)

    def load_public_sample() -> None:
        st.session_state["caplife_public_sample_active"] = True

    @st.cache_data(show_spinner=False)
    def public_sample_frame_cached(version: str) -> pd.DataFrame:
        return nasa_reference_to_caplife(pd.read_csv(PUBLIC_SAMPLE, dtype={"Cap_ID": str}))

    def public_sample_frame() -> pd.DataFrame:
        return public_sample_frame_cached(code_version())

    using_sample = st.session_state.get("caplife_public_sample_active", False)
    upload_column, sample_column = st.columns([1.7, 1])
    with upload_column, st.container(border=True, key="caplife_upload_card"):
        st.subheader("Upload measurements")
        upload_real = st.file_uploader(
            "Measurement CSV", type=["csv"], key="caplife_real_csv_v06", on_change=clear_public_sample,
        )
        with st.expander("CSV columns and units"):
            st.write(
                "Include **Cap_ID**, **Hours**, and any of the six parameters you measured: capacitance, ESR, "
                "leakage current, case expansion, generated H₂ and internal pressure. You don't need all six; "
                "leave unavailable values blank. Use the units in the template and add an hour-zero baseline "
                "when available. Record the ESR test frequency and temperature. H₂ volume is cumulative at "
                "0°C and 1 atm. Internal pressure uses absolute kPa."
            )
            st.download_button(
                "Download CSV template", blank_template_csv(),
                "CapLife_measurements_template.csv", mime="text/csv",
            )
        st.caption("Uploads are processed on the app server. Share data you have permission to use.")
    with sample_column, st.container(border=True, key="caplife_sample_card"):
        st.subheader("Explore the NASA example")
        st.write("Follow measured capacitance and ESR (near 10 kHz) in eight capacitors from the ES14 study.")
        if using_sample:
            st.caption("NASA ES14 is loaded.")
            st.button("Clear example", on_click=clear_public_sample)
            st.download_button(
                "Download example CSV", public_sample_frame().to_csv(index=False),
                "nasa_es14_caplife.csv", mime="text/csv",
            )
        else:
            st.button("Load public example", type="primary", on_click=load_public_sample)
        st.markdown("[NASA source ↗](https://data.phmsociety.org/nasa/#12-capacitor-electrical-stress)")
    if not using_sample and upload_real is None:
        st.stop()

    if not using_sample and upload_real.size > 8 * 1024 * 1024:
        st.error("The CSV exceeds the 8 MB limit.")
        st.stop()
    try:
        uploaded_frame = public_sample_frame() if using_sample else read_experimental_csv(upload_real.getvalue())
        real_report = assess_real_measurements(uploaded_frame)
    except (RealDataValidationError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        st.error(f"Upload could not be assessed: {exc}")
        st.info("Correct the source CSV and upload it again. No values were generated to replace missing readings.")
        st.stop()

    st.header("Measurement results")
    st.caption("NASA ES14 example" if using_sample else upload_real.name)

    def show_observed_threshold(capacitance_history: pd.DataFrame) -> None:
        with st.expander("Check an observed capacitance-loss threshold"):
            threshold = st.number_input("Capacitance-loss threshold (%)", min_value=0.1, max_value=99.9,
                                        value=20.0, step=0.1)
            status = observed_capacitance_threshold(capacitance_history, threshold)
            st.dataframe(status.round(2), hide_index=True)
            st.caption(
                "The last below-threshold reading and the first at or above it bracket the first recorded crossing. "
                "Confirmation requires two consecutive available readings at or above the threshold. "
                "A later reading below the threshold reveals a reversal. These are measurements of "
                "capacitance loss. The device's service-life criterion may differ."
            )
            st.download_button("Download measured threshold status", status.to_csv(index=False),
                               "caplife_observed_threshold.csv", mime="text/csv")

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
        for c in ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um", "H2_Generated_mL_STP", "Pressure_Abs_kPa")
    )))
    st.success("Basic data checks passed. This does not establish measurement validity or real-world reliability.")
    for warning in group_capacitor_warnings(warnings):
        st.warning(warning)

    overview, signals, research, exports = st.tabs(["Overview", "Signal history", "Research tools", "Downloads"])
    with overview:
        st.subheader("Measurement quality")
        st.caption("Zero recorded readings means 'not supplied' or 'no measurements' — not a zero sensor reading.")
        st.dataframe(quality, hide_index=True, width="stretch")

        st.subheader("Research-data readiness")
        audit = audit_measurement_readiness(real_report)
        st.caption(
            "This audit flags missing baselines, sparse histories, unrecorded conditions "
            "and measurement gaps. It does not certify sensor accuracy or predict RUL."
        )
        st.dataframe(audit["summary"], hide_index=True, width="stretch")
        with st.expander("Recommended measurement and documentation improvements"):
            st.dataframe(audit["actions"], hide_index=True, width="stretch")
        with st.expander("Recorded operating and test conditions"):
            st.dataframe(audit["conditions"], hide_index=True, width="stretch")

        st.subheader("Observed degradation")
        st.dataframe(changes.round(3), hide_index=True, width="stretch")
        st.caption(
            "Changes are calculated only with an available hour-zero baseline; percentages "
            "refer to the observed raw measurement, not to remaining life. "
            "Case expansion is expressed as an absolute µm difference. "
            "Inconsistent ESR test conditions suppress the ESR ratio."
        )
        capacitance_history = readings[["Cap_ID", "Hours"]].rename(columns={"Hours": "Age_h"})
        capacitance_history["Capacitance_loss_pct"] = 100 * (1 - readings["C_ratio"])
        notes = reference_age_observations(capacitance_history)
        share = notes["median_early_share"]
        if share is not None and share >= 0.5:
            low, high = notes["first_step_loss_range"]
            first_age_low, first_age_high = notes["first_step_age_range"]
            first_age = (f"{first_age_low:,.0f} h" if first_age_low == first_age_high
                         else f"{first_age_low:,.0f}–{first_age_high:,.0f} h")
            st.info(
                f"**Much of the capacitance loss appeared early.** Within the first "
                f"{notes['early_fraction']:.0%} of the recorded time span (about "
                f"{notes['early_cutoff_h']:,.0f} h), the median capacitor had reached "
                f"{share:.0%} of its latest loss. The first reading after the baseline "
                f"({first_age}) already showed {low:.1f}–{high:.1f}% loss. Early threshold "
                "crossings can reflect this initial step, so check the test protocol before "
                "reading them as gradual aging."
            )
        show_observed_threshold(capacitance_history)

    with signals:
        st.subheader("Measured history")
        cap_choice = st.selectbox("Capacitor ID", sorted(readings["Cap_ID"].unique()))
        cap_rows = readings.loc[readings["Cap_ID"] == cap_choice].sort_values("Hours")
        cap_notes = reference_age_observations(capacitance_history.loc[capacitance_history["Cap_ID"].eq(cap_choice)])
        gap = cap_notes["largest_gap"]
        if gap and (cap_notes["largest_gap_fraction_of_span"] or 0) >= 0.2:
            st.caption(
                f"Longest interval without readings: {gap['start_h']:,.1f}–{gap['end_h']:,.1f} h "
                f"({gap['hours']:,.0f} h, {cap_notes['largest_gap_fraction_of_span']:.0%} of this "
                "capacitor's recorded span)."
            )
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
        # Two additional measured channels. Missing channels remain missing;
        # no assumed physics output is passed off as sensor data.
        present_measured_gas = [
            ("H2_Generated_mL_STP", "Measured H₂ generation (mL at 0°C, 1 atm)"),
            ("Pressure_Abs_kPa", "Measured internal absolute pressure (kPa)"),
        ]
        if any(col in cap_rows and cap_rows[col].notna().any() for col, _ in present_measured_gas):
            st.markdown("**Additional directly measured indicators (if recorded)**")
            mleft, mright = st.columns(2)
            for place, (column, label) in zip((mleft, mright), present_measured_gas):
                with place:
                    st.markdown(f"**{label}**")
                    if column in cap_rows and cap_rows[column].notna().any():
                        st.scatter_chart(cap_rows[["Hours", column]].dropna(), x="Hours", y=column, height=240)
                    else:
                        st.caption("No direct measurements supplied for this channel.")
        with st.expander("View processed measurements"):
            st.dataframe(cap_rows, hide_index=True, width="stretch")
        gas_df = None
        scenario_params = None
        # v0.8 PREVIEW: optional assumption-based calculation; never treated as a measurement.

    with research:
        st.subheader("H₂ and internal pressure scenario")
        st.caption(
            "Explore how **assumed** hydrogen production from a fraction of measured leakage "
            "current could change gas quantity and internal pressure over the recorded history. "
            "This is **not** a pressure measurement, vent-opening model, EOL assessment, or RUL prediction."
        )
        show_gas_preview = st.checkbox(
            "Enable assumption-based H₂ / pressure scenario",
            value=False,
            key="caplife_gas_scenario_v08",
        )
        if show_gas_preview:
            from caplife.gas_pressure import (
                GasPressureInputError, calculate_h2_pressure_history,
            )
            with st.expander("Physics assumptions — please review before interpreting", expanded=True):
                st.write(
                    "**Leakage current is not equivalent to hydrogen-generation current.** "
                    "This preview uses a user-selected assumed fraction of leakage current; "
                    "it then assumes a fixed fraction of generated hydrogen stays in the gas space. "
                    "Core_C is used as an assumed gas-temperature proxy. The gas is ideal and the gas space is fixed; the model does not describe "
                    "electrolyte vapor, diffusion, dissolution, venting, swelling, or actual gas chemistry. "
                    "Values shown are a *scenario*, not experimentally validated quantities."
                )
                scenario_left, scenario_right = st.columns(2)
                with scenario_left:
                    assumed_headspace = st.number_input(
                        "Assumed internal free-gas volume (mL)",
                        min_value=0.01, value=5.0, step=0.5,
                        key="caplife_gas_space_ml_v08",
                    )
                    assumed_h2_fraction = st.slider(
                        "Assumed fraction of leakage current producing H₂",
                        0.0, 1.0, 0.10, 0.05,
                        key="caplife_gas_h2_fraction_v08",
                    )
                with scenario_right:
                    assumed_initial_p = st.number_input(
                        "Assumed initial absolute pressure (kPa; NOT gauge)",
                        min_value=0.01, value=101.325, step=1.0,
                        key="caplife_gas_initial_p_v08",
                    )
                    assumed_retained = st.slider(
                        "Assumed fraction of formed H₂ retained in gas space",
                        0.0, 1.0, 0.50, 0.05,
                        key="caplife_gas_retention_v08",
                    )
            try:
                gas_df = calculate_h2_pressure_history(
                    cap_rows,
                    headspace_ml=assumed_headspace,
                    initial_pressure_abs_kpa=assumed_initial_p,
                    h2_current_fraction=assumed_h2_fraction,
                    retained_fraction=assumed_retained,
                )
            except GasPressureInputError as exc:
                st.info(f"Gas/pressure scenario unavailable for this capacitor: {exc}")
                st.caption(
                    "For this first preview, the selected capacitor must have a 0-hour reading "
                    "and complete Leakage_uA and Core_C readings at each recorded time. "
                    "No missing values are invented or interpolated."
                )
            else:
                scenario_params = {
                    "headspace_ml": float(assumed_headspace),
                    "initial_pressure_abs_kpa": float(assumed_initial_p),
                    "h2_current_fraction": float(assumed_h2_fraction),
                    "retained_fraction": float(assumed_retained),
                    "reference_H2_volume": "0 C, 1 atm",
                    "input_provenance": "Measured leakage-current history; estimated hydrogen and pressure",
                    "model_status": "UNVALIDATED hypothetical physics scenario; NOT directly measured",
                }
                last = gas_df.iloc[-1]
                indicators = st.columns(3)
                indicators[0].metric(
                    "Calculated H₂ formed (STP equivalent)",
                    f"{last['H2_Generated_mL_STP_est']:.3g} mL",
                )
                indicators[1].metric(
                    "Calculated pressure (absolute)",
                    f"{last['Pressure_Abs_kPa_est']:.1f} kPa",
                )
                indicators[2].metric(
                    "Calculated pressure change from initial",
                    f"{last['Pressure_Change_From_Initial_kPa_est']:+.1f} kPa",
                )
                st.caption(
                    "H₂ volume is expressed at 0°C and 1 atm. The pressure model uses the "
                    "assumed physical headspace, not the STP gas volume. Thermal pressure "
                    "changes of the initial gas are included. Positive pressure change is "
                    "not necessarily only caused by H₂."
                )
                chart_left, chart_right = st.columns(2)
                with chart_left:
                    st.markdown("**Calculated H₂ formed and retained over time (mmol)**")
                    st.line_chart(
                        gas_df.set_index("Hours")[[
                            "H2_Generated_mmol_est", "H2_Retained_mmol_est"
                        ]], height=260,
                    )
                with chart_right:
                    st.markdown("**Calculated internal pressure over time (absolute kPa)**")
                    st.line_chart(
                        gas_df.set_index("Hours")["Pressure_Abs_kPa_est"],
                        height=260,
                    )
                with st.expander("Inspect calculation table and equations"):
                    st.dataframe(gas_df, hide_index=True, width="stretch")
                    st.latex(r"Q(t) = \int_0^t I_{\rm leakage}(\tau) \, d\tau")
                    st.latex(r"n_{H_2,\rm generated}(t) = \eta \frac{Q(t)}{2F}")
                    st.latex(r"P_{\rm abs}(t) = P_0\frac{T(t)}{T_0} + \frac{n_{H_2,\rm retained}(t) R T(t)}{V_{\rm headspace}}")
                    st.caption(
                        "The integral uses trapezoidal approximation between observed leakage "
                        "readings. η and retention are assumptions, not inferred physical constants."
                    )
                st.info(
                    "This is an **assumption-based estimate**, not measured H₂/pressure. "
                    "When you download a report, the scenario will be included in separately "
                    "labeled columns or sheets/pages, never merged into raw measurements."
                )

        # v0.9: descriptive rates from six direct measurement channels only.
        # This module NEVER consumes hypothetical H2 / pressure scenario outputs.
        from caplife.degradation_rates import calculate_degradation_rates
        st.subheader("Measured degradation rates")
        st.caption(
            "Rate = observed change divided by elapsed hours. Acceleration = change in "
            "successive interval rates divided by elapsed midpoint time. Both use "
            "measured points only: missing readings are never filled or bridged. "
            "No health score, failure threshold, RUL, or forecasting is calculated."
        )
        rate_report = calculate_degradation_rates(readings)
        rate_summary = rate_report["summary"].loc[
            lambda df: df["Cap_ID"] == cap_choice
        ]
        # Display very small accelerations in scientific notation without
        # incorrectly rounding a nonzero value to zero.
        rate_summary_display = rate_summary.copy()
        for numeric_column in (
            "Time_weighted_rate_per_hour", "Latest_rate_per_hour",
            "Latest_acceleration_per_hour2",
        ):
            rate_summary_display[numeric_column] = rate_summary_display[numeric_column].map(
                lambda value: "—" if pd.isna(value) else f"{value:+.5g}"
            )
        st.dataframe(rate_summary_display, hide_index=True, width="stretch")
        for warning in rate_report["warnings"]:
            if str(cap_choice) in warning:
                st.warning(warning)
        with st.expander("View interval-by-interval rates for all six indicators", expanded=False):
            rate_rows = rate_report["intervals"].loc[
                lambda df: df["Cap_ID"] == cap_choice
            ]
            two_cols = st.columns(2)
            for chart_idx, (_, display_name, unit) in enumerate((
                ("Capacitance_uF", "Capacitance", "µF"),
                ("ESR_ohm", "ESR", "Ω"),
                ("Leakage_uA", "Leakage current", "µA"),
                ("Case_Expansion_um", "Case expansion", "µm"),
                ("H2_Generated_mL_STP", "H₂ generated (directly measured)", "mL STP"),
                ("Pressure_Abs_kPa", "Internal absolute pressure (directly measured)", "kPa absolute"),
            )):
                with two_cols[chart_idx % 2]:
                    st.markdown(f"**{display_name}: {unit}/h**")
                    present = rate_rows.loc[rate_rows["Indicator"] == display_name]
                    if present.empty:
                        st.caption("Insufficient adjacent comparable measurements.")
                    else:
                        st.scatter_chart(
                            present[["Midpoint_Hour", "Rate_per_hour"]],
                            x="Midpoint_Hour", y="Rate_per_hour", height=220,
                        )
            st.caption(
                "Dots represent rates between adjacent recorded timestamps, plotted at "
                "interval midpoints. They are NOT continuously measured rates. "
                "Each graph uses its own units and scale."
            )
            st.dataframe(rate_rows.round(10), hide_index=True, width="stretch")
        with st.expander("Inspect rate acceleration calculations", expanded=False):
            accel_rows = rate_report["accelerations"].loc[
                lambda df: df["Cap_ID"] == cap_choice
            ]
            st.caption(
                "Acceleration here means a change in *measurement rate*, not mechanical "
                "acceleration. It requires three consecutive nonmissing readings "
                "and has units of indicator per hour squared. Negative capacitance "
                "acceleration can indicate faster loss, but neither sign is a health verdict."
            )
            if accel_rows.empty:
                st.info("No comparable three-point sequences are available for this capacitor.")
            else:
                chosen_indicator = st.selectbox(
                    "Indicator for rate-acceleration chart",
                    sorted(accel_rows["Indicator"].unique()),
                    key="caplife_v09_accel_indicator",
                )
                accel_subset = accel_rows.loc[accel_rows["Indicator"] == chosen_indicator]
                st.scatter_chart(
                    accel_subset[["At_Hours", "Acceleration_per_hour2"]],
                    x="At_Hours", y="Acceleration_per_hour2", height=250,
                )
                st.dataframe(accel_rows.round(12), hide_index=True, width="stretch")
        st.download_button(
            "Download observed interval-rate table (CSV)",
            data=rate_report["intervals"].to_csv(index=False).encode("utf-8-sig"),
            file_name="caplife_interval_rates.csv", mime="text/csv",
            key="caplife_v09_rate_csv",
        )
        st.caption(
            "Rates and changes may reflect ambient/core temperature, voltage, load, "
            "measurement methods, or test conditions—not only aging. A slope sign "
            "alone is not a failure or health classification."
        )
        st.caption(
            "The Excel and PDF downloads also include these descriptive rates. "
            "CSV measured histories remain unchanged; the separate rate CSV is above. "
            "Rates have not been validated for health inference."
        )

    with exports:
        st.subheader("Download your assessment")
        from caplife.report_exports import csv_report_bytes, csv_measurement_history_bytes, xlsx_report_bytes, pdf_report_bytes

        chosen_format = st.radio(
            "Choose one file format",
            ("CSV (.csv)", "Excel (.xlsx)", "PDF (.pdf)"),
            horizontal=True,
            index=1,
            key="caplife_report_format_v07",
        )
        if chosen_format == "CSV (.csv)":
            csv_content = st.radio(
                "What should the CSV contain?",
                ("Complete measured history (all capacitors)", "Per-capacitor assessment summary"),
                horizontal=True,
                key="caplife_csv_content_v08",
            )
            if csv_content.startswith("Complete measured history"):
                st.caption(
                    "Every recorded measurement row, including all available six indicators "
                    "and their baseline-normalized ratios. Physics scenario estimates are "
                    "never mixed into the measured history."
                )
                output_bytes = csv_measurement_history_bytes(real_report)
                export_basename = "caplife_measured_history"
            else:
                st.caption(
                    "One row per capacitor with observed changes. If enabled, assumption-based "
                    "gas/pressure results appear in clearly named estimate columns."
                )
                output_bytes = csv_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params)
                export_basename = "caplife_assessment_summary"
            extension = "csv"
            mime_type = "text/csv"
        elif chosen_format == "Excel (.xlsx)":
            st.caption("Includes measured histories, six-indicator rate/acceleration tables and separate charts; optional physics estimates remain separately labeled.")
            output_bytes = xlsx_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params, rate_report=rate_report)
            export_basename = "caplife_assessment"
            extension = "xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            st.caption("Includes descriptive rate and acceleration appendices for the selected capacitor, plus optional separately labeled physics scenario.")
            output_bytes = pdf_report_bytes(real_report, audit, cap_choice, gas_df, scenario_params, rate_report=rate_report)
            export_basename = "caplife_assessment"
            extension = "pdf"
            mime_type = "application/pdf"
        st.download_button(
            f"Download {chosen_format} assessment",
            data=output_bytes,
            file_name=f"{export_basename}.{extension}",
            mime=mime_type,
            type="primary",
        )
        with st.expander("Advanced: download every assessment table as separate CSV files (ZIP)"):
            original_zip = assessment_zip_with_readiness_bytes(real_report, audit)
            if gas_df is not None:
                # Add explicitly named assumption-based files without altering the
                # existing measured and audit CSVs or inventing readings.
                from zipfile import ZIP_DEFLATED, ZipFile
                with BytesIO() as new_zip_buf:
                    with ZipFile(BytesIO(original_zip), "r") as original, ZipFile(new_zip_buf, "w", ZIP_DEFLATED) as archive:
                        for filename in original.namelist():
                            archive.writestr(filename, original.read(filename))
                        archive.writestr("hypothetical_H2_pressure_scenario_estimates.csv", gas_df.to_csv(index=False))
                        archive.writestr("hypothetical_H2_pressure_scenario_assumptions.json", json.dumps(scenario_params, indent=2))
                    original_zip = new_zip_buf.getvalue()
            st.download_button(
                "Download complete CSV package (ZIP)",
                data=original_zip,
                file_name="caplife_assessment_tables.zip",
                mime="application/zip",
            )
        st.info(
            "**Limitations:** This mode does not determine health, EOL, RUL, uncertainty "
            "intervals, or maintenance actions. A complete dataset can still have inaccurate "
            "sensors, changing operating conditions, or incomparable test settings. "
            "The uploaded file is processed by the web-app server; check hosting/privacy "
            "policies before submitting confidential experiments."
        )
        st.caption("CapLife AI · six-indicator observation plus optional hypothetical physics · no real-data neural-network prediction")

    st.stop()

# Previous synthetic RUL page is intentionally preserved below this line.


@st.cache_resource(show_spinner="Loading the demonstration neural network…")
def get_engine(version: str = ""):
    return load_engine(ARTIFACTS)


try:
    net, metadata = get_engine(code_version())
except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError):
    st.error("Model files are missing or incompatible. See README.md for training instructions.")
    st.code("python -m caplife.train --root .", language="bash")
    st.stop()

@st.cache_data(show_spinner=False)
def load_synthetic_calibration(version: str = ""):
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


synthetic_calibration, calibration_issue = load_synthetic_calibration(code_version())


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
    st.caption("Synthetic-only trend intervals; not valid for real AEC measurements")
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
        width="stretch",
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
        value=default_prediction_hour(cap_df, time_options),
        format_func=lambda hour: f"{hour:,.0f} hours",
    )

# This historical slice is used BOTH for inference and for the visible charts.
# No measurements after the selected time are provided to the neural network.
history_to_show = cap_df.loc[cap_df["Hours"] <= prediction_hour].copy()
st.caption(
    f"Prediction will use {len(history_to_show)} readings up to "
    f"{prediction_hour:,.0f} hours. Future measurements are excluded."
)

if st.button("Analyze capacitor", type="primary", width="stretch"):
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
        st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
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
    st.dataframe(history_to_show, width="stretch", hide_index=True)
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
    st.dataframe(cap_df, width="stretch", hide_index=True)
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

if st.button("Analyze RUL across the available history", width="stretch"):
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
        st.dataframe(trajectory, width="stretch", hide_index=True)
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
    "CapLife AI synthetic demo · GRU + illustrative multi-indicator trend + "
    "synthetic-only trend intervals · No calibrated uncertainty for real capacitors"
)
