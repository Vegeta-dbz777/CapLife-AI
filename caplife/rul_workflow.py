"""Train, test and apply a six-indicator neural remaining-life model on a user's data.

The workflow answers one question before it forecasts anything: does a neural
network trained on some of the user's capacitors predict the documented
endpoint of capacitors it never saw? Capacitors are split into training,
calibration and test groups without looking at their outcomes. The training
group fits the models, the calibration group sets the width of the 80% range,
and the test group scores them. Forecasts for capacitors without an outcome
row are released only when the neural model passes every test check, and they
use exactly the model and range that were tested.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import math

import numpy as np
import pandas as pd

from caplife.real_data import MEASUREMENTS, RealDataValidationError, validate_real_measurements
from caplife.remaining_time_validation import BOUNDS
from caplife.six_indicator_validation import (
    FEATURES, INTERVAL_MASS, OUTCOME_COLUMNS, SIGNALS, calibrated_band, fit_candidates,
    joined_landmarks, landmark_features, model_inputs,
)

SPLIT_BY_EXPERIMENT = "experiment"
SPLIT_BY_DEVICE = "device"
ROLES = ("training", "calibration", "test")
MODEL_LABELS = {
    "neural": "Neural network",
    "log_linear": "Log-linear model",
    "intercept_only": "Average lifetime only",
}

# Smallest calibration group for which an 80% split-conformal radius is finite:
# ceil((n + 1) * 0.8) <= n first holds at n = 4.
MIN_CALIBRATION_DEVICES = 4
MIN_TRAINING_DEVICES = 5
MIN_TEST_DEVICES = 4
# Release checks for the neural model on the held-out test group.
MIN_TEST_FAILURES = 3
MIN_TEST_COVERAGE = 0.7
MAX_CENSOR_CONTRADICTIONS = 0.2
# Forecast inputs may exceed the training minimum or maximum by at most this
# fraction of the training range before the forecast is withheld.
TRAINING_RANGE_MARGIN = 0.10
FEATURE_LABELS = {
    **{f"{signal}_change": f"{MEASUREMENTS[signal][0].lower()} change" for signal in SIGNALS},
    **{f"{signal}_rate_per_h": f"{MEASUREMENTS[signal][0].lower()} rate" for signal in SIGNALS},
}

MEASUREMENT_TEMPLATE_COLUMNS = ("Cap_ID", "Hours", *SIGNALS, "ESR_Test_Hz", "ESR_Test_Temp_C")


def measurement_template_csv() -> str:
    return pd.DataFrame(columns=MEASUREMENT_TEMPLATE_COLUMNS).to_csv(index=False)


def outcome_template_csv() -> str:
    return pd.DataFrame(columns=OUTCOME_COLUMNS).to_csv(index=False)


def read_outcomes(payload: bytes) -> pd.DataFrame:
    """Read outcome rows as literal text so IDs and blank failure ages are preserved."""
    try:
        frame = pd.read_csv(BytesIO(payload), keep_default_na=False, encoding="utf-8-sig",
                            dtype={"Cap_ID": str, "Experiment_ID": str, "Endpoint": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise RealDataValidationError(f"The outcome CSV could not be read. {error}") from error
    if not isinstance(frame.index, pd.RangeIndex):
        raise RealDataValidationError("Outcome rows must match the header column count.")
    return frame


def device_roles(frame: pd.DataFrame, method: str, calibration_experiment: str | None = None,
                 test_experiment: str | None = None) -> pd.Series:
    """Assign each eligible capacitor to training, calibration or test without using outcomes."""
    if method == SPLIT_BY_EXPERIMENT:
        experiments = set(frame["experiment"])
        if calibration_experiment not in experiments or test_experiment not in experiments:
            raise ValueError("Choose calibration and test experiments that contain eligible capacitors.")
        if calibration_experiment == test_experiment:
            raise ValueError("Use different experiments for calibration and testing.")
        roles = np.select(
            [frame["experiment"].eq(calibration_experiment), frame["experiment"].eq(test_experiment)],
            ["calibration", "test"], "training",
        )
        return pd.Series(roles, index=frame["device"].to_numpy(), name="role")
    if method != SPLIT_BY_DEVICE:
        raise ValueError(f"Unknown split method: {method}")
    # A hash of the ID gives a fixed, outcome-blind order that does not depend on row order.
    order = sorted(frame["device"], key=lambda device: hashlib.sha256(device.encode("utf-8")).hexdigest())
    n = len(order)
    n_calibration = max(MIN_CALIBRATION_DEVICES, round(0.2 * n))
    n_test = max(MIN_TEST_DEVICES, round(0.2 * n))
    if n - n_calibration - n_test < MIN_TRAINING_DEVICES:
        needed = MIN_CALIBRATION_DEVICES + MIN_TEST_DEVICES + MIN_TRAINING_DEVICES
        raise ValueError(f"Splitting by capacitor needs at least {needed} eligible capacitors; {n} are eligible.")
    roles = ["calibration"] * n_calibration + ["test"] * n_test + ["training"] * (n - n_calibration - n_test)
    return pd.Series(roles, index=order, name="role").reindex(frame["device"].to_numpy())


def release_checks(fitted: dict, calibration: pd.DataFrame) -> list[str]:
    """Reasons the neural forecasts must be withheld; empty when every check passes."""
    neural, baseline = fitted["neural"], fitted["intercept_only"]
    if neural["radius"] is None:
        still_working = int((~np.isfinite(calibration[BOUNDS[1]].to_numpy(dtype=float))).sum())
        return [f"The {len(calibration)} calibration capacitors could not set a finite 80% range. "
                f"{still_working} of them were still working at their last check, so their failure time "
                "has no upper limit, and the conservative calibration needs nearly all of them to have failed. "
                "Add capacitors with observed failures, or calibrate on an experiment with more failures."]
    scores = neural["scores"]["test_calibrated_interval"]
    reasons = []
    failures = scores["bounded_or_exact_events"]
    if failures < MIN_TEST_FAILURES:
        reasons.append(f"Only {failures} test capacitor(s) have an observed failure; "
                       f"at least {MIN_TEST_FAILURES} are needed to score the predictions.")
    else:
        coverage = scores["definite_80pct_band_coverage_observed_events"]
        if coverage < MIN_TEST_COVERAGE:
            reasons.append(f"The 80% range contained {coverage:.0%} of observed test failures; "
                           f"at least {MIN_TEST_COVERAGE:.0%} is required.")
        reference = baseline["scores"]["test_calibrated_interval"] or baseline["scores"]["test_model_interval"]
        neural_miss = scores["mean_distance_to_observed_window_h"]
        baseline_miss = reference["mean_distance_to_observed_window_h"]
        if baseline_miss is not None and neural_miss > baseline_miss:
            reasons.append(f"The neural network missed test failures by {neural_miss:,.0f} h on average, "
                           f"more than the {baseline_miss:,.0f} h of simply predicting the average lifetime.")
    censored = scores["right_censored_origins"]
    if censored and scores["band_below_right_censor_count"] / censored > MAX_CENSOR_CONTRADICTIONS:
        reasons.append(f"For {scores['band_below_right_censor_count']} of {censored} test capacitors that were "
                       "still working at their last check, the whole 80% range ended before that check.")
    return reasons


def _test_table(test: pd.DataFrame, neural: dict) -> pd.DataFrame:
    x = model_inputs(test, neural["features"])
    if neural["radius"] is not None:
        band = calibrated_band(neural["model"], neural["radius"], x)
    else:
        band = neural["model"].quantiles(x, (0.1, 0.5, 0.9))
    age = test["age_h"].to_numpy(dtype=float)
    lower, upper = (test[column].to_numpy(dtype=float) + age for column in BOUNDS)
    low, mid, high = (age + band[:, i] for i in range(3))
    observed = np.isfinite(upper)
    result = np.where(
        observed,
        np.where((low <= lower) & (high >= upper), "Inside range",
                 np.where((high >= lower) & (low <= upper), "Partly inside", "Outside range")),
        np.where(high >= lower, "Consistent (still working)", "Range ended too early"),
    )
    return pd.DataFrame({
        "Capacitor": test["device"].to_numpy(),
        "Experiment": test["experiment"].to_numpy(),
        "Forecast made at (h)": age,
        "Predicted failure age (h)": mid,
        "80% range from (h)": low,
        "80% range to (h)": high,
        "Last seen working (h)": lower,
        "Found failed by (h)": np.where(observed, upper, np.nan),
        "Result": result,
    })


def _forecasts(raw: pd.DataFrame, neural: dict, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    readings = validate_real_measurements(raw)
    lows, highs = train.loc[:, list(FEATURES)].min(), train.loc[:, list(FEATURES)].max()
    rows, withheld = [], {}
    for device, history in readings.groupby("Cap_ID", sort=True):
        landmark = landmark_features(history)
        if isinstance(landmark, str):
            withheld[device] = landmark
            continue
        age, values = landmark
        span = (highs - lows).replace(0, 1)
        beyond = (np.maximum(values - highs, 0) + np.maximum(lows - values, 0)) / span
        if (beyond > TRAINING_RANGE_MARGIN).any():
            worst = beyond.idxmax()
            withheld[device] = (f"Early readings fall outside what the training capacitors showed "
                                f"({FEATURE_LABELS[worst]} is {beyond[worst]:.0%} of the training range beyond it)")
            continue
        low, mid, high = age + calibrated_band(neural["model"], neural["radius"], values[None, :])[0]
        latest = float(history["Hours"].max())
        outlived = latest > high
        rows.append({
            "Capacitor": device,
            "Forecast made at (h)": age,
            "Latest reading (h)": latest,
            "Predicted failure age (h)": mid,
            "80% range from (h)": low,
            "80% range to (h)": high,
            "Remaining from latest, median (h)": max(mid - latest, 0.0) if latest < mid else np.nan,
            "Remaining from latest, 80% range from (h)": np.nan if outlived else max(low - latest, 0.0),
            "Remaining from latest, 80% range to (h)": np.nan if outlived else high - latest,
            "Status": ("Outlived the predicted range" if outlived
                       else "Past the median prediction" if latest >= mid else "Forecast"),
        })
    columns = ["Capacitor", "Forecast made at (h)", "Latest reading (h)", "Predicted failure age (h)",
               "80% range from (h)", "80% range to (h)", "Remaining from latest, median (h)",
               "Remaining from latest, 80% range from (h)", "Remaining from latest, 80% range to (h)", "Status"]
    return pd.DataFrame(rows, columns=columns), withheld


def _model_table(fitted: dict) -> pd.DataFrame:
    rows = []
    for name in ("neural", "log_linear", "intercept_only"):
        calibrated = fitted[name]["scores"]["test_calibrated_interval"]
        scores = calibrated or fitted[name]["scores"]["test_model_interval"]
        rows.append({
            "Model": MODEL_LABELS[name],
            "Average miss (h)": scores["mean_distance_to_observed_window_h"],
            "Failures inside 80% range": scores["definite_80pct_band_coverage_observed_events"],
            "Typical 80% range width (h)": scores["median_band_width_h"],
            "Range": "Calibrated" if calibrated else "Model's own (uncalibrated)",
        })
    return pd.DataFrame(rows)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if value is pd.NA:
        return None
    return value


def run_rul_workflow(measurements: pd.DataFrame, outcomes: pd.DataFrame, method: str,
                     calibration_experiment: str | None = None,
                     test_experiment: str | None = None) -> dict:
    """Validate a neural remaining-life model on held-out capacitors and forecast only if it passes."""
    missing = [signal for signal in SIGNALS if signal not in measurements.columns]
    if missing:
        names = ", ".join(MEASUREMENTS[signal][0] for signal in missing)
        raise RealDataValidationError(f"The measurement CSV needs all six indicators. Missing: {names}.")
    missing = [column for column in OUTCOME_COLUMNS if column not in outcomes.columns]
    if missing or outcomes.empty:
        raise RealDataValidationError("The outcome CSV needs one row per capacitor with columns "
                                      + ", ".join(OUTCOME_COLUMNS) + ".")
    outcomes = outcomes.loc[:, list(OUTCOME_COLUMNS)]
    raw_ids = validate_real_measurements(measurements)["Cap_ID"]
    measured_ids = set(raw_ids)
    labelled_ids = outcomes["Cap_ID"].astype("string").str.strip()
    unknown = sorted(set(labelled_ids.dropna()) - measured_ids)
    if unknown:
        shown = ", ".join(unknown[:5]) + (" …" if len(unknown) > 5 else "")
        raise RealDataValidationError(f"Outcome rows name capacitors with no measurements: {shown}")

    measurement_ids = measurements["Cap_ID"].astype("string").str.strip()
    labelled = measurement_ids.isin(set(labelled_ids))
    frame, audit = joined_landmarks(measurements.loc[labelled].reset_index(drop=True), outcomes)
    if frame.empty:
        raise ValueError("No capacitor has the readings and follow-up needed to train a model.")
    roles = device_roles(frame, method, calibration_experiment, test_experiment)
    frame = frame.assign(role=roles.reindex(frame["device"].to_numpy()).to_numpy())
    parts = {role: frame.loc[frame["role"].eq(role)] for role in ROLES}
    minimum = {"training": MIN_TRAINING_DEVICES, "calibration": MIN_CALIBRATION_DEVICES, "test": MIN_TEST_DEVICES}
    for role, part in parts.items():
        if len(part) < minimum[role]:
            raise ValueError(f"The {role} group has {len(part)} eligible capacitor(s); at least {minimum[role]} are needed.")
    if not np.isfinite(parts["training"][BOUNDS[1]]).any():
        raise ValueError("At least one training capacitor needs an observed failure.")

    fitted = fit_candidates(parts["training"], parts["calibration"], parts["test"])
    reasons = release_checks(fitted, parts["calibration"])
    forecast_devices = sorted(measured_ids - set(labelled_ids))
    forecasts, withheld = None, {}
    if not reasons and forecast_devices:
        forecasts, withheld = _forecasts(measurements.loc[~labelled].reset_index(drop=True),
                                         fitted["neural"], parts["training"])

    report = {
        **audit,
        "split_method": method,
        "experiments_by_role": {role: sorted(map(str, part["experiment"].unique())) for role, part in parts.items()},
        "devices_by_role": {role: int(len(part)) for role, part in parts.items()},
        "target_interval_mass": INTERVAL_MASS,
        "release_checks": {
            "min_test_failures": MIN_TEST_FAILURES, "min_test_coverage": MIN_TEST_COVERAGE,
            "max_censor_contradictions": MAX_CENSOR_CONTRADICTIONS,
        },
        "passed": not reasons,
        "withheld_reasons": reasons,
        "models": {name: item["scores"] for name, item in fitted.items()},
        "forecast_devices": forecast_devices,
        "forecasts": [] if forecasts is None else forecasts.to_dict(orient="records"),
        "forecasts_withheld": withheld,
    }
    return {
        "report": _json_safe(report),
        "passed": not reasons,
        "reasons": reasons,
        "audit": audit,
        "roles": {role: len(part) for role, part in parts.items()},
        "experiments_by_role": report["experiments_by_role"],
        "model_table": _model_table(fitted),
        "test_table": _test_table(parts["test"], fitted["neural"]),
        "forecast_devices": forecast_devices,
        "forecasts": forecasts,
        "forecasts_withheld": withheld,
    }
