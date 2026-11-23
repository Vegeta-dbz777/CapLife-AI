"""Train, test and apply a neural remaining-life model on a user's own capacitors.

The workflow answers one question before it forecasts anything: does a neural
network trained on some of the user's capacitors predict the documented
endpoint of capacitors it never saw? Capacitors are split into training,
calibration and test groups without looking at their outcomes. The training
group fits the models, the calibration group sets the width of the 80% range,
and the test group scores them. Forecasts for capacitors without an outcome
row are released only when the neural model passes every test check, and they
use exactly the model and range that were tested.

Every reading from a capacitor's third onward is a "landmark": the changes in
each parameter since the first reading, their recent rates and the time since
the first reading, with the remaining time to the endpoint as the target. All
landmarks train the models, weighted so every capacitor counts equally.
Calibration and testing use one landmark per capacitor, chosen from its ID and
not its outcome, so their checks stay independent. Forecasts are made at each
capacitor's latest reading. The parameters used are those every capacitor with
an outcome has measured at least three times, from one to all six.
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
    INTERVAL_MASS, OUTCOME_COLUMNS, SIGNALS, calibrate_log_radius, calibrated_band, model_inputs,
)
from caplife.remaining_time_validation import QUANTILES, duration_scores
from caplife.survival import fit_duration

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
    "age_since_first_h": "time since the first reading",
}
MIN_READINGS = 3  # a landmark needs the first reading and two more


def features_for(signals: tuple[str, ...]) -> tuple[str, ...]:
    return (*(f"{signal}_change" for signal in signals), *(f"{signal}_rate_per_h" for signal in signals),
            "age_since_first_h")


def signals_for(readings: pd.DataFrame, devices) -> tuple[str, ...]:
    """Parameters that every one of these capacitors has measured at least three times."""
    devices = list(devices)
    chosen = []
    for signal in SIGNALS:
        if signal not in readings:
            continue
        counts = readings.loc[readings["Cap_ID"].isin(devices)].groupby("Cap_ID")[signal].count()
        if len(devices) and counts.reindex(devices, fill_value=0).ge(MIN_READINGS).all():
            chosen.append(signal)
    return tuple(chosen)


def landmarks(history: pd.DataFrame, signals: tuple[str, ...]) -> list[tuple[float, np.ndarray]] | str:
    """Features at every reading from the third onward, each from readings up to it; or why there are none.

    Changes are measured from the first reading with every parameter (capacitance as % loss, ESR as % rise,
    the others in their own units); rates are the slope over the last three readings.
    """
    complete = history.dropna(subset=list(signals)).sort_values("Hours")
    if len(complete) < MIN_READINGS:
        return f"Needs at least {MIN_READINGS} readings with every parameter used"
    hours = complete["Hours"].to_numpy(dtype=float)
    # Changed ESR test conditions make later ESR readings incomparable: stop the landmarks there.
    if "ESR_ohm" in signals:
        for column in ("ESR_Test_Hz", "ESR_Test_Temp_C"):
            if column in complete and complete[column].notna().any():
                values = complete[column].to_numpy()
                first = values[~pd.isna(values)][0]
                changed = np.flatnonzero(~pd.isna(values) & (values != first))
                if changed.size:
                    hours = hours[:changed[0]]
        if len(hours) < MIN_READINGS:
            return "Recorded ESR test conditions change before the third reading"
    measured = complete.loc[:, list(signals)].to_numpy(dtype=float)[:len(hours)]
    change = measured - measured[0]
    for index, signal in enumerate(signals):
        if signal == "Capacitance_uF":
            change[:, index] *= -100 / measured[0, index]
        elif signal == "ESR_ohm":
            change[:, index] *= 100 / measured[0, index]
    found = []
    for k in range(MIN_READINGS - 1, len(hours)):
        recent = slice(k - 2, k + 1)
        rate = np.polyfit(hours[recent] - hours[k], change[recent], 1)[0]
        found.append((float(hours[k]), np.concatenate([change[k], np.atleast_1d(rate), [hours[k] - hours[0]]])))
    return found


def _pick(device: str, count: int) -> int:
    """One landmark per capacitor for calibration and testing, fixed by its ID, not its outcome."""
    return int(hashlib.sha256(f"landmark:{device}".encode("utf-8")).hexdigest(), 16) % count


def _labels(outcomes: pd.DataFrame) -> pd.DataFrame:
    labels = outcomes.loc[:, list(OUTCOME_COLUMNS)].copy()
    for column in ("Cap_ID", "Experiment_ID", "Endpoint"):
        labels[column] = labels[column].astype("string").str.strip()
        if labels[column].isna().any() or labels[column].eq("").any():
            raise ValueError(f"{column} must be recorded for every outcome.")
    if labels["Cap_ID"].duplicated().any():
        raise ValueError("Include exactly one outcome row per capacitor.")
    if labels["Endpoint"].nunique() != 1:
        raise ValueError("All experiments must use the same documented endpoint.")
    for column in ("Last_alive_h", "Failure_by_h"):
        text = labels[column].astype("string").str.strip().replace("", pd.NA)
        try:
            labels[column] = pd.to_numeric(text, errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{column} must be a number of hours (blank Failure_by_h means still working).") from error
    lo, hi = labels["Last_alive_h"], labels["Failure_by_h"].fillna(np.inf)
    if not np.isfinite(lo).all() or (lo < 0).any() or (hi < lo).any():
        raise ValueError("Invalid outcome ages: Last_alive_h must be at most Failure_by_h. A blank Failure_by_h "
                         "means the capacitor was still working.")
    return labels


def all_landmarks(readings: pd.DataFrame, labels: pd.DataFrame, signals: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """Every usable landmark of every capacitor with an outcome, with its remaining-time bounds."""
    features = features_for(signals)
    by_device = {device: group for device, group in readings.groupby("Cap_ID", sort=False)}
    rows, dropped = [], {}
    for label in labels.itertuples(index=False):
        found = landmarks(by_device[label.Cap_ID], signals)
        if isinstance(found, str):
            dropped[label.Cap_ID] = found
            continue
        # The evaluation landmark is chosen from the readings and the ID alone, never from the outcome, and
        # never the last reading (often the check that found the capacitor failed).
        candidates = found[:-1] or found
        evaluation_age = candidates[_pick(label.Cap_ID, len(candidates))][0]
        usable = [(age, values) for age, values in found if age < label.Last_alive_h]
        if evaluation_age >= label.Last_alive_h:
            dropped[label.Cap_ID] = "Insufficient documented follow-up after its evaluation reading"
            continue
        upper = math.inf if pd.isna(label.Failure_by_h) else float(label.Failure_by_h)
        for age, values in usable:
            rows.append({
                "device": label.Cap_ID, "experiment": label.Experiment_ID, "age_h": age,
                BOUNDS[0]: float(label.Last_alive_h) - age, BOUNDS[1]: upper - age,
                "evaluation": age == evaluation_age, "weight": 1 / len(usable),
                **dict(zip(features, values, strict=True)),
            })
    frame = pd.DataFrame(rows, columns=["device", "experiment", "age_h", *BOUNDS, "evaluation", "weight", *features])
    return frame, {
        "endpoint": str(labels["Endpoint"].iloc[0]),
        "devices_submitted": len(labels),
        "devices_excluded": dropped,
        "experiments": sorted(str(value) for value in labels["Experiment_ID"].unique()),
        "parameters": list(signals),
    }


def fit_candidates(train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame,
                   features: tuple[str, ...]) -> dict:
    """Fit each candidate on every training landmark, calibrate it, then score the test capacitors."""
    fitted = {}
    for name, used, hidden in (("intercept_only", (), False), ("log_linear", features, False),
                               ("neural", features, True)):
        x_train, x_cal, x_test = (model_inputs(part, used) for part in (train, calibration, test))
        model = fit_duration(x_train, train[BOUNDS[0]], train[BOUNDS[1]], train["weight"].to_numpy(dtype=float), hidden)
        radius = calibrate_log_radius(calibration, model.location(x_cal))
        coverage = None if radius is None else duration_scores(test, calibrated_band(model, radius, x_test))
        fitted[name] = {
            "model": model, "features": used, "radius": radius,
            "scores": {
                "test_model_interval": duration_scores(test, model.quantiles(x_test, QUANTILES)),
                "test_calibrated_interval": coverage,
                "calibration_log_radius": radius,
            },
        }
    return fitted

MEASUREMENT_TEMPLATE_COLUMNS = ("Cap_ID", "Hours", *SIGNALS, "ESR_Test_Hz", "ESR_Test_Temp_C")


def measurement_template_csv() -> str:
    return pd.DataFrame(columns=MEASUREMENT_TEMPLATE_COLUMNS).to_csv(index=False)


def outcome_template_csv() -> str:
    return pd.DataFrame(columns=OUTCOME_COLUMNS).to_csv(index=False)


def read_outcomes(payload: bytes, time_unit: str = "hours") -> pd.DataFrame:
    """Read outcome rows (CSV or Excel) as literal text so IDs and blank failure ages are preserved.

    Decimal commas in the two ages become decimal points, and ages in days, weeks or months are
    converted to hours. ``frame.attrs["notes"]`` lists every adjustment.
    """
    from caplife.file_reader import _DECIMAL_COMMA, TIME_UNITS, read_table
    frame, notes = read_table(payload)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in ("Last_alive_h", "Failure_by_h"):
        if column not in frame:
            continue
        text = frame[column].astype(str).str.strip()
        comma = text.str.match(_DECIMAL_COMMA)
        if comma.any():
            notes.append(f"Read decimal commas in {column} as decimal points.")
        text = text.where(~comma, text.str.replace(",", ".", regex=False))
        if time_unit != "hours":
            numbers = pd.to_numeric(text, errors="coerce") * TIME_UNITS[time_unit]
            text = text.where(numbers.isna(), numbers.map(lambda value: f"{value:.10g}"))
        frame[column] = text
    if time_unit != "hours":
        notes.append(f"Converted outcome ages from {time_unit} to hours.")
    frame.attrs["notes"] = notes
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


def _forecasts(raw: pd.DataFrame, neural: dict, train: pd.DataFrame,
               signals: tuple[str, ...]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Forecasts at each capacitor's latest reading, why any were withheld, and the forecast at every reading."""
    readings = validate_real_measurements(raw)
    features = neural["features"]
    lows, highs = train.loc[:, list(features)].min(), train.loc[:, list(features)].max()
    span = (highs - lows).replace(0, 1)
    rows, withheld, history_rows = [], {}, []
    for device, history in readings.groupby("Cap_ID", sort=True):
        lacking = [MEASUREMENTS[signal][0].lower() for signal in signals
                   if signal not in history or history[signal].count() < MIN_READINGS]
        if lacking:
            withheld[device] = f"Fewer than {MIN_READINGS} readings of " + ", ".join(lacking)
            continue
        found = landmarks(history, signals)
        if isinstance(found, str):
            withheld[device] = found
            continue
        for age, values in found:
            low, mid, high = calibrated_band(neural["model"], neural["radius"], values[None, :])[0]
            history_rows.append({"Capacitor": device, "Hours": age, "RUL_low_h": low, "RUL_median_h": mid,
                                 "RUL_high_h": high})
        age, values = found[-1]
        beyond = (np.maximum(values - highs, 0) + np.maximum(lows - values, 0)) / span
        if (beyond > TRAINING_RANGE_MARGIN).any():
            worst = beyond.idxmax()
            withheld[device] = (f"The latest readings fall outside what the training capacitors showed "
                                f"({FEATURE_LABELS[worst]} is {beyond[worst]:.0%} of the training range beyond it)")
            continue
        low, mid, high = calibrated_band(neural["model"], neural["radius"], values[None, :])[0]
        rows.append({
            "Capacitor": device,
            "Forecast made at (h)": age,
            "Latest reading (h)": float(history["Hours"].max()),
            "Predicted failure age (h)": age + mid,
            "80% range from (h)": age + low,
            "80% range to (h)": age + high,
            "Remaining from latest, median (h)": mid,
            "Remaining from latest, 80% range from (h)": low,
            "Remaining from latest, 80% range to (h)": high,
            "Status": "Forecast",
        })
    columns = ["Capacitor", "Forecast made at (h)", "Latest reading (h)", "Predicted failure age (h)",
               "80% range from (h)", "80% range to (h)", "Remaining from latest, median (h)",
               "Remaining from latest, 80% range from (h)", "Remaining from latest, 80% range to (h)", "Status"]
    history_columns = ["Capacitor", "Hours", "RUL_low_h", "RUL_median_h", "RUL_high_h"]
    return (pd.DataFrame(rows, columns=columns), withheld,
            pd.DataFrame(history_rows, columns=history_columns))


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
    missing = [column for column in OUTCOME_COLUMNS if column not in outcomes.columns]
    if missing or outcomes.empty:
        raise RealDataValidationError("The outcome file needs one row per capacitor with columns "
                                      + ", ".join(OUTCOME_COLUMNS) + ".")
    readings = validate_real_measurements(measurements)
    labels = _labels(outcomes)
    measured_ids = set(readings["Cap_ID"])
    unknown = sorted(set(labels["Cap_ID"]) - measured_ids)
    if unknown:
        shown = ", ".join(unknown[:5]) + (" …" if len(unknown) > 5 else "")
        raise RealDataValidationError(f"Outcome rows name capacitors with no measurements: {shown}")
    signals = signals_for(readings, labels["Cap_ID"])
    if not signals:
        raise RealDataValidationError(
            f"No parameter has at least {MIN_READINGS} readings for every capacitor with an outcome. Each of the six "
            "can be used, but every capacitor with an outcome needs the same parameters.")
    features = features_for(signals)

    frame, audit = all_landmarks(readings, labels, signals)
    if frame.empty:
        raise ValueError("No capacitor has the readings and follow-up needed to train a model.")
    devices = frame.loc[frame["evaluation"], ["device", "experiment"]]
    roles = device_roles(devices, method, calibration_experiment, test_experiment)
    frame = frame.assign(role=roles.reindex(frame["device"].to_numpy()).to_numpy())
    parts = {role: frame.loc[frame["role"].eq(role)] for role in ROLES}
    minimum = {"training": MIN_TRAINING_DEVICES, "calibration": MIN_CALIBRATION_DEVICES, "test": MIN_TEST_DEVICES}
    for role, part in parts.items():
        if part["device"].nunique() < minimum[role]:
            raise ValueError(f"The {role} group has {part['device'].nunique()} eligible capacitor(s); "
                             f"at least {minimum[role]} are needed.")
    if not np.isfinite(parts["training"][BOUNDS[1]]).any():
        raise ValueError("At least one training capacitor needs an observed failure.")
    # Training learns from every landmark; calibration and testing use one per capacitor.
    calibration = parts["calibration"].loc[parts["calibration"]["evaluation"]]
    test = parts["test"].loc[parts["test"]["evaluation"]]

    fitted = fit_candidates(parts["training"], calibration, test, features)
    reasons = release_checks(fitted, calibration)
    labelled_ids = set(labels["Cap_ID"])
    forecast_devices = sorted(measured_ids - labelled_ids)
    forecasts, withheld, forecast_history = None, {}, None
    if not reasons and forecast_devices:
        unlabelled = readings.loc[~readings["Cap_ID"].isin(labelled_ids)].reset_index(drop=True)
        forecasts, withheld, forecast_history = _forecasts(unlabelled, fitted["neural"], parts["training"], signals)
    parts = {role: part.loc[part["evaluation"]] for role, part in parts.items()}

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
        "parameters": list(signals),
        "training_landmarks": int(len(frame.loc[frame["role"].eq("training")])),
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
        "forecast_history": forecast_history,
        "parameters": signals,
        "training_landmarks": report["training_landmarks"],
    }
