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
an outcome has measured at least three times, from one to all six. Operating
conditions (core and ambient temperature, applied over rated voltage, ripple
over rated ripple current, operating frequency) join the features when every
capacitor with an outcome has them recorded, at the value in effect at each
landmark.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from caplife.real_data import MEASUREMENTS, RealDataValidationError, validate_real_measurements
from caplife.remaining_time_validation import BOUNDS
from caplife.six_indicator_validation import (
    INTERVAL_MASS, OUTCOME_COLUMNS, SIGNALS, calibrate_log_radius, calibrated_band,
)
from caplife.remaining_time_validation import QUANTILES, duration_scores
from caplife.survival import fit_duration
from caplife.six_rul import CONDITION_LABELS, CONDITIONS, condition_values

# Optional: "limit" (a parameter reached its end-of-life limit) or "sudden" (open circuit, short, vent);
# left blank, it is worked out from the readings.
OPTIONAL_OUTCOME_COLUMNS = ("Failure_mode",)
SPLIT_BY_EXPERIMENT = "experiment"
SPLIT_BY_DEVICE = "device"
SPLIT_ROTATE = "rotation"  # every experiment but the calibration one is the test in turn
ROLES = ("training", "calibration", "test")
MODEL_LABELS = {
    "trained": "Your trained model",
    "built_in": "Built-in model",
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
def _name(signal: str) -> str:
    """A parameter's name mid-sentence: lower case except abbreviations such as ESR and H₂."""
    name = MEASUREMENTS[signal][0].split(" (")[0]
    return name if name.split()[0].isupper() or name.startswith("H₂") else name.lower()


FEATURE_LABELS = {
    **{f"{signal}_change": f"{_name(signal)} change" for signal in SIGNALS},
    **{f"{signal}_rate_per_h": f"{_name(signal)} rate" for signal in SIGNALS},
    "age_since_first_h": "time since the first reading",
    **{condition: CONDITION_LABELS[condition].lower() for condition in CONDITIONS},
}
MIN_READINGS = 3  # a landmark needs the first reading and two more


def features_for(signals: tuple[str, ...], conditions: tuple[str, ...] = ()) -> tuple[str, ...]:
    return (*(f"{signal}_change" for signal in signals), *(f"{signal}_rate_per_h" for signal in signals),
            "age_since_first_h", *conditions)


def _condition_inputs(history: pd.DataFrame) -> pd.DataFrame:
    """Operating conditions at every reading as model inputs (frequency on a log scale)."""
    values = condition_values(history)
    return values.assign(frequency=np.log10(values["frequency"]))


def conditions_for(readings: pd.DataFrame, devices) -> tuple[str, ...]:
    """Operating conditions recorded for every one of these capacitors (carried forward between readings)."""
    groups = [readings.loc[readings["Cap_ID"].eq(device)] for device in devices]
    if not groups:
        return ()
    found = [condition_values(group) for group in groups]
    return tuple(condition for condition in CONDITIONS if all(values[condition].notna().all() for values in found))


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


def landmarks(history: pd.DataFrame, signals: tuple[str, ...],
              conditions: tuple[str, ...] = ()) -> list[tuple[float, np.ndarray]] | str:
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
    operating = _condition_inputs(complete).loc[:, list(conditions)].to_numpy(dtype=float)[:len(hours)]
    if conditions and not np.isfinite(operating).all():
        return "Operating conditions are not recorded: " + ", ".join(
            FEATURE_LABELS[condition] for condition, ok in zip(conditions, np.isfinite(operating).all(axis=0)) if not ok)
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
        found.append((float(hours[k]), np.concatenate([change[k], np.atleast_1d(rate), [hours[k] - hours[0]],
                                                       operating[k]])))
    return found


def _pick(device: str, count: int) -> int:
    """One landmark per capacitor for calibration and testing, fixed by its ID, not its outcome."""
    return int(hashlib.sha256(f"landmark:{device}".encode("utf-8")).hexdigest(), 16) % count


def _labels(outcomes: pd.DataFrame) -> pd.DataFrame:
    optional = [column for column in OPTIONAL_OUTCOME_COLUMNS if column in outcomes.columns]
    labels = outcomes.loc[:, [*OUTCOME_COLUMNS, *optional]].copy()
    if "Failure_mode" in labels:
        mode = labels["Failure_mode"].astype("string").str.strip().str.lower().fillna("")
        wrong = sorted(set(mode) - {"", "limit", "sudden"})
        if wrong:
            raise ValueError("Failure_mode must be limit, sudden or blank (blank: worked out from the readings); "
                             f"found {', '.join(wrong[:3])}.")
        labels["Failure_mode"] = mode
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


def all_landmarks(readings: pd.DataFrame, labels: pd.DataFrame, signals: tuple[str, ...],
                  conditions: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """Every usable landmark of every capacitor with an outcome, with its remaining-time bounds."""
    features = features_for(signals, conditions)
    by_device = {device: group for device, group in readings.groupby("Cap_ID", sort=False)}
    rows, dropped = [], {}
    for label in labels.itertuples(index=False):
        found = landmarks(by_device[label.Cap_ID], signals, conditions)
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
        "operating_conditions": list(conditions),
    }


MEASUREMENT_TEMPLATE_COLUMNS = ("Cap_ID", "Hours", *SIGNALS, "ESR_Test_Hz", "ESR_Test_Temp_C", "Ambient_C", "Core_C",
                                "Voltage_V", "Rated_V", "Ripple_A", "Rated_Ripple_A", "Ripple_Hz")


def measurement_template_csv() -> str:
    return pd.DataFrame(columns=MEASUREMENT_TEMPLATE_COLUMNS).to_csv(index=False)


def outcome_template_csv() -> str:
    return pd.DataFrame(columns=[*OUTCOME_COLUMNS, *OPTIONAL_OUTCOME_COLUMNS]).to_csv(index=False)


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



# --- Training, testing and forecasting ----------------------------------------------------------

def _histories(readings: pd.DataFrame) -> dict:
    return {device: group.sort_values("Hours").reset_index(drop=True)
            for device, group in readings.groupby("Cap_ID", sort=False)}


def _bands(model, histories: dict, rows: pd.DataFrame, signals: tuple[str, ...], limits: dict) -> np.ndarray:
    """(low, median, high) remaining life predicted at each row's evaluation reading, from the readings up to it."""
    bands = []
    for row in rows.itertuples(index=False):
        history = histories[row.device]
        last = model.assess(history.loc[history["Hours"] <= row.age_h].reset_index(drop=True), signals, limits).iloc[-1]
        bands.append((last["RUL_low_h"], last["RUL_median_h"], last["RUL_high_h"]))
    return np.asarray(bands, dtype=float).reshape(-1, 3)


def _average_lifetime(train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """The do-nothing baseline: one remaining-time distribution for every capacitor, calibrated like the others."""
    zeros = lambda part: np.zeros((len(part), 1))
    model = fit_duration(zeros(train), train[BOUNDS[0]], train[BOUNDS[1]], train["weight"].to_numpy(dtype=float),
                         hidden=False)
    radius = calibrate_log_radius(calibration, model.location(zeros(calibration)))
    if radius is None:
        return model.quantiles(zeros(test), QUANTILES), False
    return calibrated_band(model, radius, zeros(test)), True


def _scores(test: pd.DataFrame, band: np.ndarray) -> dict:
    return duration_scores(test, band)


def release_checks(trained: dict, average: dict, calibration: pd.DataFrame, calibrated: bool) -> list[str]:
    """Reasons the trained model's forecasts must be withheld; empty when every check passes."""
    if not calibrated:
        still_working = int((~np.isfinite(calibration[BOUNDS[1]].to_numpy(dtype=float))).sum())
        return [f"The {len(calibration)} calibration capacitors could not set a finite 80% range. "
                f"{still_working} of them were still working at their last check, so their failure time "
                "has no upper limit, and the conservative calibration needs nearly all of them to have failed. "
                "Add capacitors with observed failures, or calibrate on an experiment with more failures."]
    reasons = []
    failures = trained["bounded_or_exact_events"]
    if failures < MIN_TEST_FAILURES:
        reasons.append(f"Only {failures} test capacitor(s) have an observed failure; "
                       f"at least {MIN_TEST_FAILURES} are needed to score the predictions.")
    else:
        coverage = trained["definite_80pct_band_coverage_observed_events"]
        if coverage < MIN_TEST_COVERAGE:
            reasons.append(f"The 80% range contained {coverage:.0%} of observed test failures; "
                           f"at least {MIN_TEST_COVERAGE:.0%} is required.")
        miss, baseline = trained["mean_distance_to_observed_window_h"], average["mean_distance_to_observed_window_h"]
        if baseline is not None and miss > baseline:
            reasons.append(f"The trained model missed test failures by {miss:,.0f} h on average, more than the "
                           f"{baseline:,.0f} h of simply predicting the average lifetime: your readings don't "
                           "predict these failures better than the average lifetime.")
    censored = trained["right_censored_origins"]
    if censored and trained["band_below_right_censor_count"] / censored > MAX_CENSOR_CONTRADICTIONS:
        reasons.append(f"For {trained['band_below_right_censor_count']} of {censored} test capacitors that were "
                       "still working at their last check, the whole 80% range ended before that check.")
    return reasons


def _test_table(test: pd.DataFrame, band: np.ndarray) -> pd.DataFrame:
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


def _model_table(scores: dict, ranges: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "Model": MODEL_LABELS[name],
        "Average miss (h)": scores[name]["mean_distance_to_observed_window_h"],
        "Failures inside 80% range": scores[name]["definite_80pct_band_coverage_observed_events"],
        "Typical 80% range width (h)": scores[name]["median_band_width_h"],
        "Range": ranges[name],
    } for name in MODEL_LABELS])


def _forecasts(readings: pd.DataFrame, model, train: pd.DataFrame, signals: tuple[str, ...],
               conditions: tuple[str, ...], limits: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Forecasts at each capacitor's latest reading, why any were withheld, and the forecast at every reading."""
    features = features_for(signals, conditions)
    lows, highs = train.loc[:, list(features)].min(), train.loc[:, list(features)].max()
    span = (highs - lows).replace(0, 1)
    sudden = getattr(model, "sudden", None) is not None
    rows, withheld, history_rows = [], {}, []
    for device, history in sorted(_histories(readings).items()):
        lacking = [_name(signal) for signal in signals
                   if signal not in history or history[signal].count() < MIN_READINGS]
        if lacking:
            withheld[device] = f"Fewer than {MIN_READINGS} readings of " + ", ".join(lacking)
            continue
        found = landmarks(history, signals, conditions)
        if isinstance(found, str):
            withheld[device] = found
            continue
        table = model.assess(history, signals, limits).dropna(subset=["RUL_median_h"])
        for row in table.itertuples(index=False):
            history_rows.append({"Capacitor": device, "Hours": row.Hours, "RUL_low_h": row.RUL_low_h,
                                 "RUL_median_h": row.RUL_median_h, "RUL_high_h": row.RUL_high_h})
        values = found[-1][1]
        beyond = (np.maximum(values - highs, 0) + np.maximum(lows - values, 0)) / span
        if (beyond > TRAINING_RANGE_MARGIN).any():
            worst = beyond.idxmax()
            withheld[device] = (f"The latest readings fall outside what the training capacitors showed "
                                f"({FEATURE_LABELS[worst]} is {beyond[worst]:.0%} of the training range beyond it)")
            continue
        last = table.iloc[-1]
        age, low, mid, high = (float(last[column]) for column in ("Hours", "RUL_low_h", "RUL_median_h", "RUL_high_h"))
        row = {
            "Capacitor": device,
            "Forecast made at (h)": age,
            "Latest reading (h)": float(history["Hours"].max()),
            "Predicted failure age (h)": age + mid,
            "80% range from (h)": age + low,
            "80% range to (h)": age + high,
            "Remaining from latest, median (h)": mid,
            "Remaining from latest, 80% range from (h)": low,
            "Remaining from latest, 80% range to (h)": high,
        }
        if sudden:
            row["Chance of sudden failure (%)"] = 100 * float(last.get("Sudden_failure_chance", 0.0))
        rows.append({**row, "Status": "Forecast"})
    columns = ["Capacitor", "Forecast made at (h)", "Latest reading (h)", "Predicted failure age (h)",
               "80% range from (h)", "80% range to (h)", "Remaining from latest, median (h)",
               "Remaining from latest, 80% range from (h)", "Remaining from latest, 80% range to (h)",
               *(["Chance of sudden failure (%)"] if sudden else []), "Status"]
    history_columns = ["Capacitor", "Hours", "RUL_low_h", "RUL_median_h", "RUL_high_h"]
    return (pd.DataFrame(rows, columns=columns), withheld,
            pd.DataFrame(history_rows, columns=history_columns))


def _limit_text(signal: str, limits: dict) -> str:
    """One limit as the limit boxes show it, e.g. "capacitance loss 20%" or "ESR rise 100%"."""
    from caplife.six_rul import LIMIT_KEYS
    value = limits[LIMIT_KEYS[signal]]
    text = {"Capacitance_uF": f"capacitance loss {value:g}%", "ESR_ohm": f"ESR rise {(value - 1) * 100:g}%",
            "Leakage_uA": f"leakage current rise {(value - 1) * 100:g}%", "Case_Expansion_um": f"case expansion {value:g} µm",
            "H2_Generated_mL_STP": f"H₂ generated {value:g} mL", "Pressure_Abs_kPa": f"internal pressure {value:g} kPa"}
    return text[signal]


def _tested(readings: pd.DataFrame, devices, signals: tuple[str, ...], modes: dict, limits: dict) -> list:
    """What the capacitors with an outcome covered, as (item, what it was) rows."""
    used = readings.loc[readings["Cap_ID"].isin(list(devices))]
    rows = [("Parameters measured", ", ".join(MEASUREMENTS[signal][0].split(" (")[0] for signal in signals))]
    values = pd.concat([condition_values(group) for _, group in used.groupby("Cap_ID")])
    shown = {"core_temperature": ("°C", 1), "ambient_temperature": ("°C", 1), "voltage_ratio": ("% of rated", 100),
             "ripple_ratio": ("% of rated", 100), "frequency": ("Hz", 1)}
    for condition in CONDITIONS:
        column = values[condition].dropna()
        if column.empty:
            rows.append((CONDITION_LABELS[condition], "not recorded"))
            continue
        unit, scale = shown[condition]
        low, high = column.min() * scale, column.max() * scale
        gap = "" if unit.startswith("%") else " "
        text = (f"{low:,.0f}{gap}{unit}" if np.isclose(low, high) else f"{low:,.0f} to {high:,.0f}{gap}{unit}")
        rows.append((CONDITION_LABELS[condition], text))
    rows.append(("Ages measured", f"up to {used['Hours'].max():,.0f} h"))
    counts = pd.Series(modes).value_counts()
    rows.append(("Failures", f"{counts.get('limit', 0)} reached a limit, {counts.get('sudden', 0)} sudden, "
                             f"{counts.get('running', 0)} still working when testing stopped"))
    rows.append(("End-of-life limits", ", ".join(_limit_text(signal, limits) for signal in signals)))
    return rows


def _turn_row(experiment: str, test: pd.DataFrame, scores: dict, reasons: list[str]) -> dict:
    trained = scores["trained"]
    return {
        "Test experiment": experiment,
        "Test capacitors": int(test["device"].nunique()),
        "Observed failures": int(trained["bounded_or_exact_events"]),
        "Average miss (h)": trained["mean_distance_to_observed_window_h"],
        "Failures inside 80% range": trained["definite_80pct_band_coverage_observed_events"],
        "Typical 80% range width (h)": trained["median_band_width_h"],
        "Built-in model miss (h)": scores["built_in"]["mean_distance_to_observed_window_h"],
        "Average lifetime miss (h)": scores["intercept_only"]["mean_distance_to_observed_window_h"],
        "On its own": "Passes" if not reasons else "Would not pass",
    }


def run_rul_workflow(measurements: pd.DataFrame, outcomes: pd.DataFrame, method: str,
                     calibration_experiment: str | None = None, test_experiment: str | None = None,
                     limits: dict | None = None, progress=None) -> dict:
    """Train the six-parameter model further on the user's capacitors, test it on held-out ones, and
    forecast only if it passes.

    With ``SPLIT_ROTATE`` every experiment except the calibration one is the test in turn (the model then
    trains on the others); the checks apply to all those test capacitors together, and the forecasts come
    from a final model trained on every experiment except the calibration one. ``progress(share, text)``
    is told how far training has got."""
    from caplife import fine_tune
    from caplife.six_rul import full_limits

    progress = progress or (lambda share, text: None)
    missing = [column for column in OUTCOME_COLUMNS if column not in outcomes.columns]
    if missing or outcomes.empty:
        raise RealDataValidationError("The outcome file needs one row per capacitor with columns "
                                      + ", ".join(OUTCOME_COLUMNS) + ".")
    limits = full_limits(limits)
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
    conditions = conditions_for(readings, labels["Cap_ID"])

    frame, audit = all_landmarks(readings, labels, signals, conditions)
    if frame.empty:
        raise ValueError("No capacitor has the readings and follow-up needed to train a model.")
    devices = frame.loc[frame["evaluation"], ["device", "experiment"]]
    histories = _histories(readings)
    by_label = {label.Cap_ID: label for label in labels.itertuples(index=False)}
    used = sorted(set(frame["device"]))
    modes = {device: fine_tune.failure_mode(histories[device], by_label[device], signals, limits) for device in used}
    base = fine_tune.built_in()

    def fit(roles: pd.Series, needs_test: bool = True, turn: str = "") -> dict:
        """Train, calibrate and (when there is a test group) score one split."""
        assigned = frame.assign(role=roles.reindex(frame["device"].to_numpy()).to_numpy())
        parts = {role: assigned.loc[assigned["role"].eq(role)] for role in ROLES}
        minimum = {"training": MIN_TRAINING_DEVICES, "calibration": MIN_CALIBRATION_DEVICES,
                   "test": MIN_TEST_DEVICES if needs_test else 0}
        for role, part in parts.items():
            if part["device"].nunique() < minimum[role]:
                raise ValueError(f"The {role} group{turn} has {part['device'].nunique()} eligible capacitor(s); "
                                 f"at least {minimum[role]} are needed.")
        if not np.isfinite(parts["training"][BOUNDS[1]]).any():
            raise ValueError(f"At least one training capacitor{turn} needs an observed failure.")
        # Calibration and testing use one reading per capacitor.
        calibration = parts["calibration"].loc[parts["calibration"]["evaluation"]]
        test = parts["test"].loc[parts["test"]["evaluation"]]
        model = fine_tune.train(histories, by_label, modes, sorted(set(parts["training"]["device"])),
                                signals, conditions, limits)
        # The range is calibrated on every reading of the calibration capacitors, early and late in life.
        adjustment = fine_tune.calibrate(model, histories, parts["calibration"], signals, limits)
        model.set_adjustment(*(adjustment or (0.0, 0.0)))
        result = {"assigned": assigned, "parts": parts, "calibration": calibration, "test": test, "model": model,
                  "adjustment": adjustment, "calibrated": adjustment is not None}
        if len(test):
            average, average_calibrated = _average_lifetime(parts["training"], calibration, test)
            result["bands"] = {"trained": _bands(model, histories, test, signals, limits),
                               "built_in": _bands(base, histories, test, signals, limits), "intercept_only": average}
            result["average_calibrated"] = average_calibrated
            result["scores"] = {name: _scores(test, band) for name, band in result["bands"].items()}
        return result

    turns = None
    if method == SPLIT_ROTATE:
        experiments = sorted(set(map(str, devices["experiment"])))
        if calibration_experiment not in experiments:
            raise ValueError("Choose a calibration experiment that contains eligible capacitors.")
        tests = [experiment for experiment in experiments if experiment != calibration_experiment]
        if len(tests) < 2:
            raise ValueError("Testing every experiment in turn needs at least three experiments.")
        done, rows = [], []
        for number, experiment in enumerate(tests):
            progress(number / (len(tests) + 1), f"Testing on {experiment} ({number + 1} of {len(tests)})…")
            split = fit(device_roles(devices, SPLIT_BY_EXPERIMENT, calibration_experiment, experiment),
                        turn=f" when testing on {experiment}")
            done.append(split)
            rows.append(_turn_row(experiment, split["test"], split["scores"],
                                  release_checks(split["scores"]["trained"], split["scores"]["intercept_only"],
                                                 split["calibration"], split["calibrated"])))
        progress(len(tests) / (len(tests) + 1), "Training the final model on every experiment but the calibration one…")
        final = fit(pd.Series(np.where(devices["experiment"].astype(str).eq(calibration_experiment), "calibration",
                                       "training"), index=devices["device"].to_numpy()), needs_test=False)
        test = pd.concat([split["test"] for split in done])
        bands = {name: np.concatenate([split["bands"][name] for split in done]) for name in MODEL_LABELS}
        scores = {name: _scores(test, band) for name, band in bands.items()}
        calibrated = final["calibrated"] and all(split["calibrated"] for split in done)
        average_calibrated = all(split["average_calibrated"] for split in done)
        turns = pd.DataFrame(rows)
        test_experiments = tests
    else:
        progress(0.0, "Training and testing…")
        final = fit(device_roles(devices, method, calibration_experiment, test_experiment))
        test, bands, scores = final["test"], final["bands"], final["scores"]
        calibrated, average_calibrated = final["calibrated"], final["average_calibrated"]
        test_experiments = sorted(map(str, test["experiment"].unique()))
    progress(1.0, "Done")

    trained_model, parts, adjustment = final["model"], final["parts"], final["adjustment"]
    calibration = final["calibration"]
    ranges = {"trained": "Calibrated on your capacitors" if calibrated else "Not calibrated",
              "built_in": "Built-in calibration",
              "intercept_only": "Calibrated on your capacitors" if average_calibrated else "Not calibrated"}
    reasons = release_checks(scores["trained"], scores["intercept_only"], calibration, calibrated)
    warnings = []
    trained_miss, built_in_miss = (scores[name]["mean_distance_to_observed_window_h"] for name in ("trained", "built_in"))
    if trained_miss is not None and built_in_miss is not None and trained_miss > built_in_miss:
        warnings.append(f"Your trained model missed test failures by {trained_miss:,.0f} h on average; the built-in "
                        f"model by {built_in_miss:,.0f} h. On these test capacitors, training on your data did not "
                        "improve on the built-in model.")

    labelled_ids = set(labels["Cap_ID"])
    forecast_devices = sorted(measured_ids - labelled_ids)
    forecasts, withheld, forecast_history = None, {}, None
    if not reasons and forecast_devices:
        unlabelled = readings.loc[~readings["Cap_ID"].isin(labelled_ids)].reset_index(drop=True)
        forecasts, withheld, forecast_history = _forecasts(unlabelled, trained_model, parts["training"], signals,
                                                           conditions, limits)
    mode_counts = {mode: int(count) for mode, count in pd.Series(modes).value_counts().items()}
    evaluation = {role: part.loc[part["evaluation"]] for role, part in parts.items()}
    experiments_by_role = {
        "training": sorted(map(str, evaluation["training"]["experiment"].unique())),
        "calibration": sorted(map(str, evaluation["calibration"]["experiment"].unique())),
        "test": test_experiments,
    }
    roles = {"training": len(evaluation["training"]), "calibration": len(evaluation["calibration"]), "test": len(test)}

    report = {
        **audit,
        "split_method": method,
        "experiments_by_role": experiments_by_role,
        "devices_by_role": roles,
        "end_of_life_limits": limits,
        "failure_modes": mode_counts,
        "target_interval_mass": INTERVAL_MASS,
        "release_checks": {
            "min_test_failures": MIN_TEST_FAILURES, "min_test_coverage": MIN_TEST_COVERAGE,
            "max_censor_contradictions": MAX_CENSOR_CONTRADICTIONS,
        },
        "passed": not reasons,
        "withheld_reasons": reasons,
        "warnings": warnings,
        "calibration_log_adjustment": None if adjustment is None else {"early": adjustment[0], "late": adjustment[1]},
        "models": scores,
        "each_experiment_as_test": None if turns is None else turns.to_dict(orient="records"),
        "forecast_devices": forecast_devices,
        "forecasts": [] if forecasts is None else forecasts.to_dict(orient="records"),
        "forecasts_withheld": withheld,
        "parameters": list(signals),
        "operating_conditions": list(conditions),
        "training_landmarks": int(len(parts["training"])),
    }
    return {
        "report": _json_safe(report),
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "audit": audit,
        "roles": roles,
        "experiments_by_role": experiments_by_role,
        "model_table": _model_table(scores, ranges),
        "test_table": _test_table(test, bands["trained"]),
        "turns": turns,
        "forecast_devices": forecast_devices,
        "forecasts": forecasts,
        "forecasts_withheld": withheld,
        "forecast_history": forecast_history,
        "parameters": signals,
        "operating_conditions": conditions,
        "training_landmarks": report["training_landmarks"],
        "failure_modes": mode_counts,
        "limits": limits,
        "tested": _tested(readings, used, signals, modes, limits),
        "model": trained_model,
    }
