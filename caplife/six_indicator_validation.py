"""Train and validate remaining-life models on paired six-indicator histories."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caplife.measurement_review import read_experimental_csv
from caplife.real_data import MEASUREMENTS, validate_real_measurements
from caplife.remaining_time_validation import BOUNDS, QUANTILES, duration_scores
from caplife.survival import fit_duration

SIGNALS = tuple(MEASUREMENTS)
FEATURES = tuple(f"{signal}_{stat}" for stat in ("change", "rate_per_h") for signal in SIGNALS)
OUTCOME_COLUMNS = ("Cap_ID", "Experiment_ID", "Endpoint", "Last_alive_h", "Failure_by_h")
INTERVAL_MASS = 0.8


def joined_landmarks(measurements: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(SIGNALS).difference(measurements.columns)
    if missing:
        raise ValueError(f"Missing measured indicators: {', '.join(sorted(missing))}")
    if (set(outcomes.columns) != set(OUTCOME_COLUMNS) or outcomes.empty
            or outcomes.columns.duplicated().any() or not isinstance(outcomes.index, pd.RangeIndex)):
        raise ValueError(f"Expected a nonempty outcome table with columns {', '.join(OUTCOME_COLUMNS)}.")
    readings = validate_real_measurements(measurements)
    labels = outcomes.loc[:, list(OUTCOME_COLUMNS)].copy()
    for column in ("Cap_ID", "Experiment_ID", "Endpoint"):
        labels[column] = labels[column].astype("string").str.strip()
        if labels[column].isna().any() or labels[column].eq("").any():
            raise ValueError(f"{column} must be recorded for every outcome.")
    if labels["Cap_ID"].duplicated().any() or set(labels["Cap_ID"]) != set(readings["Cap_ID"]):
        raise ValueError("Include exactly one outcome per unique physical capacitor ID in the measurements.")
    if labels["Endpoint"].nunique() != 1:
        raise ValueError("All experiments must use the same documented endpoint.")
    for column in ("Last_alive_h", "Failure_by_h"):
        text = labels[column].astype("string").str.strip().replace("", pd.NA)
        labels[column] = pd.to_numeric(text, errors="raise").astype(float)
    lo = labels["Last_alive_h"]
    hi = labels["Failure_by_h"].fillna(np.inf)
    if (not np.isfinite(lo).all() or (lo < 0).any() or (hi < lo).any()
            or not np.isfinite(labels["Failure_by_h"].dropna()).all()):
        raise ValueError("Invalid outcome bounds. A blank Failure_by_h is right-censored.")

    history_by_device = {device: group for device, group in readings.groupby("Cap_ID", sort=False)}
    rows, dropped = [], {}
    for label in labels.itertuples(index=False):
        history = history_by_device[label.Cap_ID]
        upper = math.inf if pd.isna(label.Failure_by_h) else float(label.Failure_by_h)
        # Keep one prospective landmark per physical capacitor.
        complete = history.dropna(subset=list(SIGNALS)).iloc[:3]
        if len(complete) != 3 or complete.iloc[0]["Hours"] != 0:
            dropped[label.Cap_ID] = "Requires a complete zero-hour baseline and two later complete readings"
            continue
        age = float(complete.iloc[-1]["Hours"])
        if age >= label.Last_alive_h:
            dropped[label.Cap_ID] = "Insufficient documented follow-up after the landmark"
            continue
        prefix = history.loc[history["Hours"].le(age) & history["ESR_ohm"].notna()]
        if any(column in prefix and prefix[column].nunique() > 1
               for column in ("ESR_Test_Hz", "ESR_Test_Temp_C")):
            dropped[label.Cap_ID] = "Recorded ESR test conditions change before the landmark"
            continue
        measured = complete.loc[:, list(SIGNALS)].to_numpy(dtype=float)
        change = measured - measured[0]
        change[:, 0] *= -100 / measured[0, 0]
        change[:, 1] *= 100 / measured[0, 1]
        relative_age = complete["Hours"].to_numpy(dtype=float) - age
        rate = np.polyfit(relative_age, change, 1)[0]
        rows.append({
            "device": label.Cap_ID, "experiment": label.Experiment_ID, "age_h": age,
            BOUNDS[0]: float(label.Last_alive_h) - age, BOUNDS[1]: upper - age,
            **dict(zip(FEATURES, np.concatenate([change[-1], rate]), strict=True)),
        })
    frame = pd.DataFrame(rows, columns=["device", "experiment", "age_h", *BOUNDS, *FEATURES])
    return frame, {
        "endpoint": str(labels["Endpoint"].iloc[0]),
        "devices_submitted": len(labels),
        "devices_excluded": dropped,
        "experiments": sorted(str(value) for value in labels["Experiment_ID"].unique()),
    }


def calibrate_log_radius(calibration: pd.DataFrame, locations: np.ndarray) -> float | None:
    """Conservatively calibrate against all event times consistent with a record."""
    lower, upper = (calibration[column].to_numpy(dtype=float) for column in BOUNDS)
    scores = np.maximum(np.abs(np.log(lower) - locations), np.abs(np.log(upper) - locations))
    rank = math.ceil((len(scores) + 1) * INTERVAL_MASS)
    if rank > len(scores):
        return None
    radius = float(np.sort(scores)[rank - 1])
    return radius if np.isfinite(radius) else None


def validate_model(measurements: pd.DataFrame, outcomes: pd.DataFrame,
                   calibration_experiment: str, test_experiment: str) -> dict:
    frame, audit = joined_landmarks(measurements, outcomes)
    if calibration_experiment == test_experiment:
        raise ValueError("Use different experiments for calibration and final testing.")
    calibration = frame.loc[frame["experiment"].eq(calibration_experiment)]
    test = frame.loc[frame["experiment"].eq(test_experiment)]
    train = frame.loc[~frame["experiment"].isin([calibration_experiment, test_experiment])]
    if any(part.empty for part in (train, calibration, test)):
        raise ValueError("Training, calibration and final testing each need a distinct experiment with eligible devices.")
    if not np.isfinite(train[BOUNDS[1]]).any():
        raise ValueError("Training requires a documented observed endpoint.")
    result = {
        **audit,
        "training_experiments": sorted(str(value) for value in train["experiment"].unique()),
        "calibration_experiment": calibration_experiment, "test_experiment": test_experiment,
        "devices": {name: len(part) for name, part in
                    (("training", train), ("calibration", calibration), ("test", test))},
        "features": FEATURES, "target_interval_mass": INTERVAL_MASS, "models": {},
    }
    candidates = (
        ("intercept_only", (), False),
        ("log_linear", FEATURES, False),
        ("neural", FEATURES, True),
    )
    for name, features, hidden in candidates:
        x_train, x_cal, x_test = [
            part.loc[:, list(features)].to_numpy(dtype=float) if features else np.zeros((len(part), 1))
            for part in (train, calibration, test)
        ]
        model = fit_duration(x_train, train[BOUNDS[0]], train[BOUNDS[1]], np.ones(len(train)), hidden)
        radius = calibrate_log_radius(calibration, model.location(x_cal))
        coverage = None
        if radius is not None:
            location = model.location(x_test)
            band = np.exp(location[:, None] + np.array([-radius, 0, radius]))
            coverage = duration_scores(test, band)
        result["models"][name] = {
            "test_model_interval": duration_scores(test, model.quantiles(x_test, QUANTILES)),
            "test_calibrated_interval": coverage,
            "calibration_log_radius": radius,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--calibration-experiment", required=True)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    measurements, outcomes = args.measurements.read_bytes(), args.outcomes.read_bytes()
    labels = pd.read_csv(BytesIO(outcomes), keep_default_na=False, dtype={
        "Cap_ID": str, "Experiment_ID": str, "Endpoint": str,
    })
    result = validate_model(read_experimental_csv(measurements), labels,
                            args.calibration_experiment, args.test_experiment)
    result["input_sha256"] = {name: hashlib.sha256(payload).hexdigest() for name, payload
                             in (("measurements", measurements), ("outcomes", outcomes))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
