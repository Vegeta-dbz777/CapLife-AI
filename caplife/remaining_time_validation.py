"""Compare censored remaining-time models with whole experiments held out."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caplife.neural_benchmark import FEATURES
from caplife.public_benchmark import EOS2_URLS, POFINN_URL, load_eos2, load_pofinn, read_source
from caplife.release_validation import CAMPAIGN, GROUP_FOR_COHORT, GROUPS, load_eis_loss, make_origins
from caplife.survival import fit_duration

QUANTILES = (0.1, 0.5, 0.9)
BOUNDS = ("remaining_lower_h", "remaining_upper_h")


def device_weights(frame: pd.DataFrame) -> np.ndarray:
    return 1 / frame.groupby("device")["device"].transform("size").to_numpy(dtype=float)


def duration_scores(test: pd.DataFrame, band: np.ndarray) -> dict:
    lower, upper = (test[column].to_numpy(dtype=float) for column in BOUNDS)
    band_low, point, band_high = band.T
    exact = lower == upper
    possible = np.where(exact, band_high >= lower, band_high > lower) & (band_low <= upper)
    definite = (band_low <= lower) & (band_high >= upper)
    distance = np.maximum.reduce([lower - point, point - upper, np.zeros(len(point))])
    observed = np.isfinite(upper)

    def mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        weights = device_weights(test.iloc[mask])
        return round(float(np.average(values[mask], weights=weights)), 6)

    by_device = test["device"].nunique()
    return {
        "devices": int(by_device),
        "origins": len(test),
        "bounded_or_exact_events": int(observed.sum()),
        "right_censored_origins": int((~observed).sum()),
        "mean_distance_to_observed_window_h": mean(distance, observed),
        "median_band_width_h": round(float(np.median(band_high - band_low)), 6),
        "possible_80pct_band_coverage_observed_events": mean(possible, observed),
        "definite_80pct_band_coverage_observed_events": mean(definite, observed),
        "band_below_right_censor_count": int((~possible & ~observed).sum()),
    }


def forecast_scores(test: pd.DataFrame, probability: np.ndarray, horizon: float) -> dict:
    lower, upper = (test[column].to_numpy(dtype=float) for column in BOUNDS)
    positive = upper <= horizon
    observed = positive | (lower >= horizon)
    alert = probability >= 0.5
    squared_error = (probability - positive) ** 2
    weights = device_weights(test)
    error_if_below = probability ** 2
    error_if_above = (1 - probability) ** 2
    # The per-window extremes give conservative outer bounds across unresolved cases.
    errors = [np.where(observed, squared_error, estimate(error_if_below, error_if_above))
              for estimate in (np.minimum, np.maximum)]
    resolved_weights = device_weights(test.iloc[observed])
    return {
        "resolved": int(observed.sum()),
        "brier": round(float(np.average(squared_error[observed], weights=resolved_weights)), 6)
        if observed.any() else None,
        "brier_bounds_all_origins": [round(float(np.average(error, weights=weights)), 6) for error in errors],
        "correct_alerts": int((alert & positive).sum()),
        "false_alerts": int((alert & observed & ~positive).sum()),
        "missed_crossings": int((~alert & positive).sum()),
        "devices_with_correct_alert": int(test.loc[alert & positive, "device"].nunique()),
        "devices_with_false_alert": int(test.loc[alert & observed & ~positive, "device"].nunique()),
    }


def evaluate_durations(train: pd.DataFrame, test: pd.DataFrame,
                       features: tuple[str, ...], horizons: tuple[int, ...] = (),
                       administrative_horizon: float | None = None) -> dict:
    lower, upper = (train[column].to_numpy(dtype=float) for column in BOUNDS)
    if administrative_horizon is not None:
        if not horizons or max(horizons) > administrative_horizon:
            raise ValueError("Restricted forecasts require horizons within the training follow-up limit.")
        lower = np.minimum(lower, administrative_horizon)
        upper = np.where(upper <= administrative_horizon, upper, np.inf)
    weights = device_weights(train)
    first = ~test["device"].duplicated().to_numpy()
    observed_range = train.loc[:, list(features)]
    above_or_below = (test.loc[:, list(features)].lt(observed_range.min())
                      | test.loc[:, list(features)].gt(observed_range.max())).any(axis=1)
    result = {
        "training_devices": int(train["device"].nunique()),
        "test_devices": int(test["device"].nunique()),
        "origins_outside_training_feature_ranges": int(above_or_below.sum()),
        "models": {},
    }
    for name, columns, hidden in (
        ("intercept_only", (), False), ("log_linear", features, False), ("neural", features, True),
    ):
        x_train, x_test = (part.loc[:, list(columns)].to_numpy(dtype=float) if columns
                           else np.zeros((len(part), 1)) for part in (train, test))
        fitted = fit_duration(x_train, lower, upper, weights, hidden=hidden)
        scores = {
            "horizons_h": {str(horizon): forecast_scores(test, fitted.probability_by(x_test, horizon), horizon)
                           for horizon in horizons},
        }
        if administrative_horizon is None:
            band = fitted.quantiles(x_test, QUANTILES)
            scores["all_origins"] = duration_scores(test, band)
            scores["first_origin_per_device"] = duration_scores(test.iloc[first], band[first])
        result["models"][name] = scores
    return result


def benchmark(losses: dict[str, pd.DataFrame]) -> dict:
    threshold, horizons = 20, (24, 48, 72)
    results = {}
    for setting, repeat in (("first_crossing", False), ("repeat_confirmation", True)):
        origins = pd.concat([make_origins(frame, name, 48, threshold, repeat)
                             for name, frame in losses.items()], ignore_index=True)
        if repeat:
            observed = np.isfinite(origins["remaining_upper_h"])
            # The recorded second reading is the confirmation event, at its exact inspection age.
            origins.loc[observed, "remaining_lower_h"] = origins.loc[observed, "remaining_upper_h"]
        source = origins["cohort"].map(GROUP_FOR_COHORT)
        results[setting] = {}
        for group in GROUPS:
            held_out = source.eq(group)
            train, test = origins.loc[~held_out], origins.loc[held_out]
            held_result = evaluate_durations(train, test, FEATURES, horizons)
            held_result["original_48h_trend"] = forecast_scores(test, test["trend"].to_numpy(), 48)
            held_result["fixed_72h_training_followup"] = evaluate_durations(
                train, test, FEATURES, horizons, administrative_horizon=72,
            )
            results[setting][group] = held_result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path)
    parser.add_argument("--pofinn", type=Path)
    for cohort in GROUPS[CAMPAIGN]:
        parser.add_argument(f"--{cohort.rsplit(' ', 1)[1].lower()}-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/remaining_time_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    eos, mat = load_eos2(read_source(args.eos2, EOS2_URLS))
    pofinn = read_source(args.pofinn, (POFINN_URL,))
    sessions = {name: getattr(args, name.rsplit(" ", 1)[1].lower() + "_sessions")
                for name in GROUPS[CAMPAIGN]}
    losses = {"NASA EOS-2": eos, "PoFINN": load_pofinn(pofinn),
              **{name: load_eis_loss(path) for name, path in sessions.items()}}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in
              {"EOS_DataSet.mat": mat, "data_combined.csv": pofinn,
               **{f"{name} sessions": path.read_bytes() for name, path in sessions.items()}}.items()}
    result = {
        "endpoint": "20 percent capacitance loss relative to initial capacitance",
        "nominal_interval_mass": 0.8,
        "features": FEATURES,
        "source_sha256": hashes,
        "validation": benchmark(losses),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
