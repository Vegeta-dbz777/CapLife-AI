"""Exploratory remaining-life evaluation on published AEC failure observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caplife.remaining_time_validation import BOUNDS, evaluate_durations

SOURCE = "https://doi.org/10.1109/TPEL.2023.3263125"
FEATURES = ("capacitance_loss_pp", "esr_increase_pp", "capacitance_loss_rate_pph", "esr_increase_rate_pph")


def landmark_examples(markers: pd.DataFrame, outcomes: pd.DataFrame,
                      minimum_readings: int = 3) -> pd.DataFrame:
    """Use the first eligible point per device and only pre-existing observations."""
    if minimum_readings < 2:
        raise ValueError("At least two observations are needed to estimate a rate.")
    signals = ["test_age_h", "capacitance_pct_of_initial", "esr_pct_of_initial"]
    finite_markers = markers.loc[:, signals].dropna().to_numpy()
    if (not np.isfinite(finite_markers).all() or markers["test_age_h"].lt(0).any()
            or markers.duplicated(["device_id", "test_age_h"]).any()
            or outcomes["device_id"].duplicated().any()):
        raise ValueError("Measurement ages must be finite, nonnegative, and unique within each device.")
    known = set(outcomes["device_id"])
    if not set(markers["device_id"]).issubset(known):
        raise ValueError("An outcome record is required for each measured device.")

    by_device = dict(tuple(markers.groupby("device_id", sort=False)))
    rows = []
    for outcome in outcomes.itertuples(index=False):
        earliest, latest = outcome.failure_age_lower_h, outcome.failure_age_upper_h
        if outcome.event_time_precision == "annotated_terminal_hour" and earliest == latest:
            # The published ages are printed at full-hour resolution.
            earliest, latest = max(0, earliest - 0.5), latest + 0.5
        if not np.isfinite(earliest) or earliest < 0 or np.isnan(latest) or latest < earliest:
            raise ValueError(f"Invalid failure bounds for {outcome.device_id}.")
        if outcome.device_id not in by_device:
            continue
        measured = by_device[outcome.device_id].dropna(subset=signals).sort_values("test_age_h")
        if not measured["case"].eq(outcome.case).all() or (measured[signals[1:]] <= 0).any().any():
            raise ValueError(f"Inconsistent case or indicator readings for {outcome.device_id}.")
        if len(measured) < minimum_readings or measured.iloc[0]["test_age_h"] != 0:
            continue
        history = measured.iloc[:minimum_readings]
        origin = float(history.iloc[-1]["test_age_h"])
        if origin >= earliest:
            continue
        t = history["test_age_h"].to_numpy(dtype=float) - origin
        c_loss = 100 - history["capacitance_pct_of_initial"].to_numpy(dtype=float)
        esr_increase = history["esr_pct_of_initial"].to_numpy(dtype=float) - 100
        rows.append({
            "device": outcome.device_id, "case": outcome.case, "origin_h": origin,
            BOUNDS[0]: earliest - origin, BOUNDS[1]: latest - origin,
            FEATURES[0]: c_loss[-1], FEATURES[1]: esr_increase[-1],
            FEATURES[2]: np.polyfit(t, c_loss, 1)[0],
            FEATURES[3]: np.polyfit(t, esr_increase, 1)[0],
        })
    return pd.DataFrame(rows, columns=["device", "case", "origin_h", *BOUNDS, *FEATURES])


def benchmark(markers: pd.DataFrame, endpoints: pd.DataFrame) -> dict:
    row_counts = {}
    for observations in (3, 2):
        origins = landmark_examples(markers, endpoints, observations)
        folds = {}
        for case, devices in endpoints.groupby("case"):
            train, test = origins.loc[origins["case"].ne(case)], origins.loc[origins["case"].eq(case)]
            folds[str(case)] = {
                "devices_in_source": len(devices), "devices_with_eligible_history": len(test),
                "metrics": evaluate_durations(train, test, FEATURES)
                if not test.empty and train["case"].nunique() else None,
            }
        columns = ["device", "case", "origin_h", *BOUNDS]
        row_counts[str(observations)] = {
            "held_out_cases": folds,
            "landmarks": json.loads(origins.loc[:, columns].round(3).to_json(orient="records")),
        }
    return {
        "source": SOURCE,
        "endpoint": "Reported catastrophic open- or short-circuit failure",
        "observations": "Approximate capacitance and ESR markers from published figures",
        "devices": int(endpoints["device_id"].nunique()),
        "experiments": 1,
        "features": FEATURES,
        "six_indicator_devices": 0,
        "folds_by_minimum_observations": row_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/physical_failure_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = benchmark(pd.read_csv(args.markers), pd.read_csv(args.endpoints))
    result["source_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in (("markers", args.markers), ("endpoints", args.endpoints))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
