"""Evaluate alternative threshold-warning models with entire experiments held out."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caplife.nasa_eis import SOURCE_URL
from caplife.neural_benchmark import FEATURES, evaluate_transfer
from caplife.public_benchmark import EOS2_URLS, POFINN_URL, SOURCES, load_eos2, load_pofinn, read_source

CAMPAIGN = "NASA ES10/12/14"
GROUPS = {
    "NASA EOS-2": ("NASA EOS-2",),
    "PoFINN": ("PoFINN",),
    CAMPAIGN: ("NASA ES10", "NASA ES12", "NASA ES14"),
}
GROUP_FOR_COHORT = {cohort: study for study, cohorts in GROUPS.items() for cohort in cohorts}
PROJECTED_FEATURES = ("projected_margin", "robust_margin", "relative_headroom")
CANDIDATES = {
    "persistence": "Below-threshold persistence",
    "trend": "Three-reading linear trend",
    "robust": "Recent median-slope trend",
    "calibrated_trend": "Logistic calibration of projected loss",
    "neural": "Capacitance and rates neural network",
    "neural_projected": "Projection-based neural network",
}
SCENARIOS = {
    "primary": (48, 20, False, False),
    "shorter_horizon": (24, 20, False, False),
    "longer_horizon": (72, 20, False, False),
    "lower_threshold": (48, 18, False, False),
    "higher_threshold": (48, 22, False, False),
    "repeated_crossing": (48, 20, True, False),
    "alternate_capacitance": (48, 20, False, True),
    "equal_experiment_training_weight": (48, 20, False, False),
}


def load_eis_loss(path: Path, fitted: bool = False) -> pd.DataFrame:
    signal = "Fitted_capacitance_uF" if fitted else "Capacitance_uF"
    measurements = pd.read_csv(path).dropna(subset=[signal])
    pivot = measurements.pivot_table(index="Age_h", columns="Cap_ID", values=signal, aggfunc="median")
    if pivot.empty or pivot.index[0] != 0 or not np.isfinite(pivot.iloc[0]).all() or (pivot <= 0).any().any():
        raise ValueError(f"Expected a positive hour-zero capacitance for every device in {path.name}.")
    return 100 * (1 - pivot.div(pivot.iloc[0]))


def make_origins(loss: pd.DataFrame, cohort: str, horizon: float,
                 threshold: float, repeated: bool = False) -> pd.DataFrame:
    """Label crossings between actual inspections and withhold unresolved outcomes."""
    ages = loss.index.to_numpy(dtype=float)
    if not np.isfinite(ages).all() or not np.all(np.diff(ages) > 0):
        raise ValueError("Inspection ages must be finite and strictly increasing.")
    rows = []
    for device, series in loss.items():
        available = series.notna()
        y, t = series[available].to_numpy(dtype=float), ages[available.to_numpy()]
        if not np.isfinite(y).all():
            raise ValueError("Nonmissing loss measurements must be finite.")
        above = y >= threshold - 1e-10
        crossing = (above & np.r_[False, above[:-1]]) if repeated and len(y) else above
        hits = np.flatnonzero(crossing)
        crossing_i = hits[0] if hits.size else len(t)
        last_below = t[crossing_i - 1] if crossing_i else -np.inf
        first_above = t[crossing_i] if hits.size else np.inf
        for i in range(2, min(crossing_i, len(t) - 1)):
            current = y[i]
            history_time, history_loss = t[i - 2:i + 1], y[i - 2:i + 1]
            slope = np.polyfit(history_time - t[i], history_loss, 1)[0]
            longer_time, longer_loss = t[max(0, i - 4):i + 1], y[max(0, i - 4):i + 1]
            left, right = np.triu_indices(len(longer_time), k=1)
            robust_slope = np.median((longer_loss[right] - longer_loss[left])
                                     / (longer_time[right] - longer_time[left]))
            older, recent = np.diff(history_loss) / np.diff(history_time)
            projection = (current + horizon * slope - threshold) / threshold
            robust_projection = (current + horizon * robust_slope - threshold) / threshold
            end = t[i] + horizon
            rows.append({
                "device": f"{cohort} / {device}", "cohort": cohort, "age_h": t[i],
                "loss_pct": current, "older_rate_pct_per_h": older, "recent_rate_pct_per_h": recent,
                "relative_headroom": (threshold - current) / threshold,
                "projected_margin": projection, "robust_margin": robust_projection,
                "outcome": 1.0 if first_above <= end else 0.0 if last_below >= end else np.nan,
                "persistence": 0.0, "trend": float(projection >= 0), "robust": float(robust_projection >= 0),
            })
    return pd.DataFrame(rows)


def score_predictions(detail: pd.DataFrame) -> dict:
    resolved = detail[detail["outcome"].notna()]
    positive = resolved["outcome"].eq(1)
    by_method = {}
    for method in CANDIDATES:
        prediction = resolved[method]
        alert = prediction.ge(0.5)
        device_brier = (prediction - resolved["outcome"]).pow(2).groupby(resolved["device"]).mean()
        by_method[method] = {
            "brier": round(float(device_brier.mean()), 6) if len(device_brier) else None,
            "true_alerts": int((alert & positive).sum()), "false_alerts": int((alert & ~positive).sum()),
            "missed_crossings": int((~alert & positive).sum()),
        }
    return {
        "capacitors": int(detail["device"].nunique()),
        "resolved": len(resolved), "unresolved": len(detail) - len(resolved),
        "positive_origins": int(positive.sum()),
        "crossing_devices": int(resolved.loc[positive, "device"].nunique()),
        "by_method": by_method,
    }


def evaluate_group(train: pd.DataFrame, test: pd.DataFrame, equal_experiments: bool) -> dict:
    weight_by = "experiment" if equal_experiments else None
    raw, _ = evaluate_transfer(train, test, FEATURES, weight_by)
    projection, _ = evaluate_transfer(train, test, PROJECTED_FEATURES, weight_by)
    calibrated, _ = evaluate_transfer(train, test, ("projected_margin",), weight_by)
    scores = raw.assign(
        neural_projected=projection["neural"],
        calibrated_trend=calibrated["linear"],
    )
    resolved = scores.loc[scores["outcome"].notna()]
    calibration = {}
    for method in ("calibrated_trend", "neural", "neural_projected"):
        quantized = pd.cut(resolved[method], [0, .25, .5, .75, 1], include_lowest=True, labels=False)
        weights = 1 / resolved.groupby("device")["device"].transform("size")
        calibration[method] = [
            {
                "bin_upper": (int(bucket) + 1) * .25,
                "origins": len(part), "devices": int(part["device"].nunique()),
                "mean_score": round(float(np.average(part[method], weights=weights.loc[part.index])), 6),
                "observed_frequency": round(float(np.average(part["outcome"], weights=weights.loc[part.index])), 6),
            }
            for bucket, part in resolved.groupby(quantized)
        ]
    return {
        **score_predictions(scores),
        "cohorts": {cohort: score_predictions(part) for cohort, part in scores.groupby("cohort")},
        "outside_training_ranges": {
            "neural": int((~raw.loc[resolved.index, "within_training_feature_ranges"]).sum()),
            "neural_projected": int((~projection.loc[resolved.index, "within_training_feature_ranges"]).sum()),
        },
        "calibration": calibration,
    }


def crossing_ages(loss: pd.DataFrame, threshold: float = 20) -> dict:
    first_crossings = {}
    for device, series in loss.items():
        peaks = series[series >= threshold - 1e-10]
        first_crossings[device] = None if peaks.empty else float(peaks.index.min())
    return first_crossings


def build_validation(losses: dict[str, pd.DataFrame], fitted_losses: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for scenario, (horizon, threshold, repeated, fitted) in SCENARIOS.items():
        sources = losses | fitted_losses if fitted else losses
        origins = pd.concat([make_origins(frame, name, horizon, threshold, repeated)
                             for name, frame in sources.items()], ignore_index=True)
        origins["experiment"] = origins["cohort"].map(GROUP_FOR_COHORT)
        equal_experiments = scenario == "equal_experiment_training_weight"
        held_out = {}
        for experiment, cohorts in GROUPS.items():
            test = origins["experiment"].eq(experiment)
            held_out[experiment] = evaluate_group(origins.loc[~test], origins.loc[test], equal_experiments)
            held_out[experiment]["devices_in_experiment"] = sum(len(sources[name].columns) for name in cohorts)
        results[scenario] = {
            "horizon_h": horizon, "threshold_pct": threshold, "two_consecutive_readings": repeated,
            "use_fitted_capacitance_in_multivoltage_experiment": fitted,
            "equal_experiment_training_weight": equal_experiments,
            "held_out": held_out,
        }

    return {
        "sources": {**SOURCES, CAMPAIGN: SOURCE_URL},
        "methods": CANDIDATES,
        "cohort_ages": {
            name: {"devices": len(frame.columns), "start_h": float(frame.index[0]), "end_h": float(frame.index[-1]),
                   "first_crossings_h": crossing_ages(frame)}
            for name, frame in losses.items()
        },
        "alternative_first_crossings_h": {name: crossing_ages(frame) for name, frame in fitted_losses.items()},
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path)
    parser.add_argument("--pofinn", type=Path)
    parser.add_argument("--es10-sessions", type=Path, required=True)
    parser.add_argument("--es12-sessions", type=Path, required=True)
    parser.add_argument("--es14-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    eos, nasa_mat = load_eos2(read_source(args.eos2, EOS2_URLS))
    pofinn = read_source(args.pofinn, (POFINN_URL,))
    paths = {"NASA ES10": args.es10_sessions, "NASA ES12": args.es12_sessions, "NASA ES14": args.es14_sessions}
    result = build_validation({
        "NASA EOS-2": eos, "PoFINN": load_pofinn(pofinn),
        **{name: load_eis_loss(path) for name, path in paths.items()},
    }, {name: load_eis_loss(path, fitted=True) for name, path in paths.items()})
    payloads = {
        "EOS_DataSet.mat": nasa_mat, "data_combined.csv": pofinn,
        **{f"{name} sessions": path.read_bytes() for name, path in paths.items()},
    }
    result["source_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
