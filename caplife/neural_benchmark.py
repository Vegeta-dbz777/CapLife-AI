"""Train and compare short-horizon models on independent public aging studies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from caplife.public_benchmark import (
    EOS2_URLS, POFINN_URL, SOURCES, THRESHOLD_PCT, load_eos2, load_pofinn, read_source,
)

HORIZON_H = 48
FEATURES = ("loss_pct", "older_rate_pct_per_h", "recent_rate_pct_per_h")
METHODS = {
    "frequency": "Training event frequency",
    "trend": "Extrapolated three-reading trend",
    "linear": "Regularized logistic regression",
    "neural": "Small neural network",
}
SEEDS = (7, 19, 31)


def make_examples(loss: pd.DataFrame) -> pd.DataFrame:
    """Label only horizons resolved by original inspections and use past-only features."""
    ages = loss.index.to_numpy(dtype=float)
    if (len(ages) < 4 or not np.isfinite(ages).all() or not np.all(np.diff(ages) > 0)
            or not np.isfinite(loss).all().all()):
        raise ValueError("Expected complete measurements at increasing inspection ages.")
    rows = []
    for device, series in loss.items():
        values = series.to_numpy(dtype=float)
        hits = np.flatnonzero(values >= THRESHOLD_PCT)
        crossing_index = int(hits[0]) if hits.size else len(ages)
        observed_above = ages[crossing_index] if hits.size else np.inf
        last_below = ages[crossing_index - 1] if crossing_index > 0 else -np.inf
        for i in range(2, min(crossing_index, len(ages) - 1)):
            recent = values[i - 2:i + 1]
            history = ages[i - 2:i + 1]
            origin = ages[i]
            end = origin + HORIZON_H
            outcome = 1.0 if observed_above <= end else 0.0 if last_below >= end else np.nan
            slope = np.polyfit(history, recent, 1)[0]
            older_rate, recent_rate = np.diff(recent) / np.diff(history)
            rows.append({
                "device": device, "origin_h": origin, "loss_pct": recent[-1],
                "older_rate_pct_per_h": older_rate, "recent_rate_pct_per_h": recent_rate,
                "outcome": outcome, "trend": float(recent[-1] + HORIZON_H * slope >= THRESHOLD_PCT),
                "last_below_h": last_below, "first_crossing_h": observed_above,
            })
    return pd.DataFrame(rows)


def fit_classifier(x: np.ndarray, y: np.ndarray, weights: np.ndarray, hidden: bool, seed: int) -> nn.Module:
    """Use matched training for the linear and nonlinear probability models."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = (nn.Sequential(nn.Linear(x.shape[1], 4), nn.Tanh(), nn.Linear(4, 1))
                 if hidden else nn.Linear(x.shape[1], 1)).double()
    inputs = torch.from_numpy(x)
    outcomes = torch.from_numpy(y)
    importance = torch.from_numpy(weights)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(inputs).flatten()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, outcomes, weight=importance, reduction="sum",
        )
        penalty = sum(parameter.square().sum() for name, parameter in model.named_parameters()
                      if name.endswith("weight"))
        objective = loss + 0.02 * penalty
        objective.backward()
        return objective

    optimizer.step(closure)
    return model.eval()


def evaluate_transfer(train: pd.DataFrame, test: pd.DataFrame,
                      features: tuple[str, ...] = FEATURES,
                      equal_weight_by: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    train = train.loc[train["outcome"].notna()]
    if train["outcome"].nunique() != 2:
        raise ValueError("Training requires both resolved crossing and no-crossing examples.")
    weights = 1 / train.groupby("device")["device"].transform("size").to_numpy(dtype=float)
    if equal_weight_by is not None:
        weights /= train.groupby(equal_weight_by)["device"].transform("nunique").to_numpy(dtype=float)
    weights /= weights.sum()
    raw_train = train[list(features)].to_numpy(dtype=float)
    raw_test = test[list(features)].to_numpy(dtype=float)
    center = np.average(raw_train, axis=0, weights=weights)
    scale = np.sqrt(np.average((raw_train - center) ** 2, axis=0, weights=weights))
    scale[scale == 0] = 1
    x = (raw_train - center) / scale
    y = train["outcome"].to_numpy(dtype=float)
    z = torch.from_numpy((raw_test - center) / scale)
    neural_models = [fit_classifier(x, y, weights, True, seed) for seed in SEEDS]
    logistic = fit_classifier(x, y, weights, False, SEEDS[0])
    with torch.no_grad():
        predictions = torch.stack([model(z).flatten().sigmoid() for model in neural_models]).numpy()
        linear = logistic(z).flatten().sigmoid().numpy()
    supported = ((raw_test >= raw_train.min(axis=0)) & (raw_test <= raw_train.max(axis=0))).all(axis=1)
    detail = test.assign(
        frequency=float(np.average(y, weights=weights)), linear=linear,
        neural=predictions.mean(axis=0), neural_seed_low=predictions.min(axis=0),
        neural_seed_high=predictions.max(axis=0), within_training_feature_ranges=supported,
    )
    resolved = detail.loc[detail["outcome"].notna()]
    summary = []
    for method in METHODS:
        error = (resolved[method] - resolved["outcome"]) ** 2
        per_device_error = error.groupby(resolved["device"]).mean()
        alert = resolved[method].ge(0.5)
        hit = resolved["outcome"].eq(1)
        summary.append({
            "method": METHODS[method], "training_devices": int(train["device"].nunique()),
            "test_devices": int(resolved["device"].nunique()), "resolved": len(resolved),
            "unresolved": len(detail) - len(resolved), "resolved_crossings": int(hit.sum()),
            "brier_score": float(per_device_error.mean()), "true_alerts": int((alert & hit).sum()),
            "false_alerts": int((alert & ~hit).sum()), "missed_crossings": int((~alert & hit).sum()),
        })
    return detail, summary


def load_es10(path: Path) -> pd.DataFrame:
    """Reconstruct device-relative loss from the audited public session export."""
    data = pd.read_csv(path).rename(columns={
        "Recorded_Reference_Age_h": "age_h", "Median_Cs_100Hz_uF": "capacitance",
    })
    if not np.isfinite(data[["age_h", "capacitance"]]).all().all() or (data["capacitance"] <= 0).any():
        raise ValueError("ES10 sessions require finite ages and positive capacitance.")
    capacitance = data.pivot_table(index="age_h", columns="Cap_ID", values="capacitance", aggfunc="median")
    if capacitance.index[0] != 0:
        raise ValueError("ES10 sessions require age-zero measurements.")
    return 100 * (1 - capacitance.div(capacitance.iloc[0]))


def build_benchmark(eos2: bytes, pofinn: bytes, es10_path: Path | None = None) -> dict:
    eos2_loss, eos2_mat = load_eos2(eos2)
    sources = {"NASA EOS-2": eos2_loss, "PoFINN": load_pofinn(pofinn)}
    examples = {name: make_examples(loss) for name, loss in sources.items()}
    results = {}
    for train_study, test_study in (("NASA EOS-2", "PoFINN"), ("PoFINN", "NASA EOS-2")):
        detail, summary = evaluate_transfer(examples[train_study], examples[test_study])
        detail = detail.replace([np.inf, -np.inf], np.nan).round(6)
        traces = sources[test_study].rename_axis("age_h").reset_index().melt(
            id_vars="age_h", var_name="device", value_name="loss_pct",
        )
        results[test_study] = {
            "train_study": train_study, "summary": summary,
            "predictions": json.loads(detail.to_json(orient="records", double_precision=6)),
            "traces": json.loads(traces.to_json(orient="records", double_precision=6)),
        }
    benchmark = {
        "horizon_h": HORIZON_H, "threshold_pct": THRESHOLD_PCT, "sources": SOURCES,
        "source_sha256": {
            "EOS_DataSet.mat": hashlib.sha256(eos2_mat).hexdigest(),
            "data_combined.csv": hashlib.sha256(pofinn).hexdigest(),
        },
        "results": results,
    }
    if es10_path is not None:
        es10 = make_examples(load_es10(es10_path))
        pooled = pd.concat([frame.assign(device=name + " / " + frame["device"])
                            for name, frame in examples.items()], ignore_index=True)
        detail, summary = evaluate_transfer(pooled, es10)
        benchmark["es10_stress_check"] = {
            "source": "https://data.phmsociety.org/nasa/",
            "session_export_sha256": hashlib.sha256(es10_path.read_bytes()).hexdigest(),
            "summary": summary,
            "outside_training_ranges": int((~detail["within_training_feature_ranges"]).sum()),
        }
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path)
    parser.add_argument("--pofinn", type=Path)
    parser.add_argument("--es10-sessions", type=Path, help="Audited NASA ES10 session CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts/neural_benchmark.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = build_benchmark(read_source(args.eos2, EOS2_URLS), read_source(args.pofinn, (POFINN_URL,)),
                             args.es10_sessions)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
