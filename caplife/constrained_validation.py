"""Exploratory protocol using identical release benchmark origins.

Run with PYTHONPATH=CapLife-AI and the project's analysis environment.
No feature or model fit uses measurements after a forecast's origin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from caplife.neural_benchmark import FEATURES, evaluate_transfer
from caplife.public_benchmark import load_eos2, load_pofinn
from caplife.release_validation import (
    GROUPS, GROUP_FOR_COHORT, PROJECTED_FEATURES, SCENARIOS, load_eis_loss, make_origins,
)

CANDIDATES = ("monotone_neural", "monotone_logit", "soft_physical")
WEIGHT_DECAY = 0.006


def robust_origin_features(loss: pd.DataFrame, cohort: str, horizon: float, threshold: float,
                           repeated_crossing: bool) -> pd.DataFrame:
    origins = make_origins(loss, cohort, horizon, threshold, repeated_crossing)
    additional = []
    for device, rows in origins.groupby("device", sort=False):
        cap = device.split(" / ", maxsplit=1)[1]
        observations = loss[cap].dropna()
        t = observations.index.to_numpy(dtype=float)
        y = observations.to_numpy(dtype=float)
        for row in rows.itertuples():
            i = int(np.searchsorted(t, row.age_h))
            assert t[i] == row.age_h
            window_t = t[max(0, i - 4): i + 1]
            window_y = y[max(0, i - 4): i + 1]
            left, right = np.triu_indices(len(window_t), k=1)
            robust = float(np.median((window_y[right] - window_y[left]) / (window_t[right] - window_t[left])))
            recent = (y[i] - y[i - 1]) / (t[i] - t[i - 1])
            older = (y[i - 1] - y[i - 2]) / (t[i - 1] - t[i - 2])
            durable = max(min(older, recent, robust), 0.0)
            fast = max(min(recent, 2*max(robust, 0.0)), 0.0)
            slow_projection = (y[i] + horizon * durable - threshold) / threshold
            fast_projection = (y[i] + horizon * fast - threshold) / threshold
            additional.append((row.Index, slow_projection, fast_projection))
    extra = pd.DataFrame(additional, columns=["row", "sustained", "tempered"]).set_index("row")
    return origins.join(extra)


def inputs(data: pd.DataFrame) -> np.ndarray:
    """Dimensionless, clipped physical scores without a fitted normalization."""
    headroom = -data["relative_headroom"].to_numpy()
    rapid = data["projected_margin"].to_numpy()
    softened = data["tempered"].to_numpy()
    sustained = data["sustained"].to_numpy()
    return np.clip(np.column_stack([headroom, rapid, softened, sustained]), -1, 1)


def physical_score(frame: pd.DataFrame) -> np.ndarray:
    """Fixed soft physical rule, with growing evidence across rate estimates."""
    design = inputs(frame)
    margin = design[:, 2] * 0.50 + design[:, 1] * 0.30 + design[:, 3] * 0.20
    return expit(margin / 0.15)


def train_model(train: pd.DataFrame, test: pd.DataFrame, is_neural: bool) -> np.ndarray:
    """Fit a monotone nonlinear or linear logit with equal weight per capacitor."""
    x, z = inputs(train), inputs(test)
    y = train["outcome"].to_numpy()
    weight = 1 / train.groupby("device")["device"].transform("size").to_numpy(dtype=float)
    weight /= weight.sum()
    d = x.shape[1]
    hidden = 2

    def forecast(theta: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not is_neural:
            return values @ theta[:-1] + theta[-1], np.column_stack([values, np.ones(len(values))])
        slopes = theta[:hidden*d].reshape(hidden, d)
        scores = np.tanh(values @ slopes.T + theta[hidden*d: hidden*d+hidden])
        levels = theta[hidden*d+hidden: -1]
        derivative = (1 - scores**2) * levels
        jacobian = np.column_stack([np.einsum("ij,ik->ijk", derivative, values).reshape(len(values), -1),
                                    derivative, scores, np.ones(len(values))])
        return scores @ levels + theta[-1], jacobian

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits, jacobian = forecast(theta, x)
        penalty = theta.copy()
        penalty[-1] = 0
        deviance = np.dot(weight, np.logaddexp(0, logits) - y * logits)
        gradient = ((expit(logits) - y) * weight) @ jacobian
        return deviance + WEIGHT_DECAY * np.dot(penalty, penalty), gradient + 2 * WEIGHT_DECAY * penalty

    if is_neural:
        limits = [(0, 8)]*(hidden*d) + [(-8, 8)]*hidden + [(0,8)]*hidden + [(-16, 16)]
        starts = [np.r_[np.random.default_rng(seed).uniform(0.15, 0.9, hidden*d),
                        np.random.default_rng(seed+1).uniform(-0.3, 0.3, hidden),
                        np.full(hidden, 0.6), -1]
                  for seed in (7, 19, 31)]
    else:
        limits = [(0, 8)]*d + [(-16, 16)]
        starts = [np.r_[np.full(d, 0.4), -1]]
    models = [minimize(objective, start, method="L-BFGS-B", bounds=limits, jac=True) for start in starts]
    assert all(model.success for model in models)
    return np.mean([expit(forecast(model.x, z)[0]) for model in models], axis=0)


def predict_candidate(train: pd.DataFrame, test: pd.DataFrame, method: str) -> np.ndarray:
    if method == "soft_physical":
        return physical_score(test)
    return train_model(train, test, method == "monotone_neural")


def score_predictions(data: pd.DataFrame, values: np.ndarray, cutoff: float) -> dict:
    truth = data["outcome"].astype(bool).to_numpy()
    alert = values >= cutoff
    weight = 1 / data.groupby("device")["device"].transform("size").to_numpy(dtype=float)
    value_cost = float(np.average((values - truth)**2, weights=weight))
    return {
        "brier_device": round(value_cost, 6),
        "auprc_window": round(float(average_precision_score(truth, values)), 6),
        "auprc_device_weighted": round(float(average_precision_score(truth, values, sample_weight=weight)), 6),
        "auroc_device_weighted": round(float(roc_auc_score(truth, values, sample_weight=weight)), 6),
        "true_alerts": int((alert & truth).sum()),
        "false_alerts": int((alert & ~truth).sum()),
        "missed_crossing_windows": int((~alert & truth).sum()),
        "devices_with_detected_crossings": int(data.loc[alert & truth, "device"].nunique()),
        "alert_cutoff": round(cutoff, 6),
    }


def train_cutoff(train: pd.DataFrame, method: str) -> float:
    """Train-only two-way transfer to select a cutoff with a 5% FPR budget.

    The bound is a benchmark budget. There are too few studies to infer it will
    be satisfied on a new site. Ties prefer stricter cutoffs.
    """
    oof = []
    for exp in train["experiment"].unique():
        test = train["experiment"].eq(exp)
        part = train.loc[test].copy()
        part["score"] = predict_candidate(train.loc[~test], part, method)
        oof.append(part)
    oof = pd.concat(oof)
    choices = np.r_[np.arange(0.05, 1.0, 0.025), 1.0]
    viable = []
    for cutoff in choices:
        recall = []
        for _, part in oof.groupby("experiment"):
            truth = part["outcome"].eq(1).to_numpy()
            alert = part["score"].ge(cutoff).to_numpy()
            weight = 1 / part.groupby("device")["device"].transform("size").to_numpy(dtype=float)
            fpr = np.average(alert[~truth], weights=weight[~truth])
            if fpr > 0.05:
                break
            if np.any(truth):
                recall.append(np.average(alert[truth], weights=weight[truth]))
        else:
            viable.append((np.mean(recall), cutoff))
    return float(max(viable)[1]) if viable else 1.0


def evaluate_case(origins: pd.DataFrame, scenario: str) -> dict:
    report = {}
    for exp in GROUPS:
        test_rows = origins["experiment"].eq(exp)
        train, test = origins.loc[~test_rows], origins.loc[test_rows]
        train_resolved, test_resolved = train.dropna(subset="outcome"), test.dropna(subset="outcome")
        count = int(test_resolved["outcome"].sum())
        entry = {"resolved": len(test_resolved), "unresolved": len(test) - len(test_resolved),
                 "positive_origins": count, "devices": int(test_resolved["device"].nunique()),
                 "devices_with_crossing_windows": int(test_resolved.loc[test_resolved["outcome"].eq(1), "device"].nunique()),
                 "models": {}}
        if 0 < count < len(test_resolved):
            for method in CANDIDATES:
                proba = predict_candidate(train_resolved, test_resolved, method)
                row = {"fixed_0p5": score_predictions(test_resolved, proba, 0.5)}
                if scenario == "primary":
                    row["train_selected"] = score_predictions(test_resolved, proba, train_cutoff(train_resolved, method))
                entry["models"][method] = row
            for method in ("trend", "robust", "persistence"):
                entry["models"][method] = {"fixed_0p5": score_predictions(test_resolved, test_resolved[method].to_numpy(), 0.5)}
            if scenario == "primary":
                raw, _ = evaluate_transfer(train, test, FEATURES)
                projected, _ = evaluate_transfer(train, test, PROJECTED_FEATURES)
                calibrated, _ = evaluate_transfer(train, test, ("projected_margin",))
                for method, values in {
                    "existing_neural": raw.loc[test_resolved.index, "neural"],
                    "existing_projected_neural": projected.loc[test_resolved.index, "neural"],
                    "existing_calibrated_trend": calibrated.loc[test_resolved.index, "linear"],
                }.items():
                    entry["models"][method] = {"fixed_0p5": score_predictions(test_resolved, values.to_numpy(), 0.5)}
        report[exp] = entry
    return report


def main() -> None:
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pofinn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    data, pofinn = options.data_dir, options.pofinn
    eos2, eos_mat = load_eos2((data / "eos2.zip").read_bytes())
    pofinn_raw = pofinn.read_bytes()
    source_payloads = {"EOS_DataSet.mat": eos_mat, "data_combined.csv": pofinn_raw}
    losses = {"NASA EOS-2": eos2, "PoFINN": load_pofinn(pofinn_raw)}
    alternate = {}
    for cohort in ("ES10", "ES12", "ES14"):
        file = data / f"{cohort}.csv"
        source_payloads[f"NASA {cohort} sessions"] = file.read_bytes()
        losses[f"NASA {cohort}"] = load_eis_loss(file)
        alternate[f"NASA {cohort}"] = load_eis_loss(file, fitted=True)
    report = {}
    for scenario, (horizon, threshold, repeated, fitted) in SCENARIOS.items():
        if scenario == "equal_experiment_training_weight":
            continue
        sources = (losses | alternate) if fitted else losses
        origins = pd.concat([robust_origin_features(loss, cohort, horizon, threshold, repeated)
                             for cohort, loss in sources.items()], ignore_index=True)
        origins["experiment"] = origins["cohort"].map(GROUP_FOR_COHORT)
        report[scenario] = {
            "horizon_h": horizon,
            "threshold_percent": threshold,
            "repeat_confirmation": repeated,
            "alternative_capacitance": fitted,
            "held_out": evaluate_case(origins, scenario),
        }
    result = {
        "source_sha256": {name: hashlib.sha256(payload).hexdigest()
                          for name, payload in source_payloads.items()},
        "fixed_hard_alert_cutoff": 0.5,
        "candidate_design": "fixed before these exploratory runs",
        "fixed_soft_physical": {
            "projection_coefficients": {"tempered": 0.5, "recent_ols": 0.3, "sustained": 0.2},
            "sigmoid_scale_fraction_of_threshold": 0.15,
            "margin_clip": [-1, 1],
        },
        "learned_scores": {
            "l2_regularization": WEIGHT_DECAY,
            "neural_hidden_units": 2,
            "neural_fit_seeds": [7, 19, 31],
            "training_weighting": "equal total weight per device",
        },
        "primary_cutoff_selection": {
            "folds": "two-way transfer between the two training experiments only",
            "max_weighted_train_fold_fpr": 0.05,
            "search_grid": [round(float(x), 3) for x in np.r_[np.arange(0.05, 1, 0.025), 1]],
            "optimization": "mean device-weighted recall across inner folds",
            "tie_break": "choose the higher cutoff",
        },
        "scenarios": report,
    }
    options.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
