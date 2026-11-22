"""Reproduce the cross-study forecasts and inspection-age sensitivity check."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm


LANDMARK_H = 94
# Pair actual inspection ages to keep every forecast on its study's measured clock.
INSPECTION_PAIRS_H = ((47, 46), (71, 70), (94, 94), (116, 118), (139, 142), (161, 166))
THRESHOLD_PCT = 20
BAND_MASS = 0.80
EOS2_URLS = (
    "https://data.nasa.gov/docs/legacy/EOS_DataSet.zip",
    "https://phm-datasets.s3.amazonaws.com/NASA/14.+Capacitor+Electrical+Stress+-+2.zip",
)
POFINN_URL = (
    "https://raw.githubusercontent.com/Anurag-Dutta/PoFINN/"
    "6fe0fa7dd3df31b2dd62bd148182cb85f19494c7/DATA/data_combined.csv"
)
SOURCES = {
    "NASA EOS-2": "https://data.nasa.gov/dataset/capacitor-electrical-stress-2",
    "PoFINN": "https://github.com/Anurag-Dutta/PoFINN/blob/"
    "6fe0fa7dd3df31b2dd62bd148182cb85f19494c7/DATA/data_combined.csv",
}


def read_source(path: Path | None, urls: tuple[str, ...]) -> bytes:
    if path is not None:
        return path.read_bytes()
    for url in urls:
        try:
            with urlopen(url, timeout=20) as response:
                return response.read()
        except OSError as error:
            last_error = error
    raise OSError(f"Could not retrieve {urls[0]}. Supply a local source file.") from last_error


def load_eos2(payload: bytes) -> tuple[pd.DataFrame, bytes]:
    if payload.startswith(b"PK\x03\x04"):
        with ZipFile(io.BytesIO(payload)) as archive:
            [member] = [name for name in archive.namelist()
                        if Path(name).name == "EOS_DataSet.mat"]
            payload = archive.read(member)
    data = loadmat(io.BytesIO(payload), simplify_cells=True)
    loss = pd.DataFrame(data["C"], index=data["aging_time"],
                        columns=[f"C{i}" for i in range(1, 7)])
    return loss, payload


def load_pofinn(payload: bytes) -> pd.DataFrame:
    capacitance = pd.read_csv(io.BytesIO(payload), index_col="Ageing Time")
    capacitance = capacitance.loc[:, [f"C{i}" for i in range(1, 9)]]
    return 100 * (1 - capacitance.div(capacitance.iloc[0]))


def landmark_outcomes(loss: pd.DataFrame, landmark_h: int = LANDMARK_H) -> pd.DataFrame:
    """Use originals to bracket first recorded crossings and preserve right censoring."""
    ages = loss.index.to_numpy(dtype=float)
    if (not np.isfinite(ages).all() or not np.isfinite(loss).all().all()
            or ages[0] != 0 or not np.all(np.diff(ages) > 0)
            or landmark_h not in ages):
        raise ValueError(f"Expected complete, increasing inspections from zero including {landmark_h} h.")
    history = loss.loc[:landmark_h]
    if len(history) < 3 or landmark_h >= ages[-1]:
        raise ValueError("The origin must have three inspections and later follow-up.")
    eligible = loss.columns[~(history >= THRESHOLD_PCT).any()]
    if eligible.empty:
        raise ValueError("No devices are still below the threshold at the forecast origin.")

    rows = []
    for device, series in loss[eligible].items():
        values = series.to_numpy()
        hits = np.flatnonzero(values >= THRESHOLD_PCT)
        lower_h, upper_h = (ages[hits[0] - 1], ages[hits[0]]) if hits.size else (ages[-1], np.inf)
        recent = history[device].iloc[-3:]
        slope = np.polyfit(recent.index.to_numpy(dtype=float), recent.to_numpy(), 1)[0]
        if slope <= 0:
            raise ValueError(f"Cannot extrapolate a positive loss trend for {device}.")
        rows.append({
            "device": device,
            "loss_at_origin_pct": float(recent.iloc[-1]),
            "rul_lower_h": lower_h - landmark_h,
            "rul_upper_h": upper_h - landmark_h,
            "trend_h": (THRESHOLD_PCT - recent.iloc[-1]) / slope,
        })
    return pd.DataFrame(rows)


def fit_aft(train: pd.DataFrame, offset: np.ndarray) -> tuple[float, float]:
    """Fit a log-normal time model using crossing windows and right-censored records."""
    lower = train["rul_lower_h"].to_numpy()
    upper = train["rul_upper_h"].to_numpy()
    finite = np.isfinite(upper)
    log_lower = np.log(lower, where=lower > 0, out=np.full_like(lower, -np.inf))
    log_upper = np.log(upper)

    def neg_log_likelihood(parameters: np.ndarray) -> float:
        shift, log_scale = parameters
        scale = np.exp(log_scale)
        z_lower = (log_lower - offset - shift) / scale
        z_upper = (log_upper[finite] - offset[finite] - shift) / scale
        z_left = z_lower[finite]
        # Evaluate each interval in the nearer tail to avoid subtracting rounded CDFs.
        use_survival = z_left >= 0
        larger = np.where(use_survival, log_ndtr(-z_left), log_ndtr(z_upper))
        smaller = np.where(use_survival, log_ndtr(-z_upper), log_ndtr(z_left))
        log_interval = larger + np.log(-np.expm1(smaller - larger))
        return float(-log_interval.sum() - log_ndtr(-z_lower[~finite]).sum())

    initial_shift = np.median(log_upper[finite] - offset[finite])
    result = minimize(neg_log_likelihood, [initial_shift, np.log(0.5)], method="L-BFGS-B",
                      bounds=[(None, None), (np.log(0.15), np.log(3))])
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Could not fit the time model: {result.message}")
    shift, log_scale = result.x
    return float(shift), float(np.exp(log_scale))


def evaluate(train: pd.DataFrame, test: pd.DataFrame, train_name: str,
             test_name: str) -> tuple[list[dict], list[dict]]:
    age_shift, age_scale = fit_aft(train, np.zeros(len(train)))
    trend_shift, trend_scale = fit_aft(train, np.log(train["trend_h"].to_numpy()))
    band_z = norm.ppf((1 + BAND_MASS) / 2)
    trend = test["trend_h"].to_numpy()
    lower = test["rul_lower_h"].to_numpy()
    upper = test["rul_upper_h"].to_numpy()
    finite = np.isfinite(upper)
    summaries, predictions = [], []

    for model, point, scale in (
        ("Age only", np.full(len(test), np.exp(age_shift)), age_scale),
        ("Recent trend", trend, None),
        ("Recent trend with study calibration", trend * np.exp(trend_shift), trend_scale),
    ):
        point_hit = (point > lower) & (point <= upper)
        gap = np.maximum.reduce([lower - point, point - upper, np.zeros(len(test))])
        low_band = point * np.exp(-band_z * scale) if scale is not None else None
        high_band = point * np.exp(band_z * scale) if scale is not None else None
        overlap = (high_band > lower) & (low_band <= upper) if low_band is not None else None
        contains = (low_band <= lower) & (high_band >= upper) if low_band is not None else None
        summaries.append({
            "train_study": train_name,
            "test_study": test_name,
            "model": model,
            "devices": len(test),
            "observed_crossings": int(finite.sum()),
            "right_censored": int((~finite).sum()),
            "mean_gap_h": round(float(gap[finite].mean()), 4),
            "point_hits": int(point_hit[finite].sum()),
            "band_overlaps": int(overlap[finite].sum()) if overlap is not None else None,
            "band_contains": int(contains[finite].sum()) if contains is not None else None,
            "median_band_width_h": round(float(np.median(high_band - low_band)), 4)
            if low_band is not None else None,
            "censored_point_at_or_before_last_below": int((point[~finite] <= lower[~finite]).sum()),
        })
        for i, row in enumerate(test.itertuples(index=False)):
            predictions.append({
                "train_study": train_name,
                "test_study": test_name,
                "model": model,
                "device": row.device,
                "loss_at_origin_pct": round(row.loss_at_origin_pct, 4),
                "rul_lower_h": row.rul_lower_h,
                "rul_upper_h": row.rul_upper_h if finite[i] else None,
                "forecast_h": round(float(point[i]), 4),
                "band_lower_h": round(float(low_band[i]), 4) if low_band is not None else None,
                "band_upper_h": round(float(high_band[i]), 4) if high_band is not None else None,
            })
    return summaries, predictions


def build_benchmark(eos2: bytes, pofinn: bytes) -> dict:
    eos2_loss, eos2_mat = load_eos2(eos2)
    pofinn_loss = load_pofinn(pofinn)
    summary, predictions, sensitivity = [], [], []
    for eos2_age, pofinn_age in INSPECTION_PAIRS_H:
        studies = {
            "NASA EOS-2": landmark_outcomes(eos2_loss, eos2_age),
            "PoFINN": landmark_outcomes(pofinn_loss, pofinn_age),
        }
        for train_study, test_study in (("NASA EOS-2", "PoFINN"), ("PoFINN", "NASA EOS-2")):
            metrics, detail = evaluate(studies[train_study], studies[test_study], train_study, test_study)
            if eos2_age == pofinn_age == LANDMARK_H:
                summary.extend(metrics)
                predictions.extend(detail)
            sensitivity.extend({"eos2_age_h": eos2_age, "pofinn_age_h": pofinn_age, **row}
                               for row in metrics)
    return {
        "reference_age_h": LANDMARK_H,
        "capacitance_loss_threshold_pct": THRESHOLD_PCT,
        "nominal_band_mass": BAND_MASS,
        "sources": SOURCES,
        "source_sha256": {
            "EOS_DataSet.mat": hashlib.sha256(eos2_mat).hexdigest(),
            "data_combined.csv": hashlib.sha256(pofinn).hexdigest(),
        },
        "summary": summary,
        "predictions": predictions,
        "inspection_age_sensitivity": sensitivity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path, help="Local NASA EOS-2 ZIP or MAT file")
    parser.add_argument("--pofinn", type=Path, help="Local original PoFINN data_combined.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts/public_benchmark.json"))
    args = parser.parse_args()
    result = build_benchmark(read_source(args.eos2, EOS2_URLS), read_source(args.pofinn, (POFINN_URL,)))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
