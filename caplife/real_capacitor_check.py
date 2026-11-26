"""Test the built-in six-parameter model on real capacitors from public aging studies.

At every reading before a capacitor crossed its capacitance-loss limit, the model predicts the remaining time to
that limit; the prediction is compared with when the capacitor actually crossed it (interpolated between the
readings either side). A trend line through the last three readings is the simple comparison.

Run ``python -m caplife.real_capacitor_check`` after retraining the built-in model; the page shows the stored
results only while they belong to the model in ``artifacts/six_rul``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caplife.public_benchmark import EOS2_URLS, POFINN_URL, load_eos2, load_pofinn, read_source

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "six_rul"
OUTPUT = MODEL_DIR / "real_capacitor_check.json"
ES14_FILE = ROOT / "sample_data" / "nasa_es14_reference.csv"
# ES14 capacitance drops about 14% in the first two hours (running in), so its baseline is the reading near 23 h.
ES14_SETTLED_H = 23
STUDIES = {
    "NASA EOS-2": {"limit_pct": 15, "source": "https://data.nasa.gov/dataset/capacitor-electrical-stress-2",
                   "stress": "electrical overstress"},
    "PoFINN": {"limit_pct": 20, "source": "https://github.com/Anurag-Dutta/PoFINN",
               "stress": "electrical overstress"},
    "NASA ES14": {"limit_pct": 5, "source": "sample_data/nasa_es14_reference.csv",
                  "stress": "electrical stress, slow aging"},
}


def model_fingerprint(model_dir: Path = MODEL_DIR) -> str:
    """Identifies the built-in model the results belong to."""
    digest = hashlib.sha256()
    for name in ("model.pt", "metadata.json"):
        digest.update((model_dir / name).read_bytes())
    return digest.hexdigest()[:16]


def from_loss(loss: pd.DataFrame, study: str) -> pd.DataFrame:
    """Readings from a table of capacitance loss (%) by age, one column per capacitor."""
    rows = [{"Cap_ID": f"{study}-{cap}", "Hours": float(hours), "Capacitance_uF": 1000 * (1 - pct / 100)}
            for cap in loss.columns for hours, pct in loss[cap].dropna().items()]
    return pd.DataFrame(rows)


def es14_readings(path: Path = ES14_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["Age_h"] >= ES14_SETTLED_H].copy()
    frame["Hours"] = frame["Age_h"] - ES14_SETTLED_H
    frame = frame.groupby(["Cap_ID", "Hours"], as_index=False).mean(numeric_only=True)
    frame["Cap_ID"] = "ES14-" + frame["Cap_ID"].astype(str)
    return frame.rename(columns={"Resistance_Ohm": "ESR_ohm"})[["Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm"]]


def _loss(history: pd.DataFrame) -> np.ndarray:
    return ((1 - history["Capacitance_uF"] / history["Capacitance_uF"].iloc[0]) * 100).to_numpy()


def crossing_h(history: pd.DataFrame, limit_pct: float) -> float | None:
    """When the capacitance loss reached the limit, interpolated between readings; None if it never did."""
    loss, hours = _loss(history), history["Hours"].to_numpy()
    hit = np.flatnonzero(loss >= limit_pct)
    if not hit.size:
        return None
    k = hit[0]
    if k == 0:
        return float(hours[0])
    return float(hours[k - 1] + (limit_pct - loss[k - 1]) / (loss[k] - loss[k - 1]) * (hours[k] - hours[k - 1]))


def trend_h(history: pd.DataFrame, limit_pct: float) -> float:
    """Time to the limit along a straight line through the last three readings."""
    loss = _loss(history)
    slope = np.polyfit(history["Hours"].to_numpy()[-3:], loss[-3:], 1)[0]
    return float((limit_pct - loss[-1]) / slope) if slope > 0 else np.inf


def evaluate(model, readings: pd.DataFrame, limit_pct: float, measured: tuple[str, ...]) -> dict:
    from caplife.six_rul import full_limits

    limits = full_limits({"capacitance_loss_pct": limit_pct})
    rows = []
    for _, history in readings.groupby("Cap_ID"):
        history = history.sort_values("Hours").reset_index(drop=True)
        end = crossing_h(history, limit_pct)
        if end is None:
            continue
        assessed = model.assess(history, measured, limits)
        for k in range(2, len(history)):  # a prediction needs at least three readings
            age = history["Hours"].iloc[k]
            if age >= end:
                break
            last = assessed.iloc[k]
            rows.append({"truth": end - age, "low": last["RUL_low_h"], "mid": last["RUL_median_h"],
                         "high": last["RUL_high_h"], "trend": trend_h(history.iloc[:k + 1], limit_pct)})
    seen = pd.DataFrame(rows)
    relative = lambda column: float(np.median(np.abs(seen[column] - seen["truth"]) / np.maximum(seen["truth"], 1.0)))
    return {
        "capacitors": int(readings["Cap_ID"].nunique()),
        "crossed": int(sum(crossing_h(h.sort_values("Hours"), limit_pct) is not None
                           for _, h in readings.groupby("Cap_ID"))),
        "predictions": len(seen),
        "median_true_rul_h": float(seen["truth"].median()),
        "median_predicted_rul_h": float(seen["mid"].median()),
        "median_absolute_error_h": float(np.median(np.abs(seen["mid"] - seen["truth"]))),
        "median_relative_error": relative("mid"),
        "trend_median_relative_error": relative("trend"),
        "interval_coverage": float(((seen["truth"] >= seen["low"]) & (seen["truth"] <= seen["high"])).mean()),
        "share_failed_before_range": float((seen["truth"] < seen["low"]).mean()),
        "share_failed_after_range": float((seen["truth"] > seen["high"]).mean()),
    }


def build(eos2: bytes, pofinn: bytes, model_dir: Path = MODEL_DIR) -> dict:
    from caplife.six_rul import SixRULModel

    model = SixRULModel(model_dir)
    readings = {"NASA EOS-2": from_loss(load_eos2(eos2)[0], "EOS2"), "PoFINN": from_loss(load_pofinn(pofinn), "PoFINN"),
                "NASA ES14": es14_readings()}
    studies = []
    for name, study in STUDIES.items():
        measured = ("Capacitance_uF", "ESR_ohm") if "ESR_ohm" in readings[name] else ("Capacitance_uF",)
        result = evaluate(model, readings[name], study["limit_pct"], measured)
        studies.append({"study": name, **study, "measured": list(measured),
                        "age_range_h": float(readings[name]["Hours"].max()), **result})
    return {"model_fingerprint": model_fingerprint(model_dir), "studies": studies}


def stored(model_dir: Path = MODEL_DIR) -> dict | None:
    """The stored results, or None when they are missing or belong to another version of the model."""
    path = model_dir / OUTPUT.name
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result if result.get("model_fingerprint") == model_fingerprint(model_dir) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path, help="Local NASA EOS-2 ZIP or MAT file")
    parser.add_argument("--pofinn", type=Path, help="Local original PoFINN data_combined.csv")
    args = parser.parse_args()
    result = build(read_source(args.eos2, EOS2_URLS), read_source(args.pofinn, (POFINN_URL,)))
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
