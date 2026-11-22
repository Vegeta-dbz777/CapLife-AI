"""Compare capacitance and ESR features using device- and study-held-out tests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.io import loadmat

from caplife.neural_benchmark import FEATURES, HORIZON_H, evaluate_transfer
from caplife.public_benchmark import EOS2_URLS, load_eos2, read_source

ARTICLE = "https://doi.org/10.3390/electronics14224452"
FIGURE_ROOT = "https://mdpi-res.com/electronics/electronics-14-04452/article_deploy/html/images"
COLORS = np.array([
    [77, 77, 77], [255, 65, 65], [0, 106, 226], [0, 175, 110],
    [197, 122, 226], [221, 154, 0], [0, 255, 190],
])
ESR_FEATURES = ("esr_older_log_rate", "esr_recent_log_rate")


def digitize_figure(payload: bytes, capacitance: bool) -> pd.DataFrame:
    """Read distinguishable trace centers at 20-hour samples from 560 to 960 h."""
    pixels = np.array(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=float)
    # Pixel positions of the 0 and 800 h ticks and two y ticks in the published PNGs.
    if capacitance:
        shape, x0, x800, y0, y_top, scale = (1615, 2032), 288, 1683, 1290, 177, 20
        start, end, trace_width, gray_clearance = 160, 1110, 12, 40
    else:
        shape, x0, x800, y0, y_top, scale = (1588, 1937), 246, 1597, 1245, 102, 0.16
        start, end, trace_width, gray_clearance = 115, 1030, 13, 45
    if pixels.shape[:2] != shape:
        raise ValueError("Figure dimensions differ from the calibrated public images.")
    table = pd.DataFrame(np.nan, index=pd.Index(range(560, 961, 20), name="age_h"),
                         columns=[f"Cap{i}" for i in range(1, 9)])
    for age in table.index:
        x = round(x0 + (x800 - x0) * age / 800)
        strip = pixels[start:end, x - 2:x + 3]
        distance = np.linalg.norm(strip[:, :, None, :] - COLORS, axis=-1).min(axis=1)
        for j in range(len(COLORS)):
            rows = np.flatnonzero(distance[:, j] < 23)
            if len(rows) >= 2 and np.ptp(rows) <= trace_width:
                y = start + np.average(rows, weights=25 - distance[rows, j])
                table.loc[age, f"Cap{j + 1}"] = scale * (y0 - y) / (y0 - y_top)
        dark = np.flatnonzero(distance[:, 0] < 23)
        if not len(dark):
            continue
        gray = np.all(strip == strip[:, :, 0, None], axis=2)
        gray &= (strip[:, :, 0] >= 180) & (strip[:, :, 0] <= (208 if capacitance else 209))
        gray_rows = np.flatnonzero(gray.sum(axis=1) >= 2)
        gray_rows = gray_rows[gray_rows < np.median(dark) - gray_clearance]
        if len(gray_rows) >= 2 and np.ptp(gray_rows) <= trace_width:
            table.loc[age, "Cap8"] = scale * (y0 - start - np.median(gray_rows)) / (y0 - y_top)
    return table


def paired_examples(loss: pd.DataFrame, esr: pd.DataFrame, threshold: float,
                    margin: float = 0.0) -> pd.DataFrame:
    """Use identical past-only inputs and crossing labels for both model variants."""
    if not loss.index.equals(esr.index) or not loss.columns.equals(esr.columns):
        raise ValueError("Capacitance and ESR must share the original device and inspection grid.")
    ages = loss.index.to_numpy(dtype=float)
    if len(ages) < 3 or not np.isfinite(ages).all() or np.any(np.diff(ages) <= 0):
        raise ValueError("The inspection grid must contain at least three strictly increasing finite ages.")
    if not np.isfinite([threshold, margin]).all() or not 0 <= margin < threshold:
        raise ValueError("The crossing threshold must exceed the nonnegative margin.")
    rows = []
    for device in loss:
        capacitance = loss[device].to_numpy(dtype=float)
        resistance = esr[device].to_numpy(dtype=float)
        possible = np.flatnonzero(capacitance >= threshold - margin)
        definite = np.flatnonzero(capacitance >= threshold + margin)
        first_possible = possible[0] if possible.size else len(ages)
        first_above = ages[definite[0]] if definite.size else np.inf
        earlier = np.flatnonzero(capacitance[:first_possible] < threshold - margin)
        if not len(earlier):
            continue
        last_below = ages[earlier[-1]]
        for i in range(2, min(first_possible, len(ages) - 1)):
            cap_history, esr_history = capacitance[i - 2:i + 1], resistance[i - 2:i + 1]
            if (not np.isfinite(cap_history).all() or not np.isfinite(esr_history).all()
                    or (esr_history <= 0).any()):
                continue
            history = ages[i - 2:i + 1]
            end = ages[i] + HORIZON_H
            outcome = 1 if first_above <= end else 0 if last_below >= end else np.nan
            slope = np.polyfit(history, cap_history, 1)[0]
            cap_rates = np.diff(cap_history) / np.diff(history)
            # Log ratios cancel fixed ESR units while retaining relative change per hour.
            esr_rates = np.diff(np.log(esr_history)) / np.diff(history)
            rows.append({
                "device": device, "origin_h": float(ages[i]), "loss_pct": cap_history[-1],
                "older_rate_pct_per_h": cap_rates[0], "recent_rate_pct_per_h": cap_rates[1],
                "esr_older_log_rate": esr_rates[0], "esr_recent_log_rate": esr_rates[1],
                "outcome": outcome,
                "trend": float(cap_history[-1] + HORIZON_H * slope >= threshold),
            })
    return pd.DataFrame(rows)


def compare(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    detail, _ = evaluate_transfer(train, test)
    augmented, _ = evaluate_transfer(train, test, (*FEATURES, *ESR_FEATURES))
    return detail.assign(neural_esr=augmented["neural"], esr_in_range=augmented["within_training_feature_ranges"])


def leave_device_out(examples: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([compare(examples.loc[examples["device"] != device], test)
                      for device, test in examples.groupby("device")], ignore_index=True)


def method_scores(resolved: pd.DataFrame) -> dict:
    cross = resolved["outcome"].eq(1)
    metrics = {}
    for field in ("trend", "linear", "neural", "neural_esr"):
        error = (resolved[field] - resolved["outcome"]) ** 2
        alert = resolved[field].ge(0.5)
        metrics[field] = {
            "brier": round(float(error.groupby(resolved["device"]).mean().mean()), 6),
            "true_alerts": int((alert & cross).sum()),
            "false_alerts": int((alert & ~cross).sum()),
            "missed_crossings": int((~alert & cross).sum()),
        }
    return metrics


def score(detail: pd.DataFrame) -> dict:
    resolved = detail.loc[detail["outcome"].notna()]
    supported = resolved.loc[resolved["esr_in_range"]]
    # Compare feature sets on each held-out device before averaging differences.
    paired = ((resolved["neural_esr"] - resolved["outcome"]) ** 2
              - (resolved["neural"] - resolved["outcome"]) ** 2).groupby(resolved["device"]).mean()
    weights = 1 / resolved.groupby("device")["device"].transform("size")
    calibration = {}
    for field in ("neural", "neural_esr"):
        # Fixed bins are descriptive. Repeated windows from one device are correlated.
        bins = pd.cut(resolved[field], [0., 0.25, 0.50, 0.75, 1.0], include_lowest=True)
        calibration[field] = []
        for label, part in resolved.groupby(bins, observed=True):
            w = weights.loc[part.index]
            calibration[field].append({
                "score_range": str(label), "origins": len(part), "devices": int(part["device"].nunique()),
                "mean_score": round(float(np.average(part[field], weights=w)), 4),
                "observed_frequency": round(float(np.average(part["outcome"], weights=w)), 4),
            })
    return {
        "devices": int(resolved["device"].nunique()), "resolved": len(resolved),
        "unresolved": len(detail) - len(resolved), "crossings": int(resolved["outcome"].eq(1).sum()),
        "outside_augmented_training_ranges": len(resolved) - len(supported),
        "by_method": method_scores(resolved),
        "within_augmented_training_ranges": {
            "devices": int(supported["device"].nunique()), "resolved": len(supported),
            "crossings": int(supported["outcome"].eq(1).sum()),
            "by_method": method_scores(supported) if len(supported) else None,
        },
        "per_device_brier_difference": paired.round(6).to_dict(),
        "calibration": calibration,
    }


def build_validation(eos_zip: bytes, figure_a: bytes, figure_b: bytes) -> dict:
    eos_c, eos_mat = load_eos2(eos_zip)
    eos_esr = pd.DataFrame(1 + loadmat(io.BytesIO(eos_mat), simplify_cells=True)["ESR"] / 100,
                           index=eos_c.index, columns=eos_c.columns)
    jian_c, jian_esr = digitize_figure(figure_a, True), digitize_figure(figure_b, False)
    eos12 = paired_examples(eos_c, eos_esr, 12)
    jian12 = paired_examples(jian_c, jian_esr, 12, 0.10)
    return {
        "horizon_h": HORIZON_H,
        "sources": {"Jian et al.": ARTICLE,
                    "NASA EOS-2": "https://data.nasa.gov/dataset/capacitor-electrical-stress-2"},
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in
                          (("EOS_DataSet.mat", eos_mat), ("Figure 4(a)", figure_a), ("Figure 4(b)", figure_b))},
        "eos2_20pct_device_holdout": score(leave_device_out(paired_examples(eos_c, eos_esr, 20))),
        "eos2_12pct_device_holdout": score(leave_device_out(eos12)),
        "jian_12pct_device_holdout": score(leave_device_out(jian12)),
        "eos2_to_jian_12pct": score(compare(eos12, jian12)),
        "jian_to_eos2_12pct": score(compare(jian12, eos12)),
        "figure_margin_sensitivity": {
            str(margin): {
                "eos2_to_jian": score(compare(eos12, paired_examples(jian_c, jian_esr, 12, margin))),
                "jian_to_eos2": score(compare(paired_examples(jian_c, jian_esr, 12, margin), eos12)),
            }
            for margin in (0.05, 0.15)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos2", type=Path)
    parser.add_argument("--figure-a", type=Path)
    parser.add_argument("--figure-b", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/esr_validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = build_validation(
        read_source(args.eos2, EOS2_URLS),
        read_source(args.figure_a, (f"{FIGURE_ROOT}/electronics-14-04452-g004a.png",)),
        read_source(args.figure_b, (f"{FIGURE_ROOT}/electronics-14-04452-g004b.png",)),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
