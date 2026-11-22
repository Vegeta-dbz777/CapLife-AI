"""Six-indicator neural RUL model trained on simulated capacitors.

A GRU reads up to the last ``WINDOW`` readings of one capacitor. Each indicator
is expressed as the fraction of its end-of-life limit already used, so the
network sees how quickly health is being consumed. It outputs the 10th, 50th
and 90th percentiles of log remaining life. A separate calibration fleet then
widens or narrows the 10–90 range (conformalized quantile regression) so that
about 80% of true remaining lives fall inside it, and an independent test
fleet measures accuracy and coverage. All fleets come from ``caplife.six_sim``:
the model demonstrates the method and has not been validated on real devices.

Train with ``python -m caplife.six_rul --output artifacts/six_rul``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from caplife.six_sim import ATMOSPHERE_KPA, EOL_LIMITS, SIGNALS, simulate_fleet

WINDOW = 8
MIN_READINGS = 3
QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_MASS = 0.8
RUL_OFFSET_H = 24.0
# Nominal limits for the two indicators that are not end-of-life criteria
# themselves: the expansion and generated H2 at which a typical simulated
# capacitor reaches the vent pressure.
NOMINAL_LIMITS = {"Case_Expansion_um": 52.0, "H2_Generated_mL_STP": 6.0}
STEP_FEATURES = ("log_age", "log_gap", *(f"{signal}_used" for signal in SIGNALS), "padding")
INDICATOR_LABELS = {
    "Capacitance_uF": "Capacitance", "ESR_ohm": "ESR", "Leakage_uA": "Leakage current",
    "Case_Expansion_um": "Case expansion", "H2_Generated_mL_STP": "H₂ generated",
    "Pressure_Abs_kPa": "Internal pressure",
}
SPLITS = {"train": (1600, 101), "validation": (300, 202), "calibration": (400, 303), "test": (400, 404)}


def limit_used(readings: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each indicator's end-of-life limit used, relative to the first reading."""
    first = readings.iloc[0]
    return pd.DataFrame({
        "Capacitance_uF": (1 - readings["Capacitance_uF"] / first["Capacitance_uF"]) * 100
        / EOL_LIMITS["capacitance_loss_pct"],
        "ESR_ohm": (readings["ESR_ohm"] / first["ESR_ohm"] - 1) / (EOL_LIMITS["esr_ratio"] - 1),
        "Leakage_uA": (readings["Leakage_uA"] / first["Leakage_uA"] - 1) / (EOL_LIMITS["leakage_ratio"] - 1),
        "Case_Expansion_um": (readings["Case_Expansion_um"] - first["Case_Expansion_um"])
        / NOMINAL_LIMITS["Case_Expansion_um"],
        "H2_Generated_mL_STP": (readings["H2_Generated_mL_STP"] - first["H2_Generated_mL_STP"])
        / NOMINAL_LIMITS["H2_Generated_mL_STP"],
        "Pressure_Abs_kPa": (readings["Pressure_Abs_kPa"] - ATMOSPHERE_KPA)
        / (EOL_LIMITS["pressure_abs_kpa"] - ATMOSPHERE_KPA),
    }, index=readings.index)


def sequences(readings: pd.DataFrame) -> np.ndarray:
    """One ``WINDOW`` × features array per reading from the third onward, using only past readings."""
    age = (readings["Hours"] - readings["Hours"].iloc[0]).to_numpy(dtype=float)
    gap = np.concatenate([[0.0], np.diff(age)])
    steps = np.column_stack([np.log1p(age / 1000), np.log1p(gap / 1000),
                             limit_used(readings).to_numpy(dtype=float), np.zeros(len(age))])
    windows = []
    for end in range(MIN_READINGS - 1, len(steps)):
        window = steps[max(0, end - WINDOW + 1):end + 1]
        if len(window) < WINDOW:
            padding = np.zeros((WINDOW - len(window), steps.shape[1]))
            padding[:, -1] = 1.0
            window = np.vstack([padding, window])
        windows.append(window)
    return np.asarray(windows, dtype=np.float32).reshape(-1, WINDOW, len(STEP_FEATURES))


class SixRULNet(nn.Module):
    """GRU encoder with a head that returns ordered 10th, 50th and 90th percentiles of log RUL."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(len(STEP_FEATURES), hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(x)
        median, below, above = self.head(state[-1]).unbind(-1)
        softplus = nn.functional.softplus
        return torch.stack([median - softplus(below), median, median + softplus(above)], dim=-1)


def _dataset(measurements: pd.DataFrame, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eol = meta.set_index("Cap_ID")["eol_h"]
    xs, ys, devices = [], [], []
    for device, readings in measurements.groupby("Cap_ID", sort=False):
        x = sequences(readings)
        hours = readings["Hours"].to_numpy(dtype=float)[MIN_READINGS - 1:]
        xs.append(x)
        ys.append(np.log(eol[device] - hours + RUL_OFFSET_H))
        devices += [device] * len(x)
    return np.concatenate(xs), np.concatenate(ys).astype(np.float32), np.asarray(devices)


def _pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES, dtype=prediction.dtype)
    error = target[:, None] - prediction
    return torch.maximum(levels * error, (levels - 1) * error).mean()


def _predict(net: SixRULNet, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.from_numpy(x)).numpy()


def _trend_baseline(measurements: pd.DataFrame) -> np.ndarray:
    """Hours until the first indicator reaches its limit, extrapolating the last three readings linearly."""
    predictions = []
    for _, readings in measurements.groupby("Cap_ID", sort=False):
        used = limit_used(readings).to_numpy(dtype=float)
        hours = readings["Hours"].to_numpy(dtype=float)
        for end in range(MIN_READINGS - 1, len(readings)):
            recent = slice(end - 2, end + 1)
            slopes = np.polyfit(hours[recent], used[recent], 1)[0]
            current = used[end]
            with np.errstate(divide="ignore", invalid="ignore"):
                remaining = np.where(slopes > 0, (1 - current) / slopes, np.inf)
            remaining = remaining[np.isfinite(remaining)]
            predictions.append(max(float(remaining.min()), 0.0) if remaining.size else np.nan)
    return np.asarray(predictions)


def _metrics(rul: np.ndarray, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> dict:
    inside = (rul >= low) & (rul <= high)
    return {
        "predictions": int(len(rul)),
        "median_absolute_error_h": round(float(np.median(np.abs(mid - rul))), 1),
        "mean_absolute_error_h": round(float(np.mean(np.abs(mid - rul))), 1),
        "median_relative_error": round(float(np.median(np.abs(mid - rul) / np.maximum(rul, RUL_OFFSET_H))), 4),
        "interval_coverage": round(float(inside.mean()), 4),
        "median_interval_width_h": round(float(np.median(high - low)), 1),
    }


def train(output: Path, seed: int = 7, epochs: int = 60) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    fleets = {name: simulate_fleet(n, fleet_seed, prefix=name[:3].upper()) for name, (n, fleet_seed) in SPLITS.items()}
    data = {name: _dataset(*fleet) for name, fleet in fleets.items()}
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    net = SixRULNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    x_t, y_t = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_v, y_v = torch.from_numpy(x_val), torch.from_numpy(y_val)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale = math.inf, None, 0
    for epoch in range(epochs):
        net.train()
        for batch in torch.randperm(len(x_t), generator=generator).split(512):
            optimizer.zero_grad()
            loss = _pinball(net(x_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val = float(_pinball(net(x_v), y_v))
        scheduler.step(val)
        if val < best - 1e-4:
            best, best_state, stale, best_epoch = val, {k: v.clone() for k, v in net.state_dict().items()}, 0, epoch + 1
        else:
            stale += 1
            if stale >= 10:
                break
    net.load_state_dict(best_state)
    net.eval()

    # Conformalized quantile regression with one randomly chosen reading per calibration
    # capacitor, so the calibration scores are independent across capacitors.
    x_cal, y_cal, cal_devices = data["calibration"]
    rng = np.random.default_rng(seed)
    pick = np.array([rng.choice(np.flatnonzero(cal_devices == device)) for device in np.unique(cal_devices)])
    q_cal = _predict(net, x_cal[pick])
    scores = np.maximum(q_cal[:, 0] - y_cal[pick], y_cal[pick] - q_cal[:, 2])
    rank = math.ceil((len(scores) + 1) * INTERVAL_MASS)
    adjustment = float(np.sort(scores)[rank - 1])

    x_test, y_test, _ = data["test"]
    q_test = _predict(net, x_test)
    rul = np.exp(y_test) - RUL_OFFSET_H
    to_hours = lambda values: np.maximum(np.exp(values) - RUL_OFFSET_H, 0.0)
    neural = _metrics(rul, to_hours(q_test[:, 0] - adjustment), to_hours(q_test[:, 1]),
                      to_hours(q_test[:, 2] + adjustment))
    uncalibrated = _metrics(rul, to_hours(q_test[:, 0]), to_hours(q_test[:, 1]), to_hours(q_test[:, 2]))
    trend = _trend_baseline(fleets["test"][0])
    available = np.isfinite(trend)
    trend_metrics = {
        "predictions_with_estimate": int(available.sum()),
        "median_absolute_error_h": round(float(np.median(np.abs(trend[available] - rul[available]))), 1),
        "neural_median_absolute_error_h_same_points": round(
            float(np.median(np.abs(to_hours(q_test[available, 1]) - rul[available]))), 1),
    }
    train_used = x_train[:, :, 2:8][x_train[:, :, -1] == 0]
    metadata = {
        "model": "SixRULNet GRU, hidden 64, quantile outputs (10th, 50th, 90th percentile of log RUL)",
        "training_data": "Simulated capacitors from caplife.six_sim; not real measurements",
        "signals": list(SIGNALS), "step_features": list(STEP_FEATURES), "window": WINDOW,
        "min_readings": MIN_READINGS, "rul_offset_h": RUL_OFFSET_H,
        "eol_limits": EOL_LIMITS, "nominal_limits": NOMINAL_LIMITS,
        "interval_mass": INTERVAL_MASS, "conformal_adjustment_log": round(adjustment, 6),
        "fleets": {name: {"capacitors": n, "seed": fleet_seed, "predictions": int(len(data[name][1]))}
                   for name, (n, fleet_seed) in SPLITS.items()},
        "training_seed": seed, "best_epoch": best_epoch, "validation_pinball": round(best, 5),
        "limit_used_training_range": {
            "min": [round(float(v), 4) for v in train_used.min(axis=0)],
            "max": [round(float(v), 4) for v in train_used.max(axis=0)],
        },
        "test_metrics": {"neural_calibrated": neural, "neural_uncalibrated": uncalibrated, "trend_baseline": trend_metrics},
        "end_of_life_causes_in_test_fleet": fleets["test"][1]["eol_cause"].str.split(", ").str[0]
        .value_counts().to_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output / "model.pt")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class SixRULModel:
    """Loaded model plus the health and remaining-life calculations the app shows."""

    def __init__(self, artifact_dir: Path):
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.net = SixRULNet()
        self.net.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.adjustment = self.metadata["conformal_adjustment_log"]
        self.train_max = np.asarray(self.metadata["limit_used_training_range"]["max"])

    def assess(self, readings: pd.DataFrame) -> pd.DataFrame:
        """Health and remaining life at every reading from the third onward, each using only past readings."""
        readings = readings.sort_values("Hours").reset_index(drop=True)
        if (readings.loc[0, ["Capacitance_uF", "ESR_ohm", "Leakage_uA"]] <= 0).any():
            raise ValueError("The first reading needs positive capacitance, ESR and leakage current to serve as a baseline.")
        used = limit_used(readings)
        rows = []
        if len(readings) >= MIN_READINGS:
            q = _predict(self.net, sequences(readings))
            low = np.maximum(np.exp(q[:, 0] - self.adjustment) - RUL_OFFSET_H, 0.0)
            mid = np.maximum(np.exp(q[:, 1]) - RUL_OFFSET_H, 0.0)
            high = np.maximum(np.exp(q[:, 2] + self.adjustment) - RUL_OFFSET_H, 0.0)
        for index in range(len(readings)):
            fractions = used.iloc[index]
            row = {"Hours": float(readings.loc[index, "Hours"])}
            for signal in SIGNALS:
                row[f"{signal}_health_pct"] = float(100 * np.clip(1 - fractions[signal], 0, 1))
            row["Overall_health_pct"] = min(row[f"{signal}_health_pct"] for signal in SIGNALS)
            row["Limiting_indicator"] = INDICATOR_LABELS[fractions.idxmax()]
            criteria = [s for s in SIGNALS if s not in NOMINAL_LIMITS]
            reached = bool((fractions[criteria] >= 1 - 1e-9).any())
            row["End_of_life_reached"] = reached
            k = index - (MIN_READINGS - 1)
            if k >= 0 and not reached:
                row.update({"RUL_low_h": low[k], "RUL_median_h": mid[k], "RUL_high_h": high[k]})
                row["Outside_training_range"] = bool((fractions.to_numpy() > self.train_max * 1.1).any())
            elif reached:
                row.update({"RUL_low_h": 0.0, "RUL_median_h": 0.0, "RUL_high_h": 0.0, "Outside_training_range": False})
            rows.append(row)
        return pd.DataFrame(rows)


def example_fleet(seed: int = 2027, devices_per_cause: int = 2) -> tuple[pd.DataFrame, dict]:
    """Simulated capacitors observed part of the way to end of life, and their true end-of-life ages.

    Two capacitors per end-of-life cause are taken from a larger simulated fleet, each cut off at a
    different stage of its life, so the example shows every failure mode and a range of health.
    """
    measurements, meta = simulate_fleet(120, seed, prefix="POOL")
    meta = meta.assign(cause=meta["eol_cause"].str.split(", ").str[0])
    chosen = pd.concat([meta.loc[meta["cause"].eq(cause)].head(devices_per_cause)
                        for cause in ("capacitance", "ESR", "leakage", "pressure")])
    stages = np.random.default_rng(seed).permutation(np.linspace(0.35, 0.9, len(chosen)))
    kept, truth = [], {}
    for number, (row, stage) in enumerate(zip(chosen.itertuples(index=False), stages), start=1):
        name = f"DEMO_{number:02d}"
        readings = measurements.loc[measurements["Cap_ID"].eq(row.Cap_ID)]
        part = readings.loc[readings["Hours"] <= row.eol_h * stage]
        part = part if len(part) >= MIN_READINGS + 2 else readings.iloc[:MIN_READINGS + 2]
        kept.append(part.assign(Cap_ID=name))
        truth[name] = {"eol_h": row.eol_h, "eol_cause": row.eol_cause}
    return pd.concat(kept, ignore_index=True), truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("artifacts/six_rul"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    metadata = train(args.output, args.seed)
    print(json.dumps(metadata["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
