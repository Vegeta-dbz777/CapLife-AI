"""Six-indicator neural RUL model trained on simulated capacitors, tolerant of missing indicators.

A GRU reads up to the last ``WINDOW`` readings of one capacitor. Each indicator
is expressed as the fraction of its end-of-life limit already used, so the
network sees how quickly health is being consumed, and carries a flag saying
whether it was measured at all. During training, indicators are hidden at
random, so the network learns to predict from any combination of them. It
outputs the 10th, 50th and 90th percentiles of log remaining life.

A separate calibration fleet then widens or narrows the 10–90 range
(conformalized quantile regression) for every combination of measured
indicators, so that about 80% of true remaining lives fall inside it whichever
indicators a user has. An independent test fleet measures accuracy and coverage
for every combination. All fleets come from ``caplife.six_sim``: the model
demonstrates the method and has not been validated on real devices.

Train with ``python -m caplife.six_rul --output artifacts/six_rul``.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
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
STEP_FEATURES = ("log_age", "log_gap", *(f"{signal}_used" for signal in SIGNALS),
                 *(f"{signal}_measured" for signal in SIGNALS), "padding")
USED, MEASURED = slice(2, 2 + len(SIGNALS)), slice(2 + len(SIGNALS), 2 + 2 * len(SIGNALS))
SUBSETS = tuple(subset for size in range(len(SIGNALS), 0, -1) for subset in combinations(SIGNALS, size))
COMMON_SUBSET = ("Capacitance_uF", "ESR_ohm", "Leakage_uA")
# Subsets for which the slower trend-line baseline is also scored.
BASELINE_SUBSETS = (SIGNALS, COMMON_SUBSET, ("Capacitance_uF", "ESR_ohm"), ("Capacitance_uF",))
INDICATOR_LABELS = {
    "Capacitance_uF": "Capacitance", "ESR_ohm": "ESR", "Leakage_uA": "Leakage current",
    "Case_Expansion_um": "Case expansion", "H2_Generated_mL_STP": "H₂ generated",
    "Pressure_Abs_kPa": "Internal pressure",
}
HIDDEN = 64
SPLITS = {"train": (1600, 101), "validation": (300, 202), "calibration": (400, 303), "test": (400, 404)}


def subset_key(signals) -> str:
    """Stable name for a combination of measured indicators, in SIGNALS order."""
    chosen = set(signals)
    return "+".join(signal for signal in SIGNALS if signal in chosen)


def measured_signals(readings: pd.DataFrame) -> tuple[str, ...]:
    """Indicators present in every reading of this history."""
    return tuple(signal for signal in SIGNALS if signal in readings and readings[signal].notna().all())


def limit_used(readings: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each indicator's end-of-life limit used so far (NaN if not measured).

    Aging damage does not heal: a capacitor whose ESR doubled and then partly fell back has still
    used up that part of its life. Each value is therefore the worse of the current reading and the
    worst level held so far, where "held" means the median of three consecutive readings, so a
    single outlying reading cannot mark lasting damage. Readings must be in time order.
    """
    current = current_limit_used(readings)
    held = current.rolling(3, min_periods=1).median().cummax()
    return np.maximum(current, held)


def current_limit_used(readings: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each indicator's end-of-life limit used at each reading alone, relative to the first."""
    first = readings.iloc[0]
    nan = pd.Series(np.nan, index=readings.index)
    column = lambda name: readings[name] if name in readings else nan
    base = lambda name: first[name] if name in readings else np.nan
    return pd.DataFrame({
        "Capacitance_uF": (1 - column("Capacitance_uF") / base("Capacitance_uF")) * 100
        / EOL_LIMITS["capacitance_loss_pct"],
        "ESR_ohm": (column("ESR_ohm") / base("ESR_ohm") - 1) / (EOL_LIMITS["esr_ratio"] - 1),
        "Leakage_uA": (column("Leakage_uA") / base("Leakage_uA") - 1) / (EOL_LIMITS["leakage_ratio"] - 1),
        "Case_Expansion_um": (column("Case_Expansion_um") - base("Case_Expansion_um"))
        / NOMINAL_LIMITS["Case_Expansion_um"],
        "H2_Generated_mL_STP": (column("H2_Generated_mL_STP") - base("H2_Generated_mL_STP"))
        / NOMINAL_LIMITS["H2_Generated_mL_STP"],
        "Pressure_Abs_kPa": (column("Pressure_Abs_kPa") - ATMOSPHERE_KPA)
        / (EOL_LIMITS["pressure_abs_kpa"] - ATMOSPHERE_KPA),
    }, index=readings.index)


def sequences(readings: pd.DataFrame, measured: tuple[str, ...] | None = None) -> np.ndarray:
    """One ``WINDOW`` × features array per reading from the third onward, using only past readings.

    Indicators outside ``measured`` (default: those present in every reading) enter as zero
    with their measured flag off.
    """
    measured = measured_signals(readings) if measured is None else tuple(measured)
    mask = np.array([signal in measured for signal in SIGNALS], dtype=float)
    age = (readings["Hours"] - readings["Hours"].iloc[0]).to_numpy(dtype=float)
    gap = np.concatenate([[0.0], np.diff(age)])
    used = np.nan_to_num(limit_used(readings).to_numpy(dtype=float)) * mask
    steps = np.column_stack([np.log1p(age / 1000), np.log1p(gap / 1000), used,
                             np.tile(mask, (len(age), 1)), np.zeros(len(age))])
    windows = []
    for end in range(MIN_READINGS - 1, len(steps)):
        window = steps[max(0, end - WINDOW + 1):end + 1]
        if len(window) < WINDOW:
            padding = np.zeros((WINDOW - len(window), steps.shape[1]))
            padding[:, -1] = 1.0
            window = np.vstack([padding, window])
        windows.append(window)
    return np.asarray(windows, dtype=np.float32).reshape(-1, WINDOW, len(STEP_FEATURES))


def hide_indicators(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Copy of full-indicator sequences with unkept indicators zeroed and flagged as not measured.

    ``keep`` is (n, len(SIGNALS)) booleans, one row per sequence.
    """
    hidden = x.copy()
    real = hidden[:, :, -1:] == 0  # padding rows stay all zero
    keep = keep[:, None, :].astype(hidden.dtype)
    hidden[:, :, USED] *= keep
    hidden[:, :, MEASURED] = keep * real
    return hidden


def _subset_mask(subset) -> np.ndarray:
    return np.array([signal in subset for signal in SIGNALS])


def _random_masks(rng: np.random.Generator, n: int) -> np.ndarray:
    """Training mix: 25% all six, 15% capacitance + ESR + leakage, 60% random non-empty combinations."""
    masks = rng.random((n, len(SIGNALS))) < 0.5
    empty = ~masks.any(axis=1)
    masks[empty, rng.integers(0, len(SIGNALS), empty.sum())] = True
    choice = rng.random(n)
    masks[choice < 0.25] = True
    masks[(choice >= 0.25) & (choice < 0.40)] = _subset_mask(COMMON_SUBSET)
    return masks


class SixRULNet(nn.Module):
    """GRU encoder with a head that returns ordered 10th, 50th and 90th percentiles of log RUL."""

    def __init__(self, hidden: int = HIDDEN):
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
        x = sequences(readings, SIGNALS)
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


def _trend_baseline(measurements: pd.DataFrame, subset=SIGNALS) -> np.ndarray:
    """Hours until the first measured indicator reaches its limit, extrapolating the last three readings."""
    columns = [SIGNALS.index(signal) for signal in subset]
    predictions = []
    for _, readings in measurements.groupby("Cap_ID", sort=False):
        used = limit_used(readings).to_numpy(dtype=float)[:, columns]
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


def _to_hours(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(values) - RUL_OFFSET_H, 0.0)


def train(output: Path, seed: int = 7, epochs: int = 80, hidden: int = HIDDEN) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    fleets = {name: simulate_fleet(n, fleet_seed, prefix=name[:3].upper()) for name, (n, fleet_seed) in SPLITS.items()}
    data = {name: _dataset(*fleet) for name, fleet in fleets.items()}
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["validation"]
    rng = np.random.default_rng(seed)
    # A fixed mix of hidden indicators for validation, so early stopping compares like with like.
    x_v = torch.from_numpy(hide_indicators(x_val, _random_masks(rng, len(x_val))))
    y_v = torch.from_numpy(y_val)
    y_t = torch.from_numpy(y_train)
    net = SixRULNet(hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    best, best_state, stale = math.inf, None, 0
    for epoch in range(epochs):
        net.train()
        # New random hidden indicators for every sequence in every epoch.
        x_t = torch.from_numpy(hide_indicators(x_train, _random_masks(rng, len(x_train))))
        for batch in torch.from_numpy(rng.permutation(len(x_t))).split(512):
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
            if stale >= 12:
                break
    net.load_state_dict(best_state)
    net.eval()

    # Conformalized quantile regression for every combination of measured indicators, with one
    # randomly chosen reading per calibration capacitor so the scores are independent.
    x_cal, y_cal, cal_devices = data["calibration"]
    pick = np.array([rng.choice(np.flatnonzero(cal_devices == device)) for device in np.unique(cal_devices)])
    rank = math.ceil((len(pick) + 1) * INTERVAL_MASS)
    x_test, y_test, _ = data["test"]
    rul = np.exp(y_test) - RUL_OFFSET_H
    adjustments, by_subset = {}, {}
    for subset in SUBSETS:
        keep = _subset_mask(subset)
        q_cal = _predict(net, hide_indicators(x_cal[pick], np.tile(keep, (len(pick), 1))))
        scores = np.maximum(q_cal[:, 0] - y_cal[pick], y_cal[pick] - q_cal[:, 2])
        adjustment = float(np.sort(scores)[rank - 1])
        q_test = _predict(net, hide_indicators(x_test, np.tile(keep, (len(x_test), 1))))
        key = subset_key(subset)
        adjustments[key] = round(adjustment, 6)
        by_subset[key] = _metrics(rul, _to_hours(q_test[:, 0] - adjustment), _to_hours(q_test[:, 1]),
                                  _to_hours(q_test[:, 2] + adjustment))
    baselines = {}
    for subset in BASELINE_SUBSETS:
        trend = _trend_baseline(fleets["test"][0], subset)
        available = np.isfinite(trend)
        baselines[subset_key(subset)] = {
            "predictions_with_estimate": int(available.sum()),
            "median_absolute_error_h": round(float(np.median(np.abs(trend[available] - rul[available]))), 1),
        }
    full = subset_key(SIGNALS)
    train_used = x_train[:, :, USED][x_train[:, :, -1] == 0]
    metadata = {
        "hidden": hidden,
        "model": f"SixRULNet GRU, hidden {hidden}, quantile outputs (10th, 50th, 90th percentile of log RUL), "
                 "trained with randomly hidden indicators",
        "training_data": "Simulated capacitors from caplife.six_sim; not real measurements",
        "signals": list(SIGNALS), "step_features": list(STEP_FEATURES), "window": WINDOW,
        "min_readings": MIN_READINGS, "rul_offset_h": RUL_OFFSET_H,
        "eol_limits": EOL_LIMITS, "nominal_limits": NOMINAL_LIMITS, "interval_mass": INTERVAL_MASS,
        "conformal_adjustment_log": adjustments[full],
        "conformal_adjustment_log_by_subset": adjustments,
        "fleets": {name: {"capacitors": n, "seed": fleet_seed, "predictions": int(len(data[name][1]))}
                   for name, (n, fleet_seed) in SPLITS.items()},
        "training_seed": seed, "best_epoch": best_epoch, "validation_pinball": round(best, 5),
        "limit_used_training_range": {
            "min": [round(float(v), 4) for v in train_used.min(axis=0)],
            "max": [round(float(v), 4) for v in train_used.max(axis=0)],
        },
        "test_metrics": {"neural_calibrated": by_subset[full], "trend_baseline": baselines[full]},
        "test_metrics_by_subset": by_subset,
        "trend_baseline_by_subset": baselines,
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
        self.net = SixRULNet(self.metadata.get("hidden", HIDDEN))
        self.net.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.adjustments = self.metadata["conformal_adjustment_log_by_subset"]
        self.train_max = np.asarray(self.metadata["limit_used_training_range"]["max"])

    def metrics_for(self, measured) -> dict:
        """Test-fleet accuracy for this combination of measured indicators."""
        return self.metadata["test_metrics_by_subset"][subset_key(measured)]

    def assess(self, readings: pd.DataFrame, measured: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Health and remaining life at every reading from the third onward, each using only past readings.

        Only the ``measured`` indicators (default: those present in every reading) are used; the
        health columns of the others are NaN.
        """
        readings = readings.sort_values("Hours").reset_index(drop=True)
        measured = measured_signals(readings) if measured is None else tuple(measured)
        if not measured:
            raise ValueError("Supply at least one of the six indicators in every reading.")
        ratio_based = [signal for signal in ("Capacitance_uF", "ESR_ohm", "Leakage_uA") if signal in measured]
        if (readings.loc[0, ratio_based] <= 0).any():
            raise ValueError("The first reading needs positive capacitance, ESR and leakage current "
                             "(where measured) to serve as a baseline.")
        used = limit_used(readings)[list(measured)]
        adjustment = self.adjustments[subset_key(measured)]
        columns = [SIGNALS.index(signal) for signal in measured]
        rows = []
        if len(readings) >= MIN_READINGS:
            q = _predict(self.net, sequences(readings, measured))
            low, mid, high = _to_hours(q[:, 0] - adjustment), _to_hours(q[:, 1]), _to_hours(q[:, 2] + adjustment)
        criteria = [signal for signal in measured if signal not in NOMINAL_LIMITS]
        for index in range(len(readings)):
            fractions = used.iloc[index]
            row = {"Hours": float(readings.loc[index, "Hours"])}
            for signal in SIGNALS:
                row[f"{signal}_health_pct"] = (float(100 * np.clip(1 - fractions[signal], 0, 1))
                                               if signal in measured else np.nan)
            row["Overall_health_pct"] = min(row[f"{signal}_health_pct"] for signal in measured)
            row["Limiting_indicator"] = INDICATOR_LABELS[fractions.idxmax()]
            reached = bool(criteria) and bool((fractions[criteria] >= 1 - 1e-9).any())
            row["End_of_life_reached"] = reached
            k = index - (MIN_READINGS - 1)
            if k >= 0 and not reached:
                row.update({"RUL_low_h": low[k], "RUL_median_h": mid[k], "RUL_high_h": high[k]})
                row["Outside_training_range"] = bool(
                    (fractions.to_numpy() > self.train_max[columns] * 1.1).any())
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
