"""Six-indicator neural RUL and parameter-trend models trained on simulated capacitors.

A GRU reads up to the last ``WINDOW`` readings of one capacitor. Each indicator
is expressed as the fraction of its end-of-life limit already used, so the
network sees how quickly health is being consumed, and carries a flag saying
whether it was measured at all and the limit it is measured against. Each
reading also carries the operating conditions recorded with it (core and
ambient temperature, applied over rated voltage, ripple over rated ripple
current, operating frequency), each with a flag saying whether it was recorded.
The conditions at the newest reading are taken to hold from then on, so a user
can ask what happens under different planned conditions. Limits
are the user's own (a capacitor series' tolerance): training capacitors get
limits drawn at random from ``LIMIT_RANGES``, so the network predicts the time
to whatever limits a user sets inside those ranges. During training, indicators
are also hidden at random, and so are the operating conditions, so the network
predicts from any combination of them. It outputs the 10th, 50th and 90th percentiles of log remaining life.

A second GRU forecasts each indicator's path from the latest reading to end of
life: the fraction of its limit used at 10%, 20%, ..., 100% of the remaining
life, as the 10th, 50th and 90th percentiles. Its median can only rise, because
aging damage does not heal.

Separate calibration capacitors widen or narrow the 10-90 ranges of both models
(conformalized quantile regression) so that about 80% of true values fall inside
them, and independent test capacitors measure accuracy and coverage. All
capacitors come from ``caplife.six_sim``: the models demonstrate the method and
have not been validated on real devices.

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

from caplife.six_sim import ATMOSPHERE_KPA, EOL_LIMITS, LIMIT_RANGES, SIGNALS, simulate_fleet

WINDOW = 8
MIN_READINGS = 3
QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_MASS = 0.8
RUL_OFFSET_H = 24.0
# Expansion and generated H2 are measured from the first reading.
EOL_CAUSES = ("capacitance", "ESR", "leakage", "expansion", "H2", "pressure")
LIMIT_KEYS = {"Capacitance_uF": "capacitance_loss_pct", "ESR_ohm": "esr_ratio", "Leakage_uA": "leakage_ratio",
              "Case_Expansion_um": "expansion_um", "H2_Generated_mL_STP": "h2_ml",
              "Pressure_Abs_kPa": "pressure_abs_kpa"}
# Operating conditions the networks read with every reading, and the columns each needs.
CONDITIONS = ("core_temperature", "ambient_temperature", "voltage_ratio", "ripple_ratio", "frequency")
CONDITION_SOURCES = {"core_temperature": ("Core_C",), "ambient_temperature": ("Ambient_C",),
                     "voltage_ratio": ("Voltage_V", "Rated_V"), "ripple_ratio": ("Ripple_A", "Rated_Ripple_A"),
                     "frequency": ("Ripple_Hz",)}
CONDITION_LABELS = {"core_temperature": "Core temperature", "ambient_temperature": "Ambient temperature",
                    "voltage_ratio": "Applied / rated voltage", "ripple_ratio": "Ripple / rated ripple current",
                    "frequency": "Operating frequency"}
# The ranges the simulated training capacitors cover, in the units shown to users.
CONDITION_TRAINING_RANGES = {"core_temperature": (60.0, 125.0), "ambient_temperature": (60.0, 105.0),
                             "voltage_ratio": (0.5, 1.0), "ripple_ratio": (0.2, 1.3), "frequency": (100.0, 100_000.0)}
STEP_FEATURES = ("log_age", "log_gap", *(f"{signal}_used" for signal in SIGNALS),
                 *(f"{signal}_measured" for signal in SIGNALS), *(f"{signal}_limit" for signal in SIGNALS),
                 *CONDITIONS, *(f"{condition}_recorded" for condition in CONDITIONS), "padding")
N = len(SIGNALS)
K = len(CONDITIONS)
USED, MEASURED, LIMIT = slice(2, 2 + N), slice(2 + N, 2 + 2 * N), slice(2 + 2 * N, 2 + 3 * N)
COND, COND_KNOWN = slice(2 + 3 * N, 2 + 3 * N + K), slice(2 + 3 * N + K, 2 + 3 * N + 2 * K)
# The trend forecast gives each indicator at 10%, 20%, ..., 100% of the remaining life.
HORIZON_POINTS = 10
HORIZON_FRACTIONS = np.arange(1, HORIZON_POINTS + 1) / HORIZON_POINTS
SUBSETS = tuple(subset for size in range(len(SIGNALS), 0, -1) for subset in combinations(SIGNALS, size))
COMMON_SUBSET = ("Capacitance_uF", "ESR_ohm", "Leakage_uA")
# Subsets for which the slower trend-line baseline is also scored.
BASELINE_SUBSETS = (SIGNALS, COMMON_SUBSET, ("Capacitance_uF", "ESR_ohm"), ("Capacitance_uF",))
INDICATOR_LABELS = {
    "Capacitance_uF": "Capacitance", "ESR_ohm": "ESR", "Leakage_uA": "Leakage current",
    "Case_Expansion_um": "Case expansion", "H2_Generated_mL_STP": "H₂ generated",
    "Pressure_Abs_kPa": "Internal pressure",
}
HIDDEN = 128  # 64 before: the larger network, on twice the capacitors, cut the typical error by about a fifth
# Capacitors, seed, whether their limits vary, and the share with extra inspections right before end
# of life. User limits often put a capacitor close to one, so the calibration and random-limit test
# capacitors include such readings. "test" keeps the default limits and inspection plan so its
# accuracy is comparable with earlier models.
SPLITS = {"train": (4800, 101, True, 0.0), "validation": (400, 202, True, 0.0),
          "calibration": (500, 303, True, 0.5), "test": (400, 404, False, 0.0), "test_custom": (400, 505, True, 0.5)}
# Within 5% of a limit, remaining life is short and harder to predict, so the 80% range is
# calibrated separately there (Mondrian conformal prediction).
NEAR_LIMIT = 0.95


def subset_key(signals) -> str:
    """Stable name for a combination of measured indicators, in SIGNALS order."""
    chosen = set(signals)
    return "+".join(signal for signal in SIGNALS if signal in chosen)


def measured_signals(readings: pd.DataFrame) -> tuple[str, ...]:
    """Indicators present in every reading of this history."""
    return tuple(signal for signal in SIGNALS if signal in readings and readings[signal].notna().all())


def full_limits(limits: dict | None = None) -> dict:
    """The default limits with any user limits laid over them."""
    return {**EOL_LIMITS, **(limits or {})}


def _allowance(limits: dict) -> dict:
    """How far each indicator may move from its baseline before end of life, in its own terms."""
    return {"Capacitance_uF": limits["capacitance_loss_pct"], "ESR_ohm": limits["esr_ratio"] - 1,
            "Leakage_uA": limits["leakage_ratio"] - 1, "Case_Expansion_um": limits["expansion_um"],
            "H2_Generated_mL_STP": limits["h2_ml"], "Pressure_Abs_kPa": limits["pressure_abs_kpa"] - ATMOSPHERE_KPA}


def limit_features(limits: dict | None = None) -> np.ndarray:
    """Each limit relative to its default, as the network sees it (0 = default)."""
    given, default = _allowance(full_limits(limits)), _allowance(EOL_LIMITS)
    return np.array([np.log(given[signal] / default[signal]) for signal in SIGNALS])


def limits_outside_training(limits: dict | None) -> list[str]:
    """Limits outside the ranges the model was trained on."""
    limits = full_limits(limits)
    return [key for key, (low, high) in LIMIT_RANGES.items() if not low - 1e-9 <= limits[key] <= high + 1e-9]


def value_at_fraction(signal: str, baseline: pd.Series, fraction, limits: dict | None = None):
    """The indicator value at which the given fraction of its limit is used (inverse of the fractions)."""
    allowed = _allowance(full_limits(limits))[signal]
    if signal == "Capacitance_uF":
        return baseline[signal] * (1 - fraction * allowed / 100)
    if signal in ("ESR_ohm", "Leakage_uA"):
        return baseline[signal] * (1 + fraction * allowed)
    if signal == "Pressure_Abs_kPa":
        return ATMOSPHERE_KPA + fraction * allowed
    return baseline[signal] + fraction * allowed


def limit_used(readings: pd.DataFrame, limits: dict | None = None) -> pd.DataFrame:
    """Fraction of each indicator's end-of-life limit used so far (NaN if not measured).

    Aging damage does not heal: a capacitor whose ESR doubled and then partly fell back has still
    used up that part of its life. Each value is therefore the worse of the current reading and the
    worst level held so far, where "held" means the median of three consecutive readings, so a
    single outlying reading cannot mark lasting damage. Readings must be in time order.
    """
    return np.maximum(current_limit_used(readings, limits), held_limit_used(readings, limits))


def held_limit_used(readings: pd.DataFrame, limits: dict | None = None) -> pd.DataFrame:
    """Worst fraction of each limit held so far, by the median of three consecutive readings."""
    return current_limit_used(readings, limits).rolling(3, min_periods=1).median().cummax()


def _network_inputs(readings: pd.DataFrame, limits: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Limit fractions the network sees: (past readings, newest reading).

    Damage does not heal, and one outlying reading should not drag down the predictions that
    follow it. When predicting at reading t, every earlier reading s already has its neighbour
    s + 1, so it is cleaned with the median of readings s - 1, s and s + 1 and the worst cleaned
    level is held. The newest reading has no later neighbour and enters as measured, or the
    worst cleaned level before it, whichever is higher. Nothing after t is used.
    """
    current = current_limit_used(readings, limits)
    cleaned = current.rolling(3, center=True, min_periods=1).median()
    cleaned.iloc[0] = current.iloc[0]
    past = cleaned.cummax()
    newest = np.maximum(current, past.shift(1).fillna(current))
    return past.to_numpy(dtype=float), newest.to_numpy(dtype=float)


def current_limit_used(readings: pd.DataFrame, limits: dict | None = None) -> pd.DataFrame:
    """Fraction of each indicator's end-of-life limit used at each reading alone, relative to the first."""
    allowed = _allowance(full_limits(limits))
    first = readings.iloc[0]
    nan = pd.Series(np.nan, index=readings.index)
    column = lambda name: readings[name] if name in readings else nan
    base = lambda name: first[name] if name in readings else np.nan
    return pd.DataFrame({
        "Capacitance_uF": (1 - column("Capacitance_uF") / base("Capacitance_uF")) * 100 / allowed["Capacitance_uF"],
        "ESR_ohm": (column("ESR_ohm") / base("ESR_ohm") - 1) / allowed["ESR_ohm"],
        "Leakage_uA": (column("Leakage_uA") / base("Leakage_uA") - 1) / allowed["Leakage_uA"],
        "Case_Expansion_um": (column("Case_Expansion_um") - base("Case_Expansion_um")) / allowed["Case_Expansion_um"],
        "H2_Generated_mL_STP": (column("H2_Generated_mL_STP") - base("H2_Generated_mL_STP"))
        / allowed["H2_Generated_mL_STP"],
        "Pressure_Abs_kPa": (column("Pressure_Abs_kPa") - ATMOSPHERE_KPA) / allowed["Pressure_Abs_kPa"],
    }, index=readings.index)


def condition_values(readings: pd.DataFrame) -> pd.DataFrame:
    """Each operating condition at every reading in user units (NaN where it cannot be worked out).

    A condition recorded at some readings carries forward (and back to the first reading), since
    operating conditions hold until they are changed.
    """
    column = lambda name: (pd.to_numeric(readings[name], errors="coerce") if name in readings
                           else pd.Series(np.nan, index=readings.index))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = pd.DataFrame({
            "core_temperature": column("Core_C"), "ambient_temperature": column("Ambient_C"),
            "voltage_ratio": column("Voltage_V") / column("Rated_V"),
            "ripple_ratio": column("Ripple_A") / column("Rated_Ripple_A"), "frequency": column("Ripple_Hz"),
        }, index=readings.index)
    values = values.where(np.isfinite(values))
    values.loc[:, "frequency"] = values["frequency"].where(values["frequency"] > 0)
    return values.ffill().bfill()


def condition_features(values: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Network inputs for ``condition_values``: scaled values (0 where unknown) and recorded flags."""
    scaled = np.column_stack([(values["core_temperature"] - 85) / 20, (values["ambient_temperature"] - 85) / 20,
                              values["voltage_ratio"] - 0.75, values["ripple_ratio"] - 0.75,
                              np.log10(values["frequency"] / 120) / 3])
    known = np.isfinite(scaled)
    return np.where(known, scaled, 0.0), known.astype(float)


def conditions_outside_training(values: pd.Series) -> list[str]:
    """Recorded conditions outside the ranges the model was trained on (labels)."""
    return [CONDITION_LABELS[name] for name, (low, high) in CONDITION_TRAINING_RANGES.items()
            if np.isfinite(values.get(name, np.nan)) and not low * 0.98 <= values[name] <= high * 1.02]


def sequences(readings: pd.DataFrame, measured: tuple[str, ...] | None = None,
              limits: dict | None = None) -> np.ndarray:
    """One ``WINDOW`` × features array per reading from the third onward, using only past readings.

    Indicators outside ``measured`` (default: those present in every reading) enter as zero
    with their measured flag off.
    """
    measured = measured_signals(readings) if measured is None else tuple(measured)
    mask = np.array([signal in measured for signal in SIGNALS], dtype=float)
    age = (readings["Hours"] - readings["Hours"].iloc[0]).to_numpy(dtype=float)
    gap = np.concatenate([[0.0], np.diff(age)])
    past, newest = (np.nan_to_num(values) * mask for values in _network_inputs(readings, limits))
    conditions, recorded = condition_features(condition_values(readings))
    steps = np.column_stack([np.log1p(age / 1000), np.log1p(gap / 1000), past,
                             np.tile(mask, (len(age), 1)), np.tile(limit_features(limits) * mask, (len(age), 1)),
                             conditions, recorded, np.zeros(len(age))])
    windows = []
    for end in range(MIN_READINGS - 1, len(steps)):
        window = steps[max(0, end - WINDOW + 1):end + 1].copy()
        window[-1, USED] = newest[end]
        if len(window) < WINDOW:
            padding = np.zeros((WINDOW - len(window), steps.shape[1]))
            padding[:, -1] = 1.0
            window = np.vstack([padding, window])
        windows.append(window)
    return np.asarray(windows, dtype=np.float32).reshape(-1, WINDOW, len(STEP_FEATURES))


def hide_indicators(x: np.ndarray, keep: np.ndarray, keep_conditions: np.ndarray | None = None) -> np.ndarray:
    """Copy of full-indicator sequences with unkept indicators zeroed and flagged as not measured.

    ``keep`` is (n, len(SIGNALS)) booleans, one row per sequence; ``keep_conditions`` (n, len(CONDITIONS))
    likewise hides operating conditions (default: all kept).
    """
    hidden = x.copy()
    real = hidden[:, :, -1:] == 0  # padding rows stay all zero
    keep = keep[:, None, :].astype(hidden.dtype)
    hidden[:, :, USED] *= keep
    hidden[:, :, LIMIT] *= keep
    hidden[:, :, MEASURED] = keep * real
    if keep_conditions is not None:
        keep_conditions = keep_conditions[:, None, :].astype(hidden.dtype)
        hidden[:, :, COND] *= keep_conditions
        hidden[:, :, COND_KNOWN] *= keep_conditions
    return hidden


def no_conditions(n: int) -> np.ndarray:
    return np.zeros((n, K), dtype=bool)


def _random_condition_masks(rng: np.random.Generator, n: int) -> np.ndarray:
    """Training mix: 35% every condition recorded, 30% none, 35% each recorded with probability 1/2."""
    masks = rng.random((n, K)) < 0.5
    choice = rng.random(n)
    masks[choice < 0.35] = True
    masks[(choice >= 0.35) & (choice < 0.65)] = False
    return masks


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


class TrajectoryNet(nn.Module):
    """GRU encoder whose head forecasts each indicator's fraction of its limit at ``HORIZON_POINTS``
    points to end of life: 10th, 50th and 90th percentiles, the median never falling."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.rnn = nn.GRU(len(STEP_FEATURES), hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, HORIZON_POINTS * N * 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(x)
        raw = self.head(state[-1]).view(-1, HORIZON_POINTS, N, 3)
        softplus = nn.functional.softplus
        anchor = x[:, -1, USED]  # the newest fractions (zero for unmeasured indicators)
        median = anchor[:, None, :] + torch.cumsum(softplus(raw[..., 0]), dim=1)
        return torch.stack([median - softplus(raw[..., 1]), median, median + softplus(raw[..., 2])], dim=-1)


def _future_fractions(frame: pd.DataFrame, when: np.ndarray, hours: np.ndarray, truth: dict,
                      limits: dict) -> np.ndarray:
    """True (noise-free) fractions of each limit at the times ``when``, against the measured baseline."""
    future = pd.DataFrame({signal: np.interp(when.ravel(), hours, truth[signal]) for signal in SIGNALS})
    both = pd.concat([frame.iloc[[0]][list(SIGNALS)], future], ignore_index=True)
    return current_limit_used(both, limits).to_numpy(dtype=float)[1:].reshape(*when.shape, N)


def _fleet_data(n: int, seed: int, prefix: str, vary_limits: bool, near_end_share: float = 0.0) -> dict:
    """Simulated capacitors with network inputs, log RUL targets and future-fraction targets."""
    parts = {"x": [], "y": [], "path": [], "devices": []}

    def collect(frame, meta, hours, truth):
        limits = {key[len("limit_"):]: value for key, value in meta.items() if key.startswith("limit_")}
        ages = frame["Hours"].to_numpy(dtype=float)[MIN_READINGS - 1:]
        remaining = meta["eol_h"] - ages
        parts["x"].append(sequences(frame, SIGNALS, limits))
        parts["y"].append(np.log(remaining + RUL_OFFSET_H))
        when = ages[:, None] + remaining[:, None] * HORIZON_FRACTIONS[None, :]
        parts["path"].append(_future_fractions(frame, when, hours, truth, limits))
        parts["devices"] += [meta["Cap_ID"]] * len(ages)

    measurements, meta = simulate_fleet(n, seed, prefix=prefix, vary_limits=vary_limits, on_truth=collect,
                                        near_end_share=near_end_share)
    return {"measurements": measurements, "meta": meta, "x": np.concatenate(parts["x"]),
            "y": np.concatenate(parts["y"]).astype(np.float32),
            "path": np.concatenate(parts["path"]).astype(np.float32), "devices": np.asarray(parts["devices"])}


def _pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(QUANTILES, dtype=prediction.dtype)
    error = target[:, None] - prediction
    return torch.maximum(levels * error, (levels - 1) * error).mean()


def _path_pinball(prediction: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Pinball loss of path quantiles, counting only the measured indicators."""
    levels = torch.tensor(QUANTILES, dtype=prediction.dtype)
    error = target[..., None] - prediction
    loss = torch.maximum(levels * error, (levels - 1) * error)
    weight = keep[:, None, :, None].to(loss.dtype).expand_as(loss)
    return (loss * weight).sum() / weight.sum()


def _predict(net: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return net(torch.from_numpy(x)).numpy()


def _limits_by_device(meta: pd.DataFrame) -> dict:
    columns = [column for column in meta if column.startswith("limit_")]
    return {row["Cap_ID"]: {column[len("limit_"):]: row[column] for column in columns}
            for row in meta.to_dict("records")}


def _trend_baseline(measurements: pd.DataFrame, meta: pd.DataFrame, subset=SIGNALS) -> np.ndarray:
    """Hours until the first measured indicator reaches its limit, extrapolating the last three readings."""
    columns = [SIGNALS.index(signal) for signal in subset]
    limits = _limits_by_device(meta)
    predictions = []
    for device, readings in measurements.groupby("Cap_ID", sort=False):
        used = limit_used(readings, limits[device]).to_numpy(dtype=float)[:, columns]
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


def _fit(net: nn.Module, loss_fn, x_train, target_train, x_val, target_val, rng, epochs: int):
    """Adam with plateau learning-rate decay and early stopping on a fixed validation mix."""
    keep_val = _random_masks(rng, len(x_val))
    x_v = torch.from_numpy(hide_indicators(x_val, keep_val, _random_condition_masks(rng, len(x_val))))
    t_v, k_v = torch.from_numpy(target_val), torch.from_numpy(keep_val)
    t_t = torch.from_numpy(target_train)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    best, best_state, stale, best_epoch = math.inf, None, 0, 0
    for epoch in range(epochs):
        net.train()
        keep = _random_masks(rng, len(x_train))  # new hidden indicators and conditions every epoch
        x_t = torch.from_numpy(hide_indicators(x_train, keep, _random_condition_masks(rng, len(x_train))))
        k_t = torch.from_numpy(keep)
        for batch in torch.from_numpy(rng.permutation(len(x_t))).split(512):
            optimizer.zero_grad()
            loss = loss_fn(net(x_t[batch]), t_t[batch], k_t[batch])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val = float(loss_fn(net(x_v), t_v, k_v))
        scheduler.step(val)
        if val < best - 1e-5:
            best, best_state, stale, best_epoch = val, {k: v.clone() for k, v in net.state_dict().items()}, 0, epoch + 1
        else:
            stale += 1
            if stale >= 12:
                break
    net.load_state_dict(best_state)
    net.eval()
    return best, best_epoch


def _conformal(scores: np.ndarray) -> float:
    """The split-conformal quantile of nonconformity scores for an ``INTERVAL_MASS`` range."""
    rank = min(math.ceil((len(scores) + 1) * INTERVAL_MASS), len(scores))
    return float(np.sort(scores)[rank - 1])


def _path_metrics(path: np.ndarray, truth: np.ndarray, measured: np.ndarray) -> dict:
    """Accuracy and 80% coverage of forecast paths, in fractions of each limit, per indicator."""
    out = {}
    for index, signal in enumerate(SIGNALS):
        if not measured[index]:
            continue
        low, mid, high = (path[:, :, index, q] for q in range(3))
        target = truth[:, :, index]
        out[signal] = {
            "median_abs_error_halfway": round(float(np.median(np.abs(mid[:, HORIZON_POINTS // 2 - 1]
                                                                  - target[:, HORIZON_POINTS // 2 - 1]))), 4),
            "median_abs_error_at_end_of_life": round(float(np.median(np.abs(mid[:, -1] - target[:, -1]))), 4),
            "interval_coverage": round(float(((target >= low) & (target <= high)).mean()), 4),
        }
    return out


def path_adjustment(path_net: nn.Module, cal: dict, rng: np.random.Generator) -> np.ndarray:
    """One additive range adjustment per forecast point and indicator (split-conformal).

    Every calibration reading is used, with the training mix of measured indicators and recorded
    conditions: one reading per capacitor left too few scores per point for a stable 80% range.
    """
    n = len(cal["x"])
    keep = _random_masks(rng, n)
    predicted = _predict(path_net, hide_indicators(cal["x"], keep, _random_condition_masks(rng, n)))
    adjust = np.zeros((HORIZON_POINTS, N))
    for index in range(N):
        rows = keep[:, index]
        scores = np.maximum(predicted[rows, :, index, 0] - cal["path"][rows, :, index],
                            cal["path"][rows, :, index] - predicted[rows, :, index, 2])
        level = math.ceil((rows.sum() + 1) * INTERVAL_MASS) / rows.sum()
        adjust[:, index] = np.quantile(scores, min(level, 1.0), axis=0)
    return adjust


def trajectory_metrics(path_net: nn.Module, data: dict, path_adjust: np.ndarray) -> dict:
    """Forecast-path accuracy and coverage on the test fleets, with and without operating conditions."""
    metrics = {}
    for name in ("test", "test_custom"):
        test = data[name]
        for label, subset in (("all six", SIGNALS), ("capacitance, ESR, leakage", COMMON_SUBSET)):
            keep = np.tile(_subset_mask(subset), (len(test["x"]), 1))
            for suffix, conditions in (("", None), (", no operating conditions", no_conditions(len(test["x"])))):
                p = _predict(path_net, hide_indicators(test["x"], keep, conditions))
                p[..., 0] -= path_adjust
                p[..., 2] += path_adjust
                metrics[f"{name}, {label}{suffix}"] = _path_metrics(p, test["path"], _subset_mask(subset))
    return metrics


def train(output: Path, seed: int = 7, epochs: int = 80, hidden: int = HIDDEN) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = {name: _fleet_data(n, fleet_seed, name[:3].upper() + ("C" if name == "test_custom" else ""), vary, near)
            for name, (n, fleet_seed, vary, near) in SPLITS.items()}
    train_data, val = data["train"], data["validation"]
    rng = np.random.default_rng(seed)

    net = SixRULNet(hidden)
    best, best_epoch = _fit(net, lambda q, y, keep: _pinball(q, y), train_data["x"], train_data["y"],
                            val["x"], val["y"], rng, epochs)
    path_net = TrajectoryNet(hidden)
    path_best, path_epoch = _fit(path_net, _path_pinball, train_data["x"], train_data["path"],
                                 val["x"], val["path"], rng, epochs)

    # Conformalized quantile regression for every combination of measured indicators, separately for
    # readings within 5% of a limit and the rest. Away from the limits, one randomly chosen reading per
    # calibration capacitor keeps the scores independent; near a limit every such reading is used.
    cal = data["calibration"]
    pick = np.array([rng.choice(np.flatnonzero(cal["devices"] == device)) for device in np.unique(cal["devices"])])
    # Each capacitor is scored twice, with all operating conditions recorded and with none, so the
    # ranges hold whether or not a user records them.
    adjustments, near_adjustments, by_subset, custom_by_subset, bare_by_subset = {}, {}, {}, {}, {}
    n_cal = len(cal["x"])
    for subset in SUBSETS:
        keep = np.tile(_subset_mask(subset), (n_cal, 1))
        x_cal = np.concatenate([hide_indicators(cal["x"], keep), hide_indicators(cal["x"], keep, no_conditions(n_cal))])
        q_cal = _predict(net, x_cal)
        y_cal = np.concatenate([cal["y"], cal["y"]])
        scores = np.maximum(q_cal[:, 0] - y_cal, y_cal - q_cal[:, 2])
        near = x_cal[:, -1, USED].max(axis=1) >= NEAR_LIMIT
        both = np.concatenate([pick, pick + n_cal])
        far_rows = both[~near[both]]
        adjustment = _conformal(scores[far_rows])
        near_adjustment = _conformal(scores[near]) if near.sum() >= 30 else adjustment
        key = subset_key(subset)
        adjustments[key], near_adjustments[key] = round(adjustment, 6), round(near_adjustment, 6)
        for name, store, conditions in (("test", by_subset, True), ("test_custom", custom_by_subset, True),
                                        ("test", bare_by_subset, False)):
            test = data[name]
            n_test = len(test["x"])
            x_test = hide_indicators(test["x"], np.tile(_subset_mask(subset), (n_test, 1)),
                                     None if conditions else no_conditions(n_test))
            q = _predict(net, x_test)
            pad = np.where(x_test[:, -1, USED].max(axis=1) >= NEAR_LIMIT, near_adjustment, adjustment)
            store[key] = _metrics(np.exp(test["y"]) - RUL_OFFSET_H, _to_hours(q[:, 0] - pad),
                                  _to_hours(q[:, 1]), _to_hours(q[:, 2] + pad))

    path_adjust = path_adjustment(path_net, cal, rng)
    path_metrics = trajectory_metrics(path_net, data, path_adjust)

    # Accuracy by how close the nearest parameter already is to its limit (random-limit test capacitors).
    custom = data["test_custom"]
    q = _predict(net, custom["x"])
    full = subset_key(SIGNALS)
    nearest = custom["x"][:, -1, USED].max(axis=1)
    adjustment = np.where(nearest >= NEAR_LIMIT, near_adjustments[full], adjustments[full])
    by_closeness = {}
    for low_edge, high_edge in ((0.0, 0.8), (0.8, 0.95), (0.95, 1.0)):
        rows = (nearest >= low_edge) & (nearest < high_edge)
        by_closeness[f"{low_edge:.2f}-{high_edge:.2f}"] = _metrics(
            np.exp(custom["y"][rows]) - RUL_OFFSET_H, _to_hours(q[rows, 0] - adjustment[rows]),
            _to_hours(q[rows, 1]), _to_hours(q[rows, 2] + adjustment[rows]))

    baselines = {}
    test = data["test"]
    rul = np.exp(test["y"]) - RUL_OFFSET_H
    for subset in BASELINE_SUBSETS:
        trend = _trend_baseline(test["measurements"], test["meta"], subset)
        available = np.isfinite(trend)
        baselines[subset_key(subset)] = {
            "predictions_with_estimate": int(available.sum()),
            "median_absolute_error_h": round(float(np.median(np.abs(trend[available] - rul[available]))), 1),
        }
    x_train = train_data["x"]
    train_used = x_train[:, :, USED][x_train[:, :, -1] == 0]
    metadata = {
        "hidden": hidden,
        "model": f"SixRULNet GRU, hidden {hidden}, quantile outputs (10th, 50th, 90th percentile of log RUL), "
                 "trained with randomly hidden indicators and operating conditions and random end-of-life limits",
        "conditions": list(CONDITIONS), "condition_training_ranges": CONDITION_TRAINING_RANGES,
        "trajectory_model": f"TrajectoryNet GRU, hidden {hidden}: 10th, 50th and 90th percentiles of each "
                            f"indicator's fraction of its limit at {HORIZON_POINTS} points to end of life",
        "training_data": "Simulated capacitors from caplife.six_sim; not real measurements",
        "signals": list(SIGNALS), "step_features": list(STEP_FEATURES), "window": WINDOW,
        "min_readings": MIN_READINGS, "rul_offset_h": RUL_OFFSET_H, "horizon_points": HORIZON_POINTS,
        "eol_limits": EOL_LIMITS, "limit_ranges": LIMIT_RANGES, "interval_mass": INTERVAL_MASS,
        "conformal_adjustment_log": adjustments[full],
        "conformal_adjustment_log_by_subset": adjustments,
        "near_limit_fraction": NEAR_LIMIT,
        "conformal_adjustment_log_near_limit_by_subset": near_adjustments,
        "trajectory_conformal_adjustment": np.round(path_adjust, 6).tolist(),
        "fleets": {name: {"capacitors": n, "seed": fleet_seed, "limits": "random" if vary else "default",
                          "near_end_share": near, "predictions": int(len(data[name]["y"]))}
                   for name, (n, fleet_seed, vary, near) in SPLITS.items()},
        "training_seed": seed, "best_epoch": best_epoch, "validation_pinball": round(best, 5),
        "trajectory_best_epoch": path_epoch, "trajectory_validation_pinball": round(path_best, 5),
        "limit_used_training_range": {
            "min": [round(float(v), 4) for v in train_used.min(axis=0)],
            "max": [round(float(v), 4) for v in train_used.max(axis=0)],
        },
        "test_metrics": {"neural_calibrated": by_subset[full], "trend_baseline": baselines[full],
                         "neural_calibrated_custom_limits": custom_by_subset[full],
                         "neural_calibrated_without_conditions": bare_by_subset[full]},
        "test_metrics_by_subset": by_subset,
        "test_metrics_by_subset_without_conditions": bare_by_subset,
        "test_metrics_by_subset_custom_limits": custom_by_subset,
        "trend_baseline_by_subset": baselines,
        "trajectory_test_metrics": path_metrics,
        "test_metrics_custom_limits_by_closeness_to_limit": by_closeness,
        "end_of_life_causes_in_test_fleet": test["meta"]["eol_cause"].str.split(", ").str[0]
        .value_counts().to_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output / "model.pt")
    torch.save(path_net.state_dict(), output / "trajectory.pt")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class SixRULModel:
    """Loaded models plus the health, remaining-life and trend calculations the app shows."""

    def __init__(self, artifact_dir: Path):
        self.metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        hidden = self.metadata.get("hidden", HIDDEN)
        self.net = SixRULNet(hidden)
        self.net.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.path_net = TrajectoryNet(hidden)
        self.path_net.load_state_dict(torch.load(artifact_dir / "trajectory.pt", map_location="cpu",
                                                 weights_only=True))
        self.path_net.eval()
        self.path_adjust = np.asarray(self.metadata["trajectory_conformal_adjustment"])
        self.adjustments = self.metadata["conformal_adjustment_log_by_subset"]
        self.near_adjustments = self.metadata.get("conformal_adjustment_log_near_limit_by_subset", self.adjustments)
        self.near_limit = self.metadata.get("near_limit_fraction", NEAR_LIMIT)
        self.train_max = np.asarray(self.metadata["limit_used_training_range"]["max"])

    def metrics_for(self, measured, with_conditions: bool = True) -> dict:
        """Test-fleet accuracy for this combination of measured indicators, with or without conditions."""
        key = "test_metrics_by_subset" if with_conditions else "test_metrics_by_subset_without_conditions"
        return self.metadata.get(key, self.metadata["test_metrics_by_subset"])[subset_key(measured)]

    @staticmethod
    def _prepare(readings: pd.DataFrame, measured) -> tuple[pd.DataFrame, tuple[str, ...]]:
        readings = readings.sort_values("Hours").reset_index(drop=True)
        measured = measured_signals(readings) if measured is None else tuple(measured)
        if not measured:
            raise ValueError("Supply at least one of the six indicators in every reading.")
        ratio_based = [signal for signal in ("Capacitance_uF", "ESR_ohm", "Leakage_uA") if signal in measured]
        if (readings.loc[0, ratio_based] <= 0).any():
            raise ValueError("The first reading needs positive capacitance, ESR and leakage current "
                             "(where measured) to serve as a baseline.")
        return readings, measured

    def assess(self, readings: pd.DataFrame, measured: tuple[str, ...] | None = None,
               limits: dict | None = None) -> pd.DataFrame:
        """Health and remaining life at every reading from the third onward, each using only past readings.

        Only the ``measured`` indicators (default: those present in every reading) are used; the
        health columns of the others are NaN. ``limits`` are the user's end-of-life limits
        (defaults for any not given).
        """
        readings, measured = self._prepare(readings, measured)
        used = limit_used(readings, limits)[list(measured)]
        seen = pd.DataFrame(_network_inputs(readings, limits)[1], columns=SIGNALS)[list(measured)]  # newest inputs
        key = subset_key(measured)
        columns = [SIGNALS.index(signal) for signal in measured]
        rows = []
        if len(readings) >= MIN_READINGS:
            q = _predict(self.net, sequences(readings, measured, limits))
            # Remaining life can only grow when a limit is loosened, so it is never predicted below that
            # for the limits tightened to the defaults (each quantile is monotone in the limits).
            floor = {key: min(value, EOL_LIMITS[key]) for key, value in full_limits(limits).items()}
            if floor != full_limits(limits):
                q = np.maximum(q, _predict(self.net, sequences(readings, measured, floor)))
            near = seen.to_numpy()[MIN_READINGS - 1:].max(axis=1) >= self.near_limit
            pad = np.where(near, self.near_adjustments[key], self.adjustments[key])
            low, mid, high = _to_hours(q[:, 0] - pad), _to_hours(q[:, 1]), _to_hours(q[:, 2] + pad)
        for index in range(len(readings)):
            fractions = used.iloc[index]
            row = {"Hours": float(readings.loc[index, "Hours"])}
            for signal in SIGNALS:
                row[f"{signal}_health_pct"] = (float(100 * np.clip(1 - fractions[signal], 0, 1))
                                               if signal in measured else np.nan)
            row["Overall_health_pct"] = min(row[f"{signal}_health_pct"] for signal in measured)
            row["Limiting_indicator"] = INDICATOR_LABELS[fractions.idxmax()]
            reached = bool((fractions >= 1 - 1e-9).any())
            row["End_of_life_reached"] = reached
            k = index - (MIN_READINGS - 1)
            if k >= 0 and not reached:
                row.update({"RUL_low_h": low[k], "RUL_median_h": mid[k], "RUL_high_h": high[k]})
                row["Outside_training_range"] = bool(
                    (seen.iloc[index].to_numpy() > self.train_max[columns] * 1.1).any())
            elif reached:
                row.update({"RUL_low_h": 0.0, "RUL_median_h": 0.0, "RUL_high_h": 0.0, "Outside_training_range": False})
            rows.append(row)
        return pd.DataFrame(rows)

    def forecast(self, readings: pd.DataFrame, remaining_h: float, measured: tuple[str, ...] | None = None,
                 limits: dict | None = None) -> pd.DataFrame:
        """Each measured indicator's forecast path from the latest reading to end of life.

        ``remaining_h`` (the predicted median remaining life) sets the time axis: the network gives
        each indicator's value at 10%, 20%, ..., 100% of it, as an 80% range and a median, starting
        from the latest reading's level (the worst level held, since damage does not heal).
        """
        readings, measured = self._prepare(readings, measured)
        if len(readings) < MIN_READINGS:
            raise ValueError(f"A forecast needs at least {MIN_READINGS} readings.")
        x = sequences(readings, measured, limits)[-1:]
        path = _predict(self.path_net, x)[0]  # (points, indicators, quantiles) in fractions of limits
        path[..., 0] -= self.path_adjust
        path[..., 2] += self.path_adjust
        start = x[0, -1, USED]
        last_h = float(readings["Hours"].iloc[-1])
        hours = np.concatenate([[last_h], last_h + remaining_h * HORIZON_FRACTIONS])
        baseline = readings.iloc[0]
        rows = []
        for signal in measured:
            index = SIGNALS.index(signal)
            low = np.concatenate([[start[index]], np.maximum(path[:, index, 0], start[index])])
            mid = np.concatenate([[start[index]], path[:, index, 1]])
            high = np.concatenate([[start[index]], path[:, index, 2]])
            for when, fractions in zip(hours, zip(low, mid, high)):
                values = [value_at_fraction(signal, baseline, f, limits) for f in fractions]
                rows.append({"Hours": when, "signal": signal, "fraction_median": fractions[1],
                             "low": min(values), "median": values[1], "high": max(values)})
        return pd.DataFrame(rows)


def example_fleet(seed: int = 2027, devices_per_cause: int = 2) -> tuple[pd.DataFrame, dict]:
    """Simulated capacitors observed part of the way to end of life, and their true end-of-life ages.

    Two capacitors per end-of-life cause are taken from a larger simulated fleet, each cut off at a
    different stage of its life, so the example shows every failure mode and a range of health.
    """
    measurements, meta = simulate_fleet(120, seed, prefix="POOL")
    meta = meta.assign(cause=meta["eol_cause"].str.split(", ").str[0])
    chosen = pd.concat([meta.loc[meta["cause"].eq(cause)].head(devices_per_cause)
                        for cause in EOL_CAUSES])
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
