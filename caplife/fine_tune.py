"""Fine-tune the built-in six-parameter models on a lab's own tested capacitors (prototype).

The result is a model of the same kind as the built-in one (``SixRULModel``): the same two
networks, the same inputs (any combination of the six parameters, end-of-life limits, operating
conditions) and the same outputs (remaining life with an 80% range, each parameter's path to end
of life), so every page feature works with it. Training starts from the built-in weights and
mixes in simulated capacitors, so what the built-in model knows is kept where the lab's data say
nothing.

End of life comes two ways. Wear-out is a parameter reaching its limit, which the two networks
predict. A sudden failure (open circuit, short, vent) happens while every parameter is still
inside its limits; when a lab's outcomes include such failures, a separate duration model learns
how their risk depends on age, the parameters and the operating conditions. Remaining life is
then whichever comes first, and its 80% range is calibrated on the lab's calibration capacitors.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, ndtri

from caplife import six_rul
from caplife.rul_workflow import landmarks as duration_landmarks
from caplife.six_rul import (
    HORIZON_FRACTIONS, INTERVAL_MASS, MIN_READINGS, QUANTILES, RUL_OFFSET_H, SIGNALS, SixRULModel,
    current_limit_used, full_limits, hide_indicators, limit_used, sequences,
)
from caplife.six_sim import LIMIT_RANGES
from caplife.survival import DurationModel, fit_duration

LIMIT_MODE, SUDDEN_MODE, RUNNING_MODE = "limit", "sudden", "running"
ALTERNATE_LIMITS = 4          # extra limit sets per capacitor, so the model keeps answering for any limits
REPLAY_CAPACITORS = 300       # simulated capacitors mixed in, so the built-in knowledge is kept
REPLAY_WEIGHT = 0.5
EPOCHS = 30
LEARNING_RATE = 5e-4
Z90 = float(ndtri(0.9))
# The 80% range is calibrated separately early and late in life: once some parameter has used this share of
# its limit, the readings show how fast the capacitor is wearing and the range can be much tighter.
LATE_STAGE = 0.5
BUILT_IN_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"
REPLAY_SEED = 4242


@lru_cache(maxsize=1)
def built_in() -> SixRULModel:
    return SixRULModel(BUILT_IN_DIR)


@lru_cache(maxsize=1)
def replay_data() -> dict:
    """Simulated capacitors like the built-in training ones, made once per app start (a few seconds)."""
    data = six_rul._fleet_data(REPLAY_CAPACITORS, REPLAY_SEED, "REP", True, 0.0)
    return {"x": data["x"], "y": data["y"], "path": data["path"]}


# --- What happened to each capacitor ------------------------------------------------------------

def _fractions(readings: pd.DataFrame, signals: tuple[str, ...], limits: dict) -> np.ndarray:
    """The largest fraction of any measured parameter's limit used at each reading."""
    return limit_used(readings, limits)[list(signals)].max(axis=1).to_numpy(dtype=float)


def crossing_age(readings: pd.DataFrame, signals: tuple[str, ...], limits: dict) -> float | None:
    """Age at which a measured parameter first reached its limit, interpolated between readings."""
    used = _fractions(readings, signals, limits)
    hours = readings["Hours"].to_numpy(dtype=float)
    reached = np.flatnonzero(used >= 1)
    if not reached.size:
        return None
    k = reached[0]
    if k == 0:
        return float(hours[0])
    share = (1 - used[k - 1]) / max(used[k] - used[k - 1], 1e-9)
    return float(hours[k - 1] + share * (hours[k] - hours[k - 1]))


def failure_mode(readings: pd.DataFrame, label, signals: tuple[str, ...], limits: dict) -> str:
    """Wear-out if the parameters were on course to reach a limit by the failure check, else sudden."""
    stated = str(getattr(label, "Failure_mode", "") or "").strip().lower()
    if stated in (LIMIT_MODE, SUDDEN_MODE):
        return stated
    if not np.isfinite(label.Failure_by_h):
        return RUNNING_MODE
    if crossing_age(readings, signals, limits) is not None:
        return LIMIT_MODE
    before = readings.loc[readings["Hours"] <= label.Failure_by_h]
    used = _fractions(before, signals, limits)[-3:]
    hours = before["Hours"].to_numpy(dtype=float)[-3:]
    if len(used) < 2:
        return SUDDEN_MODE
    slope, intercept = np.polyfit(hours, used, 1)
    return LIMIT_MODE if slope * label.Failure_by_h + intercept >= 0.97 else SUDDEN_MODE


def wear_out_target(readings: pd.DataFrame, label, mode: str, signals: tuple[str, ...],
                    limits: dict) -> tuple[float, bool]:
    """(age, observed): when wear-out happened for these limits, or the age it was still ahead at."""
    crossed = crossing_age(readings, signals, limits)
    if crossed is not None:
        return crossed, True
    if mode == LIMIT_MODE and limits == full_limits(limits) and np.isfinite(label.Failure_by_h):
        return 0.5 * (label.Last_alive_h + label.Failure_by_h), True
    last = float(readings["Hours"].iloc[-1])
    return (min(last, label.Last_alive_h) if mode != RUNNING_MODE else max(last, label.Last_alive_h)), False


def random_limits(rng: np.random.Generator) -> dict:
    """Limits drawn like the built-in training: log-uniform inside the trained ranges."""
    return {key: float(np.exp(rng.uniform(np.log(low), np.log(high)))) for key, (low, high) in LIMIT_RANGES.items()}


# --- Training examples --------------------------------------------------------------------------

@dataclass
class Examples:
    x: list = field(default_factory=list)          # network input windows
    y: list = field(default_factory=list)          # log(remaining + offset), a lower bound where censored
    observed: list = field(default_factory=list)   # False: wear-out was still ahead (censored)
    path: list = field(default_factory=list)       # future fractions of each limit (NaN unknown)
    measured: list = field(default_factory=list)   # which parameters are measured
    weight: list = field(default_factory=list)     # every reading counts equally (see add_capacitor)

    def arrays(self) -> dict:
        return {"x": np.concatenate(self.x).astype(np.float32), "y": np.concatenate(self.y).astype(np.float32),
                "observed": np.concatenate(self.observed), "path": np.concatenate(self.path).astype(np.float32),
                "measured": np.concatenate(self.measured), "weight": np.concatenate(self.weight).astype(np.float32)}


def _future_fractions(readings: pd.DataFrame, ages: np.ndarray, end: float, limits: dict) -> np.ndarray:
    """Measured fractions of each limit at 10%, 20%, ..., 100% of the way to ``end`` (NaN past the last reading)."""
    hours = readings["Hours"].to_numpy(dtype=float)
    used = current_limit_used(readings, limits).rolling(3, center=True, min_periods=1).median().cummax()
    used = used.to_numpy(dtype=float)
    when = ages[:, None] + (end - ages)[:, None] * HORIZON_FRACTIONS[None, :]
    out = np.full((*when.shape, len(SIGNALS)), np.nan)
    inside = when <= hours[-1] + 1e-9
    for index in range(len(SIGNALS)):
        if np.isfinite(used[:, index]).all():
            out[..., index] = np.where(inside, np.interp(when, hours, used[:, index]), np.nan)
    return out


def add_capacitor(examples: Examples, readings: pd.DataFrame, label, mode: str, signals: tuple[str, ...],
                  limits: dict, rng: np.random.Generator) -> None:
    """Every reading from the third onward, for the lab's limits and some random ones."""
    mask = np.array([signal in signals for signal in SIGNALS])
    limit_sets = [limits] + [random_limits(rng) for _ in range(ALTERNATE_LIMITS)]
    for chosen in limit_sets:
        end, observed = wear_out_target(readings, label, mode, signals, chosen)
        x = sequences(readings, signals, chosen)
        ages = readings["Hours"].to_numpy(dtype=float)[MIN_READINGS - 1:]
        keep = ages < end
        if not keep.any():
            continue
        x, ages = x[keep], ages[keep]
        examples.x.append(x)
        examples.y.append(np.log(end - ages + RUL_OFFSET_H))
        examples.observed.append(np.full(len(ages), observed))
        examples.path.append(_future_fractions(readings, ages, end, chosen) if observed
                             else np.full((len(ages), len(HORIZON_FRACTIONS), len(SIGNALS)), np.nan))
        examples.measured.append(np.tile(mask, (len(ages), 1)))
        # Every reading counts equally, as in the built-in training. Weighting each capacitor equally
        # instead would let short-lived capacitors, with few readings, dominate the early ages.
        examples.weight.append(np.full(len(ages), 1 / len(limit_sets)))


# --- Fine-tuning --------------------------------------------------------------------------------

def _censored_pinball(prediction, target, observed, weight):
    """Pinball loss; where wear-out was still ahead, only predictions below the bound are penalised."""
    levels = torch.tensor(QUANTILES, dtype=prediction.dtype)
    error = target[:, None] - prediction
    loss = torch.maximum(levels * error, (levels - 1) * error)
    loss = torch.where(observed[:, None], loss, levels * torch.relu(error))
    return (loss.mean(dim=1) * weight).sum() / weight.sum()


def _masked_path_pinball(prediction, target, keep, weight):
    levels = torch.tensor(QUANTILES, dtype=prediction.dtype)
    known = torch.isfinite(target) & keep[:, None, :]
    error = torch.nan_to_num(target)[..., None] - prediction
    loss = torch.maximum(levels * error, (levels - 1) * error).mean(dim=-1)
    w = known.to(loss.dtype) * weight[:, None, None]
    return (loss * w).sum() / w.sum().clamp_min(1e-9)


def _hide_some(rng: np.random.Generator, measured: np.ndarray) -> np.ndarray:
    """Half the time, hide some of the measured parameters (never all), so other combinations keep working."""
    keep = measured.copy()
    for row in np.flatnonzero(rng.random(len(keep)) < 0.5):
        on = np.flatnonzero(keep[row])
        if len(on) > 1:
            drop = rng.choice(on, rng.integers(1, len(on)), replace=False)
            keep[row, drop] = False
    return keep


def fine_tune(base: SixRULModel, data: dict, replay: dict, seed: int = 7, epochs: int = EPOCHS) -> tuple:
    """Both networks, continued from the built-in weights on the lab's examples plus simulated replay."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net, path_net = copy.deepcopy(base.net), copy.deepcopy(base.path_net)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    path_opt = torch.optim.Adam(path_net.parameters(), lr=LEARNING_RATE)
    y, observed, weight = (torch.from_numpy(data[k]) for k in ("y", "observed", "weight"))
    path = torch.from_numpy(data["path"])
    has_path = np.isfinite(data["path"]).any(axis=(1, 2))
    replay_y, replay_path = torch.from_numpy(replay["y"]), torch.from_numpy(replay["path"])
    n = len(data["x"])
    for _ in range(epochs):
        net.train()
        path_net.train()
        keep = _hide_some(rng, data["measured"])
        x = torch.from_numpy(hide_indicators(data["x"], keep, six_rul._random_condition_masks(rng, n) | (rng.random((n, 1)) < 0.7)))
        k = torch.from_numpy(keep)
        for batch in torch.from_numpy(rng.permutation(n)).split(128):
            sim = torch.from_numpy(rng.choice(len(replay["x"]), len(batch), replace=False))
            sim_keep = six_rul._random_masks(rng, len(batch))
            sim_x = torch.from_numpy(hide_indicators(replay["x"][sim.numpy()], sim_keep,
                                                     six_rul._random_condition_masks(rng, len(batch))))
            ones = torch.ones(len(batch))
            opt.zero_grad()
            loss = (_censored_pinball(net(x[batch]), y[batch], observed[batch], weight[batch])
                    + REPLAY_WEIGHT * _censored_pinball(net(sim_x), replay_y[sim], torch.ones(len(batch), dtype=torch.bool), ones))
            loss.backward()
            opt.step()
            rows = batch[torch.from_numpy(has_path[batch.numpy()])]
            if len(rows):
                path_opt.zero_grad()
                path_loss = (_masked_path_pinball(path_net(x[rows]), path[rows], k[rows], weight[rows])
                             + REPLAY_WEIGHT * six_rul._path_pinball(path_net(sim_x), replay_path[sim],
                                                                     torch.from_numpy(sim_keep)))
                path_loss.backward()
                path_opt.step()
    net.eval()
    path_net.eval()
    return net, path_net


# --- Sudden failures ----------------------------------------------------------------------------

@dataclass
class SuddenModel:
    """Time to a sudden failure from a reading: a log-normal whose centre depends on the readings."""
    duration: object
    signals: tuple[str, ...]
    conditions: tuple[str, ...]
    lows: np.ndarray
    highs: np.ndarray

    def features(self, readings: pd.DataFrame) -> dict[float, np.ndarray]:
        found = duration_landmarks(readings, self.signals, self.conditions)
        return {} if isinstance(found, str) else dict(found)

    def location_scale(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        # Beyond what the lab's capacitors showed (a voltage never tested, say), hold each input at the
        # nearest value seen, so the risk follows the closest tested case instead of extrapolating.
        return self.duration.location(np.clip(values, self.lows, self.highs)), self.duration.scale


def fit_sudden(histories: dict, labels: pd.DataFrame, modes: dict, wear_ends: dict, signals, conditions):
    """A sudden-failure duration model, or None when the lab saw no sudden failures."""
    if not any(modes[device] == SUDDEN_MODE for device in modes):
        return None
    rows, lower, upper, weight = [], [], [], []
    for label in labels.itertuples(index=False):
        device = label.Cap_ID
        found = duration_landmarks(histories[device], signals, conditions)
        if isinstance(found, str):
            continue
        sudden = modes[device] == SUDDEN_MODE
        ends = (label.Last_alive_h, label.Failure_by_h) if sudden else (wear_ends[device], math.inf)
        usable = [(age, values) for age, values in found if age < ends[0]]
        for age, values in usable:
            rows.append(values)
            lower.append(ends[0] - age)
            upper.append(ends[1] - age)
            weight.append(1.0)  # every reading counts equally (see add_capacitor)
    rows = np.asarray(rows)
    duration = fit_duration(rows, np.asarray(lower), np.asarray(upper), np.asarray(weight), hidden=True)
    return SuddenModel(duration, tuple(signals), tuple(conditions), rows.min(axis=0), rows.max(axis=0))


# --- Combining wear-out and sudden failure ------------------------------------------------------

GRID = np.exp(np.linspace(np.log(1.0), np.log(2e6), 1200))  # hours ahead


def _wear_cdf(q: np.ndarray) -> np.ndarray:
    """Wear-out CDF on GRID from the 10th/50th/90th percentiles of log(remaining + offset): a split normal."""
    z = np.log(GRID + RUL_OFFSET_H)
    low = max((q[1] - q[0]) / Z90, 1e-3)
    high = max((q[2] - q[1]) / Z90, 1e-3)
    return ndtr((z - q[1]) / np.where(z < q[1], low, high))


def combine(q: np.ndarray, sudden: tuple[float, float] | None) -> tuple[np.ndarray, float]:
    """Percentiles (log space) of whichever comes first, and the chance that it is the sudden failure."""
    if sudden is None:
        return q, 0.0
    wear = _wear_cdf(q)
    location, scale = sudden
    sudden_cdf = ndtr((np.log(GRID) - location) / scale)
    alive = (1 - wear) * (1 - sudden_cdf)
    first = 1 - alive
    out = np.interp(QUANTILES, first, np.log(GRID + RUL_OFFSET_H))
    density = np.diff(sudden_cdf)
    chance = float(np.sum(density * (1 - wear[1:])))
    return out, chance


# --- The trained model --------------------------------------------------------------------------

class TrainedModel(SixRULModel):
    """A ``SixRULModel`` with fine-tuned networks, the lab's calibration and, optionally, sudden failures."""

    def __init__(self, base: SixRULModel, net, path_net, adjustment: float, sudden: SuddenModel | None,
                 path_adjust: np.ndarray | None = None, late_adjustment: float | None = None):
        self.metadata = copy.deepcopy(base.metadata)
        self.metadata["hidden"] = net.rnn.hidden_size  # a model saved with an earlier built-in model may be smaller
        self.net, self.path_net = net, path_net
        self.path_adjust = base.path_adjust if path_adjust is None else path_adjust
        self.adjustments = dict(base.adjustments)
        self.near_limit = LATE_STAGE  # the built-in "near a limit" widening becomes the late-life one
        self.train_max = base.train_max
        self.sudden = sudden
        self.set_adjustment(adjustment, late_adjustment)

    def set_adjustment(self, early: float, late: float | None = None) -> None:
        """The lab's calibrated widening of the 80% range early and late in life (one value: both)."""
        self.adjustments = {key: early for key in self.adjustments}
        self.near_adjustments = {key: early if late is None else late for key in self.adjustments}

    def late(self, readings: pd.DataFrame, measured, limits) -> np.ndarray:
        """At every reading from the third: whether it is late in life (some parameter past LATE_STAGE of its limit)."""
        seen = pd.DataFrame(six_rul._network_inputs(readings, limits)[1], columns=SIGNALS)[list(measured)]
        return seen.to_numpy()[MIN_READINGS - 1:].max(axis=1) >= self.near_limit

    def raw_quantiles(self, readings: pd.DataFrame, measured, limits) -> np.ndarray:
        """Wear-out percentiles of log(remaining + offset) at every reading from the third, before calibration."""
        return six_rul._predict(self.net, sequences(readings, measured, limits))

    def combined(self, readings: pd.DataFrame, measured, limits) -> tuple[np.ndarray, np.ndarray]:
        """Percentiles of whichever comes first, and the chance it is sudden, at every reading from the third."""
        q = self.raw_quantiles(readings, measured, limits)
        if self.sudden is None:
            return q, np.zeros(len(q))
        found = self.sudden.features(readings)
        hours = readings["Hours"].to_numpy(dtype=float)[MIN_READINGS - 1:]
        out, chance = q.copy(), np.zeros(len(q))
        for index, age in enumerate(hours):
            if age in found:
                location, scale = self.sudden.location_scale(found[age][None, :])
                out[index], chance[index] = combine(q[index], (float(location[0]), scale))
        return out, chance

    def assess(self, readings: pd.DataFrame, measured=None, limits=None) -> pd.DataFrame:
        table = super().assess(readings, measured, limits)
        if self.sudden is None or len(readings) < MIN_READINGS:
            return table
        readings, measured = self._prepare(readings, measured)
        q, chance = self.combined(readings, measured, limits)
        key = six_rul.subset_key(measured)
        pad = np.where(self.late(readings, measured, limits), self.near_adjustments[key], self.adjustments[key])
        rows = table.index[MIN_READINGS - 1:]
        live = ~table.loc[rows, "End_of_life_reached"].to_numpy(dtype=bool)
        table.loc[rows[live], "RUL_low_h"] = six_rul._to_hours(q[live, 0] - pad[live])
        table.loc[rows[live], "RUL_median_h"] = six_rul._to_hours(q[live, 1])
        table.loc[rows[live], "RUL_high_h"] = six_rul._to_hours(q[live, 2] + pad[live])
        table.loc[rows, "Sudden_failure_chance"] = chance
        return table


def calibrate(model: TrainedModel, histories: dict, rows: pd.DataFrame, measured, limits) -> tuple | None:
    """Split-conformal widening (log space) of the 80% range, (early, late) in life; None when it cannot be finite.

    Every reading of every calibration capacitor is scored, conservatively for failures known only to lie
    between two checks, and each capacitor counts equally. Within each stage the widening is the weighted
    quantile at 80% x (1 + 1/capacitors), the usual finite-sample conformal level. Using every reading instead
    of one per capacitor keeps a single unlucky reading from setting the whole range.
    """
    found = []
    for device, group in rows.groupby("device", sort=False):
        history = histories[device]
        q, _ = model.combined(history, measured, limits)
        late = model.late(history, measured, limits)
        ages = history["Hours"].to_numpy(dtype=float)[MIN_READINGS - 1:]
        for row in group.itertuples(index=False):
            index = int(np.flatnonzero(np.isclose(ages, row.age_h))[0])
            low = np.log(row.remaining_lower_h + RUL_OFFSET_H)
            high = np.log(row.remaining_upper_h + RUL_OFFSET_H) if np.isfinite(row.remaining_upper_h) else np.inf
            found.append((device, bool(late[index]), max(q[index, 0] - low, high - q[index, 2])))
    scores = pd.DataFrame(found, columns=["device", "late", "score"])

    def widening(part: pd.DataFrame) -> float | None:
        if part.empty:
            return None
        devices = part["device"].nunique()
        level = min(1.0, INTERVAL_MASS * (1 + 1 / devices))
        weights = 1 / part.groupby("device")["device"].transform("size").to_numpy(dtype=float)
        order = np.argsort(part["score"].to_numpy())
        values, cumulative = part["score"].to_numpy()[order], np.cumsum(weights[order]) / weights.sum()
        value = float(values[min(np.searchsorted(cumulative, level - 1e-12), len(values) - 1)])
        return max(value, 0.0) if np.isfinite(value) else None

    early, late = widening(scores.loc[~scores["late"]]), widening(scores.loc[scores["late"]])
    if early is None and late is None:
        return None
    if (early is None and not scores.loc[~scores["late"]].empty) or (late is None and not scores.loc[scores["late"]].empty):
        return None  # a stage whose range cannot be finite (still-working capacitors decide it)
    return (early if early is not None else late, late if late is not None else early)


def train(histories: dict, labels: dict, modes: dict, training: list, signals: tuple[str, ...],
          conditions: tuple[str, ...], limits: dict, seed: int = 7) -> TrainedModel:
    """The trained model from the training capacitors, before the lab's calibration."""
    base = built_in()
    rng = np.random.default_rng(seed)
    examples = Examples()
    for device in training:
        add_capacitor(examples, histories[device], labels[device], modes[device], signals, limits, rng)
    if not examples.x:
        raise ValueError("No training capacitor has readings before its end of life to learn from.")
    net, path_net = fine_tune(base, examples.arrays(), replay_data(), seed)
    wear_ends = {device: wear_out_target(histories[device], labels[device], modes[device], signals, limits)[0]
                 for device in training}
    training_labels = pd.DataFrame([labels[device] for device in training])
    sudden = fit_sudden(histories, training_labels, {d: modes[d] for d in training}, wear_ends, signals, conditions)
    return TrainedModel(base, net, path_net, 0.0, sudden)


# --- Saving and loading -------------------------------------------------------------------------

# Bump when the model inputs or the package layout change; older packages then ask to be trained again.
MODEL_FORMAT = 1


class ModelFormatError(ValueError):
    """A saved model this version of CapLife cannot use (made by an older or newer version)."""


def _tensor_bytes(state: dict) -> bytes:
    buffer = BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _tensors(content: bytes) -> dict:
    # weights_only: a saved model can only hold numbers, never code.
    return torch.load(BytesIO(content), map_location="cpu", weights_only=True)


def _inputs_signature() -> dict:
    from caplife.rul_workflow import FEATURE_LABELS
    return {"step_features": list(six_rul.STEP_FEATURES), "window": six_rul.WINDOW,
            "duration_features": sorted(FEATURE_LABELS)}


def package(model: TrainedModel, info: dict) -> dict[str, bytes]:
    """The trained model as files for the user's account: network weights and a JSON description."""
    sudden = None
    files = {"model.pt": _tensor_bytes(model.net.state_dict()),
             "trajectory.pt": _tensor_bytes(model.path_net.state_dict())}
    if model.sudden is not None:
        duration = model.sudden.duration
        files["sudden.pt"] = _tensor_bytes(duration.network.state_dict())
        sudden = {"signals": list(model.sudden.signals), "conditions": list(model.sudden.conditions),
                  "hidden": isinstance(duration.network, torch.nn.Sequential),
                  "center": duration.center.tolist(), "spread": duration.spread.tolist(), "scale": duration.scale,
                  "lows": model.sudden.lows.tolist(), "highs": model.sudden.highs.tolist()}
    description = {
        "format": MODEL_FORMAT, "inputs": _inputs_signature(), "hidden": model.metadata.get("hidden", six_rul.HIDDEN),
        "adjustment": next(iter(model.adjustments.values())),
        "late_adjustment": next(iter(model.near_adjustments.values())), "late_from": model.near_limit,
        "path_adjust": np.asarray(model.path_adjust).tolist(),
        "sudden": sudden, "info": info,
    }
    files["model.json"] = json.dumps(description, allow_nan=False).encode("utf-8")
    return files


def describe(files: dict[str, bytes]) -> dict:
    """The saved model's description (its ``info``) and whether this version of CapLife can use it."""
    description = json.loads(files["model.json"])
    usable = description.get("format") == MODEL_FORMAT and description.get("inputs") == _inputs_signature()
    return {**description.get("info", {}), "usable": usable}


def load(files: dict[str, bytes]) -> TrainedModel:
    """A saved model, ready for the pages; ``ModelFormatError`` when this version cannot use it."""
    description = json.loads(files["model.json"])
    if description.get("format") != MODEL_FORMAT or description.get("inputs") != _inputs_signature():
        raise ModelFormatError("This model was saved by a different version of CapLife. Train it again on "
                               "Train The AI to use it.")
    base = built_in()
    net, path_net = six_rul.SixRULNet(description["hidden"]), six_rul.TrajectoryNet(description["hidden"])
    net.load_state_dict(_tensors(files["model.pt"]))
    path_net.load_state_dict(_tensors(files["trajectory.pt"]))
    net.eval()
    path_net.eval()
    sudden = None
    if description["sudden"] is not None:
        part = description["sudden"]
        width = len(part["center"])
        network = (torch.nn.Sequential(torch.nn.Linear(width, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
                   if part["hidden"] else torch.nn.Linear(width, 1)).double()
        network.load_state_dict(_tensors(files["sudden.pt"]))
        network.eval()
        duration = DurationModel(np.asarray(part["center"]), np.asarray(part["spread"]), network, part["scale"])
        sudden = SuddenModel(duration, tuple(part["signals"]), tuple(part["conditions"]),
                             np.asarray(part["lows"]), np.asarray(part["highs"]))
    # Models saved before early/late calibration have one widening for the whole life.
    model = TrainedModel(base, net, path_net, description["adjustment"], sudden,
                         np.asarray(description["path_adjust"]), description.get("late_adjustment"))
    return model
