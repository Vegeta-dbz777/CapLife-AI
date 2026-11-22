"""Fit log-normal remaining-time distributions to observed event bounds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch import nn


def event_log_likelihood(location: torch.Tensor, scale: torch.Tensor,
                         lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Return log-density for exact times and log-probability for censoring."""
    exact = upper.eq(lower)
    interval = upper.isfinite() & ~exact
    right_censored = ~upper.isfinite()
    result = torch.zeros_like(location)

    if exact.any():
        log_time = lower[exact].log()
        z = (log_time - location[exact]) / scale
        result[exact] = -0.5 * (z.square() + np.log(2 * np.pi)) - scale.log() - log_time
    if right_censored.any():
        known = right_censored & lower.gt(0)
        result[known] = torch.special.log_ndtr((location[known] - lower[known].log()) / scale)
    if interval.any():
        z_hi = (upper[interval].log() - location[interval]) / scale
        log_mass = torch.special.log_ndtr(z_hi).clone()
        finite_left = lower[interval].gt(0)
        z_lo = (lower[interval][finite_left].log() - location[interval][finite_left]) / scale
        z_end = z_hi[finite_left]
        # Use the survival function in the upper tail to keep the difference resolved.
        use_survival = z_lo >= 0
        log_big = torch.special.log_ndtr(torch.where(use_survival, -z_lo, z_end))
        log_small = torch.special.log_ndtr(torch.where(use_survival, -z_end, z_lo))
        difference = (log_small - log_big).clamp_max(-torch.finfo(z_lo.dtype).eps)
        log_mass[finite_left] = log_big + torch.log(-torch.expm1(difference))
        result[interval] = log_mass
    return result


@dataclass
class DurationModel:
    """Training normalization and a conditional log-normal duration model."""

    center: np.ndarray
    spread: np.ndarray
    network: nn.Module
    scale: float

    def location(self, features: np.ndarray) -> np.ndarray:
        values = torch.from_numpy((np.asarray(features, dtype=float) - self.center) / self.spread)
        with torch.no_grad():
            return self.network(values).reshape(-1).numpy()

    def probability_by(self, features: np.ndarray, hours: float) -> np.ndarray:
        if not np.isfinite(hours) or hours <= 0:
            raise ValueError("A forecast horizon must be positive and finite.")
        return ndtr((np.log(hours) - self.location(features)) / self.scale)

    def quantiles(self, features: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
        return np.exp(self.location(features)[:, None] + self.scale * ndtri(probabilities))


def fit_duration(
    features: np.ndarray, lower: np.ndarray, upper: np.ndarray, weights: np.ndarray,
    hidden: bool = True, seed: int = 7,
) -> DurationModel:
    """Train a regularized model, including samples without an observed failure."""
    features, lower, upper, weights = (np.asarray(values, dtype=float)
                                      for values in (features, lower, upper, weights))
    if (features.ndim != 2 or lower.shape != (len(features),)
            or upper.shape != lower.shape or weights.shape != lower.shape
            or not np.isfinite(features).all() or not np.isfinite(lower).all()
            or (lower < 0).any() or np.isnan(upper).any() or (upper <= 0).any()
            or (upper < lower).any() or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Expected finite covariates, valid observed time bounds and positive weights.")
    observed = np.isfinite(upper)
    if not observed.any():
        raise ValueError("Duration training requires at least one observed event.")
    weights = weights / weights.sum()
    center = np.average(features, axis=0, weights=weights)
    spread = np.sqrt(np.average((features - center) ** 2, axis=0, weights=weights))
    spread[spread == 0] = 1
    x = torch.from_numpy((features - center) / spread)
    low, high, importance = (torch.from_numpy(value) for value in (lower, upper, weights))
    width = features.shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        network = (nn.Sequential(nn.Linear(width, 4), nn.Tanh(), nn.Linear(4, 1))
                   if hidden else nn.Linear(width, 1)).double()

    output = network[-1] if isinstance(network, nn.Sequential) else network
    with torch.no_grad():
        output.bias.fill_(float(np.log(upper[observed]).mean()))
    raw_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.double))

    def scale() -> torch.Tensor:
        # Bound the residual spread so tiny datasets cannot collapse it to zero.
        return 0.05 + nn.functional.softplus(raw_scale)

    optimizer = torch.optim.LBFGS([*network.parameters(), raw_scale], max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        likelihood = event_log_likelihood(network(x).flatten(), scale(), low, high)
        penalty = sum(weight.square().sum() for name, weight in network.named_parameters()
                      if name.endswith("weight"))
        objective = -(importance * likelihood).sum() + 0.025 * penalty
        objective.backward()
        return objective

    optimizer.step(closure)
    with torch.no_grad():
        if not torch.isfinite(network(x)).all() or not torch.isfinite(scale()):
            raise RuntimeError("The remaining-time model did not converge to a finite fit.")
    return DurationModel(center, spread, network.eval(), float(scale().detach()))
