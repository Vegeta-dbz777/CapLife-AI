import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import lognorm

from caplife.physical_failure_validation import FEATURES, landmark_examples
from caplife.survival import event_log_likelihood, fit_duration


def test_neural_time_likelihood_preserves_interval_event_and_censored_samples():
    location = torch.tensor([2.0] * 5, dtype=torch.double, requires_grad=True)
    sigma = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    lower = torch.tensor([0, 3, 3, 60, 5], dtype=torch.double)
    upper = torch.tensor([4, 3, np.inf, 61, 9], dtype=torch.double)
    actual = event_log_likelihood(location, sigma, lower, upper)
    law = lognorm(s=0.7, scale=np.exp(2))
    expected = [law.logcdf(4), law.logpdf(3), law.logsf(3),
                np.log(law.sf(60) - law.sf(61)), np.log(law.cdf(9) - law.cdf(5))]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-10)
    actual.sum().backward()
    assert torch.isfinite(location.grad).all() and torch.isfinite(sigma.grad)


def test_six_covariate_training_and_distribution_are_numerically_valid():
    rng = np.random.default_rng(9)
    sensors = rng.normal(size=(30, 6))
    times = np.exp(4 + sensors[:, 0] / 3)
    weights = np.ones(30)
    upper = times + 1
    upper[::5] = np.inf
    engine = fit_duration(sensors, times - 1, upper, weights)
    bands = engine.quantiles(sensors, (0.1, 0.5, 0.9))
    assert np.isfinite(bands).all() and (np.diff(bands, axis=1) > 0).all()
    assert (engine.probability_by(sensors, 50) <= engine.probability_by(sensors, 100)).all()
    with pytest.raises(ValueError, match="at least one observed event"):
        fit_duration(sensors, times - 1, np.full(30, np.inf), weights)


def test_physical_failure_features_use_only_readings_available_at_the_landmark():
    markers = pd.DataFrame({
        "device_id": ["a"] * 4, "case": [1] * 4, "test_age_h": [0, 10, 20, 30],
        "capacitance_pct_of_initial": [100, 98, 96, 90],
        "esr_pct_of_initial": [100, 105, 110, 150],
    })
    endpoint = pd.DataFrame({
        "device_id": ["a"], "case": [1], "event_time_precision": ["annotated_terminal_hour"],
        "failure_age_lower_h": [50], "failure_age_upper_h": [50],
    })
    original = landmark_examples(markers, endpoint)
    assert original.loc[0, "origin_h"] == 20
    assert original.loc[0, "remaining_lower_h"] == 29.5
    assert original.loc[0, "remaining_upper_h"] == 30.5
    np.testing.assert_allclose(original.loc[:, list(FEATURES)].to_numpy(), [[4, 10, .2, .5]])
    markers.loc[3, "esr_pct_of_initial"] = 500
    pd.testing.assert_frame_equal(original, landmark_examples(markers, endpoint))
    assert landmark_examples(markers.iloc[1:], endpoint).empty
