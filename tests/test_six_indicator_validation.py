import numpy as np
import pandas as pd
import pytest

from caplife.remaining_time_validation import BOUNDS
from caplife.six_indicator_validation import (
    FEATURES, SIGNALS, calibrate_log_radius, joined_landmarks, validate_model,
)


def histories_and_outcomes():
    measurements, outcomes = [], []
    for group in ("train", "calibration", "test"):
        for i in range(6):
            device = f"{group}_{i}"
            event = 100 + 12 * i
            for age in (0, 10, 20, 30):
                signals = (100 - age * (i + 1) / 20, .1 + age * (i + 1) / 5000,
                           .5 + age * i / 100, age / 10, age * (i + 1) / 80, 100 + age * i / 70)
                measurements.append({"Cap_ID": device, "Hours": age,
                                     **dict(zip(SIGNALS, signals, strict=True))})
            outcomes.append({
                "Cap_ID": device, "Experiment_ID": group, "Endpoint": "Observed functional failure",
                "Last_alive_h": event - 2, "Failure_by_h": "" if group == "train" and i == 0 else str(event),
            })
    return pd.DataFrame(measurements), pd.DataFrame(outcomes)


def test_train_calibration_and_holdout_are_separate_and_events_can_be_censored():
    measurements, outcomes = histories_and_outcomes()
    frame, _ = joined_landmarks(measurements, outcomes)
    assert frame.loc[frame["device"].eq("train_0"), BOUNDS[1]].item() == np.inf
    edited = measurements.copy()
    edited.loc[edited["Hours"].eq(30), list(SIGNALS)] *= 3
    pd.testing.assert_frame_equal(frame, joined_landmarks(edited, outcomes)[0])

    original = validate_model(measurements, outcomes, "calibration", "test")
    assert original["devices"] == dict(training=6, calibration=6, test=6)
    assert original["features"] == FEATURES and len(FEATURES) == 12
    assert original["models"]["neural"]["test_calibrated_interval"] is not None
    outcomes.loc[outcomes["Experiment_ID"].eq("test"), ["Last_alive_h", "Failure_by_h"]] = [200, "201"]
    changed = validate_model(measurements, outcomes, "calibration", "test")
    for model, score in original["models"].items():
        assert score["calibration_log_radius"] == changed["models"][model]["calibration_log_radius"]


def test_refuses_unpaired_data_and_withholds_unsupported_calibration():
    measurements, outcomes = histories_and_outcomes()
    with pytest.raises(ValueError, match="Missing measured indicators"):
        joined_landmarks(measurements.drop(columns=SIGNALS[-1]), outcomes)
    with pytest.raises(ValueError, match="exactly one outcome"):
        joined_landmarks(measurements, outcomes.iloc[1:])
    bounds = pd.DataFrame({BOUNDS[0]: [10] * 5, BOUNDS[1]: [20, 20, 20, 20, np.inf]})
    assert calibrate_log_radius(bounds, np.log([15] * 5)) is None
    assert calibrate_log_radius(bounds.iloc[:3], np.log([15] * 3)) is None
