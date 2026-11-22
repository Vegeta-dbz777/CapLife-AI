import json

import pandas as pd
import pytest

from caplife.real_data import RealDataValidationError
from caplife.rul_example import example_dataset
from caplife.rul_workflow import (
    SPLIT_BY_DEVICE, SPLIT_BY_EXPERIMENT, device_roles, outcome_template_csv, read_outcomes,
    run_rul_workflow,
)
from caplife.six_indicator_validation import OUTCOME_COLUMNS

PREDICTED = ["Predicted failure age (h)", "80% range from (h)", "80% range to (h)"]


@pytest.fixture(scope="module")
def example():
    return example_dataset()


@pytest.fixture(scope="module")
def passing(example):
    return run_rul_workflow(*example, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")


def test_passing_model_forecasts_only_capacitors_without_outcomes(example, passing):
    measurements, outcomes = example
    assert passing["passed"] and passing["reasons"] == []
    assert passing["roles"] == {"training": 16, "calibration": 8, "test": 8}
    forecast = set(passing["forecasts"]["Capacitor"]) | set(passing["forecasts_withheld"])
    assert forecast == set(measurements["Cap_ID"]) - set(outcomes["Cap_ID"])
    table = passing["forecasts"]
    assert (table["80% range from (h)"] <= table["Predicted failure age (h)"]).all()
    assert (table["Predicted failure age (h)"] <= table["80% range to (h)"]).all()
    json.dumps(passing["report"], allow_nan=False)


def test_test_outcomes_cannot_change_the_model_or_forecasts(example, passing):
    measurements, outcomes = example
    edited = outcomes.copy()
    test_rows = edited["Experiment_ID"].eq("SYN_EXP_B")
    edited.loc[test_rows, ["Last_alive_h", "Failure_by_h"]] = [5000.0, "5050"]
    changed = run_rul_workflow(measurements, edited, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    pd.testing.assert_frame_equal(passing["test_table"][PREDICTED], changed["test_table"][PREDICTED])
    for model, scores in passing["report"]["models"].items():
        assert scores["calibration_log_radius"] == changed["report"]["models"][model]["calibration_log_radius"]
    # Every test failure now falls far outside the ranges, so the forecasts are withheld.
    assert not changed["passed"] and changed["forecasts"] is None
    assert any("80% range contained" in reason for reason in changed["reasons"])


def test_later_readings_do_not_change_a_forecast(example, passing):
    measurements, outcomes = example
    edited = measurements.copy()
    later = edited["Cap_ID"].eq("SYN_RUNNING_2") & edited["Hours"].gt(100)
    edited.loc[later, "Capacitance_uF"] *= 0.9
    changed = run_rul_workflow(edited, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    row = lambda result: result["forecasts"].set_index("Capacitor").loc["SYN_RUNNING_2", PREDICTED]
    pd.testing.assert_series_equal(row(passing), row(changed))


def test_device_split_is_fixed_by_id_not_by_row_order_or_outcome():
    frame = pd.DataFrame({"device": [f"cap_{i}" for i in range(20)], "experiment": "one"})
    roles = device_roles(frame, SPLIT_BY_DEVICE)
    shuffled = device_roles(frame.sample(frac=1, random_state=3), SPLIT_BY_DEVICE)
    assert roles.to_dict() == shuffled.to_dict()
    assert roles.value_counts().to_dict() == {"training": 12, "calibration": 4, "test": 4}
    with pytest.raises(ValueError, match="at least 13 eligible"):
        device_roles(frame.iloc[:12], SPLIT_BY_DEVICE)


def test_inputs_are_checked_before_training(example):
    measurements, outcomes = example
    with pytest.raises(RealDataValidationError, match="Missing: Internal pressure"):
        run_rul_workflow(measurements.drop(columns="Pressure_Abs_kPa"), outcomes, SPLIT_BY_DEVICE)
    stray = pd.concat([outcomes, outcomes.iloc[[0]].assign(Cap_ID="NOT_MEASURED")], ignore_index=True)
    with pytest.raises(RealDataValidationError, match="no measurements: NOT_MEASURED"):
        run_rul_workflow(measurements, stray, SPLIT_BY_DEVICE)
    with pytest.raises(ValueError, match="different experiments"):
        run_rul_workflow(measurements, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_A", "SYN_EXP_A")


def test_uncalibrated_split_withholds_forecasts_and_explains_why(example):
    result = run_rul_workflow(*example, SPLIT_BY_EXPERIMENT, "SYN_EXP_D", "SYN_EXP_A")
    assert not result["passed"] and result["forecasts"] is None
    assert "still working at their last check" in result["reasons"][0]
    assert set(result["model_table"]["Range"]) == {"Model's own (uncalibrated)"}


def test_outcome_template_round_trips_blank_failure_ages():
    assert read_outcomes(outcome_template_csv().encode()).columns.tolist() == list(OUTCOME_COLUMNS)
    frame = read_outcomes(b"Cap_ID,Experiment_ID,Endpoint,Last_alive_h,Failure_by_h\nNA,e1,open,90,\n")
    assert frame.loc[0, "Cap_ID"] == "NA" and frame.loc[0, "Failure_by_h"] == ""
