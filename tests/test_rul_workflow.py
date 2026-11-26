import io
import json

import pandas as pd
import pytest

from caplife.real_data import RealDataValidationError
from caplife.rul_example import example_dataset
from caplife.rul_workflow import (
    SPLIT_BY_DEVICE, SPLIT_BY_EXPERIMENT, SPLIT_ROTATE, device_roles, outcome_template_csv, read_outcomes,
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
    models = passing["model_table"].set_index("Model")
    assert list(models.index) == ["Your trained model", "Built-in model", "Average lifetime only"]
    # Training on the example's capacitors beats both the built-in model and the average lifetime.
    assert models.loc["Your trained model", "Average miss (h)"] < models.loc["Built-in model", "Average miss (h)"]
    assert models.loc["Your trained model", "Average miss (h)"] < models.loc["Average lifetime only", "Average miss (h)"]
    assert passing["failure_modes"] == {"limit": 24, "running": 7} and passing["model"].sudden is None
    assert passing["roles"] == {"training": 16, "calibration": 7, "test": 8}
    # SYN_EXP_C_2 has no documented follow-up after its (outcome-blind) evaluation reading.
    assert list(passing["audit"]["devices_excluded"]) == ["SYN_EXP_C_2"]
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
    assert passing["report"]["calibration_log_adjustment"] == changed["report"]["calibration_log_adjustment"]
    # Every test failure now falls far outside the ranges, so the forecasts are withheld.
    assert not changed["passed"] and changed["forecasts"] is None
    assert any("80% range contained" in reason for reason in changed["reasons"])


def test_forecasts_use_the_latest_reading_and_each_point_only_earlier_readings(example, passing):
    measurements, outcomes = example
    forecast = passing["forecasts"].set_index("Capacitor").loc["SYN_RUNNING_2"]
    latest = measurements.loc[measurements["Cap_ID"].eq("SYN_RUNNING_2"), "Hours"].max()
    assert forecast["Forecast made at (h)"] == latest
    edited = measurements.copy()
    later = edited["Cap_ID"].eq("SYN_RUNNING_2") & edited["Hours"].gt(100)
    edited.loc[later, "Capacitance_uF"] *= 0.9
    changed = run_rul_workflow(edited, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    history = lambda result: result["forecast_history"].query("Capacitor == 'SYN_RUNNING_2' and Hours <= 100")
    pd.testing.assert_frame_equal(history(passing), history(changed))  # earlier points are unaffected
    assert not changed["forecasts"].set_index("Capacitor").loc["SYN_RUNNING_2"].equals(forecast)


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
    two_each = measurements[["Cap_ID", "Hours", "Capacitance_uF"]].copy()
    two_each.loc[two_each.groupby("Cap_ID").cumcount() >= 2, "Capacitance_uF"] = float("nan")
    with pytest.raises(RealDataValidationError, match="No parameter has at least 3 readings"):
        run_rul_workflow(two_each, outcomes, SPLIT_BY_DEVICE)
    stray = pd.concat([outcomes, outcomes.iloc[[0]].assign(Cap_ID="NOT_MEASURED")], ignore_index=True)
    with pytest.raises(RealDataValidationError, match="no measurements: NOT_MEASURED"):
        run_rul_workflow(measurements, stray, SPLIT_BY_DEVICE)
    with pytest.raises(ValueError, match="different experiments"):
        run_rul_workflow(measurements, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_A", "SYN_EXP_A")


def test_uncalibrated_split_withholds_forecasts_and_explains_why(example):
    result = run_rul_workflow(*example, SPLIT_BY_EXPERIMENT, "SYN_EXP_D", "SYN_EXP_A")
    assert not result["passed"] and result["forecasts"] is None
    assert "still working at their last check" in result["reasons"][0]
    assert set(result["model_table"]["Range"]) == {"Not calibrated", "Built-in calibration"}


def test_outcome_template_round_trips_blank_failure_ages():
    assert read_outcomes(outcome_template_csv().encode()).columns.tolist() == [*OUTCOME_COLUMNS, "Failure_mode"]
    frame = read_outcomes(b"Cap_ID,Experiment_ID,Endpoint,Last_alive_h,Failure_by_h\nNA,e1,open,90,\n")
    assert frame.loc[0, "Cap_ID"] == "NA" and frame.loc[0, "Failure_by_h"] == ""


def test_fewer_parameters_train_on_the_ones_every_capacitor_has(example):
    measurements, outcomes = example
    three = measurements[["Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm", "Leakage_uA"]]
    result = run_rul_workflow(three, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    assert result["parameters"] == ("Capacitance_uF", "ESR_ohm", "Leakage_uA")
    assert result["training_landmarks"] > result["roles"]["training"]  # every reading from the third on
    assert len(result["test_table"]) == result["roles"]["test"]  # one evaluation reading per test capacitor


def test_recorded_operating_conditions_join_the_features(example):
    import numpy as np
    measurements, outcomes = example
    # Every capacitor records the same rated values and frequency; ambient differs by experiment.
    ambient = measurements["Cap_ID"].str.extract(r"_(\w)_")[0].map({"A": 70.0, "B": 85.0, "C": 95.0}).fillna(80.0)
    with_conditions = measurements.assign(Ambient_C=ambient, Voltage_V=20.0, Rated_V=25.0, Ripple_Hz=10_000.0)
    result = run_rul_workflow(with_conditions, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    assert result["operating_conditions"] == ("ambient_temperature", "voltage_ratio", "frequency")
    assert result["report"]["operating_conditions"] == ["ambient_temperature", "voltage_ratio", "frequency"]
    # A capacitor to forecast that lacks a used condition is withheld with the reason.
    running = with_conditions["Cap_ID"].isin(set(measurements["Cap_ID"]) - set(outcomes["Cap_ID"]))
    first = with_conditions.loc[running, "Cap_ID"].iloc[0]
    missing = with_conditions.copy()
    missing.loc[missing["Cap_ID"].eq(first), "Ambient_C"] = np.nan
    again = run_rul_workflow(missing, outcomes, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    if again["passed"]:
        assert "ambient temperature" in again["forecasts_withheld"][first]


def test_stated_sudden_failures_add_a_sudden_failure_part(example):
    measurements, outcomes = example
    stated = outcomes.assign(Failure_mode="")
    # Five training capacitors that failed are recorded as sudden failures (an open circuit, say).
    failed = stated["Failure_by_h"].ne("") & stated["Experiment_ID"].isin(["SYN_EXP_A", "SYN_EXP_D"])
    stated.loc[stated.index[failed][:5], "Failure_mode"] = "Sudden"
    result = run_rul_workflow(measurements, stated, SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    assert result["failure_modes"]["sudden"] == 5
    assert result["model"].sudden is not None
    history = measurements.loc[measurements["Cap_ID"].eq("SYN_RUNNING_2")].reset_index(drop=True)
    last = result["model"].assess(history, result["parameters"]).iloc[-1]
    assert 0 < last["Sudden_failure_chance"] < 1
    with pytest.raises(ValueError, match="Failure_mode must be limit, sudden or blank"):
        run_rul_workflow(measurements, stated.assign(Failure_mode="melted"), SPLIT_BY_DEVICE)


def test_the_trained_model_answers_for_other_limits(passing, example):
    measurements, _ = example
    history = measurements.loc[measurements["Cap_ID"].eq("SYN_RUNNING_2")].reset_index(drop=True)
    remaining = [passing["model"].assess(history, passing["parameters"], {"capacitance_loss_pct": loss})
                 .iloc[-1]["RUL_median_h"] for loss in (15.0, 20.0, 30.0)]
    assert remaining[0] < remaining[1] < remaining[2]  # a looser limit leaves more life
    path = passing["model"].forecast(history, remaining[1], passing["parameters"])
    assert set(path["signal"]) == set(passing["parameters"])


def test_a_saved_model_loads_back_identical_and_refuses_other_versions(passing, example):
    from caplife import fine_tune
    measurements, _ = example
    files = fine_tune.package(passing["model"], {"trained_on": "Simulated example"})
    loaded = fine_tune.load(files)
    history = measurements.loc[measurements["Cap_ID"].eq("SYN_RUNNING_2")].reset_index(drop=True)
    pd.testing.assert_frame_equal(passing["model"].assess(history, passing["parameters"]),
                                  loaded.assess(history, passing["parameters"]))
    assert fine_tune.describe(files) == {"trained_on": "Simulated example", "usable": True}
    older = {**files, "model.json": files["model.json"].replace(b'"format": 1', b'"format": 0')}
    assert not fine_tune.describe(older)["usable"]
    with pytest.raises(fine_tune.ModelFormatError, match="Train it again"):
        fine_tune.load(older)


def test_the_range_is_calibrated_separately_early_and_late_in_life(passing):
    from caplife import fine_tune
    widening = passing["report"]["calibration_log_adjustment"]
    assert set(widening) == {"early", "late"}
    model = passing["model"]
    assert set(model.adjustments.values()) == {widening["early"]}
    assert set(model.near_adjustments.values()) == {widening["late"]}
    assert model.near_limit == fine_tune.LATE_STAGE
    # A model saved before early/late calibration had one widening; it still loads, with it throughout.
    files = fine_tune.package(model, {})
    description = json.loads(files["model.json"])
    for key in ("late_adjustment", "late_from"):
        description.pop(key)
    older = fine_tune.load({**files, "model.json": json.dumps(description).encode()})
    assert set(older.near_adjustments.values()) == {widening["early"]}



def test_every_experiment_in_turn_pools_the_tests_and_forecasts_from_a_final_model(example):
    measurements, outcomes = example
    steps = []
    result = run_rul_workflow(measurements, outcomes, SPLIT_ROTATE, "SYN_EXP_A",
                              progress=lambda share, text: steps.append(share))
    turns = result["turns"]
    assert turns["Test experiment"].tolist() == ["SYN_EXP_B", "SYN_EXP_C", "SYN_EXP_D"]
    # Every eligible capacitor outside the calibration experiment is a test capacitor once.
    assert result["roles"]["test"] == int(turns["Test capacitors"].sum()) == len(result["test_table"])
    assert set(result["test_table"]["Experiment"]) == {"SYN_EXP_B", "SYN_EXP_C", "SYN_EXP_D"}
    # The final model trains on every experiment except the calibration one.
    assert result["experiments_by_role"]["training"] == ["SYN_EXP_B", "SYN_EXP_C", "SYN_EXP_D"]
    assert steps[0] == 0 and steps[-1] == 1 and steps == sorted(steps)
    json.dumps(result["report"], allow_nan=False)
    with pytest.raises(ValueError, match="calibration experiment"):
        run_rul_workflow(measurements, outcomes, SPLIT_ROTATE, "NOT_AN_EXPERIMENT")


def test_new_readings_and_outcomes_are_added_to_a_models_training_data():
    from caplife.retrain import merge_measurements, merge_outcomes
    old = b"Cap_ID,Hours,Capacitance_uF\nA,0,100\nA,10,99\n"
    new = b"Cap,Time (h),Capacitance\nA,20,98\nB,0,100\nA,10,99.5\n"
    merged, note = merge_measurements(old, new)
    frame = pd.read_csv(io.BytesIO(merged))
    assert frame[["Cap_ID", "Hours"]].values.tolist() == [["A", 0], ["A", 10], ["A", 20], ["B", 0]]
    assert frame.loc[1, "Capacitance_uF"] == 99.5  # the newer reading wins
    assert note == "Added 2 readings, including 1 new capacitor."
    header = "Cap_ID,Experiment_ID,Endpoint,Last_alive_h,Failure_by_h\n"
    merged, note = merge_outcomes((header + "A,E1,x,10,\n").encode(), (header + "A,E1,x,20,30\nB,E2,x,10,\n").encode())
    frame = pd.read_csv(io.BytesIO(merged), dtype=str, keep_default_na=False)
    assert frame.values.tolist() == [["A", "E1", "x", "20", "30"], ["B", "E2", "x", "10", ""]]
    assert note == "Outcomes: 1 new outcome, 1 outcome updated."
