from pathlib import Path

import numpy as np
import pandas as pd

from caplife import operating_conditions as conditions
from caplife.file_reader import _LOOKUP, _key
from caplife.six_rul import COND, COND_KNOWN, K, SixRULModel, condition_values, hide_indicators, sequences
from caplife.six_sim import CONDITION_COLUMNS, SIGNALS, simulate_fleet

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"


def _history(**conditions_by_column):
    frame = pd.DataFrame({"Cap_ID": "A", "Hours": [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
                          "Capacitance_uF": [1000, 990, 978, 965, 950], "ESR_ohm": [0.08, 0.083, 0.087, 0.092, 0.098],
                          "Leakage_uA": [8, 8.6, 9.4, 10.5, 11.9]})
    for column, values in conditions_by_column.items():
        frame[column] = values
    return frame


def test_headers_for_operating_conditions_are_recognised():
    for header, column in (("Operating frequency (Hz)", "Ripple_Hz"), ("Rated ripple current (A)", "Rated_Ripple_A"),
                           ("Applied voltage (V)", "Voltage_V"), ("Rated voltage", "Rated_V"),
                           ("Ripple current (A)", "Ripple_A"), ("Core temperature (°C)", "Core_C")):
        assert _LOOKUP[_key(header)] == column


def test_simulated_readings_record_conditions_and_some_change():
    measurements, meta = simulate_fleet(40, seed=11)
    assert set(CONDITION_COLUMNS) <= set(measurements.columns)
    assert (measurements["Voltage_V"] <= measurements["Rated_V"] + 1e-9).all()
    changed = meta["change_h"] < meta["eol_h"]
    assert 0 < changed.mean() < 1
    device = meta.loc[changed, "Cap_ID"].iloc[0]
    assert measurements.loc[measurements["Cap_ID"] == device, "Ambient_C"].round().nunique() > 1


def test_condition_features_are_ratios_carried_forward_and_flagged():
    frame = _history(Voltage_V=[40, np.nan, 40, 40, 45], Rated_V=50, Ripple_A=1.0, Rated_Ripple_A=2.0,
                     Ripple_Hz=10_000, Ambient_C=85)
    values = condition_values(frame)
    assert values["voltage_ratio"].tolist() == [0.8, 0.8, 0.8, 0.8, 0.9]  # the gap carries forward
    assert values["ripple_ratio"].eq(0.5).all() and values["core_temperature"].isna().all()
    x = sequences(frame, ("Capacitance_uF", "ESR_ohm", "Leakage_uA"))
    known = x[-1, -1, COND_KNOWN]
    assert known.tolist() == [0, 1, 1, 1, 1]  # no core temperature recorded
    hidden = hide_indicators(x, np.ones((len(x), len(SIGNALS)), bool), np.zeros((len(x), K), bool))
    assert not hidden[:, :, COND].any() and not hidden[:, :, COND_KNOWN].any()


def test_planned_conditions_replace_only_the_newest_reading():
    frame = _history(Ambient_C=70.0, Core_C=80.0, Ripple_A=1.0, Rated_Ripple_A=1.0, Ripple_Hz=120.0)
    planned = conditions.with_planned(frame, {"Ambient_C": 90.0})
    assert planned["Ambient_C"].tolist()[:-1] == frame["Ambient_C"].tolist()[:-1]
    assert planned["Ambient_C"].iloc[-1] == 90.0 and planned["Core_C"].iloc[-1] == 100.0  # core follows ambient
    halved = conditions.with_planned(frame, {"Ripple_A": 0.5})
    assert halved["Core_C"].iloc[-1] == 80.0 - 12 * (1 - 0.25)  # less ripple heating


def test_stress_notes_judge_voltage_ripple_and_temperature():
    notes = conditions.stress_notes(pd.Series({"Voltage_V": 55.0, "Rated_V": 50.0, "Ripple_A": 0.5,
                                               "Rated_Ripple_A": 1.0, "Ripple_Hz": 120.0, "Core_C": 112.0}))
    levels = [level for level, _ in notes]
    assert levels[0] == "error" and "above rating" in notes[0][1]
    assert levels[1] == "info"
    assert levels[2] == "error" and "112 °C" in notes[2][1]


def test_model_predicts_shorter_life_under_harsher_planned_conditions():
    model = SixRULModel(ARTIFACT)
    measurements, _ = simulate_fleet(6, seed=31)
    shorter = 0
    for _, readings in measurements.groupby("Cap_ID"):
        readings = readings.iloc[:8].reset_index(drop=True)
        base = model.assess(readings).iloc[-1]["RUL_median_h"]
        recorded = conditions.latest(readings)
        hot = conditions.with_planned(readings, {"Ambient_C": recorded["Ambient_C"] + 20,
                                                 "Voltage_V": recorded["Rated_V"]})
        shorter += model.assess(hot).iloc[-1]["RUL_median_h"] < base
    assert shorter >= 5


def test_aging_rule_factor_follows_ten_degree_and_voltage_rules():
    recorded = pd.Series({"Ambient_C": 80.0, "Core_C": 90.0, "Voltage_V": 40.0, "Rated_V": 50.0})
    assert np.isclose(conditions.aging_rule_factor(recorded, {"Ambient_C": 70.0}), 2.0)  # 10 °C cooler: twice
    assert np.isclose(conditions.aging_rule_factor(recorded, {"Voltage_V": 20.0}), 8.0)  # half the voltage: 2³
    assert conditions.aging_rule_factor(pd.Series({"Voltage_V": 40.0, "Rated_V": 50.0}), {"Voltage_V": 20.0}) is None


def test_changed_conditions_reach_the_assessment_of_that_capacitor_only():
    import json
    from caplife import six_rul_page as page
    measurements, _ = simulate_fleet(3, seed=31)
    measurements = measurements.groupby("Cap_ID").head(8)
    first, second = sorted(measurements["Cap_ID"].unique())[:2]
    recorded = conditions.latest(measurements.loc[measurements["Cap_ID"] == first])
    planned = {first: {"Ambient_C": float(recorded["Ambient_C"]) + 20}}
    base = page._assess_cached(measurements, json.dumps({}), json.dumps({}), "t")
    hot = page._assess_cached(measurements, json.dumps({}), json.dumps(planned), "t")
    assert hot[first][0]["Ambient_C"].iloc[-1] == planned[first]["Ambient_C"]
    assert hot[first][1]["RUL_median_h"].iloc[-1] < base[first][1]["RUL_median_h"].iloc[-1]
    assert hot[first][1]["RUL_median_h"].iloc[:-1].equals(base[first][1]["RUL_median_h"].iloc[:-1])  # history unchanged
    assert hot[second][1].equals(base[second][1])


def test_judgements_have_short_labels_for_badges():
    found = conditions.judgements(pd.Series({"Voltage_V": 24.5, "Rated_V": 25.0, "Core_C": 70.0}))
    assert [(level, short) for level, short, _ in found] == [("warning", "Voltage 98% of rated"), ("info", "Core 70 °C")]


def test_alternative_history_matches_the_simulator_had_it_always_run_cooler():
    import caplife.six_sim as sim
    rng = np.random.default_rng(3)
    device = sim._device(rng)
    while np.isfinite(device["change_h"]):
        device = sim._device(rng)
    hours = np.linspace(0, 0.8 * device["tau_h"], 12)
    state = sim._state(device, hours)
    conditions_now = sim.conditions_at(device, hours)
    readings = pd.DataFrame({"Cap_ID": "A", "Hours": hours, **state,
                             **{column: conditions_now[column] for column in CONDITION_COLUMNS}})
    planned = {"Ambient_C": device["ambient_C"] - 15, "Voltage_V": 0.6 * device["rated_V"],
               "Ripple_A": 0.5 * device["rated_ripple_A"]}
    estimate = conditions.alternative_history(readings, planned)
    # The truth: the same capacitor run at the planned conditions from the start.
    ratio = (planned["Voltage_V"] / device["rated_V"], planned["Ripple_A"] / device["rated_ripple_A"])
    core, acceleration = (float(v) for v in sim.operating_stress(planned["Ambient_C"], *ratio, device["ripple_Hz"]))
    tau_rated = device["tau_h"] * float(sim.operating_stress(device["ambient_C"], device["voltage_ratio"],
                                                             device["ripple_ratio"], device["ripple_Hz"])[1])
    other = {**device, "ambient_C": planned["Ambient_C"], "voltage_ratio": ratio[0], "ripple_ratio": ratio[1],
             "core_C": core, "ambient2_C": planned["Ambient_C"], "voltage2_ratio": ratio[0], "ripple2_ratio": ratio[1],
             "core2_C": core, "tau_h": tau_rated / acceleration, "tau2_h": tau_rated / acceleration}
    truth = sim._state(other, estimate["Hours"].to_numpy())
    assert estimate["Hours"].iloc[-1] > hours[-1]  # milder conditions: the same aging takes longer
    for signal in SIGNALS:
        assert np.allclose(estimate[signal], truth[signal], rtol=2e-3, atol=1e-6), signal
    # Without any recorded temperature there is nothing to scale from.
    assert conditions.alternative_history(readings.drop(columns=["Core_C", "Ambient_C"]), planned) is None
