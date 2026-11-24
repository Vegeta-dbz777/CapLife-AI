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
