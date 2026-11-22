import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caplife.six_rul import MIN_READINGS, SixRULModel, example_fleet, limit_used, sequences
from caplife.six_sim import EOL_LIMITS, SIGNALS, end_of_life_causes, simulate_fleet

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "six_rul"


@pytest.fixture(scope="module")
def model():
    return SixRULModel(ARTIFACTS)


def test_simulator_is_reproducible_and_stops_before_end_of_life():
    first, meta = simulate_fleet(20, seed=5)
    again, _ = simulate_fleet(20, seed=5)
    pd.testing.assert_frame_equal(first, again)
    last_hours = first.groupby("Cap_ID")["Hours"].max()
    assert (last_hours < meta.set_index("Cap_ID")["eol_h"]).all()
    for _, readings in first.groupby("Cap_ID"):
        assert readings["H2_Generated_mL_STP"].is_monotonic_increasing
        assert (readings["Case_Expansion_um"] >= 0).all()
    assert set(meta["eol_cause"].str.split(", ").str[0]) <= {"capacitance", "ESR", "leakage", "pressure"}


def test_end_of_life_rules_match_the_documented_limits():
    base = {"Capacitance_uF": 100, "ESR_ohm": 0.1, "Leakage_uA": 10, "Pressure_Abs_kPa": 101.3}
    assert end_of_life_causes(base, base) == []
    assert end_of_life_causes(base, {**base, "Capacitance_uF": 100 - EOL_LIMITS["capacitance_loss_pct"]}) == ["capacitance"]
    assert end_of_life_causes(base, {**base, "ESR_ohm": 0.2, "Pressure_Abs_kPa": 250}) == ["ESR", "pressure"]


def test_a_prediction_uses_only_readings_up_to_its_age(model):
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_03")].reset_index(drop=True)
    edited = readings.copy()
    edited.loc[8:, "Capacitance_uF"] *= 0.9
    original, changed = model.assess(readings), model.assess(edited)
    columns = ["RUL_low_h", "RUL_median_h", "RUL_high_h"]
    pd.testing.assert_frame_equal(original.loc[:7, columns], changed.loc[:7, columns])
    assert not original.loc[8:, columns].equals(changed.loc[8:, columns])
    assert len(sequences(readings)) == len(readings) - MIN_READINGS + 1


def test_assessment_returns_ordered_ranges_and_bounded_health(model):
    measurements, truth = example_fleet()
    for device, readings in measurements.groupby("Cap_ID"):
        result = model.assess(readings)
        predicted = result.dropna(subset=["RUL_median_h"])
        assert len(predicted) == len(readings) - MIN_READINGS + 1
        assert (predicted["RUL_low_h"] <= predicted["RUL_median_h"]).all()
        assert (predicted["RUL_median_h"] <= predicted["RUL_high_h"]).all()
        health = result.filter(like="_health_pct")
        assert ((health >= 0) & (health <= 100)).all().all()
        assert truth[device]["eol_h"] > readings["Hours"].max()


def test_too_few_readings_give_health_but_no_remaining_life(model):
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_01")].head(MIN_READINGS - 1)
    result = model.assess(readings)
    assert "RUL_median_h" not in result
    assert result["Overall_health_pct"].between(0, 100).all()


def test_met_limit_reports_end_of_life_and_zero_remaining_life(model):
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_01")].reset_index(drop=True)
    readings.loc[len(readings) - 1, "ESR_ohm"] = readings.loc[0, "ESR_ohm"] * EOL_LIMITS["esr_ratio"]
    last = model.assess(readings).iloc[-1]
    assert last["End_of_life_reached"] and last["RUL_median_h"] == 0
    assert last["ESR_ohm_health_pct"] == 0
    assert limit_used(readings)["ESR_ohm"].iloc[-1] == pytest.approx(1.0)


def test_saved_model_beats_the_trend_baseline_and_is_calibrated():
    metrics = json.loads((ARTIFACTS / "metadata.json").read_text())["test_metrics"]
    neural = metrics["neural_calibrated"]
    assert 0.75 <= neural["interval_coverage"] <= 0.85
    assert neural["median_absolute_error_h"] < metrics["trend_baseline"]["median_absolute_error_h"]
    assert np.isfinite(neural["median_interval_width_h"])
    assert list(json.loads((ARTIFACTS / "metadata.json").read_text())["signals"]) == list(SIGNALS)


def test_zero_baseline_leakage_is_refused_rather_than_reported_as_end_of_life(model):
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_01")].reset_index(drop=True)
    readings.loc[0, "Leakage_uA"] = 0.0
    with pytest.raises(ValueError, match="baseline"):
        model.assess(readings)


def test_sparkline_draws_rising_flat_and_too_short_series():
    import base64
    from caplife.six_rul_page import _sparkline

    def drawing(html):
        return base64.b64decode(html.split("base64,")[1].split('"')[0]).decode()

    rising = _sparkline(pd.Series([0, 10, 20]), pd.Series([1.0, 2.0, 3.0]), "ESR", "watch")
    assert rising.startswith('<img class="caplife-spark"') and 'alt="ESR trend over 3 readings"' in rising
    svg = drawing(rising)
    points = svg.split('<polyline points="')[1].split('"')[0].split()
    ys = [float(point.split(",")[1]) for point in points]
    assert ys[0] > ys[-1]  # higher values are drawn higher up
    assert "#d4881a" in svg  # the Watch colour
    assert "4.0,20.0" in drawing(_sparkline(pd.Series([0, 10]), pd.Series([5.0, 5.0]), "H2"))  # flat: middle
    assert _sparkline(pd.Series([0]), pd.Series([1.0]), "C") == ""


def test_saved_times_follow_the_viewers_time_zone():
    from datetime import datetime
    from caplife.accounts import local_time
    assert local_time(datetime(2026, 11, 22, 20, 24), "Europe/Copenhagen") == "2026-11-22 21:24"
    assert local_time(datetime(2026, 11, 22, 20, 24), None) == "2026-11-22 20:24 UTC"
    assert local_time(datetime(2026, 11, 22, 20, 24), "Not/AZone") == "2026-11-22 20:24 UTC"
