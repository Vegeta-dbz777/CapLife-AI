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
        assert readings["Case_Expansion_um"].iloc[0] == 0
        assert readings["Case_Expansion_um"].min() > -1.5  # small negatives are sensor noise only
    assert set(meta["eol_cause"].str.split(", ").str[0]) <= {"capacitance", "ESR", "leakage", "expansion", "H2",
                                                             "pressure"}


def test_end_of_life_rules_match_the_documented_limits():
    base = {"Capacitance_uF": 100, "ESR_ohm": 0.1, "Leakage_uA": 10, "Case_Expansion_um": 0.0,
            "H2_Generated_mL_STP": 0.0, "Pressure_Abs_kPa": 101.3}
    assert end_of_life_causes(base, base) == []
    assert end_of_life_causes(base, {**base, "Case_Expansion_um": EOL_LIMITS["expansion_um"]}) == ["expansion"]
    assert end_of_life_causes(base, {**base, "H2_Generated_mL_STP": EOL_LIMITS["h2_ml"]}) == ["H2"]
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


def test_simulated_gas_readings_start_at_atmosphere_and_accumulate():
    # Regression: cumulative H2 was once integrated over an unsorted time array, which pulled
    # pressure below atmospheric and held H2 at zero for long-lived capacitors.
    from caplife.six_sim import ATMOSPHERE_KPA
    measurements, _ = simulate_fleet(60, seed=2027)
    first = measurements.groupby("Cap_ID").first()
    assert (first["Pressure_Abs_kPa"] - ATMOSPHERE_KPA).abs().max() < 2.5  # measurement noise only
    final_h2 = measurements.groupby("Cap_ID")["H2_Generated_mL_STP"].last()
    assert (final_h2 > 0).all()
    for _, readings in measurements.groupby("Cap_ID"):
        assert readings["Pressure_Abs_kPa"].iloc[-1] > readings["Pressure_Abs_kPa"].iloc[0]


def test_chart_axis_fits_small_trends_and_reports_far_limits():
    from caplife.six_rul_page import _chart_domain
    # H2 rising 0 -> 0.15 mL with a 6 mL limit: axis fits the data, limit reported above.
    low, high, where = _chart_domain(pd.Series([0.0, 0.05, 0.15]), 6.0, 2)
    assert where == "above" and high < 1.0 and low < 0.0
    # Capacitance 1000 -> 985 uF with an 800 uF limit far below.
    assert _chart_domain(pd.Series([1000.0, 985.0]), 800.0, 1)[2] == "below"
    # A limit close to the readings is kept on the chart.
    low, high, where = _chart_domain(pd.Series([100.0, 180.0]), 250.0, 1)
    assert where is None and high >= 250.0


def test_three_parameter_histories_get_health_and_remaining_life(model):
    from caplife.six_rul import COMMON_SUBSET
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_02"), ["Cap_ID", "Hours", *COMMON_SUBSET]]
    result = model.assess(readings)
    last = result.iloc[-1]
    assert pd.notna(last["RUL_median_h"]) and last["RUL_low_h"] <= last["RUL_median_h"] <= last["RUL_high_h"]
    assert np.isnan(last["Pressure_Abs_kPa_health_pct"]) and not np.isnan(last["ESR_ohm_health_pct"])
    assert last["Overall_health_pct"] == min(last[f"{s}_health_pct"] for s in COMMON_SUBSET)
    assert model.metrics_for(COMMON_SUBSET)["predictions"] > 0


def test_hidden_indicators_in_training_match_a_real_partial_file():
    from caplife.six_rul import COMMON_SUBSET, _subset_mask, hide_indicators
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_03")]
    full = sequences(readings, SIGNALS)
    hidden = hide_indicators(full, np.tile(_subset_mask(COMMON_SUBSET), (len(full), 1)))
    assert np.allclose(hidden, sequences(readings[["Cap_ID", "Hours", *COMMON_SUBSET]]))


def test_every_parameter_combination_is_calibrated_and_tested():
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text())
    by_subset = metadata["test_metrics_by_subset"]
    assert len(by_subset) == 63 == len(metadata["conformal_adjustment_log_by_subset"])
    assert all(0.74 <= metrics["interval_coverage"] <= 0.87 for metrics in by_subset.values())
    # Fewer parameters must not look more precise than all six.
    full = by_subset["+".join(SIGNALS)]["median_absolute_error_h"]
    assert by_subset["Capacitance_uF"]["median_absolute_error_h"] > full


def test_parameter_lists_keep_abbreviations_capitalised():
    from caplife.six_rul_page import _names
    assert _names(("Capacitance_uF", "ESR_ohm", "Leakage_uA")) == "capacitance, ESR and leakage current"
    assert _names(("H2_Generated_mL_STP",)) == "H₂ generated"


def test_capacitor_past_its_limit_from_the_start_gets_a_clear_message_not_a_flat_chart(model):
    from caplife.measurement_review import nasa_reference_to_caplife
    from caplife.six_rul_page import _always_at_end_of_life, _end_of_life_message
    nasa = nasa_reference_to_caplife(pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data"
                                                 / "nasa_es14_reference.csv", dtype={"Cap_ID": str}))
    readings = nasa.loc[nasa["Cap_ID"].eq("ES14C1"), ["Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm"]]
    assessment = model.assess(readings)
    # ES14C1's ESR doubled within 2 h of electrical overstress, and it lost over 20% of its
    # capacitance within the first day.
    assert _always_at_end_of_life(assessment)
    assert _end_of_life_message(assessment) == ("End-of-life limit first met at 2 h (ESR) and still met at "
                                                "the latest reading, 5,106 h (Capacitance).")
    healthy = model.assess(example_fleet()[0].query("Cap_ID == 'DEMO_01'"))
    assert not _always_at_end_of_life(healthy)


def test_aging_damage_does_not_heal_but_a_single_outlier_is_ignored():
    from caplife.six_rul import current_limit_used
    readings = pd.DataFrame({"Cap_ID": "X", "Hours": [0, 10, 20, 30, 40, 50, 60],
                             "ESR_ohm": [0.10, 0.19, 0.19, 0.19, 0.15, 0.14, 0.14]})
    held = limit_used(readings)["ESR_ohm"]
    assert held.is_monotonic_increasing and held.iloc[-1] == pytest.approx(0.9)  # ESR held at 1.9x
    assert current_limit_used(readings)["ESR_ohm"].iloc[-1] == pytest.approx(0.4)
    spike = readings.assign(ESR_ohm=[0.10, 0.11, 0.30, 0.11, 0.12, 0.12, 0.12])
    assert limit_used(spike)["ESR_ohm"].iloc[-1] < 0.25  # one 3x reading is not lasting damage
    assert limit_used(spike)["ESR_ohm"].iloc[2] >= 1  # but that reading itself meets the limit


def test_nasa_overstress_capacitors_stay_at_end_of_life_once_a_limit_is_held():
    from caplife.measurement_review import nasa_reference_to_caplife
    from caplife.six_rul_page import _status
    model = SixRULModel(ARTIFACTS)
    nasa = nasa_reference_to_caplife(pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data"
                                                 / "nasa_es14_reference.csv", dtype={"Cap_ID": str}))
    for _, readings in nasa.groupby("Cap_ID"):
        reached = model.assess(readings[["Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm"]])["End_of_life_reached"]
        flags = reached.to_numpy()
        held = next(i for i in range(len(flags) - 1) if flags[i] and flags[i + 1])
        assert flags[held:].all()  # once a limit is held, the capacitor stays at end of life
    assert _status(53, False) == "Watch"  # ESR up 47% of the way to its limit is not "Good"
    assert _status(75, False) == "Good" and _status(30, False) == "Critical"


def test_one_faulty_reading_only_affects_its_own_prediction(model):
    measurements, _ = example_fleet()
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_03")].reset_index(drop=True)
    spiked = readings.copy()
    middle = len(readings) // 2
    spiked.loc[middle, "ESR_ohm"] = readings.loc[0, "ESR_ohm"] * 2.3  # one reading past the ESR limit
    clean, faulty = model.assess(readings), model.assess(spiked)
    assert faulty.loc[middle, "End_of_life_reached"] and faulty.loc[middle, "RUL_median_h"] == 0
    after = slice(middle + 1, None)
    assert not faulty.loc[after, "End_of_life_reached"].any()
    ratio = faulty.loc[after, "RUL_median_h"] / clean.loc[after, "RUL_median_h"]
    assert ratio.between(0.9, 1.1).all()  # later predictions are not dragged down by the spike


def test_every_indicator_can_end_a_capacitors_life(model):
    measurements, truth = example_fleet()
    assert {cause.split(", ")[0] for cause in (t["eol_cause"] for t in truth.values())} == {
        "capacitance", "ESR", "leakage", "expansion", "H2", "pressure"}
    readings = measurements.loc[measurements["Cap_ID"].eq("DEMO_01")].reset_index(drop=True)
    for signal, past in (("Case_Expansion_um", EOL_LIMITS["expansion_um"] + 1),
                         ("H2_Generated_mL_STP", EOL_LIMITS["h2_ml"] + 0.1)):
        bulged = readings.copy()
        bulged.loc[len(bulged) - 3:, signal] = past  # held over the last three readings
        last = model.assess(bulged).iloc[-1]
        assert last["End_of_life_reached"] and last[f"{signal}_health_pct"] == 0
