import json
from pathlib import Path

import numpy as np
import pandas as pd

from caplife.public_benchmark import evaluate, landmark_outcomes, screen_forecasts, summarize_screening


def test_first_crossing_and_right_censoring():
    loss = pd.DataFrame(
        {"Crossing": [0, 4, 8, 20, 16, 22], "Censored": [0, 4, 8, 12, 16, 18]},
        index=[0, 47, 94, 118, 142, 166],
    )
    actual = landmark_outcomes(loss).set_index("device")
    assert actual.loc["Crossing", ["rul_lower_h", "rul_upper_h"]].tolist() == [0, 24]
    assert actual.loc["Censored", "rul_lower_h"] == 72
    assert np.isinf(actual.loc["Censored", "rul_upper_h"])
    assert actual.loc["Crossing", "trend_h"] == actual.loc["Censored", "trend_h"]

    later = landmark_outcomes(loss, 142).set_index("device")
    assert later.index.tolist() == ["Censored"]
    assert later.loc["Censored", "rul_lower_h"] == 24
    assert np.isinf(later.loc["Censored", "rul_upper_h"])


def test_held_out_outcomes_do_not_change_forecasts():
    data = json.loads(Path("artifacts/public_benchmark.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["predictions"])
    trend_rows = rows.loc[rows["model"] == "Recent trend"].rename(columns={"forecast_h": "trend_h"})
    trend_rows["rul_upper_h"] = trend_rows["rul_upper_h"].fillna(np.inf)
    train = trend_rows.loc[trend_rows["test_study"] == "NASA EOS-2"]
    test = trend_rows.loc[trend_rows["test_study"] == "PoFINN"]
    changed = test.assign(rul_lower_h=test["rul_lower_h"] + 50, rul_upper_h=test["rul_upper_h"] + 50)

    summary, original = evaluate(train, test, "NASA EOS-2", "PoFINN")
    changed_summary, recomputed = evaluate(train, changed, "NASA EOS-2", "PoFINN")

    forecast_columns = ["forecast_h", "band_lower_h", "band_upper_h"]
    pd.testing.assert_frame_equal(pd.DataFrame(original)[forecast_columns],
                                  pd.DataFrame(recomputed)[forecast_columns])
    assert summary != changed_summary
    assert all(row["observed_crossings"] == 8 for row in summary)


def test_screen_uses_only_observed_history_and_reports_zero_retained():
    train = pd.DataFrame({"loss_at_origin_pct": [8.0, 12.0]})
    observations = pd.DataFrame({
        "device": ["In range", "Out of range", "Long band", "Fluctuation"],
        "loss_at_origin_pct": [8.0, 15.0, 10.0, 10.0],
        "recent_history_h": [20.0] * 4,
        "recent_loss_increases": [True, True, True, False],
    })
    details = [
        {"model": "Recent trend with study calibration", "device": device, "test_study": "test",
         "forecast_h": 10.0, "band_lower_h": 5.0, "band_upper_h": upper,
         "rul_lower_h": 10.0, "rul_upper_h": 22.0}
        for device, upper in zip(observations["device"], [20.0, 20.0, 21.0, 20.0])
    ]
    result = screen_forecasts(details, train, observations, 94)
    assert [row["decision"] for row in result] == [
        "Show forecast", "Show forecast", "Forecast extends beyond recent history span",
        "Recent readings fluctuate",
    ]
    changed = screen_forecasts([dict(row, rul_upper_h=None) for row in details], train, observations, 94)
    assert [row["decision"] for row in changed] == [row["decision"] for row in result]
    assert [row["shown"] for row in summarize_screening(result)] == [4, 2, 1]
    assert summarize_screening(changed)[-1]["mean_gap_h"] is None
    zero = screen_forecasts(details, pd.DataFrame({"loss_at_origin_pct": [1, 2]}), observations, 94)
    assert summarize_screening(zero)[-1]["shown"] == 0
