import numpy as np
import pandas as pd

from caplife.neural_benchmark import evaluate_transfer, make_examples
from caplife.real_data import observed_capacitance_threshold


def test_horizon_labels_respect_crossing_windows_and_censoring():
    loss = pd.DataFrame({"crosses": [0, 3, 5, 10, 20, 18], "censored": [0, 3, 5, 10, 14, 19]},
                        index=[0, 20, 40, 60, 100, 120])
    rows = make_examples(loss).set_index(["device", "origin_h"])
    assert np.isnan(rows.loc[("crosses", 40), "outcome"])
    assert rows.loc[("crosses", 60), "outcome"] == 1
    assert ("crosses", 100) not in rows.index
    assert rows.loc[("censored", 60), "outcome"] == 0
    assert np.isnan(rows.loc[("censored", 100), "outcome"])
    assert rows.loc[("crosses", 60), "recent_rate_pct_per_h"] == 0.25


def test_changing_held_out_labels_cannot_change_predictions():
    loss = pd.DataFrame({"early": [0, 3, 7, 14, 22, 25, 26], "late": [0, 2, 3, 5, 10, 12, 20]},
                        index=[0, 20, 40, 60, 80, 100, 120])
    train = make_examples(loss)
    test = train.assign(device=train["device"].map({"early": "a", "late": "b"}))
    original, first = evaluate_transfer(train, test)
    flipped, second = evaluate_transfer(train, test.assign(outcome=1 - test["outcome"]))
    pd.testing.assert_series_equal(original["neural"], flipped["neural"])
    assert first != second


def test_observed_threshold_reports_crossing_and_missing_baseline():
    history = pd.DataFrame({
        "Cap_ID": ["a", "a", "a", "a", "b", "b", "c", "c", "d"],
        "Age_h": [0, 10, 20, 30, 5, 12, 0, 15, 0],
        "Capacitance_loss_pct": [0, 12, 100 * (1 - 0.8), 18, np.nan, np.nan, 0, 3, 0],
    })
    status = observed_capacitance_threshold(history, 20).set_index("Capacitor")
    assert status.loc["a", "First at or above threshold (h)"] == 20
    assert status.loc["a", "Last below before crossing (h)"] == 10
    assert status.loc["a", "Latest measured loss (%)"] == 18
    assert status.loc["b", "Status"] == "No comparable capacitance readings"
    assert status.loc["c", "Last below before crossing (h)"] == 15
    assert np.isnan(status.loc["c", "First at or above threshold (h)"])
    assert status.loc["d", "Status"] == "Need a baseline and a later reading"
