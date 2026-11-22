import numpy as np
import pandas as pd
import pytest

from caplife.esr_validation import ESR_FEATURES, paired_examples


def test_esr_features_ignore_units_and_future_measurements():
    loss = pd.DataFrame({"cap": [0, 3, 6, 9, 10, 13]}, index=[0, 20, 40, 60, 80, 100])
    esr = pd.DataFrame({"cap": [1, 1.1, 1.21, 1.4, 1.5, 1.6]}, index=loss.index)
    scaled = esr * 1_000
    scaled.loc[60:, "cap"] *= 7

    original = paired_examples(loss, esr, 12).set_index("origin_h")
    changed = paired_examples(loss, scaled, 12).set_index("origin_h")
    expected = np.log(1.1) / 20
    np.testing.assert_allclose(original.loc[40, list(ESR_FEATURES)].to_numpy(dtype=float), expected)
    np.testing.assert_allclose(changed.loc[40, list(ESR_FEATURES)].to_numpy(dtype=float), expected)
    pd.testing.assert_series_equal(original.loc[40].drop(list(ESR_FEATURES)),
                                   changed.loc[40].drop(list(ESR_FEATURES)))
    assert original.loc[60, "esr_recent_log_rate"] != changed.loc[60, "esr_recent_log_rate"]


def test_ambiguous_crossing_and_missing_esr_are_preserved():
    ages = [0, 20, 40, 60, 80, 100, 120]
    loss = pd.DataFrame({"cap": [0, 2, 4, 6, 11, 11.95, 12.2]}, index=ages)
    esr = pd.DataFrame({"cap": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}, index=ages)
    rows = paired_examples(loss, esr, 12, 0.1).set_index("origin_h")
    assert np.isnan(rows.loc[40, "outcome"])
    assert np.isnan(rows.loc[60, "outcome"])
    assert rows.loc[80, "outcome"] == 1
    assert 100 not in rows.index

    esr.loc[20, "cap"] = np.nan
    assert paired_examples(loss, esr, 12, 0.1)["origin_h"].tolist() == [80]


def test_esr_examples_reject_invalid_inspection_grids():
    loss = pd.DataFrame({"cap": [0, 2, 3]}, index=[0, 20, 20])
    with pytest.raises(ValueError, match="strictly increasing"):
        paired_examples(loss, loss, 12)
