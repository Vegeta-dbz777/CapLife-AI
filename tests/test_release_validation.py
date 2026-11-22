import numpy as np
import pandas as pd

from caplife.nasa_eis import acquisition_date, spectrum_values
from caplife.neural_benchmark import make_examples
from caplife.release_validation import make_origins


def test_public_validation_uses_past_data_and_preserves_interval_labels():
    loss = pd.DataFrame({"a": [0, 7, 12, 16, 20, 18, 20, 21]}, index=[0, 20, 40, 60, 100, 120, 140, 160])
    ordinary = make_origins(loss, "test", 48, 20).set_index("age_h")
    original = make_examples(loss).set_index("origin_h")
    pd.testing.assert_series_equal(ordinary["outcome"].rename_axis("origin_h"), original["outcome"])
    assert np.isnan(ordinary.loc[40, "outcome"])
    assert ordinary.loc[60, "outcome"] == 1
    assert ordinary.loc[60, "remaining_lower_h"] == 0
    assert ordinary.loc[60, "remaining_upper_h"] == 40

    future_changes = loss.copy()
    future_changes.loc[100:, "a"] = 17
    new = make_origins(future_changes, "test", 48, 20).set_index("age_h")
    label_columns = ["outcome", "remaining_lower_h", "remaining_upper_h"]
    pd.testing.assert_series_equal(ordinary.loc[60].drop(label_columns), new.loc[60].drop(label_columns))
    assert new.loc[60, "outcome"] == 0
    assert np.isposinf(new.loc[60, "remaining_upper_h"])

    repeated = make_origins(loss, "test", 48, 20, repeated=True).set_index("age_h")
    assert repeated.loc[60, "outcome"] == 0
    assert repeated.loc[120, "outcome"] == 1


def test_eis_extracts_correct_units_and_requires_coherent_impedance():
    frequency = np.array([25, 102, 1000, 9950])
    spectrum = np.zeros((10, len(frequency)))
    spectrum[0] = frequency
    spectrum[1] = 0.2
    spectrum[2] = 1 / (2 * np.pi * frequency * 0.002)
    spectrum[3] = np.hypot(spectrum[1], spectrum[2])
    spectrum[8] = 2_000
    np.testing.assert_allclose(spectrum_values(spectrum), (2_000, 0.2, 2_000))

    spectrum[0, 1] = 125
    spectrum[2, 1] = 1 / (2 * np.pi * 125 * 0.002)
    spectrum[3, 1] = np.hypot(spectrum[1, 1], spectrum[2, 1])
    assert np.isnan(spectrum_values(spectrum)[0])
    assert np.isfinite(spectrum_values(spectrum)[1:]).all()
    spectrum[3, 2] = 5
    assert np.isnan(spectrum_values(spectrum)).all()


def test_eis_accepts_the_archive_acquisition_header():
    caption = "Acquisition started on : 09/29/2014 13:13:12"
    chars = np.array([ord(char) for char in caption], dtype="uint16").reshape(-1, 1)
    assert acquisition_date(chars) == "2014-09-29"
