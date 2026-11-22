import io

import pandas as pd
import pytest

from caplife.measurement_review import (
    group_capacitor_warnings, nasa_reference_to_caplife, observed_capacitance_threshold, read_experimental_csv,
)
from caplife.real_data import RealDataValidationError, assess_real_measurements


def test_literal_capacitor_id_and_missing_measurement_are_preserved():
    raw = read_experimental_csv(b"Cap_ID,Hours,Capacitance_uF\nNA,0,100\nNA,10,NA\nNA,20,80\n")
    measurements = assess_real_measurements(raw)["measurements"]
    assert measurements["Cap_ID"].tolist() == ["NA"] * 3
    assert pd.isna(measurements.loc[1, "C_ratio"])
    assert measurements.loc[2, "C_ratio"] == 0.8


@pytest.mark.parametrize("payload", [
    b'Cap_ID,Hours,"Hours",Capacitance_uF\na,0,1,100\n',
    b"Cap_ID,Hours,Capacitance_uF\na,0,100,5\na,10,80,6\n",
])
def test_ambiguous_csv_is_rejected(payload):
    with pytest.raises(RealDataValidationError, match="Duplicate CSV column headers|header column count"):
        read_experimental_csv(payload)


def test_threshold_export_distinguishes_first_confirmation_and_reversal():
    readings = pd.read_csv(io.StringIO(
        "Cap_ID,Age_h,Capacitance_loss_pct\n"
        "A,0,0\nA,20,20\nA,30,\nA,40,18\nA,80,21\nA,120,22\n"
        "B,0,0\nB,20,20\nB,40,18\n"
    ))
    status = observed_capacitance_threshold(readings, 20).set_index("Capacitor")
    assert status.loc["A", "First at or above threshold (h)"] == 20
    assert status.loc["A", "First later reading below threshold (h)"] == 40
    assert status.loc["A", "First confirmed by two readings (h)"] == 120
    assert pd.isna(status.loc["B", "First confirmed by two readings (h)"])


def test_nasa_reference_rows_become_caplife_esr_readings():
    reference = pd.read_csv(io.StringIO(
        "Cap_ID,Age_h,Capacitance_uF,Resistance_Ohm\n"
        "A,0,100,0.10\nA,0,102,0.12\nA,10,95,0.15\n"
    ), dtype={"Cap_ID": str})
    frame = nasa_reference_to_caplife(reference)
    assert frame.columns.tolist() == ["Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm", "ESR_Test_Hz"]
    assert frame["Hours"].tolist() == [0.0, 10.0]
    assert frame.loc[0, "Capacitance_uF"] == 101 and frame.loc[0, "ESR_ohm"] == pytest.approx(0.11)
    assert (frame["ESR_Test_Hz"] == 10_000).all()
    report = assess_real_measurements(frame)
    assert report["measurements"]["C_ratio"].iloc[-1] == pytest.approx(95 / 101)


def test_identical_warnings_for_several_capacitors_are_merged():
    merged = group_capacitor_warnings([
        "C1: ESR test conditions are only partly documented; provisional.",
        "C2: ESR test conditions are only partly documented; provisional.",
        "C3: Only one reading.",
    ])
    assert merged == ["ESR test conditions are only partly documented; provisional. (2 capacitors: C1, C2)",
                      "C3: Only one reading."]
