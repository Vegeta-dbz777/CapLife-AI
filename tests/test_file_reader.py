from io import BytesIO

import pandas as pd
import pytest

from caplife.file_reader import read_measurement_file
from caplife.real_data import RealDataValidationError, validate_real_measurements
from caplife.rul_workflow import read_outcomes

ROWS = pd.DataFrame({
    "Cap_ID": ["A", "A", "A", "B", "B", "B"],
    "Hours": [0, 100, 200, 0, 100, 200],
    "Capacitance_uF": [1000.5, 990.25, 980.0, 950.0, 945.5, 940.0],
    "ESR_ohm": [0.0854, 0.0861, 0.0870, 0.09, 0.091, 0.093],
    "Pressure_Abs_kPa": [101.3, 110.2, 120.4, 101.4, 105.0, 109.9],
})


def read(payload: bytes, **options):
    frame, notes = read_measurement_file(payload, **options)
    return validate_real_measurements(frame), notes


def test_european_excel_csv_with_semicolons_and_decimal_commas():
    frame, notes = read(ROWS.to_csv(index=False, sep=";", decimal=",").encode("cp1252"))
    assert frame["ESR_ohm"].iloc[0] == pytest.approx(0.0854)
    assert any("semicolon" in n for n in notes) and any("decimal commas" in n for n in notes)


def test_tab_separated_and_excel_files():
    assert len(read(ROWS.to_csv(index=False, sep="\t").encode())[0]) == 6
    buffer = BytesIO()
    ROWS.to_excel(buffer, index=False)
    frame, notes = read(buffer.getvalue(), filename="lab.xlsx")
    assert frame["Capacitance_uF"].tolist() == ROWS["Capacitance_uF"].tolist()
    assert any("Excel" in n for n in notes)


def test_column_name_variations_are_recognised():
    renamed = ROWS.rename(columns={"Cap_ID": "capacitor id", "Hours": "Time (h) ", "Capacitance_uF": "Capacitance (µF)",
                                   "ESR_ohm": "ESR (Ω)", "Pressure_Abs_kPa": "Pressure"})
    frame, notes = read(renamed.to_csv(index=False).encode())
    assert {"Cap_ID", "Hours", "Capacitance_uF", "ESR_ohm", "Pressure_Abs_kPa"} <= set(frame.columns)
    assert any("Recognised" in n for n in notes)


def test_missing_markers_bad_rows_and_repeats_are_handled_without_rejecting_the_file():
    text = ROWS.astype(str)
    text.loc[1, "ESR_ohm"] = "n/a"
    text.loc[2, "Capacitance_uF"] = "-"
    extra = pd.DataFrame([{**text.iloc[4].to_dict()}, {**text.iloc[0].to_dict(), "Cap_ID": ""},
                          {**text.iloc[0].to_dict(), "Hours": "-5", "Cap_ID": "C"}])
    frame, notes = read(pd.concat([text, extra]).to_csv(index=False).encode())
    assert len(frame) == 6 and pd.isna(frame.loc[1, "ESR_ohm"]) and pd.isna(frame.loc[2, "Capacitance_uF"])
    joined = " ".join(notes)
    assert "'n/a' or '-'" in joined and "without a capacitor ID" in joined and "negative time" in joined
    assert "repeated reading" in joined and "B at 100 h" in joined


def test_gauge_pressure_is_detected_or_follows_the_choice():
    gauge = ROWS.assign(Pressure_Abs_kPa=ROWS["Pressure_Abs_kPa"] - 101.325)
    frame, notes = read(gauge.to_csv(index=False).encode())
    assert frame["Pressure_Abs_kPa"].min() > 100 and any("gauge" in n for n in notes)
    absolute, notes = read(ROWS.to_csv(index=False).encode())
    assert absolute["Pressure_Abs_kPa"].iloc[0] == pytest.approx(101.3) and not any("gauge" in n for n in notes)
    forced, _ = read(ROWS.to_csv(index=False).encode(), pressure="gauge")
    assert forced["Pressure_Abs_kPa"].iloc[0] == pytest.approx(202.625)


def test_time_in_days_is_converted_from_the_column_name_or_the_setting():
    frame, notes = read(ROWS.rename(columns={"Hours": "Days"}).to_csv(index=False).encode())
    assert frame["Hours"].max() == 200 * 24 and any("days to hours" in n for n in notes)
    frame, _ = read(ROWS.to_csv(index=False).encode(), time_unit="weeks")
    assert frame["Hours"].max() == 200 * 168


def test_misaligned_rows_and_missing_columns_still_fail_clearly():
    with pytest.raises(RealDataValidationError, match="Line 2 does not match the header column count"):
        read_measurement_file(b"Cap_ID,Hours,Capacitance_uF\na,0,100,5\n")
    with pytest.raises(RealDataValidationError, match="Could not find a Hours column"):
        read_measurement_file(b"Cap_ID,Capacitance_uF\na,100\n")


def test_outcomes_accept_semicolons_decimal_commas_and_days():
    payload = "Cap_ID;Experiment_ID;Endpoint;Last_alive_h;Failure_by_h\nA;E1;C loss 20%;100,5;120\nB;E1;C loss 20%;90;\n"
    frame = read_outcomes(payload.encode(), time_unit="days")
    assert frame["Last_alive_h"].tolist() == ["2412", "2160"] and frame["Failure_by_h"].tolist() == ["2880", ""]
