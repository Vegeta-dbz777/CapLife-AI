import io

import pandas as pd
import pytest

from caplife.column_match import MEASUREMENTS, NOT_USED, OUTCOMES, apply, guess, preview, problems
from caplife.file_reader import read_measurement_file
from caplife.rul_workflow import read_outcomes

LAB = ("Serial number;Test time (days);Cap (nF);ESR (mOhm);Leakage (mA);Pressure [bar];Remarks\n"
       "A1;0;1000000;80;0,012;1,015;new\n"
       "A1;10;990000;84;0,013;1,02;\n"
       "A1;20;n/a;90;0,014;1,03;\n").encode()


def test_guesses_columns_and_units_from_their_names():
    table = preview(LAB, "lab.csv", MEASUREMENTS)
    assert table["Use as"].tolist() == ["Capacitor ID", "Time (days)", "Capacitance (nF)", "ESR (mΩ)",
                                        "Leakage current (mA)", "Internal pressure (bar)", NOT_USED]
    assert table["Example values"].iloc[0] == "A1, A1, A1"
    assert problems(table["Use as"].tolist(), MEASUREMENTS) == []


def test_matched_file_is_read_in_caplife_units():
    choices = preview(LAB, "lab.csv", MEASUREMENTS)["Use as"].tolist()
    frame, _ = read_measurement_file(apply(LAB, "lab.csv", choices, MEASUREMENTS), "lab.csv")
    assert frame["Hours"].tolist() == [0, 240, 480]
    assert frame["Capacitance_uF"].iloc[0] == pytest.approx(1000)
    assert pd.isna(frame["Capacitance_uF"].iloc[2])  # a missing marker stays missing
    assert frame["ESR_ohm"].iloc[0] == pytest.approx(0.08)
    assert frame["Leakage_uA"].iloc[0] == pytest.approx(12)
    assert frame["Pressure_Abs_kPa"].iloc[0] == pytest.approx(101.5)
    assert "Remarks" not in frame


def test_standard_files_pass_through_unchanged():
    standard = b"Cap_ID,Hours,Capacitance_uF,Notes\nC1,0,1000,x\nC1,10,990,\n"
    choices = guess(["Cap_ID", "Hours", "Capacitance_uF", "Notes"], MEASUREMENTS)
    assert apply(standard, "a.csv", choices, MEASUREMENTS) is standard
    outcome = b"Cap_ID,Experiment_ID,Endpoint,Last_alive_h,Failure_by_h\nC1,E1,Capacitance loss,100,120\n"
    columns = ["Cap_ID", "Experiment_ID", "Endpoint", "Last_alive_h", "Failure_by_h"]
    assert apply(outcome, "o.csv", guess(columns, OUTCOMES), OUTCOMES) is outcome


def test_excel_with_other_names_is_rewritten_as_csv_the_reader_accepts():
    buffer = io.BytesIO()
    pd.DataFrame({"Unit": ["U1", "U1"], "Age (weeks)": [0, 1], "C [uF]": [100.0, 99.0]}).to_excel(buffer, index=False)
    choices = guess(["Unit", "Age (weeks)", "C [uF]"], MEASUREMENTS)
    assert choices == ["Capacitor ID", "Time (weeks)", "Capacitance (µF)"]
    frame, _ = read_measurement_file(apply(buffer.getvalue(), "lab.xlsx", choices, MEASUREMENTS), "lab.xlsx")
    assert frame["Hours"].tolist() == [0, 168] and frame["Capacitance_uF"].tolist() == [100, 99]


def test_outcome_columns_are_matched_by_meaning():
    payload = b"Capacitor,Batch,Criterion,Last alive (hours),Failed at (h)\nC1,B1,20% loss,100,120\nC2,B1,20% loss,150,\n"
    choices = preview(payload, "o.csv", OUTCOMES)["Use as"].tolist()
    assert problems(choices, OUTCOMES) == []
    frame = read_outcomes(apply(payload, "o.csv", choices, OUTCOMES))
    assert frame.columns.tolist() == ["Cap_ID", "Experiment_ID", "Endpoint", "Last_alive_h", "Failure_by_h"]
    assert frame["Failure_by_h"].tolist() == ["120", ""]


def test_missing_and_repeated_columns_are_explained():
    assert problems(["Capacitance (µF)", NOT_USED], MEASUREMENTS) == [
        "Choose the column to use as Capacitor ID.", "Choose the column to use as Time."]
    assert problems(["Capacitor ID", "Time (hours)", "Time (days)", "ESR (Ω)", "ESR (mΩ)"], MEASUREMENTS) == [
        "More than one column is set as Time. Keep one.", "More than one column is set as ESR. Keep one."]
