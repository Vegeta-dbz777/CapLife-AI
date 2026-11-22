import io
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader

from caplife.config import FEATURES
from caplife.degradation_rates import calculate_degradation_rates
from caplife.measurement_review import read_experimental_csv, reference_age_observations
from caplife.predict import default_prediction_hour
from caplife.readiness import audit_measurement_readiness
from caplife.real_data import assess_real_measurements, assess_reference_age_measurements
from caplife.report_exports import pdf_report_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_reference_age_notes_report_largest_gap_and_early_loss():
    raw = read_experimental_csv(
        b"Cap_ID,Age_h,Capacitance_uF,Resistance_Ohm\n"
        b"A,0,100,1\nA,2,88,1.1\nA,20,85,1.2\nA,100,84,1.2\nA,1000,80,1.3\n"
        b"B,0,100,1\nB,2,90,1.1\nB,20,88,1.2\nB,60,87,1.2\nB,1000,84,1.3\n",
        reference_age=True,
    )
    _, history = assess_reference_age_measurements(raw)
    notes = reference_age_observations(history)
    assert notes["largest_gap"] == {"Cap_ID": "B", "start_h": 60.0, "end_h": 1000.0, "hours": 940.0}
    assert notes["first_step_loss_range"] == pytest.approx((10.0, 12.0))
    assert notes["first_step_age_range"] == (2.0, 2.0)
    # A: 16% of its latest 20% loss by 100 h; B: 13% of its latest 16% by 60 h.
    assert notes["early_cutoff_h"] == 100.0
    assert abs(notes["median_early_share"] - (0.8 + 13 / 16) / 2) < 1e-12


def test_public_example_notes_flag_early_loss_and_sampling_gap():
    raw = read_experimental_csv((ROOT / "sample_data" / "nasa_es14_reference.csv").read_bytes(), reference_age=True)
    _, history = assess_reference_age_measurements(raw)
    notes = reference_age_observations(history)
    assert notes["median_early_share"] >= 0.5
    assert notes["largest_gap_fraction_of_span"] >= 0.2


def test_equal_rates_have_exactly_zero_acceleration():
    measurements = assess_real_measurements(read_experimental_csv(
        b"Cap_ID,Hours,Leakage_uA\nA,0,10\nA,300,10.8\nA,600,11.6\nA,900,12.4\n"
    ))["measurements"]
    accelerations = calculate_degradation_rates(measurements)["accelerations"]
    leakage = accelerations.loc[accelerations["Indicator"].str.startswith("Leakage")]
    assert len(leakage) == 2
    assert leakage["Acceleration_per_hour2"].eq(0.0).all()


def test_six_indicator_pdf_has_no_near_blank_spillover_page():
    rows = ["Cap_ID,Hours,Capacitance_uF,ESR_ohm,Leakage_uA,Case_Expansion_um,H2_Generated_mL_STP,Pressure_Abs_kPa,"
            "Ambient_C,Core_C,Voltage_V,Rated_V,Ripple_A,Ripple_Hz,ESR_Test_Hz,ESR_Test_Temp_C"]
    for cap in range(3):
        for k in range(8):
            rows.append(f"C{cap},{300 * k},{1000 - 15 * k},{0.1 + 0.005 * k},{10 + 0.8 * k},{4 * k},{0.04 * k * k},"
                        f"{101.3 + 0.6 * k * k},105,112,25,35,1.2,20000,100000,25")
    report = assess_real_measurements(read_experimental_csv("\n".join(rows).encode()))
    audit = audit_measurement_readiness(report)
    rates = calculate_degradation_rates(report["measurements"])
    pages = PdfReader(io.BytesIO(pdf_report_bytes(report, audit, "C0", rate_report=rates))).pages
    texts = [page.extract_text() for page in pages]
    assert "Measured history" in texts[1]
    assert all(len(text.split()) > 40 for text in texts)
    assert "Research documentation priorities" in "".join(texts[:2])


def test_demo_defaults_to_last_hour_before_illustrative_eol():
    hours = [0, 100, 200, 300, 400, 500, 600, 700]
    cap = pd.DataFrame({feature: 0.5 for feature in FEATURES} | {
        "Cap_ID": "A", "Hours": hours, "C_ratio": [1, .98, .96, .94, .92, .9, .85, .79],
        "ESR_ratio": 1.1, "Leakage_ratio": 1.0, "Case_Strain_Index": 0.1,
    })
    assert default_prediction_hour(cap, hours[5:]) == 600
    cap["C_ratio"] = 0.5
    assert default_prediction_hour(cap, hours[5:]) == 700
