"""Open every workspace of the Streamlit app headlessly and exercise its main actions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MAIN_KEY = "caplife_workflow_v07"
MONITOR = "Six-parameter RUL prediction (demonstration model)"
TRAIN = "Train an RUL model on uploaded outcomes"
MY_DATA = "My saved datasets"

# Shown label -> internal workspace value, as defined in app.py; the only three workspaces.
WORKSPACES = {"Monitor and Predict": MONITOR, "Train The AI": TRAIN, "My Data": MY_DATA}


def open_workspace(label: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert app.sidebar.radio(key=MAIN_KEY).options == list(WORKSPACES)
    assert len(app.sidebar.radio) == 1  # no Research list any more
    app.sidebar.radio(key=MAIN_KEY).set_value(WORKSPACES[label]).run()
    assert not app.exception, app.exception
    return app


def metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


@pytest.mark.parametrize("label", list(WORKSPACES))
def test_every_workspace_opens_without_error(label):
    app = open_workspace(label)
    assert app.header[0].value == label


def test_monitor_example_shows_fleet_and_capacitor_detail():
    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    assert not app.exception
    assert metric(app, "Capacitors") == "12"  # two simulated capacitors per end-of-life cause
    assert metric(app, "Remaining useful life").endswith(" h")
    app.selectbox(key="caplife_six_device").set_value("DEMO_04").run()
    assert not app.exception
    from caplife.six_rul import SixRULModel, example_fleet
    from caplife.six_rul_page import ARTIFACT_DIR
    measurements, _ = example_fleet()
    expected = SixRULModel(ARTIFACT_DIR).assess(measurements.query("Cap_ID == 'DEMO_04'")).iloc[-1]
    assert metric(app, "Overall health") == f"{expected['Overall_health_pct']:.0f}%"


def test_monitor_offers_pdf_and_excel_reports_and_csv_files():
    import io
    import json

    import openpyxl

    from caplife import operating_conditions as conditions
    from caplife import six_rul_page as page
    from caplife.monitor_report import pdf_bytes, xlsx_bytes
    from caplife.six_rul import example_fleet, full_limits

    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    assert not app.exception
    measurements, _ = example_fleet()
    limits = {device: full_limits() for device in measurements["Cap_ID"].unique()}
    results = page._assess_cached(measurements, json.dumps(limits, sort_keys=True), json.dumps({}), "test")
    fleet = page._fleet_table(results, None, limits)
    readings, assessment = results["DEMO_01"]
    report = page._report("Simulated example", fleet, "DEMO_01", readings, assessment, page._measured(readings),
                          limits["DEMO_01"], limits, conditions.latest(readings), {}, None, measurements)
    assert pdf_bytes(report)[:4] == b"%PDF"
    sheets = openpyxl.load_workbook(io.BytesIO(xlsx_bytes(report))).sheetnames
    assert {"Overview", "Fleet", "DEMO_01 history", "Limits", "Measurements"} <= set(sheets)


def test_monitor_runs_with_only_three_parameters():
    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    app.checkbox(key="caplife_six_three_only").check().run()
    assert not app.exception, app.exception
    assert any("Using 3 of 6 parameters" in c.value for c in app.caption)
    assert not app.info
    assert metric(app, "Remaining useful life").endswith(" h")


def test_monitor_takes_user_limits_and_forecasts_parameter_trends():
    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    device = app.selectbox(key="caplife_six_device").value
    boxes = [box for box in app.number_input if box.key.startswith(f"caplife_six_limit_{device}_")]
    assert len(boxes) == 6
    assert any("Predicted parameter trends to end of life" in m.value for m in app.markdown)
    before = metric(app, "Overall health")
    app.number_input(key=f"caplife_six_limit_{device}_Capacitance_uF").set_value(40.0).run()
    assert not app.exception
    assert metric(app, "Overall health") != before or before == "100%"
    assert any("Custom limits:" in c.value for c in app.caption)


def test_train_the_ai_example_passes_and_forecasts():
    app = open_workspace("Train The AI")
    app.radio(key="caplife_rul_source").set_value("Use the synthetic example").run()
    app.selectbox(key="caplife_rul_calibration").set_value("SYN_EXP_C").run()
    app.selectbox(key="caplife_rul_test").set_value("SYN_EXP_B").run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception
    assert any("Passed." in s.value for s in app.success)
    assert any("Remaining-life prediction over time" in m.value for m in app.markdown)
    assert any("Parameters used (6 of 6" in c.value for c in app.caption)


def test_my_data_asks_to_sign_in_when_signed_out():
    app = open_workspace("My Data")
    assert any("Sign in" in i.value for i in app.info)


def test_privacy_page_is_served_on_the_app_domain():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params["page"] = "privacy"
    app.run()
    assert not app.exception
    assert app.title[0].value == "Privacy policy"
    text = " ".join(m.value for m in app.markdown)
    assert "name**, **email address" in text and "My Data" in text
    assert not app.sidebar.radio  # the policy is a standalone page


def test_app_opens_on_home_and_each_card_opens_its_workspace():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert app.sidebar.radio(key=MAIN_KEY).value is None  # no workspace chosen: Home
    assert "Home" not in app.sidebar.radio(key=MAIN_KEY).options
    assert any("Choose where to start" in m.value for m in app.markdown)
    cards = {"caplife_home_open_monitor": "Monitor and Predict", "caplife_home_open_train": "Train The AI",
             "caplife_home_open_data": "My Data"}
    assert {b.key for b in app.button if b.key and b.key.startswith("caplife_home_open_")} == set(cards)
    for key, label in cards.items():
        app.sidebar.button(key="caplife_brand_home").click().run()
        app.button(key=key).click().run()
        assert not app.exception
        assert app.sidebar.radio(key=MAIN_KEY).value == WORKSPACES[label]
        assert app.header[0].value == label


def test_brand_returns_home_from_any_page_and_keeps_the_work():
    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    app.sidebar.button(key="caplife_brand_home").click().run()
    assert not app.exception
    assert app.sidebar.radio(key=MAIN_KEY).value is None
    assert any("Choose where to start" in m.value for m in app.markdown)
    app.sidebar.radio(key=MAIN_KEY).set_value(MONITOR).run()
    assert app.radio(key="caplife_six_source").value == "Use simulated example capacitors"  # still chosen


def test_upload_pages_have_no_column_guides_or_file_settings():
    for label in ("Monitor and Predict", "Train The AI"):
        app = open_workspace(label)
        assert not any(e.label in ("Columns and units", "What goes in each file", "File settings: time unit and pressure")
                       for e in app.expander)
