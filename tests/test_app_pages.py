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


def test_monitor_pdf_report_is_the_page_as_shown():
    """The PDF follows the page: the limit and condition boxes as shown, and the page's own charts."""
    import io
    import json

    import openpyxl
    from pypdf import PdfReader

    from caplife import operating_conditions as conditions
    from caplife import six_rul_page as page
    from caplife.monitor_report import pdf_bytes, xlsx_bytes
    from caplife.six_rul import example_fleet, full_limits

    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    device = app.selectbox(key="caplife_six_device").value
    app.number_input(key=f"caplife_six_limit_{device}_ESR_ohm").set_value(150.0).run()
    recorded = app.number_input(key=f"caplife_six_condition_{device}_Ambient_C").value
    app.number_input(key=f"caplife_six_condition_{device}_Ambient_C").set_value(recorded + 15).run()
    assert not app.exception, app.exception
    view = app.session_state[page.REPORT_VIEW]
    assert ("ESR rise (%)", "150.00", True) in view.limits
    ambient = next(box for box in view.conditions if box[0].startswith("Ambient"))
    assert ambient[1] == f"{recorded + 15:,.2f}" and ambient[2] == f"{recorded:,.2f}"
    assert ("violet", "Changed conditions in use") in view.badges
    titles = [section.title for section in view.sections]
    assert titles == ["Six-parameter history", "Predicted parameter trends to end of life",
                      "Remaining-life prediction over time"]
    assert len(view.sections[0].charts) == 6 and len(view.sections[1].charts) == 6

    measurements, _ = example_fleet()
    limits = {d: full_limits() for d in measurements["Cap_ID"].unique()}
    results = page._assess_cached(measurements, json.dumps(limits, sort_keys=True), json.dumps({}), "test")
    readings, assessment = results[device]
    report = page._report("Simulated example", view.fleet, device, readings, assessment, page._measured(readings),
                          limits[device], limits, conditions.latest(readings), {}, None, measurements, view)
    pdf = pdf_bytes(report)
    assert pdf[:4] == b"%PDF"
    text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    for words in ("Fleet overview", f"End-of-life limits for {device}", "ESR rise (%)", "150.00",
                  f"Operating conditions for {device}", f"Recorded: {recorded:,.2f}", "Changed conditions in use",
                  "Six-parameter history", "Predicted parameter trends to end of life",
                  "Remaining-life prediction over time"):
        assert words in text, words
    sheets = openpyxl.load_workbook(io.BytesIO(xlsx_bytes(report)))
    assert {"Overview", "Fleet", f"{device} history", "Limits", "Measurements"} <= set(sheets.sheetnames)
    assert "ESR rise (%)" in [cell.value for cell in sheets["Limits"][1]]


def test_monitor_runs_with_only_three_parameters():
    from caplife import operating_conditions as conditions
    from caplife.six_rul import COMMON_SUBSET, example_fleet
    measurements, _ = example_fleet()
    three = measurements[["Cap_ID", "Hours", *COMMON_SUBSET, *conditions.recorded_columns(measurements)]]
    app = open_workspace("Monitor and Predict")
    assert not app.checkbox  # the example has no "only three parameters" option any more
    app.get("file_uploader")[0].upload("three.csv", three.to_csv(index=False).encode(), "text/csv").run()
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
    app.radio(key="caplife_rul_source").set_value("Use simulated example capacitors").run()
    app.selectbox(key="caplife_rul_calibration").set_value("SYN_EXP_C").run()
    app.selectbox(key="caplife_rul_test").set_value("SYN_EXP_B").run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception
    assert any("Passed." in s.value for s in app.success)
    assert any("Remaining-life prediction over time" in m.value for m in app.markdown)
    assert any("Parameters 6 of 6" in m.value for m in app.markdown)  # a tag under 1 · Data used
    assert not app.info  # the example's note is in the Data help, not a box
    # The limits that defined failure start at the defaults; the comparison includes the built-in model.
    assert app.number_input(key="caplife_rul_limit_Capacitance_uF").value == 20.0
    assert app.number_input(key="caplife_rul_limit_ESR_ohm").value == 100.0  # shown as % rise
    models = next(d.value for d in app.dataframe if "Model" in d.value)["Model"].tolist()
    assert models == ["Your trained model", "Built-in model", "Average lifetime only"]
    assert any(e.label == "What your data tested" for e in app.expander)
    # A different limit is a different training run, counted as another setting tried.
    app.number_input(key="caplife_rul_limit_Capacitance_uF").set_value(25.0).run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception
    assert any("2 different settings" in w.value for w in app.warning)


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


def test_each_chart_zooms_on_its_own_and_resets():
    """A drag zooms one chart; a click or tap does not undo it; Reset zoom or a double-click does."""
    from caplife import six_rul_page as page
    app = open_workspace("Monitor and Predict")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    device = app.selectbox(key="caplife_six_device").value
    history = f"caplife_six_zoom_{device}_history_Capacitance_uF"
    rul = f"caplife_six_zoom_{device}_rul"
    # The drag selection is named zoom_0, then zoom_1 after a zoom (a fresh, empty one), and so on.
    drag = lambda start, stop, n=0: {"selection": {f"zoom_{n}": {"Hours": [start, stop]}, "reset": []}}

    def domains(spec):
        found = [spec.get("encoding", {}).get("x", {}).get("scale", {}).get("domain")]
        for layer in spec.get("layer", []):
            found += domains(layer)
        return found

    chart = lambda section, index: app.session_state[page.REPORT_VIEW].sections[section].charts[index][1].to_dict()
    for key, hours in ((history, (1000.0, 5000.0)), (rul, (2000.0, 9000.0))):  # as a drag in the browser
        app.session_state[key] = drag(*hours)
    app.run()
    assert not app.exception, app.exception
    assert [1000.0, 5000.0] in domains(chart(0, 0))        # the zoomed chart
    assert [1000.0, 5000.0] not in domains(chart(0, 1))    # the others keep their range
    assert [2000.0, 9000.0] in domains(chart(2, 0))
    assert {b.key for b in app.button if b.key and b.key.startswith("caplife_six_zoom_")} == \
        {f"{history}_reset", f"{rul}_reset"}
    app.session_state[rul] = {"selection": {"zoom_1": {}, "reset": []}}  # a click or tap: the zoom stays
    app.run()
    assert [2000.0, 9000.0] in domains(chart(2, 0))
    app.session_state[rul] = drag(3000.0, 6000.0, 1)  # zoom again, inside the zoom
    app.run()
    assert [3000.0, 6000.0] in domains(chart(2, 0))
    app.session_state[rul] = {"selection": {"zoom_2": {}, "reset": [{"Hours": 4500.0}]}}  # a double-click
    app.run()
    assert [3000.0, 6000.0] not in domains(chart(2, 0))
    app.button(key=f"{history}_reset").click().run()  # Reset zoom by the title
    assert [1000.0, 5000.0] not in domains(chart(0, 0))


def test_uploaded_file_and_choices_are_kept_when_visiting_other_pages():
    from caplife.six_rul import example_fleet
    app = open_workspace("Monitor and Predict")
    measurements, _ = example_fleet()
    app.get("file_uploader")[0].upload("bench.csv", measurements.to_csv(index=False).encode(), "text/csv").run()
    assert not app.exception, app.exception
    app.selectbox(key="caplife_six_device").set_value("DEMO_04").run()
    app.number_input(key="caplife_six_limit_DEMO_04_Capacitance_uF").set_value(25.0).run()
    for elsewhere in (WORKSPACES["Train The AI"], None, WORKSPACES["My Data"]):  # None is Home
        if elsewhere is None:
            app.sidebar.button(key="caplife_brand_home").click().run()
        else:
            app.sidebar.radio(key=MAIN_KEY).set_value(elsewhere).run()
        app.sidebar.radio(key=MAIN_KEY).set_value(MONITOR).run()
        assert not app.exception, app.exception
        assert any("Using **bench.csv**" in c.value for c in app.caption)  # the upload, kept
        assert app.selectbox(key="caplife_six_device").value == "DEMO_04"
        assert app.number_input(key="caplife_six_limit_DEMO_04_Capacitance_uF").value == 25.0
        assert metric(app, "Capacitors") == "12"
    app.button(key="caplife_six_upload_forget").click().run()
    assert not any("Using **bench.csv**" in c.value for c in app.caption)
    assert not app.metric


def test_train_the_ai_report_pdf_and_excel_match_the_page():
    import io
    from openpyxl import load_workbook
    from pypdf import PdfReader
    from caplife.train_report import pdf_bytes, xlsx_bytes
    app = open_workspace("Train The AI")
    app.radio(key="caplife_rul_source").set_value("Use simulated example capacitors").run()
    app.selectbox(key="caplife_rul_calibration").set_value("SYN_EXP_C").run()
    app.selectbox(key="caplife_rul_test").set_value("SYN_EXP_B").run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception
    view = app.session_state["caplife_rul_report_view"]
    assert view.source == "Simulated example" and view.passed and view.rul_charts
    pdf = pdf_bytes(view)
    assert pdf.startswith(b"%PDF")
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    for heading in ("1 · Data used", "2 · Test on capacitors the model never saw", "3 · Verdict",
                    "4 · Remaining-life forecasts", "Passed."):
        assert heading in text
    sheets = load_workbook(io.BytesIO(xlsx_bytes(view))).sheetnames
    assert sheets[0] == "Overview" and {"Model comparison", "Test results", "Forecasts"} <= set(sheets)
