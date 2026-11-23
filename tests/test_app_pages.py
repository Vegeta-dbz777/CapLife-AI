"""Open every workspace of the Streamlit app headlessly and exercise its main actions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
MAIN_KEY, RESEARCH_KEY = "caplife_workflow_v07", "caplife_research_page"

# Shown label -> (sidebar list, internal workspace value), as defined in app.py.
WORKSPACES = {
    "Analyze measurements": (MAIN_KEY, "Experimental measurements (v0.8)"),
    "RUL prediction": (MAIN_KEY, "Six-parameter RUL prediction (demonstration model)"),
    "Train RUL model": (MAIN_KEY, "Train an RUL model on uploaded outcomes"),
    "Forecast benchmarks": (RESEARCH_KEY, "Public-data forecast benchmark"),
    "Neural research": (RESEARCH_KEY, "Public-data neural experiment"),
    "Earlier GRU demo": (RESEARCH_KEY, "Synthetic RUL demonstration (v0.5)"),
}


def open_workspace(label: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert app.sidebar.radio(key=MAIN_KEY).options == [k for k, (g, _) in WORKSPACES.items() if g == MAIN_KEY]
    assert app.sidebar.radio(key=RESEARCH_KEY).options == [k for k, (g, _) in WORKSPACES.items() if g == RESEARCH_KEY]
    group, value = WORKSPACES[label]
    app.sidebar.radio(key=group).set_value(value).run()
    assert not app.exception, app.exception
    return app


def metric(app: AppTest, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


@pytest.mark.parametrize("label", list(WORKSPACES))
def test_every_workspace_opens_without_error(label):
    open_workspace(label)


def test_public_example_loads_eight_capacitors():
    app = open_workspace("Analyze measurements")
    next(b for b in app.button if b.label == "Load public example").click().run()
    assert not app.exception
    assert metric(app, "Capacitors") == "8"
    # 584 NASA rows; each capacitor's two 0 h readings are combined into one.
    assert metric(app, "Measurement rows") == "576"
    assert all(r.label != "CSV format" for r in app.radio)
    assert any("Much of the capacitance loss appeared early" in i.value for i in app.info)
    assert sum("ESR test conditions are only partly documented" in w.value for w in app.warning) == 1


def test_rul_prediction_example_shows_fleet_and_capacitor_detail():
    app = open_workspace("RUL prediction")
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


def test_train_rul_model_example_passes_and_forecasts():
    app = open_workspace("Train RUL model")
    app.radio(key="caplife_rul_source").set_value("Use the synthetic example").run()
    app.selectbox(key="caplife_rul_calibration").set_value("SYN_EXP_C").run()
    app.selectbox(key="caplife_rul_test").set_value("SYN_EXP_B").run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception
    assert any("Passed." in s.value for s in app.success)


def test_earlier_gru_demo_analyzes_a_capacitor():
    app = open_workspace("Earlier GRU demo")
    next(b for b in app.button if b.label == "Analyze capacitor").click().run()
    assert not app.exception
    assert metric(app, "GRU remaining life (DEMO)").endswith(" h")


def test_privacy_page_is_served_on_the_app_domain():
    app = AppTest.from_file(APP, default_timeout=60)
    app.query_params["page"] = "privacy"
    app.run()
    assert not app.exception
    assert app.title[0].value == "Privacy policy"
    text = " ".join(m.value for m in app.markdown)
    assert "name**, **email address" in text and "My data" in text
    assert not app.sidebar.radio  # the policy is a standalone page


def test_choosing_a_page_in_one_sidebar_list_clears_the_other():
    app = open_workspace("Neural research")
    assert app.sidebar.radio(key=MAIN_KEY).value is None
    assert app.header[0].value == "Public-data neural experiment"
    app.sidebar.radio(key=MAIN_KEY).set_value("Six-parameter RUL prediction (demonstration model)").run()
    assert not app.exception
    assert app.sidebar.radio(key=RESEARCH_KEY).value is None
    assert app.header[0].value == "Six-parameter health and remaining-life prediction"


def test_rul_prediction_runs_with_only_three_parameters():
    app = open_workspace("RUL prediction")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    app.checkbox(key="caplife_six_three_only").check().run()
    assert not app.exception, app.exception
    assert any("Using 3 of 6 parameters" in i.value for i in app.info)
    assert metric(app, "Remaining useful life").endswith(" h")
