"""Open every workspace of the Streamlit app headlessly and exercise its main actions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
WORKSPACE_KEY = "caplife_workflow_v07"


# Shown label -> internal workspace value, as defined by WORKFLOWS in app.py.
WORKSPACES = {
    "Analyze measurements": "Experimental measurements (v0.8)",
    "RUL prediction": "Six-parameter RUL prediction (demonstration model)",
    "Train RUL model": "Train an RUL model on uploaded outcomes",
    "Forecast benchmarks": "Public-data forecast benchmark",
    "Neural research": "Public-data neural experiment",
    "Earlier GRU demo": "Synthetic RUL demonstration (v0.5)",
}


def open_workspace(label: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    radio = app.sidebar.radio(key=WORKSPACE_KEY)
    assert radio.options == list(WORKSPACES)
    radio.set_value(WORKSPACES[label]).run()
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
    assert metric(app, "Measurement rows") == "584"


def test_rul_prediction_example_shows_fleet_and_capacitor_detail():
    app = open_workspace("RUL prediction")
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    assert not app.exception
    assert metric(app, "Capacitors") == "8"
    assert metric(app, "Remaining useful life").endswith(" h")
    app.selectbox(key="caplife_six_device").set_value("DEMO_04").run()
    assert not app.exception
    assert metric(app, "Overall health") == "26%"


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
