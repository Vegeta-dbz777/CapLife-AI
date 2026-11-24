import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from caplife.tables import centered_config

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_every_column_is_centred_and_existing_settings_are_kept():
    frame = pd.DataFrame({"Capacitor": ["A"], "Health": [50], "Life (h)": [1200.0], "Note": ["x"], "Hidden": [1]})
    config = centered_config(frame, {
        "Capacitor": st.column_config.TextColumn(pinned=True),
        "Health": st.column_config.ProgressColumn(min_value=0, max_value=100),
        "Life (h)": st.column_config.NumberColumn(format="%.0f"),
        "Note": "Remark",
        "Hidden": None,
    }, show_index=False)
    assert config["Capacitor"]["alignment"] == "center" and config["Capacitor"]["pinned"] is True
    assert config["Life (h)"]["alignment"] == "center" and config["Life (h)"]["type_config"]["format"] == "%.0f"
    assert config["Note"]["alignment"] == "center" and config["Note"]["label"] == "Remark"
    assert "alignment" not in config["Health"] or config["Health"].get("alignment") is None  # bars fill the cell
    assert config["Hidden"] is None  # stays hidden
    assert "_index" not in config
    assert centered_config(frame, None)["_index"]["alignment"] == "center"


def test_rendered_tables_carry_centred_columns():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.radio(key="caplife_workflow_v07").set_value("Six-parameter RUL prediction (demonstration model)").run()
    app.radio(key="caplife_six_source").set_value("Use simulated example capacitors").run()
    assert not app.exception
    columns = json.loads(app.dataframe[0].proto.columns)
    assert columns["Capacitor"]["alignment"] == "center" and columns["Capacitor"]["pinned"] is True
    assert columns["Remaining life, median (h)"]["alignment"] == "center"
