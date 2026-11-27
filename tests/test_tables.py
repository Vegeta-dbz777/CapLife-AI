import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from caplife.tables import centered_config

APP = str(Path(__file__).resolve().parents[1] / "app.py")

def upload_example(app):
    """Monitor and Predict with the twelve simulated example capacitors, uploaded as a file."""
    from caplife.six_rul import example_fleet
    measurements, _ = example_fleet()
    app.get("file_uploader")[0].upload("example.csv", measurements.to_csv(index=False).encode(), "text/csv").run()
    return app



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
    upload_example(app)
    assert not app.exception
    fleet = next(d for d in app.dataframe if "Capacitor" in d.value.columns)  # after the file's Columns table
    columns = json.loads(fleet.proto.columns)
    assert columns["Capacitor"]["alignment"] == "center" and columns["Capacitor"]["pinned"] is True
    assert columns["Remaining life, median (h)"]["alignment"] == "center"


def test_sentence_length_text_wraps_instead_of_being_cut():
    from caplife.tables import CHARS_PER_LINE, WRAP_WIDTH, fitted_width, long_text_lengths, wrapped_row_height
    sentence = "Document absolute-vs-gauge pressure, ambient reference, sensor calibration and probe geometry."
    frame = pd.DataFrame({"Cap_ID": ["A", "B"], "Recommendation": [sentence, None], "Short": ["ok", "fine"]})
    assert len(sentence) > CHARS_PER_LINE and set(long_text_lengths(frame)) == {"Recommendation"}
    config = centered_config(frame, {"Cap_ID": st.column_config.TextColumn(pinned=True)}, show_index=False)
    assert config["Recommendation"]["alignment"] == "left" and config["Recommendation"]["width"] == WRAP_WIDTH
    assert config["Short"]["alignment"] == "center" and config["Cap_ID"]["alignment"] == "center"
    assert config["Short"].get("width") is None  # short text keeps Streamlit's own sizing
    height = wrapped_row_height(long_text_lengths(frame))
    assert height is not None and height >= 2 * 20  # room for at least two lines
    # A width chosen by the caller is kept.
    kept = centered_config(frame, {"Recommendation": st.column_config.TextColumn(width="small")}, show_index=False)
    assert kept["Recommendation"]["width"] == "small"
    # Short text only: one-line rows, Streamlit's default height.
    assert wrapped_row_height(long_text_lengths(frame[["Cap_ID", "Short"]])) is None


def test_a_long_entry_far_down_the_column_still_fits():
    from caplife.tables import fitted_width
    # Streamlit sizes columns from the first rows; here those are all a dash.
    frame = pd.DataFrame({"Limits": ["—"] * 30 + ["Leakage current: 3 of 9 missing"]})
    config = centered_config(frame, None, show_index=False)
    assert config["Limits"]["alignment"] == "center"
    assert config["Limits"]["width"] == fitted_width(len("Leakage current: 3 of 9 missing")) >= 31 * 7
