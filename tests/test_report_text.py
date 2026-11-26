"""Page text as it appears in the PDF and Excel reports."""
import io

import pandas as pd
from openpyxl import load_workbook

from caplife.monitor_report import _markup
from caplife.train_report import TrainView, xlsx_bytes


def test_pdf_markup_keeps_bold_and_line_breaks_and_escapes_the_rest():
    assert _markup("**Forecasts withheld.** Did not pass:\n• a < b") == \
        "<b>Forecasts withheld.</b> Did not pass:<br/>• a &lt; b"


def test_train_workbook_shows_whole_hours_and_plain_text():
    test = pd.DataFrame({"Capacitor": ["A"], "Predicted failure age (h)": [1384.28], "Result": ["Inside range"]})
    models = pd.DataFrame({"Model": ["Neural network"], "Failures inside 80% range": [0.875]})
    view = TrainView(
        source="bench", data_metrics=[("Capacitors measured", "1")], split="Split", tags=[], data_note="",
        excluded={}, model_table=models, test_legend="", test_chart=None, test_table=test, passed=False,
        verdict="**Forecasts withheld.** The neural network did not pass:", reasons=["Too few failures."],
        checks="Checks: at least 3 observed test failures.", warnings=[], forecast_message=None, forecasts=None,
        rul_charts=[], withheld={}, withheld_note="", tables={"Model comparison": models, "Test results": test})
    book = load_workbook(io.BytesIO(xlsx_bytes(view)))
    overview = dict(row for row in book["Overview"].iter_rows(values_only=True) if row[0])
    assert overview["Verdict"] == "Forecasts withheld. The neural network did not pass:"
    assert overview["Reason"] == "Too few failures."
    assert overview["Checks"] == "at least 3 observed test failures."
    age = book["Test results"]["B2"]
    assert age.value == 1384.28 and age.number_format == "#,##0"  # full value, shown as whole hours
    assert book["Model comparison"]["B2"].number_format == "0%"
