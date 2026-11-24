"""Trend charts for degradation rates (Analyze measurements): points joined in time order."""
from __future__ import annotations

import altair as alt
import pandas as pd

from caplife.chart_hover import NO_TOOLTIP, hoverable

TEAL, ZERO = "#2f9e8f", "#9fb0bd"


def rate_trend(frame: pd.DataFrame, x: str, y: str, x_title: str, y_title: str, height: int = 230) -> alt.Chart:
    """A line through the values in time order, with points, a zero line and easy-to-hover tooltips."""
    data = frame[[x, y]].dropna().sort_values(x)
    tip = [alt.Tooltip(f"{x}:Q", title=x_title, format=",.0f"), alt.Tooltip(f"{y}:Q", title=y_title, format=".4~g")]
    x_enc = alt.X(f"{x}:Q", title=x_title)
    y_enc = alt.Y(f"{y}:Q", title=y_title)
    zero = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(color=ZERO, strokeDash=[4, 4], opacity=.7).encode(
        y="zero:Q", tooltip=NO_TOOLTIP)
    line = alt.Chart(data).mark_line(color=TEAL, strokeWidth=2, point=alt.OverlayMarkDef(
        color=TEAL, size=34, filled=True)).encode(x=x_enc, y=y_enc, tooltip=tip)
    return alt.layer(zero, line, *hoverable(data, x_enc, y_enc, tip, TEAL)).properties(height=height)
