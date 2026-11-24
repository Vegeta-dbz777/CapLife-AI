"""Chart hover helpers shared by the RUL pages."""
from __future__ import annotations

import itertools

import altair as alt
import pandas as pd

NO_TOOLTIP = alt.value(None)  # Streamlit gives every mark a raw-number tooltip unless it is switched off
HIT_SIZE = 520  # invisible hover circle around each point: about a 13 px radius
_hover_names = itertools.count()


def hoverable(data: pd.DataFrame, x, y, tooltip: list, color: str) -> list:
    """An easy-to-hit hover area around each point (with its tooltip) and a ring on the hovered point."""
    hovered = alt.selection_point(name=f"hover_{next(_hover_names)}", on="pointerover", clear="pointerout",
                                  empty=False)
    hit = alt.Chart(data).mark_circle(size=HIT_SIZE, opacity=0.001, cursor="pointer").encode(
        x=x, y=y, tooltip=tooltip).add_params(hovered)
    ring = alt.Chart(data).mark_circle(size=170, fill=None, stroke=color, strokeWidth=2.5).encode(
        x=x, y=y, opacity=alt.condition(hovered, alt.value(1), alt.value(0)), tooltip=NO_TOOLTIP)
    return [ring, hit]
