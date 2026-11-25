"""Chart hover helpers shared by the RUL pages."""
from __future__ import annotations

import itertools

import altair as alt
import pandas as pd

NO_TOOLTIP = alt.value(None)  # Streamlit gives every mark a raw-number tooltip unless it is switched off
HIT_SIZE = 520  # invisible hover circle around each point: about a 13 px radius
ON_LINE_HIT_SIZE = 1250  # points on a vertical line: about 20 px, so the point wins over the line
_hover_names = itertools.count()


def restart_names() -> None:
    """Number the next chart's hover selections from 0 again. Names only need to be unique within a chart;
    the same names on every run keep an unchanged chart identical, so the browser does not rebuild it
    (a rebuild in the middle of a double-click loses the double-click)."""
    global _hover_names
    _hover_names = itertools.count()


def hoverable(data: pd.DataFrame, x, y, tooltip: list, color: str, on_line: bool = False) -> list:
    """An easy-to-hit hover area around each point (with its tooltip) and a ring on the hovered point.

    ``on_line`` is for points that sit on a vertical line (the latest reading, an end of life): a larger
    hover area and ring, so a small slip of the cursor does not select the line instead."""
    hovered = alt.selection_point(name=f"hover_{next(_hover_names)}", on="pointerover", clear="pointerout",
                                  empty=False)
    hit = alt.Chart(data).mark_circle(size=ON_LINE_HIT_SIZE if on_line else HIT_SIZE, opacity=0.001,
                                      cursor="pointer").encode(x=x, y=y, tooltip=tooltip).add_params(hovered)
    ring = alt.Chart(data).mark_circle(size=320 if on_line else 170, fill=None, stroke=color,
                                       strokeWidth=3 if on_line else 2.5).encode(
        x=x, y=y, opacity=alt.condition(hovered, alt.value(1), alt.value(0)), tooltip=NO_TOOLTIP)
    return [ring, hit]


def hoverable_rule(data: pd.DataFrame, x, tooltip: list, color: str) -> list:
    """A vertical line made easy to hover (about 14 px wide) that glows in its colour while hovered."""
    hovered = alt.selection_point(name=f"hover_{next(_hover_names)}", on="pointerover", clear="pointerout",
                                  empty=False)
    hit = alt.Chart(data).mark_rule(strokeWidth=14, opacity=0.001, cursor="pointer").encode(
        x=x, tooltip=tooltip).add_params(hovered)
    glow = alt.Chart(data).mark_rule(color=color, strokeWidth=9).encode(
        x=x, opacity=alt.condition(hovered, alt.value(0.35), alt.value(0)), tooltip=NO_TOOLTIP)
    return [glow, hit]


def points_over_lines(hovers: list) -> list:
    """Hover layers with the vertical lines' first, so a point on a line is hovered before the line."""
    is_rule = lambda layer: getattr(layer.mark, "type", layer.mark) == "rule"
    return [layer for layer in hovers if is_rule(layer)] + [layer for layer in hovers if not is_rule(layer)]
