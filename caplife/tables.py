"""Every table in the app goes through ``table``: ``st.dataframe`` with all cells centred.

Each column (and the index, when it is shown) is centred. A column's existing settings (number
format, pinning, label) are kept. Progress-bar columns fill their cell, so they are left as they are.

Sentence-length text (recommendations, notes) would be cut off in a one-line cell, so such columns
get a fixed minimum width; text too long for one line is left-aligned and the rows are made tall
enough for it to wrap.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

CENTER = "center"
LONG_TEXT = 48          # characters: longer text gets a wider column (and wraps past CHARS_PER_LINE)
WRAP_WIDTH = 480        # pixels given to a wrapping column (it can grow to fill the table)
CHARS_PER_LINE = 62     # a conservative fit for WRAP_WIDTH
LINE_HEIGHT, ROW_PADDING, MAX_LINES = 20, 16, 6


def _columns(data) -> list:
    frame = getattr(data, "data", data)  # a pandas Styler wraps its DataFrame as .data
    if isinstance(frame, pd.Series):
        return [frame.name if frame.name is not None else "value"]
    return list(getattr(frame, "columns", []))


def long_text_lengths(data) -> dict:
    """Longest text in each column whose text runs past ``LONG_TEXT`` characters."""
    frame = getattr(data, "data", data)
    if not isinstance(frame, pd.DataFrame):
        return {}
    lengths = {}
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object or pd.api.types.is_string_dtype(column):
            longest = int(column.dropna().astype(str).str.len().max() or 0) if column.notna().any() else 0
            if longest > LONG_TEXT:
                lengths[name] = longest
    return lengths


def wrapped_row_height(lengths: dict) -> int | None:
    """Row height that fits the longest wrapped text, or None when nothing needs wrapping."""
    if not lengths:
        return None
    lines = min(MAX_LINES, max(math.ceil(length / CHARS_PER_LINE) for length in lengths.values()))
    return lines * LINE_HEIGHT + ROW_PADDING if lines > 1 else None


def centered_config(data, column_config: dict | None = None, show_index: bool = True) -> dict:
    """``column_config`` with every column centred, keeping whatever each column already had.

    Sentence-length text columns get ``WRAP_WIDTH`` unless their configuration already sets a width;
    those long enough to wrap are left-aligned.
    """
    config = dict(column_config or {})
    names = [str(name) if not isinstance(name, (str, int)) else name for name in _columns(data)]
    long_text = long_text_lengths(data)
    if show_index:
        names.append("_index")
    for name in names:
        current = config.get(name)
        if current is None and name in config:  # None hides a column: leave it hidden
            continue
        settings = {"alignment": CENTER}
        if name in long_text:  # wide enough for one line of CHARS_PER_LINE; text that wraps reads left-aligned
            settings = {"alignment": "left" if long_text[name] > CHARS_PER_LINE else CENTER, "width": WRAP_WIDTH}
        if isinstance(current, str):  # a plain label
            config[name] = st.column_config.Column(current, **settings)
        elif isinstance(current, dict):
            if (current.get("type_config") or {}).get("type") != "progress":
                if current.get("width") is not None:
                    settings.pop("width", None)
                config[name] = {**current, **settings}
        else:
            config[name] = st.column_config.Column(**settings)
    return config


def table(data=None, *args, column_config: dict | None = None, hide_index: bool | None = None, **kwargs):
    """``st.dataframe`` with every column centred and sentence-length text wrapped."""
    config = centered_config(data, column_config, show_index=not hide_index)
    if kwargs.get("row_height") is None:
        kwargs["row_height"] = wrapped_row_height(long_text_lengths(data))
    return st.dataframe(data, *args, column_config=config, hide_index=hide_index, **kwargs)
