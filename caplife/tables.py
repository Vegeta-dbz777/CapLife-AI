"""Every table in the app goes through ``table``: ``st.dataframe`` with all cells centred.

Each column (and the index, when it is shown) is centred. A column's existing settings (number
format, pinning, label) are kept. Progress-bar columns fill their cell, so they are left as they are.

Streamlit sizes a column from its first rows only, so a longer entry further down (or any
sentence-length text) would be cut off. Text columns therefore get a width that fits their longest
entry and header; text too long for one line is left-aligned, gets ``WRAP_WIDTH`` and the rows are
made tall enough for it to wrap.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

CENTER = "center"
FIT_FROM = 12           # characters: shorter text always fits Streamlit's own column sizing
CHAR_WIDTH, CELL_PADDING = 7.6, 30   # pixels, a generous estimate for the app's 14 px font
WRAP_WIDTH = 480        # pixels given to a wrapping column (it can grow to fill the table)
CHARS_PER_LINE = 62     # text longer than this wraps; a conservative fit for WRAP_WIDTH
LINE_HEIGHT, ROW_PADDING, MAX_LINES = 20, 16, 6


def _columns(data) -> list:
    frame = getattr(data, "data", data)  # a pandas Styler wraps its DataFrame as .data
    if isinstance(frame, pd.Series):
        return [frame.name if frame.name is not None else "value"]
    return list(getattr(frame, "columns", []))


def text_lengths(data) -> dict:
    """Longest entry (or header) of each text column, in characters, where it is ``FIT_FROM`` or more."""
    frame = getattr(data, "data", data)
    if not isinstance(frame, pd.DataFrame):
        return {}
    lengths = {}
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object or pd.api.types.is_string_dtype(column):
            longest = int(column.dropna().astype(str).str.len().max()) if column.notna().any() else 0
            longest = max(longest, len(str(name)))
            if longest >= FIT_FROM:
                lengths[name] = longest
    return lengths


def long_text_lengths(data) -> dict:
    """Text columns whose longest entry needs more than one line."""
    return {name: length for name, length in text_lengths(data).items() if length > CHARS_PER_LINE}


def fitted_width(length: int) -> int:
    """Pixels for one line of ``length`` characters (up to ``WRAP_WIDTH``)."""
    return min(WRAP_WIDTH, math.ceil(length * CHAR_WIDTH + CELL_PADDING))


def wrapped_row_height(lengths: dict) -> int | None:
    """Row height that fits the longest wrapped text, or None when nothing needs wrapping."""
    if not lengths:
        return None
    lines = min(MAX_LINES, max(math.ceil(length / CHARS_PER_LINE) for length in lengths.values()))
    return lines * LINE_HEIGHT + ROW_PADDING if lines > 1 else None


def centered_config(data, column_config: dict | None = None, show_index: bool = True) -> dict:
    """``column_config`` with every column centred, keeping whatever each column already had.

    Text columns get a width that fits their longest entry unless their configuration already sets
    one; text that wraps is left-aligned.
    """
    config = dict(column_config or {})
    names = [str(name) if not isinstance(name, (str, int)) else name for name in _columns(data)]
    text = text_lengths(data)
    if show_index:
        names.append("_index")
    for name in names:
        current = config.get(name)
        if current is None and name in config:  # None hides a column: leave it hidden
            continue
        settings = {"alignment": CENTER}
        if name in text:  # wide enough for the longest entry; text that wraps reads left-aligned
            wraps = text[name] > CHARS_PER_LINE
            settings = {"alignment": "left" if wraps else CENTER, "width": fitted_width(text[name])}
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


def whole_hours(data, column_config: dict | None) -> tuple:
    """Numbers in hours (a column named "... (h)") as whole hours with thousands separators, as the cards show them.

    A column whose own settings give another number format keeps it.
    """
    if not isinstance(data, pd.DataFrame):
        return data, column_config
    hours = [name for name in data.columns
             if str(name).endswith("(h)") and pd.api.types.is_numeric_dtype(data[name])]
    if not hours:
        return data, column_config
    data, config = data.copy(), dict(column_config or {})
    for name in hours:
        data[name] = data[name].round().astype("Int64")
        own = config.get(name)
        if own is None and name in config:  # None hides a column: leave it hidden
            continue
        if own is None or isinstance(own, str):  # nothing set, or a plain label
            config[name] = st.column_config.NumberColumn(own, format="localized")
        elif isinstance(own, dict) and (own.get("type_config") or {}).get("format") in (None, "%.0f"):
            config[name] = {**own, "type_config": {**(own.get("type_config") or {"type": "number"}),
                                                   "format": "localized"}}
    return data, config


def whole_percents(data, column_config: dict | None) -> tuple:
    """Shares shown as percent ("percent" format: 0.875 shows as 87.5%) as whole percents (88%), as the cards
    show them."""
    if not isinstance(data, pd.DataFrame) or not column_config:
        return data, column_config
    shares = [name for name, own in column_config.items()
              if isinstance(own, dict) and (own.get("type_config") or {}).get("format") == "percent"
              and name in data.columns and pd.api.types.is_numeric_dtype(data[name])]
    if not shares:
        return data, column_config
    data, config = data.copy(), dict(column_config)
    for name in shares:
        data[name] = data[name] * 100
        config[name] = {**config[name], "type_config": {**config[name]["type_config"], "format": "%.0f%%"}}
    return data, config


def table(data=None, *args, column_config: dict | None = None, hide_index: bool | None = None, **kwargs):
    """``st.dataframe`` with every column centred, hours and percents shown whole and sentence-length text
    wrapped."""
    data, column_config = whole_hours(data, column_config)
    data, column_config = whole_percents(data, column_config)
    config = centered_config(data, column_config, show_index=not hide_index)
    if kwargs.get("row_height") is None:
        kwargs["row_height"] = wrapped_row_height(long_text_lengths(data))
    return st.dataframe(data, *args, column_config=config, hide_index=hide_index, **kwargs)
