"""Every table in the app goes through ``table``: ``st.dataframe`` with all cells centred.

Each column (and the index, when it is shown) is centred. A column's existing settings (number
format, pinning, label) are kept. Progress-bar columns fill their cell, so they are left as they are.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

CENTER = "center"


def _columns(data) -> list:
    frame = getattr(data, "data", data)  # a pandas Styler wraps its DataFrame as .data
    if isinstance(frame, pd.Series):
        return [frame.name if frame.name is not None else "value"]
    return list(getattr(frame, "columns", []))


def centered_config(data, column_config: dict | None = None, show_index: bool = True) -> dict:
    """``column_config`` with every column centred, keeping whatever each column already had."""
    config = dict(column_config or {})
    names = [str(name) if not isinstance(name, (str, int)) else name for name in _columns(data)]
    if show_index:
        names.append("_index")
    for name in names:
        current = config.get(name)
        if current is None and name in config:  # None hides a column: leave it hidden
            continue
        if isinstance(current, str):  # a plain label
            config[name] = st.column_config.Column(current, alignment=CENTER)
        elif isinstance(current, dict):
            if (current.get("type_config") or {}).get("type") != "progress":
                config[name] = {**current, "alignment": CENTER}
        else:
            config[name] = st.column_config.Column(alignment=CENTER)
    return config


def table(data=None, *args, column_config: dict | None = None, hide_index: bool | None = None, **kwargs):
    """``st.dataframe`` with every column centred."""
    config = centered_config(data, column_config, show_index=not hide_index)
    return st.dataframe(data, *args, column_config=config, hide_index=hide_index, **kwargs)
