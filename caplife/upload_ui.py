"""Shared Streamlit pieces for measurement uploads: file settings and reading notes."""
from __future__ import annotations

import streamlit as st

from caplife.file_reader import TIME_UNITS

UPLOAD_TYPES = ["csv", "xlsx"]
PRESSURE_CHOICES = {"Detect automatically": "auto", "Absolute": "absolute", "Gauge (relative to atmosphere)": "gauge"}


def file_settings(key: str, time_help: str | None = None) -> tuple[str, str]:
    """Time unit and pressure type of an uploaded file; returns (time_unit, pressure)."""
    with st.expander("File settings: time unit and pressure"):
        left, right = st.columns(2)
        unit = left.selectbox(
            "Time column is in", list(TIME_UNITS), key=f"{key}_time_unit",
            help=time_help or "Used when the column name does not say. A column named Days or Weeks is converted "
                              "to hours automatically.")
        pressure = right.selectbox(
            "Pressure column is", list(PRESSURE_CHOICES), key=f"{key}_pressure",
            help="Absolute pressure is about 101.3 kPa at the start. Gauge pressure starts near 0 kPa and is "
                 "converted by adding 101.3 kPa. Detect automatically treats a capacitor whose first reading is "
                 "below 80 kPa as gauge.")
    return unit, PRESSURE_CHOICES[pressure]


def show_notes(notes: list[str]) -> None:
    """How the file was read: every adjustment, so nothing happens silently."""
    if notes:
        st.info("**File read with these adjustments:**\n" + "\n".join(f"- {note}" for note in notes))
