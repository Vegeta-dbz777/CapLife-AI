"""Shared Streamlit pieces for measurement uploads: accepted file types and reading notes."""
from __future__ import annotations

import streamlit as st

UPLOAD_TYPES = ["csv", "xlsx"]


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def show_notes(notes: list[str]) -> None:
    """How the file was read: every adjustment, listed in a closed expander so nothing happens silently."""
    if notes:
        with st.expander(f"File read with {_plural(len(notes), 'adjustment')}", icon=":material/tune:"):
            st.markdown("\n".join(f"- {note}" for note in notes))


def status_row(passed: str, warnings: list[str], file_notes: list[str] | None = None) -> None:
    """One tidy line of status tags (checks passed, notes to review, file adjustments) and one closed
    expander with the notes and adjustments, instead of a stack of message boxes."""
    file_notes = file_notes or []
    row = st.container(horizontal=True, vertical_alignment="center")
    row.badge("Basic checks passed", icon=":material/check_circle:", color="green", help=passed)
    if warnings:
        row.badge(f"{_plural(len(warnings), 'note')} to review", icon=":material/warning:", color="orange",
                  help="Open Notes below to read them.")
    if file_notes:
        row.badge(f"File read with {_plural(len(file_notes), 'adjustment')}", icon=":material/tune:", color="blue",
                  help="Open Notes below to see how columns and values were read.")
    if warnings or file_notes:
        parts = []
        if warnings:
            parts.append("**To review**\n" + "\n".join(f"- {warning}" for warning in warnings))
        if file_notes:
            parts.append("**How the file was read**\n" + "\n".join(f"- {note}" for note in file_notes))
        with st.expander("Notes", icon=":material/info:"):
            st.markdown("\n\n".join(parts))
