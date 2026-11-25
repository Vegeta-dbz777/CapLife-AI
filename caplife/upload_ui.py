"""Shared Streamlit pieces for measurement uploads: accepted file types, uploads kept across pages,
and reading notes."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

UPLOAD_TYPES = ["csv", "xlsx"]


@dataclass(frozen=True)
class KeptFile:
    """An uploaded file kept in the session, read like Streamlit's UploadedFile."""
    name: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def getvalue(self) -> bytes:
        return self.data


PAGE_VISIT = "caplife_page_visit"  # counts arrivals on a page from another page (set in app.py)


def kept_file_uploader(label: str, key: str, types=UPLOAD_TYPES) -> KeptFile | None:
    """A file uploader whose file stays with the session when the user visits another page.

    Streamlit does not keep an uploader's file reliably once its page is left: back on the page the
    browser shows an empty uploader, and it may report that as the file being removed. So the file is
    kept here, and the uploader gets a fresh key on every visit to the page. Back on the page the
    uploader is empty and the kept file is used, with a note and a Remove button; uploading another
    file replaces it."""
    store = f"{key}_kept"
    upload = st.file_uploader(label, type=types, key=f"{key}_{st.session_state.get(PAGE_VISIT, 0)}")
    if upload is not None:
        kept = st.session_state.get(store)
        if kept is None or kept.name != upload.name or kept.size != upload.size:
            st.session_state[store] = KeptFile(upload.name, upload.getvalue())
        return st.session_state[store]
    kept = st.session_state.get(store)
    if kept is None:
        return None
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    row.caption(f"Using **{kept.name}** ({kept.size / 1024:,.1f} KB), uploaded earlier.", width="content")
    row.button("Remove", key=f"{key}_forget", type="tertiary", icon=":material/close:",
               on_click=st.session_state.pop, args=(store, None), help="Forget this file and upload another.")
    return kept


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
