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


@st.cache_data(max_entries=16, show_spinner=False)
def _column_preview(payload: bytes, filename: str, kind: str):
    from caplife.column_match import preview
    return preview(payload, filename, kind)


@st.cache_data(max_entries=16, show_spinner=False)
def _apply_columns(payload: bytes, filename: str, choices: tuple[str, ...], kind: str) -> bytes:
    from caplife.column_match import apply
    return apply(payload, filename, list(choices), kind)


def _choose(store: str, digest: str, editor: str) -> None:
    """Keep the edited "Use as" choices, and rebuild the table from them on the next run."""
    choices = st.session_state[store]["choices"][digest]
    for row, change in st.session_state.get(editor, {}).get("edited_rows", {}).items():
        if "Use as" in change and change["Use as"]:
            choices[int(row)] = change["Use as"]
    st.session_state[store]["round"] += 1
    st.session_state[store]["edited"].add(digest)  # keep the table open while the user works through it


def matched_file(name: str, payload: bytes, kind: str, key: str) -> bytes | None:
    """Let the user say what each column of an uploaded file is, and return the file as CapLife reads it.

    The choices start from a guess based on the column names and are kept for the session, per file. Returns
    None, with the reason shown, until the file has every column CapLife needs."""
    import hashlib

    from caplife.column_match import NOT_USED, TARGETS, problems
    from caplife.real_data import RealDataValidationError

    try:
        frame = _column_preview(payload, name, kind)
    except (RealDataValidationError, ValueError, UnicodeError) as error:
        st.error(f"{name} could not be read: {error}")
        return None
    store = f"{key}_columns"
    state = st.session_state.setdefault(store, {"choices": {}, "round": 0, "edited": set()})
    digest = hashlib.sha256(payload).hexdigest()[:16]
    choices = state["choices"].setdefault(digest, list(frame["Use as"]))
    frame = frame.assign(**{"Use as": choices})
    found = problems(choices, kind)
    used = sum(choice != NOT_USED for choice in choices)
    title = (f"Columns of {name}: {used} of {len(choices)} used" if not found
             else f"Columns of {name}: choose what they are")
    with st.expander(title, icon=":material/view_column:", expanded=bool(found) or digest in state["edited"]):
        editor = f"{key}_columns_{digest}_{state['round']}"
        st.data_editor(
            frame, key=editor, hide_index=True, width="stretch", disabled=["Column in your file", "Example values"],
            on_change=_choose, args=(store, digest, editor),
            column_config={
                "Column in your file": st.column_config.TextColumn(pinned=True),
                "Example values": st.column_config.TextColumn(help="The first values in the column."),
                "Use as": st.column_config.SelectboxColumn(
                    options=[NOT_USED, *(target.label for target in TARGETS[kind])], required=True,
                    help="What this column holds. Pick the unit your values are in: they are converted to "
                         "CapLife's units. Columns marked Not used are left out."),
            })
    if found:
        st.warning(" ".join(found), icon=":material/view_column:")
        return None
    try:
        return _apply_columns(payload, name, tuple(choices), kind)
    except (RealDataValidationError, ValueError) as error:
        st.error(f"{name} could not be read: {error}")
        return None
