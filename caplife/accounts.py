"""Optional sign-in and per-user saving for the Streamlit app.

Both features switch on only when configured in Streamlit secrets (see
docs/accounts-and-storage.md): an ``[auth]`` section enables sign-in through an
OpenID Connect provider such as Google, and ``[storage] url`` points at a SQL
database. Without them the app runs as a public demo and nothing is stored.
"""
from __future__ import annotations

from pathlib import Path

import os

import pandas as pd
import streamlit as st

from caplife.tables import table

from caplife.storage import KINDS, DatasetStore, StorageError


def _secret(section: str):
    try:
        return st.secrets[section] if section in st.secrets else None
    except Exception:  # No secrets file at all
        return None


def viewer_timezone() -> str | None:
    """The visitor's browser time zone (for example "Europe/Copenhagen"), if the browser reported one."""
    try:
        return st.context.timezone or None
    except Exception:
        return None


def local_time(value, timezone: str | None = None) -> str:
    """A stored UTC time in the visitor's time zone; falls back to UTC, labelled as such."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if timezone:
        try:
            return f"{stamp.tz_convert(timezone):%Y-%m-%d %H:%M}"
        except Exception:  # Unknown zone name from the browser
            pass
    return f"{stamp:%Y-%m-%d %H:%M} UTC"


def auth_configured() -> bool:
    return _secret("auth") is not None


def _login_provider() -> str | None:
    """None for a single provider configured directly under [auth], else the first named provider."""
    auth = _secret("auth")
    if auth is None or "client_id" in auth:
        return None
    named = [key for key, value in auth.items() if hasattr(value, "keys")]
    return named[0] if named else None


def storage_url() -> str | None:
    storage = _secret("storage")
    if storage is not None and storage.get("url"):
        return str(storage["url"])
    return os.environ.get("CAPLIFE_DATABASE_URL") or None


@st.cache_resource(show_spinner=False)
def _open_store(url: str) -> DatasetStore:
    return DatasetStore(url)


def store() -> DatasetStore | None:
    url = storage_url()
    if not url:
        return None
    try:
        return _open_store(url)
    except Exception:  # Unreachable database: run without saving rather than fail the page.
        st.warning("Saved datasets are unavailable right now because the database could not be reached.")
        return None


def current_user() -> dict | None:
    """The signed-in user's stable account ID, email and name, or None."""
    if not auth_configured() or not st.user.get("is_logged_in", False):
        return None
    subject = st.user.get("sub")
    issuer = st.user.get("iss") or "oidc"
    email = st.user.get("email")
    account = f"{issuer}|{subject}" if subject else email
    if not account:
        return None
    return {"id": str(account), "email": email or "", "name": st.user.get("name") or email or "Signed-in user"}


SIGN_IN_TO_SEE = "Sign in (top right) to save datasets and see them here. Only you can see your datasets."


def saving_available() -> bool:
    return current_user() is not None and store() is not None


def account_bar(open_my_data=None) -> None:
    """Sign-in controls in the top-right corner of every page, shown only when sign-in is configured.

    Signed out: a Sign in button. Signed in: an account menu with the email, My Data and Sign out.
    """
    if not auth_configured():
        return
    user = current_user()
    # The control sits in the app's top bar (assets/interface.css), placed by a small script.
    st.html(Path(__file__).resolve().parents[1] / "assets" / "account-placement.html", unsafe_allow_javascript=True)
    with st.container(horizontal=True, horizontal_alignment="right", key="caplife_account_bar"):
        if user is None:
            provider = _login_provider()
            st.button("Sign in", key="caplife_sign_in", icon=":material/login:", on_click=st.login,
                      args=(provider,) if provider else (),
                      help="Sign in to save datasets privately and reopen them later.")
            return
        who = user["email"] or user["name"] or "Account"
        with st.popover(who, icon=":material/account_circle:", key="caplife_account_menu"):
            st.caption(f"Signed in as **{who}**")
            if open_my_data is not None:
                st.button("My Data", key="caplife_account_my_data", icon=":material/folder_open:",
                          on_click=open_my_data, width="stretch")
            st.button("Sign out", key="caplife_sign_out", icon=":material/logout:", on_click=st.logout,
                      width="stretch")


def save_controls(kind: str, files: dict[str, bytes], default_name: str, key: str) -> None:
    """Let a signed-in user keep the current files in their account."""
    if not saving_available():
        if auth_configured() and current_user() is None and storage_url():
            st.caption("Sign in (top right) to save this dataset to your account.")
        return
    with st.expander("Save this dataset to your account"):
        name = st.text_input("Name", value=default_name[:80], max_chars=80, key=f"{key}_name")
        if st.button("Save dataset", key=f"{key}_save"):
            try:
                store().save(current_user()["id"], kind, name, files)
            except StorageError as error:
                st.error(str(error))
            else:
                st.success(f"Saved “{name.strip()}”. Open it any time from **My Data** or this page.")


@st.cache_data(show_spinner=False, ttl=600, max_entries=16)
def _load_cached(url: str, owner: str, dataset_id: int) -> dict[str, bytes]:
    return _open_store(url).load(owner, dataset_id)


def saved_picker(kind: str, key: str) -> tuple[dict[str, bytes], str] | None:
    """Choose one of the user's saved datasets of this kind; returns its files and name.

    Offered to everyone: signed-out visitors are asked to sign in, as on My Data."""
    user = current_user()
    if user is None:
        st.info(SIGN_IN_TO_SEE)
        return None
    data_store = store()
    if data_store is None:
        st.info("Saving is not set up on this deployment yet.")
        return None
    saved = data_store.list(user["id"], kind)
    if saved.empty:
        st.info("You have no saved datasets of this kind yet. Upload one and save it to your account.")
        return None
    timezone = viewer_timezone()
    names = {int(row.id): row.name for row in saved.itertuples(index=False)}
    labels = {int(row.id): f"{row.name} · {local_time(row.created_at, timezone)}"
              for row in saved.itertuples(index=False)}
    chosen = st.selectbox("Saved dataset", list(labels), format_func=labels.get, key=f"{key}_saved", persist_state="session")
    try:
        return _load_cached(storage_url(), user["id"], int(chosen)), names[int(chosen)]
    except StorageError as error:
        st.error(str(error))
        return None


def _check_files(kind: str, files: dict[str, bytes], name: str) -> str | None:
    """Read the files the way their page will; returns why they cannot be used, or None."""
    from caplife.file_reader import read_measurement_file
    from caplife.measurement_review import read_experimental_csv
    from caplife.real_data import RealDataValidationError, validate_real_measurements
    from caplife.rul_workflow import read_outcomes
    try:
        if kind == "six_parameter":
            validate_real_measurements(read_measurement_file(files["measurements"], name)[0])
        else:
            read_experimental_csv(files["measurements"])
            read_outcomes(files["outcomes"], "hours")
    except (RealDataValidationError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeError) as error:
        return str(error)
    return None


def _upload_dataset(user: dict, data_store: DatasetStore) -> None:
    """Upload CSV or Excel files on My Data and save them, without going through a workspace."""
    round_ = st.session_state.setdefault("caplife_mydata_upload_round", 0)  # a new round empties the uploaders
    with st.container(border=True):
        st.subheader("Upload a dataset", help="Files are checked the way their page reads them, then saved to your "
                                              "account. Open them later from **My saved datasets** on that page.")
        kind = st.radio("For", list(KINDS), format_func=KINDS.get, horizontal=True, key="caplife_mydata_kind",
                        persist_state="session")
        types = ["csv", "xlsx"]
        if kind == "six_parameter":
            uploads = {"measurements": st.file_uploader("Measurement file (CSV or Excel)", type=types,
                                                        key=f"caplife_mydata_up_six_{round_}")}
        else:
            left, right = st.columns(2)
            uploads = {"measurements": left.file_uploader("Measurement history (CSV or Excel)", type=types,
                                                          key=f"caplife_mydata_up_meas_{round_}"),
                       "outcomes": right.file_uploader("Outcomes (CSV or Excel)", type=types,
                                                       key=f"caplife_mydata_up_out_{round_}")}
        first = uploads["measurements"]
        name = st.text_input("Name", value=first.name.rsplit(".", 1)[0][:80] if first else "", max_chars=80,
                             key=f"caplife_mydata_up_name_{round_}_{first.name if first else ''}")
        ready = all(uploads.values()) and bool(name.strip())
        if st.button("Save dataset", type="primary", disabled=not ready, key="caplife_mydata_upload_save"):
            files = {label: upload.getvalue() for label, upload in uploads.items()}
            problem = _check_files(kind, files, first.name)
            if problem:
                st.error(f"The file could not be used: {problem}")
                return
            try:
                data_store.save(user["id"], kind, name, files)
            except StorageError as error:
                st.error(str(error))
                return
            st.session_state["caplife_mydata_upload_round"] = round_ + 1
            st.session_state["caplife_mydata_saved_note"] = f"Saved “{' '.join(name.split())}” for {KINDS[kind]}."
            st.rerun()
        note = st.session_state.get("caplife_mydata_saved_note")
        if note:
            # The cross sits inside the green box (assets/interface.css) and closes it.
            with st.container(key="caplife_mydata_saved_box"):
                st.success(note, icon=":material/check_circle:")
                st.button("", icon=":material/close:", type="tertiary", help="Close",
                          key="caplife_mydata_saved_close", on_click=st.session_state.pop,
                          args=("caplife_mydata_saved_note", None))


SORT_ORDERS = {"newest": ":material/arrow_downward: Newest first", "oldest": ":material/arrow_upward: Oldest first"}
FILE_LABELS = {"measurements": "Measurements", "outcomes": "Outcomes"}


def _delete_dataset(owner: str, dataset_id: int) -> None:
    store().delete(owner, dataset_id)
    _load_cached.clear()


def _dataset_row(user: dict, row, timezone) -> None:
    """One saved dataset: its name and details, a download button per file and a delete button."""
    dataset_id = int(row.id)
    files = _load_cached(storage_url(), user["id"], dataset_id)
    with st.container(border=True, horizontal=True, vertical_alignment="center", gap="small",
                      key=f"caplife_mydata_row_{dataset_id}"):
        with st.container(gap=None):
            st.markdown(f"**{row.name}**")
            st.caption(f"For {KINDS[row.kind]} · saved {local_time(row.created_at, timezone)} · "
                       f"{row.size_bytes / 1024:,.1f} KB")
        for label, content in files.items():
            extension = "xlsx" if content[:4] == b"PK\x03\x04" else "csv"
            shown = FILE_LABELS.get(label, label.title()) if len(files) > 1 else "Download"
            st.download_button(shown, content, f"{row.name} - {label}.{extension}", icon=":material/download:",
                               key=f"caplife_mydata_dl_{dataset_id}_{label}", width="content",
                               help=f"Download the {label} file of “{row.name}” as you saved it "
                                    f"({extension.upper()}).")
        with st.popover("", icon=":material/delete:", help=f"Delete “{row.name}”", width="content",
                        key=f"caplife_mydata_del_menu_{dataset_id}"):
            st.write(f"Delete **{row.name}**? This cannot be undone.")
            st.button("Delete", type="primary", icon=":material/delete:", key=f"caplife_mydata_remove_{dataset_id}",
                      on_click=_delete_dataset, args=(user["id"], dataset_id))


def render_my_data() -> None:
    """The My Data page: list, open and delete the signed-in user's datasets."""
    st.header("My Data")
    user, data_store = current_user(), store()
    if user is None:
        st.info(SIGN_IN_TO_SEE)
        return
    if data_store is None:
        st.info("Saving is not set up on this deployment yet.")
        return
    st.write(f"Datasets saved by **{user['email'] or user['name']}**. Only you can see them.")
    saved = data_store.list(user["id"])
    if saved.empty:
        st.info("Nothing saved yet. Upload a dataset below to keep it for later.")
    else:
        timezone = viewer_timezone()
        order = st.segmented_control(
            "Sort", list(SORT_ORDERS), format_func=SORT_ORDERS.get, default="newest", required=True,
            key="caplife_mydata_sort", label_visibility="collapsed", persist_state="session")
        if order == "oldest":  # the store lists newest first
            saved = saved.iloc[::-1]
        for row in saved.itertuples(index=False):
            _dataset_row(user, row, timezone)
    _upload_dataset(user, data_store)
    with st.expander("Delete everything"):
        st.write("Permanently remove every dataset saved to your account.")
        confirm = st.checkbox("I understand this cannot be undone", key="caplife_mydata_confirm")
        if st.button("Delete all my data", disabled=not confirm, key="caplife_mydata_delete_all"):
            removed = data_store.delete_all(user["id"])
            _load_cached.clear()
            st.success(f"Deleted {removed} dataset(s).")
    st.caption("Datasets are stored in the database configured for this app, compressed and labelled with your "
               "account ID. Signing out does not delete them; use the buttons above.")
