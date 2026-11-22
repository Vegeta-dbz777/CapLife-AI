"""Optional sign-in and per-user saving for the Streamlit app.

Both features switch on only when configured in Streamlit secrets (see
docs/accounts-and-storage.md): an ``[auth]`` section enables sign-in through an
OpenID Connect provider such as Google, and ``[storage] url`` points at a SQL
database. Without them the app runs as a public demo and nothing is stored.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

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


def saving_available() -> bool:
    return current_user() is not None and store() is not None


def sidebar_account() -> None:
    """Sign-in controls, shown only when sign-in is configured."""
    if not auth_configured():
        return
    user = current_user()
    if user is None:
        st.caption("Sign in to save datasets privately and reopen them later.")
        provider = _login_provider()
        st.button("Sign in", key="caplife_sign_in", on_click=st.login, args=(provider,) if provider else ())
    else:
        st.caption(f"Signed in as **{user['email'] or user['name']}**")
        st.button("Sign out", key="caplife_sign_out", on_click=st.logout)


def save_controls(kind: str, files: dict[str, bytes], default_name: str, key: str) -> None:
    """Let a signed-in user keep the current files in their account."""
    if not saving_available():
        if auth_configured() and current_user() is None and storage_url():
            st.caption("Sign in from the sidebar to save this dataset to your account.")
        return
    with st.expander("Save this dataset to your account"):
        name = st.text_input("Name", value=default_name[:80], max_chars=80, key=f"{key}_name")
        if st.button("Save dataset", key=f"{key}_save"):
            try:
                store().save(current_user()["id"], kind, name, files)
            except StorageError as error:
                st.error(str(error))
            else:
                st.success(f"Saved “{name.strip()}”. Open it any time from **My data** or this page.")


@st.cache_data(show_spinner=False, ttl=600, max_entries=16)
def _load_cached(url: str, owner: str, dataset_id: int) -> dict[str, bytes]:
    return _open_store(url).load(owner, dataset_id)


def saved_picker(kind: str, key: str) -> tuple[dict[str, bytes], str] | None:
    """Choose one of the user's saved datasets of this kind; returns its files and name."""
    user, data_store = current_user(), store()
    if user is None or data_store is None:
        return None
    saved = data_store.list(user["id"], kind)
    if saved.empty:
        st.info("You have no saved datasets of this kind yet. Upload one and save it to your account.")
        return None
    timezone = viewer_timezone()
    names = {int(row.id): row.name for row in saved.itertuples(index=False)}
    labels = {int(row.id): f"{row.name} · {local_time(row.created_at, timezone)}"
              for row in saved.itertuples(index=False)}
    chosen = st.selectbox("Saved dataset", list(labels), format_func=labels.get, key=f"{key}_saved")
    try:
        return _load_cached(storage_url(), user["id"], int(chosen)), names[int(chosen)]
    except StorageError as error:
        st.error(str(error))
        return None


def render_my_data() -> None:
    """The My data page: list, open and delete the signed-in user's datasets."""
    st.header("My data")
    user, data_store = current_user(), store()
    if user is None:
        st.info("Sign in from the sidebar to see your saved datasets.")
        return
    if data_store is None:
        st.info("Saving is not set up on this deployment yet.")
        return
    st.write(f"Datasets saved by **{user['email'] or user['name']}**. Only you can see them.")
    saved = data_store.list(user["id"])
    if saved.empty:
        st.info("Nothing saved yet. Upload a dataset on **RUL prediction** or **Train RUL model** and save it.")
    else:
        table = pd.DataFrame({
            "Name": saved["name"], "Type": saved["kind"].map(KINDS),
            "Saved": [local_time(value, viewer_timezone()) for value in saved["created_at"]],
            "Size (KB)": (saved["size_bytes"] / 1024).round(1),
        })
        st.dataframe(table, hide_index=True, width="stretch")
        labels = {int(row.id): f"{row.name} · {KINDS[row.kind]}" for row in saved.itertuples(index=False)}
        chosen = st.selectbox("Dataset", list(labels), format_func=labels.get, key="caplife_mydata_pick")
        files = data_store.load(user["id"], chosen)
        cols = st.columns(len(files) + 1)
        for column, (label, content) in zip(cols, files.items()):
            column.download_button(f"Download {label} CSV", content, f"{label}.csv", mime="text/csv",
                                   key=f"caplife_mydata_dl_{label}")
        if cols[-1].button("Delete this dataset", key="caplife_mydata_delete"):
            data_store.delete(user["id"], chosen)
            _load_cached.clear()
            st.rerun()
    with st.expander("Delete everything"):
        st.write("Permanently remove every dataset saved to your account.")
        confirm = st.checkbox("I understand this cannot be undone", key="caplife_mydata_confirm")
        if st.button("Delete all my data", disabled=not confirm, key="caplife_mydata_delete_all"):
            removed = data_store.delete_all(user["id"])
            _load_cached.clear()
            st.success(f"Deleted {removed} dataset(s).")
    st.caption("Datasets are stored in the database configured for this app, compressed and labelled with your "
               "account ID. Signing out does not delete them; use the buttons above.")
