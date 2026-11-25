"""CapLife AI: monitor capacitor health and predict remaining life, train the AI, and keep your data."""
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


def _forget_outdated_caplife_modules() -> None:
    """Make the next imports load the current ``caplife`` code after a repository update.

    Streamlit Community Cloud pulls an update into the running server and reruns this
    script, but modules imported by an earlier run stay in ``sys.modules`` with their old
    code, so a page could keep showing behaviour that has already been replaced.
    """
    package = sys.modules.get("caplife")
    if package is None:
        return
    fingerprint = getattr(package, "source_fingerprint", None)
    if fingerprint is not None and fingerprint() == getattr(package, "LOADED_FINGERPRINT", None):
        return
    for name in [name for name in sys.modules if name == "caplife" or name.startswith("caplife.")]:
        del sys.modules[name]


_forget_outdated_caplife_modules()

from caplife.accounts import account_bar, render_my_data


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PUBLIC_APP_URL = "https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/"
MONITOR_WORKFLOW = "Six-parameter RUL prediction (demonstration model)"
TRAIN_WORKFLOW = "Train an RUL model on uploaded outcomes"
MY_DATA_WORKFLOW = "My saved datasets"
HOME_WORKFLOW = "Home"  # shown when no workspace is chosen; the CapLife AI brand opens it
WORKFLOWS = {
    MONITOR_WORKFLOW: "Monitor and Predict",
    TRAIN_WORKFLOW: "Train The AI",
    MY_DATA_WORKFLOW: "My Data",
}
MAIN_KEY = "caplife_workflow_v07"

st.set_page_config(
    page_title="CapLife AI | Capacitor health and remaining life",
    page_icon="⚡",
    layout="wide",
)
st.html(ASSETS / "interface.css")
st.html(ASSETS / "hide-repo-buttons.html", unsafe_allow_javascript=True)
st.html(ASSETS / "hide-profile-badge.html", unsafe_allow_javascript=True)

if st.query_params.get("page") == "privacy":
    from caplife.privacy import render as render_privacy
    render_privacy(PUBLIC_APP_URL)
    st.stop()


def _open(workflow: str | None) -> None:
    # None shows Home; uploaded files and other page state are kept.
    st.session_state[MAIN_KEY] = workflow


with st.sidebar:
    st.button("CapLife AI", key="caplife_brand_home", on_click=_open, args=(None,), help="Home")
    st.html(ASSETS / "theme-switch.html", unsafe_allow_javascript=True)
    st.session_state.setdefault(MAIN_KEY, None)  # nothing chosen: Home
    if st.session_state[MAIN_KEY] not in (None, *WORKFLOWS):  # a page from an earlier version
        st.session_state[MAIN_KEY] = None
    st.radio("Workspace", tuple(WORKFLOWS), format_func=WORKFLOWS.__getitem__, key=MAIN_KEY)
    workflow = st.session_state[MAIN_KEY] or HOME_WORKFLOW
    # Arriving on a page from another one: uploaders start empty and use the file kept for them.
    if st.session_state.get("caplife_shown_page") != workflow:
        st.session_state["caplife_shown_page"] = workflow
        st.session_state["caplife_page_visit"] = st.session_state.get("caplife_page_visit", 0) + 1
    st.divider()
    st.markdown(f"[Privacy]({PUBLIC_APP_URL}?page=privacy)")

account_bar(lambda: _open(MY_DATA_WORKFLOW))

if workflow == HOME_WORKFLOW:
    from caplife.home_page import render as render_home
    render_home(ASSETS / "capacitor-hero.webp", _open,
                {"monitor": MONITOR_WORKFLOW, "train": TRAIN_WORKFLOW, "data": MY_DATA_WORKFLOW})
elif workflow == MY_DATA_WORKFLOW:
    render_my_data()
elif workflow == MONITOR_WORKFLOW:
    from caplife.six_rul_page import render as render_monitor
    render_monitor()
elif workflow == TRAIN_WORKFLOW:
    from caplife.rul_page import render as render_train
    render_train()
# Ending the run with st.stop() keeps the choices made on the pages not shown (data source, uploads, limits):
# a run that finishes normally would drop the state of every widget it did not draw.
st.stop()
