"""Sign-in and saved datasets in the running app, with a stand-in for the identity provider."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import caplife.accounts as accounts
from caplife.rul_example import example_dataset
from caplife.six_rul import example_fleet
from caplife.storage import DatasetStore

APP = str(Path(__file__).resolve().parents[1] / "app.py")
WORKSPACE_KEY = "caplife_workflow_v07"
ALICE = {"id": "https://accounts.example|alice", "email": "alice@example.org", "name": "Alice"}


@pytest.fixture()
def database(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'caplife.db'}"
    monkeypatch.setattr(accounts, "storage_url", lambda: url)
    accounts._open_store.clear()
    accounts._load_cached.clear()
    return DatasetStore(url)


def run_app(user=None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.secrets["auth"] = {"redirect_uri": "http://localhost:8501/oauth2callback", "cookie_secret": "test"}
    app.run()
    assert not app.exception, app.exception
    return app


def test_signed_out_visitors_see_sign_in_and_no_my_data(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = run_app()
    assert any(b.label == "Sign in" for b in app.button)  # top-right corner, not the sidebar
    assert not any(b.label == "Sign in" for b in app.sidebar.button)
    assert "My saved datasets" not in app.sidebar.radio(key=WORKSPACE_KEY).options
    assert "My data" not in app.sidebar.radio(key=WORKSPACE_KEY).options


def test_without_auth_settings_the_app_shows_no_account_controls():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert not any(b.label in ("Sign in", "Sign out") for b in app.button)
    assert not any(b.label in ("Sign in", "Sign out") for b in app.sidebar.button)


def test_signed_in_user_reopens_a_saved_six_parameter_dataset(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    measurements, _ = example_fleet()
    database.save(ALICE["id"], "six_parameter", "Bench batch", {"measurements": measurements.to_csv(index=False).encode()})
    database.save("someone-else", "six_parameter", "Not Alice's", {"measurements": b"Cap_ID,Hours\n"})
    app = run_app()
    assert any(b.label == "Sign out" for b in app.button)  # in the top-right account menu
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Six-parameter RUL prediction (demonstration model)").run()
    app.radio(key="caplife_six_source").set_value("My saved datasets").run()
    assert not app.exception, app.exception
    picker = app.selectbox(key="caplife_six_saved")
    assert len(picker.options) == 1 and picker.options[0].startswith("Bench batch")
    assert next(m.value for m in app.metric if m.label == "Capacitors") == "12"


def test_signed_in_user_reopens_a_saved_training_dataset(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    measurements, outcomes = example_dataset()
    database.save(ALICE["id"], "train_rul", "Lab run", {"measurements": measurements.to_csv(index=False).encode(),
                                                        "outcomes": outcomes.to_csv(index=False).encode()})
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Train an RUL model on uploaded outcomes").run()
    app.radio(key="caplife_rul_source").set_value("My saved datasets").run()
    assert not app.exception, app.exception
    assert next(m.value for m in app.metric if m.label == "Capacitors measured") == "37"


def test_my_data_lists_and_deletes_only_the_users_datasets(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    database.save(ALICE["id"], "six_parameter", "First", {"measurements": b"Cap_ID,Hours\nA,0\n"})
    database.save(ALICE["id"], "six_parameter", "Second", {"measurements": b"Cap_ID,Hours\nA,0\n"})
    other = database.save("someone-else", "six_parameter", "Theirs", {"measurements": b"Cap_ID,Hours\n"})
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert not app.exception, app.exception
    assert len(app.selectbox(key="caplife_mydata_pick").options) == 2
    next(b for b in app.button if b.key == "caplife_mydata_delete").click().run()
    assert database.list(ALICE["id"])["name"].tolist() == ["First"]
    app.checkbox(key="caplife_mydata_confirm").check().run()
    next(b for b in app.button if b.key == "caplife_mydata_delete_all").click().run()
    assert database.list(ALICE["id"]).empty
    assert database.list("someone-else")["id"].tolist() == [other]


SAVE_SCRIPT = """
from caplife.accounts import save_controls
save_controls("six_parameter", {"measurements": b"Cap_ID,Hours\\nA,0\\n"}, "bench_upload", "probe")
"""


def test_save_button_stores_the_dataset_under_the_users_account(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = AppTest.from_string(SAVE_SCRIPT, default_timeout=60)
    app.secrets["auth"] = {"redirect_uri": "x", "cookie_secret": "x"}
    app.run()
    app.text_input(key="probe_name").input("My bench run").run()
    next(b for b in app.button if b.key == "probe_save").click().run()
    assert not app.exception, app.exception
    assert any("Saved" in s.value for s in app.success)
    saved = database.list(ALICE["id"])
    assert saved["name"].tolist() == ["My bench run"]
    assert database.load(ALICE["id"], saved["id"].iloc[0]) == {"measurements": b"Cap_ID,Hours\nA,0\n"}


def test_signed_out_visitor_is_asked_to_sign_in_before_saving(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = AppTest.from_string(SAVE_SCRIPT, default_timeout=60)
    app.secrets["auth"] = {"redirect_uri": "x", "cookie_secret": "x"}
    app.run()
    assert any("Sign in (top right)" in c.value for c in app.caption)
    assert not any(b.key == "probe_save" for b in app.button)


def test_sign_in_dependencies_are_installed():
    # Streamlit's st.login needs Authlib's httpx integration; a missing httpx only fails at sign-in time.
    import authlib.integrations.httpx_client  # noqa: F401
    from streamlit.auth_util import validate_auth_credentials  # noqa: F401


def test_signing_out_while_on_my_data_returns_to_the_home_page(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert app.header[0].value == "My data"
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app.run()
    assert not app.exception, app.exception
    assert app.sidebar.radio(key=WORKSPACE_KEY).value is None  # back on Home
    assert any("Choose where to start" in m.value for m in app.markdown)
    assert "My data" not in app.sidebar.radio(key=WORKSPACE_KEY).options


def test_account_menu_opens_my_data(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.button(key="caplife_account_my_data").click().run()
    assert not app.exception, app.exception
    assert app.header[0].value == "My data"
    assert app.sidebar.radio(key=WORKSPACE_KEY).value == "My saved datasets"
