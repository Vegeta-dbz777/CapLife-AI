"""Sign-in and saved datasets in the running app, with a stand-in for the identity provider."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import caplife.accounts as accounts
from caplife.rul_example import example_dataset
from caplife.six_rul import example_fleet
from caplife.storage import DatasetStore

APP = str(Path(__file__).resolve().parents[1] / "app.py")
WORKSPACE_KEY = "caplife_workflow_v07"
ALICE = {"id": "https://accounts.example|alice", "email": "alice@example.org", "name": "Alice"}

def upload_example(app):
    """Monitor and Predict with the twelve simulated example capacitors, uploaded as a file."""
    from caplife.six_rul import example_fleet
    measurements, _ = example_fleet()
    app.get("file_uploader")[0].upload("example.csv", measurements.to_csv(index=False).encode(), "text/csv").run()
    return app



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


def test_signed_out_visitors_see_sign_in_and_my_data_asks_them_to_sign_in(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = run_app()
    assert any(b.label == "Sign in" for b in app.button)  # top-right corner, not the sidebar
    assert not any(b.label == "Sign in" for b in app.sidebar.button)
    # The sidebar shows its labels; My Data is always listed, and asks visitors to sign in.
    assert app.sidebar.radio(key=WORKSPACE_KEY).options == ["Monitor and Predict", "Train The AI", "My Data"]
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert any("Sign in" in i.value for i in app.info) and not app.dataframe


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
    rows = [b.key for b in app.button if b.key and b.key.startswith("caplife_mydata_remove_")]
    assert len(rows) == 2  # one row, with its own delete button, per dataset of Alice's
    second = int(database.list(ALICE["id"]).query("name == 'Second'")["id"].iloc[0])
    app.button(key=f"caplife_mydata_remove_{second}").click().run()
    assert not app.exception, app.exception
    assert database.list(ALICE["id"])["name"].tolist() == ["First"]
    assert len([b for b in app.button if b.key and b.key.startswith("caplife_mydata_remove_")]) == 1
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


def test_signing_out_while_on_my_data_asks_to_sign_in_again(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert app.header[0].value == "My Data"
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app.run()
    assert not app.exception, app.exception
    # My Data is always in the sidebar; signed out, it asks to sign in and shows nobody's datasets.
    assert app.sidebar.radio(key=WORKSPACE_KEY).value == "My saved datasets"
    assert any("Sign in" in i.value for i in app.info)
    assert not app.dataframe


def test_account_menu_opens_my_data(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.button(key="caplife_account_my_data").click().run()
    assert not app.exception, app.exception
    assert app.header[0].value == "My Data"
    assert app.sidebar.radio(key=WORKSPACE_KEY).value == "My saved datasets"


PAGES = {"Six-parameter RUL prediction (demonstration model)": "caplife_six_source",
         "Train an RUL model on uploaded outcomes": "caplife_rul_source"}


@pytest.mark.parametrize("workspace", list(PAGES))
def test_saved_datasets_are_offered_to_signed_out_visitors_who_are_asked_to_sign_in(database, monkeypatch, workspace):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value(workspace).run()
    assert "My saved datasets" in app.radio(key=PAGES[workspace]).options
    app.radio(key=PAGES[workspace]).set_value("My saved datasets").run()
    assert not app.exception, app.exception
    assert [i.value for i in app.info] == [accounts.SIGN_IN_TO_SEE]  # the same message as My Data
    assert not app.selectbox and not app.metric


@pytest.mark.parametrize("workspace", list(PAGES))
def test_signed_in_user_without_saved_datasets_is_told_so(database, monkeypatch, workspace):
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value(workspace).run()
    app.radio(key=PAGES[workspace]).set_value("My saved datasets").run()
    assert not app.exception, app.exception
    assert any("no saved datasets of this kind yet" in i.value for i in app.info)


def test_saved_datasets_are_offered_even_without_sign_in_configured():
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    for workspace, key in PAGES.items():
        app.sidebar.radio(key=WORKSPACE_KEY).set_value(workspace).run()
        app.radio(key=key).set_value("My saved datasets").run()
        assert not app.exception, app.exception
        assert any("Sign in" in i.value for i in app.info)


def open_my_data(user=ALICE, monkeypatch=None) -> AppTest:
    monkeypatch.setattr(accounts, "current_user", lambda: user)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert not app.exception, app.exception
    return app


def test_my_data_uploads_and_saves_a_monitor_dataset_without_visiting_the_page(database, monkeypatch):
    app = open_my_data(monkeypatch=monkeypatch)
    assert any("No datasets saved yet" in i.value for i in app.info)
    assert app.button(key="caplife_mydata_upload_save").disabled  # nothing uploaded yet
    measurements, _ = example_fleet()
    app.get("file_uploader")[0].upload("bench.csv", measurements.to_csv(index=False).encode(), "text/csv").run()
    assert app.text_input[0].value == "bench"
    app.button(key="caplife_mydata_upload_save").click().run()
    assert not app.exception, app.exception
    assert any("Saved “bench” for Monitor and Predict" in s.value for s in app.success)
    app.run()
    assert app.success  # stays until closed
    app.button(key="caplife_mydata_saved_close").click().run()
    assert not app.success
    saved = database.list(ALICE["id"])
    assert saved[["name", "kind"]].values.tolist() == [["bench", "six_parameter"]]
    dataset = int(saved["id"].iloc[0])
    assert [b.label for b in app.get("download_button") if b.key.startswith(f"caplife_mydata_dl_{dataset}_")] == \
        ["Download"]
    # It opens from My saved datasets on Monitor and Predict.
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Six-parameter RUL prediction (demonstration model)").run()
    app.radio(key="caplife_six_source").set_value("My saved datasets").run()
    assert next(m.value for m in app.metric if m.label == "Capacitors") == "12"


def test_my_data_saves_a_training_dataset_with_both_files(database, monkeypatch):
    app = open_my_data(monkeypatch=monkeypatch)
    app.radio(key="caplife_mydata_kind").set_value("train_rul").run()
    measurements, outcomes = example_dataset()
    uploaders = app.get("file_uploader")
    assert len(uploaders) == 2
    uploaders[0].upload("lab.csv", measurements.to_csv(index=False).encode(), "text/csv").run()
    assert app.button(key="caplife_mydata_upload_save").disabled  # outcomes still missing
    app.get("file_uploader")[1].upload("lab_outcomes.csv", outcomes.to_csv(index=False).encode(), "text/csv").run()
    app.button(key="caplife_mydata_upload_save").click().run()
    assert not app.exception, app.exception
    assert database.list(ALICE["id"])["kind"].tolist() == ["train_rul"]
    assert [b.label for b in app.get("download_button") if b.key.startswith("caplife_mydata_dl_")] == \
        ["Measurements", "Outcomes"]  # one button per file, on the dataset's own row


def test_my_data_saves_an_excel_workbook(database, monkeypatch):
    import io
    app = open_my_data(monkeypatch=monkeypatch)
    measurements, _ = example_fleet()
    workbook = io.BytesIO()
    measurements.to_excel(workbook, index=False)
    app.get("file_uploader")[0].upload("bench.xlsx", workbook.getvalue(),
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet").run()
    app.button(key="caplife_mydata_upload_save").click().run()
    assert not app.exception, app.exception
    assert not app.error
    dataset = int(database.list(ALICE["id"])["id"].iloc[0])
    assert database.load(ALICE["id"], dataset)["measurements"] == workbook.getvalue()


def test_my_data_refuses_a_file_its_page_cannot_read(database, monkeypatch):
    app = open_my_data(monkeypatch=monkeypatch)
    app.get("file_uploader")[0].upload("notes.csv", b"hello,world\n1,2\n", "text/csv").run()
    assert not app.exception, app.exception
    # Nothing in it is a capacitor ID or a time, so it cannot be saved until its columns are matched.
    assert any("Capacitor ID" in w.value for w in app.warning)
    assert app.button(key="caplife_mydata_upload_save").disabled
    app.get("file_uploader")[0].upload("bad.csv", b"Cap_ID,Hours,Capacitance_uF\nA,0,x\n", "text/csv").run()
    app.button(key="caplife_mydata_upload_save").click().run()
    assert not app.exception, app.exception
    assert any("could not be used" in e.value for e in app.error)
    assert database.list(ALICE["id"]).empty


def test_my_data_sorts_newest_or_oldest_first(database, monkeypatch):
    for name in ("First", "Second", "Third"):
        database.save(ALICE["id"], "six_parameter", name, {"measurements": b"Cap_ID,Hours\nA,0\n"})
    app = open_my_data(monkeypatch=monkeypatch)
    names = lambda: [m.value.strip("*") for m in app.markdown if m.value in ("**First**", "**Second**", "**Third**")]
    assert names() == ["Third", "Second", "First"]  # newest first by default
    app.get("button_group")[0].set_value("oldest").run()
    assert not app.exception, app.exception
    assert names() == ["First", "Second", "Third"]


def train_example(app: AppTest) -> AppTest:
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Train an RUL model on uploaded outcomes").run()
    app.radio(key="caplife_rul_source").set_value("Use simulated example capacitors").run()
    app.radio(key="caplife_rul_split_v2").set_value("One test experiment").run()
    app.selectbox(key="caplife_rul_calibration").set_value("SYN_EXP_C").run()
    app.selectbox(key="caplife_rul_test").set_value("SYN_EXP_B").run()
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception, app.exception
    return app


def test_signed_in_user_saves_a_trained_model_and_sees_it_in_my_data(database, monkeypatch):
    from caplife.storage import MODEL_KIND
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = train_example(run_app())
    assert app.text_input(key="caplife_rul_model_name").value == "Simulated example model"
    app.text_input(key="caplife_rul_model_name").set_value("Lab model").run()
    app.button(key="caplife_rul_model_save").click().run()
    assert any("Saved “Lab model”" in s.value for s in app.success)
    saved = database.list(ALICE["id"], MODEL_KIND)
    assert saved["name"].tolist() == ["Lab model"]
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    assert not app.exception, app.exception
    text = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
    assert "Lab model" in text and "Trained on Simulated example" in text and "Test: missed by" in text
    model_id = int(saved["id"].iloc[0])
    app.button(key=f"caplife_mydata_model_remove_{model_id}").click().run()
    assert database.list(ALICE["id"], MODEL_KIND).empty


def test_signed_out_visitors_are_asked_to_sign_in_to_save_a_model(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = train_example(run_app())
    assert not [b for b in app.button if b.key == "caplife_rul_model_save"]
    assert any("Sign in (top right) to save this model" in c.value for c in app.caption)


@pytest.fixture(scope="module")
def lab_model_files():
    from caplife import fine_tune
    from caplife.rul_workflow import SPLIT_BY_EXPERIMENT, run_rul_workflow
    result = run_rul_workflow(*example_dataset(), SPLIT_BY_EXPERIMENT, "SYN_EXP_C", "SYN_EXP_B")
    info = {"trained_on": "Simulated example", "endpoint": result["audit"]["endpoint"], "sudden": False,
            "parameter_names": ["Capacitance"], "tested": [list(row) for row in result["tested"]],
            "scores": {"trained": {"miss_h": 100.0, "inside": 1.0}, "built_in": {"miss_h": 200.0, "inside": 0.5},
                       "average": {"miss_h": 480.0, "inside": 1.0}}}
    return fine_tune.package(result["model"], info)


def open_monitor(app: AppTest) -> AppTest:
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Six-parameter RUL prediction (demonstration model)").run()
    upload_example(app)
    assert not app.exception, app.exception
    return app


def test_monitor_and_predict_runs_on_a_chosen_trained_model(database, monkeypatch, lab_model_files):
    from caplife.six_rul_page import REPORT_VIEW
    from caplife.storage import MODEL_KIND
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    model_id = database.save(ALICE["id"], MODEL_KIND, "Lab model", lab_model_files)
    app = open_monitor(run_app())
    built_in = app.session_state[REPORT_VIEW]
    assert built_in.model == "Built-in model"
    app.radio(key="caplife_six_model_source").set_value("My trained model").run()
    assert not app.exception, app.exception
    assert app.selectbox(key="caplife_six_model_id").value == model_id
    view = app.session_state[REPORT_VIEW]
    assert view.model == "Lab model (trained on Simulated example)"
    assert app.session_state["caplife_six_model_key"] == f"trained:{ALICE['id']}:{model_id}"
    # Every remaining life on the page now comes from the trained model.
    column = "Remaining life, median (h)"
    assert not view.fleet[column].equals(built_in.fleet[column])
    assert any(e.label == "About “Lab model” and how well it did on its test" for e in app.expander)
    # Back to the built-in model, the page is as before.
    app.radio(key="caplife_six_model_source").set_value("Built-in model").run()
    pd.testing.assert_series_equal(app.session_state[REPORT_VIEW].fleet[column], built_in.fleet[column])


def test_choosing_a_trained_model_signed_out_asks_to_sign_in(database, monkeypatch):
    monkeypatch.setattr(accounts, "current_user", lambda: None)
    app = open_monitor(run_app())
    app.radio(key="caplife_six_model_source").set_value("My trained model").run()
    assert not app.exception, app.exception
    assert any("Sign in (top right) to use the models you trained" in i.value for i in app.info)
    assert app.session_state["caplife_six_model_key"] == "built-in"


def test_an_app_update_gets_a_fresh_store(database, monkeypatch):
    # Streamlit Cloud updates the code without restarting: a store made by the old code must not be reused.
    first = accounts.store()
    assert accounts.store() is first
    monkeypatch.setattr(accounts, "code_version", lambda: "after-an-update")
    assert accounts.store() is not first


def test_a_saved_model_is_trained_again_with_its_data_settings_and_new_outcomes(database, monkeypatch):
    import io
    import json

    from caplife.retrain import training_data
    from caplife.rul_page import RETRAIN, _example_csvs
    from caplife.storage import MODEL_KIND
    monkeypatch.setattr(accounts, "current_user", lambda: ALICE)
    app = run_app()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("Train an RUL model on uploaded outcomes").run()
    app.radio(key="caplife_rul_source").set_value("Use simulated example capacitors").run()
    app.number_input(key="caplife_rul_limit_Capacitance_uF").set_value(25.0).run()
    app = train_example(app)
    app.text_input(key="caplife_rul_model_name").set_value("Lab model").run()
    app.button(key="caplife_rul_model_save").click().run()
    saved = database.list(ALICE["id"], MODEL_KIND)
    model_id = int(saved["id"].iloc[0])
    files = database.load(ALICE["id"], model_id)
    measurements, outcomes = _example_csvs()
    assert training_data(files) == {"measurements": measurements, "outcomes": outcomes}
    settings = json.loads(files["model.json"])["info"]["settings"]
    assert settings["method"] == "experiment" and settings["calibration"] == "SYN_EXP_C"
    assert settings["test"] == "SYN_EXP_B" and settings["limits"]["capacitance_loss_pct"] == 25.0

    # Somewhere else in the meantime the settings change; Train again brings back the model's own.
    app.radio(key="caplife_rul_split_v2").set_value("By capacitor").run()
    app.number_input(key="caplife_rul_limit_Capacitance_uF").set_value(20.0).run()
    app.sidebar.radio(key=WORKSPACE_KEY).set_value("My saved datasets").run()
    app.button(key=f"caplife_mydata_model_retrain_{model_id}").click().run()
    assert not app.exception, app.exception
    assert app.sidebar.radio(key=WORKSPACE_KEY).value == "Train an RUL model on uploaded outcomes"
    assert app.radio(key="caplife_rul_source").value == RETRAIN
    assert app.selectbox(key="caplife_rul_retrain_model").value == model_id
    assert app.radio(key="caplife_rul_split_v2").value == "One test experiment"
    assert app.selectbox(key="caplife_rul_calibration").value == "SYN_EXP_C"
    assert app.number_input(key="caplife_rul_limit_Capacitance_uF").value == 25.0

    # New outcomes: one capacitor that was still running has since failed.
    frame = pd.read_csv(io.BytesIO(outcomes), dtype=str, keep_default_na=False)
    running = frame.index[frame["Failure_by_h"].eq("")][0]
    frame.loc[running, "Failure_by_h"] = str(float(frame.loc[running, "Last_alive_h"]) + 500)
    app.get("file_uploader")[1].upload("new_outcomes.csv", frame.to_csv(index=False).encode(), "text/csv").run()
    assert not app.exception, app.exception
    notes = " ".join(e.markdown[0].value for e in app.expander if e.label.startswith("File read with"))
    assert f"Outcomes: 0 new outcomes, {len(frame)} outcomes updated." in notes
    next(b for b in app.button if b.label == "Train and test the model").click().run()
    assert not app.exception, app.exception
    assert app.text_input(key="caplife_rul_model_name").value == "Lab model"
    assert app.checkbox(key=f"caplife_rul_model_replace_{model_id}").value
    app.button(key="caplife_rul_model_save").click().run()
    assert any("in place of “Lab model”" in s.value for s in app.success)
    saved = database.list(ALICE["id"], MODEL_KIND)
    assert saved["id"].tolist() == [model_id]  # replaced, not added
    kept = training_data(database.load(ALICE["id"], model_id))
    assert kept["measurements"] == measurements
    stored = pd.read_csv(io.BytesIO(kept["outcomes"]), dtype=str, keep_default_na=False)
    failed = stored.set_index("Cap_ID").loc[frame.loc[running, "Cap_ID"], "Failure_by_h"]
    assert float(failed) == float(frame.loc[running, "Failure_by_h"])
    assert len(stored) == len(frame)
