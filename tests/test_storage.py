import os

import pytest
from sqlalchemy import delete

from caplife.storage import (
    MAX_DATASETS_PER_USER, MAX_MODELS_PER_USER, MODEL_KIND, DatasetStore, StorageError, datasets,
)

CSV = b"Cap_ID,Hours,Capacitance_uF\nA,0,100\nA,10,99\n"
# Set CAPLIFE_TEST_POSTGRES_URL (as the GitHub workflow does) to also run against PostgreSQL.
BACKENDS = ["sqlite"] + (["postgresql"] if os.environ.get("CAPLIFE_TEST_POSTGRES_URL") else [])


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path):
    if request.param == "sqlite":
        return DatasetStore(f"sqlite:///{tmp_path / 'caplife.db'}")
    data_store = DatasetStore(os.environ["CAPLIFE_TEST_POSTGRES_URL"])
    with data_store.engine.begin() as connection:
        connection.execute(delete(datasets))
    return data_store


def test_saved_files_round_trip_and_list_newest_first(store):
    first = store.save("alice", "six_parameter", "  Batch   one ", {"measurements": CSV})
    second = store.save("alice", "train_rul", "Batch two", {"measurements": CSV, "outcomes": b"x,y\n1,2\n"})
    listed = store.list("alice")
    assert listed["id"].tolist() == [second, first]
    assert listed["name"].tolist() == ["Batch two", "Batch one"]
    assert store.load("alice", first) == {"measurements": CSV}
    assert store.list("alice", "six_parameter")["id"].tolist() == [first]


def test_users_cannot_see_open_or_delete_each_others_data(store):
    dataset = store.save("alice", "six_parameter", "Private", {"measurements": CSV})
    assert store.list("bob").empty
    with pytest.raises(StorageError, match="another account"):
        store.load("bob", dataset)
    assert store.delete("bob", dataset) is False
    assert store.delete_all("bob") == 0
    assert store.load("alice", dataset) == {"measurements": CSV}


def test_delete_and_delete_all_remove_only_the_owners_data(store):
    keep = store.save("bob", "six_parameter", "Bob's", {"measurements": CSV})
    one = store.save("alice", "six_parameter", "One", {"measurements": CSV})
    store.save("alice", "six_parameter", "Two", {"measurements": CSV})
    assert store.delete("alice", one) is True
    assert store.list("alice")["name"].tolist() == ["Two"]
    assert store.delete_all("alice") == 1
    assert store.list("alice").empty and store.list("bob")["id"].tolist() == [keep]


def test_invalid_saves_are_refused_with_clear_messages(store):
    with pytest.raises(StorageError, match="Sign in"):
        store.save("", "six_parameter", "x", {"measurements": CSV})
    with pytest.raises(StorageError, match="name"):
        store.save("alice", "six_parameter", "   ", {"measurements": CSV})
    with pytest.raises(StorageError, match="kind"):
        store.save("alice", "other", "x", {"measurements": CSV})
    for index in range(MAX_DATASETS_PER_USER):
        store.save("alice", "six_parameter", f"n{index}", {"measurements": CSV})
    with pytest.raises(StorageError, match=f"up to {MAX_DATASETS_PER_USER}"):
        store.save("alice", "six_parameter", "one too many", {"measurements": CSV})


def test_excel_and_windows_encoded_files_are_kept_byte_for_byte(store):
    workbook = b"PK\x03\x04" + bytes(range(256))
    windows_csv = "Cap_ID,Hours,Température\nA,0,85\n".encode("cp1252")
    dataset = store.save("alice", "train_rul", "Mixed", {"measurements": workbook, "outcomes": windows_csv})
    assert store.load("alice", dataset) == {"measurements": workbook, "outcomes": windows_csv}


def test_trained_models_have_their_own_allowance(store):
    weights = bytes(range(256)) * 4  # binary, like saved network weights
    for index in range(MAX_DATASETS_PER_USER):
        store.save("alice", "six_parameter", f"n{index}", {"measurements": CSV})
    # A full set of datasets does not stop a model being saved, and models stop at their own limit.
    for index in range(MAX_MODELS_PER_USER):
        store.save("alice", MODEL_KIND, f"m{index}", {"model.pt": weights})
    with pytest.raises(StorageError, match=f"up to {MAX_MODELS_PER_USER} trained models"):
        store.save("alice", MODEL_KIND, "one too many", {"model.pt": weights})
    listed = store.list("alice", MODEL_KIND)
    assert len(listed) == MAX_MODELS_PER_USER
    assert store.load("alice", int(listed["id"].iloc[0])) == {"model.pt": weights}


def test_a_model_is_replaced_in_place_and_only_by_its_owner(store):
    from caplife.storage import MODEL_KIND, StorageError
    first = store.save("alice", MODEL_KIND, "Lab model", {"model.json": b"{}"})
    assert store.replace("alice", first, "Lab model v2", {"model.json": b'{"v": 2}'}) == first
    saved = store.list("alice", MODEL_KIND)
    assert saved["id"].tolist() == [first] and saved["name"].tolist() == ["Lab model v2"]
    assert store.load("alice", first) == {"model.json": b'{"v": 2}'}
    with pytest.raises(StorageError):
        store.replace("mallory", first, "Mine now", {"model.json": b"{}"})
    assert store.load("alice", first) == {"model.json": b'{"v": 2}'}
