"""Private per-user storage of uploaded datasets in a SQL database.

Every read and delete is filtered by the owner's account ID, so one user can
never list, open or remove another user's datasets. Files are stored
gzip-compressed. The module works with any SQLAlchemy URL: PostgreSQL (for
example Neon or Supabase) in production, SQLite for local development and tests.
"""
from __future__ import annotations

from base64 import b64decode, b64encode
from datetime import datetime, timezone
import gzip
import json

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Integer, LargeBinary, MetaData, String, Table, create_engine, delete, func, insert,
    select, update,
)

MAX_DATASETS_PER_USER = 25
MAX_MODELS_PER_USER = 10
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_NAME_LENGTH = 80
KINDS = {"six_parameter": "Monitor and Predict", "train_rul": "Train The AI"}  # the page each kind opens on
MODEL_KIND = "trained_model"  # a model trained on Train The AI, kept beside the datasets

_metadata = MetaData()
datasets = Table(
    "caplife_datasets", _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("owner", String(320), nullable=False, index=True),
    Column("kind", String(32), nullable=False),
    Column("name", String(MAX_NAME_LENGTH), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("payload", LargeBinary, nullable=False),
)


def _encode(content: bytes) -> str | dict[str, str]:
    """UTF-8 text is kept as text (as datasets were always stored); anything else, such as an Excel
    workbook or a CSV in a Windows encoding, is kept byte for byte."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"base64": b64encode(content).decode("ascii")}


def _decode(value: str | dict[str, str]) -> bytes:
    return b64decode(value["base64"]) if isinstance(value, dict) else value.encode("utf-8")


class StorageError(ValueError):
    """A request the store refuses, with a message suitable for the user."""


class DatasetStore:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True)
        _metadata.create_all(self.engine)

    def save(self, owner: str, kind: str, name: str, files: dict[str, bytes]) -> int:
        if not owner:
            raise StorageError("Sign in to save datasets.")
        if kind not in KINDS and kind != MODEL_KIND:
            raise StorageError(f"Unknown dataset kind: {kind}")
        is_model = kind == MODEL_KIND
        name = " ".join(name.split())[:MAX_NAME_LENGTH]
        if not name:
            raise StorageError("Give the model a name." if is_model else "Give the dataset a name.")
        if not files or any(len(content) > MAX_FILE_BYTES for content in files.values()):
            raise StorageError("Each saved file must be between 1 byte and 8 MB.")
        payload = gzip.compress(json.dumps({label: _encode(content) for label, content in files.items()}).encode())
        with self.engine.begin() as connection:
            # Models and datasets have their own allowances, so saving one never crowds out the other.
            same = (datasets.c.kind == MODEL_KIND) if is_model else (datasets.c.kind != MODEL_KIND)
            count = connection.execute(
                select(func.count()).select_from(datasets).where(datasets.c.owner == owner, same)).scalar_one()
            if is_model and count >= MAX_MODELS_PER_USER:
                raise StorageError(f"You can keep up to {MAX_MODELS_PER_USER} trained models. Delete one in "
                                   "My Data to save another.")
            if not is_model and count >= MAX_DATASETS_PER_USER:
                raise StorageError(f"You can keep up to {MAX_DATASETS_PER_USER} datasets. Delete one to save another.")
            result = connection.execute(insert(datasets).values(
                owner=owner, kind=kind, name=name, created_at=datetime.now(timezone.utc),
                size_bytes=sum(len(content) for content in files.values()), payload=payload))
            return int(result.inserted_primary_key[0])

    def replace(self, owner: str, dataset_id: int, name: str, files: dict[str, bytes]) -> int:
        """Put new files and a new name in place of a saved item of the same owner, keeping its ID and kind
        (used when a model is trained again). The save time becomes now."""
        name = " ".join(name.split())[:MAX_NAME_LENGTH]
        if not name:
            raise StorageError("Give the model a name.")
        if not files or any(len(content) > MAX_FILE_BYTES for content in files.values()):
            raise StorageError("Each saved file must be between 1 byte and 8 MB.")
        payload = gzip.compress(json.dumps({label: _encode(content) for label, content in files.items()}).encode())
        with self.engine.begin() as connection:
            result = connection.execute(update(datasets).where(
                datasets.c.owner == owner, datasets.c.id == int(dataset_id)).values(
                name=name, created_at=datetime.now(timezone.utc), payload=payload,
                size_bytes=sum(len(content) for content in files.values())))
        if result.rowcount != 1:
            raise StorageError("That model no longer exists. Save this one as a new model.")
        return int(dataset_id)

    def list(self, owner: str, kind: str | None = None) -> pd.DataFrame:
        query = (select(datasets.c.id, datasets.c.kind, datasets.c.name, datasets.c.created_at,
                        datasets.c.size_bytes)
                 .where(datasets.c.owner == owner).order_by(datasets.c.created_at.desc(), datasets.c.id.desc()))
        if kind is not None:
            query = query.where(datasets.c.kind == kind)
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return pd.DataFrame(rows, columns=["id", "kind", "name", "created_at", "size_bytes"])

    def load(self, owner: str, dataset_id: int) -> dict[str, bytes]:
        with self.engine.connect() as connection:
            payload = connection.execute(select(datasets.c.payload).where(
                datasets.c.owner == owner, datasets.c.id == int(dataset_id))).scalar_one_or_none()
        if payload is None:
            raise StorageError("That dataset does not exist or belongs to another account.")
        return {label: _decode(value) for label, value in json.loads(gzip.decompress(payload)).items()}

    def delete(self, owner: str, dataset_id: int) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(delete(datasets).where(
                datasets.c.owner == owner, datasets.c.id == int(dataset_id)))
        return result.rowcount == 1

    def delete_all(self, owner: str) -> int:
        with self.engine.begin() as connection:
            return connection.execute(delete(datasets).where(datasets.c.owner == owner)).rowcount
