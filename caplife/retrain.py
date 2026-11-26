"""Train a saved model again: its own training data, with newer readings and outcomes added on top.

A saved model keeps the files it was trained on (``TRAINING_FILES``) and the settings it was trained with, so it
can be trained again later with the same limits and split when more capacitors have been tested or more have
failed.
"""
from __future__ import annotations

import pandas as pd

from caplife.file_reader import read_measurement_file
from caplife.rul_workflow import OPTIONAL_OUTCOME_COLUMNS, OUTCOME_COLUMNS, read_outcomes

TRAINING_FILES = {"measurements": "training_measurements", "outcomes": "training_outcomes"}


def training_data(files: dict[str, bytes]) -> dict[str, bytes] | None:
    """The measurement and outcome files a saved model was trained on, or None if it predates keeping them."""
    if not all(name in files for name in TRAINING_FILES.values()):
        return None
    return {kind: files[name] for kind, name in TRAINING_FILES.items()}


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def merge_measurements(old: bytes, new: bytes) -> tuple[bytes, str]:
    """Earlier readings with the new ones added; a reading at the same capacitor and age is taken from the new
    file. Returns the merged CSV and a sentence saying what was added."""
    before, _ = read_measurement_file(old)
    added, _ = read_measurement_file(new)
    merged = pd.concat([before, added], ignore_index=True)
    merged = merged.drop_duplicates(["Cap_ID", "Hours"], keep="last").sort_values(["Cap_ID", "Hours"])
    readings = len(merged) - len(before)
    capacitors = merged["Cap_ID"].nunique() - before["Cap_ID"].nunique()
    note = (f"Added {_plural(readings, 'reading')}"
            + (f", including {_plural(capacitors, 'new capacitor')}" if capacitors else "") + ".")
    return merged.to_csv(index=False).encode("utf-8"), note


def merge_outcomes(old: bytes, new: bytes) -> tuple[bytes, str]:
    """Earlier outcomes with the new ones added; a capacitor in both files takes its outcome from the new file
    (for example one that was still working and has since failed)."""
    before, added = read_outcomes(old), read_outcomes(new)
    columns = [*OUTCOME_COLUMNS, *(c for c in OPTIONAL_OUTCOME_COLUMNS if c in before or c in added)]
    before, added = (frame.reindex(columns=columns).fillna("") for frame in (before, added))
    for frame in (before, added):
        frame["Cap_ID"] = frame["Cap_ID"].astype(str).str.strip()
    updated = int(before["Cap_ID"].isin(set(added["Cap_ID"])).sum())
    merged = pd.concat([before.loc[~before["Cap_ID"].isin(set(added["Cap_ID"]))], added], ignore_index=True)
    fresh = len(merged) - len(before)
    parts = [f"{_plural(fresh, 'new outcome')}"] + ([f"{_plural(updated, 'outcome')} updated"] if updated else [])
    return merged.to_csv(index=False).encode("utf-8"), "Outcomes: " + ", ".join(parts) + "."
