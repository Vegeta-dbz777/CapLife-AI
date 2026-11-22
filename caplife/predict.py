"""Reusable inference engine called by the website or Python scripts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import validate_measurements, last_window, scale_windows, out_of_training_range
from .model import RULGRU
from .simulator import end_of_life_causes


def load_engine(artifact_dir: str | Path = "artifacts"):
    path = Path(artifact_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    net = RULGRU(num_features=len(metadata["features"]), hidden_dim=metadata["hidden_dim"])
    net.load_state_dict(torch.load(path / "gru_demo_state.pt", map_location="cpu", weights_only=True))
    net.eval()
    return net, metadata


def assess_capacitor(cap_df: pd.DataFrame, net: RULGRU, metadata: dict) -> dict:
    data = validate_measurements(cap_df)
    ids = data["Cap_ID"].unique()
    if len(ids) != 1:
        raise ValueError("Select exactly one capacitor before prediction.")
    recent = data.iloc[-metadata["sequence_length"]:]
    bounds_warnings = out_of_training_range(recent, metadata.get("training_bounds", {}))
    latest = recent.iloc[-1].to_dict()
    failure_reasons = end_of_life_causes(latest)
    if failure_reasons:
        # The deterministic tutorial EOL rule supersedes NN output.
        pred_rul = 0.0
    else:
        window = last_window(data, metadata["sequence_length"])
        normalized = scale_windows(window, metadata["scaler"])
        with torch.no_grad():
            pred_rul = float(net(torch.from_numpy(normalized)).item()) * metadata["rul_scale_hours"]
    return {
        "cap_id": ids[0],
        "hours_observed": float(latest["Hours"]),
        "rul_hours": max(0.0, pred_rul),
        "status": "EOL threshold reached" if failure_reasons else "No demo EOL threshold reached",
        "eol_causes": failure_reasons,
        "warnings": bounds_warnings,
        "model_id": metadata["model_id"],
        "data_provenance": metadata["training_data_provenance"],
    }


def default_prediction_hour(cap_df: pd.DataFrame, options: list[float]) -> float:
    """Latest selectable hour before the illustrative EOL rule fires.

    Starting there shows a GRU estimate on first use instead of the rule's
    zero. Falls back to the latest option when every option is past the rule.
    """
    rows = cap_df.set_index("Hours")
    for hour in reversed(options):
        row = rows.loc[hour]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        if not end_of_life_causes(row.to_dict()):
            return hour
    return options[-1]
