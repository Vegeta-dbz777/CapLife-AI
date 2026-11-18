"""Strict input contract and leak-free sequence preprocessing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FEATURES, SEQ_LEN


class DataValidationError(ValueError):
    pass


def validate_measurements(df: pd.DataFrame, seq_len: int = SEQ_LEN,
                          min_rows_per_cap: bool = True) -> pd.DataFrame:
    required = ["Cap_ID", *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise DataValidationError("The uploaded dataset contains no records.")
    if len(df) > 100_000:
        raise DataValidationError("Prototype limit: at most 100,000 measurement rows.")
    copy = df.copy()
    if copy["Cap_ID"].isna().any() or (copy["Cap_ID"].astype(str).str.strip() == "").any():
        raise DataValidationError("Cap_ID must not be empty.")
    copy["Cap_ID"] = copy["Cap_ID"].astype(str).str.strip()
    for c in FEATURES:
        copy[c] = pd.to_numeric(copy[c], errors="coerce")
        if not np.isfinite(copy[c].to_numpy(dtype=float)).all():
            raise DataValidationError(f"{c} contains missing or nonnumeric values.")
    if (copy["Hours"] < 0).any():
        raise DataValidationError("Hours must be nonnegative.")
    if (copy["Ripple_Hz"] <= 0).any() or (copy["Ripple_A"] < 0).any():
        raise DataValidationError("Ripple_Hz must be positive and Ripple_A nonnegative.")
    if (copy["Voltage_ratio"] <= 0).any():
        raise DataValidationError("Voltage_ratio must be positive.")
    for c in ["C_ratio", "ESR_ratio", "Leakage_ratio"]:
        if (copy[c] <= 0).any():
            raise DataValidationError(f"{c} must be positive.")
    if (copy["Case_Strain_Index"] < 0).any():
        raise DataValidationError("Case_Strain_Index must be nonnegative.")
    copy = copy.sort_values(["Cap_ID", "Hours"]).reset_index(drop=True)
    if copy.duplicated(["Cap_ID", "Hours"]).any():
        raise DataValidationError("Duplicate measurement hours for a Cap_ID. Combine duplicates first.")
    if min_rows_per_cap:
        short = copy.groupby("Cap_ID").size()
        short = short[short < seq_len]
        if len(short):
            raise DataValidationError(
                f"Each capacitor needs at least {seq_len} readings. "
                f"Too few for: {', '.join(short.index[:5])}"
            )
    return copy


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    arr = df[FEATURES].to_numpy(dtype=np.float32).copy()
    arr[:, FEATURES.index("Ripple_Hz")] = np.log10(arr[:, FEATURES.index("Ripple_Hz")])
    return arr


def windowize(df: pd.DataFrame, seq_len: int = SEQ_LEN,
              require_labels: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    windows, targets, cap_ids = [], [], []
    for cap_id, cap_df in df.groupby("Cap_ID", sort=False):
        feats = feature_matrix(cap_df)
        labels = cap_df["RUL_hours"].to_numpy(dtype=float) if "RUL_hours" in cap_df else None
        for end in range(seq_len - 1, len(cap_df)):
            if require_labels and (labels is None or not np.isfinite(labels[end])):
                continue
            windows.append(feats[end - seq_len + 1:end + 1])
            targets.append(float(labels[end]) if labels is not None else np.nan)
            cap_ids.append(str(cap_id))
    if not windows:
        raise DataValidationError("No usable sequences. Need repeated measurements and observed EOL labels.")
    return np.stack(windows).astype(np.float32), np.array(targets, dtype=np.float32), cap_ids


def last_window(cap_df: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    if len(cap_df) < seq_len:
        raise DataValidationError(f"Need at least {seq_len} readings for this capacitor.")
    cap_df = cap_df.sort_values("Hours")
    return feature_matrix(cap_df.iloc[-seq_len:])[None, :, :]


def make_scaler(X_train: np.ndarray) -> dict:
    means = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    stds = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return {"mean": means.tolist(), "std": stds.tolist()}


def scale_windows(X: np.ndarray, scaler: dict) -> np.ndarray:
    means = np.asarray(scaler["mean"], dtype=np.float32)
    stds = np.asarray(scaler["std"], dtype=np.float32)
    return ((X - means) / stds).astype(np.float32)


def out_of_training_range(cap_df: pd.DataFrame, training_bounds: dict) -> list[str]:
    """Conservative checks; outside-training input is not fit for a claim of reliability."""
    warnings = []
    for c in FEATURES:
        if c not in training_bounds:
            continue
        low, high = training_bounds[c]
        if cap_df[c].min() < low or cap_df[c].max() > high:
            warnings.append(f"{c} is outside the demo training range ({low:g} to {high:g}).")
    return warnings
