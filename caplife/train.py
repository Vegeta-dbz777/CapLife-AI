"""Reproducible group-separated training for the demonstrator GRU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import FEATURES, SEQ_LEN, RUL_SCALE_HOURS
from .data import windowize, make_scaler, scale_windows
from .model import RULGRU
from .simulator import simulate_capacitors


def train_and_save(root: str | Path, n_caps: int = 180, epochs: int = 25,
                   seed: int = 42, hidden_dim: int = 32) -> dict:
    root = Path(root)
    artifacts = root / "artifacts"
    samples = root / "sample_data"
    artifacts.mkdir(parents=True, exist_ok=True)
    samples.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    df = simulate_capacitors(n_caps=n_caps, seed=seed)
    df.to_csv(samples / "demo_capacitors.csv", index=False)
    template = df.drop(columns=["RUL_hours", "EOL_observed", "EOL_cause"]).head(6)
    template.to_csv(samples / "upload_template.csv", index=False)
    known_caps = df.loc[df["EOL_observed"], "Cap_ID"].unique()
    rng = np.random.default_rng(seed)
    cap_ids = rng.permutation(known_caps)
    n_train = int(len(cap_ids) * 0.7)
    n_val = int(len(cap_ids) * 0.15)
    ids_train = set(cap_ids[:n_train])
    ids_val = set(cap_ids[n_train:n_train + n_val])
    ids_test = set(cap_ids[n_train + n_val:])
    split = {}
    for name, ids in [("train", ids_train), ("val", ids_val), ("test", ids_test)]:
        split_df = df.loc[df["Cap_ID"].isin(ids)]
        X, y, _ = windowize(split_df, seq_len=SEQ_LEN)
        split[name] = (X, y)
    scaler = make_scaler(split["train"][0])
    def tensors(name):
        X, y = split[name]
        return torch.from_numpy(scale_windows(X, scaler)), torch.from_numpy(y / RUL_SCALE_HOURS)
    xtr, ytr = tensors("train")
    xval, yval = tensors("val")
    xte, yte = tensors("test")
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    net = RULGRU(len(FEATURES), hidden_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    loss_fn = torch.nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    best_epoch = None
    no_improve = 0
    for epoch in range(1, epochs + 1):
        net.train()
        for xbatch, ybatch in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xbatch), ybatch)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            vloss = float(loss_fn(net(xval), yval).item())
        print(f"Epoch {epoch:02d}: validation loss = {vloss:.5f}", flush=True)
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 7:
                print("Early stopped on validation set.", flush=True)
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ypred = (net(xte).numpy() * RUL_SCALE_HOURS).clip(min=0)
    ytrue = yte.numpy() * RUL_SCALE_HOURS
    metrics = {
        "test_MAE_hours": round(float(mean_absolute_error(ytrue, ypred)), 2),
        "test_RMSE_hours": round(float(np.sqrt(mean_squared_error(ytrue, ypred))), 2),
        "test_R2": round(float(r2_score(ytrue, ypred)), 4),
        "test_measurement_windows": int(len(yte)),
        "train_capacitors": len(ids_train),
        "validation_capacitors": len(ids_val),
        "test_capacitors": len(ids_test),
        "excluded_censored_capacitors": int(df["Cap_ID"].nunique() - len(known_caps)),
        "best_epoch": best_epoch,
    }
    minmax = df[df["Cap_ID"].isin(ids_train)]
    training_bounds = {f: [float(minmax[f].min()), float(minmax[f].max())] for f in FEATURES}
    metadata = {
        "model_id": "caplife-gru-demo-v0.1",
        "model_type": "GRU time-series regression",
        "hidden_dim": hidden_dim,
        "sequence_length": SEQ_LEN,
        "features": FEATURES,
        "rul_scale_hours": RUL_SCALE_HOURS,
        "scaler": scaler,
        "training_bounds": training_bounds,
        "training_data_provenance": "Invented synthetic degradation trajectories; NO real AEC experiments",
        "validated_for_real_world_use": False,
        "unit_conventions": {
            "Hours": "elapsed operational hours",
            "Ambient_C": "degrees Celsius",
            "Core_C": "degrees Celsius",
            "Voltage_ratio": "applied voltage / rated voltage",
            "Ripple_A": "RMS ripple current in amperes",
            "Ripple_Hz": "ripple current frequency in hertz",
            "C_ratio": "measured capacitance / initial capacitance",
            "ESR_ratio": "measured ESR / initial ESR, measured at a consistent specified frequency and temperature",
            "Leakage_ratio": "measured leakage / initial reference leakage, controlled measurement protocol",
            "Case_Strain_Index": "nonnegative hypothetical normalized strain; NOT measured gas pressure",
        },
        "metrics": metrics,
        "warning": "DEMONSTRATION ONLY. No reliability or safety claims. No calibrated prediction intervals.",
    }
    torch.save(net.state_dict(), artifacts / "gru_demo_state.pt")
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out synthetic capacitor results:", json.dumps(metrics, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--caps", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    train_and_save(args.root, n_caps=args.caps, epochs=args.epochs)
