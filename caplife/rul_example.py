"""Synthetic six-indicator histories and outcomes for trying the RUL training workflow.

Every value here is generated. The data only demonstrate the file formats and
the train / calibrate / test / forecast steps; results on them say nothing
about real capacitors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENDPOINT = "Capacitance loss of 20% or more (synthetic)"
EXPERIMENTS = {"SYN_EXP_A": 1.0, "SYN_EXP_B": 1.12, "SYN_EXP_C": 0.92, "SYN_EXP_D": 1.05}
DEVICES_PER_EXPERIMENT = 8
STEP_H = 50
EXPERIMENT_END_H = {"SYN_EXP_D": 900}
RUNNING_DEVICES = 5


def _readings(rng, device: str, life_h: float, until_h: float) -> list[dict]:
    rows = []
    for hours in np.arange(0, until_h + 1, STEP_H):
        loss = 20 * (hours / life_h) ** 1.3
        noisy = loss + rng.normal(0, 0.25) if hours else 0.0
        rows.append({
            "Cap_ID": device, "Hours": float(hours),
            "Capacitance_uF": round(1000 * (1 - noisy / 100), 2),
            "ESR_ohm": round(0.1 * (1 + 0.022 * loss * (1 + rng.normal(0, 0.04))), 5),
            "Leakage_uA": round(10 * (1 + 0.03 * loss) + rng.normal(0, 0.05), 3),
            "Case_Expansion_um": round(max(2.5 * loss + rng.normal(0, 0.4), 0), 2),
            "H2_Generated_mL_STP": round(max(0.02 * loss ** 1.5 + rng.normal(0, 0.01), 0), 4),
            "Pressure_Abs_kPa": round(101.3 + 0.9 * loss + rng.normal(0, 0.15), 2),
            "ESR_Test_Hz": 100000, "ESR_Test_Temp_C": 25,
        })
    return rows


def example_dataset(seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measurements for 37 synthetic capacitors and outcomes for the 32 that finished testing."""
    rng = np.random.default_rng(seed)
    measurements, outcomes = [], []
    for experiment, stress in EXPERIMENTS.items():
        end = EXPERIMENT_END_H.get(experiment, np.inf)
        for index in range(1, DEVICES_PER_EXPERIMENT + 1):
            device = f"{experiment}_{index}"
            # Device-to-device variation in how fast the capacitor ages.
            life = 1200 / (stress * rng.lognormal(0, 0.3))
            failed_at = STEP_H * np.ceil(life / STEP_H)
            if failed_at <= end:
                measurements += _readings(rng, device, life, failed_at)
                outcomes.append({"Cap_ID": device, "Experiment_ID": experiment, "Endpoint": ENDPOINT,
                                 "Last_alive_h": failed_at - STEP_H, "Failure_by_h": failed_at})
            else:
                measurements += _readings(rng, device, life, end)
                outcomes.append({"Cap_ID": device, "Experiment_ID": experiment, "Endpoint": ENDPOINT,
                                 "Last_alive_h": end, "Failure_by_h": ""})
    for index in range(1, RUNNING_DEVICES + 1):
        life = 1200 / rng.lognormal(0, 0.3)
        observed_to = STEP_H * np.floor(min(life * 0.75, 200 + 100 * index) / STEP_H)
        measurements += _readings(rng, f"SYN_RUNNING_{index}", life, max(observed_to, 2 * STEP_H))
    return pd.DataFrame(measurements), pd.DataFrame(outcomes).astype({"Failure_by_h": str})
