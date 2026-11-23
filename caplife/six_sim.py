"""Physics-inspired simulator of six-indicator aluminum electrolytic capacitor aging.

This generates *training data for a demonstration model*. The relations follow
mechanisms described in the AEC literature, but every coefficient is chosen for
illustration and none is calibrated against a real capacitor series:

- Aging rate doubles for every 10 °C of core temperature (Arrhenius rule of
  thumb) and rises with applied voltage. Ripple current heats the core.
- Electrolyte loss lowers capacitance and raises ESR.
- Leakage current grows with aging. A fraction of it electrolyses water and
  generates hydrogen (Faraday's law: 0.116 mL of H₂ at STP per ampere-second).
- Retained hydrogen raises the absolute internal pressure in a fixed headspace,
  and the pressure rise bulges the case.

End of life is the first of six limits, one per indicator: capacitance loss of
20%, ESR at twice its initial value, leakage at five times its initial value,
case expansion of 52 µm, 6 mL of generated hydrogen, or internal pressure
reaching the vent pressure. The first two are common criteria in the AEC
literature; the others are illustrative. The expansion and hydrogen limits are
the levels at which a typical simulated capacitor reaches the vent pressure, so
a stiff-cased or gas-retaining capacitor can still vent first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNALS = ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um",
           "H2_Generated_mL_STP", "Pressure_Abs_kPa")
EOL_LIMITS = {
    "capacitance_loss_pct": 20.0,
    "esr_ratio": 2.0,
    "leakage_ratio": 5.0,
    "pressure_abs_kpa": 250.0,
    "expansion_um": 52.0,  # bulge of the case relative to the first reading
    "h2_ml": 6.0,  # cumulative hydrogen generated, at 0 °C and 1 atm
}
ATMOSPHERE_KPA = 101.325
FARADAY_ML_PER_AS = 0.116  # mL of H2 at 0 °C and 1 atm per ampere-second
MEASUREMENT_TEMP_K = 298.15
H2_EFFICIENCY = 0.003  # median fraction of leakage charge that generates H2
EXPANSION_NOISE_UM = 0.25  # standard deviation of a precise laser displacement measurement
NOISE = {"Capacitance_uF": 0.003, "ESR_ohm": 0.015, "Leakage_uA": 0.03, "H2_Generated_mL_STP": 0.02}


def _device(rng: np.random.Generator) -> dict:
    ambient = rng.uniform(70, 105)
    ripple = rng.uniform(0.2, 1.2)  # ratio of rated ripple current
    voltage = rng.uniform(0.6, 1.0)  # ratio of rated voltage
    core = ambient + 12 * ripple ** 2
    acceleration = 2 ** ((core - 105) / 10) * voltage ** 3
    return {
        "ambient_C": ambient, "ripple_ratio": ripple, "voltage_ratio": voltage, "core_C": core,
        # Hours for the electrolyte-loss coordinate to reach 1 at rated conditions.
        "tau_h": 4000 * rng.lognormal(0, 0.25) / acceleration,
        "capacitance_uF": 1000 * rng.normal(1, 0.05),
        "esr_ohm": 0.08 * rng.normal(1, 0.08),
        "leakage_uA": 8 * rng.lognormal(0, 0.2),
        "c_shape": rng.uniform(1.0, 1.5),
        "c_rate": rng.lognormal(0, 0.25),
        "esr_gain": rng.lognormal(-0.3, 0.3),
        "leak_gain": rng.lognormal(0, 0.35),
        "h2_efficiency": rng.lognormal(np.log(H2_EFFICIENCY), 0.6),
        "retained_fraction": rng.uniform(0.3, 0.7),
        "headspace_mL": rng.uniform(1.5, 3.0),
        "expansion_um_per_kPa": rng.uniform(0.25, 0.45),
    }


def _state(device: dict, hours: np.ndarray) -> dict[str, np.ndarray]:
    """Noise-free indicators at the given ages, plus cumulative H2 by trapezoids."""
    loss = hours / device["tau_h"]
    capacitance = device["capacitance_uF"] * (1 - 0.2 * (device["c_rate"] * loss) ** device["c_shape"])
    esr = device["esr_ohm"] * (1 + device["esr_gain"] * loss ** 2 / (1 - 0.3 * np.minimum(loss, 2.5)))
    leakage = device["leakage_uA"] * (1 + 3 * device["leak_gain"] * loss ** 2)
    # Leakage at the operating core temperature drives gas generation.
    operating_leakage_a = leakage * 1e-6 * 2 ** ((device["core_C"] - 25) / 20) * device["voltage_ratio"]
    rate_ml_per_h = FARADAY_ML_PER_AS * operating_leakage_a * 3600 * device["h2_efficiency"]
    h2 = np.concatenate([[0.0], np.cumsum(np.diff(hours) * (rate_ml_per_h[1:] + rate_ml_per_h[:-1]) / 2)])
    retained = h2 * device["retained_fraction"]
    # Pressure reported at the 25 °C measurement temperature.
    pressure = ATMOSPHERE_KPA * (1 + retained / device["headspace_mL"] * MEASUREMENT_TEMP_K / 273.15)
    expansion = device["expansion_um_per_kPa"] * (pressure - ATMOSPHERE_KPA)
    return {"Capacitance_uF": capacitance, "ESR_ohm": esr, "Leakage_uA": leakage,
            "Case_Expansion_um": expansion, "H2_Generated_mL_STP": h2, "Pressure_Abs_kPa": pressure}


def end_of_life_causes(baseline: dict, reading: dict) -> list[str]:
    """Which end-of-life limits a reading meets, relative to its baseline."""
    # A small tolerance so a reading exactly at a limit counts despite floating-point rounding.
    tolerance = 1e-9
    causes = []
    if 100 * (1 - reading["Capacitance_uF"] / baseline["Capacitance_uF"]) >= EOL_LIMITS["capacitance_loss_pct"] - tolerance:
        causes.append("capacitance")
    if reading["ESR_ohm"] / baseline["ESR_ohm"] >= EOL_LIMITS["esr_ratio"] - tolerance:
        causes.append("ESR")
    if reading["Leakage_uA"] / baseline["Leakage_uA"] >= EOL_LIMITS["leakage_ratio"] - tolerance:
        causes.append("leakage")
    if reading["Case_Expansion_um"] - baseline["Case_Expansion_um"] >= EOL_LIMITS["expansion_um"] - tolerance:
        causes.append("expansion")
    if reading["H2_Generated_mL_STP"] - baseline["H2_Generated_mL_STP"] >= EOL_LIMITS["h2_ml"] - tolerance:
        causes.append("H2")
    if reading["Pressure_Abs_kPa"] >= EOL_LIMITS["pressure_abs_kpa"] - tolerance:
        causes.append("pressure")
    return causes


def simulate_device(rng: np.random.Generator, cap_id: str, fine_step_h: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Noisy inspections until end of life, and the true (noise-free) end-of-life age and cause."""
    device = _device(rng)
    horizon = 4 * device["tau_h"]
    fine = np.arange(0, horizon + fine_step_h, fine_step_h)
    truth = _state(device, fine)
    base = {name: values[0] for name, values in truth.items()}
    eol_index = next(i for i in range(len(fine)) if end_of_life_causes(base, {k: v[i] for k, v in truth.items()}))
    eol_h = float(fine[eol_index])
    causes = end_of_life_causes(base, {k: v[eol_index] for k, v in truth.items()})

    # Inspections at a device-specific interval with jitter, stopping before end of life.
    interval = eol_h / rng.uniform(12, 40)
    times = [0.0]
    while True:
        step = interval * rng.uniform(0.7, 1.3)
        if times[-1] + step >= eol_h:
            break
        times.append(times[-1] + step)
    times = np.round(np.array(times), 1)
    # Cumulative H2 is integrated on the sorted dense grid (``truth``) and then read off at the
    # inspection times, so it does not depend on inspection spacing.
    state = {k: np.interp(times, fine, v) for k, v in truth.items()}
    frame = pd.DataFrame({"Cap_ID": cap_id, "Hours": times})
    for name in SIGNALS:
        values = state[name].copy()
        if name in NOISE:
            values *= 1 + rng.normal(0, NOISE[name], len(values))
        elif name == "Case_Expansion_um":
            values += rng.normal(0, EXPANSION_NOISE_UM, len(values))
        else:
            values += rng.normal(0, 0.4, len(values))
        frame[name] = values
    frame.loc[0, "H2_Generated_mL_STP"] = 0.0
    frame.loc[0, "Case_Expansion_um"] = 0.0
    frame["H2_Generated_mL_STP"] = frame["H2_Generated_mL_STP"].clip(lower=0).cummax()
    # Expansion is a change from the first reading, so a precise sensor can report small negative
    # values early on; they are kept rather than clipped to zero.
    rounding = {"Capacitance_uF": 2, "ESR_ohm": 5, "Leakage_uA": 3, "Case_Expansion_um": 2,
                "H2_Generated_mL_STP": 4, "Pressure_Abs_kPa": 2}
    frame = frame.round(rounding)
    meta = {"Cap_ID": cap_id, "eol_h": eol_h, "eol_cause": ", ".join(causes),
            **{k: round(float(v), 4) for k, v in device.items()}}
    return frame, meta


def simulate_fleet(n_devices: int, seed: int, prefix: str = "SIM") -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frames, metas = [], []
    for index in range(1, n_devices + 1):
        frame, meta = simulate_device(rng, f"{prefix}_{index:04d}")
        frames.append(frame)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)
