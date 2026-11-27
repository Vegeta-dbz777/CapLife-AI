"""Physics-inspired simulator of six-indicator aluminum electrolytic capacitor aging.

This generates *training data for a demonstration model*. The relations follow
mechanisms described in the AEC literature, but every coefficient is chosen for
illustration and none is calibrated against a real capacitor series:

- Aging rate doubles for every 10 °C of core temperature (Arrhenius rule of
  thumb, for a 105 °C series) and rises with the cube of applied voltage over
  rated voltage. Ripple current heats the core by about 12 K at the rated ripple
  current; at higher operating frequencies ESR is lower, so the same current
  heats less (up to 1.4 times the rated current for the same heating at
  10 kHz and above).
- Each capacitor has its own rated voltage, rated ripple current, operating
  frequency, ambient temperature, applied voltage and ripple current, and half
  of the capacitors move to new operating conditions part-way through life, at
  an inspection that records them. Every inspection records the conditions in
  effect from then on.
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

import math

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
# Users can set their own limit for each indicator (a capacitor series' own tolerance). The model is
# trained on limits drawn from these ranges, so it can predict the time to any limit inside them.
LIMIT_RANGES = {
    "capacitance_loss_pct": (5.0, 40.0),
    "esr_ratio": (1.2, 4.0),
    "leakage_ratio": (1.5, 10.0),
    "expansion_um": (10.0, 100.0),
    "h2_ml": (1.0, 12.0),
    "pressure_abs_kpa": (130.0, 350.0),
}
DEFAULT_LIMIT_SHARE = 0.3  # share of training capacitors that keep the default limits


def random_limits(rng: np.random.Generator) -> dict:
    """The default limits, or each limit drawn log-uniformly from its range (pressure: its rise)."""
    if rng.random() < DEFAULT_LIMIT_SHARE:
        return dict(EOL_LIMITS)
    limits = {}
    for key, (low, high) in LIMIT_RANGES.items():
        if key in ("esr_ratio", "leakage_ratio"):  # the allowed rise is what varies by orders of magnitude
            limits[key] = 1 + float(np.exp(rng.uniform(np.log(low - 1), np.log(high - 1))))
        elif key == "pressure_abs_kpa":
            rise = np.exp(rng.uniform(np.log(low - ATMOSPHERE_KPA), np.log(high - ATMOSPHERE_KPA)))
            limits[key] = ATMOSPHERE_KPA + float(rise)
        else:
            limits[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return limits
ATMOSPHERE_KPA = 101.325
FARADAY_ML_PER_AS = 0.116  # mL of H2 at 0 °C and 1 atm per ampere-second
MEASUREMENT_TEMP_K = 298.15
H2_EFFICIENCY = 0.003  # median fraction of leakage charge that generates H2
EXPANSION_NOISE_UM = 0.25  # standard deviation of a precise laser displacement measurement
NOISE = {"Capacitance_uF": 0.003, "ESR_ohm": 0.015, "Leakage_uA": 0.03, "H2_Generated_mL_STP": 0.02}


CONDITION_COLUMNS = ("Ambient_C", "Core_C", "Voltage_V", "Rated_V", "Ripple_A", "Rated_Ripple_A", "Ripple_Hz")
RATED_VOLTAGES = (16.0, 25.0, 35.0, 50.0, 63.0, 100.0, 160.0, 250.0, 400.0, 450.0)
RATED_TEMP_C = 105.0
CHANGE_SHARE = 0.5  # share of capacitors whose operating conditions change part-way through life


def frequency_multiplier(hz):
    """Allowed ripple current relative to the rated (120 Hz) value: 1 at 120 Hz, 1.4 from 10 kHz."""
    return 1 + 0.4 * np.clip(np.log10(np.asarray(hz, dtype=float) / 120) / 2, 0, 1)


# The physics constants above, as one capacitor family; ``random_family`` draws other families.
STANDARD_FAMILY = {"life_scale": 1.0, "doubling_c": 10.0, "voltage_exponent": 3.0, "ripple_heating_k": 12.0,
                   "c_shape": (1.0, 1.5)}


def random_family(rng: np.random.Generator) -> dict:
    """Physics constants of a capacitor family unlike the standard one: real series differ in base life,
    in how strongly temperature, voltage and ripple speed aging, and in how capacitance loss accelerates.
    A model trained on many families has to read each capacitor's own trend instead of trusting one set of
    constants."""
    shape_low = rng.uniform(0.8, 1.6)
    return {"life_scale": float(np.exp(rng.normal(0.0, 1.0))), "doubling_c": rng.uniform(7.0, 14.0),
            "voltage_exponent": rng.uniform(1.5, 5.0), "ripple_heating_k": rng.uniform(6.0, 18.0),
            "c_shape": (shape_low, shape_low + rng.uniform(0.1, 0.6))}


def operating_stress(ambient_c, voltage_ratio, ripple_ratio, hz, family: dict | None = None):
    """Core temperature and aging acceleration (1 at rated conditions without ripple heating)."""
    family = family or STANDARD_FAMILY
    core = np.asarray(ambient_c, dtype=float) + family["ripple_heating_k"] * (np.asarray(ripple_ratio, dtype=float)
                                                                            / frequency_multiplier(hz)) ** 2
    return core, (2 ** ((core - RATED_TEMP_C) / family["doubling_c"])
                  * np.asarray(voltage_ratio, dtype=float) ** family["voltage_exponent"])


def _operating(rng: np.random.Generator) -> tuple[float, float, float]:
    """Ambient temperature, applied/rated voltage and ripple/rated ripple current."""
    return rng.uniform(60, 105), rng.uniform(0.5, 1.0), rng.uniform(0.2, 1.3)


def _device(rng: np.random.Generator, family: dict | None = None) -> dict:
    family = family or STANDARD_FAMILY
    ambient, voltage, ripple = _operating(rng)
    hz = 120.0 if rng.random() < 0.45 else float(np.exp(rng.uniform(np.log(1e3), np.log(1e5))))
    core, acceleration = (float(v) for v in operating_stress(ambient, voltage, ripple, hz, family))
    # Hours for the electrolyte-loss coordinate to reach 1 at rated conditions.
    tau_rated = 4000 * family["life_scale"] * rng.lognormal(0, 0.25)
    device = {
        "ambient_C": ambient, "ripple_ratio": ripple, "voltage_ratio": voltage, "core_C": core,
        "ripple_Hz": hz, "rated_V": float(rng.choice(RATED_VOLTAGES)),
        "rated_ripple_A": float(rng.lognormal(np.log(1.5), 0.5)),
        "tau_h": tau_rated / acceleration, "change_h": np.inf,
    }
    if rng.random() < CHANGE_SHARE:
        ambient2, voltage2, ripple2 = _operating(rng)
        core2, acceleration2 = (float(v) for v in operating_stress(ambient2, voltage2, ripple2, hz, family))
        # On the 0.1 h grid inspections are recorded on, so the inspection at the change records the new ones.
        change_h = math.ceil(rng.uniform(0.15, 0.7) * device["tau_h"] * 10) / 10
        device.update(change_h=change_h, ambient2_C=ambient2,
                      voltage2_ratio=voltage2, ripple2_ratio=ripple2, core2_C=core2, tau2_h=tau_rated / acceleration2)
    else:
        device.update(ambient2_C=ambient, voltage2_ratio=voltage, ripple2_ratio=ripple, core2_C=core,
                      tau2_h=device["tau_h"])
    return {
        **device,
        "capacitance_uF": 1000 * rng.normal(1, 0.05),
        "esr_ohm": 0.08 * rng.normal(1, 0.08),
        "leakage_uA": 8 * rng.lognormal(0, 0.2),
        "c_shape": rng.uniform(*family["c_shape"]),
        "c_rate": rng.lognormal(0, 0.25),
        "esr_gain": rng.lognormal(-0.3, 0.3),
        "leak_gain": rng.lognormal(0, 0.35),
        "h2_efficiency": rng.lognormal(np.log(H2_EFFICIENCY), 0.6),
        "retained_fraction": rng.uniform(0.3, 0.7),
        "headspace_mL": rng.uniform(1.5, 3.0),
        "expansion_um_per_kPa": rng.uniform(0.25, 0.45),
    }


def conditions_at(device: dict, hours: np.ndarray) -> dict[str, np.ndarray]:
    """Operating conditions in effect at each age (the second set from ``change_h`` on)."""
    later = np.asarray(hours, dtype=float) >= device["change_h"]
    pick = lambda first, second: np.where(later, device[second], device[first])
    voltage, ripple = pick("voltage_ratio", "voltage2_ratio"), pick("ripple_ratio", "ripple2_ratio")
    return {"Ambient_C": pick("ambient_C", "ambient2_C"), "Core_C": pick("core_C", "core2_C"),
            "Voltage_V": voltage * device["rated_V"], "Rated_V": np.full(later.shape, device["rated_V"]),
            "Ripple_A": ripple * device["rated_ripple_A"],
            "Rated_Ripple_A": np.full(later.shape, device["rated_ripple_A"]),
            "Ripple_Hz": np.full(later.shape, device["ripple_Hz"]), "voltage_ratio": voltage}


def _state(device: dict, hours: np.ndarray) -> dict[str, np.ndarray]:
    """Noise-free indicators at the given ages, plus cumulative H2 by trapezoids."""
    before = np.minimum(hours, device["change_h"])
    loss = before / device["tau_h"] + (hours - before) / device["tau2_h"]
    conditions = conditions_at(device, hours)
    capacitance = device["capacitance_uF"] * (1 - 0.2 * (device["c_rate"] * loss) ** device["c_shape"])
    esr = device["esr_ohm"] * (1 + device["esr_gain"] * loss ** 2 / (1 - 0.3 * np.minimum(loss, 2.5)))
    leakage = device["leakage_uA"] * (1 + 3 * device["leak_gain"] * loss ** 2)
    # Leakage at the operating core temperature drives gas generation.
    operating_leakage_a = leakage * 1e-6 * 2 ** ((conditions["Core_C"] - 25) / 20) * conditions["voltage_ratio"]
    rate_ml_per_h = FARADAY_ML_PER_AS * operating_leakage_a * 3600 * device["h2_efficiency"]
    h2 = np.concatenate([[0.0], np.cumsum(np.diff(hours) * (rate_ml_per_h[1:] + rate_ml_per_h[:-1]) / 2)])
    retained = h2 * device["retained_fraction"]
    # Pressure reported at the 25 °C measurement temperature.
    pressure = ATMOSPHERE_KPA * (1 + retained / device["headspace_mL"] * MEASUREMENT_TEMP_K / 273.15)
    expansion = device["expansion_um_per_kPa"] * (pressure - ATMOSPHERE_KPA)
    return {"Capacitance_uF": capacitance, "ESR_ohm": esr, "Leakage_uA": leakage,
            "Case_Expansion_um": expansion, "H2_Generated_mL_STP": h2, "Pressure_Abs_kPa": pressure}


def end_of_life_causes(baseline: dict, reading: dict, limits: dict | None = None) -> list[str]:
    """Which end-of-life limits a reading meets, relative to its baseline (default limits unless given)."""
    limits = EOL_LIMITS if limits is None else limits
    # A small tolerance so a reading exactly at a limit counts despite floating-point rounding.
    tolerance = 1e-9
    causes = []
    if 100 * (1 - reading["Capacitance_uF"] / baseline["Capacitance_uF"]) >= limits["capacitance_loss_pct"] - tolerance:
        causes.append("capacitance")
    if reading["ESR_ohm"] / baseline["ESR_ohm"] >= limits["esr_ratio"] - tolerance:
        causes.append("ESR")
    if reading["Leakage_uA"] / baseline["Leakage_uA"] >= limits["leakage_ratio"] - tolerance:
        causes.append("leakage")
    if reading["Case_Expansion_um"] - baseline["Case_Expansion_um"] >= limits["expansion_um"] - tolerance:
        causes.append("expansion")
    if reading["H2_Generated_mL_STP"] - baseline["H2_Generated_mL_STP"] >= limits["h2_ml"] - tolerance:
        causes.append("H2")
    if reading["Pressure_Abs_kPa"] >= limits["pressure_abs_kpa"] - tolerance:
        causes.append("pressure")
    return causes


def simulate_device(rng: np.random.Generator, cap_id: str, fine_step_h: float = 5.0,
                    limits: dict | None = None, keep_truth: bool = False, near_end_share: float = 0.0,
                    family: dict | None = None):
    """Noisy inspections until end of life, and the true (noise-free) end-of-life age and cause.

    ``limits`` (default: ``EOL_LIMITS``) decides end of life. With ``keep_truth`` the noise-free
    trajectory is returned as well, as ``(hours, {indicator: values})``. A ``near_end_share`` of
    capacitors get one or two extra inspections in the last 3% of their life, so a model also
    learns from readings taken right next to a limit.
    """
    limits = EOL_LIMITS if limits is None else limits
    while True:
        device = _device(rng, family)
        # Long enough for the loosest limits; coarser steps for very slow capacitors keep it small, finer ones
        # for very fast ones (a short-lived family) keep end of life precise.
        horizon = 12 * max(device["tau_h"], device["tau2_h"])
        step = max(min(fine_step_h, horizon / 2000), horizon / 40000)
        fine = np.arange(0, horizon + step, step)
        truth = _state(device, fine)
        base = {name: values[0] for name, values in truth.items()}
        eol_index = next((i for i in range(1, len(fine))
                          if end_of_life_causes(base, {k: v[i] for k, v in truth.items()}, limits)), None)
        if eol_index is not None:
            break
    eol_h = float(fine[eol_index])
    causes = end_of_life_causes(base, {k: v[eol_index] for k, v in truth.items()}, limits)

    # Inspections at a device-specific interval with jitter, stopping before end of life.
    interval = eol_h / rng.uniform(12, 40)
    times = [0.0]
    while True:
        step = interval * rng.uniform(0.7, 1.3)
        if times[-1] + step >= eol_h:
            break
        times.append(times[-1] + step)
    if device["change_h"] < eol_h:
        # Conditions are changed at an inspection, which records the new ones before any aging under them
        # (the situation of a user entering planned conditions).
        times = sorted([t for t in times if abs(t - device["change_h"]) >= 1.0] + [device["change_h"]])
    if near_end_share and rng.random() < near_end_share:
        extra = eol_h * (1 - rng.uniform(0.003, 0.03, rng.integers(1, 3)))
        times += sorted(t for t in extra if t > times[-1] + 1.0)
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
    conditions = conditions_at(device, times)
    frame["Ambient_C"] = np.round(conditions["Ambient_C"] + rng.normal(0, 0.3, len(times)), 1)
    frame["Core_C"] = np.round(conditions["Core_C"] + rng.normal(0, 0.5, len(times)), 1)
    frame["Voltage_V"] = np.round(conditions["Voltage_V"], 2)
    frame["Rated_V"] = conditions["Rated_V"]
    frame["Ripple_A"] = np.round(conditions["Ripple_A"] * (1 + rng.normal(0, 0.01, len(times))), 3)
    frame["Rated_Ripple_A"] = np.round(conditions["Rated_Ripple_A"], 3)
    frame["Ripple_Hz"] = np.round(conditions["Ripple_Hz"], 0)
    frame.loc[0, "H2_Generated_mL_STP"] = 0.0
    frame.loc[0, "Case_Expansion_um"] = 0.0
    frame["H2_Generated_mL_STP"] = frame["H2_Generated_mL_STP"].clip(lower=0).cummax()
    # Expansion is a change from the first reading, so a precise sensor can report small negative
    # values early on; they are kept rather than clipped to zero.
    rounding = {"Capacitance_uF": 2, "ESR_ohm": 5, "Leakage_uA": 3, "Case_Expansion_um": 2,
                "H2_Generated_mL_STP": 4, "Pressure_Abs_kPa": 2}
    frame = frame.round(rounding)
    meta = {"Cap_ID": cap_id, "eol_h": eol_h, "eol_cause": ", ".join(causes),
            **{f"limit_{k}": float(v) for k, v in limits.items()},
            **{k: round(float(v), 4) for k, v in device.items()}}
    if keep_truth:
        return frame, meta, (fine, truth)
    return frame, meta


def simulate_fleet(n_devices: int, seed: int, prefix: str = "SIM", vary_limits: bool = False,
                   on_truth=None, near_end_share: float = 0.0,
                   family_share: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fleet of simulated capacitors with the default limits, or random ones with ``vary_limits``.

    A ``family_share`` of the capacitors come from randomly drawn families (``random_family``).

    ``on_truth(frame, meta, hours, truth)`` is called for each capacitor with its noise-free
    trajectory, which is not kept (it can be large).
    """
    rng = np.random.default_rng(seed)
    limit_rng = np.random.default_rng(seed + 10_000)
    family_rng = np.random.default_rng(seed + 20_000)  # its own stream: fleets without families stay unchanged
    frames, metas = [], []
    for index in range(1, n_devices + 1):
        limits = random_limits(limit_rng) if vary_limits else None
        family = random_family(family_rng) if family_share and family_rng.random() < family_share else None
        frame, meta, (hours, truth) = simulate_device(rng, f"{prefix}_{index:04d}", limits=limits, keep_truth=True,
                                                      near_end_share=near_end_share, family=family)
        if on_truth is not None:
            on_truth(frame, meta, hours, truth)
        frames.append(frame)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)
