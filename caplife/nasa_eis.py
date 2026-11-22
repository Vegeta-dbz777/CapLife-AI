"""Extract reference-age capacitance histories from the public NASA EIS archive."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/12.+Capacitor+Electrical+Stress.zip"
SOURCE_URL = "https://data.phmsociety.org/nasa/#12-capacitor-electrical-stress"
CAMPAIGN = ("ES10", "ES12", "ES14")


def matlab_text(dataset) -> str:
    return "".join(chr(int(code)) for code in dataset[()].ravel() if code).strip()


def reference_schedule(file, eis) -> list[tuple[str, float]]:
    table = eis["EIS_Reference_Table"][()]
    schedule = [
        (datetime.strptime(matlab_text(file[date_ref]), "%m/%d/%Y").date().isoformat(),
         float(matlab_text(file[age_ref])))
        for date_ref, age_ref in zip(table[0], table[3], strict=True)
    ]
    ages = np.array([age for _, age in schedule])
    if not np.isfinite(ages).all() or ages[0] != 0 or (np.diff(ages) < 0).any():
        raise ValueError("NASA's EIS reference ages must increase from zero.")
    return schedule


def acquisition_date(header) -> str | None:
    if header.dtype != np.dtype("uint16") or header.ndim != 2:
        return None
    for column in header[()].T:
        line = "".join(chr(int(code)) for code in column if code).strip()
        label, _, value = line.partition(":")
        if label.strip().lower() == "acquisition started on":
            date = value.strip().split()[0]
            return datetime.strptime(date, "%m/%d/%Y").date().isoformat()
    return None


def spectrum_values(matrix: np.ndarray) -> tuple[float, float, float]:
    """Read series capacitance and real impedance, plus a low-frequency RC check."""
    if matrix.ndim != 2 or matrix.shape[0] < 10:
        return np.nan, np.nan, np.nan
    frequency, real_z, minus_imag_z, magnitude = matrix[:4]
    valid = np.isfinite(frequency) & (frequency > 0)
    valid &= np.isfinite(real_z) & np.isfinite(minus_imag_z) & np.isfinite(magnitude)
    if not valid.any() or not np.allclose(
        np.hypot(real_z[valid], minus_imag_z[valid]), magnitude[valid], rtol=0.01, atol=0.001
    ):
        return np.nan, np.nan, np.nan

    def near(target: float, signal: np.ndarray) -> float:
        indices = np.flatnonzero(valid)
        index = indices[np.abs(frequency[indices] - target).argmin()]
        value = signal[index]
        close_enough = abs(frequency[index] / target - 1) <= 0.12
        return float(value) if np.isfinite(value) and value > 0 and close_enough else np.nan

    capacitance = near(100, matrix[8])
    resistance = near(10_000, real_z)
    band = valid & (frequency >= 20) & (frequency <= 2_000) & (minus_imag_z > 0)
    fitted_capacitance = np.nan
    if band.sum() >= 3:
        x = 1 / (2 * np.pi * frequency[band])
        fitted_capacitance = float(1e6 * (x @ x) / (x @ minus_imag_z[band]))
    return capacitance, resistance, fitted_capacitance


def extract_eis(path: Path, cohort: str) -> pd.DataFrame:
    import h5py  # Required only for regenerating the research measurement tables.

    rows = []
    with h5py.File(path, "r") as file:
        eis = file[f"{cohort}/EIS_Data"]
        schedule = reference_schedule(file, eis)
        for device in sorted(key for key in eis if key.startswith(f"{cohort}C")):
            measurements = eis[device]["EIS_Measurement"]
            headers, spectra = (measurements[key][()].ravel() for key in ("Header", "Data"))
            for entry, ((date, age), header_ref, spectrum_ref) in enumerate(zip(schedule, headers, spectra, strict=True)):
                values, unmatched = [], 0
                if header_ref and spectrum_ref:
                    scan_headers, scan_spectra = (file[ref][()].ravel() for ref in (header_ref, spectrum_ref))
                    for header, spectrum in zip(scan_headers, scan_spectra, strict=True):
                        if spectrum and h5py.check_dtype(ref=file[spectrum].dtype) is None:
                            matrix = file[spectrum]
                            if matrix.ndim != 2 or matrix.shape[0] < 10:
                                continue
                            if not header or acquisition_date(file[header]) != date:
                                unmatched += 1
                                continue
                            values.append(spectrum_values(matrix[()].astype(float)))
                medians = [
                    float(np.median(valid)) if len(valid := column[np.isfinite(column)]) else np.nan
                    for column in np.array(values, dtype=float).reshape(-1, 3).T
                ]
                rows.append({
                    "Cap_ID": device, "Entry": entry, "Age_h": age, "Reference_date": date,
                    "Capacitance_uF": medians[0], "Resistance_Ohm": medians[1],
                    "Fitted_capacitance_uF": medians[2], "Matched_scans": len(values),
                    "Unmatched_spectra": unmatched,
                })
    result = pd.DataFrame(rows)
    if result.empty or result["Capacitance_uF"].isna().all():
        raise ValueError(f"No usable capacitor histories found for {cohort}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat", type=Path)
    parser.add_argument("cohort", choices=CAMPAIGN)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract_eis(args.mat, args.cohort).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
