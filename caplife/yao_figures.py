"""Digitize plotted observation markers in Figures 7 and 8 of Bo et al. (2023).

These are approximate read-outs of published markers. Interpolated guide curves
are excluded. The original raster figure coordinates are retained for auditing.
"""
from pathlib import Path
import csv
import hashlib
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label

SOURCE_URL = "https://vbn.aau.dk/ws/portalfiles/portal/639038831/letter_3.26.pdf"
EXPECTED_SHA256 = "00a5144f9e761244df97c0112bed0cd5dbc1480e1846c93fef554bdbff3905b3"
# Approximate placement references on the 1600 px wide views. At extraction time,
# actual axis positions, tic positions, and marker centers come from the PNGs.
PANELS = {
    7: [
        (1, 545, 736, 22, 185, 45, 170, 200),
        (2, 542, 736, 490, 657, 514, 642, 200),
        (3, 542, 738, 950, 1118, 974, 1102, 200),
        (4, 1350, 1539, 22, 183, 45, 170, 200),
        (5, 1345, 1534, 493, 653, 517, 638, 200),
        (6, 1346, 1534, 951, 1110, 975, 1097, 200),
    ],
    8: [
        (1, 535, 739, 31, 197, 53, 181, 200),
        (2, 521, 725, 505, 678, 530, 662, 200),
        (3, 1333, 1537, 25, 197, 50, 182, 200),
        (4, 1331, 1537, 504, 679, 529, 662, 200),
    ],
}


def peak(values, above, below):
    return int(np.argmax(values[above:below + 1])) + above


def runs(mask):
    indexes = np.flatnonzero(mask)
    return [g for g in np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1) if len(g)]


def calibrate(rgb, panel):
    _, approx_l, approx_r, approx_t, approx_b, approx_100c, approx_100r, increment = panel
    sx = rgb.shape[1] / 1600
    l0, r0, t0, b0, c1000, r1000 = [round(a * sx) for a in (approx_l, approx_r, approx_t, approx_b, approx_100c, approx_100r)]
    red, green, blue = rgb.transpose(2, 0, 1)
    dark = rgb.max(axis=2) < 94
    blue_axis = (blue > 100) & (green > 32) & (green < 180) & (blue - red > 70)
    red_axis = (red > 120) & (green > 22) & (green < 182) & (red - blue > 85)
    top = peak(dark[:, l0 + 10:r0 - 10].sum(axis=1), t0 - 6, t0 + 6)
    bot = peak(dark[:, l0 + 10:r0 - 10].sum(axis=1), b0 - 6, b0 + 6)
    lft = peak(blue_axis[top + 5:bot - 5, :].sum(axis=0), l0 - 5, l0 + 5)
    rgt = peak(red_axis[top + 5:bot - 5, :].sum(axis=0), r0 - 5, r0 + 5)
    cap_ref_y = peak(blue_axis[:, lft + 2:lft + 7].sum(axis=1), c1000 - 5, c1000 + 5)
    esr_ref_y = peak(red_axis[:, rgt - 8:rgt - 1].sum(axis=1), r1000 - 7, r1000 + 7)
    bottom_ticks = dark[bot - 6:bot - 1].sum(axis=0) >= 1
    bottom_ticks[:lft + 10] = False
    bottom_ticks[rgt - 2:] = False
    tick_centres = [float(np.mean(g)) for g in runs(bottom_ticks)]
    # Restrict ticks to the regular x-axis spacing and keep only groups confirmed
    # by more than one candidate for regular spacing where possible.
    return lft, rgt, top, bot, cap_ref_y, esr_ref_y, tick_centres


def marker_centres(rgb, rectangle, color):
    lft, rgt, top, bot = rectangle
    x1, x2 = lft - 16, rgt + 5
    y1, y2 = top - 4, bot + 16
    crop = rgb[y1:y2, x1:x2]
    red, green, blue = crop.transpose(2, 0, 1)
    if color == "blue":
        points = (blue > 180) & (red < 100) & (green < 82)
    else:
        points = (red > 196) & (green < 38) & (blue < 70)
    labels, count = label(binary_dilation(points, iterations=1), structure=np.ones((3, 3), int))
    markers = []
    for k in range(1, count + 1):
        ys, xs = np.where((labels == k) & points)
        if len(xs) < 19:
            continue
        width = int(xs.max() - xs.min())
        height = int(ys.max() - ys.min())
        if 8 <= width <= 28 and 8 <= height <= 29:
            markers.append((round((xs.min() + xs.max()) / 2 + x1, 1),
                            round((ys.min() + ys.max()) / 2 + y1, 1), len(xs)))
    return sorted(markers)


# Figure panel labels were verified directly against the source images.
# The terminal annotation encodes test termination at catastrophic failure for
# these ordinary specimens. Whole-hour labels have unspecified rounding.
# Figure 9 only establishes "open circuit" at the 17 h electrical check.
EVENTS = [
    (7, 1, 1, 572, 572, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 545.2, 545.4, "terminal arrow 572 h"),
    (7, 2, 1, 468, 468, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 420.5, 421.5, "terminal arrow 468 h"),
    (7, 3, 1, 642, 642, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 560.0, 561.0, "terminal arrow 642 h"),
    (7, 4, 2, 581, 581, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 527.7, 527.9, "terminal arrow 581 h"),
    (7, 5, 2, 491, 492, "bracket_from_terminal_annotations", "explosion_electrical_mode_not_stated", 380.8, 381.2, "electrical/force plot ends 491 h; thermal trace and explosion documented at end 492 h"),
    (7, 6, 2, 726, 726, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 595.0, 596.0, "terminal arrow 726 h"),
    (8, 1, 1, 1177, 1177, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 1078.7, 1078.9, "terminal arrow 1177 h"),
    (8, 2, 1, 963, 963, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 883.7, 883.9, "terminal arrow 963 h"),
    (8, 3, 2, 811, 811, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 724.5, 724.6, "terminal arrow 811 h"),
    (8, 4, 2, 866, 866, "annotated_terminal_hour", "open_or_short_not_stated_by_device", 734.2, 734.6, "terminal arrow 866 h"),
    (9, 3, "", 0, 17, "bracket_from_electrical_checks", "open_circuit", 12.453, 12.557, "case 3 sample 3 identified in fig. 10; 0 h normal and 17 h open, no intermediate direct electrical measurement"),
]


def matched_esr(rgb, x, rectangle):
    _, _, top, bot = rectangle
    x1, x2 = int(x) - 14, int(x) + 15
    crop = rgb[top - 4:bot + 16, x1:x2]
    red, green, blue = crop.transpose(2, 0, 1)
    mask = (red > 196) & (green < 53) & (blue < 70)
    ys, xs = np.where(mask)
    if len(xs) < 20 or not 8 <= np.ptp(xs) <= 28 or not 8 <= np.ptp(ys) <= 29:
        raise ValueError(f"Could not confirm ESR cross near x={x}")
    return (xs.min() + xs.max()) / 2 + x1, (ys.min() + ys.max()) / 2 + top - 4


def unit_id(figure, device):
    return f"yao_case_{figure - 6}_sample_{device}"


def read_markers(rgb, panel, figure):
    device = panel[0]
    lft, rgt, top, bot, cap_axis_y, esr_axis_y, ticks = calibrate(rgb, panel)
    axes = lft, rgt, top, bot
    input_points = marker_centres(rgb, axes, "blue")
    paired_esr = [matched_esr(rgb, x, axes) for x, _, _ in input_points]
    if len(input_points) < 4 or len(ticks) < 2:
        raise ValueError(f"Too few visible marks or axis ticks for figure {figure} sample {device}")
    hours_per_px = panel[-1] / np.median(np.diff([lft, *ticks]))
    cap100_y = input_points[0][1]
    esr100_y = paired_esr[0][1]
    if abs(cap100_y - cap_axis_y) > 6 or abs(esr100_y - esr_axis_y) > 8:
        raise ValueError(f"Broken 100%-tick reference for figure {figure} sample {device}")
    for i, ((cap_x, cap_y, _), (esr_x, esr_y)) in enumerate(zip(input_points, paired_esr)):
        if abs(cap_x - esr_x) > 8:
            raise ValueError(f"Unmatched marker in figure {figure} sample {device}")
        # Blue circles are partly occluded where the two curves cross.
        # The red crosses provide the more stable timing reference.
        age = 0 if i == 0 else (esr_x - lft) * hours_per_px
        yield {
            "device_id": unit_id(figure, device), "case": figure - 6,
            "sample": device, "figure": figure,
            "test_age_h": round(float(age), 2),
            "capacitance_pct_of_initial": round(100 + 30 * (cap100_y - cap_y) / (bot - cap100_y), 2),
            "esr_pct_of_initial": round(100 + 200 * (esr100_y - esr_y) / (esr100_y - top), 2),
            "capacitance_readout_x_px": cap_x, "capacitance_readout_y_px": cap_y,
            "esr_readout_x_px": esr_x, "esr_readout_y_px": esr_y,
            "provenance": "digitized_published_observation_markers",
        }


def write_csv(path, rows):
    with path.open("w", newline="") as of:
        writer = csv.DictWriter(of, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    out.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("The PDF differs from the audited source.")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "figure"
        subprocess.run(["pdfimages", "-png", "-f", "4", "-l", "5", str(pdf), str(target)], check=True)
        for i, figure in enumerate(PANELS):
            rgb = np.array(Image.open(f"{target}-{i * 2:03}.png").convert("RGB"), dtype=np.int32)
            for panel in PANELS[figure]:
                rows.extend(read_markers(rgb, panel, figure))
    write_csv(out / "yao_digitized_electrical.csv", rows)
    write_csv(out / "yao_physical_endpoints.csv", [
        {
            "device_id": unit_id(figure, device), "case": figure - 6, "sample": device,
            "type": type_id, "figure": figure, "failure_age_lower_h": lower,
            "failure_age_upper_h": upper, "event_time_precision": precision,
            "physical_failure_mode": cause, "vent_inset_lower_h": vent_lower,
            "vent_inset_upper_h": vent_upper, "evidence": evidence,
            "doi": "10.1109/TPEL.2023.3263125",
        }
        for figure, device, type_id, lower, upper, precision, cause, vent_lower, vent_upper, evidence in EVENTS
    ])
    counts = {row["device_id"]: 0 for row in rows}
    for row in rows:
        counts[row["device_id"]] += 1
    print(f"{len(rows)} paired plotted observations across {len(counts)} devices", counts)
    print(f"{len(EVENTS)} physical endpoints, including 2 conservatively bracketed")


if __name__ == "__main__":
    main()
