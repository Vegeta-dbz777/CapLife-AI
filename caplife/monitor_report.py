"""Downloadable reports for Monitor and Predict: a PDF and an Excel workbook.

Both lead with the remaining-life predictions (fleet table and the selected capacitor), then the
limits, operating conditions and a data check, and the Excel workbook adds every table in full.
The predictions come from a demonstration model trained on simulated capacitors; both reports say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from io import BytesIO

import numpy as np
import pandas as pd

from caplife.six_rul import INDICATOR_LABELS, value_at_fraction
from caplife.six_sim import ATMOSPHERE_KPA, SIGNALS

DISCLAIMER = ("Demonstration model: the neural networks learned from simulated capacitors, not laboratory measurements. "
              "Their accuracy on real capacitors has not been tested. Not a maintenance or safety tool.")
UNITS = {"Capacitance_uF": "µF", "ESR_ohm": "Ω", "Leakage_uA": "µA", "Case_Expansion_um": "µm",
         "H2_Generated_mL_STP": "mL", "Pressure_Abs_kPa": "kPa"}
CONDITION_NAMES = {"Ambient_C": "Ambient temperature (°C)", "Core_C": "Core temperature (°C)",
                   "Voltage_V": "Applied voltage (V)", "Rated_V": "Rated voltage (V)", "Ripple_A": "Ripple current (A)",
                   "Rated_Ripple_A": "Rated ripple current (A)", "Ripple_Hz": "Operating frequency (Hz)"}
LIMIT_NAMES = {"capacitance_loss_pct": "Capacitance loss (%)", "esr_ratio": "ESR (× initial)",
               "leakage_ratio": "Leakage current (× initial)", "expansion_um": "Case expansion (µm)",
               "h2_ml": "H₂ generated (mL)", "pressure_abs_kpa": "Internal pressure (kPa, absolute)"}


@dataclass
class MonitorReport:
    """Everything Monitor and Predict shows, for one data set and one selected capacitor."""
    source: str
    fleet: pd.DataFrame
    device: str
    readings: pd.DataFrame
    assessment: pd.DataFrame
    measured: tuple[str, ...]
    limits: dict
    all_limits: dict
    recorded_conditions: pd.Series
    changed_conditions: dict = field(default_factory=dict)
    forecast: pd.DataFrame | None = None
    data_check: pd.DataFrame | None = None
    measurements: pd.DataFrame | None = None
    accuracy: dict | None = None


def _hours(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):,.0f} h"


def _detail(report: MonitorReport) -> list[tuple[str, str]]:
    last = report.assessment.iloc[-1]
    rows = [("Readings", f"{len(report.readings)} up to {float(last['Hours']):,.0f} h"),
            ("Parameters used", ", ".join(INDICATOR_LABELS[s] for s in report.measured)),
            ("Overall health", f"{float(last['Overall_health_pct']):.0f}% (limiting: {last['Limiting_indicator']})")]
    if bool(last.get("End_of_life_reached")):
        rows.append(("Remaining useful life", "0 h: an end-of-life limit is met"))
    elif pd.notna(last.get("RUL_median_h")):
        rows += [("Remaining useful life", _hours(last["RUL_median_h"])),
                 ("80% range", f"{float(last['RUL_low_h']):,.0f} – {float(last['RUL_high_h']):,.0f} h"),
                 ("Predicted end-of-life age", _hours(float(last["Hours"]) + float(last["RUL_median_h"])))]
    if report.changed_conditions:
        rows.append(("Operating conditions", "changed by the user from the latest reading on (see Conditions)"))
    return rows


def _limit_rows(limits: dict) -> list[tuple[str, str]]:
    return [(LIMIT_NAMES[key], f"{value:g}") for key, value in limits.items() if key in LIMIT_NAMES]


def _condition_table(report: MonitorReport) -> pd.DataFrame:
    rows = []
    for column, name in CONDITION_NAMES.items():
        recorded = report.recorded_conditions.get(column, np.nan)
        changed = report.changed_conditions.get(column)
        if pd.notna(recorded) or changed is not None:
            rows.append({"Condition": name, "Recorded (latest)": "—" if pd.isna(recorded) else f"{recorded:,.4g}",
                         "Used for the prediction": f"{(changed if changed is not None else recorded):,.4g}"})
    return pd.DataFrame(rows, columns=["Condition", "Recorded (latest)", "Used for the prediction"])


# ---------------------------------------------------------------- charts (PNG, for the PDF)

def _parameter_png(report: MonitorReport) -> bytes:
    from matplotlib.figure import Figure
    signals = list(report.measured)
    columns = 3 if len(signals) > 4 else 2
    rows = int(np.ceil(len(signals) / columns))
    fig = Figure(figsize=(11, 3.0 * rows), dpi=130)
    axes = np.atleast_1d(fig.subplots(rows, columns)).ravel()
    baseline = report.readings.iloc[0]
    latest = float(report.readings["Hours"].iloc[-1])
    for ax, signal in zip(axes, signals):
        ax.plot(report.readings["Hours"], report.readings[signal], "o-", color="#2f9e8f", ms=3, lw=1.4,
                label="Measured")
        if report.forecast is not None and len(report.forecast):
            path = report.forecast.loc[report.forecast["signal"].eq(signal)]
            ax.fill_between(path["Hours"], path["low"], path["high"], color="#8e6bd8", alpha=.22, lw=0)
            ax.plot(path["Hours"], path["median"], "--", color="#8e6bd8", lw=1.8, label="Forecast")
        limit = float(value_at_fraction(signal, baseline, 1.0, report.limits))
        ax.axhline(limit, color="#d9822b", ls=":", lw=1.3, label="Limit")
        ax.axvline(latest, color="#9fb0bd", lw=1.2)
        ax.set_title(f"{INDICATOR_LABELS[signal]} ({UNITS[signal]})", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.25)
    for ax in axes[len(signals):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()


def _rul_png(report: MonitorReport) -> bytes | None:
    data = report.assessment.dropna(subset=["RUL_median_h"])
    if data.empty:
        return None
    from matplotlib.figure import Figure
    fig = Figure(figsize=(11, 3.0), dpi=130)
    ax = fig.subplots()
    ax.fill_between(data["Hours"], data["RUL_low_h"], data["RUL_high_h"], color="#8e6bd8", alpha=.2, lw=0)
    ax.plot(data["Hours"], data["RUL_median_h"], "o-", color="#8e6bd8", ms=3, lw=1.4, label="Predicted at each reading")
    last = data.iloc[-1]
    latest, mid = float(last["Hours"]), float(last["RUL_median_h"])
    if mid > 0:
        ax.plot([latest, latest + mid], [mid, 0], "--", color="#8e6bd8", lw=1.6, label="Countdown to end of life")
        ax.axvspan(latest + float(last["RUL_low_h"]), latest + float(last["RUL_high_h"]), color="#9aa5b1", alpha=.18,
                   label="End of life, 80% range")
    ax.axvline(latest, color="#9fb0bd", lw=1.2)
    ax.set_xlabel("Capacitor age (h)", fontsize=8)
    ax.set_ylabel("Remaining useful life (h)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getvalue()


# ---------------------------------------------------------------- PDF

def pdf_bytes(report: MonitorReport) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(buffer, pagesize=(page_w, page_h), leftMargin=32, rightMargin=32, topMargin=30,
                            bottomMargin=30, title="CapLife AI · Monitor and Predict", author="CapLife AI")
    width = page_w - 64
    styles = getSampleStyleSheet()
    navy, light = colors.HexColor("#15243D"), colors.HexColor("#E8F2F7")
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=21, leading=25, textColor=navy, alignment=0)
    head = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12.5, textColor=navy, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=8.8, leading=12, textColor=colors.HexColor("#334B62"))
    cell = ParagraphStyle("c", parent=styles["Normal"], fontSize=7.4, leading=9.5)
    bold = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold", textColor=navy)
    text = lambda value: escape(str(value)).replace("₂", "2").replace("Ω", "ohm").replace("µ", "u")

    def grid(rows, widths, header=True):
        table = Table([[Paragraph(text(v), bold if header and i == 0 else cell) for v in row]
                       for i, row in enumerate(rows)], colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        style = [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F5F8FB")]),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF3")))
        table.setStyle(TableStyle(style))
        return table

    fleet = report.fleet
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [Paragraph("CapLife AI · Monitor and Predict", title),
             Paragraph(text(f"{report.source} · generated {stamp}"), body), Spacer(1, 6)]
    attention = int(fleet["Status"].isin(["Watch", "Critical", "End of life"]).sum())
    cards = Table([[f"CAPACITORS  {len(fleet)}", f"NEED ATTENTION  {attention}",
                    f"SHORTEST REMAINING LIFE  {_hours(fleet['Remaining life, median (h)'].min())}"]],
                  colWidths=[width / 3] * 3, rowHeights=[30])
    cards.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), light), ("TEXTCOLOR", (0, 0), (-1, -1), navy),
                               ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [cards, Spacer(1, 6), Paragraph(text(DISCLAIMER), body)]
    if report.accuracy:
        story.append(Paragraph(text(
            f"Typical error on simulated test capacitors with these parameters: "
            f"{report.accuracy['median_absolute_error_h']:,.0f} h ({report.accuracy['median_relative_error']:.0%} of "
            f"remaining life); the 80% range held the true value {report.accuracy['interval_coverage']:.0%} of the time."),
            body))

    story.append(Paragraph("Fleet overview", head))
    columns = ["Capacitor", "Parameters", "Latest reading (h)", "Overall health (%)", "Limiting indicator", "Status",
               "Remaining life, median (h)", "Remaining life, 80% range (h)", "Predicted end-of-life age (h)"]
    shown = [c for c in columns if c in fleet]
    rows = [["Remaining life (h)" if c == "Remaining life, median (h)" else "80% range (h)"
             if c == "Remaining life, 80% range (h)" else "End-of-life age (h)" if c.startswith("Predicted") else c
             for c in shown]]
    for record in fleet[shown].itertuples(index=False):
        rows.append([("—" if pd.isna(v) else f"{v:,.0f}") if isinstance(v, (int, float, np.floating, np.integer))
                     else v for v in record])
    story.append(grid(rows, [width / len(shown)] * len(shown)))

    story += [PageBreak(), Paragraph(text(f"Capacitor {report.device}"), title)]
    left = grid([("Item", "Value"), *_detail(report)], [140, 250])
    right = grid([("End-of-life limit", "Value"), *_limit_rows(report.limits)], [170, 80])
    story.append(Table([[left, right]], colWidths=[width * 0.58, width * 0.42], hAlign="LEFT",
                       style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
    conditions = _condition_table(report)
    if len(conditions):
        story.append(Paragraph("Operating conditions", head))
        story.append(grid([list(conditions.columns), *conditions.values.tolist()], [200, 120, 140]))
    last = report.assessment.iloc[-1]
    health = [("Parameter", "Latest value", "Health")]
    for signal in report.measured:
        health.append((INDICATOR_LABELS[signal], f"{float(report.readings[signal].iloc[-1]):,.4g} {UNITS[signal]}",
                       f"{float(last[f'{signal}_health_pct']):.0f}%"))
    story += [Paragraph("Parameter health", head), grid(health, [170, 140, 80])]

    story += [PageBreak(), Paragraph("Parameter history and forecast", head)]
    png = _parameter_png(report)
    image_height = min(page_h - 120, width * (3.0 * int(np.ceil(len(report.measured) / (3 if len(report.measured) > 4 else 2)))) / 11)
    story.append(Image(BytesIO(png), width=width, height=image_height))
    rul = _rul_png(report)
    if rul is not None:
        story += [PageBreak(), Paragraph("Remaining-life prediction over time", head),
                  Image(BytesIO(rul), width=width, height=width * 3.0 / 11),
                  Paragraph(text("Left of the grey line: the prediction made at each reading from the readings "
                                 "available then. Right of it: the countdown from the latest reading to the predicted "
                                 "end of life, with the end of life's 80% range shaded."), body)]

    if report.data_check is not None and len(report.data_check):
        story += [PageBreak(), Paragraph("Data check", head),
                  Paragraph(text("What could limit each capacitor's assessment: missing readings, no 0 h reading, "
                                 "sampling gaps or changing ESR test settings."), body), Spacer(1, 4)]
        check = report.data_check
        rows = [["Capacitor", "Readiness", "What limits it", "Readings", "Last reading (h)"]]
        for record in check.itertuples(index=False):
            rows.append([record.Cap_ID, record.Data_readiness, record.Limits, f"{record.Measurements}",
                         f"{record.Last_hour:,.0f}"])
        story.append(grid(rows, [110, 90, width - 110 - 90 - 70 - 90, 70, 90]))
    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------- Excel

def xlsx_bytes(report: MonitorReport) -> bytes:
    import xlsxwriter

    memory = BytesIO()
    book = xlsxwriter.Workbook(memory, {"in_memory": True, "strings_to_formulas": False, "strings_to_urls": False,
                                        "nan_inf_to_errors": True})
    navy, blue = "#15243D", "#226D93"
    title = book.add_format({"bold": True, "font_size": 18, "font_color": "white", "bg_color": navy})
    header = book.add_format({"bold": True, "font_color": "white", "bg_color": blue, "text_wrap": True,
                              "valign": "vcenter"})
    note = book.add_format({"font_color": "#8B3E06", "bg_color": "#FFF4DD", "text_wrap": True, "valign": "vcenter"})
    label = book.add_format({"bold": True, "font_color": navy})

    overview = book.add_worksheet("Overview")
    overview.set_column("A:A", 30)
    overview.set_column("B:B", 60)
    overview.merge_range("A1:B1", "CapLife AI · Monitor and Predict", title)
    overview.set_row(0, 28)
    info = [("Data", report.source), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            ("Capacitors", len(report.fleet)), ("Selected capacitor", report.device), *_detail(report)]
    for row, (key, value) in enumerate(info, start=2):
        overview.write(row, 0, key, label)
        overview.write(row, 1, value)
    overview.merge_range(len(info) + 3, 0, len(info) + 5, 1, DISCLAIMER, note)

    def sheet(name: str, frame: pd.DataFrame) -> None:
        tab = book.add_worksheet(name[:31])
        tab.freeze_panes(1, 0)
        for index, column in enumerate(frame.columns):
            tab.write(0, index, str(column), header)
            tab.set_column(index, index, min(40, max(12, len(str(column)) + 2)))
        for row, record in enumerate(frame.itertuples(index=False), start=1):
            for index, value in enumerate(record):
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    tab.write_blank(row, index, None)
                elif isinstance(value, (np.integer, np.floating)):
                    tab.write_number(row, index, float(value))
                elif isinstance(value, (bool, np.bool_)):
                    tab.write_boolean(row, index, bool(value))
                else:
                    tab.write(row, index, value)

    sheet("Fleet", report.fleet)
    detail = report.readings.merge(report.assessment, on="Hours", how="left")
    sheet(f"{report.device} history", detail)
    if report.forecast is not None and len(report.forecast):
        sheet(f"{report.device} forecast", report.forecast.rename(columns={"signal": "Parameter"}))
    limits = pd.DataFrame([{"Capacitor": device, **{LIMIT_NAMES[k]: v for k, v in values.items() if k in LIMIT_NAMES}}
                           for device, values in report.all_limits.items()])
    sheet("Limits", limits)
    conditions = _condition_table(report)
    if len(conditions):
        sheet(f"{report.device} conditions", conditions)
    if report.data_check is not None and len(report.data_check):
        sheet("Data check", report.data_check)
    if report.measurements is not None:
        sheet("Measurements", report.measurements)
    book.close()
    return memory.getvalue()


__all__ = ["MonitorReport", "pdf_bytes", "xlsx_bytes", "ATMOSPHERE_KPA", "SIGNALS"]
