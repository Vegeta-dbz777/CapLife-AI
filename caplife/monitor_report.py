"""Downloadable reports for Monitor and Predict: a PDF and an Excel workbook.

The PDF is the page as the user sees it: the fleet overview, then the selected capacitor with the
limits and operating conditions in its input boxes, its health cards and the page's own charts
(exported from the same Altair charts), in the page's order. The Excel workbook has every table in
full. The predictions come from a demonstration model trained on simulated capacitors; both say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from caplife.six_rul import INDICATOR_LABELS

DISCLAIMER = ("Demonstration model: the neural networks learned from simulated capacitors, not laboratory measurements. "
              "Their accuracy on real capacitors has not been tested. Not a maintenance or safety tool.")
UNITS = {"Capacitance_uF": "µF", "ESR_ohm": "Ω", "Leakage_uA": "µA", "Case_Expansion_um": "µm",
         "H2_Generated_mL_STP": "mL", "Pressure_Abs_kPa": "kPa"}
CONDITION_NAMES = {"Ambient_C": "Ambient temperature (°C)", "Core_C": "Core temperature (°C)",
                   "Voltage_V": "Applied voltage (V)", "Rated_V": "Rated voltage (V)", "Ripple_A": "Ripple current (A)",
                   "Rated_Ripple_A": "Rated ripple current (A)", "Ripple_Hz": "Operating frequency (Hz)"}
# As the limit boxes on the page show them: ESR and leakage as a rise in percent.
LIMIT_NAMES = {"capacitance_loss_pct": "Capacitance loss (%)", "esr_ratio": "ESR rise (%)",
               "leakage_ratio": "Leakage current rise (%)", "expansion_um": "Case expansion (µm)",
               "h2_ml": "H₂ generated (mL)", "pressure_abs_kpa": "Internal pressure (kPa, absolute)"}


def _shown_limit(key: str, value: float) -> float:
    return round((value - 1) * 100, 6) if key in ("esr_ratio", "leakage_ratio") else float(value)


@dataclass
class ChartSection:
    """A chart section of the page: its title, the captions above the charts, and (caption, chart) pairs."""
    title: str
    notes: list[str]
    charts: list[tuple[str, object]]
    columns: int = 2


@dataclass
class PageView:
    """What Monitor and Predict shows for the selected capacitor, in the page's order and wording."""
    source: str
    fleet_metrics: list[tuple[str, str]]
    fleet: pd.DataFrame
    fleet_note: str | None
    device: str
    parameter_note: str | None
    limits: list[tuple[str, str, bool]]                 # label, value as in the box, measured
    conditions: list[tuple[str, str, str | None]]        # label, value in use, recorded value when changed
    badges: list[tuple[str, str]]                        # colour, text
    messages: list[tuple[str, str]]                      # error / warning / info, text
    metrics: list[tuple[str, str, str | None]]           # label, value, change
    metrics_note: str | None
    health_cards: list[dict]
    sections: list[ChartSection]
    model: str = "Built-in model"                         # which model gave remaining life and trends
    plan: pd.DataFrame | None = None                      # the maintenance planning list
    plan_horizon: float | None = None
    disclaimer: str = DISCLAIMER


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
    view: PageView | None = None


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
    return [(LIMIT_NAMES[key], f"{_shown_limit(key, value):g}") for key, value in limits.items() if key in LIMIT_NAMES]


def _condition_table(report: MonitorReport) -> pd.DataFrame:
    rows = []
    for column, name in CONDITION_NAMES.items():
        recorded = report.recorded_conditions.get(column, np.nan)
        changed = report.changed_conditions.get(column)
        if pd.notna(recorded) or changed is not None:
            rows.append({"Condition": name, "Recorded (latest)": "—" if pd.isna(recorded) else f"{recorded:,.4g}",
                         "Used for the prediction": f"{(changed if changed is not None else recorded):,.4g}"})
    return pd.DataFrame(rows, columns=["Condition", "Recorded (latest)", "Used for the prediction"])


# ---------------------------------------------------------------- charts (the page's own, as PNG)

def _fonts_dir() -> str:
    """DejaVu fonts shipped with matplotlib: they have µ, Ω, ₂ and arrows, on any server."""
    import matplotlib
    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf")


_FONTS_READY = False


def _register_fonts() -> None:
    global _FONTS_READY
    if _FONTS_READY:
        return
    import vl_convert
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    folder = Path(_fonts_dir())
    pdfmetrics.registerFont(TTFont("DejaVu", str(folder / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(folder / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
    vl_convert.register_font_directory(str(folder))
    _FONTS_READY = True


def chart_png(chart, width: int) -> bytes:
    """An Altair chart from the page as a PNG, in the page's light theme (hover helpers stay invisible)."""
    import vl_convert
    _register_fonts()
    text, grid = "#31333F", "#E6EAF1"
    import altair as alt
    styled = (chart.properties(width=width)
              .configure(font="DejaVu Sans", background="white")
              .configure_view(stroke=None)
              .configure_axis(labelColor=text, titleColor=text, gridColor=grid, domainColor=grid, tickColor=grid,
                              labelFontSize=11, titleFontSize=11, titleFontWeight="normal", labelPadding=4)
              .configure_text(font="DejaVu Sans"))
    # Streamlit swaps Altair's global data transformer while it draws a chart, under this lock; a report
    # built meanwhile (the download runs as the page redraws) would otherwise lose the charts' data.
    try:
        from streamlit.elements.vega_charts import _altair_globals_lock as lock
    except ImportError:  # pragma: no cover - other Streamlit versions
        from contextlib import nullcontext
        lock = nullcontext()
    with lock, alt.data_transformers.enable("default"):
        spec = styled.to_dict()
    return vl_convert.vegalite_to_png(spec, scale=2)


# ---------------------------------------------------------------- PDF

STATUS_COLOURS = {"good": "#0f7a5f", "watch": "#a86200", "critical": "#b42318", "end-of-life": "#b42318"}
BADGE_COLOURS = {"red": ("#FDECEA", "#B42318"), "orange": ("#FFF1DE", "#A86200"), "green": ("#E3F4EE", "#0F7A5F"),
                 "violet": ("#EFE9FB", "#6B4BB8"), "gray": ("#EEF1F4", "#4A5565")}
MESSAGE_COLOURS = {"error": ("#FDECEA", "#B42318"), "warning": ("#FFF6E0", "#8A5300"), "info": ("#E7F0FB", "#1F5FAD")}


def _markup(text: str) -> str:
    """Page text (with **bold** and line breaks) as reportlab paragraph markup."""
    parts = escape(str(text)).split("**")
    return "".join(f"<b>{part}</b>" if index % 2 else part for index, part in enumerate(parts)).replace("\n", "<br/>")


def _fleet_cell(column: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if column == "Overall health (%)":
        return f"{float(value):.0f}%"
    if column in ("Voltage / rated", "Ripple / rated"):
        return f"{float(value):.0f}%"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.0f}"
    return str(value)


def pdf_bytes(report: MonitorReport) -> bytes:
    from reportlab.graphics.shapes import Circle, Drawing, PolyLine, Polygon, Rect
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    view = report.view
    if view is None:
        raise ValueError("The PDF report is built from the page as shown; open the capacitor on the page first.")
    _register_fonts()
    buffer = BytesIO()
    page_w, page_h = landscape(A4)
    margin = 30
    doc = SimpleDocTemplate(buffer, pagesize=(page_w, page_h), leftMargin=margin, rightMargin=margin, topMargin=28,
                            bottomMargin=28, title=f"CapLife AI · Monitor and Predict · {view.device}",
                            author="CapLife AI")
    width = page_w - 2 * margin
    ink, muted, line = colors.HexColor("#31333F"), colors.HexColor("#5F6B7A"), colors.HexColor("#D5DDE5")
    style = lambda name, size, **kw: ParagraphStyle(name, fontName=kw.pop("font", "DejaVu"), fontSize=size,
                                                    leading=kw.pop("leading", size * 1.3), textColor=kw.pop("color", ink),
                                                    **kw)
    title = style("title", 20, font="DejaVu-Bold", leading=24)
    h2 = style("h2", 14, font="DejaVu-Bold", spaceBefore=12, spaceAfter=6)
    h3 = style("h3", 11.5, font="DejaVu-Bold", spaceBefore=10, spaceAfter=4)
    body = style("body", 8.6)
    caption = style("caption", 8, color=muted)
    small = style("small", 7.4, color=muted)
    cell = style("cell", 7.2, leading=9)
    head_cell = style("head", 7.2, leading=9, color=muted)
    label = style("label", 8, color=ink)
    value_style = style("value", 9)
    metric_label = style("mlabel", 8, color=muted)
    metric_value = style("mvalue", 17, leading=21)
    para = lambda text, st=body: Paragraph(_markup(text), st)

    def box(content, padding=8, border=line, top=None):
        frame = Table([[content]], colWidths=[width], hAlign="LEFT")
        rules = [("BOX", (0, 0), (-1, -1), 0.8, border), ("LEFTPADDING", (0, 0), (-1, -1), padding),
                 ("RIGHTPADDING", (0, 0), (-1, -1), padding), ("TOPPADDING", (0, 0), (-1, -1), padding),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), padding)]
        frame.setStyle(TableStyle(rules))
        return frame

    def metric_row(items):
        cells = []
        for name, value, change in items:
            content = [Paragraph(_markup(name), metric_label), Paragraph(_markup(value), metric_value)]
            if change:
                colour = "#0F7A5F" if change.strip().startswith("+") else "#B42318"
                content.append(Paragraph(_markup(change), style("delta", 8, color=colors.HexColor(colour))))
            cells.append(content)
        gap = 8
        column = (width - gap * (len(cells) - 1)) / len(cells)
        row = Table([cells], colWidths=[column] * len(cells), hAlign="LEFT")
        row.setStyle(TableStyle([("BOX", (i, 0), (i, 0), 0.8, line) for i in range(len(cells))]
                                + [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                   ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        return row

    def input_grid(heading, boxes):
        """Input boxes three to a row, like the page: label above, value in a box."""
        inner = width - 16
        column = (inner - 2 * 10) / 3
        rows = []
        for start in range(0, len(boxes), 3):
            cells = []
            for name, value, extra in boxes[start:start + 3]:
                field = Table([[Paragraph(_markup(value), value_style)]], colWidths=[column])
                field.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, line),
                                           ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                parts = [Paragraph(_markup(name), label), Spacer(1, 2), field]
                if extra:
                    parts.append(Paragraph(_markup(extra), small))
                cells.append(parts)
            cells += [""] * (3 - len(cells))
            rows.append(cells)
        grid = Table(rows, colWidths=[column + 10, column + 10, column - 4])
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return box([Paragraph(_markup(f"**{heading}**"), body), Spacer(1, 6), grid])

    def badge_row(badges):
        if not badges:
            return None
        from reportlab.pdfbase.pdfmetrics import stringWidth
        cells, widths = [], []
        for colour, text in badges:
            background, foreground = BADGE_COLOURS.get(colour, BADGE_COLOURS["gray"])
            chip_width = stringWidth(text, "DejaVu", 7.6) + 14
            chip = Table([[Paragraph(_markup(text), style("chip", 7.6, color=colors.HexColor(foreground)))]],
                         colWidths=[chip_width])
            chip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                                      ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                      ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
            cells.append(chip)
            widths.append(chip_width + 6)
        row = Table([cells], colWidths=widths, hAlign="LEFT")
        row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
        return row

    def message(level, text):
        background, foreground = MESSAGE_COLOURS.get(level, MESSAGE_COLOURS["info"])
        frame = Table([[Paragraph(_markup(text), style("msg", 8.6, color=colors.HexColor(foreground)))]],
                      colWidths=[width], hAlign="LEFT")
        frame.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(background)),
                                   ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
        return frame

    def health_card(card, card_width):
        status = colors.HexColor(STATUS_COLOURS.get(card["status_key"], STATUS_COLOURS["good"]))
        inner = card_width - 16
        top = Table([[Paragraph(_markup(f"**{card['label']}**"), body),
                      Paragraph(_markup(f"**{card['status']}**"), style("chipc", 8.6, color=status, alignment=2))]],
                    colWidths=[inner * 0.65, inner * 0.35])
        top.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        spark = Drawing(inner, 30)
        hours, values = np.asarray(card["hours"], float), np.asarray(card["values"], float)
        keep = np.isfinite(values)
        hours, values = hours[keep], values[keep]
        if len(hours) >= 2:
            xs = (hours - hours.min()) / ((hours.max() - hours.min()) or 1) * (inner - 6) + 3
            span = values.max() - values.min()
            ys = (np.full_like(values, 15.0) if span == 0 else 3 + (values - values.min()) / span * 24)
            points = [v for pair in zip(xs, ys) for v in pair]
            fill = colors.Color(status.red, status.green, status.blue, alpha=0.14)
            spark.add(Polygon([xs[0], 0, *points, xs[-1], 0], fillColor=fill, strokeColor=None))
            spark.add(PolyLine(points, strokeColor=status, strokeWidth=1.4))
            spark.add(Circle(xs[-1], ys[-1], 2.2, fillColor=status, strokeColor=None))
        bar = Drawing(inner, 6)
        bar.add(Rect(0, 0, inner, 5, rx=2.5, ry=2.5, fillColor=colors.HexColor("#E6EAF1"), strokeColor=None))
        bar.add(Rect(0, 0, max(inner * card["health"] / 100, 0.1), 5, rx=2.5, ry=2.5, fillColor=status,
                     strokeColor=None))
        foot = Table([[Paragraph(f"{card['health']:.0f}% health", small),
                       Paragraph(_markup(f"Limit: {card['limit']}"), style("lim", 7.4, color=muted, alignment=2))]],
                     colWidths=[inner * 0.4, inner * 0.6])
        foot.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        content = [top, Spacer(1, 3),
                   Paragraph(f"{escape(card['value'])} <font size=8 color='#5F6B7A'>{escape(card['unit'])}</font>",
                             style("cv", 15, leading=18)),
                   Paragraph(_markup(card["change"]), small), Spacer(1, 3), spark, Spacer(1, 3), bar, Spacer(1, 3),
                   foot]
        frame = Table([[content]], colWidths=[card_width])
        frame.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, line), ("LINEABOVE", (0, 0), (-1, 0), 3, status),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                                   ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        return frame

    def chart_image(chart, target_width):
        pixels = 560 if target_width < width * 0.75 else 1100
        png = chart_png(chart, pixels)
        w, h = ImageReader(BytesIO(png)).getSize()
        return Image(BytesIO(png), width=target_width, height=target_width * h / w)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [Paragraph("Monitor and Predict", title), Spacer(1, 4),
             Paragraph(_markup(f"CapLife AI · {view.source} · model: {view.model} · generated {stamp}"), caption), Spacer(1, 4)]

    # Fleet overview
    story += [Paragraph("Fleet overview", h2), metric_row([(n, v, None) for n, v in view.fleet_metrics]),
              Spacer(1, 8)]
    fleet = view.fleet
    columns = list(fleet.columns)
    rows = [[Paragraph(_markup(c), head_cell) for c in columns]]
    for record in fleet.itertuples(index=False):
        rows.append([Paragraph(_markup(_fleet_cell(c, v)), cell) for c, v in zip(columns, record)])
    weights = [1.5 if "range" in c or "indicator" in c else 1.0 for c in columns]
    table = Table(rows, colWidths=[width * w / sum(weights) for w in weights], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F2F6")),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(table)
    if view.fleet_note:
        story += [Spacer(1, 4), para(view.fleet_note, caption)]

    # Maintenance planning, as on the page
    if view.plan is not None and view.plan_horizon is not None:
        story += [Paragraph(f"Maintenance planning: the next {view.plan_horizon:,.0f} h", h2)]
        if view.plan.empty:
            story.append(para(f"No capacitor is likely to reach end of life within the next "
                              f"{view.plan_horizon:,.0f} h: every 80% range starts later.", body))
        else:
            plan = view.plan.rename(columns={"Chance within plan (%)": f"Chance within {view.plan_horizon:,.0f} h (%)"})
            columns = list(plan.columns)
            rows = [[Paragraph(_markup(c), head_cell) for c in columns]]
            for record in plan.itertuples(index=False):
                rows.append([Paragraph(_markup(_fleet_cell(c, v)), cell) for c, v in zip(columns, record)])
            grid = Table(rows, colWidths=[width / len(columns)] * len(columns), repeatRows=1, hAlign="LEFT")
            grid.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F2F6")),
                                      ("LINEBELOW", (0, 0), (-1, -1), 0.4, line),
                                      ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [Spacer(1, 4), grid, Spacer(1, 3),
                      para("Counted from each capacitor's latest reading. Could fail from: the early end of the 80% "
                           "range; most likely: the predicted remaining life.", caption)]

    # Capacitor detail: inputs, metrics and health, as on the page
    detail = [Paragraph("Capacitor detail", h2), para(f"Capacitor: **{view.device}**", body), Spacer(1, 4)]
    if view.parameter_note:
        detail += [para(view.parameter_note, caption), Spacer(1, 4)]
    detail.append(input_grid(f"End-of-life limits for {view.device}",
                             [(name, value if measured else f"{value} (not measured)", None)
                              for name, value, measured in view.limits]))
    story += [KeepTogether(detail), Spacer(1, 8)]
    conditions = [(name, value, f"Recorded: {recorded}" if recorded is not None else None)
                  for name, value, recorded in view.conditions]
    block = input_grid(f"Operating conditions for {view.device}", conditions)
    chips = badge_row(view.badges)
    story += [KeepTogether([block] + ([Spacer(1, 4), chips] if chips else [])), Spacer(1, 8)]
    for level, text in view.messages:
        story += [message(level, text), Spacer(1, 6)]
    if view.metrics:
        story += [KeepTogether([metric_row(view.metrics)] +
                               ([Spacer(1, 3), para(view.metrics_note, caption)] if view.metrics_note else [])),
                  Spacer(1, 8)]
    if view.health_cards:
        gap = 8
        card_width = (width - 2 * gap) / 3
        cards = [health_card(card, card_width) for card in view.health_cards]
        grid_rows = [cards[i:i + 3] + [""] * (3 - len(cards[i:i + 3])) for i in range(0, len(cards), 3)]
        grid = Table(grid_rows, colWidths=[card_width + gap] * 2 + [card_width])
        grid.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), gap),
                                  ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(grid)

    # The page's charts, section by section
    for section in view.sections:
        heading = [Paragraph(_markup(section.title), h3)] + [para(note, caption) for note in section.notes]
        if not section.charts:
            story.append(KeepTogether(heading))
            continue
        if section.columns == 1:
            images = [chart_image(chart, width) for _, chart in section.charts]
            story.append(KeepTogether(heading + [Spacer(1, 4), *images]))
            continue
        gap = 14
        column = (width - gap) / 2
        pairs = [section.charts[i:i + 2] for i in range(0, len(section.charts), 2)]
        for index, pair in enumerate(pairs):
            cells = [[para(text, caption), Spacer(1, 2), chart_image(chart, column)] for text, chart in pair]
            cells += [""] * (2 - len(cells))
            row = Table([cells], colWidths=[column + gap, column])
            row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
            story.append(KeepTogether((heading + [Spacer(1, 4)] if index == 0 else []) + [row]))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("DejaVu", 6.8)
        canvas.setFillColor(muted)
        canvas.drawString(margin, 14, view.disclaimer)
        canvas.drawRightString(page_w - margin, 14, f"{view.device} · page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
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
    hours = book.add_format({"num_format": "#,##0"})  # shown as whole hours; the cell keeps the full value

    overview = book.add_worksheet("Overview")
    overview.set_column("A:A", 30)
    overview.set_column("B:B", 60)
    overview.merge_range("A1:B1", "CapLife AI · Monitor and Predict", title)
    overview.set_row(0, 28)
    view = report.view
    info = [("Data", report.source), ("Model", view.model if view else "Built-in model"),
            ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            ("Capacitors", len(report.fleet)), ("Selected capacitor", report.device), *_detail(report)]
    for row, (key, value) in enumerate(info, start=2):
        overview.write(row, 0, key, label)
        overview.write(row, 1, value)
    overview.merge_range(len(info) + 3, 0, len(info) + 5, 1, view.disclaimer if view else DISCLAIMER, note)

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
                    tab.write_number(row, index, float(value), hours if str(frame.columns[index]).endswith("(h)")
                                     else None)
                elif isinstance(value, (bool, np.bool_)):
                    tab.write_boolean(row, index, bool(value))
                else:
                    tab.write(row, index, value)

    sheet("Fleet", report.fleet)
    if view is not None and view.plan is not None and view.plan_horizon is not None:
        sheet(f"Plan, next {view.plan_horizon:,.0f} h".replace(",", ""), view.plan)
    detail = report.readings.merge(report.assessment, on="Hours", how="left")
    sheet(f"{report.device} history", detail)
    if report.forecast is not None and len(report.forecast):
        sheet(f"{report.device} forecast", report.forecast.rename(columns={"signal": "Parameter"}))
    limits = pd.DataFrame([{"Capacitor": device, **{LIMIT_NAMES[k]: _shown_limit(k, v) for k, v in values.items()
                                                    if k in LIMIT_NAMES}}
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


__all__ = ["ChartSection", "MonitorReport", "PageView", "chart_png", "pdf_bytes", "xlsx_bytes"]
