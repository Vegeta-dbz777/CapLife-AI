"""Downloadable reports for Train The AI: a PDF and an Excel workbook.

The PDF is the page as the user sees it, in the page's order: the data and the split, the data used,
the test on capacitors the model never saw (the model table and the page's own test chart), the
verdict, and the remaining-life forecasts with every forecast capacitor's remaining-life chart. The
Excel workbook has every table in full. Charts are the page's Altair charts, exported as on Monitor
and Predict's report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd

from caplife.monitor_report import BADGE_COLOURS, MESSAGE_COLOURS, _markup, _register_fonts, chart_png

NOTE = ("Forecasts describe the uploaded capacitors and endpoint only; accuracy under other operating conditions, "
        "product series or endpoints is untested. A research prototype, not a maintenance or safety tool.")
TAG_COLOURS = {**BADGE_COLOURS, "blue": ("#E7F0FB", "#1F5FAD")}


@dataclass
class TrainView:
    """What Train The AI shows after training, in the page's order and wording."""
    source: str
    data_metrics: list[tuple[str, str]]
    split: str
    tags: list[tuple[str, str]]                     # colour, text
    data_note: str
    excluded: dict
    model_table: pd.DataFrame
    test_legend: str
    test_chart: object
    test_table: pd.DataFrame
    passed: bool
    verdict: str
    reasons: list[str]
    checks: str
    warnings: list[str]
    forecast_message: str | None
    forecasts: pd.DataFrame | None                  # as the page's table shows them
    rul_charts: list[tuple[str, object]]            # (summary, chart) for every forecast capacitor
    withheld: dict
    withheld_note: str
    tested: pd.DataFrame | None = None              # what the user's data covered
    tested_note: str = ""
    tables: dict = field(default_factory=dict)      # every table in full, for the Excel workbook


def _cell(column: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if "inside 80% range" in column:
            return f"{float(value):.1%}"
        return f"{float(value):,.0f}"
    return str(value)


def pdf_bytes(view: TrainView) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    _register_fonts()
    buffer = BytesIO()
    page_w, page_h = landscape(A4)
    margin = 30
    doc = SimpleDocTemplate(buffer, pagesize=(page_w, page_h), leftMargin=margin, rightMargin=margin, topMargin=28,
                            bottomMargin=28, title="CapLife AI · Train The AI", author="CapLife AI")
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
    cell = style("cell", 7.2, leading=9)
    head_cell = style("head", 7.2, leading=9, color=muted)
    metric_label = style("mlabel", 8, color=muted)
    metric_value = style("mvalue", 17, leading=21)
    para = lambda text, st=body: Paragraph(_markup(text), st)

    def metric_row(items):
        gap = 8
        column = (width - gap * (len(items) - 1)) / len(items)
        row = Table([[[Paragraph(_markup(n), metric_label), Paragraph(_markup(v), metric_value)] for n, v in items]],
                    colWidths=[column] * len(items), hAlign="LEFT")
        row.setStyle(TableStyle([("BOX", (i, 0), (i, 0), 0.8, line) for i in range(len(items))]
                                + [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                   ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        return row

    def tags(items):
        cells, widths = [], []
        for colour, text in items:
            background, foreground = TAG_COLOURS.get(colour, TAG_COLOURS["gray"])
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

    def grid(frame: pd.DataFrame, shares: list[float] | None = None):
        columns = list(frame.columns)
        rows = [[Paragraph(_markup(c), head_cell) for c in columns]]
        for record in frame.itertuples(index=False):
            rows.append([Paragraph(_markup(_cell(c, v)), cell) for c, v in zip(columns, record)])
        shares_ = shares or [1 / len(columns)] * len(columns)
        table = Table(rows, colWidths=[width * share for share in shares_], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F2F6")),
                                   ("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return table

    def chart_image(chart):
        png = chart_png(chart, 1100)
        w, h = ImageReader(BytesIO(png)).getSize()
        return Image(BytesIO(png), width=width, height=width * h / w)

    def reasons(items: dict) -> list:
        return [para(f"• **{device}**: {reason}", caption) for device, reason in items.items()]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [Paragraph("Train The AI", title), Spacer(1, 4),
             para(f"CapLife AI · {view.source} · generated {stamp}", caption), Spacer(1, 8),
             metric_row(view.data_metrics), Spacer(1, 6), para(view.split, caption)]

    story += [Paragraph("1 · Data used", h2), tags(view.tags), Spacer(1, 4), para(view.data_note, caption)]
    if view.excluded:
        story += [Spacer(1, 4), para(f"**{len(view.excluded)} capacitor(s) with outcomes could not be used**",
                                     caption), *reasons(view.excluded)]
    if view.tested is not None and len(view.tested):
        story += [Spacer(1, 6), para("**What your data tested**", caption), Spacer(1, 3), grid(view.tested, [0.25, 0.75]),
                  Spacer(1, 2), para(view.tested_note, caption)]

    story += [Paragraph("2 · Test on capacitors the model never saw", h2), grid(view.model_table), Spacer(1, 8),
              KeepTogether([para(view.test_legend, caption), Spacer(1, 3), chart_image(view.test_chart)]),
              Spacer(1, 6), para("**Each test capacitor**", caption), Spacer(1, 3), grid(view.test_table)]

    story += [Paragraph("3 · Verdict", h2),
              message("info" if view.passed else "warning",
                      view.verdict + "".join(f"\n• {reason}" for reason in view.reasons)),
              Spacer(1, 4), para(view.checks, caption)]
    for warning in view.warnings:
        story += [Spacer(1, 4), message("warning", warning)]

    story += [Paragraph("4 · Remaining-life forecasts", h2)]
    if view.forecast_message:
        story.append(para(view.forecast_message, body))
    if view.forecasts is not None and len(view.forecasts):
        story += [grid(view.forecasts), Spacer(1, 6)]
    if view.rul_charts:
        story.append(Paragraph("Remaining-life prediction over time", h3))
        for summary, chart in view.rul_charts:
            story.append(KeepTogether([para(summary, caption), Spacer(1, 3), chart_image(chart), Spacer(1, 8)]))
    if view.withheld:
        story += [para(f"**Forecast withheld for {len(view.withheld)} capacitor(s)**", caption),
                  *reasons(view.withheld), para(view.withheld_note, caption)]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("DejaVu", 6.8)
        canvas.setFillColor(muted)
        canvas.drawString(margin, 14, NOTE)
        canvas.drawRightString(page_w - margin, 14, f"Train The AI · page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def xlsx_bytes(view: TrainView) -> bytes:
    """Every table in full: an overview, then one sheet per table."""
    import xlsxwriter

    memory = BytesIO()
    book = xlsxwriter.Workbook(memory, {"in_memory": True, "strings_to_formulas": False, "strings_to_urls": False,
                                        "nan_inf_to_errors": True})
    navy, blue = "#15243D", "#226D93"
    title = book.add_format({"bold": True, "font_size": 18, "font_color": "white", "bg_color": navy})
    header = book.add_format({"bold": True, "font_color": "white", "bg_color": blue, "text_wrap": True,
                              "valign": "vcenter"})
    label = book.add_format({"bold": True, "font_color": navy})
    wrap = book.add_format({"text_wrap": True, "valign": "top"})
    hours = book.add_format({"num_format": "#,##0"})  # shown as whole hours; the cell keeps the full value
    percent = book.add_format({"num_format": "0.0%"})

    overview = book.add_worksheet("Overview")
    overview.set_column("A:A", 30)
    overview.set_column("B:B", 90)
    overview.merge_range("A1:B1", "CapLife AI · Train The AI", title)
    overview.set_row(0, 28)
    info = [("Data", view.source), ("Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
            *view.data_metrics, ("Split", view.split), ("Data used", " · ".join(text for _, text in view.tags)),
            ("Details", view.data_note), ("Verdict", view.verdict.replace("**", "")),
            *(("Reason", r) for r in view.reasons), ("Checks", view.checks.removeprefix("Checks: ")), *(("Note", w) for w in view.warnings), ("About the forecasts", NOTE)]
    for row, (key, value) in enumerate(info, start=2):
        overview.write(row, 0, key, label)
        overview.write(row, 1, value, wrap)

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
                elif isinstance(value, (np.integer, np.floating, int, float)) and not isinstance(value, bool):
                    name = str(frame.columns[index])
                    tab.write_number(row, index, float(value), hours if name.endswith("(h)") else
                                     percent if "inside 80% range" in name else None)
                else:
                    tab.write(row, index, "" if value is None else str(value))

    for name, frame in view.tables.items():
        if frame is not None and len(frame):
            sheet(name, frame)
    book.close()
    return memory.getvalue()


__all__ = ["TrainView", "pdf_bytes", "xlsx_bytes"]
