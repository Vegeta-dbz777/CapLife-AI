"""CapLife AI research-only downloads: CSV, XLSX, and PDF.

All three outputs are generated in memory from the same already-validated v0.8
experimental report. Never infer EOL, RUL, health, uncertainty, or missing data.
"""
from __future__ import annotations

from io import BytesIO
import math
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

INDICATORS = (
    ("Capacitance", "Capacitance_change_percent", "Observed_Change_percent", "%"),
    ("ESR", "ESR_change_percent", "Observed_Change_percent", "%"),
    ("Leakage current", "Leakage_change_percent", "Observed_Change_percent", "%"),
    ("Case expansion", "Case_expansion_change_um", "Observed_Change_um", "µm"),
    ("H₂ generated (measured)", "H2_measured_change_mL_STP", "Observed_Change_mL_STP", "mL at STP"),
    ("Internal pressure (measured)", "Pressure_measured_change_kPa", "Observed_Change_kPa", "kPa abs change"),
)
GRAPH_CHANNELS = (
    ("C_ratio", "Capacitance / C₀", "ratio"),
    ("ESR_ratio", "ESR / ESR₀", "ratio"),
    ("Leakage_ratio", "Leakage / initial leakage", "ratio"),
    ("Case_Expansion_um", "Case expansion", "µm"),
)
EXTRA_MEASURED_GRAPHS = (
    ("H2_Generated_mL_STP", "Measured H₂ generated (STP)", "mL at STP"),
    ("Pressure_Abs_kPa", "Measured internal pressure", "kPa absolute"),
)

def _graph_channels_for(history: pd.DataFrame):
    # Old four-indicator uploads still produce exactly four measurement charts.
    additional = tuple(item for item in EXTRA_MEASURED_GRAPHS
                       if item[0] in history and history[item[0]].notna().any())
    return GRAPH_CHANNELS + additional

DISCLAIMER = (
    "RESEARCH PROTOTYPE — DESCRIPTIVE MEASUREMENTS ONLY. "
    "No validated capacitor health, end-of-life, remaining-life, confidence interval, "
    "or maintenance recommendation is provided. Synthetic test/example data are not "
    "real experimental results. Missing readings are not imputed."
)


def summary_table(report: dict, audit: dict, selected_cap=None, gas_scenario=None, assumptions=None) -> pd.DataFrame:
    """One row per capacitor. Flattened summary suitable for a single CSV file."""
    summaries = audit["summary"].copy()
    changes = report["changes"]
    for label, destination, origin, _ in INDICATORS:
        per_cap = changes.loc[changes["Indicator"] == label, ["Cap_ID", origin, "Status"]].copy()
        per_cap.rename(columns={origin: destination, "Status": destination + "_status"}, inplace=True)
        summaries = summaries.merge(per_cap, on="Cap_ID", how="left", validate="one_to_one")
    if gas_scenario is not None and len(gas_scenario):
        # Calculated scenario is explicitly tied to the selected capacitor; NEVER
        # silently propagate its estimated values to other uploaded capacitors.
        cap = str(selected_cap)
        last = gas_scenario.iloc[-1]
        summaries["Scenario_H2_generated_mL_STP_est"] = np.where(
            summaries["Cap_ID"] == cap, float(last["H2_Generated_mL_STP_est"]), np.nan)
        summaries["Scenario_Pressure_abs_kPa_est"] = np.where(
            summaries["Cap_ID"] == cap, float(last["Pressure_Abs_kPa_est"]), np.nan)
        summaries["Scenario_Pressure_change_kPa_est"] = np.where(
            summaries["Cap_ID"] == cap, float(last["Pressure_Change_From_Initial_kPa_est"]), np.nan)
        summaries["Scenario_Provenance"] = np.where(
            summaries["Cap_ID"] == cap, "Assumption-based physics estimate; NOT measured", "")
    return summaries


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent user-provided IDs from becoming formulas on opening CSV in Excel."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        def safe(value):
            if not isinstance(value, str):
                return value
            # A leading whitespace/control character can conceal spreadsheet formulas.
            if value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value
        result[column] = result[column].map(safe)
    return result


def csv_report_bytes(report: dict, audit: dict, selected_cap=None, gas_scenario=None, assumptions=None) -> bytes:
    """A plain CSV summary with one row per capacitor; legacy behavior preserved."""
    table = _csv_safe(summary_table(report, audit, selected_cap, gas_scenario, assumptions))
    return table.to_csv(index=False, na_rep="").encode("utf-8-sig")


def csv_measurement_history_bytes(report: dict) -> bytes:
    """A complete CSV of all observed rows for every capacitor.

    Raw measured values and normalized ratios are retained as supplied/derived
    by the validated measurement pipeline. Hypothetical physics estimates are
    never appended to this measured-history file.
    """
    measured = report["measurements"].sort_values(["Cap_ID", "Hours"], kind="stable")
    return _csv_safe(measured).to_csv(index=False, na_rep="").encode("utf-8-sig")


def _xlsx_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def xlsx_report_bytes(report: dict, audit: dict, selected_cap: str, gas_scenario=None, assumptions=None) -> bytes:
    """Detailed multi-tab workbook with 4–6 recorded measurement plots."""
    import xlsxwriter

    memory = BytesIO()
    wb = xlsxwriter.Workbook(memory, {"in_memory": True, "strings_to_formulas": False, "strings_to_urls": False, "nan_inf_to_errors": False})
    navy, blue, pale = "#15243D", "#226D93", "#EDF4F8"
    title = wb.add_format({"bold": True, "font_size": 20, "font_color": "white", "bg_color": navy, "valign": "vcenter"})
    banner = wb.add_format({"font_color": "white", "bg_color": navy, "font_size": 10})
    header = wb.add_format({"bold": True, "font_color": "white", "bg_color": blue, "text_wrap": True, "valign": "vcenter", "bottom": 1, "bottom_color": "#D9E7ED"})
    body = wb.add_format({"valign": "top", "font_size": 10, "bottom": 1, "bottom_color": "#EDF0F4"})
    muted = wb.add_format({"font_color": "#53657C", "text_wrap": True, "valign": "top", "font_size": 10})
    note_box = wb.add_format({"font_color": "#8B3E06", "bg_color": "#FFF4DD", "text_wrap": True, "valign": "vcenter"})
    big_value = wb.add_format({"bold": True, "font_size": 18, "font_color": blue})

    overview = wb.add_worksheet("Overview")
    overview.hide_gridlines(2)
    overview.set_tab_color(blue)
    overview.set_column("A:A", 25)
    overview.set_column("B:M", 15)
    overview.merge_range("A1:M2", "CapLife AI  /  Experimental measurement report", title)
    overview.merge_range("A3:M3", "Version 0.8 • six-indicator research-only descriptive assessment", banner)
    overview.set_row(0, 22)
    overview.set_row(1, 26)
    overview.write("A5", "CAPACITORS", header)
    overview.write("D5", "MEASUREMENT ROWS", header)
    overview.write("G5", "SELECTED CAPACITOR", header)
    overview.write("K5", "RECORDED CHANNELS", header)
    overview.write("A6", int(report["measurements"]["Cap_ID"].nunique()), big_value)
    overview.write("D6", len(report["measurements"]), big_value)
    overview.write("G6", selected_cap, big_value)
    recorded = sum(report["measurements"][c].notna().any() for c in ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um", "H2_Generated_mL_STP", "Pressure_Abs_kPa") if c in report["measurements"])
    overview.write("K6", int(recorded), big_value)
    overview.merge_range("A8:M10", DISCLAIMER, note_box)
    overview.set_row(7, 28)
    overview.set_row(8, 26)
    overview.write("A12", "All-capacitor summary", header)
    overview.merge_range("A13:M14", "The full summary is on the Summary sheet; the Charts sheet shows recorded points for the selected capacitor. None of these outputs is a prognostic result.", muted)
    overview.write("A17", "Limitations", header)
    overview.merge_range("A18:M21", "Percentage changes require a measured hour-zero baseline. ESR comparisons depend on consistent documented test conditions. Test and operating conditions, sensor calibration, and provenance require independent review. The CSV download offers either the complete measured history or a per-capacitor summary; the XLSX file contains the detailed tables.", muted)
    overview.freeze_panes(3, 0)

    sheets = [
        ("Summary", summary_table(report, audit, selected_cap, gas_scenario, assumptions)),
        ("Measurements", report["measurements"]),
        ("Quality", report["quality"]),
        ("Observed Changes", report["changes"]),
        ("Readiness", audit["summary"]),
        ("Review Actions", audit["actions"]),
        ("Operating Conditions", audit["conditions"]),
    ]
    for tab, df in sheets:
        sheet = wb.add_worksheet(tab)
        sheet.hide_gridlines(2)
        sheet.freeze_panes(1, 0)
        sheet.set_tab_color("#23789C" if tab == "Summary" else "#90AFC1")
        sheet.set_row(0, 34)
        for ci, col in enumerate(df.columns):
            sheet.write(0, ci, str(col), header)
            if col in {"Recommendation", "Status", "Data_readiness"}:
                sheet.set_column(ci, ci, 48)
            elif col == "Cap_ID":
                sheet.set_column(ci, ci, 25)
            else:
                sheet.set_column(ci, ci, min(30, max(16, len(str(col)) + 2)))
        for ri, record in enumerate(df.itertuples(index=False, name=None), start=1):
            for ci, value in enumerate(record):
                value = _xlsx_value(value)
                if value is None:
                    sheet.write_blank(ri, ci, None, body)
                else:
                    sheet.write(ri, ci, value, body)
            if ri % 2 == 0:
                sheet.set_row(ri, None, None, {"level": 0})
        if len(df) and len(df.columns):
            sheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    history = report["measurements"].loc[report["measurements"]["Cap_ID"] == selected_cap].sort_values("Hours")
    if history.empty:
        wb.close()
        raise ValueError("Selected capacitor is absent from the validated report.")
    chart_data = wb.add_worksheet("Chart Data")
    chart_data.hide_gridlines(2)
    channels = _graph_channels_for(history)
    cols = ["Hours", *[x[0] for x in channels]]
    chart_data.set_column(0, len(cols) - 1, 25)
    for j, col in enumerate(cols):
        chart_data.write(0, j, col, header)
        if col not in history.columns:
            continue
        for i, val in enumerate(history[col], start=1):
            converted = _xlsx_value(val)
            if converted is not None:
                chart_data.write(i, j, converted)
    chart_sheet = wb.add_worksheet("Charts")
    chart_sheet.hide_gridlines(2)
    chart_sheet.set_tab_color("#0A8E87")
    chart_sheet.set_column("A:R", 11)
    chart_sheet.merge_range("A1:R2", f"Recorded degradation trends  /  {selected_cap}", title)
    chart_sheet.merge_range("A3:R3", "Actual measurement points only • blanks are not interpolated • no EOL or RUL estimate", muted)
    for channel_idx, (col, label, unit) in enumerate(channels, start=1):
        plot = wb.add_chart({"type": "scatter", "subtype": "marker_only"})
        plot.add_series({
            "name": label,
            "categories": ["Chart Data", 1, 0, len(history), 0],
            "values": ["Chart Data", 1, channel_idx, len(history), channel_idx],
            "marker": {"type": "circle", "size": 5, "border": {"color": blue}, "fill": {"color": blue}},
        })
        plot.set_title({"name": label})
        plot.set_x_axis({"name": "Operating hours", "min": 0})
        plot.set_y_axis({"name": unit})
        plot.set_legend({"none": True})
        plot.set_chartarea({"border": {"none": True}})
        plot.set_plotarea({"border": {"color": "#CDD8E0"}})
        plot.set_size({"width": 530, "height": 286})
        chart_positions = ("A5", "J5", "A22", "J22", "A39", "J39")
        chart_sheet.insert_chart(chart_positions[channel_idx - 1], plot)
    note_start = 57 if len(channels) > 4 else 40
    chart_sheet.merge_range(f"A{note_start}:R{note_start+2}", DISCLAIMER, note_box)

    if gas_scenario is not None and len(gas_scenario):
        # The calculated scenario has its own worksheets and line charts;
        # these cannot be mistaken for direct sensor observations.
        scenario = gas_scenario.copy()
        gas_sheet = wb.add_worksheet("Physics Scenario")
        gas_sheet.hide_gridlines(2)
        gas_sheet.freeze_panes(1, 0)
        gas_sheet.set_tab_color("#9D6B20")
        for colidx, col in enumerate(scenario.columns):
            gas_sheet.write(0, colidx, col, header)
            gas_sheet.set_column(colidx, colidx, 26 if colidx else 18)
        for rowidx, record in enumerate(scenario.itertuples(index=False, name=None), start=1):
            for colidx, value in enumerate(record):
                converted = _xlsx_value(value)
                if converted is not None:
                    gas_sheet.write(rowidx, colidx, converted, body)
        if len(scenario):
            gas_sheet.autofilter(0, 0, len(scenario), len(scenario.columns)-1)
        assumption_sheet = wb.add_worksheet("Scenario Assumptions")
        assumption_sheet.hide_gridlines(2)
        assumption_sheet.set_column("A:A", 43)
        assumption_sheet.set_column("B:B", 24)
        assumption_sheet.merge_range("A1:D2", "Assumption-based physics preview — NOT sensor measurements", title)
        assumption_sheet.write("A4", "Selected capacitor", header)
        assumption_sheet.write("B4", selected_cap, body)
        for idx, (name, value) in enumerate(sorted((assumptions or {}).items()), start=5):
            assumption_sheet.write(idx-1, 0, str(name), body)
            assumption_sheet.write(idx-1, 1, _xlsx_value(value), body)
        assumption_sheet.merge_range("A13:D17", "Faraday-law conversion assumes an unverified fraction of leakage current generates H₂. A fixed fraction is retained in a rigid headspace. The result is hypothetical and has NOT been calibrated or validated against physical gas/pressure measurements. No EOL or RUL prediction.", note_box)
        physics_chart_sheet = wb.add_worksheet("Physics Charts")
        physics_chart_sheet.hide_gridlines(2)
        physics_chart_sheet.set_column("A:R", 11)
        physics_chart_sheet.merge_range("A1:R2", f"Assumption-based gas / pressure scenario  /  {selected_cap}", title)
        physics_chart_sheet.merge_range("A3:R3", "Calculated values ONLY — not measured, not validated, no vent limit or RUL", muted)
        time_col = scenario.columns.get_loc("Hours")
        for xpos, column, label in (("A5", "H2_Generated_mL_STP_est", "Calculated H₂ generated (mL at STP)"), ("J5", "Pressure_Abs_kPa_est", "Calculated absolute pressure (kPa)")):
            if column not in scenario:
                continue
            val_col = scenario.columns.get_loc(column)
            line = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
            line.add_series({"name": label, "categories": ["Physics Scenario", 1, time_col, len(scenario), time_col], "values": ["Physics Scenario", 1, val_col, len(scenario), val_col], "line": {"color": "#C98222", "width": 2}, "marker": {"type": "circle", "size": 5}})
            line.set_title({"name": label})
            line.set_x_axis({"name": "Operating hours"})
            line.set_legend({"none": True})
            line.set_size({"width": 530, "height": 290})
            physics_chart_sheet.insert_chart(xpos, line)
        physics_chart_sheet.merge_range("A24:R28", "This is a deterministic assumption-based estimate, not a measured 5th/6th channel or a real-world RUL prediction. The gas headspace, current fraction, retention fraction and initial pressure are user assumptions.", note_box)
    wb.close()
    return memory.getvalue()


def _pdf_txt(value) -> str:
    """Plain text safe to include in ReportLab paragraphs."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    return escape(str(value)[:140].replace("₂", "2").replace("₀", "0"))


def _fmt(value, suffix="") -> str:
    if value is None or pd.isna(value):
        return "Not available"
    try:
        return f"{float(value):+,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _trend_png(history: pd.DataFrame) -> bytes:
    from matplotlib.figure import Figure

    channels = _graph_channels_for(history)
    extra = len(channels) > 4
    fig = Figure(figsize=(11.7, 7.4 if extra else 5.5), dpi=140, facecolor="white")
    axes = fig.subplots(3 if extra else 2, 2)
    for ax, (col, label, unit) in zip(axes.flat, channels):
        valid = history[["Hours", col]].dropna() if col in history.columns else history.iloc[0:0][["Hours"]]
        if not valid.empty:
            ax.scatter(valid["Hours"], valid[col], s=28, color="#24759A", zorder=3)
            ax.set_xlim(left=0)
        else:
            ax.text(.5, .5, "No comparable observations", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#62788C")
        ax.set_title(label, fontsize=11, weight="bold", color="#172B45")
        ax.set_xlabel("Operating hours", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(alpha=.3)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(hspace=.57 if extra else .48, wspace=.29, left=.08, right=.98, bottom=.10, top=.94)
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=130)
    return stream.getvalue()


def pdf_report_bytes(report: dict, audit: dict, selected_cap: str, gas_scenario=None, assumptions=None) -> bytes:
    """Print-ready A4 landscape report. Summary covers all caps, plots selected cap."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    )

    buf = BytesIO()
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(buf, pagesize=(page_w, page_h), leftMargin=34, rightMargin=34, topMargin=34, bottomMargin=34, title="CapLife AI Research Measurement Assessment", author="CapLife AI")
    usable_width = page_w - 68
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CapTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=colors.HexColor("#15243D"), spaceAfter=9))
    styles.add(ParagraphStyle(name="CapHead", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=colors.HexColor("#15243D"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="CapBody", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#334B62")))
    styles.add(ParagraphStyle(name="CapCell", parent=styles["Normal"], fontSize=7, leading=10, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CapBold", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#15243D")))
    styles.add(ParagraphStyle(name="CapSub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#46728A"), spaceAfter=12))
    story = []
    story.append(Paragraph("CapLife AI / Measurement assessment", styles["CapTitle"]))
    story.append(Paragraph("Experimental measurements (v0.8)  |  Descriptive research report  |  No real-world lifetime prediction", styles["CapSub"]))
    n_caps = int(report["measurements"]["Cap_ID"].nunique())
    n_rows = len(report["measurements"])
    n_rec = int(sum(report["measurements"][c].notna().any() for c in ("Capacitance_uF", "ESR_ohm", "Leakage_uA", "Case_Expansion_um", "H2_Generated_mL_STP", "Pressure_Abs_kPa") if c in report["measurements"]))
    cards = Table([[f"CAPACITORS  {n_caps}", f"MEASUREMENT ROWS  {n_rows:,}", f"RECORDED CHANNELS  {n_rec}", f"PLOT DETAIL  {_pdf_txt(selected_cap)}"]], colWidths=[usable_width/4]*4, rowHeights=[35])
    cards.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#E8F2F7")),("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#173C59")),("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),10)]))
    story.extend([cards, Spacer(1, 8), Paragraph(escape(DISCLAIMER), styles["CapBody"])])
    story.append(Paragraph("All-capacitor observed changes", styles["CapHead"]))
    summary = summary_table(report, audit, selected_cap, gas_scenario, assumptions)
    headings = ["Capacitor", "Points", "Coverage", "C change", "ESR change", "Leakage", "Expansion", "H2 measured", "Pressure measured", "Readiness"]
    # Allow common descriptive IDs to fit on one line while retaining readiness space.
    fixed_widths = [124, 39, 50, 62, 62, 62, 65, 69, 74]
    widths = fixed_widths + [usable_width - sum(fixed_widths)]
    cells = [[Paragraph(_pdf_txt(x), styles["CapBold"]) for x in headings]]
    for _, r in summary.iterrows():
        cells.append([
            Paragraph(_pdf_txt(r["Cap_ID"]), styles["CapCell"]),
            str(int(r["Measurements"])),
            f"{r['Observed_coverage_percent']:.0f}%",
            _fmt(r["Capacitance_change_percent"], "%"),
            _fmt(r["ESR_change_percent"], "%"),
            _fmt(r["Leakage_change_percent"], "%"),
            _fmt(r["Case_expansion_change_um"], " µm"),
            _fmt(r["H2_measured_change_mL_STP"], " mL"),
            _fmt(r["Pressure_measured_change_kPa"], " kPa"),
            Paragraph(_pdf_txt(r["Data_readiness"]), styles["CapCell"]),
        ])
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#D9EAF3")),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, colors.HexColor("#F5F8FB")]),
        ("TEXTCOLOR",(0,0),(-1,-1),colors.HexColor("#17324A")),
        ("FONTSIZE",(0,1),(-1,-1),7.7),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("LINEBELOW",(0,0),(-1,0),0.6,colors.HexColor("#9DB9CA")),
    ]))
    story.append(t)
    story.append(Paragraph("Measurement coverage / selected capacitor", styles["CapHead"]))
    cap_quality = report["quality"].loc[report["quality"]["Cap_ID"] == selected_cap]
    quality_cells = [[Paragraph(x, styles["CapBold"]) for x in ("Indicator", "Available", "Missing", "Completeness", "Status")]]
    for _, q in cap_quality.iterrows():
        quality_cells.append([
            Paragraph(_pdf_txt(q["Indicator"]), styles["CapCell"]),
            str(int(q["Available"])), str(int(q["Missing"])),
            f"{q['Completeness_percent']:.0f}%", Paragraph(_pdf_txt(q["Status"]), styles["CapCell"]),
        ])
    qt = Table(quality_cells, colWidths=[170, 85, 85, 100, usable_width-440], repeatRows=1)
    qt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,0), colors.HexColor("#D9EAF3")),
        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, colors.HexColor("#F5F8FB")]),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("FONTSIZE", (0,1),(-1,-1), 8),
    ]))
    story.append(qt)
    actions_for_cap = audit["actions"].loc[audit["actions"]["Cap_ID"] == selected_cap]
    if not actions_for_cap.empty:
        story.append(Paragraph("Research documentation priorities", styles["CapHead"]))
        for _, action in actions_for_cap.head(5).iterrows():
            story.append(Paragraph(
                f"<b>{_pdf_txt(action['Category'])}:</b> {_pdf_txt(action['Recommendation'])}",
                styles["CapBody"],
            ))
        if len(actions_for_cap) > 5:
            story.append(Paragraph("Further recommendations are available in the Excel report.", styles["CapBody"]))
    # Keep caveats with the measured chart page to avoid a nearly blank spillover
    # sheet when six channels add extra rows to the overview and quality table.
    story.append(PageBreak())

    history = report["measurements"].loc[report["measurements"]["Cap_ID"] == selected_cap].sort_values("Hours")
    if history.empty:
        raise ValueError("Selected capacitor is absent from the validated report.")
    story.append(Paragraph(f"Measured history / {_pdf_txt(selected_cap)}", styles["CapTitle"]))
    story.append(Paragraph("Observed data only. Points are not connected or interpolated across gaps.", styles["CapSub"]))
    plot_bytes = _trend_png(history)
    from reportlab.lib.utils import ImageReader
    extra_plots = len(_graph_channels_for(history)) > 4
    # Six measured plots need to share the page with operating conditions and
    # the research-limitations notice. 0.62 fits all three without orphaning
    # a single paragraph on a nearly empty extra page.
    plot_width = usable_width * (0.62 if extra_plots else 0.78)
    plot_img = Image(BytesIO(plot_bytes), width=plot_width, height=plot_width * (7.4 if extra_plots else 5.5) / 11.7, hAlign="CENTER")
    story.append(plot_img)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Operating and test conditions recorded", styles["CapHead"]))
    conds = audit["conditions"].loc[audit["conditions"]["Cap_ID"] == selected_cap]
    descriptions = []
    for _, c in conds.iterrows():
        if c["Available"]:
            if c["Min_observed"] == c["Max_observed"]:
                span = f"{c['Min_observed']:g}"
            else:
                span = f"{c['Min_observed']:g}–{c['Max_observed']:g}"
            descriptions.append(f"{_pdf_txt(c['Condition'])}: {span} {_pdf_txt(c['Unit'])}")
    if descriptions:
        for i in range(0, len(descriptions), 2):
            cells_row = descriptions[i:i+2]
            while len(cells_row) < 2:
                cells_row.append("")
            story.append(Table([[Paragraph(_pdf_txt(x).replace("&amp;", "&"), styles["CapBody"]) for x in cells_row]], colWidths=[usable_width/2]*2))
    else:
        story.append(Paragraph("No operating or test conditions were documented in the uploaded data.", styles["CapBody"]))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Research limitations: Test-condition comparability, baseline quality, sensor calibration, and provenance require independent review. This PDF reports no capacitor health or RUL.", styles["CapBody"]))

    if gas_scenario is not None and len(gas_scenario):
        story.append(PageBreak())
        story.append(Paragraph("Physics scenario / assumptions only", styles["CapTitle"]))
        story.append(Paragraph(
            "Calculated hydrogen and pressure. These are NOT sensor measurements, and no RUL or vent-opening threshold is inferred.",
            styles["CapSub"],
        ))
        assumption_pairs = list(sorted((assumptions or {}).items()))
        scenario_info = [[Paragraph("Scenario input", styles["CapBold"]), Paragraph("Assumed value", styles["CapBold"])]]
        scenario_info += [[Paragraph(_pdf_txt(name), styles["CapCell"]), Paragraph(_pdf_txt(value), styles["CapCell"])]
                          for name, value in assumption_pairs]
        scenario_table = Table(scenario_info, colWidths=[usable_width/2]*2, hAlign="LEFT")
        scenario_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D9EAF3")),
                                         ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F5F8FB")]),
                                         ("BOTTOMPADDING",(0,0),(-1,-1),4), ("TOPPADDING",(0,0),(-1,-1),4)]))
        story.append(scenario_table)
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 3), dpi=120)
        ax1, ax2 = fig.subplots(1, 2)
        ax1.plot(gas_scenario["Hours"], gas_scenario["H2_Generated_mL_STP_est"], "o-", color="#C98222")
        ax1.set_title("Calculated H2 formed at STP", fontsize=11)
        ax1.set_ylabel("mL at 0 C and 1 atm")
        ax2.plot(gas_scenario["Hours"], gas_scenario["Pressure_Abs_kPa_est"], "o-", color="#C98222")
        ax2.set_title("Calculated internal pressure", fontsize=11)
        ax2.set_ylabel("kPa absolute")
        for ax in (ax1, ax2):
            ax.set_xlabel("Operating hours")
            ax.grid(alpha=.25)
        fig.subplots_adjust(wspace=.38, left=.07, right=.97, bottom=.20, top=.83)
        plot_buf = BytesIO()
        fig.savefig(plot_buf, format="png")
        story.append(Spacer(1, 10))
        story.append(Image(BytesIO(plot_buf.getvalue()), width=usable_width*.87, height=usable_width*.87*.30, hAlign="CENTER"))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Model limits: Constant retention and headspace volume; leakage is not directly hydrogen-producing current. No diffusion kinetics, gas dissolution, venting, or validated internal-gas calibration. Keep these estimates separate from direct measurements.", styles["CapBody"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CCD8E1"))
        canvas.line(34, 26, page_w-34, 26)
        canvas.setFont("Helvetica", 7.3)
        canvas.setFillColor(colors.HexColor("#617286"))
        canvas.drawString(34, 15, "CAPLIFE AI  •  RESEARCH-ONLY MEASUREMENT AUDIT")
        canvas.drawRightString(page_w-34, 15, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
