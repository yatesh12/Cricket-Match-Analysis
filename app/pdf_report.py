"""CricketIQ — Pre-Match Broadcast Hot Zone Report Generator (PDF).

Auto-generates a professional PDF report for each upcoming IPL match showing:
  - Predicted top 5 peak engagement windows
  - Recommended ad slot placements
  - Estimated revenue per slot in ₹
  - Team and match context

Uses ReportLab for PDF generation.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.broadcast_monetisation import AD_RATES


class HotZonePDFReport:
    """Generate pre-match 'Predicted Hot Zone Report' PDF."""

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import inch, cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            self._ready = True
        except ImportError:
            self._ready = False

    def generate(
        self,
        match_data: dict,
        output_path: str,
    ) -> str:
        """Generate the PDF report for a single match.

        Args:
            match_data: dict from HotZoneReport.generate_match_report()
            output_path: full path for the output PDF file.

        Returns:
            Path to generated PDF.
        """
        if not self._ready:
            return self._generate_text_fallback(match_data, output_path)

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Title"],
            fontSize=22, spaceAfter=12,
        )
        story.append(Paragraph("CricketIQ — Pre-Match Broadcast Report", title_style))
        story.append(Spacer(1, 6))

        # Meta
        meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, textColor=colors.gray)
        story.append(Paragraph(f"Match: {match_data.get('match_id', 'N/A')}", meta_style))
        story.append(Paragraph(f"Team: {match_data.get('team1', 'N/A')}", meta_style))
        story.append(Paragraph(f"Generated: {match_data.get('generated_at', datetime.now().isoformat())}", meta_style))
        story.append(Spacer(1, 12))

        # Summary metrics
        summary_data = [
            ["Metric", "Value"],
            ["Total Overs", str(match_data.get("total_overs", 0))],
            ["Peak Engagement Windows", str(match_data.get("peak_overs", 0))],
            ["Peak Threshold (pctile)", f"{match_data.get('peak_threshold', 0.8)*100:.0f}th"],
            ["Estimated Ad Revenue", f"₹{match_data.get('estimated_ad_revenue_cr', 0)} crore"],
        ]
        summary_table = Table(summary_data, colWidths=[250, 150])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8eaf6")]),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Ad Rate Card
        story.append(Paragraph("Ad Rate Card (IPL Season Estimates)", styles["Heading2"]))
        rate_data = [
            ["Slot Type", "Rate per 30s", "Slots per Over", "Rate per Over"],
            ["Peak (p > 75th pctile)", f"₹{AD_RATES['peak']/100000:.0f}L", "4", f"₹{AD_RATES['peak']*4/100000:.0f}L"],
            ["Standard (p 50-75)", f"₹{AD_RATES['standard']/100000:.0f}L", "4", f"₹{AD_RATES['standard']*4/100000:.0f}L"],
            ["Low (p < 50)", f"₹{AD_RATES['low']/100000:.0f}L", "4", f"₹{AD_RATES['low']*4/100000:.0f}L"],
        ]
        rate_table = Table(rate_data, colWidths=[180, 120, 100, 120])
        rate_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ffebee")]),
        ]))
        story.append(rate_table)
        story.append(Spacer(1, 20))

        # Top 5 Hot Zones
        story.append(Paragraph("Top 5 Predicted Peak Engagement Windows", styles["Heading2"]))
        hot_zones = match_data.get("top_5_hot_zones", [])
        zone_data = [["#", "Innings", "Over", "Excitement Score", "Ad Value (₹)"]]
        for i, zone in enumerate(hot_zones, 1):
            val = AD_RATES["peak"] * 4
            zone_data.append([
                str(i),
                str(zone.get("innings", "")),
                str(zone.get("over", "")),
                f"{zone.get('excitement_normalised', 0):.3f}",
                f"₹{val/100000:.0f}L",
            ])
        zone_table = Table(zone_data, colWidths=[30, 70, 60, 130, 100])
        zone_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#004d40")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e0f2f1")]),
        ]))
        story.append(zone_table)
        story.append(Spacer(1, 20))

        # Full over-by-over detail
        hot_over_details = match_data.get("hot_zone_overs", [])
        if hot_over_details:
            story.append(Paragraph("All Peak Engagement Overs", styles["Heading2"]))
            detail_data = [["Innings", "Over", "Excitement", "Runs", "Wickets"]]
            for d in hot_over_details:
                detail_data.append([
                    str(d.get("innings", "")),
                    str(d.get("over", "")),
                    f"{d.get('excitement_normalised', 0):.3f}",
                    str(d.get("runs_scored", 0)),
                    str(d.get("wickets", 0)),
                ])
            detail_table = Table(detail_data, colWidths=[70, 60, 100, 60, 70])
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a148c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(detail_table)

        # Footer
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            "Footer", parent=styles["Normal"],
            fontSize=8, textColor=colors.gray, alignment=1,
        )
        story.append(Paragraph("CricketIQ — Enterprise Cricket Analytics Platform", footer_style))
        story.append(Paragraph("Confidential — for authorised recipients only", footer_style))

        doc.build(story)
        return output_path

    def _generate_text_fallback(self, match_data: dict, output_path: str) -> str:
        """Fallback plain-text report when ReportLab is not available."""
        lines = []
        lines.append("=" * 60)
        lines.append("  CricketIQ — Pre-Match Broadcast Report (TEXT)")
        lines.append("=" * 60)
        lines.append(f"  Match: {match_data.get('match_id', 'N/A')}")
        lines.append(f"  Team: {match_data.get('team1', 'N/A')}")
        lines.append(f"  Generated: {match_data.get('generated_at', 'N/A')}")
        lines.append("")
        lines.append(f"  Estimated Ad Revenue: ₹{match_data.get('estimated_ad_revenue_cr', 0)} crore")
        lines.append(f"  Peak Windows: {match_data.get('peak_overs', 0)}/{match_data.get('total_overs', 0)} overs")
        lines.append("")
        lines.append("  Top 5 Peak Windows:")
        for i, zone in enumerate(match_data.get("top_5_hot_zones", []), 1):
            lines.append(f"    {i}. Innings {zone.get('innings')}, Over {zone.get('over')}: "
                         f"excitement = {zone.get('excitement_normalised', 0):.3f}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("CricketIQ — Confidential")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return output_path


def generate_match_report_pdf(match_data: dict, output_dir: str = "outputs/reports") -> str:
    """Generate PDF report for a match and return the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    match_id = match_data.get("match_id", "unknown_match")
    sanitised = match_id.replace("/", "_").replace(" ", "_")
    output_path = str(Path(output_dir) / f"{sanitised}_hot_zone_report.pdf")

    generator = HotZonePDFReport()
    return generator.generate(match_data, output_path)
