from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "treeheap_edge_ai_evidence_brief.zh.md"
OUTPUT = ROOT / "TreeHeap-端侧低功耗AI技术证据简报.pdf"

FONT = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
if not FONT.exists():
    FONT = ROOT / "NotoSansSC-VF.ttf"
if not FONT.exists():
    FONT = Path(r"C:\Windows\Fonts\Deng.ttf")
if not FONT.exists():
    FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
if FONT.suffix.lower() == ".ttc":
    FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
pdfmetrics.registerFont(TTFont("NotoSC", str(FONT)))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ZHTitle", parent=styles["Title"], fontName="NotoSC", fontSize=20,
    leading=26, alignment=TA_CENTER, spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="ZHHeading", parent=styles["Heading2"], fontName="NotoSC", fontSize=13,
    leading=18, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#17365D"),
))
styles.add(ParagraphStyle(
    name="ZHBody", parent=styles["BodyText"], fontName="NotoSC", fontSize=9.5,
    leading=15, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="ZHSmall", parent=styles["BodyText"], fontName="NotoSC", fontSize=8,
    leading=11, textColor=colors.HexColor("#555555"),
))


def para(text, style="ZHBody"):
    text = html.escape(text).replace("`", "")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return Paragraph(text, styles[style])


def build():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    table_rows = []
    in_table = False

    for line in lines:
        if line.startswith("# "):
            story.append(para(line[2:], "ZHTitle"))
            continue
        if line.startswith("## "):
            story.append(para(line[3:], "ZHHeading"))
            continue
        if line.startswith("### "):
            story.append(para(line[4:], "ZHHeading"))
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                continue
            table_rows.append(cells)
            in_table = True
            continue
        if in_table:
            if table_rows:
                widths = [42 * mm, 42 * mm, 42 * mm]
                table = Table([[para(c, "ZHSmall") for c in row] for row in table_rows], colWidths=widths, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8CA6BF")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 5)])
                table_rows = []
            in_table = False
        if not line.strip():
            continue
        if line.startswith("- "):
            story.append(para("• " + line[2:]))
        elif re.match(r"^\d+\. ", line):
            story.append(para(line))
        elif line.startswith("`") and line.endswith("`"):
            story.append(para(line.strip("`"), "ZHSmall"))
        elif line.startswith("> "):
            story.append(para("引用：" + line[2:]))
        elif not line.startswith("版本：") and not line.startswith("用途：") and not line.startswith("证据原则："):
            story.append(para(line))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("NotoSC", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 10 * mm, "SameTime / TreeHeap | 技术证据简报 | 2026-09-29")
        canvas.drawRightString(192 * mm, 10 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="TreeHeap 端侧低功耗 AI 技术证据简报",
        author="SameTime / Houming818",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
