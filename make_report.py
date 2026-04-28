"""
make_report.py
--------------
Renders REPORT.md to REPORT.pdf using the same reportlab styling as the
April 14 check-in PDF, so the final report looks consistent.

Run:  python3 make_report.py
"""
import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


HERE = Path(__file__).parent
SRC  = HERE / "REPORT.md"
OUT  = HERE / "REPORT.pdf"


# ── Styles (matching make_checkin.py for consistency) ────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle("Title",
    parent=styles["Title"], fontSize=20, leading=24,
    textColor=colors.HexColor("#1a1a2e"), spaceAfter=4, alignment=TA_CENTER)

subtitle_style = ParagraphStyle("Subtitle",
    parent=styles["Normal"], fontSize=11, leading=14,
    textColor=colors.HexColor("#444444"), spaceAfter=2, alignment=TA_CENTER)

h1_style = ParagraphStyle("H1",
    parent=styles["Heading1"], fontSize=14, leading=18,
    textColor=colors.HexColor("#1a1a2e"),
    spaceBefore=18, spaceAfter=8)

h2_style = ParagraphStyle("H2",
    parent=styles["Heading2"], fontSize=11.5, leading=15,
    textColor=colors.HexColor("#2c3e7a"),
    spaceBefore=12, spaceAfter=4)

body_style = ParagraphStyle("Body",
    parent=styles["Normal"], fontSize=10.5, leading=15,
    textColor=colors.HexColor("#222222"), spaceAfter=6,
    alignment=TA_LEFT,
    firstLineIndent=0)

caption_style = ParagraphStyle("Caption",
    parent=styles["Normal"], fontSize=9, leading=12,
    textColor=colors.HexColor("#555555"), spaceAfter=10,
    alignment=TA_CENTER, fontName="Helvetica-Oblique")


def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#cccccc"),
                      spaceBefore=6, spaceAfter=6)

def sp(n=6):
    return Spacer(1, n)


def md_inline(s: str) -> str:
    """Convert minimal markdown inline syntax to reportlab-friendly HTML."""
    # Bold **text** -> <b>text</b>
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Italic *text* (avoid eating the bold case)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    # Backtick code `x` -> <font face="Courier">x</font>
    s = re.sub(r"`([^`]+?)`",
               r'<font face="Courier" size="9.5">\1</font>', s)
    return s


def render(md: str):
    """Walk the markdown and yield reportlab flowables."""
    flowables = []
    lines = md.splitlines()
    i = 0
    in_paragraph = []

    def flush_paragraph():
        if in_paragraph:
            text = " ".join(in_paragraph).strip()
            if text:
                flowables.append(Paragraph(md_inline(text), body_style))
            in_paragraph.clear()

    while i < len(lines):
        line = lines[i].rstrip()

        # Image: ![caption](path.png)
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)\s*$", line.strip())
        if m:
            flush_paragraph()
            caption, path = m.group(1), m.group(2)
            img_path = HERE / path
            if img_path.exists():
                img = Image(str(img_path), width=5.8 * inch, height=5.8 * inch * 0.6)
                img.hAlign = "CENTER"
                flowables.append(sp(4))
                flowables.append(img)
                if caption:
                    flowables.append(Paragraph(md_inline(caption), caption_style))
                else:
                    flowables.append(sp(8))
            else:
                flowables.append(Paragraph(f"[missing image: {path}]", body_style))
            i += 1
            continue

        # Title (single # at top of file)
        if line.startswith("# "):
            flush_paragraph()
            flowables.append(sp(6))
            flowables.append(Paragraph(md_inline(line[2:]), title_style))
            i += 1
            continue

        # Heading 2
        if line.startswith("## "):
            flush_paragraph()
            flowables.append(Paragraph(md_inline(line[3:]), h1_style))
            i += 1
            continue

        # Heading 3
        if line.startswith("### "):
            flush_paragraph()
            flowables.append(Paragraph(md_inline(line[4:]), h2_style))
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            flush_paragraph()
            flowables.append(hr())
            i += 1
            continue

        # Author / date subtitle (bold-prefixed line right after title)
        if line.startswith("**") and line.endswith("**"):
            flush_paragraph()
            inner = line.strip("*").strip()
            flowables.append(Paragraph(inner, subtitle_style))
            i += 1
            continue

        if line.startswith("**Authors:**") or line.startswith("**Date:**"):
            flush_paragraph()
            flowables.append(Paragraph(md_inline(line), subtitle_style))
            i += 1
            continue

        # Blank line ends a paragraph
        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        # Otherwise accumulate paragraph text
        in_paragraph.append(line.strip())
        i += 1

    flush_paragraph()
    return flowables


def main():
    md = SRC.read_text(encoding="utf-8")
    story = render(md)

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=1*inch, rightMargin=1*inch,
        topMargin=0.9*inch, bottomMargin=0.9*inch,
        title="GM Mode: NBA Trade Package Optimizer",
        author="Jason Fang and Jonathan Mehrotra",
    )
    doc.build(story)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
