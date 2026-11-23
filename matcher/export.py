"""Export a cover letter as DOCX or PDF in DIN 5008 (Form B) layout.

Form B positions used (A4, from the top edge):
  0–45 mm   sender block (right-aligned)
  45–90 mm  address field: 5 mm return line + recipient
  ~98 mm    date line (right), then subject (bold), salutation, body, closing, name
Margins: left 25 mm, right 20 mm.
"""
from __future__ import annotations

import io
from typing import Any
from xml.sax.saxutils import escape


def _sender_lines(profile: dict[str, Any]) -> list[str]:
    keys = ("full_name", "street", "postal_city", "phone", "email")
    return [str(profile.get(k)).strip() for k in keys if str(profile.get(k) or "").strip()]


def _recipient_lines(recipient: dict[str, Any]) -> list[str]:
    keys = ("company", "contact", "street", "postal_city")
    return [str(recipient.get(k)).strip() for k in keys if str(recipient.get(k) or "").strip()]


def _return_line(profile: dict[str, Any]) -> str:
    parts = [profile.get("full_name"), profile.get("street"), profile.get("postal_city")]
    return " · ".join(str(p).strip() for p in parts if str(p or "").strip())


def _date_line(place: str, date_str: str) -> str:
    return f"{place}, {date_str}" if place else date_str


# --------------------------------------------------------------------------- DOCX
def to_docx(letter: dict[str, Any], profile: dict[str, Any], recipient: dict[str, Any],
            place: str, date_str: str) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Mm(297), Mm(210)
    sec.top_margin, sec.bottom_margin = Mm(20), Mm(20)
    sec.left_margin, sec.right_margin = Mm(25), Mm(20)

    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.15

    def para(text: str = "", align=None, bold=False, size=None, underline=False, space_after=0):
        p = doc.add_paragraph()
        if align is not None:
            p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text)
        run.bold = bold
        run.underline = underline
        if size:
            run.font.size = Pt(size)
        return p

    # sender block (top 20–45 mm)
    sender = _sender_lines(profile)
    for i, line in enumerate(sender):
        para(line, align=WD_ALIGN_PARAGRAPH.RIGHT, bold=(i == 0), size=10)
    # pad to the address field (~45 mm): each 10 pt line ≈ 4.1 mm
    for _ in range(max(0, 6 - len(sender))):
        para("", size=10)

    # address field
    para(_return_line(profile), size=7, underline=True, space_after=4)
    rec = _recipient_lines(recipient)
    for line in rec:
        para(line)
    for _ in range(max(1, 8 - len(rec))):
        para("")

    para(_date_line(place, date_str), align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=18)
    para(letter.get("subject", ""), bold=True, space_after=18)
    para(letter.get("salutation", ""), space_after=11)
    for p in letter.get("paragraphs", []):
        para(p, space_after=11).alignment = WD_ALIGN_PARAGRAPH.LEFT
    para(letter.get("closing", ""), space_after=36)
    para(profile.get("full_name", ""))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------- PDF
def to_pdf(letter: dict[str, Any], profile: dict[str, Any], recipient: dict[str, Any],
           place: str, date_str: str) -> bytes:
    """Build the PDF; if it doesn't fit on one page, retry with slightly smaller type and spacing."""
    from pypdf import PdfReader

    pdf = b""
    for font_size, gap in ((11, 4), (10.5, 3.2), (10, 2.6), (9.5, 2.2)):
        pdf = _build_pdf(letter, profile, recipient, place, date_str, font_size, gap)
        if len(PdfReader(io.BytesIO(pdf)).pages) == 1:
            break
    return pdf


def _build_pdf(letter: dict[str, Any], profile: dict[str, Any], recipient: dict[str, Any],
               place: str, date_str: str, font_size: float, gap_mm: float) -> bytes:
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=font_size, leading=font_size * 1.32,
                          alignment=TA_LEFT)
    right = ParagraphStyle("right", parent=base, alignment=TA_RIGHT)
    sender_st = ParagraphStyle("sender", parent=right, fontSize=9.5, leading=12)
    small = ParagraphStyle("small", parent=base, fontSize=7, leading=9)
    bold = ParagraphStyle("bold", parent=base, fontName="Helvetica-Bold")

    def P(text: str, style=base):
        return Paragraph(escape(text or ""), style)

    top = 20 * mm
    width = A4[0] - 25 * mm - 20 * mm

    sender = _sender_lines(profile)
    sender_html = "<br/>".join(
        f"<b>{escape(line)}</b>" if i == 0 else escape(line) for i, line in enumerate(sender))
    header = Table([[Paragraph(sender_html, sender_st)]], colWidths=[width], rowHeights=[45 * mm - top])
    header.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0)]))

    return_line = Paragraph(f"<u>{escape(_return_line(profile))}</u>", small)
    rec_html = "<br/>".join(escape(line) for line in _recipient_lines(recipient))
    address = Table([[return_line], [Paragraph(rec_html, base)]], colWidths=[85 * mm],
                    rowHeights=[5 * mm, 40 * mm], hAlign="LEFT")
    address.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1)]))

    story = [header, address, Spacer(1, 6 * mm), P(_date_line(place, date_str), right), Spacer(1, 7 * mm),
             P(letter.get("subject", ""), bold), Spacer(1, 6 * mm), P(letter.get("salutation", "")),
             Spacer(1, gap_mm * mm)]
    for p in letter.get("paragraphs", []):
        story += [P(p), Spacer(1, gap_mm * mm)]
    # closing and name always stay together
    story.append(KeepTogether([P(letter.get("closing", "")), Spacer(1, 12 * mm), P(profile.get("full_name", ""))]))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=25 * mm, rightMargin=20 * mm, topMargin=top,
                      bottomMargin=15 * mm, title=letter.get("subject", "Cover letter"),
                      author=profile.get("full_name", "")).build(story)
    return buf.getvalue()


# =========================================================================== tailored resume
def _contact_line(profile: dict[str, Any]) -> str:
    keys = ("street", "postal_city", "phone", "email")
    return " · ".join(str(profile.get(k)).strip() for k in keys if str(profile.get(k) or "").strip())


def _entry_head(e: dict[str, Any]) -> tuple[str, str]:
    """(left: 'Title — Organisation, Location', right: dates)"""
    org = ", ".join(x for x in (e.get("organisation"), e.get("location")) if x)
    return " — ".join(x for x in (e.get("title"), org) if x), e.get("dates", "")


def resume_to_docx(resume: dict[str, Any], profile: dict[str, Any]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt, RGBColor

    from .resume import HEADINGS, SECTION_ORDER

    h = HEADINGS.get(resume.get("language", "en"), HEADINGS["en"])
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Mm(297), Mm(210)
    sec.top_margin = sec.bottom_margin = Mm(15)
    sec.left_margin = sec.right_margin = Mm(18)
    text_width = Mm(210 - 36)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.1

    def para(text="", bold=False, size=None, italic=False, space_after=0, color=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        if text:
            run = p.add_run(text)
            run.bold, run.italic = bold, italic
            if size:
                run.font.size = Pt(size)
            if color:
                run.font.color.rgb = RGBColor(*color)
        return p

    def heading(text):
        p = para(text.upper(), bold=True, size=10.5, space_after=3, color=(0x1F, 0x3A, 0x5F))
        p.paragraph_format.space_before = Pt(9)
        border = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "1F3A5F")):
            bottom.set(qn(k), v)
        border.append(bottom)
        p._p.get_or_add_pPr().append(border)

    para(profile.get("full_name", ""), bold=True, size=18)
    if resume.get("headline"):
        para(resume["headline"], size=11, color=(0x1F, 0x3A, 0x5F))
    if _contact_line(profile):
        para(_contact_line(profile), size=9)

    for key in SECTION_ORDER:
        value = resume.get(key)
        if not value:
            continue
        heading(h[key])
        if key == "summary":
            para(value)
        elif key == "skills":
            for g in value:
                p = para()
                if g["category"]:
                    p.add_run(f"{g['category']}: ").bold = True
                p.add_run(", ".join(g["items"]))
        elif key in ("languages", "certifications"):
            para(" · ".join(value))
        else:
            for e in value:
                left, right = _entry_head(e)
                p = para(space_after=1)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.tab_stops.add_tab_stop(text_width, WD_TAB_ALIGNMENT.RIGHT)
                p.add_run(left).bold = True
                if right:
                    p.add_run(f"\t{right}")
                for b in e["bullets"]:
                    doc.add_paragraph(b, style="List Bullet").paragraph_format.space_after = Pt(0)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def resume_to_pdf(resume: dict[str, Any], profile: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (HRFlowable, KeepTogether, ListFlowable, ListItem, Paragraph,
                                    SimpleDocTemplate, Table, TableStyle)

    from .resume import HEADINGS, SECTION_ORDER

    h = HEADINGS.get(resume.get("language", "en"), HEADINGS["en"])
    accent = colors.HexColor("#1F3A5F")
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=12.5)
    name_st = ParagraphStyle("name", parent=base, fontName="Helvetica-Bold", fontSize=18, leading=22)
    headline_st = ParagraphStyle("headline", parent=base, fontSize=11, leading=14, textColor=accent)
    small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=11)
    head_st = ParagraphStyle("head", parent=base, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                             textColor=accent, spaceBefore=8)
    bold = ParagraphStyle("bold", parent=base, fontName="Helvetica-Bold")
    right = ParagraphStyle("right", parent=base, alignment=2)
    width = A4[0] - 36 * mm - 12  # margins, minus the frame's 6 pt padding on each side

    def P(text, style=base):
        return Paragraph(escape(text or ""), style)

    story = [P(profile.get("full_name", ""), name_st)]
    if resume.get("headline"):
        story.append(P(resume["headline"], headline_st))
    if _contact_line(profile):
        story.append(P(_contact_line(profile), small))

    for key in SECTION_ORDER:
        value = resume.get(key)
        if not value:
            continue
        block = [P(h[key].upper(), head_st), HRFlowable(width="100%", thickness=0.6, color=accent,
                                                         spaceBefore=1, spaceAfter=3)]
        if key == "summary":
            block.append(P(value))
            story += block
        elif key == "skills":
            for g in value:
                label = f"<b>{escape(g['category'])}:</b> " if g["category"] else ""
                block.append(Paragraph(label + escape(", ".join(g["items"])), base))
            story += block
        elif key in ("languages", "certifications"):
            block.append(P(" · ".join(value)))
            story += block
        else:
            for i, e in enumerate(value):
                left, dates = _entry_head(e)
                row = Table([[P(left, bold), P(dates, right)]], colWidths=[width * 0.74, width * 0.26])
                row.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                         ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                                         ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                items = [row]
                if e["bullets"]:
                    items.append(ListFlowable([ListItem(P(b), leftIndent=10) for b in e["bullets"]],
                                              bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8))
                # keep the heading with the first entry, and each entry in one piece
                story.append(KeepTogether(block + items if i == 0 else items))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                      bottomMargin=15 * mm, title=resume.get("headline") or "Resume",
                      author=profile.get("full_name", "")).build(story)
    return buf.getvalue()
