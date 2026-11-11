"""Extract plain text from a CV PDF."""
from __future__ import annotations

import io
import re

from pypdf import PdfReader


class PDFTextError(ValueError):
    pass


# Some fonts (e.g. Inter, popular in CV templates) store "case-sensitive" punctuation
# in the Private Use Area, so PDF readers return unreadable symbols instead of + - ( ).
PUA_MAP = {
    "": "+", "": "-", "": "(", "": ")",
    "": "[", "": "]", "": "{", "": "}",
    "": "–", "": "—", "": "×", "": "=",
}


def clean_text(text: str) -> str:
    for src, dst in PUA_MAP.items():
        text = text.replace(src, dst)
    text = re.sub(r"[-]", "", text)          # drop any remaining unknown private symbols
    text = text.replace(" ", " ").replace("­", "").replace("￾", "-")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # join hyphenated line breaks
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)         # "EK -RA8D1" -> "EK-RA8D1" (font spacing quirk)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise PDFTextError(f"This file could not be opened as a PDF ({exc}).") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PDFTextError("The PDF is password-protected. Please upload an unlocked version.") from exc

    text = clean_text("\n".join(page.extract_text() or "" for page in reader.pages))
    if len(text) < 100:
        raise PDFTextError(
            "Almost no text could be read from this PDF. It is probably a scanned image — "
            "export your CV as a text-based PDF (e.g. 'Save as PDF' from Word/Google Docs).")
    return text
