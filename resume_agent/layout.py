"""Measure the compiled PDF (kit §11): pages, blank space at the bottom, orphan lines.

Measuring the real PDF beats estimating characters — XCharter widths vary.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

# Calibrated on a full page from this template (A4, bottom=0.3in): the last
# baseline of a full page sits at ~18pt; one text line is ~12pt (10pt font).
FULL_PAGE_BOTTOM_Y = 18.0
LINE_PT = 12.0
ORPHAN_MAX_WORDS = 2
HEADINGS = {"education", "technologies and skills", "experience", "projects",
            "achievements & publications", "achievements", "publications & research", "awards"}


class Layout(BaseModel):
    pages: int
    free_lines: float                 # blank lines at the bottom of page 1 (only meaningful if pages == 1)
    orphans: list[str]                # bullet ids / "skills:<category>" whose last line is 1-2 words


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def measure(pdf_path: str, targets: dict[str, str]) -> Layout:
    """targets: id -> plain text of that block (bullets and skills lines)."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    ys: list[float] = []

    def visit(text, cm, tm, font, size):
        if text.strip():
            ys.append(tm[4] * cm[1] + tm[5] * cm[3] + cm[5])

    text = page.extract_text(visitor_text=visit)
    lowest = min(ys) if ys else FULL_PAGE_BOTTOM_Y
    free = max(0.0, (lowest - FULL_PAGE_BOTTOM_Y) / LINE_PT)

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    orphans: list[str] = []
    keyed = {tid: _key(t) for tid, t in targets.items()}
    for i, line in enumerate(lines):
        words = line.split()
        if not (0 < len(words) <= ORPHAN_MAX_WORDS) or line.lower() in HEADINGS or i == 0:
            continue
        k = _key(line)
        if len(k) < 2:
            continue
        for tid, tk in keyed.items():
            # the short line is the tail of a block that spans >1 line
            if tk.endswith(k) and len(tk) > len(k) + 20 and tid not in orphans:
                orphans.append(tid)
                break
    return Layout(pages=len(reader.pages), free_lines=round(free, 1), orphans=orphans)
