"""Extract the quotable numbers from a bullet so they can be checked against facts.

Identifiers that merely contain digits (ResNet-18, F0.5, Recall@1, Llama-3-8B,
T200, PS3.15, mDeBERTa-v3, Top-1000) are skipped: a digit glued to a preceding
letter, '-', '@', '.', '/' or '_' is part of a name, not a claim. Years are skipped too.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# number, optional magnitude suffix (M/K/B) that is not the start of a word
_NUM = re.compile(
    r"(?<![A-Za-z0-9@.\-_/=^])"          # not glued to an identifier
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s?([MKB])(?![A-Za-z]))?"       # 1.7M, 200K (but not "43.3 MB")
)
# numbers right after '=' are hyperparameters, e.g. "β=4.0", "(α=2"
_EQ_NUM = re.compile(r"(?<==)(\d+(?:\.\d+)?)")
_SCI = re.compile(r"\d+(?:\.\d+)?e-?\d+", re.I)
_YEAR = re.compile(r"^(19|20)\d{2}$")
# split value ranges so each end is standalone: "400–600", "2024-25".
# Only en-dash / em-dash, or a year range; a plain hyphen between digits is
# usually a model name ("Llama-3-8B").
_RANGE = re.compile(r"(\d)\s?[–—]\s?(\d)|\b((?:19|20)\d{2})-(\d{2})\b")


def _split_range(m: re.Match) -> str:
    if m.group(1):
        return f"{m.group(1)} {m.group(2)}"
    return f"{m.group(3)} {m.group(4)}"


def normalize(token: str) -> str:
    """'1,017,209' -> '1017209'; '4.0' -> '4'; '1.7 M' -> '1.7M'; '64.50' -> '64.5'."""
    t = token.strip().replace(",", "").replace(" ", "")
    suffix = ""
    if t and t[-1] in "MKB":
        t, suffix = t[:-1], t[-1]
    try:
        d = Decimal(t)
    except InvalidOperation:
        return token.strip()
    s = format(d.normalize(), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s + suffix


_IDENT_PLUS = re.compile(r"[A-Za-z]\+\+\d*")  # C++17


def extract_numbers(text: str) -> set[str]:
    text = _IDENT_PLUS.sub(" ", text)
    sci = {m.group(0).lower() for m in _SCI.finditer(text)}
    clean = _SCI.sub(" ", text)
    clean = _RANGE.sub(_split_range, clean)
    clean = clean.replace("$", " ")
    out: set[str] = set(sci)
    for m in _NUM.finditer(clean):
        raw = m.group(1)
        n = normalize(raw + (m.group(2) or ""))
        nxt = clean[m.end():m.end() + 1]
        # a year is a bare 4-digit token: "2026" yes; "2,000+" / "2000%" / "2000 users"-with-comma no
        if _YEAR.match(n) and "," not in raw and nxt not in {"+", "%"}:
            continue
        out.add(n)
    for m in _EQ_NUM.finditer(clean):
        out.add(normalize(m.group(1)))
    return out


def normalize_values(values: list[str]) -> set[str]:
    out: set[str] = set()
    for v in values:
        out |= extract_numbers(v) or {normalize(v)}
    return out
