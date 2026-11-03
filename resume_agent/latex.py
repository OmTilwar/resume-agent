"""Plain text -> LaTeX (kit §13), and the Jinja environment for templates/."""
from __future__ import annotations

import re
from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "^": r"\^{}",
}
_UNICODE = {
    "→": r"$\rightarrow$", "×": r"$\times$", "β": r"$\beta$", "α": r"$\alpha$",
    "≤": r"$\leq$", "≥": r"$\geq$", "±": r"$\pm$", "≈": r"$\approx$", "★": r"$\star$",
    "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "~": r"$\sim$",
}
_SPECIAL_RE = re.compile("|".join(re.escape(k) for k in _SPECIALS))
_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE))


def escape(text: str) -> str:
    """Escape plain text for LaTeX body. Do NOT pass already-LaTeX strings."""
    out = _SPECIAL_RE.sub(lambda m: _SPECIALS[m.group(0)], text)
    out = out.replace("<", r"$<$").replace(">", r"$>$")
    out = _UNICODE_RE.sub(lambda m: _UNICODE[m.group(0)], out)
    return out


def highlight(latex: str) -> str:
    """Wrap a LaTeX fragment in the kit's metric highlight."""
    return r"\textbf{\textcolor{NavyBlue}{" + latex + "}}"


# A metric token: optional ~ < > +, number, optional unit. Same "not part of an
# identifier" rule as knowledge.numbers; a trailing '-' (2-member, 128-dim) is
# a descriptor, not a metric.
_METRIC = re.compile(
    r"(?<![A-Za-z0-9@.\-_/=^+])"
    r"([~<>+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s?(?:%|x|ms|s|M|K|B|dB|MB|pts|Hz|kHz))?\+?)"
    r"(?![A-Za-z0-9\-])"
)
_YEARLIKE = re.compile(r"^(19|20)\d{2}$")


def escape_and_highlight(text: str, max_highlights: int = 4) -> str:
    """Escape plain text and wrap up to `max_highlights` metrics in NavyBlue bold."""
    cands = [m for m in _METRIC.finditer(text) if not _YEARLIKE.match(m.group(1))]

    def strength(m: re.Match) -> int:
        t = m.group(1)
        return 2 if re.search(r"[%.,~<>+]|[A-Za-z]", t) or int(re.sub(r"\D", "", t) or 0) > 20 else 1

    keep = sorted(sorted(cands, key=lambda m: -strength(m))[:max_highlights], key=lambda m: m.start())
    out: list[str] = []
    pos = 0
    for m in keep:
        out.append(escape(text[pos:m.start(1)]))
        out.append(highlight(escape(m.group(1))))
        pos = m.end(1)
    out.append(escape(text[pos:]))
    return "".join(out)


def env() -> jinja2.Environment:
    # LaTeX-safe delimiters: (( var )), ((* block *)), ((# comment #))
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        variable_start_string="((", variable_end_string="))",
        block_start_string="((*", block_end_string="*))",
        comment_start_string="((#", comment_end_string="#))",
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
        autoescape=False, undefined=jinja2.StrictUndefined,
    )
