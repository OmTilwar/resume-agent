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
