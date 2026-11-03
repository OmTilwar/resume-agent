"""Deterministic truth checks for a single bullet.

Phase 0 runs these over the kit's own bullet bank (tests). Phase 2's validator
node runs the same functions over LLM-written bullets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Context, Fact, FactStatus, KnowledgeBase, Project
from .numbers import extract_numbers, normalize_values


@dataclass(frozen=True)
class Violation:
    rule: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.where}: {self.message}"


def _strip_exceptions(text: str, kb: KnowledgeBase) -> str:
    out = text
    for exc in kb.rules.phrase_exceptions:
        out = re.sub(re.escape(exc), " ", out, flags=re.I)
    return out


def check_phrases(text: str, where: str, kb: KnowledgeBase, project: Project | None = None) -> list[Violation]:
    v: list[Violation] = []
    body = _strip_exceptions(text, kb)
    low = body.lower()
    for p in kb.rules.banned_phrases:
        if re.search(rf"(?<![A-Za-z]){re.escape(p.lower())}(?![A-Za-z])", low):
            v.append(Violation("banned-phrase", where, f"contains {p!r} (kit §10.1)"))
    for t in kb.rules.banned_bullet_terms:
        if t.lower() in low:
            v.append(Violation("banned-term", where, f"contains {t!r} (kit §2.2/§2.4)"))
    for pat in kb.rules.analogy_patterns:
        if re.search(pat, body, flags=re.I):
            v.append(Violation("analogy-clause", where, f"matches {pat!r} (kit §2.4.2)"))
    for pat in kb.rules.institution_guard:
        if re.search(pat, body):
            v.append(Violation("institution", where, f"matches {pat!r} — institution is IIIT Nagpur"))
    for claim in kb.experience.unconfirmed_claims:
        if claim.lower() in low:
            v.append(Violation("unconfirmed-claim", where, f"uses unconfirmed claim {claim!r} (kit §5)"))
    if project is not None:
        for t in project.banned_terms:
            if t.lower() in low:
                v.append(Violation("project-banned-term", where, f"{project.id} must never say {t!r}"))
        first = body.strip().split(" ", 1)[0] if body.strip() else ""
        if first in project.banned_verbs:
            v.append(Violation("lead-verb", where, f"{first!r} not allowed on {project.id} (team project)"))
    return v


def check_numbers(text: str, where: str, cited: list[Fact], ctx: Context, kb: KnowledgeBase) -> list[Violation]:
    """Every number in the bullet must come from a cited, allowed fact."""
    v: list[Violation] = []
    for f in cited:
        if f.status == FactStatus.BANNED:
            v.append(Violation("banned-fact", where, f"cites banned fact {f.id!r}"))
        elif not kb.fact_allowed(f, ctx):
            v.append(Violation("context-restricted", where, f"fact {f.id!r} not allowed for {ctx}"))
        elif f.eval_size is not None and f.eval_size < kb.rules.min_eval_size:
            v.append(Violation("small-eval", where, f"fact {f.id!r} eval set {f.eval_size} < {kb.rules.min_eval_size}"))
    allowed = set().union(*(normalize_values(f.values) for f in cited if kb.fact_allowed(f, ctx))) if cited else set()
    for n in sorted(extract_numbers(text) - allowed):
        v.append(Violation("unbacked-number", where, f"number {n!r} not in any cited allowed fact"))
    return v


def check_points_vs_percent(text: str, where: str) -> list[Violation]:
    """'+5.92%' after an 'X% to Y%' / 'X% -> Y%' change is a points delta, not percent (kit §2.4.5)."""
    v: list[Violation] = []
    if re.search(r"\d%\s*(?:→|->|to)\s*\d+(?:\.\d+)?%.*?\(\+\d+(?:\.\d+)?%\)", text):
        v.append(Violation("pts-vs-percent", where, "percentage-point delta written as % — use 'pts'"))
    return v


def check_bullet(text: str, where: str, cited: list[Fact], ctx: Context, kb: KnowledgeBase,
                 project: Project | None = None) -> list[Violation]:
    return (
        check_phrases(text, where, kb, project)
        + check_numbers(text, where, cited, ctx, kb)
        + check_points_vs_percent(text, where)
    )
