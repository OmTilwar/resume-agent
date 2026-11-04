"""Prompt builders. Each node sees only the slice of the KB it needs.

Contact details are never included in any prompt.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from .knowledge import Context, KnowledgeBase
from .state import JDAnalysis, JobPosting

WRITING_RULES = """\
Bullet formula: [strong past-tense verb] + [what was built] + [how: key technique/tool] + [quantified result].
Hard rules:
- Use ONLY the facts provided. Every number you write must appear verbatim in the `values` of a fact you cite in `fact_ids`. Never round, combine or invent numbers.
- A change between two percentages is in points: write "60.53% to 66.45% (+5.92 pts)", never "+5.92%".
- Length: either one line (<= 105 characters) or two full lines (180-215 characters). Nothing in between, nothing longer.
- Past tense. No first person. One idea per bullet; a semicolon may join one closely related result.
- Banned words/phrases: Responsible for, Worked on, Helped, Involved in, Various, etc., passionate, robust, cutting-edge, world-class, seamless, novel, new algorithm, state-of-the-art, production-grade, analogous to, applicable to, relevant to.
- Never mention Ragas, LangGraph, satellite imagery, React Native, Flutter.
- Use the JD's exact wording for skills only where the facts genuinely support it. If the JD wants something the facts don't show, leave it out — do not stretch.
- Keep the project's own domain; do not bolt other domains' keywords onto it.
- Plain text only: no LaTeX, no markdown. Unicode arrows (→) are fine.
"""


def _json(obj) -> str:
    """Compact JSON: indentation is whitespace the model is billed for."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


LENGTH_HINT = ("Each reference bullet shows its `chars`; they are all a correct length. When you adapt one, stay within "
               "about 10 characters of its `chars`. A one-line bullet must stay <= 105; a two-line one 180-215.")


def _sys(text: str) -> SystemMessage:
    return SystemMessage(content=text)


def analyze_messages(job: JobPosting, kb: KnowledgeBase) -> list:
    archetypes = "\n".join(
        f"- {a.id}: {a.name} (signals: {', '.join(a.signals)})" for a in kb.archetypes.values()
    )
    return [
        _sys(
            "You are a senior technical recruiter who screens ML/AI/SWE intern applications. "
            "Analyse the job description and return structured data.\n"
            "Rules: keywords must use the JD's exact wording; rank by how much the JD stresses them; "
            "'must' = required/core, 'nice' = preferred/bonus. Return exactly 10 keywords when the JD has that many. "
            "Pick archetype ids ONLY from this list:\n" + archetypes
        ),
        HumanMessage(content=f"Company: {job.company}\nRole: {job.role}\n\nJob description:\n{job.text}"),
    ]


def _keywords(analysis: JDAnalysis) -> str:
    return ", ".join(f"{k.term} ({k.importance})" for k in analysis.top_keywords)


def project_messages(project_id: str, analysis: JDAnalysis, ctx: Context, kb: KnowledgeBase,
                     n_bullets: int) -> list:
    p = kb.get_project(project_id)
    facts = [{"id": f.id, "claim": f.claim, "values": f.values} for f in kb.allowed_facts(project_id, ctx)]
    bank = [{"text": b.text, "chars": len(b.text), "fact_ids": b.fact_ids, **({"note": b.note} if b.note else {})}
            for b in kb.allowed_bullets(project_id, ctx)]
    titles = [f"{i}: {t.subtitle} | {', '.join(t.stack)}" for i, t in enumerate(p.titles)]
    payload = {
        "project": p.name,
        "what_it_is": p.one_liner,
        "project_rules": p.rules,
        "banned_terms_for_this_project": p.banned_terms,
        "banned_opening_verbs": p.banned_verbs,
        "stack_you_may_choose_from": p.stack,
        "title_options": titles,
        "facts": facts,
        "reference_bullets": bank,
    }
    return [
        _sys("You are a senior technical resume writer. Rewrite one project's bullets for a specific job.\n"
             + WRITING_RULES),
        HumanMessage(content=(
            f"Target role archetype: {analysis.archetype}; domain: {analysis.domain}.\n"
            f"JD keywords (ranked): {_keywords(analysis)}\n\n"
            f"Write exactly {n_bullets} bullets for this project, most JD-relevant first. "
            "The first bullet carries the strongest JD-relevant claim. Pick the best title option and 4-5 stack "
            "items (JD-matching first). You may reword the reference bullets, but every number must come from a "
            "cited fact. Respect each reference bullet's note (e.g. 'use only as a 4th bullet'). " + LENGTH_HINT + "\n\n"
            + _json(payload)
        )),
    ]


def experience_messages(groups: list[str], analysis: JDAnalysis, kb: KnowledgeBase) -> list:
    exp = kb.experience
    payload = {
        "allowed_titles": exp.titles,
        "facts": [{"id": f.id, "claim": f.claim, "values": f.values} for f in exp.facts],
        "reference_bullets": [
            {"group": g.id, "text": b.text, "chars": len(b.text), "fact_ids": b.fact_ids}
            for g in exp.groups if g.id in groups for b in g.bullets
        ],
        "never_claim": exp.unconfirmed_claims
        + ["that this internship's work was published (the IEEE paper is a different topic)"],
    }
    return [
        _sys("You are a senior technical resume writer. Rewrite the candidate's research-internship bullets "
             "for a specific job.\n" + WRITING_RULES),
        HumanMessage(content=(
            f"Target role archetype: {analysis.archetype}; JD keywords: {_keywords(analysis)}\n\n"
            "Write 3 bullets; put the JD's #1 theme first. Choose a title from allowed_titles. " + LENGTH_HINT + "\n\n"
            + _json(payload)
        )),
    ]


def fix_messages(payload: dict) -> list:
    """payload: {"sections": {section: {facts, reference_bullets, project_rules}},
                 "bullets": [{id, section, text, current_chars, problems}], "target_chars"}.
    Facts and rules go once per section, not once per bullet (they were ~60% of fixer input)."""
    return [
        _sys("You repair resume bullets that failed review. Fix every listed problem and keep everything that was "
             "fine. Keep the bullet's meaning and its strongest metric. Each bullet may only use the facts of its "
             "own `section`.\n" + WRITING_RULES
             + "\nReturn one fixed bullet per id, with the fact_ids it uses."),
        HumanMessage(content=_json(payload)),
    ]


def skills_messages(analysis: JDAnalysis, project_ids: list[str], kb: KnowledgeBase) -> list:
    inventory = {cat: [s.display for s in items] for cat, items in kb.skills_by_category().items()}
    stacks = {pid: kb.get_project(pid).stack for pid in project_ids}
    return [
        _sys(
            "You build the 'Technologies and Skills' section of a one-page resume.\n"
            "Rules: 4-5 lines; each line has a category name renamed to mirror the JD "
            "(e.g. 'LLMs & AI Agents', 'Retrieval & Ranking'); 8-14 entries per line; JD keywords first in "
            "each line; order lines by JD importance; drop irrelevant categories. "
            "Copy entries EXACTLY as written in the inventory, including any ' (Familiar)' suffix — never remove "
            "it. Never add a skill that is not in the inventory. Also pick 4-5 relevant courses, copied exactly."
        ),
        HumanMessage(content=(
            f"JD keywords (ranked): {_keywords(analysis)}\n"
            f"Projects on the page and their stacks: {json.dumps(stacks)}\n\n"
            f"Inventory: {_json(inventory)}\n\n"
            f"Courses: {json.dumps(kb.profile.coursework + kb.profile.coursework_shorthand)}"
        )),
    ]
