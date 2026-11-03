"""Phase 2 content validation: every bullet on the page, before LaTeX.

Deterministic checks (kit §2, §10, §11) + an LLM faithfulness judge that reads
each bullet against the claims of the facts it cites (catches wording drift the
number check can't, e.g. "3,200+ images" when the fact says "instances").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .knowledge import Context, Fact, KnowledgeBase
from .knowledge.checks import check_bullet
from .knowledge.numbers import extract_numbers
from .state import FinalBullet, ResumeState

Severity = Literal["hard", "soft"]   # hard = must not ship; soft = fix if we can


class Issue(BaseModel):
    target: str          # bullet id, e.g. "finsight#2", "experience#1"
    rule: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}/{self.severity}] {self.target}: {self.message}"


# ---- bullet addressing -----------------------------------------------------

@dataclass
class Slot:
    id: str
    section: str                 # project id or "experience"
    index: int
    bullet: FinalBullet
    facts: dict[str, Fact]       # allowed facts for this section


def slots(state: ResumeState, kb: KnowledgeBase) -> list[Slot]:
    ctx = Context(archetype=state["analysis"].archetype, company=state["job"].company)
    out = [Slot(f"experience#{i + 1}", "experience", i, b, {f.id: f for f in kb.experience.facts})
           for i, b in enumerate(state["experience"].bullets)]
    for p in state["projects"]:
        facts = {f.id: f for f in kb.allowed_facts(p.project_id, ctx)}
        out += [Slot(f"{p.project_id}#{i + 1}", p.project_id, i, b, facts) for i, b in enumerate(p.bullets)]
    return out


# ---- deterministic checks --------------------------------------------------

def visible_len(text: str) -> int:
    return len(re.sub(r"\s+", " ", text.strip()))


def length_issue(text: str, kb: KnowledgeBase, target: str) -> Issue | None:
    b = kb.rules.page_budget
    n = visible_len(text)
    if n > b["two_line_bullet_max"]:
        return Issue(target=target, rule="too-long", severity="hard",
                     message=f"{n} chars spills to a 3rd line; cut to {b['two_line_bullet_min']}-{b['two_line_bullet_max']}")
    if b["one_line_bullet_max"] + 10 < n < b["two_line_bullet_min"] - 10:
        return Issue(target=target, rule="half-line", severity="soft",
                     message=f"{n} chars leaves a half-empty 2nd line; make it <= {b['one_line_bullet_max']} "
                             f"or {b['two_line_bullet_min']}-{b['two_line_bullet_max']}")
    return None


def first_verb(text: str) -> str:
    return (text.strip().split(" ", 1)[0] if text.strip() else "").strip(",;:")


def deterministic_issues(state: ResumeState, kb: KnowledgeBase) -> list[Issue]:
    ctx = Context(archetype=state["analysis"].archetype, company=state["job"].company)
    issues: list[Issue] = []
    ss = slots(state, kb)
    for s in ss:
        project = kb.projects.get(s.section)
        cited = [s.facts[f] for f in s.bullet.fact_ids if f in s.facts]
        unknown = [f for f in s.bullet.fact_ids if f not in s.facts]
        if unknown:
            issues.append(Issue(target=s.id, rule="unknown-fact", severity="hard",
                                message=f"cites facts not allowed here: {unknown}"))
        for v in check_bullet(s.bullet.text, s.id, cited, ctx, kb, project):
            issues.append(Issue(target=s.id, rule=v.rule, severity="hard", message=v.message))
        li = length_issue(s.bullet.text, kb, s.id)
        if li:
            issues.append(li)
        if s.section != "vio" and not extract_numbers(s.bullet.text):
            issues.append(Issue(target=s.id, rule="no-number", severity="soft",
                                message="every bullet needs at least one number from its facts (kit §10.1)"))

    seen_text: dict[str, str] = {}
    for s in ss:
        k = re.sub(r"\W+", "", s.bullet.text.lower())
        if k in seen_text:
            issues.append(Issue(target=s.id, rule="duplicate", severity="hard",
                                message=f"same bullet as {seen_text[k]}"))
        seen_text.setdefault(k, s.id)

    # verbs: none more than twice; never twice in a row (kit §10.2)
    counts: dict[str, int] = {}
    prev = ""
    max_rep = kb.rules.page_budget["max_verb_repeats"]
    for s in ss:
        v = first_verb(s.bullet.text)
        counts[v] = counts.get(v, 0) + 1
        if counts[v] > max_rep:
            issues.append(Issue(target=s.id, rule="verb-repeat", severity="soft",
                                message=f"'{v}' already opens {max_rep} bullets; use a different verb"))
        elif v == prev:
            issues.append(Issue(target=s.id, rule="verb-consecutive", severity="soft",
                                message=f"previous bullet also starts with '{v}'"))
        prev = v
    return issues


# ---- LLM faithfulness judge -----------------------------------------------

class Verdict(BaseModel):
    id: str
    supported: bool = Field(description="True only if every claim in the bullet is backed by the cited facts")
    issue: str = Field("", description="If not supported: the exact unsupported or changed claim, one sentence")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


JUDGE_SYSTEM = """\
You are a strict fact-checker for a resume. For each bullet you get the claims of the facts it cites.
Mark supported=false if the bullet:
- changes a noun or unit (e.g. facts say "annotated instances", bullet says "images"; "records" vs "users"),
- claims something no cited fact states (production users, customers, team size, leadership, deployment at a company, awards),
- inflates scope or certainty (e.g. "state-of-the-art", "first", "solved"), or attributes a validation number to a leaderboard,
- swaps the project's domain.
Rewording, reordering and omitting details are FINE. Numbers have already been checked; focus on wording and claims.
Return one verdict per bullet id."""


def judge_messages(items: list[Slot]):
    from langchain_core.messages import HumanMessage, SystemMessage
    import json
    payload = [{"id": s.id, "bullet": s.bullet.text,
                "cited_fact_claims": [s.facts[f].claim for f in s.bullet.fact_ids if f in s.facts]}
               for s in items]
    return [SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=1))]


def judge_issues(state: ResumeState, kb: KnowledgeBase, call: Callable, approved: set[str],
                 skip: set[str] = frozenset()) -> tuple[list[Issue], set[str]]:
    """Judge model-written bullets not already approved (and not in `skip`). Returns (issues, newly approved)."""
    todo = [s for s in slots(state, kb)
            if s.bullet.origin == "llm" and s.bullet.text not in approved and s.id not in skip]
    if not todo:
        return [], set()
    res: Verdicts = call("judge", Verdicts, judge_messages(todo))
    by_id = {v.id: v for v in res.verdicts}
    issues, ok = [], set()
    for s in todo:
        v = by_id.get(s.id)
        if v is None:
            continue                     # judge skipped it; re-judged next round
        if v.supported:
            ok.add(s.bullet.text)
        else:
            issues.append(Issue(target=s.id, rule="unfaithful", severity="hard",
                                message=v.issue or "claim not supported by cited facts"))
    return issues, ok
