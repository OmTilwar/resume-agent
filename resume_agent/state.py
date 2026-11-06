"""Graph state + the structured schemas LLM nodes must return."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---- inputs ---------------------------------------------------------------

class JobPosting(BaseModel):
    company: str
    role: str
    text: str
    url: Optional[str] = None
    source: str = "manual"          # manual | pdf | (later) apify:linkedin, greenhouse, ...
    location: Optional[str] = None


# ---- LLM outputs ----------------------------------------------------------

class Keyword(BaseModel):
    term: str = Field(description="Exact wording from the JD")
    importance: Literal["must", "nice"]


class JDAnalysis(BaseModel):
    archetype: str = Field(description="Best-matching archetype id from the provided list")
    secondary_archetype: Optional[str] = Field(None, description="Second archetype id if the JD mixes types")
    top_keywords: list[Keyword] = Field(description="Top 10 keywords/skills, most important first, exact JD wording")
    domain: str = Field(description="Company's business domain in 1-3 plain words, e.g. 'fintech', 'healthcare', 'e-commerce' — not the role")
    responsibilities: list[str] = Field(default_factory=list, description="3-6 short phrases")


class DraftBullet(BaseModel):
    text: str = Field(description="Plain text bullet (no LaTeX, no markdown)")
    fact_ids: list[str] = Field(description="IDs of every fact this bullet's claims and numbers come from")


class ProjectDraft(BaseModel):
    title_index: int = Field(description="Index into the provided title options")
    stack: list[str] = Field(description="4-5 technologies from the project's stack, JD-matching first")
    bullets: list[DraftBullet]


class ExperienceDraft(BaseModel):
    title: str = Field(description="Exactly one of the allowed titles")
    bullets: list[DraftBullet]


class SkillLine(BaseModel):
    category: str = Field(description="Category name mirroring the JD, e.g. 'LLMs & AI Agents'")
    entries: list[str] = Field(description="8-14 items copied exactly from the inventory")


class SkillsDraft(BaseModel):
    lines: list[SkillLine]
    coursework: list[str] = Field(description="4-5 course names copied exactly from the list")


class FixedBullet(BaseModel):
    id: str
    text: str = Field(description="The repaired bullet, plain text")
    fact_ids: list[str]


class FixBatch(BaseModel):
    fixes: list[FixedBullet]


# ---- assembled pieces -----------------------------------------------------

class FinalBullet(BaseModel):
    text: str
    fact_ids: list[str]
    origin: Literal["llm", "bank"]


class ProjectBlock(BaseModel):
    project_id: str
    name: str
    subtitle: str          # work entries: the job title
    stack: list[str]
    github: str
    bullets: list[FinalBullet]
    kind: str = "project"  # "work" = an Experience entry (kit §5A): rendered under Experience, no GitHub icon
    org: str = ""
    location: str = ""
    dates: str = ""


class ExperienceBlock(BaseModel):
    title: str
    bullets: list[FinalBullet]


class ProjectScore(BaseModel):
    project_id: str
    score: int
    must_hits: list[str]
    nice_hits: list[str]
    reasons: list[str]


class Report(BaseModel):
    keyword_coverage: dict[str, list[str]]
    honest_gaps: list[str]
    interview_questions: list[str]
    replaced_bullets: list[str]
    pages: Optional[int]
    tex_path: str
    pdf_path: Optional[str]


class ResumeState(TypedDict, total=False):
    job: JobPosting
    slug: str
    out_dir: str
    analysis: JDAnalysis
    scores: list[ProjectScore]
    selected: list[str]
    exp_groups: list[str]
    projects: list[ProjectBlock]
    experience: ExperienceBlock
    skills: list[SkillLine]
    coursework: list[str]
    achievements: list[str]
    violations: Annotated[list[str], operator.add]   # parallel writers append; never return the old list
    # Phase 2
    issues: list            # list[validate.Issue] still open
    approved: list[str]     # bullet texts the judge accepted (not re-judged)
    fix_round: int
    work_ids: list[str]     # work entries on the page (kit §5A), in page order
    work_groups: list[str]  # their bullet groups that lead for this JD (kit §9 'G' column)
    length_tried: list[str] # bullet ids (slots) the fixer already got one length attempt at
    layout: object         # layout.Layout from the last compile
    layout_round: int
    layout_action: str      # trim | orphans | fill | done
    trim_level: int
    max_achievements: int
    no_fill: bool
    last_fill: Optional[str]
    orphan_tried: list[str]
    notes: list[str]
    tex: str
    tex_path: str
    pdf_path: Optional[str]
    pages: Optional[int]
    report: Report
