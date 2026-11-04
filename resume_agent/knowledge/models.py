"""Typed schema for the resume knowledge base (kb/*.yaml).

Mirrors MASTER_RESUME_KIT.md v2. The kit is the human-readable source; these
models make its rules machine-checkable (fact status, context restrictions,
retired projects, Familiar-only skills).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FactStatus(str, Enum):
    VERIFIED = "verified"  # ✅ saved artifact in the repo
    STATED = "stated"      # ⚠ README / candidate-stated
    BANNED = "banned"      # ⛔ never use, in any wording


class Context(BaseModel):
    """What we know about the target job when deciding which facts are usable."""
    archetype: Optional[str] = None
    company: Optional[str] = None


class AllowedWhen(BaseModel):
    """A fact is usable only if ANY listed condition matches the context."""
    archetypes: list[str] = Field(default_factory=list)
    company_contains: list[str] = Field(default_factory=list)

    def matches(self, ctx: Context) -> bool:
        if ctx.archetype and ctx.archetype in self.archetypes:
            return True
        company = (ctx.company or "").lower()
        return any(c.lower() in company for c in self.company_contains)


class Fact(BaseModel):
    id: str
    claim: str
    values: list[str] = Field(default_factory=list)  # numbers a bullet may quote
    status: FactStatus
    eval_size: Optional[int] = None    # items in the eval set behind a metric
    allowed_when: Optional[AllowedWhen] = None
    source: Optional[str] = None       # artifact path / commit
    note: Optional[str] = None

    @property
    def usable_in_any_context(self) -> bool:
        return self.status != FactStatus.BANNED


class Bullet(BaseModel):
    id: str
    text: str
    fact_ids: list[str]
    note: Optional[str] = None


class Title(BaseModel):
    subtitle: str
    stack: list[str]


class InterviewQA(BaseModel):
    q: str
    a: Optional[str] = None


class Project(BaseModel):
    id: str
    name: str
    tier: int                     # 1 default pool, 2 role-specific
    section: str                  # kit section, e.g. "7.1"
    github: str
    stack: list[str]
    one_liner: str
    context: Optional[str] = None   # kit's Problem/Pipeline prose — evidence for the judge
    tags: list[str]
    rules: list[str] = Field(default_factory=list)
    forbidden_title_terms: list[str] = Field(default_factory=list)
    banned_terms: list[str] = Field(default_factory=list)   # never in this project's bullets
    allowed_terms: list[str] = Field(default_factory=list)  # globally banned terms this project may use
    banned_verbs: list[str] = Field(default_factory=list)
    max_bullets: Optional[int] = None
    facts: list[Fact]
    titles: list[Title]
    bullets: list[Bullet]
    interview: list[InterviewQA] = Field(default_factory=list)

    @field_validator("tier")
    @classmethod
    def _tier(cls, v: int) -> int:
        if v not in (1, 2):
            raise ValueError("tier must be 1 or 2")
        return v

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(f"{self.id}: unknown fact {fact_id!r}")


class RetiredProject(BaseModel):
    id: str
    name: str
    reason: str
    revive_when: str


class ExperienceGroup(BaseModel):
    id: str  # "A".."E"
    name: str
    bullets: list[Bullet]


class Experience(BaseModel):
    titles: list[str]
    org: str
    location: str
    dates: str
    facts: list[Fact]
    groups: list[ExperienceGroup]
    unconfirmed_claims: list[str] = Field(default_factory=list)

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(f"experience: unknown fact {fact_id!r}")


class Skill(BaseModel):
    name: str
    category: str
    familiar: bool = False

    @property
    def display(self) -> str:
        return f"{self.name} (Familiar)" if self.familiar else self.name


class Education(BaseModel):
    institution: str
    degree: str
    dates: str
    cgpa: str


class Profile(BaseModel):
    name: str
    phone: str
    email: str
    linkedin: str
    github: str
    education: Education
    targeting: str
    positioning: list[str]
    coursework: list[str]
    coursework_shorthand: list[str]
    coursework_aliases: dict[str, str] = Field(default_factory=dict)  # shorthand -> official
    taglines: list[str] = Field(default_factory=list)

    def course_key(self, name: str) -> str:
        """Same key for a course and its shorthand (dedupe)."""
        return self.coursework_aliases.get(name, name).lower()


class Achievement(BaseModel):
    id: str
    kind: str  # publication | hackathon | competition | open_source
    text: str
    latex: Optional[str] = None
    facts: list[Fact] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    include_for: list[str] = Field(default_factory=list)
    drop_if_project: Optional[str] = None
    always_include: bool = False


class Archetype(BaseModel):
    id: str
    name: str
    signals: list[str]
    emphasize: str
    # Ordered; an inner list means "one of these" (e.g. MedVision or AuthNet).
    projects: list[str | list[str]]
    exp_groups: list[str]
    note: Optional[str] = None

    def flat_projects(self) -> list[str]:
        out: list[str] = []
        for p in self.projects:
            out.extend(p if isinstance(p, list) else [p])
        return out


class Rules(BaseModel):
    banned_phrases: list[str]          # anywhere on the page (case-insensitive)
    phrase_exceptions: list[str] = Field(default_factory=list)  # legit uses, e.g. "novel face generation"
    banned_bullet_terms: list[str]     # never in a bullet (may appear in Skills)
    analogy_patterns: list[str]        # regexes
    lead_verbs_solo_only: list[str]
    verbs: dict[str, list[str]]
    min_eval_size: int = 50
    institution_guard: list[str] = Field(default_factory=list)  # regexes
    page_budget: dict[str, int | float] = Field(default_factory=dict)


class KnowledgeBase(BaseModel):
    kit_version: str
    kit_updated: str
    profile: Profile
    experience: Experience
    skills: list[Skill]
    projects: dict[str, Project]
    retired: list[RetiredProject]
    achievements: list[Achievement]
    archetypes: dict[str, Archetype]
    rules: Rules

    # ---- query helpers (used by graph nodes later) -------------------------

    def selectable_projects(self, tiers: tuple[int, ...] = (1, 2)) -> list[Project]:
        return [p for p in self.projects.values() if p.tier in tiers]

    def get_project(self, project_id: str) -> Project:
        if any(r.id == project_id for r in self.retired):
            raise ValueError(f"{project_id!r} is a retired project (Appendix F) and can never be used")
        try:
            return self.projects[project_id]
        except KeyError:
            raise KeyError(f"unknown project {project_id!r}") from None

    @staticmethod
    def fact_allowed(fact: Fact, ctx: Context) -> bool:
        if fact.status == FactStatus.BANNED:
            return False
        if fact.allowed_when is not None and not fact.allowed_when.matches(ctx):
            return False
        return True

    def allowed_facts(self, project_id: str, ctx: Context) -> list[Fact]:
        return [f for f in self.get_project(project_id).facts if self.fact_allowed(f, ctx)]

    def allowed_bullets(self, project_id: str, ctx: Context) -> list[Bullet]:
        p = self.get_project(project_id)
        ok = {f.id for f in self.allowed_facts(project_id, ctx)}
        return [b for b in p.bullets if set(b.fact_ids) <= ok]

    def banned_facts(self, project_id: str | None = None) -> list[Fact]:
        projects = [self.get_project(project_id)] if project_id else list(self.projects.values())
        return [f for p in projects for f in p.facts if f.status == FactStatus.BANNED]

    def familiar_skills(self) -> list[Skill]:
        return [s for s in self.skills if s.familiar]

    def skills_by_category(self) -> dict[str, list[Skill]]:
        out: dict[str, list[Skill]] = {}
        for s in self.skills:
            out.setdefault(s.category, []).append(s)
        return out
