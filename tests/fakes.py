"""A deterministic stand-in for the OpenRouter models, keyed by output schema."""
from __future__ import annotations

import json

from resume_agent.state import (
    DraftBullet, ExperienceDraft, JDAnalysis, Keyword, ProjectDraft, SkillLine, SkillsDraft,
)

LLM_JD_ANALYSIS = JDAnalysis(
    archetype="llm_genai_agents",
    secondary_archetype="backend_sde",
    domain="enterprise SaaS",
    top_keywords=[
        Keyword(term="LLMs", importance="must"), Keyword(term="RAG", importance="must"),
        Keyword(term="Python", importance="must"), Keyword(term="AI agents", importance="must"),
        Keyword(term="FastAPI", importance="nice"), Keyword(term="vector databases", importance="nice"),
        Keyword(term="evaluation", importance="nice"), Keyword(term="Kubernetes", importance="nice"),
        Keyword(term="prompt engineering", importance="nice"), Keyword(term="React", importance="nice"),
    ],
    responsibilities=["build LLM features", "ship APIs"],
)

FABRICATED = "Boosted retrieval accuracy by 37% with a novel RAG pipeline used by 10,000 customers."


class _Runnable:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages):
        return self.fn(messages)


def _payload(messages) -> dict:
    text = messages[-1].content
    return json.loads(text[text.index("{"):])


def _project(messages):
    p = _payload(messages)
    bank = p["reference_bullets"]
    bullets = [DraftBullet(text=FABRICATED, fact_ids=[]),  # must be rejected
               *[DraftBullet(text=b["text"], fact_ids=b["fact_ids"]) for b in bank[:2]]]
    return ProjectDraft(title_index=0, stack=p["stack_you_may_choose_from"][:5], bullets=bullets)


def _experience(messages):
    p = _payload(messages)
    bank = p["reference_bullets"]
    return ExperienceDraft(title="Machine Learning Research Intern",
                           bullets=[DraftBullet(text=b["text"], fact_ids=b["fact_ids"]) for b in bank[:3]])


def _skills(messages):
    return SkillsDraft(
        lines=[
            SkillLine(category="LLMs & AI Agents", entries=["Large Language Models", "RAG Pipelines", "Agentic Workflows",
                      "Tool-Calling / Function-Calling", "Prompt Engineering", "LangChain", "Retrieval Benchmarking"]),
            SkillLine(category="Retrieval & Ranking", entries=["BM25", "Hybrid Search (BM25 + dense)",
                      "Vector DBs (Pinecone, ChromaDB)", "Entity Resolution", "Kubernetes Operators"]),
            SkillLine(category="Backend & APIs", entries=["FastAPI", "REST APIs", "WebSockets", "asyncio", "Pydantic"]),
            SkillLine(category="Programming", entries=["Python", "SQL", "TypeScript", "C++ (C++17)", "Bash"]),
        ],
        coursework=["Machine Learning", "Data Structures & Algorithms", "DBMS", "Underwater Basket Weaving"],
    )


def fake_factory(role: str, schema):
    table = {JDAnalysis: lambda m: LLM_JD_ANALYSIS, ProjectDraft: _project,
             ExperienceDraft: _experience, SkillsDraft: _skills}
    return _Runnable(table[schema])
