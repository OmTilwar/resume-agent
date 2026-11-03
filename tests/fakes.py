"""A deterministic stand-in for the OpenRouter models, keyed by output schema.

Scenarios it exercises:
- every project draft opens with a FABRICATED bullet; the fake fixer can't repair it,
  so after MAX_FIX rounds the fallback must swap in a bank bullet;
- the experience draft says "images" where the fact says "instances"; the fake judge
  flags it and the fake fixer repairs it (images -> instances), so it ships model-written.
"""
from __future__ import annotations

import json

from resume_agent.state import (
    DraftBullet, ExperienceDraft, FixBatch, FixedBullet, JDAnalysis, Keyword, ProjectDraft, SkillLine,
    SkillsDraft,
)
from resume_agent.validate import Verdict, Verdicts

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
DRIFTED = "Trained YOLOv8 on 3,200+ annotated images, improving mAP by 8.4% on aerial and field imagery."

CALLS: list[str] = []   # role log, for assertions


class _Runnable:
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, messages):
        return self.fn(messages)


def _payload(messages):
    text = messages[-1].content
    start = min(i for i in (text.find("{"), text.find("[")) if i >= 0)
    return json.loads(text[start:])


def _project(messages):
    p = _payload(messages)
    bank = p["reference_bullets"]
    bullets = [DraftBullet(text=FABRICATED, fact_ids=[]),
               *[DraftBullet(text=b["text"], fact_ids=b["fact_ids"]) for b in bank[:2]]]
    return ProjectDraft(title_index=0, stack=p["stack_you_may_choose_from"][:5], bullets=bullets)


def _experience(messages):
    p = _payload(messages)
    bank = [b for b in p["reference_bullets"]]
    return ExperienceDraft(title="Machine Learning Research Intern", bullets=[
        DraftBullet(text=DRIFTED, fact_ids=["ex_dataset", "ex_map"]),
        *[DraftBullet(text=b["text"], fact_ids=b["fact_ids"]) for b in bank[1:3]],
    ])


def _skills(messages):
    return SkillsDraft(
        lines=[
            SkillLine(category="LLMs & AI Agents", entries=["Large Language Models", "RAG Pipelines", "Agentic Workflows",
                      "Tool-Calling / Function-Calling", "Prompt Engineering", "LangChain", "Retrieval Benchmarking"]),
            SkillLine(category="Retrieval & Ranking", entries=["BM25", "Hybrid Search (BM25 + dense)",
                      "Vector DBs (Pinecone, ChromaDB)", "Entity Resolution", "Kubernetes Operators"]),
            SkillLine(category="Backend & APIs", entries=["FastAPI", "REST APIs", "WebSockets", "asyncio", "Pydantic",
                      "Large Language Models"]),   # duplicate of line 1 → must be dropped
            SkillLine(category="Programming", entries=["Python", "SQL", "TypeScript", "C++ (C++17)", "Bash"]),
        ],
        coursework=["Machine Learning", "Fundamentals of Machine Learning", "DBMS", "Underwater Basket Weaving"],
    )


def _judge(messages):
    items = _payload(messages)
    return Verdicts(verdicts=[
        Verdict(id=i["id"], supported="images" not in i["bullet"],
                issue="facts say 'annotated instances', bullet says 'images'" if "images" in i["bullet"] else "")
        for i in items])


def _fixer(messages):
    items = _payload(messages)
    fixes = []
    for i in items:
        if "images" in i["text"]:
            fixes.append(FixedBullet(id=i["id"], text=i["text"].replace("annotated images", "annotated instances"),
                                     fact_ids=["ex_dataset", "ex_map"]))
        else:   # a weak fixer: returns the bullet unchanged
            fixes.append(FixedBullet(id=i["id"], text=i["text"], fact_ids=[]))
    return FixBatch(fixes=fixes)


def fake_factory(role: str, schema):
    CALLS.append(role)
    table = {JDAnalysis: lambda m: LLM_JD_ANALYSIS, ProjectDraft: _project, ExperienceDraft: _experience,
             SkillsDraft: _skills, Verdicts: _judge, FixBatch: _fixer}
    return _Runnable(table[schema])
