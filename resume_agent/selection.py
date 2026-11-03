"""Kit §3 step 2 — deterministic project scoring and selection.

score = +3 per must-have keyword the project honestly demonstrates (tags / stack)
      + 1 per nice-to-have
      + 2 if the project is in the primary archetype's list (+1 secondary)
      + 2 if Tier 1
Tier 2 projects are eligible only when an archetype (primary/secondary) lists them.
"""
from __future__ import annotations

import re

from .knowledge import KnowledgeBase, Project
from .state import JDAnalysis, ProjectScore

# JD phrasing -> KB tag vocabulary. Extend as new JDs show misses.
SYNONYMS: dict[str, list[str]] = {
    "large language model": ["llm"], "large language models": ["llm"], "llms": ["llm"],
    "generative ai": ["genai", "generative-ai"], "gen ai": ["genai"], "genai": ["genai"],
    "retrieval augmented generation": ["rag"], "retrieval-augmented generation": ["rag"],
    "computer vision": ["cv", "computer-vision"], "machine learning": ["ml"],
    "deep learning": ["ml", "pytorch"], "natural language processing": ["nlp"],
    "vector database": ["vector-search"], "vector databases": ["vector-search"], "vector db": ["vector-search"],
    "semantic search": ["retrieval", "dense-retrieval"], "search": ["search", "retrieval"],
    "recommendation": ["ranking"], "recommender": ["ranking"],
    "agentic": ["agents"], "ai agents": ["agents"], "agent": ["agents"],
    "function calling": ["function-calling", "tool-calling"], "tool use": ["tool-use", "tool-calling"],
    "rest api": ["rest-api", "fastapi"], "rest apis": ["rest-api", "fastapi"], "apis": ["rest-api", "fastapi"],
    "backend": ["fastapi", "rest-api"], "microservices": ["fastapi", "production-api"],
    "speech": ["speech", "voice"], "asr": ["stt"], "text to speech": ["tts"],
    "fine-tuning": ["fine-tuning", "lora"], "fine tuning": ["fine-tuning", "lora"],
    "embeddings": ["embeddings"], "transformers": ["transformers", "cross-encoder", "bi-encoder"],
    "statistics": ["statistics", "hypothesis-testing"], "a/b testing": ["hypothesis-testing"],
    "etl": ["etl", "data-pipelines"], "data pipelines": ["data-pipelines", "etl"],
    "cloud": ["aws", "cloud-gpu"], "aws": ["aws", "s3", "sagemaker"],
    "medical": ["healthcare", "medical-imaging"], "healthcare": ["healthcare"],
    "image": ["cv"], "object detection": ["cv"], "ocr": ["ocr"],
    "c++": ["cpp", "cpp17"], "ros": ["ros2"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_/\-]", " ", s.lower())).strip()


def _vocab(p: Project) -> set[str]:
    return {_norm(t) for t in p.tags} | {_norm(s) for s in p.stack}


def _probes(keyword: str) -> set[str]:
    kw = _norm(keyword)
    out = {kw} | {_norm(x) for k, xs in SYNONYMS.items() if re.search(rf"\b{re.escape(k)}\b", keyword.lower()) for x in xs}
    out |= {p[:-1] for p in out if p.endswith("s") and len(p) > 4}   # LLMs -> LLM
    inner = re.findall(r"\(([^)]+)\)", keyword)                       # "Retrieval-Augmented Generation (RAG)"
    out |= {_norm(x) for x in inner}
    return {p for p in out if p}


def mentions(keyword: str, text: str) -> bool:
    """Does `text` (a bullet / skills line) cover this JD keyword?"""
    t = _norm(text)
    return any(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", t) for p in _probes(keyword))


def keyword_hits(keyword: str, project: Project) -> bool:
    probes = _probes(keyword)
    vocab = _vocab(project)
    for probe in probes:
        for term in vocab:
            if not term:
                continue
            if term == probe or re.search(rf"\b{re.escape(term)}\b", probe) or (
                len(probe) > 3 and re.search(rf"\b{re.escape(probe)}\b", term)
            ):
                return True
    return False


def score_projects(analysis: JDAnalysis, kb: KnowledgeBase) -> list[ProjectScore]:
    primary = kb.archetypes.get(analysis.archetype)
    secondary = kb.archetypes.get(analysis.secondary_archetype or "")
    prim_list = primary.flat_projects() if primary else []
    sec_list = secondary.flat_projects() if secondary else []

    scores: list[ProjectScore] = []
    for p in kb.selectable_projects():
        if p.tier == 2 and p.id not in prim_list + sec_list:
            continue
        must = [k.term for k in analysis.top_keywords if k.importance == "must" and keyword_hits(k.term, p)]
        nice = [k.term for k in analysis.top_keywords if k.importance == "nice" and keyword_hits(k.term, p)]
        s, reasons = 3 * len(must) + len(nice), []
        if must:
            reasons.append(f"+{3 * len(must)} must-have: {', '.join(must)}")
        if nice:
            reasons.append(f"+{len(nice)} nice-to-have: {', '.join(nice)}")
        if p.id in prim_list:
            s += 2
            reasons.append(f"+2 fits archetype {primary.id}")
        elif p.id in sec_list:
            s += 1
            reasons.append(f"+1 fits secondary archetype {secondary.id}")
        if p.tier == 1:
            s += 2
            reasons.append("+2 Tier 1")
        scores.append(ProjectScore(project_id=p.id, score=s, must_hits=must, nice_hits=nice, reasons=reasons))

    order = {pid: i for i, pid in enumerate(prim_list + sec_list)}
    scores.sort(key=lambda x: (-x.score, order.get(x.project_id, 99)))
    return scores


def select_projects(scores: list[ProjectScore], kb: KnowledgeBase, k: int = 4) -> list[str]:
    chosen: list[str] = []
    for s in scores:
        if s.project_id == "vio" and "submarine" not in [x.project_id for x in scores[: k + 1]]:
            continue  # kit §7.7: VIO only alongside the submarine framework
        chosen.append(s.project_id)
        if len(chosen) == k:
            break
    return chosen
