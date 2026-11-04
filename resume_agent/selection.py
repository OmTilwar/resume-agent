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
_LLM = ["llm", "langchain", "langgraph", "langsmith", "llm apis", "rag", "prompt engineering", "tool calling", "agents"]
SYNONYMS: dict[str, list[str]] = {
    # LLM work shows up as these on the page (LangGraph, LLM APIs, …) — a JD asking for "LLMs" is asking for them
    **{k: _LLM for k in ("large language model", "large language models", "llms", "llm", "language models")},
    "ai apis": ["llm apis", "openai", "openrouter", "groq", "function calling", "tool calling"],
    "openai": ["llm apis", "openrouter", "groq"],
    "front-end": ["frontend", "react", "next.js", "javascript", "typescript", "html/css", "vite", "full-stack"],
    "front end": ["frontend", "react", "next.js", "javascript", "typescript", "html/css", "vite", "full-stack"],
    "frontend": ["react", "next.js", "javascript", "typescript", "html/css", "vite", "full-stack"],
    "competitive programming": ["data structures & algorithms"],
    # concepts the page shows by name of the architecture, not the concept (DataCore run, 4 Nov)
    "neural network": ["deep learning", "pytorch", "resnet", "u-net", "cnn", "transformers", "metric learning"],
    "neural networks": ["deep learning", "pytorch", "resnet", "u-net", "cnn", "transformers", "metric learning"],
    "embeddings": ["embeddings", "bi-encoder", "sentence-transformers", "dense retrieval", "metric learning",
                   "fingerprints"],
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
    "transformers": ["transformers", "cross-encoder", "bi-encoder"],
    "statistics": ["statistics", "hypothesis-testing", "t-test", "mann-whitney", "p <"],
    "statistical": ["statistics", "hypothesis-testing", "t-test", "mann-whitney"],
    "a/b testing": ["hypothesis-testing"],
    "etl": ["etl", "data-pipelines"], "data pipelines": ["data-pipelines", "etl"],
    "cloud": ["aws", "cloud-gpu"], "aws": ["aws", "s3", "sagemaker"],
    "medical": ["healthcare", "medical-imaging"], "healthcare": ["healthcare"],
    "image": ["cv"], "object detection": ["cv"], "ocr": ["ocr"],
    "c++": ["cpp", "cpp17"], "ros": ["ros2"],
    # learning paradigms: what on the page actually demonstrates them
    "supervised": ["classification", "transfer learning", "learning to rank", "lightgbm", "segmentation"],
    "unsupervised": ["t-sne", "vae", "clustering", "self-training", "representation learning"],
    "data science": ["data-science", "hypothesis testing", "forecasting", "statistics"],
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
    # alternatives: "TensorFlow/PyTorch", "AWS, GCP or Azure" -> each part counts on its own
    parts = re.split(r"\s*(?:/|,|or|and)\s*", re.sub(r"\([^)]*\)", "", keyword))
    if len(parts) > 1:
        for part in parts:
            if part.strip():
                out |= _probes(part)
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


ORDER_BONUS = [3, 2, 1]   # extra for the 1st / 2nd / 3rd project in the archetype's kit §9 list
FLAGSHIP = "amazon_er"    # kit §9: "belongs on almost every ML/AI resume"


def jd_signal_hits(p: Project, jd_text: str) -> list[str]:
    t = _norm(jd_text)
    return [sig for sig in p.jd_signals if re.search(rf"(?<![a-z0-9]){re.escape(_norm(sig))}(?![a-z0-9])", t)]


def score_projects(analysis: JDAnalysis, kb: KnowledgeBase, jd_text: str = "") -> list[ProjectScore]:
    primary = kb.archetypes.get(analysis.archetype)
    secondary = kb.archetypes.get(analysis.secondary_archetype or "")
    prim_list = primary.flat_projects() if primary else []
    sec_list = secondary.flat_projects() if secondary else []
    # slot = position in the archetype's list; "MedVision or AuthNet" share a slot
    slot = {}
    for i, entry in enumerate(primary.projects if primary else []):
        for pid in (entry if isinstance(entry, list) else [entry]):
            slot[pid] = i

    scores: list[ProjectScore] = []
    for p in kb.selectable_projects():
        signals = jd_signal_hits(p, jd_text) if p.tier == 2 else []
        if p.tier == 2 and p.id not in prim_list + sec_list and not signals:
            continue   # kit §3: Tier 2 only when the archetype — or the JD's own words — call for it
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
            bonus = ORDER_BONUS[slot[p.id]] if slot[p.id] < len(ORDER_BONUS) else 0
            if bonus:   # kit §9 lists projects in priority order for each JD type
                s += bonus
                reasons.append(f"+{bonus} position {slot[p.id] + 1} in kit §9 for {primary.id}")
        elif p.id in sec_list:
            s += 1
            reasons.append(f"+1 fits secondary archetype {secondary.id}")
        if p.tier == 1:
            s += 2
            reasons.append("+2 Tier 1")
        if signals:
            s += 2
            reasons.append(f"+2 JD asks for it: {', '.join(signals)}")
        scores.append(ProjectScore(project_id=p.id, score=s, must_hits=must, nice_hits=nice, reasons=reasons))

    order = {pid: i for i, pid in enumerate(prim_list + sec_list)}
    scores.sort(key=lambda x: (-x.score, order.get(x.project_id, 99)))
    return scores


def select_projects(scores: list[ProjectScore], kb: KnowledgeBase, k: int = 4,
                    analysis: JDAnalysis | None = None) -> list[str]:
    chosen: list[str] = []
    for s in scores:
        if s.project_id == "vio" and "submarine" not in [x.project_id for x in scores[: k + 1]]:
            continue  # kit §7.7: VIO only alongside the submarine framework
        chosen.append(s.project_id)
        if len(chosen) == k:
            break
    # kit §9: the flagship stays whenever the JD type lists it — keyword wording varies between
    # runs and must not knock it off (seen live, 4 Nov: "EDA"/"survey data" instead of "AWS"/"git").
    arch = kb.archetypes.get(analysis.archetype) if analysis else None
    if arch and FLAGSHIP in arch.flat_projects() and FLAGSHIP not in chosen             and any(s.project_id == FLAGSHIP for s in scores) and len(chosen) == k:
        chosen[-1] = FLAGSHIP
        order = {s.project_id: i for i, s in enumerate(scores)}
        chosen.sort(key=lambda pid: order[pid])
    return chosen
