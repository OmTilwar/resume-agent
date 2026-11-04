"""Fixes from comparing the DataCore resume with a hand-written one (4 Nov)."""
from resume_agent.knowledge import Context, get_kb
from resume_agent.nodes import ensure_headline, fit_skills, pick_achievements
from resume_agent.selection import score_projects
from resume_agent.state import (ExperienceBlock, FinalBullet, JDAnalysis, JobPosting, Keyword, ProjectBlock,
                                SkillLine)
from resume_agent.validate import deterministic_issues

DATACORE = ("Interns work on product features, internal tools, and services powered by modern AI models. "
            "Demonstrable interest in AI/ML through personal projects, research, hackathons, or open-source "
            "contributions. Familiarity with LLMs and AI APIs such as OpenAI.")


def _analysis(keywords, archetype="ai_generalist", secondary=None):
    return JDAnalysis(archetype=archetype, secondary_archetype=secondary, domain="AI",
                      top_keywords=[Keyword(term=k, importance="must") for k in keywords])


def _state(keywords, selected=("finsight", "amazon_er", "authnet", "propvox"), jd=DATACORE, **kw):
    return {"analysis": _analysis(keywords, **kw), "selected": list(selected),
            "job": JobPosting(company="X", role="Y", text=jd)}


# ---- 1. achievements follow the JD's words; the PR is the default 3rd ----------------------------

def test_open_source_line_is_third_when_the_jd_asks_for_it():
    ach = pick_achievements(_state(["LLMs", "Python"]), get_kb())
    assert "GroundingDINO" in ach[2]


def test_open_source_line_is_the_default_third_achievement():
    ach = pick_achievements(_state(["LLMs", "Python"], jd="Build LLM features in Python."), get_kb())
    assert "GroundingDINO" in ach[2]                   # kit §8: default 3rd when space allows


def test_cv_in_jd_text_is_not_computer_vision():
    """'Send your CV' must not count as a computer-vision signal."""
    st = _state(["Excel"], selected=("retailforecast", "amazon_er", "medvision", "finsight"),
                jd="Send your CV to hr@example.com.", archetype="robotics_cpp")
    ach = pick_achievements(st, get_kb())
    assert "eYantra" in ach[2]                         # robotics keeps eYRC first; the CV mention didn't override


# ---- 2. a project's headline result stays on the page --------------------------------------

LOST = ("Designed a business entity-resolution pipeline matching 10M noisy multilingual records (US/India/France) "
        "to 1.7M reference entities, ranking 724th on the Amazon ML Challenge 2026 public leaderboard.")


def test_missing_headline_is_restored_in_place_of_its_twin():
    kb = get_kb()
    p = kb.get_project("amazon_er")
    bank = kb.allowed_bullets("amazon_er", Context())
    others = [FinalBullet(text=b.text, fact_ids=b.fact_ids, origin="bank") for b in bank if "0.9837" not in b.text][:2]
    out = ensure_headline([FinalBullet(text=LOST, fact_ids=[], origin="llm"), *others], p, bank, notes := [])
    assert "0.9837" in out[0].text and out[1:] == others
    assert "headline result" in notes[0]


def test_missing_headline_is_a_hard_issue():
    kb = get_kb()
    block = ProjectBlock(project_id="amazon_er", name="Amazon", subtitle="s", stack=[], github="g",
                         bullets=[FinalBullet(text=LOST, fact_ids=["amz_scale", "amz_rank"], origin="llm")])
    st = {**_state(["Python"]), "projects": [block],
          "experience": ExperienceBlock(title="Research Intern", bullets=[])}
    issues = [i for i in deterministic_issues(st, kb) if i.rule == "missing-headline"]
    assert issues and issues[0].severity == "hard" and issues[0].target == "amazon_er#1"


def test_every_headline_exists_in_a_kit_bullet():
    from resume_agent.knowledge.numbers import extract_numbers
    kb = get_kb()
    for p in kb.projects.values():
        for h in p.headline:
            assert any(h in extract_numbers(b.text) for b in p.bullets), (p.id, h)


# ---- 3. Skills ranked by the JD; wide off-topic lines give way ------------------------------

def test_wide_off_topic_line_is_dropped_and_lines_are_ranked():
    kb = get_kb()
    st = _state(["LLMs", "Python", "front-end development"])
    lines = [
        SkillLine(category="Retrieval & Ranking", entries=[
            "Dense Retrieval / Bi-Encoders (multilingual-e5, contrastive fine-tuning)",
            "Cross-Encoder Re-Ranking (bge-reranker-v2-m3, mDeBERTa-v3)", "BM25", "Hybrid Search (BM25 + dense)",
            "Learning-to-Rank (LightGBM)"]),
        SkillLine(category="LLMs & AI Agents", entries=["Large Language Models", "LLM APIs (Groq, OpenRouter)",
                                                         "LangGraph", "LangChain", "Prompt Engineering"]),
        SkillLine(category="Python & APIs", entries=["Python", "FastAPI", "REST APIs", "WebSockets", "asyncio"]),
        SkillLine(category="Front-End Development", entries=["React", "Next.js", "TypeScript", "JavaScript (ES6+)", "Vite"]),
        SkillLine(category="ML / DL / CV libraries", entries=["PyTorch", "Scikit-Learn", "LightGBM", "ONNX", "OpenCV"]),
    ]
    out, notes = fit_skills(lines, st, kb)
    cats = [l.category for l in out]
    assert "Retrieval & Ranking" not in cats
    assert cats[0] == "LLMs & AI Agents"                       # most JD-related line leads
    assert any("too wide" in n for n in notes)


def test_jd_entries_survive_width_trimming_before_project_only_ones():
    kb = get_kb()
    st = _state(["LLMs", "prompt engineering", "function calling"], selected=("finsight",))
    line = SkillLine(category="LLMs & AI Agents", entries=[
        "Llama-3 (llama.cpp, GGUF)", "Vision-Language Models (Qwen2-VL)", "LoRA Fine-Tuning (Unsloth)",
        "Large Language Models", "LLM APIs (Groq, OpenRouter)", "LangGraph", "LangChain",
        "Prompt Engineering", "Tool-Calling / Function-Calling", "RAG Pipelines"])
    pad = [SkillLine(category=c, entries=e) for c, e in [
        ("Python & APIs", ["Python", "FastAPI", "REST APIs", "WebSockets", "asyncio"]),
        ("Programming", ["Python", "SQL", "Bash", "TypeScript", "YAML"]),
        ("Front-End", ["React", "Next.js", "TypeScript", "JavaScript (ES6+)", "Vite"])]]
    out, _ = fit_skills([line, *pad], st, kb)
    llm = next(l for l in out if l.category == "LLMs & AI Agents")
    assert "Prompt Engineering" in llm.entries and "Tool-Calling / Function-Calling" in llm.entries


# ---- 4. Tier 2 enters when the JD's own words call for it --------------------------------

def test_tier2_enters_when_the_jd_calls_for_it(monkeypatch):
    """A Tier 2 project outside the JD's archetypes enters only when the JD's own words name one of its signals."""
    kb = get_kb()
    vox = kb.get_project("voxpipeline")
    assert vox.tier == 2
    monkeypatch.setattr(vox, "jd_signals", ["speaker diarization"])
    a = _analysis(["LLMs", "Python"], secondary="backend_sde")
    assert "voxpipeline" not in {s.project_id for s in score_projects(a, kb, "Build LLM features.")}
    scored = {s.project_id: s for s in score_projects(a, kb, "LLM features plus speaker diarization for calls.")}
    assert "voxpipeline" in scored
    assert any("JD asks for it: speaker diarization" in r for r in scored["voxpipeline"].reasons)


def test_resume_agent_is_tier1_with_a_headline():
    p = get_kb().get_project("resume_agent")
    assert p.tier == 1 and p.headline == ["97.9"]
