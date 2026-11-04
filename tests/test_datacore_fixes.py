"""Fixes from the DataCore run (4 Nov): repeated metrics, missed front-end requirement, weak Skills lines."""
from resume_agent import prompts
from resume_agent.knowledge import get_kb
from resume_agent.nodes import _fill_from_bank, _relevance, bank_swap, honest_category
from resume_agent.selection import keyword_hits, mentions
from resume_agent.state import FinalBullet, JDAnalysis, JobPosting, Keyword, SkillLine
from resume_agent.validate import length_issue, repeated_numbers

# experience#1 as the model wrote it in that run
EXP1 = "Built a YOLOv8 dense object detection model for precision agriculture, improving mAP by 8.4%."


def _exp_bank(kb):
    return [b for g in kb.experience.groups for b in g.bullets]


# ---- 1. no repeated headline metric ------------------------------------------------------------

def test_repeated_numbers():
    assert repeated_numbers("raised mAP by 8.4% on 3,200+ instances", [EXP1]) == {"8.4"}
    assert repeated_numbers("deployed at under 30ms", [EXP1]) == set()


def test_swap_prefers_a_bullet_with_new_metrics():
    kb = get_kb()
    bank = _exp_bank(kb)
    fresh = [b for b in bank if not repeated_numbers(b.text, [EXP1]) and not length_issue(b.text, kb, "x")]
    assert fresh, "fixture: the experience bank has well-sized bullets without 8.4%"
    off = FinalBullet(text="x" * 116, fact_ids=["ex_dataset", "ex_map"], origin="llm")
    swap = bank_swap(off, "experience", 1, [EXP1], bank, kb)
    assert swap is not None and not repeated_numbers(swap.text, [EXP1])


def test_fill_prefers_new_metrics():
    kb = get_kb()
    bank = _exp_bank(kb)
    kept = _fill_from_bank([FinalBullet(text=EXP1, fact_ids=["ex_map"], origin="llm")], bank, 3, 3)
    texts = [b.text for b in kept]
    assert len(kept) == 3
    assert sum("8.4" in t for t in texts) <= 2          # was 3 of 3 in the live run


def test_writer_rules_forbid_repeating_a_number():
    assert "never repeat a number" in prompts.WRITING_RULES


# ---- 2. stated requirements become keywords -------------------------------------------------

def test_analyzer_prompt_covers_every_requirement():
    msgs = prompts.analyze_messages(JobPosting(company="X", role="Y", text="z"), get_kb())
    sys = msgs[0].content
    assert "EVERY distinct required skill area" in sys and "front-end" in sys
    assert "familiarity with" in sys                       # softened requirements are still 'must'
    assert "Never use generic words" in sys


def test_front_end_keyword_matches_real_frontend_work():
    kb = get_kb()
    kw = "Basic front-end development skills"
    assert keyword_hits(kw, kb.get_project("finsight"))     # React (Vite) frontend
    assert keyword_hits(kw, kb.get_project("propvox"))      # Next.js/React
    assert mentions(kw, "React") and mentions(kw, "JavaScript (ES6+)")
    assert not keyword_hits(kw, kb.get_project("amazon_er"))


# ---- 3. Skills: real LLM tooling kept, names backed by entries ------------------------------

def _state(keywords):
    a = JDAnalysis(archetype="ai_generalist", domain="AI", top_keywords=[Keyword(term=k, importance="must") for k in keywords])
    return {"analysis": a, "selected": ["amazon_er", "medvision"]}   # projects without LangGraph in their stack


def test_llm_frameworks_count_as_relevant_to_an_llm_jd():
    relevant = _relevance(_state(["LLMs", "AI APIs such as OpenAI"]), get_kb())
    for e in ["LangGraph", "LangChain", "LLM APIs (Groq, OpenRouter)", "LangSmith (tracing)"]:
        assert relevant(e), e


def test_category_name_must_be_backed_by_entries():
    kb = get_kb()
    weak = SkillLine(category="AI APIs & Model Training", entries=["REST APIs", "WebSockets", "asyncio", "Pydantic"])
    assert honest_category(weak, kb) == "Backend & APIs"
    single = SkillLine(category="Python", entries=["Python", "TypeScript", "JavaScript (ES6+)", "HTML/CSS"])
    assert honest_category(single, kb) == "Programming"
    for good in [SkillLine(category="LLMs & AI Agents", entries=["Large Language Models", "LangGraph", "Prompt Engineering"]),
                 SkillLine(category="Retrieval & Ranking", entries=["BM25", "Hybrid Search (BM25 + dense)"]),
                 SkillLine(category="Python & SQL", entries=["Python", "SQL", "Window Functions"])]:
        assert honest_category(good, kb) == good.category


# ---- round 2 (DataCore v2 run) ----------------------------------------------------------------

from resume_agent.nodes import _ensure_required, required_skills, snap_weak   # noqa: E402
from resume_agent.validate import weak_reason                                 # noqa: E402

FS1 = ("Built a privacy-first banking assistant on a local Llama-3-8B (GGUF, llama-cpp-python) with a 4-stage pipeline: "
       "BM25 lexical search → Pinecone vector search → deterministic function-calling → grounded generation.")
FS3 = "Implemented deterministic function-calling for account queries via a 4-stage pipeline."   # repeats bullet 1
PV3 = "Added bilingual English + Hindi prompting with sentence flushing so speech starts before the LLM finishes."


def test_required_inventory_skill_is_always_in_skills():
    kb = get_kb()
    st = _state(["Proficiency in Python", "LLMs"])
    assert "Python" in required_skills(st, kb)
    lines = [SkillLine(category="LLMs & AI Agents", entries=["Large Language Models", "LangGraph", "LangChain",
                                                              "LLM APIs (Groq, OpenRouter)", "Prompt Engineering"]),
             SkillLine(category="Backend & APIs", entries=["FastAPI", "REST APIs", "WebSockets", "asyncio", "Pydantic"])]
    notes = []
    out = _ensure_required(lines, st, kb, notes)
    assert sum("Python" in l.entries for l in out) == 1
    assert any("added required 'Python'" in n for n in notes)


def test_required_skill_not_duplicated_when_present():
    kb = get_kb()
    lines = [SkillLine(category="Programming", entries=["Python", "SQL", "Bash", "TypeScript", "YAML"])]
    out = _ensure_required(lines, _state(["Python"]), kb, [])
    assert out[0].entries.count("Python") == 1


def test_weak_reasons():
    kb = get_kb()
    assert "repeats 4" in weak_reason(FS3, [FS1], kb, "finsight")
    assert weak_reason(PV3, [], kb, "propvox") == "no number"
    assert weak_reason(PV3, [], kb, "vio") is None                 # VIO bullets may have no number (kit)
    assert weak_reason("Cut CPU inference from 14.78ms to 8ms.", [FS1], kb, "authnet") is None


def test_snap_swaps_number_repeat_and_numberless_drafts():
    kb = get_kb()
    from resume_agent.knowledge import Context
    bank = [b for b in kb.allowed_bullets("finsight", Context()) if not kb.cites_last_only("finsight", b.fact_ids)]
    out = snap_weak([FinalBullet(text=FS1, fact_ids=["fs_pipeline"], origin="llm"),
                     FinalBullet(text=FS3, fact_ids=[], origin="llm")], "finsight", bank, kb, notes := [])
    assert out[0].text == FS1                                        # the first one keeps its number
    assert out[1].origin == "bank" and not repeated_numbers(out[1].text, [FS1])
    assert "repeats 4" in notes[0]


def test_concepts_match_the_architectures_on_the_page():
    assert mentions("neural networks", "Fine-tuned ResNet-18 on 5,800+ chest X-rays")
    assert mentions("neural networks", "Trained a U-Net lung segmenter from scratch")
    assert mentions("embeddings", "producing 128-dim texture fingerprints")
    assert mentions("embeddings", "Fine-tuned a multilingual-e5-base bi-encoder on 3M pairs")


def test_synonym_table_has_no_duplicate_keys():
    """A repeated key silently overrides the earlier one (it hid the new 'embeddings' synonyms once)."""
    import ast
    import collections
    from pathlib import Path
    import resume_agent.selection as sel
    tree = ast.parse(Path(sel.__file__).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "SYNONYMS")
    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    assert not [k for k, c in collections.Counter(keys).items() if c > 1]


def test_required_concept_goes_to_a_related_line_not_the_roomiest():
    """DataCore v3: 'Embeddings' landed first on 'Python & APIs' because no ML-tasks line was on the page."""
    kb = get_kb()
    lines = [SkillLine(category="Python & APIs", entries=["Python", "FastAPI", "REST APIs", "WebSockets"]),
             SkillLine(category="Neural Networks & Deep Learning",
                       entries=["PyTorch", "Scikit-Learn", "sentence-transformers", "HuggingFace Transformers",
                                "ONNX", "ONNX Runtime", "LightGBM"])]
    out = _ensure_required(lines, _state(["embeddings"]), kb, [])
    assert "Embeddings" not in out[0].entries
    assert out[1].entries[1] == "Embeddings"                  # after the line's lead entry
