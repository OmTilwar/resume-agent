"""Cost cuts (4 Nov, from LangSmith traces): fewer and smaller paid calls, same truth guarantees."""
import json

import pytest

from resume_agent import prompts
from resume_agent.graph import build_graph
from resume_agent.knowledge import Context, get_kb
from resume_agent.loop import _fix_items, fix
from resume_agent.nodes import snap_lengths
from resume_agent.state import FinalBullet, JobPosting
from resume_agent.validate import length_issue, slots

from . import fakes

HALF_LINE = ("Improved the pipeline end to end with careful engineering, many experiments and steady iteration "
             "over several weeks of work on it")          # ~130 chars: the dead zone


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    out = tmp_path_factory.mktemp("cost")
    job = JobPosting(company="Acme AI", role="LLM Engineering Intern", text="(fake JD)")
    return build_graph().invoke({"job": job, "out_dir": str(out)},
                                config={"configurable": {"llm_factory": fakes.fake_factory}, "recursion_limit": 80})


def test_half_line_fixture_is_in_the_dead_zone():
    assert length_issue(HALF_LINE, get_kb(), "t").rule == "half-line"


# ---- 1. writer output in the dead zone is swapped for a kit bullet, no call --------------------

def test_snap_swaps_off_length_draft_for_same_fact_kit_bullet():
    kb = get_kb()
    bank = kb.allowed_bullets("amazon_er", Context())
    target = next(b for b in bank if not length_issue(b.text, kb, "x") and b.fact_ids)
    draft = FinalBullet(text=HALF_LINE, fact_ids=target.fact_ids, origin="llm")
    notes = []
    out = snap_lengths([draft], "amazon_er", bank, kb, notes)
    assert out[0].origin == "bank" and not length_issue(out[0].text, kb, "x")
    assert set(out[0].fact_ids) & set(target.fact_ids)          # same facts preferred
    assert notes and "off-length" in notes[0]


def test_snap_leaves_good_and_bank_bullets_alone():
    kb = get_kb()
    bank = kb.allowed_bullets("amazon_er", Context())
    good = FinalBullet(text="Built a thing in 105 chars or less, 3x faster.", fact_ids=[], origin="llm")
    banked = FinalBullet(text=HALF_LINE, fact_ids=[], origin="bank")
    assert snap_lengths([good, banked], "amazon_er", bank, kb, []) == [good, banked]


def test_snap_keeps_draft_when_no_kit_bullet_fits():
    kb = get_kb()
    draft = FinalBullet(text=HALF_LINE, fact_ids=[], origin="llm")
    assert snap_lengths([draft], "amazon_er", [], kb, [])[0] is draft


def test_writer_prompt_shows_reference_lengths_and_is_compact():
    kb = get_kb()
    msgs = prompts.project_messages("amazon_er", fakes.LLM_JD_ANALYSIS, Context(), kb, 3)
    text = msgs[-1].content
    payload = json.loads(text[text.index("{"):])
    assert all(b["chars"] == len(b["text"]) for b in payload["reference_bullets"])
    assert "\n " not in text[text.index("{"):]                   # no indentation whitespace
    assert "within about 10 characters" in text


# ---- 2. a length problem gets ONE fixer attempt per slot ---------------------------------------

def _with_half_line(final):
    projects = [p.model_copy(deep=True) for p in final["projects"]]
    pid = projects[0].project_id
    projects[0].bullets[0] = FinalBullet(text=HALF_LINE, fact_ids=projects[0].bullets[0].fact_ids, origin="llm")
    state = {**final, "projects": projects}
    tid = f"{pid}#1"
    state["issues"] = [length_issue(HALF_LINE, get_kb(), tid)]
    return state, tid


def _recording_factory(calls):
    def factory(role, schema):
        calls.append(role)
        return fakes.fake_factory(role, schema)
    return factory


def test_first_length_problem_goes_to_the_fixer(final):
    state, tid = _with_half_line(final)
    calls = []
    upd = fix({**state, "length_tried": []}, {"configurable": {"llm_factory": _recording_factory(calls)}})
    assert calls == ["fixer"]
    assert tid in upd["length_tried"]


def test_second_length_problem_is_swapped_without_a_call(final):
    state, tid = _with_half_line(final)
    calls = []
    upd = fix({**state, "length_tried": [tid]}, {"configurable": {"llm_factory": _recording_factory(calls)}})
    assert calls == []                                             # no paid call at all
    b = upd["projects"][0].bullets[0]
    assert b.origin == "bank" and not length_issue(b.text, get_kb(), "x")
    assert any("used a verified kit bullet" in n for n in upd["notes"])


# ---- 3. fixer payload: facts once per section ----------------------------------------------

def test_fix_payload_sends_facts_once_per_section(final):
    kb = get_kb()
    ss = [s for s in slots(final, kb) if s.section == final["projects"][0].project_id]
    payload = _fix_items({s.id: ["x"] for s in ss}, final, kb)
    assert list(payload["sections"]) == [ss[0].section]
    assert len(payload["bullets"]) == len(ss) >= 2
    assert all("facts" not in b for b in payload["bullets"])


# ---- 4. the same JD is analysed once -----------------------------------------------------------

def test_analysis_is_reused_for_the_same_jd(tmp_path):
    calls = []
    cfg = {"configurable": {"llm_factory": _recording_factory(calls), "cache_dir": str(tmp_path / "c")},
           "recursion_limit": 80}
    job = JobPosting(company="Acme AI", role="LLM Intern", text="(fake JD) build RAG")
    for i in range(2):
        build_graph().invoke({"job": job, "out_dir": str(tmp_path / f"o{i}")}, config=cfg)
    assert calls.count("analyzer") == 1
    other = JobPosting(company="Acme AI", role="LLM Intern", text="(fake JD) a different posting")
    build_graph().invoke({"job": other, "out_dir": str(tmp_path / "o3")}, config=cfg)
    assert calls.count("analyzer") == 2


def test_no_cache_unless_asked(tmp_path):
    calls = []
    cfg = {"configurable": {"llm_factory": _recording_factory(calls)}, "recursion_limit": 80}
    job = JobPosting(company="Acme AI", role="LLM Intern", text="(fake JD)")
    for i in range(2):
        build_graph().invoke({"job": job, "out_dir": str(tmp_path / f"o{i}")}, config=cfg)
    assert calls.count("analyzer") == 2
