"""Phase 2: content loop (validate ⇄ fix → fallback) and layout loop, offline."""
import shutil

import pytest

from resume_agent.graph import build_graph
from resume_agent.latex import escape_and_highlight
from resume_agent.layout import Layout
from resume_agent.loop import MAX_FIX, layout_fix
from resume_agent.state import JobPosting
from resume_agent.validate import deterministic_issues, length_issue

from . import fakes


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    fakes.CALLS.clear()
    out = tmp_path_factory.mktemp("p2")
    job = JobPosting(company="Acme AI", role="LLM Engineering Intern", text="(fake JD)")
    return build_graph().invoke({"job": job, "out_dir": str(out)},
                                config={"configurable": {"llm_factory": fakes.fake_factory},
                                        "recursion_limit": 80})


def test_judge_catches_wording_drift_and_fixer_repairs_it(final):
    texts = [b.text for b in final["experience"].bullets]
    assert fakes.DRIFTED not in texts
    fixed = fakes.DRIFTED.replace("annotated images", "annotated instances")
    assert fixed in texts
    assert next(b for b in final["experience"].bullets if b.text == fixed).origin == "llm"


def test_unrepairable_bullets_fall_back_to_bank(final):
    for p in final["projects"]:
        assert all(b.text != fakes.FABRICATED for b in p.bullets)
        assert any(b.origin == "bank" for b in p.bullets)
    assert any("replaced with verified bank bullet" in n for n in final["notes"])
    assert final["fix_round"] >= MAX_FIX


def test_no_hard_issue_ships(final, kb):
    hard = [i for i in deterministic_issues(final, kb) if i.severity == "hard"]
    assert not hard, hard


def test_judge_and_fixer_were_called(final):
    assert "judge" in fakes.CALLS and "fixer" in fakes.CALLS


def test_skills_deduped_across_lines(final):
    entries = [e for l in final["skills"] for e in l.entries]
    assert entries.count("Large Language Models") == 1


def test_coursework_alias_deduped(final):
    cw = final["coursework"]
    assert not ({"Machine Learning", "Fundamentals of Machine Learning"} <= set(cw))


def test_report_has_quality_section(final, tmp_path_factory):
    from pathlib import Path
    md = Path(final["out_dir"], "report.md").read_text(encoding="utf-8")
    assert "## Quality checks" in md and "## Still open" in md


@pytest.mark.latex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_layout_measured_and_one_page(final):
    assert final["layout"] is not None
    assert final["layout"].pages == 1


# ---- unit-level ---------------------------------------------------------------

def test_length_rules(kb):
    assert length_issue("x" * 100, kb, "t") is None
    assert length_issue("x" * 200, kb, "t") is None
    assert length_issue("x" * 140, kb, "t").rule == "half-line"
    assert length_issue("x" * 240, kb, "t").rule == "too-long"


def test_pi_5b_not_highlighted():
    out = escape_and_highlight("Deployed on Raspberry Pi 5B + Hailo-8 at <30ms latency.")
    assert r"{5B}" not in out and r"$<$30ms" in out


def test_trim_order_starts_with_achievements(final):
    st = {**final, "layout_action": "trim", "trim_level": 0, "last_fill": None,
          "layout": Layout(pages=2, free_lines=0, orphans=[])}
    upd = layout_fix(st, {})
    assert upd["max_achievements"] == 2 and upd["trim_level"] == 1


def test_overflowing_fill_is_undone(final):
    p0 = final["projects"][0]
    st = {**final, "layout_action": "trim", "last_fill": p0.project_id,
          "layout": Layout(pages=2, free_lines=0, orphans=[])}
    upd = layout_fix(st, {})
    assert len(upd["projects"][0].bullets) == len(p0.bullets) - 1
    assert upd["no_fill"] is True


def test_blank_space_gets_filled_before_orphans(final):
    L = final["layout"]
    assert L.free_lines < 3 or final.get("no_fill"), L


def test_near_duplicates_flagged_and_distinct_bank_bullets_not(kb):
    from resume_agent.validate import near_duplicate
    a = ("Ensembled 4 fine-tuned cross-encoders (bge-reranker-v2-m3, mDeBERTa-v3) with a cross-fitted LightGBM "
         "stacker and query-level gate, lifting leaderboard F0.5 from 0.894 to 0.9837 across 18 iterations.")
    b = ("Ensembled 4 fine-tuned cross-encoders (2x bge-reranker-v2-m3, 2x mDeBERTa-v3) lifting leaderboard F0.5 "
         "from 0.894 to 0.9837 across 18 iterations; orphan simulation (25% entity removal) added +0.0043 F0.5.")
    assert near_duplicate(a, b)                                   # seen in the live KA run
    for pid, p in kb.projects.items():
        bs = [x for x in p.bullets if x.id != "mv_b1_health"]
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                assert not near_duplicate(bs[i].text, bs[j].text), (pid, bs[i].id, bs[j].id)


def test_no_near_duplicates_ship(final, kb):
    dups = [i for i in deterministic_issues(final, kb) if i.rule == "duplicate"]
    assert not dups, dups
