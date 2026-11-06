"""End-to-end Phase 1 graph with a fake LLM — no network, no API key."""
import shutil

import pytest

from resume_agent.graph import build_graph
from resume_agent.knowledge import get_kb
from resume_agent.latex import escape_and_highlight
from resume_agent.state import JobPosting

from .conftest import HAS_DESIGN_STENCIL, needs_design_stencil
from .fakes import FABRICATED, LLM_JD_ANALYSIS, fake_factory


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    job = JobPosting(company="Acme AI", role="LLM Engineering Intern", text="(fake JD)")
    return build_graph().invoke({"job": job, "out_dir": str(out)},
                                config={"configurable": {"llm_factory": fake_factory}})


def test_selects_llm_projects_in_order(final):
    assert final["selected"][0] == "finsight"
    assert set(final["selected"]) >= {"finsight", "propvox"}
    # an LLM JD puts Design Stencil first, so 3 projects (kit §2.1.2); without the private entry, 4
    assert len(final["selected"]) == (3 if HAS_DESIGN_STENCIL else 4)
    assert "id_vlm" not in final["selected"]


@needs_design_stencil
def test_llm_jd_puts_the_work_entry_first(final):
    assert final["work_ids"] == ["design_stencil"] and final["work_groups"][0] == "G1"
    ds = final["projects"][0]
    assert ds.kind == "work" and ds.subtitle == "Freelance AI Engineer" and ds.github == ""
    assert len(final["experience"].bullets) == 2  # the Research Intern entry gives way to 2 bullets
    tex = final["tex"]
    assert tex.index("Freelance AI Engineer") < tex.index("Research Intern") < tex.index("section*{Projects}")
    assert "Design Stencil -- Remote" in tex
    assert "github.com/OmTilwar/navaa" not in tex.lower() and "\\href{}" not in tex   # private repo: no link


def test_fabricated_bullet_never_reaches_page(final):
    assert FABRICATED not in final["tex"]
    assert "10,000 customers" not in final["tex"]
    assert any("FABRICATED" in v or "unbacked-number" in v or "banned-phrase" in v for v in final["violations"])
    for p in final["projects"]:
        assert 3 <= len(p.bullets) <= 4


def test_skills_enforce_inventory_and_familiar_tags(final):
    entries = [e for line in final["skills"] for e in line.entries]
    # the model wrote "LlamaIndex" without its tag: it may be trimmed for relevance/width,
    # but it must never appear untagged
    assert "LlamaIndex" not in entries
    familiar = {s.display for s in get_kb().familiar_skills()}
    assert all(e in familiar for e in entries if "(Familiar)" in e)
    assert "Kubernetes Operators" not in entries      # not in inventory → dropped
    assert "Underwater Basket Weaving" not in final["coursework"]


def test_tex_is_well_formed(final):
    tex = final["tex"]
    assert r"\textbf{\textcolor{NavyBlue}{" in tex
    assert "CGPA: 8.40" in tex
    assert "Machine Learning Research Intern" in tex
    assert tex.count(r"\begin{itemize}") == tex.count(r"\end{itemize}") == 6


def test_report_has_gaps_and_coverage(final):
    rep = final["report"]
    assert "Kubernetes (nice)" in rep.honest_gaps or rep.keyword_coverage["Kubernetes"]
    assert rep.keyword_coverage["Python"]
    assert len(rep.interview_questions) == 3


@pytest.mark.latex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_compiles(final):
    assert final["pdf_path"], final["violations"]
    assert final["pages"] >= 1
    print(f"\noffline graph resume: {final['pages']} page(s)")


def test_highlight_prefers_real_metrics():
    out = escape_and_highlight("Ensembled 4 cross-encoders, lifting F0.5 from 0.894 to 0.9837 across 18 iterations on 2-member team.")
    assert r"{0.894}" in out and r"{0.9837}" in out and r"{18}" in out
    assert r"\textcolor{NavyBlue}{2}" not in out          # 2-member is a descriptor
    assert "F0.5" in out and r"{0.5}" not in out


def test_project_headers_fit_one_line(final):
    from resume_agent.nodes import HEADER_MAX_CHARS
    for p in final["projects"]:
        header = f"{p.name}: {p.subtitle} | {', '.join(p.stack)}"
        assert len(header) <= HEADER_MAX_CHARS or len(p.stack) == 3, header
        low = [s.lower() for s in p.stack]
        assert not any(a != b and a in b for a in low for b in low), p.stack


def test_coverage_uses_synonyms(final):
    cov = final["report"].keyword_coverage
    assert cov["vector databases"], "Pinecone vector search should count"
    assert any("FinSight" in h for h in cov["LLMs"])


@needs_design_stencil
def test_work_entry_prompt_lists_the_jd_types_groups_first(kb):
    import json
    from resume_agent import prompts
    from resume_agent.knowledge import Context
    msgs = prompts.project_messages("design_stencil", LLM_JD_ANALYSIS, Context(), kb, 3, ["G2", "G1"])
    assert "work-experience entry" in msgs[0].content
    text = msgs[-1].content
    sent = [b["text"] for b in json.loads(text[text.index("{"):])["reference_bullets"]]
    group = {b.text: b.group for b in kb.get_project("design_stencil").bullets}
    assert [group[t] for t in sent[:2]] == ["G2", "G2"] and group[sent[2]] == "G1"
