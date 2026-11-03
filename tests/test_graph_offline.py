"""End-to-end Phase 1 graph with a fake LLM — no network, no API key."""
import shutil

import pytest

from resume_agent.graph import build_graph
from resume_agent.latex import escape_and_highlight
from resume_agent.state import JobPosting

from .fakes import FABRICATED, fake_factory


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    job = JobPosting(company="Acme AI", role="LLM Engineering Intern", text="(fake JD)")
    return build_graph().invoke({"job": job, "out_dir": str(out)},
                                config={"configurable": {"llm_factory": fake_factory}})


def test_selects_llm_projects_in_order(final):
    assert final["selected"][0] == "finsight"
    assert set(final["selected"]) >= {"finsight", "propvox"}
    assert len(final["selected"]) == 4
    assert "id_vlm" not in final["selected"]


def test_fabricated_bullet_never_reaches_page(final):
    assert FABRICATED not in final["tex"]
    assert "10,000 customers" not in final["tex"]
    assert any("FABRICATED" in v or "unbacked-number" in v or "banned-phrase" in v for v in final["violations"])
    for p in final["projects"]:
        assert 3 <= len(p.bullets) <= 4


def test_skills_enforce_inventory_and_familiar_tags(final):
    entries = [e for line in final["skills"] for e in line.entries]
    assert "LlamaIndex (Familiar)" in entries         # tag re-attached
    assert "LlamaIndex" not in entries
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
