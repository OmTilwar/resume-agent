import pytest

from resume_agent.selection import keyword_hits, score_projects, select_projects
from resume_agent.state import JDAnalysis, Keyword


def _a(arch, musts, nices=(), sec=None):
    return JDAnalysis(archetype=arch, secondary_archetype=sec, domain="x",
                      top_keywords=[Keyword(term=t, importance="must") for t in musts]
                      + [Keyword(term=t, importance="nice") for t in nices])


@pytest.mark.parametrize("kw,pid,hit", [
    ("Large Language Models", "finsight", True),
    ("Retrieval-Augmented Generation (RAG)", "finsight", True),
    ("Computer Vision", "medvision", True),
    ("DICOM", "medvision", True),
    ("ROS2", "submarine", True),
    ("Kubernetes", "medvision", False),
    ("LangGraph", "finsight", False),
])
def test_keyword_hits(kb, kw, pid, hit):
    assert keyword_hits(kw, kb.projects[pid]) is hit


def test_healthcare_jd_leads_with_medvision(kb):
    sel = select_projects(score_projects(_a("healthcare", ["medical imaging", "PyTorch", "HIPAA"]), kb), kb)
    assert sel[0] == "medvision"


def test_robotics_jd_pairs_vio_with_submarine(kb):
    sel = select_projects(score_projects(_a("robotics_cpp", ["C++", "ROS2", "sensor fusion", "EKF"]), kb), kb)
    assert sel[:2] == ["submarine", "vio"] or set(sel[:2]) == {"submarine", "vio"}


def test_tier2_excluded_unless_archetype_lists_it(kb):
    ids = {s.project_id for s in score_projects(_a("llm_genai_agents", ["LLM", "RAG"]), kb)}
    assert not ids & {"submarine", "vio", "photoretouch", "retailforecast", "id_vlm", "voxpipeline"}


def test_data_science_jd_gets_retailforecast(kb):
    sel = select_projects(score_projects(_a("data_science", ["SQL", "statistics", "forecasting"]), kb), kb)
    assert sel[0] == "retailforecast"
