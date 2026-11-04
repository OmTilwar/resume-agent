"""Structure and invariants of the knowledge base (kit §4, §7, §9, App. F)."""
from resume_agent.knowledge.loader import validate_references


def test_loads_with_expected_shape(kb):
    assert len(kb.projects) == 12
    assert {p.id for p in kb.selectable_projects(tiers=(1,))} == {
        "amazon_er", "finsight", "propvox", "medvision", "authnet", "resume_agent"}   # ResumeAgent: Tier 1 since 4 Nov
    assert kb.get_project("resume_agent").headline == ["97.9"]          # Tier 1 needs a headline (kit §10.3)
    assert len(kb.archetypes) == 22
    assert {g.id for g in kb.experience.groups} == set("ABCDE")
    assert len(kb.retired) == 2


def test_references_are_consistent(kb):
    assert validate_references(kb) == []


def test_profile_invariants(kb):
    p = kb.profile
    assert p.education.cgpa == "8.40"                      # never 8.36
    assert "Information Technology, Nagpur" in p.education.institution
    assert p.education.dates == "August 2023 - May 2027"
    assert kb.experience.dates == "August 2025 -- February 2026"
    assert set(kb.experience.titles) == {"Research Intern", "Machine Learning Research Intern"}


def test_every_project_has_enough_bank_bullets(kb):
    for p in kb.projects.values():
        assert len(p.bullets) >= 3, p.id
        assert p.titles, p.id
        if p.max_bullets:
            assert len(p.bullets) <= p.max_bullets + 1, p.id


def test_project_section_numbers_unique(kb):
    sections = [p.section for p in kb.projects.values()]
    assert len(sections) == len(set(sections))


def test_titles_respect_domain(kb):
    """Kit §2.4.1 — no domain-swapping titles."""
    for p in kb.projects.values():
        for t in p.titles:
            for term in p.forbidden_title_terms:
                assert term.lower() not in t.subtitle.lower(), (p.id, t.subtitle)


def test_github_urls(kb):
    for p in kb.projects.values():
        assert p.github.lower().startswith("https://github.com/omtilwar/"), p.id
