"""Template renders from the bank and compiles (the Phase 1 assembler will replace build_sample)."""
import re
import shutil
import subprocess

import pytest

from resume_agent.knowledge import Context
from resume_agent.latex import env, escape


def build_sample(kb):
    """A default AI-generalist resume straight from bank bullets, no LLM."""
    ctx = Context(archetype="ai_generalist")
    arch = kb.archetypes["ai_generalist"]
    projects = []
    for pid in arch.flat_projects()[:4]:
        p = kb.get_project(pid)
        t = p.titles[0]
        projects.append(dict(
            name=escape(p.name), subtitle=escape(t.subtitle), stack=[escape(s) for s in t.stack],
            github=p.github, bullets=[escape(b.text) for b in kb.allowed_bullets(pid, ctx)[:3]],
        ))
    group = next(g for g in kb.experience.groups if g.id == arch.exp_groups[0])
    skills = [dict(category=escape(c), entries=[escape(s.display) for s in items[:10]])
              for c, items in list(kb.skills_by_category().items())[:4]]
    achievements = [a.latex for a in kb.achievements
                    if a.latex and a.drop_if_project not in arch.flat_projects()[:4]][:3]
    return env().get_template("resume.tex.j2").render(
        profile=kb.profile, tagline=None,
        coursework=[escape(c) for c in kb.profile.coursework[:5]],
        skills=skills,
        experience=dict(title=kb.experience.titles[0], org=kb.experience.org,
                        location=kb.experience.location, dates=kb.experience.dates,
                        bullets=[escape(b.text) for b in group.bullets[:3]]),
        projects=projects, achievements=achievements,
    )


def test_escape():
    assert escape("60.53% → 66.45% & R&D_x #1") == r"60.53\% $\rightarrow$ 66.45\% \& R\&D\_x \#1"
    assert escape("~30x, β=4.0, <30ms") == r"$\sim$30x, $\beta$=4.0, $<$30ms"
    assert escape("val–test") == "val--test"


def test_renders_valid_structure(kb):
    tex = build_sample(kb)
    assert tex.startswith(r"\documentclass[10pt]{article}")
    assert tex.rstrip().endswith(r"\end{document}")
    assert tex.count(r"\begin{itemize}") == tex.count(r"\end{itemize}") == 6   # exp + 4 projects + achievements
    assert "CGPA: 8.40" in tex
    assert not re.search(r"[→×β–—]", tex), "raw unicode leaked into .tex"
    assert "((" not in tex and "))" not in tex


@pytest.mark.latex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_compiles(kb, tmp_path):
    (tmp_path / "sample.tex").write_text(build_sample(kb), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "sample.tex"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-3000:]
    m = re.search(r"Output written on sample\.pdf \((\d+) pages?", r.stdout)
    assert m, r.stdout[-1500:]
    print(f"\nsample resume compiled: {m.group(1)} page(s)")
