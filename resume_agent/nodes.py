"""LangGraph nodes. Each takes (state, config) and returns a partial state update.

LLM access goes through config["configurable"]["llm_factory"](role, schema) so
tests can inject a fake; the default is resume_agent.llm.structured (OpenRouter).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from . import prompts
from .knowledge import Context, KnowledgeBase, get_kb
from .knowledge.checks import check_bullet
from .latex import env, escape, escape_and_highlight
from .selection import mentions, score_projects, select_projects
from .state import (
    DraftBullet, ExperienceBlock, ExperienceDraft, FinalBullet, JDAnalysis, ProjectBlock,
    ProjectDraft, Report, ResumeState, SkillLine, SkillsDraft,
)

log = logging.getLogger("resume_agent")
OUT_ROOT = Path(__file__).resolve().parents[1] / "out"


# ---- helpers ----------------------------------------------------------------

def _factory(config: RunnableConfig):
    f = (config or {}).get("configurable", {}).get("llm_factory")
    if f is None:
        from .llm import structured
        f = structured
    return f


def _call(config: RunnableConfig, role: str, schema, messages, attempts: int = 2):
    last: Exception | None = None
    for i in range(attempts):
        try:
            out = _factory(config)(role, schema).invoke(messages)
            if out is None:
                raise ValueError("model returned no structured output")
            return out
        except Exception as e:  # noqa: BLE001 — free models fail in many ways; retry then surface
            last = e
            log.warning("%s call failed (attempt %d/%d): %s", role, i + 1, attempts, e)
    raise RuntimeError(f"{role} LLM call failed after {attempts} attempts: {last}") from last


def _ctx(state: ResumeState) -> Context:
    return Context(archetype=state["analysis"].archetype, company=state["job"].company)


def slugify(*parts: str) -> str:
    s = "_".join(p for p in parts if p)
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:60] or "job"


def _vet(drafts: list[DraftBullet], where: str, facts_by_id: dict, ctx: Context, kb: KnowledgeBase,
         project=None) -> tuple[list[FinalBullet], list[str]]:
    """Keep LLM bullets that pass every truth check; report the rest."""
    kept, problems = [], []
    for i, d in enumerate(drafts):
        unknown = [f for f in d.fact_ids if f not in facts_by_id]
        cited = [facts_by_id[f] for f in d.fact_ids if f in facts_by_id]
        v = check_bullet(d.text, f"{where}#{i + 1}", cited, ctx, kb, project)
        if unknown:
            problems.append(f"{where}#{i + 1}: cites unknown/disallowed facts {unknown}")
        elif v:
            problems.extend(str(x) for x in v)
        else:
            kept.append(FinalBullet(text=d.text.strip(), fact_ids=d.fact_ids, origin="llm"))
    return kept, problems


HEADER_MAX_CHARS = 108   # "Name: Subtitle | a, b, c" must fit one printed line next to the GitHub icon


def fit_stack(name: str, subtitle: str, wanted: list[str], allowed: list[str], default: list[str]) -> list[str]:
    """Only real stack items; no near-duplicates; 3-5 items; header fits one line (kit §10.5)."""
    lc = {s.lower(): s for s in allowed}
    stack = list(dict.fromkeys(lc[s.lower()] for s in wanted if s.lower() in lc))
    stack = [s for s in stack if not any(s.lower() != o.lower() and s.lower() in o.lower() for o in stack)]
    if len(stack) < 3:
        stack = list(dict.fromkeys(stack + [s for s in default if s not in stack]))
    stack = stack[:5]
    while len(stack) > 3 and len(f"{name}: {subtitle} | {', '.join(stack)}") > HEADER_MAX_CHARS:
        stack.pop()
    return stack


def _fill_from_bank(kept: list[FinalBullet], bank, n_min: int, n_max: int) -> list[FinalBullet]:
    used = {b.text for b in kept}
    for b in bank:
        if len(kept) >= n_min:
            break
        if b.text not in used:
            kept.append(FinalBullet(text=b.text, fact_ids=b.fact_ids, origin="bank"))
    return kept[:n_max]


# ---- nodes -----------------------------------------------------------------

def ingest(state: ResumeState, config: RunnableConfig) -> ResumeState:
    job = state["job"]
    slug = state.get("slug") or slugify(job.company, job.role)
    out_dir = Path(state.get("out_dir") or OUT_ROOT / slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"slug": slug, "out_dir": str(out_dir)}


def analyze(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    a: JDAnalysis = _call(config, "analyzer", JDAnalysis, prompts.analyze_messages(state["job"], kb))
    if a.archetype not in kb.archetypes:
        log.warning("unknown archetype %r from analyzer; defaulting to ai_generalist", a.archetype)
        a.archetype = "ai_generalist"
    if a.secondary_archetype not in kb.archetypes:
        a.secondary_archetype = None
    a.top_keywords = a.top_keywords[:10]
    return {"analysis": a}


def select(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    a = state["analysis"]
    scores = score_projects(a, kb)
    groups = list(dict.fromkeys(
        kb.archetypes[a.archetype].exp_groups
        + (kb.archetypes[a.secondary_archetype].exp_groups if a.secondary_archetype else [])
    ))
    return {"scores": scores, "selected": select_projects(scores, kb), "exp_groups": groups}


def _write_one_project(pid: str, state: ResumeState, config: RunnableConfig) -> tuple[ProjectBlock, list[str]]:
    kb, ctx = get_kb(), _ctx(state)
    p = kb.get_project(pid)
    n_max = min(p.max_bullets or 4, 4)
    n = 3
    draft: ProjectDraft = _call(config, "writer", ProjectDraft,
                                prompts.project_messages(pid, state["analysis"], ctx, kb, n))
    facts = {f.id: f for f in kb.allowed_facts(pid, ctx)}
    _, bad = _vet(draft.bullets, pid, facts, ctx, kb, p)
    # Phase 2: keep the model's bullets even if flawed — validate/fix repairs them,
    # fallback swaps in bank bullets only if repair fails.
    drafted = [FinalBullet(text=d.text.strip(), fact_ids=d.fact_ids, origin="llm") for d in draft.bullets[:n]]
    bullets = _fill_from_bank(drafted, kb.allowed_bullets(pid, ctx), n, n_max)
    idx = draft.title_index if 0 <= draft.title_index < len(p.titles) else 0
    title = p.titles[idx]
    stack = fit_stack(p.name, title.subtitle, draft.stack, p.stack, title.stack)
    block = ProjectBlock(project_id=pid, name=p.name, subtitle=title.subtitle, stack=stack,
                         github=p.github, bullets=bullets)
    return block, [f"first draft: {x}" for x in bad]


def max_concurrency(config: RunnableConfig) -> int:
    return int((config or {}).get("configurable", {}).get("max_concurrency", 4))


def write_projects(state: ResumeState, config: RunnableConfig) -> ResumeState:
    """The 4 project calls are independent: run them concurrently (capped for rate limits)."""
    from concurrent.futures import ThreadPoolExecutor
    pids = state["selected"]
    with ThreadPoolExecutor(max_workers=max(1, min(max_concurrency(config), len(pids)))) as pool:
        results = list(pool.map(lambda pid: _write_one_project(pid, state, config), pids))   # keeps rank order
    return {"projects": [b for b, _ in results], "violations": [v for _, vs in results for v in vs]}


def write_experience(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb, ctx = get_kb(), _ctx(state)
    exp = kb.experience
    draft: ExperienceDraft = _call(config, "writer", ExperienceDraft,
                                   prompts.experience_messages(state["exp_groups"], state["analysis"], kb))
    facts = {f.id: f for f in exp.facts}
    _, bad = _vet(draft.bullets, "experience", facts, ctx, kb)
    kept = [FinalBullet(text=d.text.strip(), fact_ids=d.fact_ids, origin="llm") for d in draft.bullets[:3]]
    bad = [f"first draft: {x}" for x in bad]
    bank = [b for g in exp.groups if g.id in state["exp_groups"] for b in g.bullets] + \
           [b for g in exp.groups if g.id not in state["exp_groups"] for b in g.bullets]
    bullets = _fill_from_bank(kept, bank, 3, 4)
    title = draft.title if draft.title in exp.titles else exp.titles[0]
    return {"experience": ExperienceBlock(title=title, bullets=bullets), "violations": bad}


def write_skills(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    draft: SkillsDraft = _call(config, "fast", SkillsDraft,
                               prompts.skills_messages(state["analysis"], state["selected"], kb))
    by_name = {s.name.lower(): s for s in kb.skills}
    by_name.update({s.display.lower(): s for s in kb.skills})
    lines, problems = [], []
    seen: set[str] = set()   # an entry appears on one line only
    for line in draft.lines[:5]:
        entries = []
        for e in line.entries:
            s = by_name.get(e.strip().lower())
            if s is None:
                problems.append(f"skills: dropped {e!r} (not in inventory)")
                continue
            if s.name.lower() in seen:
                continue
            seen.add(s.name.lower())
            entries.append(s.display)  # re-attaches '(Familiar)' if the model stripped it
        entries = entries[:14]
        if len(entries) >= 4:
            cat = re.sub(r"\s*\((?:nice|must|nice-to-have|must-have|required|preferred)\)\s*", " ", line.category, flags=re.I)
            lines.append(SkillLine(category=cat.strip(), entries=entries))
    if len(lines) < 4:  # fallback: inventory order
        problems.append("skills: model returned <4 usable lines; filled from inventory")
        for cat, items in kb.skills_by_category().items():
            if len(lines) >= 4:
                break
            if all(l.category != cat for l in lines):
                lines.append(SkillLine(category=cat, entries=[s.display for s in items[:12]]))

    prof = kb.profile
    allowed_courses = {c.lower(): c for c in prof.coursework + prof.coursework_shorthand}
    courses, keys = [], set()
    for c in draft.coursework:
        name = allowed_courses.get(c.strip().lower())
        if name and prof.course_key(name) not in keys:   # "Machine Learning" == "Fundamentals of ML"
            keys.add(prof.course_key(name))
            courses.append(name)
    courses = courses[:5] or prof.coursework[:5]
    return {"skills": lines, "coursework": courses, "violations": problems}


def pick_achievements(state: ResumeState, kb: KnowledgeBase) -> list[str]:
    """Kit §8, ranked: publication, hackathons, then the best extras. Callers slice to fit the page."""
    selected = set(state["selected"])
    a = state["analysis"]
    text = " ".join([k.term for k in a.top_keywords] + [a.archetype, a.domain]).lower()
    by_id = {x.id: x for x in kb.achievements}
    gd, eyrc = by_id["groundingdino_pr"], by_id["eyrc"]
    wants_gd = a.archetype in {"ml_infra", "ml_cv_engineer", "applied_scientist"} or any(t in text for t in gd.include_for)
    extras = []
    if "amazon_er" not in selected:
        extras.append(by_id["amazon_ml_challenge"])
    if a.archetype in {"robotics_cpp", "vio_slam"}:
        extras += [eyrc, gd]
    else:
        extras += [gd, eyrc] if wants_gd else [eyrc, gd]
    ranked = [by_id["ieee_publication"], by_id["hackathon_wins"], *extras]
    seen, out = set(), []
    for x in ranked:
        if x.id not in seen:
            seen.add(x.id)
            out.append(x.latex)
    return out


def assemble(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    achievements = pick_achievements(state, kb)[: state.get("max_achievements", 3)]
    exp = state["experience"]
    tex = env().get_template("resume.tex.j2").render(
        profile=kb.profile,
        tagline=None,
        coursework=[escape(c) for c in state["coursework"]],
        skills=[dict(category=escape(l.category), entries=[escape(e) for e in l.entries]) for l in state["skills"]],
        experience=dict(title=escape(exp.title), org=kb.experience.org, location=kb.experience.location,
                        dates=kb.experience.dates,
                        bullets=[escape_and_highlight(b.text) for b in exp.bullets]),
        projects=[dict(name=escape(p.name), subtitle=escape(p.subtitle), stack=[escape(s) for s in p.stack],
                       github=p.github, bullets=[escape_and_highlight(b.text) for b in p.bullets])
                  for p in state["projects"]],
        achievements=achievements,
    )
    path = Path(state["out_dir"]) / f"{state['slug']}.tex"
    path.write_text(tex, encoding="utf-8")
    return {"tex": tex, "tex_path": str(path), "achievements": achievements}


def compile_pdf(state: ResumeState, config: RunnableConfig) -> ResumeState:
    if shutil.which("pdflatex") is None:
        log.warning("pdflatex not found; skipping compile")
        return {"pdf_path": None, "pages": None}
    out_dir = Path(state["out_dir"])
    tex = Path(state["tex_path"])
    r = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out_dir}", str(tex)],
        capture_output=True, text=True, timeout=300, cwd=out_dir,
    )
    flat = r.stdout.replace("\r", "").replace("\n", "")   # MiKTeX hard-wraps the log at 79 chars
    m = re.search(r"Output written on .*?\((\d+) pages?", flat)
    if r.returncode != 0 or not m:
        tail = r.stdout[-2000:]
        return {"pdf_path": None, "pages": None,
                "violations": [f"compile failed:\n{tail}"]}
    return {"pdf_path": str(out_dir / f"{tex.stem}.pdf"), "pages": int(m.group(1))}


def report(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    a = state["analysis"]
    places: dict[str, list[str]] = {}
    for k in a.top_keywords:
        t = k.term
        hits = [f"Skills: {l.category}" for l in state["skills"]
                if mentions(t, l.category + " " + ", ".join(l.entries))]
        hits += [f"{p.name} #{i + 1}" for p in state["projects"] for i, b in enumerate(p.bullets) if mentions(t, b.text)]
        hits += [f"Experience #{i + 1}" for i, b in enumerate(state["experience"].bullets) if mentions(t, b.text)]
        places[t] = hits
    gaps = [f"{k.term} ({k.importance})" for k in a.top_keywords if not places[k.term]]
    questions = []
    for pid in state["selected"]:
        qs = kb.get_project(pid).interview
        if qs:
            questions.append(f"{kb.get_project(pid).name}: {qs[0].q}")
    replaced = [f"{p.name}: {sum(b.origin == 'bank' for b in p.bullets)} bank bullet(s)" for p in state["projects"]
                if any(b.origin == "bank" for b in p.bullets)]
    rep = Report(keyword_coverage=places, honest_gaps=gaps, interview_questions=questions[:3],
                 replaced_bullets=replaced, pages=state.get("pages"), tex_path=state["tex_path"],
                 pdf_path=state.get("pdf_path"))
    Path(state["out_dir"], "report.md").write_text(render_report(state, rep), encoding="utf-8")
    return {"report": rep}


def render_report(state: ResumeState, rep: Report) -> str:
    a, job = state["analysis"], state["job"]
    lines = [f"# {job.company} — {job.role}", "",
             f"**Archetype:** {a.archetype}" + (f" (+ {a.secondary_archetype})" if a.secondary_archetype else "")
             + f" · **Domain:** {a.domain}",
             f"**Pages:** {rep.pages if rep.pages is not None else 'not compiled'}", "",
             "## Top keywords", *[f"{i + 1}. {k.term} ({k.importance})" for i, k in enumerate(a.top_keywords)], "",
             "## Projects chosen",
             *[f"- **{s.project_id}** — score {s.score}: {'; '.join(s.reasons)}"
               for s in state["scores"] if s.project_id in state["selected"]],
             f"- Experience groups: {', '.join(state['exp_groups'])}", "",
             "## Keyword coverage",
             *[f"- {k}: {', '.join(v) if v else '—'}" for k, v in rep.keyword_coverage.items()], "",
             "## Honest gaps", *([f"- {g}" for g in rep.honest_gaps] or ["- none"]), "",
             "## Interview prep", *[f"- {q}" for q in rep.interview_questions], "",
             *_quality_section(state, rep)]
    return "\n".join(lines) + "\n"


def _quality_section(state: ResumeState, rep: Report) -> list[str]:
    L = state.get("layout")
    open_issues = state.get("issues", [])
    out = ["## Quality checks",
           f"- Content fix rounds: {state.get('fix_round', 0)} · judge-approved bullets: {len(state.get('approved', []))}",
           f"- Layout: {L.pages} page(s), {L.free_lines} blank line(s) at bottom, "
           f"orphans: {', '.join(L.orphans) or 'none'} · layout rounds: {state.get('layout_round', 0)}"
           if L else "- Layout: not measured (no PDF)"]
    out += [f"- {r}" for r in rep.replaced_bullets] or ["- every bullet on the page is model-written and passed all checks"]
    out += ["", "## Still open (review before sending)"]
    out += [f"- {i}" for i in open_issues] or ["- nothing"]
    if L and L.pages > 1:
        out.append("- **page overflow could not be fixed automatically — trim by hand (kit §11)**")
    out += ["", "## Pipeline log", *[f"- {n}" for n in state.get("notes", [])],
            *[f"- {v}" for v in state.get("violations", [])]]
    return out
