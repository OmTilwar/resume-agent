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


def _child_config(config: RunnableConfig, role: str, schema, attempt: int) -> dict:
    """Name each model call by role and nest it under the current graph step in traces."""
    cfg = config or {}
    return {
        "callbacks": cfg.get("callbacks"),
        "run_name": f"{role}:{schema.__name__}",
        "tags": [*(cfg.get("tags") or []), f"role:{role}"],
        "metadata": {**(cfg.get("metadata") or {}), "role": role, "schema": schema.__name__, "attempt": attempt},
    }


def _call(config: RunnableConfig, role: str, schema, messages, attempts: int = 2):
    last: Exception | None = None
    for i in range(attempts):
        try:
            factory = _factory(config)
            try:
                model = factory(role, schema, attempt=i + 1)   # retries move along the model chain
            except TypeError:                                 # test fakes take (role, schema) only
                model = factory(role, schema)
            out = model.invoke(messages, config=_child_config(config, role, schema, i + 1))
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


def bank_swap(bullet: FinalBullet, section: str, index: int, keep: list[str], bank, kb: KnowledgeBase,
              used: set[str] = frozenset()):
    """Best well-sized verified bullet to stand in for `bullet` (no model call): numbers the section doesn't show yet
    first, then the same facts,
    never a near-duplicate of `keep` (the block's other bullets), never a 4th-bullet-only fact in slots 1-3.
    Returns None when nothing fits."""
    from .knowledge.numbers import extract_numbers
    from .validate import length_issue, near_duplicate, repeated_numbers
    cands = [b for b in bank
             if b.text not in used and b.text not in keep and b.text != bullet.text
             and not length_issue(b.text, kb, "x")
             and not any(near_duplicate(b.text, t) for t in keep)
             and not (index < 3 and kb.cites_last_only(section, b.fact_ids))]
    if not cands:
        return None
    want = set(bullet.fact_ids)
    # new metrics first (don't show the section's headline number twice), then the same facts; ties keep kit order
    return max(cands, key=lambda b: (-len(repeated_numbers(b.text, keep)), bool(extract_numbers(b.text)),
                                     len(want & set(b.fact_ids))))


def snap_weak(bullets: list[FinalBullet], section: str, bank, kb: KnowledgeBase,
              notes: list[str], only: set[int] | None = None) -> list[FinalBullet]:
    """Model-written bullets that are off-length, have no number, or repeat a number an earlier bullet of the block
    already shows are swapped for a verified kit bullet straight away (no model call). The fixer rarely gets these
    right: LangSmith, 4 Nov — 10 of 16 first drafts were 120-169 chars and most were still off after 2 fixer rounds;
    the DataCore run shipped "…via a 4-stage pipeline" (repeating bullet 1) and a bullet with no number.
    `only`: restrict to these positions (e.g. the bullets the fixer just rewrote)."""
    from .validate import weak_reason
    out = list(bullets)
    for i, b in enumerate(out):
        if b.origin != "llm" or (only is not None and i not in only):
            continue
        why = weak_reason(b.text, [x.text for x in out[:i]], kb, section)
        if why is None:
            continue
        keep = [x.text for j, x in enumerate(out) if j != i]
        swap = bank_swap(b, section, i, keep, bank, kb)
        if swap is not None:
            out[i] = FinalBullet(text=swap.text, fact_ids=swap.fact_ids, origin="bank")
            notes.append(f"{section}#{i + 1}: draft {why} — used a verified kit bullet")
    return out


def has_headline(bullets: list[FinalBullet], project) -> bool:
    from .knowledge.numbers import extract_numbers
    return not project or not project.headline or any(
        set(project.headline) & extract_numbers(b.text) for b in bullets)


def ensure_headline(bullets: list[FinalBullet], project, bank, notes: list[str]) -> list[FinalBullet]:
    """Kit §10.3: a project's headline result must be on the page. DataCore run, 4 Nov: the fixer rewrote Amazon
    bullet 1 without 0.9837 — the flagship lost its score. Swap in the kit bullet that carries it (no model call):
    in place of a bullet saying the same thing, else at the top, dropping the block's last bullet."""
    from .validate import near_duplicate
    if has_headline(bullets, project):
        return bullets
    from .knowledge.numbers import extract_numbers
    on = {b.text for b in bullets}
    cands = [b for b in bank if set(project.headline) & extract_numbers(b.text) and b.text not in on]
    if not cands:
        return bullets
    h = FinalBullet(text=cands[0].text, fact_ids=cands[0].fact_ids, origin="bank")
    out = list(bullets)
    twin = next((i for i, b in enumerate(out) if near_duplicate(b.text, h.text)), None)
    if twin is not None:
        out[twin] = h
    else:
        out = [h] + out[:max(len(out) - 1, 0)] if out else [h]
    notes.append(f"{project.id}: headline result ({', '.join(project.headline)}) was missing — used the kit bullet that has it")
    return out


def _fill_from_bank(kept: list[FinalBullet], bank, n_min: int, n_max: int) -> list[FinalBullet]:
    """Top up to n_min from the kit bank, preferring bullets that add metrics the block doesn't show yet."""
    from .validate import near_duplicate, repeated_numbers
    kept = list(kept)
    while len(kept) < n_min:
        texts = [b.text for b in kept]
        cands = [b for b in bank if b.text not in texts and not any(near_duplicate(b.text, t) for t in texts)]             or [b for b in bank if b.text not in texts]
        if not cands:
            break
        b = min(cands, key=lambda b: len(repeated_numbers(b.text, texts)))   # stable: ties keep kit order
        kept.append(FinalBullet(text=b.text, fact_ids=b.fact_ids, origin="bank"))
    return kept[:n_max]


# ---- nodes -----------------------------------------------------------------

def ingest(state: ResumeState, config: RunnableConfig) -> ResumeState:
    job = state["job"]
    slug = state.get("slug") or slugify(job.company, job.role)
    out_dir = Path(state.get("out_dir") or OUT_ROOT / slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"slug": slug, "out_dir": str(out_dir)}


def _analysis_cache_path(config: RunnableConfig, messages) -> Path | None:
    """Same JD + same prompt + same analyzer model -> same file. Off unless the caller sets `cache_dir`."""
    cache_dir = (config or {}).get("configurable", {}).get("cache_dir")
    if not cache_dir:
        return None
    import hashlib
    import json
    from .llm import load_config
    role = load_config()["roles"]["analyzer"]
    key = json.dumps({"messages": [m.content for m in messages], "model": role["model"],
                      "fallbacks": role.get("fallbacks", []), "schema": JDAnalysis.model_json_schema()},
                     sort_keys=True, ensure_ascii=False)
    return Path(cache_dir) / "analysis" / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.json"


def analyze(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    messages = prompts.analyze_messages(state["job"], kb)
    cache = _analysis_cache_path(config, messages)
    if cache is not None and cache.exists():
        a = JDAnalysis.model_validate_json(cache.read_text(encoding="utf-8"))
        log.info("analysis: reused %s (same JD, prompt and model)", cache.name)
    else:
        a = _call(config, "analyzer", JDAnalysis, messages)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(a.model_dump_json(indent=1), encoding="utf-8")
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
    scores = score_projects(a, kb, state["job"].text)
    groups = list(dict.fromkeys(
        kb.archetypes[a.archetype].exp_groups
        + (kb.archetypes[a.secondary_archetype].exp_groups if a.secondary_archetype else [])
    ))
    return {"scores": scores, "selected": select_projects(scores, kb, analysis=a), "exp_groups": groups}


def _write_one_project(pid: str, state: ResumeState, config: RunnableConfig) -> tuple[ProjectBlock, list[str]]:
    kb, ctx = get_kb(), _ctx(state)
    p = kb.get_project(pid)
    n_max = min(p.max_bullets or 4, 4)
    n = 3
    notes: list[str] = []
    try:
        draft: ProjectDraft = _call(config, "writer", ProjectDraft,
                                    prompts.project_messages(pid, state["analysis"], ctx, kb, n))
    except RuntimeError as e:   # one failed call must not sink the run: fall back to verified bullets
        log.warning("writer failed for %s; using bank bullets: %s", pid, e)
        draft = ProjectDraft(title_index=0, stack=[], bullets=[])
        notes.append(f"{pid}: writer model failed — used verified kit bullets ({e})")
    facts = {f.id: f for f in kb.allowed_facts(pid, ctx)}
    _, bad = _vet(draft.bullets, pid, facts, ctx, kb, p)
    # Phase 2: keep the model's bullets even if flawed — validate/fix repairs them,
    # fallback swaps in bank bullets only if repair fails.
    drafted = []
    for d in draft.bullets:
        if kb.cites_last_only(pid, d.fact_ids):     # kit: weak facts only as a 4th bullet
            notes.append(f"{pid}: moved out a bullet citing a 4th-bullet-only fact ({d.text[:50]}…)")
            continue
        drafted.append(FinalBullet(text=d.text.strip(), fact_ids=d.fact_ids, origin="llm"))
    bank = [b for b in kb.allowed_bullets(pid, ctx) if not kb.cites_last_only(pid, b.fact_ids)]
    bullets = _fill_from_bank(snap_weak(drafted[:n], pid, bank, kb, notes), bank, n, n_max)
    bullets = ensure_headline(bullets, p, bank, notes)
    idx = draft.title_index if 0 <= draft.title_index < len(p.titles) else 0
    title = p.titles[idx]
    stack = fit_stack(p.name, title.subtitle, draft.stack, p.stack, title.stack)
    block = ProjectBlock(project_id=pid, name=p.name, subtitle=title.subtitle, stack=stack,
                         github=p.github, bullets=bullets)
    return block, notes + [f"first draft: {x}" for x in bad]


def max_concurrency(config: RunnableConfig) -> int:
    return int((config or {}).get("configurable", {}).get("max_concurrency", 4))


def write_projects(state: ResumeState, config: RunnableConfig) -> ResumeState:
    """The 4 project calls are independent: run them concurrently (capped for rate limits).
    ContextThreadPoolExecutor carries tracing context into the worker threads."""
    from langchain_core.runnables.config import ContextThreadPoolExecutor
    pids = state["selected"]
    with ContextThreadPoolExecutor(max_workers=max(1, min(max_concurrency(config), len(pids)))) as pool:
        results = list(pool.map(lambda pid: _write_one_project(pid, state, config), pids))   # keeps rank order
    return {"projects": [b for b, _ in results], "violations": [v for _, vs in results for v in vs]}


def write_experience(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb, ctx = get_kb(), _ctx(state)
    exp = kb.experience
    failed: list[str] = []
    try:
        draft: ExperienceDraft = _call(config, "writer", ExperienceDraft,
                                       prompts.experience_messages(state["exp_groups"], state["analysis"], kb))
    except RuntimeError as e:
        log.warning("writer failed for experience; using bank bullets: %s", e)
        draft = ExperienceDraft(title=exp.titles[0], bullets=[])
        failed.append(f"experience: writer model failed — used verified kit bullets ({e})")
    facts = {f.id: f for f in exp.facts}
    _, bad = _vet(draft.bullets, "experience", facts, ctx, kb)
    kept = [FinalBullet(text=d.text.strip(), fact_ids=d.fact_ids, origin="llm") for d in draft.bullets[:3]]
    bad = failed + [f"first draft: {x}" for x in bad]
    bank = [b for g in exp.groups if g.id in state["exp_groups"] for b in g.bullets] + \
           [b for g in exp.groups if g.id not in state["exp_groups"] for b in g.bullets]
    bullets = _fill_from_bank(snap_weak(kept, "experience", bank, kb, bad), bank, 3, 4)
    title = draft.title if draft.title in exp.titles else exp.titles[0]
    return {"experience": ExperienceBlock(title=title, bullets=bullets), "violations": bad}


# Kit §10.4: 4-5 lines, 8-14 entries each, each line <= ~1.3 printed lines.
# Template: ~115 visible chars per printed line (kit §11) -> ~150 chars per Skills line.
SKILL_LINE_MAX_CHARS = 150
SKILL_MIN_ENTRIES, SKILL_MAX_ENTRIES = 5, 14   # kit asks 8+, but width wins: long entries ("Dense Retrieval / Bi-Encoders (...)") can't fit 8
SKILL_LINES_MIN, SKILL_LINES_MAX = 4, 5


def skill_line_chars(line: SkillLine) -> int:
    return len(f"{line.category}: {', '.join(line.entries)}")


def _relevance(state: ResumeState, kb: KnowledgeBase):
    """entry -> True if it relates to a JD keyword or a chosen project's stack/tags."""
    from .selection import _norm
    keywords = [k.term for k in state["analysis"].top_keywords]
    vocab = {_norm(x) for pid in state["selected"] for p in [kb.get_project(pid)] for x in [*p.stack, *p.tags]}
    vocab = {v for v in vocab if len(v) > 2}

    from .selection import _probes
    kw_probes = {p for k in keywords for p in _probes(k)}

    def relevant(entry: str) -> bool:
        e = _norm(entry.replace("(Familiar)", ""))
        if any(mentions(k, entry) for k in keywords):
            return True
        # expand the entry the same way keywords are expanded: "Large Language Models" ~ "LLMs"
        if _probes(entry.replace("(Familiar)", "")) & (kw_probes | vocab):
            return True
        return any(re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", e) for v in vocab)
    return relevant


def _jd_relevance(state: ResumeState):
    """entry -> True only if it relates to a JD keyword (stricter than _relevance, which also counts the chosen
    projects' stacks). Used to rank: the JD's skills go first, project-only ones fill in after."""
    from .selection import _probes
    keywords = [k.term for k in state["analysis"].top_keywords]
    kw_probes = {p for k in keywords for p in _probes(k)}

    def jd(entry: str) -> int:
        """2 = the entry names a JD keyword (or its synonym); 1 = only the entry's own synonyms reach one
        ("Vision-Language Models" ~ "language models"); 0 = unrelated."""
        e = entry.replace("(Familiar)", "")
        if any(mentions(k, entry) for k in keywords):
            return 2
        return 1 if _probes(e) & kw_probes else 0
    return jd


_FILLER = {"libraries", "library", "tools", "tooling", "frameworks", "framework", "skills", "core", "stack",
            "languages", "technologies", "platforms", "methods", "other"}


def honest_category(line: SkillLine, kb: KnowledgeBase) -> str:
    """The model renames categories to mirror the JD; keep its name only if every part of it is backed by an entry
    on the line. Live run, 4 Nov: 'AI APIs & Model Training' held REST APIs, WebSockets, asyncio, Pydantic — nothing
    about training. Otherwise fall back to the entries' own inventory category ('Backend & APIs')."""
    from .selection import _norm, _probes
    cat_of: dict[str, str] = {}
    for sk in kb.skills:
        cat_of.setdefault(sk.display, sk.category)   # TypeScript is listed twice; its first home wins
    inv = [cat_of.get(e, "") for e in line.entries]

    def backed(part: str) -> bool:
        words = [w for w in _norm(part).split() if w not in _FILLER]
        if not words:
            return True
        part = " ".join(words)
        probes = _probes(part)
        return any(mentions(part, e) or probes & _probes(e.replace("(Familiar)", ""))
                   or re.search(rf"(?<![a-z0-9]){re.escape(part)}(?![a-z0-9])", _norm(c))
                   for e, c in zip(line.entries, inv))

    parts = [p for p in re.split(r"\s*(?:&|/|,|\band\b)\s*", line.category) if p.strip()]
    names_one_entry = len(parts) == 1 and _norm(parts[0]) in {_norm(e) for e in line.entries} and len(line.entries) > 1
    if parts and all(backed(p) for p in parts) and not names_one_entry:
        return line.category
    # most entries' category; ties go to the first entry's (the line leads with its most relevant entry)
    common = max(dict.fromkeys(c for c in inv if c), key=lambda c: (inv.count(c), -inv.index(c)), default="")
    if not common:
        return line.category
    return common.replace(" / ", " & ") if common.count(" / ") == 1 else common   # "Backend & APIs"; keep "ML / DL / CV"


def fit_skills(lines: list[SkillLine], state: ResumeState, kb: KnowledgeBase) -> tuple[list[SkillLine], list[str]]:
    """Enforce kit §10.4 deterministically: relevant categories, JD-relevant entries first, line width."""
    notes: list[str] = []
    relevant = _relevance(state, kb)
    jd = _jd_relevance(state)
    used = {e for l in lines for e in l.entries}

    # 1. swap out lines with nothing relevant to this JD; pad to the minimum line count
    kept = [l for l in lines if any(relevant(e) for e in l.entries)]
    for l in lines:
        if l not in kept:
            notes.append(f"skills: dropped line '{l.category}' — nothing in it relates to this JD")
            used -= set(l.entries)
    if len(kept) < SKILL_LINES_MIN:
        ranked = []
        for cat, items in kb.skills_by_category().items():
            if any(cat == l.category for l in kept):
                continue
            entries = [s.display for s in items if s.display not in used]
            score = sum(relevant(e) for e in entries)
            if score:
                ranked.append((score, cat, entries))
        for score, cat, entries in sorted(ranked, key=lambda r: -r[0]):
            if len(kept) >= SKILL_LINES_MIN:
                break
            kept.append(SkillLine(category=cat, entries=entries))
            used |= set(entries)
            notes.append(f"skills: added inventory line '{cat}' ({score} JD-relevant entries)")
    kept = kept[:SKILL_LINES_MAX]

    # 2. per line: JD-relevant entries first (stable), then trim to the width budget
    out = []
    for l in kept:
        # JD-keyword entries first, then project-only ones, so width trimming drops the latter first
        rel = sorted((e for e in l.entries if relevant(e)), key=lambda e: -jd(e))
        other = sorted((e for e in l.entries if not relevant(e)), key=lambda e: "(Familiar)" in e)
        # no padding: unrelated entries (TypeScript, CMake on a data-science JD) only fill a short line
        pad = other[:max(0, SKILL_MIN_ENTRIES - len(rel))]
        if len(other) > len(pad):
            notes.append(f"skills: '{l.category}' dropped unrelated {', '.join(other[len(pad):])}")
        entries = (rel + pad)[:SKILL_MAX_ENTRIES]
        line = SkillLine(category=l.category, entries=entries)
        dropped = []
        while skill_line_chars(line) > SKILL_LINE_MAX_CHARS and len(line.entries) > SKILL_MIN_ENTRIES:
            dropped.append(line.entries.pop())
        if dropped:
            notes.append(f"skills: trimmed '{l.category}' to fit ~1.3 lines (dropped {', '.join(reversed(dropped))})")
        name = honest_category(line, kb)
        if name != line.category and name not in {o.category for o in out}:
            notes.append(f"skills: renamed '{line.category}' to '{name}' — its entries don't back the name")
            line = SkillLine(category=name, entries=line.entries)
            while skill_line_chars(line) > SKILL_LINE_MAX_CHARS and len(line.entries) > SKILL_MIN_ENTRIES:
                line.entries.pop()
        out.append(line)

    # 3. kit §10.4: order lines by JD importance; a line that still can't fit ~1.3 printed lines gives way if the
    #    section has lines to spare (DataCore, 4 Nov: 'Retrieval & Ranking' took 2 printed lines on a JD that never
    #    mentions retrieval, while JD skills were trimmed off the LLM line)
    def score(l: SkillLine) -> int:
        return sum(jd(e) > 0 for e in l.entries) + 2 * any(mentions(k.term, l.category) for k in state["analysis"].top_keywords)
    out.sort(key=score, reverse=True)
    while len(out) > SKILL_LINES_MIN:
        wide = [l for l in out if skill_line_chars(l) > SKILL_LINE_MAX_CHARS]
        if not wide:
            break
        worst = min(wide, key=score)
        out.remove(worst)
        notes.append(f"skills: dropped line '{worst.category}' — too wide for ~1.3 lines and least related to the JD")
    return _ensure_required(out, state, kb, notes), notes


def required_skills(state: ResumeState, kb: KnowledgeBase, limit: int = 3) -> list[str]:
    """Inventory entries a must-have keyword names outright ("Python" for "Proficiency in Python")."""
    from .selection import _norm
    must = [_norm(k.term) for k in state["analysis"].top_keywords if k.importance == "must"]
    out = []
    for s in kb.skills:
        n = _norm(s.display.replace("(Familiar)", ""))
        if len(n) >= 3 and s.display not in out and any(
                n == k or re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", k) for k in must):
            out.append(s.display)
    return out[:limit]


def _ensure_required(lines: list[SkillLine], state: ResumeState, kb: KnowledgeBase, notes: list[str]) -> list[SkillLine]:
    """A required skill the candidate has must be visible in Skills, not only in project headers. DataCore run, 4 Nov:
    a Front-End line took the slot that held Python, so 'Proficiency in Python required' had no Skills entry."""
    if not lines:
        return lines
    cat_of: dict[str, str] = {}
    for sk in kb.skills:
        cat_of.setdefault(sk.display, sk.category)
    forced = []
    for e in required_skills(state, kb):
        if any(e in l.entries for l in lines):
            continue
        from .selection import _probes
        same = [l for l in lines if any(cat_of.get(x) == cat_of.get(e) for x in l.entries)]
        # else a line about the same thing ("Embeddings" → the line with sentence-transformers), else the roomiest
        near = [l for l in lines if any(mentions(e, x) or mentions(x.replace("(Familiar)", ""), e)
                                        for x in [l.category, *l.entries])]
        line = (same or near or [min(lines, key=skill_line_chars)])[0]
        # own category: lead the line; elsewhere: right after the line's first entry, so the line keeps its theme
        line.entries.insert(0 if same else min(1, len(line.entries)), e)
        forced.append(e)
        while skill_line_chars(line) > SKILL_LINE_MAX_CHARS and len(line.entries) > SKILL_MIN_ENTRIES:
            drop = next((x for x in reversed(line.entries) if x not in forced), None)
            if drop is None:
                break
            line.entries.remove(drop)
        notes.append(f"skills: added required '{e}' to '{line.category}' (the JD asks for it)")
    return lines


def write_skills(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    lines, problems = [], []
    try:
        draft: SkillsDraft = _call(config, "fast", SkillsDraft,
                                   prompts.skills_messages(state["analysis"], state["selected"], kb))
    except RuntimeError as e:   # falls through to the inventory-order fallback below
        log.warning("skills model failed; using inventory order: %s", e)
        draft = SkillsDraft(lines=[], coursework=[])
        problems.append(f"skills: model failed — filled from the skills inventory ({e})")
    by_name = {s.name.lower(): s for s in kb.skills}
    by_name.update({s.display.lower(): s for s in kb.skills})
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
    lines, fit_notes = fit_skills(lines, state, kb)
    problems += fit_notes

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
    by_id = {x.id: x for x in kb.achievements}
    gd, eyrc = by_id["groundingdino_pr"], by_id["eyrc"]
    # the JD's own words count, not just the 10 keywords (DataCore asked for "open-source contributions" and the
    # analyzer didn't pick it as a keyword, so the line was skipped with blank space left on the page)
    from .selection import _norm
    def has(t: str, s: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(_norm(t))}(?![a-z0-9])", s))
    kw_text = _norm(" ".join([k.term for k in a.top_keywords] + [a.archetype, a.domain]))
    jd_text = _norm(state["job"].text)
    # in the raw JD only unambiguous terms count ("send your CV" is not computer vision)
    jd_asks_gd = any(has(t, kw_text) or (len(t) >= 4 and has(t, jd_text)) for t in gd.include_for)
    extras = []
    if jd_asks_gd:
        extras.append(gd)                          # asked for: ahead of the Amazon line too
    if "amazon_er" not in selected:
        extras.append(by_id["amazon_ml_challenge"])
    # kit §8: the PR is "the default 3rd achievement when space allows"; robotics JDs lead with eYRC
    extras += [eyrc, gd] if a.archetype in {"robotics_cpp", "vio_slam"} else [gd, eyrc]
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
