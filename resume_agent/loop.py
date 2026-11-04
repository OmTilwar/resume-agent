"""Phase 2: self-correction loops.

content:  validate ⇄ fix (≤ MAX_FIX rounds) → fallback (bank bullets for anything still hard-failing)
layout:   compile → layout_check ⇄ layout_fix (trim / fix orphans / fill, ≤ MAX_LAYOUT rounds)
"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from . import prompts
from .knowledge import Context, KnowledgeBase, get_kb
from .knowledge.checks import check_bullet
from .layout import measure
from .nodes import _call
from .state import FinalBullet, FixBatch, ResumeState
from .validate import (Issue, Slot, deterministic_issues, judge_issues, length_issue, near_duplicate, slots,
                       visible_len)

log = logging.getLogger("resume_agent")
MAX_FIX = 2
MAX_LAYOUT = 4
MAX_TRIM = 5


# ---- state helpers ---------------------------------------------------------

def _ctx(state: ResumeState) -> Context:
    return Context(archetype=state["analysis"].archetype, company=state["job"].company)


def _copy_blocks(state: ResumeState):
    return ([p.model_copy(deep=True) for p in state["projects"]], state["experience"].model_copy(deep=True))


def _put(projects, experience, slot: Slot, bullet: FinalBullet) -> None:
    if slot.section == "experience":
        experience.bullets[slot.index] = bullet
    else:
        next(p for p in projects if p.project_id == slot.section).bullets[slot.index] = bullet


def _bank(section: str, state: ResumeState, kb: KnowledgeBase):
    if section == "experience":
        groups = state["exp_groups"]
        return [b for g in kb.experience.groups if g.id in groups for b in g.bullets] + \
               [b for g in kb.experience.groups if g.id not in groups for b in g.bullets]
    return kb.allowed_bullets(section, _ctx(state))


def _on_page(state: ResumeState) -> set[str]:
    return {b.text for p in state["projects"] for b in p.bullets} | {b.text for b in state["experience"].bullets}


def _section_texts(state: ResumeState, section: str, exclude: str | None = None) -> list[str]:
    kb = get_kb()
    return [s.bullet.text for s in slots(state, kb) if s.section == section and s.id != exclude]


def _fresh(candidate: str, existing: list[str]) -> bool:
    return not any(near_duplicate(candidate, e) for e in existing)


def _vet_one(text: str, fact_ids: list[str], slot: Slot, state: ResumeState, kb: KnowledgeBase) -> list[str]:
    """Problems with a candidate replacement (empty = acceptable)."""
    if any(f not in slot.facts for f in fact_ids):
        return ["cites facts not allowed here"]
    cited = [slot.facts[f] for f in fact_ids]
    probs = [str(v) for v in check_bullet(text, slot.id, cited, _ctx(state), kb, kb.projects.get(slot.section))]
    li = length_issue(text, kb, slot.id)
    if li and li.severity == "hard":
        probs.append(str(li))
    return probs


def _fix_items(targets: dict[str, list[str]], state: ResumeState, kb: KnowledgeBase) -> list[dict]:
    by_id = {s.id: s for s in slots(state, kb)}
    items = []
    for tid, problems in targets.items():
        s = by_id[tid]
        p = kb.projects.get(s.section)
        items.append({
            "id": tid,
            "text": s.bullet.text,
            "current_chars": visible_len(s.bullet.text),
            "problems": problems,
            "facts": [{"id": f.id, "claim": f.claim, "values": f.values} for f in s.facts.values()],
            "reference_bullets": [b.text for b in _bank(s.section, state, kb)][:4],
            "project_rules": (p.rules + [f"never say: {', '.join(p.banned_terms)}"] if p and p.banned_terms
                              else (p.rules if p else [])),
            "target_chars": "<= 105, or 180-215",
        })
    return items


def _apply_fixes(targets: dict[str, list[str]], state: ResumeState, config: RunnableConfig, kb: KnowledgeBase,
                 notes: list[str]):
    """One fixer call for all targets; keep only fixes that pass the deterministic checks."""
    projects, experience = _copy_blocks(state)
    by_id = {s.id: s for s in slots(state, kb)}
    try:
        batch: FixBatch = _call(config, "fixer", FixBatch, prompts.fix_messages(_fix_items(targets, state, kb)))
        fixes = {f.id: f for f in batch.fixes}
    except RuntimeError as e:
        notes.append(f"fixer unavailable this round: {e}")
        fixes = {}
    for tid in targets:
        f = fixes.get(tid)
        if f is None:
            continue
        probs = _vet_one(f.text, f.fact_ids, by_id[tid], state, kb)
        if probs:
            notes.append(f"fix for {tid} rejected: {probs[0]}")
            continue
        _put(projects, experience, by_id[tid], FinalBullet(text=f.text.strip(), fact_ids=f.fact_ids, origin="llm"))
    return projects, experience


# ---- content loop ------------------------------------------------------------

def validate(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    notes = list(state.get("notes", []))
    issues = deterministic_issues(state, kb)
    approved = set(state.get("approved", []))
    hard_targets = {i.target for i in issues if i.severity == "hard"}
    try:
        # only judge bullets that already pass the deterministic checks
        j_issues, ok = judge_issues(state, kb, lambda r, s, m: _call(config, r, s, m), approved, hard_targets)
        issues += j_issues
        approved |= ok
    except RuntimeError as e:
        notes.append(f"judge unavailable — bullets not wording-checked this round: {e}")
    return {"issues": issues, "approved": sorted(approved), "notes": notes,
            "fix_round": state.get("fix_round", 0)}


def route_after_validate(state: ResumeState) -> str:
    issues = state.get("issues", [])
    if not issues:
        return "assemble"
    if state.get("fix_round", 0) < MAX_FIX:
        return "fix"
    return "fallback" if any(i.severity == "hard" for i in issues) else "assemble"


def fix(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    notes = list(state.get("notes", []))
    targets: dict[str, list[str]] = {}
    for i in state["issues"]:
        targets.setdefault(i.target, []).append(f"{i.rule}: {i.message}")
    projects, experience = _apply_fixes(targets, state, config, kb, notes)
    return {"projects": projects, "experience": experience, "notes": notes,
            "fix_round": state.get("fix_round", 0) + 1}


def fallback(state: ResumeState, config: RunnableConfig) -> ResumeState:
    """Rounds exhausted: swap every still-hard-failing bullet for an unused verified bank bullet."""
    kb = get_kb()
    notes = list(state.get("notes", []))
    projects, experience = _copy_blocks(state)
    by_id = {s.id: s for s in slots(state, kb)}
    used = _on_page(state)
    for tid in sorted({i.target for i in state["issues"] if i.severity == "hard"}):
        s = by_id[tid]
        others = [t for t in _section_texts(state, s.section, exclude=tid)]
        others += [b.text for b in (experience.bullets if s.section == 'experience' else
                   next(p for p in projects if p.project_id == s.section).bullets)]
        others = [t for t in others if t != s.bullet.text]
        bank = [b for b in _bank(s.section, state, kb) if b.text not in used and _fresh(b.text, others)
                and not (s.index < 3 and kb.cites_last_only(s.section, b.fact_ids))]
        bank.sort(key=lambda b: bool(length_issue(b.text, kb, tid)))   # well-sized first
        if bank:
            b = bank[0]
            used.add(b.text)
            _put(projects, experience, s, FinalBullet(text=b.text, fact_ids=b.fact_ids, origin="bank"))
            notes.append(f"{tid}: replaced with verified bank bullet after {MAX_FIX} failed fix rounds")
        else:
            block = experience if s.section == "experience" else next(p for p in projects if p.project_id == s.section)
            if len(block.bullets) > 3 or (s.section == "experience" and len(block.bullets) > 2):
                block.bullets[s.index] = None
                notes.append(f"{tid}: dropped — still failing and no non-duplicate bank bullet left")
            else:
                notes.append(f"{tid}: still failing and no unused bank bullet — REVIEW MANUALLY")
    for p in projects:
        p.bullets = [b for b in p.bullets if b is not None]
    experience.bullets = [b for b in experience.bullets if b is not None]
    soft = [i for i in state["issues"] if i.severity == "soft"]
    return {"projects": projects, "experience": experience, "notes": notes, "issues": soft}


# ---- layout loop -------------------------------------------------------------

def layout_targets(state: ResumeState, kb: KnowledgeBase) -> dict[str, str]:
    t = {s.id: s.bullet.text for s in slots(state, kb)}
    for line in state["skills"]:
        t[f"skills:{line.category}"] = f"{line.category}: {', '.join(line.entries)}"
    return t


def layout_check(state: ResumeState, config: RunnableConfig) -> ResumeState:
    if not state.get("pdf_path"):
        return {"layout_action": "done", "layout": None}
    kb = get_kb()
    L = measure(state["pdf_path"], layout_targets(state, kb))
    rnd = state.get("layout_round", 0)
    fresh_orphans = [o for o in L.orphans if o not in state.get("orphan_tried", [])]
    if L.pages > 1:
        action = "trim" if state.get("trim_level", 0) < MAX_TRIM or state.get("last_fill") else "done"
    elif L.free_lines >= 3 and not state.get("no_fill") and rnd < MAX_LAYOUT:
        action = "fill"          # before orphans: adding content reflows lines anyway
    elif fresh_orphans and rnd < MAX_LAYOUT:
        action = "orphans"       # each orphan gets one rewrite attempt
    else:
        action = "done"
    return {"layout": L, "layout_action": action}


def route_after_layout(state: ResumeState) -> str:
    return "report" if state.get("layout_action") == "done" else "layout_fix"


def route_after_layout_fix(state: ResumeState) -> str:
    # rewritten bullets must pass the content checks (and judge) again
    return "validate" if state.get("layout_action") == "orphans" else "assemble"


def layout_fix(state: ResumeState, config: RunnableConfig) -> ResumeState:
    kb = get_kb()
    action = state["layout_action"]
    notes = list(state.get("notes", []))
    upd: ResumeState = {"layout_round": state.get("layout_round", 0) + 1}
    projects, experience = _copy_blocks(state)

    if action == "trim":
        last = state.get("last_fill")
        if last:                                   # undo the fill that overflowed, and stop filling
            if last == "achievement":
                upd["max_achievements"] = state.get("max_achievements", 3) - 1
            else:
                next(p for p in projects if p.project_id == last).bullets.pop()
            notes.append(f"layout: undid fill ({last}) — it overflowed the page")
            upd.update(projects=projects, last_fill=None, no_fill=True, notes=notes)
            return upd
        # kit §11 trim order. Skip levels that would change nothing and apply the first one that does,
        # so a single round always makes progress. Trims don't count against MAX_LAYOUT (own cap: MAX_TRIM).
        from .nodes import pick_achievements
        upd["layout_round"] = state.get("layout_round", 0)
        shown_ach = min(state.get("max_achievements", 3), len(pick_achievements(state, kb)))
        level = state.get("trim_level", 0)
        while level < MAX_TRIM:
            level += 1
            if level == 1 and shown_ach > 2:
                upd["max_achievements"] = 2
                notes.append("layout: trimmed achievements to 2")
                break
            if level == 2 and any(len(p.bullets) > 3 for p in projects):
                p = next(p for p in reversed(projects) if len(p.bullets) > 3)
                p.bullets.pop()
                notes.append(f"layout: dropped 4th bullet of {p.name}")
                break
            if level == 3 and len(state["skills"]) > 4:
                upd["skills"] = state["skills"][:4]
                notes.append("layout: cut Skills to 4 lines")
                break
            if level == 4 and len(experience.bullets) > 2:
                experience.bullets = experience.bullets[:2]
                notes.append("layout: cut Experience to 2 bullets")
                break
            if level == 5:
                longest = sorted(slots(state, kb), key=lambda s: -visible_len(s.bullet.text))[:2]
                targets = {s.id: ["page is over by a few lines: shorten to <= 105 characters, keep the main metric"]
                           for s in longest}
                projects, experience = _apply_fixes(targets, state, config, kb, notes)
                notes.append("layout: asked the fixer to shorten the 2 longest bullets")
                upd["layout_action"] = "orphans"   # route through validate again
                upd["fix_round"] = MAX_FIX
                break
        upd.update(projects=projects, experience=experience, trim_level=level, notes=notes, no_fill=True)
        return upd

    if action == "orphans":
        L = state["layout"]
        tried = list(state.get("orphan_tried", []))
        skills = [l.model_copy(deep=True) for l in state["skills"]]
        bullet_targets = {}
        for tid in [o for o in L.orphans if o not in tried]:
            tried.append(tid)
            if tid.startswith("skills:"):
                line = next((l for l in skills if f"skills:{l.category}" == tid), None)
                if line and len(line.entries) > 8:
                    dropped = line.entries.pop()
                    notes.append(f"layout: dropped '{dropped}' so {line.category} doesn't end on an orphan")
            else:
                bullet_targets[tid] = [
                    "when printed, its last line holds only 1-2 words. Rewrite it ~15 characters shorter "
                    "(or ~35 longer), staying <= 105 or within 180-215 characters, without losing the main metric."]
        if bullet_targets:
            projects, experience = _apply_fixes(bullet_targets, {**state, "skills": skills}, config, kb, notes)
        upd.update(projects=projects, experience=experience, skills=skills, notes=notes, orphan_tried=tried,
                   fix_round=MAX_FIX)   # one judge pass, then fallback — no new fix loop
        return upd

    if action == "fill":                           # kit §11 fill order
        used = _on_page(state)
        for p in projects:
            proj = kb.get_project(p.project_id)
            if len(p.bullets) >= min(4, proj.max_bullets or 4):
                continue
            cands = [b for b in kb.allowed_bullets(p.project_id, _ctx(state))
                     if b.text not in used and not length_issue(b.text, kb, "x")
                     and _fresh(b.text, [x.text for x in p.bullets])]
            if cands:
                p.bullets.append(FinalBullet(text=cands[0].text, fact_ids=cands[0].fact_ids, origin="bank"))
                notes.append(f"layout: filled blank space with a 4th {p.name} bullet")
                upd.update(projects=projects, last_fill=p.project_id, notes=notes)
                return upd
        from .nodes import pick_achievements
        if state.get("max_achievements", 3) < len(pick_achievements(state, kb)):
            upd.update(max_achievements=state.get("max_achievements", 3) + 1, last_fill="achievement",
                       notes=notes + ["layout: filled blank space with another achievement"])
            return upd
        upd.update(no_fill=True, notes=notes + ["layout: blank space left — nothing verified left to add"])
        return upd

    return upd
