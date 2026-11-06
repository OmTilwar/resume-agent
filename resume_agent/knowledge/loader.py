"""Load kb/*.yaml into a validated KnowledgeBase and cross-check references."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .models import (
    Achievement, Archetype, Experience, KnowledgeBase, Profile, Project,
    RetiredProject, Rules, Skill,
)

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "kb"


class KBError(ValueError):
    pass


def _read(path: Path):
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_skills(raw: dict[str, list[str]]) -> list[Skill]:
    skills: list[Skill] = []
    for category, items in raw.items():
        for item in items:
            name = item.strip()
            familiar = name.endswith("(Familiar)")
            if familiar:
                name = name[: -len("(Familiar)")].strip()
            skills.append(Skill(name=name, category=category, familiar=familiar))
    return skills


def load_kb(kb_dir: Path | str = DEFAULT_KB_DIR) -> KnowledgeBase:
    kb_dir = Path(kb_dir)
    meta = _read(kb_dir / "meta.yaml")
    projects: dict[str, Project] = {}
    for f in sorted((kb_dir / "projects").glob("*.yaml")) + sorted((kb_dir / "work").glob("*.yaml")):
        p = Project.model_validate(_read(f))
        if p.id in projects:
            raise KBError(f"duplicate project id {p.id!r}")
        projects[p.id] = p

    archetypes = {a["id"]: Archetype.model_validate(a) for a in _read(kb_dir / "archetypes.yaml")}

    kb = KnowledgeBase(
        kit_version=str(meta["kit_version"]),
        kit_updated=str(meta["kit_updated"]),
        profile=Profile.model_validate(_read(kb_dir / "profile.yaml")),
        experience=Experience.model_validate(_read(kb_dir / "experience.yaml")),
        skills=_parse_skills(_read(kb_dir / "skills.yaml")),
        projects=projects,
        retired=[RetiredProject.model_validate(r) for r in _read(kb_dir / "retired.yaml")],
        achievements=[Achievement.model_validate(a) for a in _read(kb_dir / "achievements.yaml")],
        archetypes=archetypes,
        rules=Rules.model_validate(_read(kb_dir / "rules.yaml")),
    )
    errors = validate_references(kb)
    if errors:
        raise KBError("knowledge base failed validation:\n  - " + "\n  - ".join(errors))
    return kb


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return load_kb()


def validate_references(kb: KnowledgeBase) -> list[str]:
    """Structural integrity: every reference points at something real."""
    errs: list[str] = []
    retired_ids = {r.id for r in kb.retired}
    group_ids = {g.id for g in kb.experience.groups}

    for p in kb.projects.values():
        fact_ids = [f.id for f in p.facts]
        if len(fact_ids) != len(set(fact_ids)):
            errs.append(f"{p.id}: duplicate fact ids")
        for b in p.bullets:
            for fid in b.fact_ids:
                if fid not in fact_ids:
                    errs.append(f"{p.id}/{b.id}: cites unknown fact {fid!r}")
                elif p.fact(fid).status.value == "banned":
                    errs.append(f"{p.id}/{b.id}: bank bullet cites BANNED fact {fid!r}")
        for t in p.titles:
            extra = [s for s in t.stack if s not in p.stack]
            if extra:
                errs.append(f"{p.id}: title {t.subtitle!r} lists tools not in project stack: {extra}")
        if p.id in retired_ids:
            errs.append(f"{p.id}: is both active and retired")

    exp_fact_ids = {f.id for f in kb.experience.facts}
    for g in kb.experience.groups:
        for b in g.bullets:
            for fid in b.fact_ids:
                if fid not in exp_fact_ids:
                    errs.append(f"experience/{b.id}: cites unknown fact {fid!r}")

    for a in kb.archetypes.values():
        for pid in a.flat_projects():
            if pid not in kb.projects:
                errs.append(f"archetype {a.id}: unknown project {pid!r}")
        for g in a.exp_groups:
            if g not in group_ids:
                errs.append(f"archetype {a.id}: unknown experience group {g!r}")

    for ach in kb.achievements:
        if ach.drop_if_project and ach.drop_if_project not in kb.projects:
            errs.append(f"achievement {ach.id}: unknown drop_if_project {ach.drop_if_project!r}")
    return errs
