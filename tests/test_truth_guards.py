"""The guarantees Phase 0 exists for: banned / retired / restricted things are unreachable."""
import pytest

from resume_agent.knowledge import Context, FactStatus
from resume_agent.knowledge.checks import check_bullet, check_phrases
from resume_agent.knowledge.numbers import extract_numbers, normalize_values

from .conftest import CONTEXTS


# ---- ⛔ banned facts -------------------------------------------------------

@pytest.mark.parametrize("ctx_name", CONTEXTS)
def test_banned_facts_never_allowed(kb, ctx_name):
    ctx = Context(**CONTEXTS[ctx_name])
    for pid in kb.projects:
        allowed = kb.allowed_facts(pid, ctx)
        assert all(f.status != FactStatus.BANNED for f in allowed), pid
        for b in kb.allowed_bullets(pid, ctx):
            for fid in b.fact_ids:
                assert kb.projects[pid].fact(fid).status != FactStatus.BANNED


def test_known_banned_facts_are_marked(kb):
    """Spot-check the ⛔ items called out in kit §7 / App. D."""
    expected = {
        "finsight": {"fs_ragas"},
        "medvision": {"mv_unet_synthetic", "mv_rag", "mv_powerbi", "mv_val_split"},
        "photoretouch": {"pr_fid"},
        "id_vlm": {"idv_strict", "idv_ragas"},
    }
    for pid, ids in expected.items():
        assert ids <= {f.id for f in kb.banned_facts(pid)}, pid


def test_banned_values_absent_from_allowed_bullets(kb):
    """A banned number must not sneak in via another bullet of the same project."""
    for pid, p in kb.projects.items():
        banned_vals = set().union(*(normalize_values(f.values) for f in kb.banned_facts(pid))) if kb.banned_facts(pid) else set()
        ok_vals = set().union(*(normalize_values(f.values) for f in p.facts if f.status != FactStatus.BANNED))
        for b in p.bullets:
            leaked = (extract_numbers(b.text) & banned_vals) - ok_vals
            assert not leaked, (pid, b.id, leaked)


# ---- retired projects (App. F) -------------------------------------------

def test_retired_projects_unreachable(kb):
    selectable = {p.id for p in kb.selectable_projects()}
    for r in kb.retired:
        assert r.id not in selectable
        assert r.id not in kb.projects
        with pytest.raises(ValueError, match="retired"):
            kb.get_project(r.id)
        for a in kb.archetypes.values():
            assert r.id not in a.flat_projects()


# ---- context-restricted facts (kit §2.5) --------------------------------

@pytest.mark.parametrize("pid,fid,ok_ctx", [
    ("medvision", "mv_sens_spec", "healthcare"),
    ("authnet", "an_map_rounded", "entrupy"),
    ("voxpipeline", "vp_tests_full", "sarvam"),
])
def test_context_restricted_facts(kb, pid, fid, ok_ctx):
    fact = kb.projects[pid].fact(fid)
    for name, c in CONTEXTS.items():
        assert kb.fact_allowed(fact, Context(**c)) == (name == ok_ctx), (fid, name)


def test_healthcare_bullet_variant_only_in_healthcare(kb):
    ids_default = {b.id for b in kb.allowed_bullets("medvision", Context())}
    ids_health = {b.id for b in kb.allowed_bullets("medvision", Context(archetype="healthcare"))}
    assert "mv_b1_health" not in ids_default
    assert "mv_b1_health" in ids_health


# ---- small eval sets (kit §2.4.4) ---------------------------------------

def test_no_usable_metric_from_tiny_eval(kb):
    for p in kb.projects.values():
        for f in p.facts:
            if f.eval_size is not None and f.eval_size < kb.rules.min_eval_size:
                assert f.status == FactStatus.BANNED, (p.id, f.id, f.eval_size)


# ---- Familiar skills (kit §2.2.3) ---------------------------------------

def test_familiar_skills_never_in_project_stacks(kb):
    familiar = {s.name.lower() for s in kb.familiar_skills()}
    assert {"ragas", "kubernetes", "power bi", "django"} <= familiar
    assert "langchain" not in familiar   # used for real in resume_agent (kit v2.1)
    for p in kb.projects.values():
        clash = familiar & {s.lower() for s in p.stack}
        assert not clash, (p.id, clash)


def test_familiar_skills_never_in_bank_bullets(kb):
    for s in kb.familiar_skills():
        for p in kb.projects.values():
            for b in p.bullets:
                assert s.name.lower() not in b.text.lower().split(), (p.id, b.id, s.name)


# ---- the bank itself passes every bullet check --------------------------

def _all_bank_bullets(kb):
    for pid, p in kb.projects.items():
        for b in p.bullets:
            yield f"{pid}/{b.id}", b, [p.fact(f) for f in b.fact_ids], p
    for g in kb.experience.groups:
        for b in g.bullets:
            yield f"experience/{b.id}", b, [kb.experience.fact(f) for f in b.fact_ids], None


def _ctx_for(kb, bullet_facts):
    """Pick the context the bullet was written for (restricted facts need theirs)."""
    for c in CONTEXTS.values():
        ctx = Context(**c)
        if all(kb.fact_allowed(f, ctx) for f in bullet_facts):
            return ctx
    return Context()


def test_bank_bullets_pass_all_checks(kb):
    failures = []
    for where, b, facts, project in _all_bank_bullets(kb):
        failures += check_bullet(b.text, where, facts, _ctx_for(kb, facts), kb, project)
    assert not failures, "\n" + "\n".join(map(str, failures))


# ---- the checks catch what they should ----------------------------------

@pytest.mark.parametrize("text,rule", [
    ("Integrated Ragas evaluation scoring 0.92 faithfulness.", "banned-term"),
    ("Built an agent with LangGraph.", "banned-term"),
    ("Developed a novel retrieval algorithm.", "banned-phrase"),
    ("Worked on a RAG chatbot.", "banned-phrase"),
    ("Built OCR analogous to land-record processing.", "analogy-clause"),
    ("B.Tech from IIT Nagpur.", "institution"),
    ("Wrote operational runbooks for the pipeline.", "unconfirmed-claim"),
])
def test_phrase_checks_fire(kb, text, rule):
    assert rule in {v.rule for v in check_phrases(text, "t", kb)}


def test_team_project_lead_verb_rejected(kb):
    p = kb.projects["amazon_er"]
    v = check_phrases("Spearheaded an entity-resolution pipeline.", "t", kb, p)
    assert "lead-verb" in {x.rule for x in v}


def test_fabricated_number_rejected(kb):
    p = kb.projects["finsight"]
    v = check_bullet("Raised Top-1 accuracy from 60.53% to 72.10% on 762 queries.", "t",
                     [p.fact("fs_retrieval_bench")], Context(), kb, p)
    assert [x.message for x in v if x.rule == "unbacked-number"] == ["number '72.1' not in any cited allowed fact"]


def test_citing_banned_fact_rejected(kb):
    p = kb.projects["medvision"]
    v = check_bullet("Reached 0.83 Dice.", "t", [p.fact("mv_unet_synthetic")], Context(), kb, p)
    assert {"banned-fact", "small-eval"} & {x.rule for x in v}
    assert "unbacked-number" in {x.rule for x in v}


def test_restricted_fact_outside_context_rejected(kb):
    p = kb.projects["medvision"]
    v = check_bullet("Reached 92.3% sensitivity.", "t", [p.fact("mv_sens_spec")],
                     Context(archetype="backend_sde"), kb, p)
    assert "context-restricted" in {x.rule for x in v}


def test_points_vs_percent(kb):
    p = kb.projects["finsight"]
    v = check_bullet("Raised Top-1 from 60.53% to 66.45% (+5.92%) on 762 queries.", "t",
                     [p.fact("fs_retrieval_bench")], Context(), kb, p)
    assert "pts-vs-percent" in {x.rule for x in v}
