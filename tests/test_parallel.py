"""The three writers (and the 4 project calls inside write_projects) run concurrently."""
import threading
import time

from resume_agent.graph import build_graph
from resume_agent.state import JobPosting, ProjectDraft, ExperienceDraft, SkillsDraft

from . import fakes

SLEEP = 0.4


def test_writers_run_in_parallel(tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    def slow_factory(role, schema):
        inner = fakes.fake_factory(role, schema)
        if schema not in (ProjectDraft, ExperienceDraft, SkillsDraft):
            return inner

        class Slow:
            def invoke(self, messages, config=None):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(SLEEP)
                with lock:
                    active[0] -= 1
                return inner.invoke(messages)
        return Slow()

    t0 = time.time()
    final = build_graph().invoke(
        {"job": JobPosting(company="Acme AI", role="LLM Intern", text="x"), "out_dir": str(tmp_path)},
        config={"configurable": {"llm_factory": slow_factory, "max_concurrency": 4}, "recursion_limit": 80})
    elapsed = time.time() - t0

    assert peak[0] >= 4, f"only {peak[0]} writer calls overlapped"
    writer_time_if_sequential = 6 * SLEEP
    assert elapsed < writer_time_if_sequential + 3.0      # generous: compile + layout loop also take time
    # order preserved and nothing duplicated by the parallel merge
    assert [p.project_id for p in final["projects"]] == final["selected"]
    drafts = [v for v in final["violations"] if v.startswith("first draft")]
    assert len(drafts) == len(set(drafts)), "violations duplicated by the parallel merge"


def test_concurrency_cap_respected(tmp_path):
    active, peak, lock = [0], [0], threading.Lock()

    def slow_factory(role, schema):
        inner = fakes.fake_factory(role, schema)

        class Slow:
            def invoke(self, messages, config=None):
                if schema is ProjectDraft:
                    with lock:
                        active[0] += 1
                        peak[0] = max(peak[0], active[0])
                    time.sleep(0.2)
                    with lock:
                        active[0] -= 1
                return inner.invoke(messages)
        return Slow()

    build_graph().invoke(
        {"job": JobPosting(company="Acme AI", role="LLM Intern", text="x"), "out_dir": str(tmp_path)},
        config={"configurable": {"llm_factory": slow_factory, "max_concurrency": 2}, "recursion_limit": 80})
    assert peak[0] == 2


def test_writer_returning_nothing_does_not_sink_the_run(tmp_path):
    """Reproduces the 4 Nov live crash: the writer returned no structured output for every project."""
    def broken_writer(role, schema):
        inner = fakes.fake_factory(role, schema)

        class R:
            def invoke(self, messages, config=None):
                return None if schema in (ProjectDraft, ExperienceDraft, SkillsDraft) else inner.invoke(messages)
        return R()

    final = build_graph().invoke(
        {"job": JobPosting(company="Acme AI", role="LLM Intern", text="x"), "out_dir": str(tmp_path)},
        config={"configurable": {"llm_factory": broken_writer}, "recursion_limit": 80})
    assert len(final["projects"]) == 4
    # every project still has 3+ bullets (kit bullets, possibly shortened later by the fixer)
    assert all(len(p.bullets) >= 3 for p in final["projects"])
    assert any("writer model failed" in v for v in final["violations"])
    assert final["tex_path"] and len(final["skills"]) >= 4


def test_retry_asks_factory_for_next_attempt(tmp_path):
    """_call passes attempt=2 on retry so the real factory can switch models."""
    seen = []

    def flaky(role, schema, attempt=1):
        seen.append((role, schema.__name__, attempt))
        inner = fakes.fake_factory(role, schema)

        class R:
            def invoke(self, messages, config=None):
                if schema is SkillsDraft and attempt == 1:
                    return None                       # first attempt "runs away"
                return inner.invoke(messages)
        return R()

    final = build_graph().invoke(
        {"job": JobPosting(company="Acme AI", role="LLM Intern", text="x"), "out_dir": str(tmp_path)},
        config={"configurable": {"llm_factory": flaky}, "recursion_limit": 80})
    assert ("fast", "SkillsDraft", 2) in seen
    assert not any("skills: model failed" in v for v in final["violations"])   # attempt 2 succeeded
