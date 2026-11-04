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
            def invoke(self, messages):
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
            def invoke(self, messages):
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
