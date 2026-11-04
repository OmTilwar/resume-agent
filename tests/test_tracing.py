"""Tracing is opt-in, and every model call is named by role and nested under its graph step."""
import os

from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.base import BaseTracer

from resume_agent import tracing
from resume_agent.graph import build_graph
from resume_agent.state import JobPosting

from . import fakes


class Recorder(BaseTracer):
    """Collects runs in memory, the way LangSmith would receive them."""
    def __init__(self):
        super().__init__()
        self.runs = []

    def _persist_run(self, run):
        pass

    def _start_trace(self, run):
        super()._start_trace(run)
        self.runs.append(run)


def test_tracing_off_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr(tracing, "ENV_FILE", tmp_path / "missing.env")
    assert tracing.configure_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_on_with_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("LANGSMITH_AGENT_ID", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.setattr(tracing, "ENV_FILE", tmp_path / "missing.env")
    assert tracing.configure_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "resume-agent-local"
    assert tracing.configure_tracing(enable=False) is False          # --no-trace wins
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_model_calls_named_and_nested(tmp_path):
    rec = Recorder()
    def traced_factory(role, schema):        # a real Runnable, so it emits runs like ChatOpenAI does
        inner = fakes.fake_factory(role, schema)
        return RunnableLambda(lambda messages: inner.invoke(messages))

    cfg = {"callbacks": [rec], "recursion_limit": 80, "configurable": {"llm_factory": traced_factory},
           **tracing.run_config("acme_llm", "Acme AI", "LLM Intern")}
    build_graph().invoke({"job": JobPosting(company="Acme AI", role="LLM Intern", text="x"), "out_dir": str(tmp_path)},
                         config=cfg)
    by_id = {r.id: r for r in rec.runs}
    root = next(r for r in rec.runs if r.parent_run_id is None)
    assert root.name == "resume:acme_llm" and "company:Acme AI" in (root.tags or [])
    calls = [r for r in rec.runs if r.name.startswith(("writer:", "judge:", "fixer:", "analyzer:", "fast:"))]
    names = {r.name for r in calls}
    assert {"analyzer:JDAnalysis", "writer:ProjectDraft", "writer:ExperienceDraft", "fast:SkillsDraft",
            "judge:Verdicts"} <= names
    # the 4 parallel project calls must hang under write_projects, not float as separate traces
    project_calls = [r for r in calls if r.name == "writer:ProjectDraft"]
    assert len(project_calls) == 4
    assert all(by_id[r.parent_run_id].name == "write_projects" for r in project_calls)


def test_explicit_environment_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("LANGSMITH_AGENT_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "resume-agent-local")
    monkeypatch.setattr(tracing, "ENV_FILE", tmp_path / "missing.env")
    tracing.configure_tracing(environment="development")
    assert os.environ["LANGSMITH_PROJECT"] == "resume-agent-development"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_agent_address_never_combined_with_project(monkeypatch, tmp_path):
    """LangSmith traces nothing if both an agent address and a project are set (seen live, 4 Nov)."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "resume-agent")
    monkeypatch.setenv("LANGSMITH_PROJECT", "resume-agent-local")
    monkeypatch.setattr(tracing, "ENV_FILE", tmp_path / "missing.env")
    assert tracing.configure_tracing(environment="development") is True
    assert "LANGSMITH_PROJECT" not in os.environ
    assert os.environ["LANGSMITH_AGENT_ENVIRONMENT"] == "development"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
