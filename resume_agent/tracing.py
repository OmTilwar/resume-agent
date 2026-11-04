"""Optional LangSmith tracing.

Turn it on by adding to resume-agent/.env:

    LANGSMITH_API_KEY=lsv2_...
    # optional:
    LANGSMITH_PROJECT=resume-agent
    LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # only for EU-region accounts

Tracing is opt-in per process: the CLI enables it when a key is present (disable
with --no-trace); benchmarks enable it only with --trace, so hundreds of judge
calls don't flood the dashboard.

What gets sent: job-description text, project facts and model outputs.
Contact details are never in any prompt (the template fills them in locally).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_PROJECT = "resume-agent-local"


def configure_tracing(enable: bool = True, environment: str | None = None) -> bool:
    """Enable LangSmith for this process if a key is configured. Returns True if tracing is on.

    Two routing modes (LangSmith refuses to trace if both are set):
      - agent address: LANGSMITH_AGENT_ID (+ LANGSMITH_AGENT_ENVIRONMENT) — used when an agent id is set;
        `environment` switches it (e.g. "development" for benchmarks),
      - project name:  LANGSMITH_PROJECT — default "resume-agent-<environment>".
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
    except ImportError:
        pass
    if not enable or not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    if os.environ.get("LANGSMITH_AGENT_ID"):
        os.environ.pop("LANGSMITH_PROJECT", None)            # never both: LangSmith would trace nothing
        if environment:
            os.environ["LANGSMITH_AGENT_ENVIRONMENT"] = environment
        else:
            os.environ.setdefault("LANGSMITH_AGENT_ENVIRONMENT", "local")
    else:
        if environment:
            os.environ["LANGSMITH_PROJECT"] = f"resume-agent-{environment}"
        else:
            os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    return True


def destination() -> str:
    """Where traces go, in words."""
    if os.environ.get("LANGSMITH_AGENT_ID"):
        return (f"agent '{os.environ['LANGSMITH_AGENT_ID']}', "
                f"environment '{os.environ.get('LANGSMITH_AGENT_ENVIRONMENT', 'local')}'")
    return f"project '{os.environ.get('LANGSMITH_PROJECT', DEFAULT_PROJECT)}'"


def project_url() -> str:
    """Best-effort pointer to the traces (UI host follows the API region)."""
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    host = "https://eu.smith.langchain.com" if "eu." in endpoint else "https://smith.langchain.com"
    return f"{host} → {destination()}"


def run_config(slug: str, company: str, role: str, extra_tags: list[str] | None = None) -> dict:
    """Run name, tags and metadata so each resume is easy to find in LangSmith."""
    return {
        "run_name": f"resume:{slug}",
        "tags": ["resume-agent", f"company:{company}", *(extra_tags or [])],
        "metadata": {"company": company, "role": role, "slug": slug},
    }


def flush() -> None:
    """Make sure buffered traces are sent before the process exits."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:  # noqa: BLE001 — tracing must never break a run
        pass
