"""Role-based chat models via OpenRouter (used from Phase 1 on).

    from resume_agent.llm import get_llm, structured
    analysis = structured("analyzer", JDAnalysis).invoke(prompt)

Requires: pip install -e ".[agent]"  and  OPENROUTER_API_KEY in .env
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(CONFIG_PATH.parents[1] / ".env")
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (put it in resume-agent/.env)")
    return key


def model_chain(role: str, attempt: int = 1) -> list[str]:
    """Models to try for this attempt. A retry starts from the NEXT model in the chain: a response cut
    off at max_tokens (e.g. a repetition runaway) counts as a success for OpenRouter, so its own
    fallback never kicks in, and asking the same model again tends to repeat the failure."""
    r = load_config()["roles"][role]
    chain = [r["model"], *r.get("fallbacks", [])]
    k = (attempt - 1) % len(chain)
    return chain[k:] + chain[:k]


def get_llm(role: str, attempt: int = 1, **overrides):
    from langchain_openai import ChatOpenAI  # lazy: Phase 0 doesn't need langchain

    cfg = load_config()
    r = cfg["roles"][role]
    chain = model_chain(role, attempt)
    return ChatOpenAI(
        model=chain[0],
        base_url=cfg["base_url"],
        api_key=_api_key(),
        temperature=overrides.pop("temperature", r.get("temperature", 0.2)),
        max_tokens=overrides.pop("max_tokens", r.get("max_tokens")),   # caps runaways (see config)
        max_retries=overrides.pop("max_retries", 3),
        timeout=overrides.pop("timeout", 120),
        default_headers={"X-Title": cfg.get("app_name", "resume-agent")},
        # OpenRouter-native fallback chain: tried in order if the primary errors / is rate-limited.
        extra_body={"models": chain},
        **overrides,
    )


def structured_for_model(model: str, schema, method: str = "function_calling", temperature: float = 0.0):
    """One specific model, NO fallbacks — for benchmarks, where results must be attributable."""
    from langchain_openai import ChatOpenAI

    cfg = load_config()
    llm = ChatOpenAI(model=model, base_url=cfg["base_url"], api_key=_api_key(), temperature=temperature,
                     max_retries=2, timeout=180, default_headers={"X-Title": cfg.get("app_name", "resume-agent")},
                     extra_body={"models": [model]})
    return llm.with_structured_output(schema, method=method)


def structured(role: str, schema, attempt: int = 1, **overrides):
    """Chat model bound to a Pydantic schema, using the method the role's model supports."""
    method = load_config()["roles"][role].get("structured", "function_calling")
    return get_llm(role, attempt=attempt, **overrides).with_structured_output(schema, method=method)
