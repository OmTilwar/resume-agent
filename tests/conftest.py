import pytest

from resume_agent.knowledge import load_kb


@pytest.fixture(scope="session")
def kb():
    return load_kb()


# Contexts that exercise every context-restricted fact (kit §2.5).
CONTEXTS = {
    "default": dict(archetype=None, company=None),
    "healthcare": dict(archetype="healthcare", company="Apollo Hospitals"),
    "entrupy": dict(archetype="texture_auth_kyc", company="Entrupy"),
    "sarvam": dict(archetype="backend_sde", company="Sarvam AI"),
    "llm": dict(archetype="llm_genai_agents", company="Acme"),
}
