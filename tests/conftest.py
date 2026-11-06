from pathlib import Path

import pytest

from resume_agent.knowledge import load_kb

# kb/work/*.yaml holds private client facts and is git-ignored (kb/work/README.md): on a public clone the
# tests that need the Design Stencil entry are skipped; locally they run.
HAS_DESIGN_STENCIL = (Path(__file__).resolve().parents[1] / "kb" / "work" / "design_stencil.yaml").exists()
needs_design_stencil = pytest.mark.skipif(not HAS_DESIGN_STENCIL,
                                          reason="private kb/work/design_stencil.yaml not present (git-ignored)")


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
