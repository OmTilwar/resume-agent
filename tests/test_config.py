from resume_agent.llm import load_config


def test_model_config_shape():
    cfg = load_config()
    assert cfg["base_url"] == "https://openrouter.ai/api/v1"
    assert set(cfg["roles"]) == {"analyzer", "writer", "fixer", "judge", "fast"}
    for role, r in cfg["roles"].items():
        models = [r["model"], *r.get("fallbacks", [])]
        assert all(m.endswith(":free") for m in models), f"{role}: non-free model configured"
        assert len(models) == len(set(models)) >= 2, role
        assert r["structured"] in {"json_schema", "function_calling"}
        assert "openrouter/free" not in models   # random router = non-reproducible
