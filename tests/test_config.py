from resume_agent.llm import load_config

JUDGE_BENCHMARKED = "nvidia/nemotron-3-ultra-550b-a55b"   # bench/results — change only after re-benchmarking


def test_model_config_shape():
    cfg = load_config()
    assert cfg["base_url"] == "https://openrouter.ai/api/v1"
    assert set(cfg["roles"]) == {"analyzer", "writer", "fixer", "judge", "fast"}
    for role, r in cfg["roles"].items():
        models = [r["model"], *r.get("fallbacks", [])]
        assert len(models) == len(set(models)) >= 2, role
        assert r["structured"] in {"json_schema", "function_calling"}
        assert "openrouter/free" not in models              # random router = non-reproducible
        assert not any(m.endswith(":batch") for m in models)  # async batch endpoints are too slow
        assert "thinkingmachines/inkling:free" not in models  # 403: partner apps only (Nov 2026)


def test_judge_is_the_benchmarked_model():
    """Benchmark numbers only describe the live pipeline if the judge chain is that model."""
    j = load_config()["roles"]["judge"]
    assert all(m.split(":")[0] == JUDGE_BENCHMARKED for m in [j["model"], *j["fallbacks"]])
