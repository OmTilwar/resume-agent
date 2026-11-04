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


def test_every_role_caps_output():
    """A skills call ran away to 16,384 tokens (~8 min, twice) on 4 Nov; caps turn that into a fast failure."""
    for role, r in load_config()["roles"].items():
        assert 0 < r["max_tokens"] <= 12000, role


def test_retry_moves_to_next_model():
    from resume_agent.llm import model_chain
    r = load_config()["roles"]["fast"]
    chain = [r["model"], *r["fallbacks"]]
    assert model_chain("fast", 1) == chain
    assert model_chain("fast", 2)[0] == chain[1]          # attempt 2 starts on the first fallback
    assert sorted(model_chain("fast", 2)) == sorted(chain)
