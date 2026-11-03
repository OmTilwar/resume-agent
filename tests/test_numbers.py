import pytest

from resume_agent.knowledge.numbers import extract_numbers, normalize


@pytest.mark.parametrize("text,expected", [
    ("raising mAP by 8.4% on a 3,200+ instance dataset", {"8.4", "3200"}),
    ("matching 10M records to 1.7M entities", {"10M", "1.7M"}),
    ("43.3 MB → 10.9 MB, 4x smaller", {"43.3", "10.9", "4"}),
    ("~400–600 ms median", {"400", "600"}),
    ("<30ms latency on Raspberry Pi 5B + Hailo-8", {"30", "5B"}),
    ("β=4.0, (α=2", {"4", "2"}),
    ("max diff 7.45e-08", {"7.45e-08"}),
    ("p < 0.05, 0.0x → 1.5x", {"0.05", "0", "1.5"}),
])
def test_extracts_claim_numbers(text, expected):
    assert extract_numbers(text) == expected


@pytest.mark.parametrize("text", [
    "ResNet-18 backbone", "Llama-3-8B (GGUF)", "llama-3.1-8b-instant", "Qwen2-VL-2B",
    "macro F0.5", "Recall@1", "PS3.15 Annex E", "bge-reranker-v2-m3", "mDeBERTa-v3",
    "C++17/ROS2", "T200/T500 thrusters", "Top-1000 finalist", "Amazon ML Challenge 2026",
    "INT8 on H100/A100", "Deepgram Nova-2", "multilingual-e5-base", "YOLOv8", "MIDV-2020",
])
def test_ignores_identifiers_and_years(text):
    assert extract_numbers(text) == set()


def test_normalize():
    assert normalize("1,017,209") == "1017209"
    assert normalize("4.0") == "4"
    assert normalize("0.70") == "0.7"
    assert normalize("1.7 M") == "1.7M"


@pytest.mark.parametrize("text,expected", [
    ("used by 2,000+ students", {"2000"}),     # found by the seeded benchmark: was mistaken for a year
    ("grew 2000% in a year", {"2000"}),
    ("Amazon ML Challenge 2026", set()),        # still a year
])
def test_year_vs_count(text, expected):
    assert extract_numbers(text) == expected
