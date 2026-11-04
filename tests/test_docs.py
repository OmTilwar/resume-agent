"""The architecture diagram must not drift from the real graph."""
import re
from pathlib import Path

from resume_agent.graph import build_graph

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_svg_shows_every_graph_node():
    nodes = {n for n in build_graph().get_graph().nodes if not n.startswith("__")}
    svg_titles = set(re.findall(r'<text class="t"[^>]*>([^<]+)</text>', (DOCS / "agent-graph.svg").read_text(encoding="utf-8")))
    assert nodes == svg_titles, f"missing in SVG: {nodes - svg_titles}; extra in SVG: {svg_titles - nodes}"


def test_mermaid_export_is_current():
    assert (DOCS / "graph.mmd").read_text(encoding="utf-8") == build_graph().get_graph().draw_mermaid(), \
        "graph changed — run: python scripts/export_graph.py"
