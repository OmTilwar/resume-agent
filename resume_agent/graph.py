"""Phase 1 graph: linear, no loops (Phase 2 adds validate -> fix)."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import ResumeState

STEPS = [
    ("ingest", nodes.ingest),
    ("analyze", nodes.analyze),
    ("select", nodes.select),
    ("write_projects", nodes.write_projects),
    ("write_experience", nodes.write_experience),
    ("write_skills", nodes.write_skills),
    ("assemble", nodes.assemble),
    ("compile", nodes.compile_pdf),
    ("report", nodes.report),
]


def build_graph():
    g = StateGraph(ResumeState)
    for name, fn in STEPS:
        g.add_node(name, fn)
    g.add_edge(START, STEPS[0][0])
    for (a, _), (b, _) in zip(STEPS, STEPS[1:]):
        g.add_edge(a, b)
    g.add_edge(STEPS[-1][0], END)
    return g.compile()
