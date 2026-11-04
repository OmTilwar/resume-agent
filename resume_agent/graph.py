"""Resume graph.

ingest → analyze → select ─┬─ write_projects     (its 4 projects run concurrently)
                           ├─ write_experience   } the three writers run in parallel;
                           └─ write_skills       } validate waits for all of them
  → validate ⇄ fix                (content loop, ≤ loop.MAX_FIX rounds)
  → [fallback]                    (bank bullets for anything still failing)
  → assemble → compile → layout_check ⇄ layout_fix   (page loop, kit §11)
  → report
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import loop, nodes
from .state import ResumeState

LINEAR = [
    ("ingest", nodes.ingest),
    ("analyze", nodes.analyze),
    ("select", nodes.select),
]
WRITERS = [
    ("write_projects", nodes.write_projects),
    ("write_experience", nodes.write_experience),
    ("write_skills", nodes.write_skills),
]


def build_graph():
    g = StateGraph(ResumeState)
    for name, fn in LINEAR + WRITERS:
        g.add_node(name, fn)
    for name, fn in [("validate", loop.validate), ("fix", loop.fix), ("fallback", loop.fallback),
                     ("assemble", nodes.assemble), ("compile", nodes.compile_pdf),
                     ("layout_check", loop.layout_check), ("layout_fix", loop.layout_fix),
                     ("report", nodes.report)]:
        g.add_node(name, fn)

    g.add_edge(START, LINEAR[0][0])
    for (a, _), (b, _) in zip(LINEAR, LINEAR[1:]):
        g.add_edge(a, b)
    # fan out: the three writers don't depend on each other
    for name, _ in WRITERS:
        g.add_edge("select", name)
    # fan in: validate runs once, after all three have finished
    g.add_edge([name for name, _ in WRITERS], "validate")

    g.add_conditional_edges("validate", loop.route_after_validate,
                            {"fix": "fix", "fallback": "fallback", "assemble": "assemble"})
    g.add_edge("fix", "validate")
    g.add_edge("fallback", "assemble")
    g.add_edge("assemble", "compile")
    g.add_edge("compile", "layout_check")
    g.add_conditional_edges("layout_check", loop.route_after_layout,
                            {"layout_fix": "layout_fix", "report": "report"})
    g.add_conditional_edges("layout_fix", loop.route_after_layout_fix,
                            {"validate": "validate", "assemble": "assemble"})
    g.add_edge("report", END)
    return g.compile()
