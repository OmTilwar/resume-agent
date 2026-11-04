"""Regenerate docs/graph.mmd from the compiled graph and render docs/agent-graph.png from the SVG.

    python scripts/export_graph.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resume_agent.graph import build_graph  # noqa: E402

DOCS = ROOT / "docs"


def mermaid() -> None:
    (DOCS / "graph.mmd").write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
    print("wrote docs/graph.mmd")


async def png(scale: int = 2) -> None:
    from playwright.async_api import async_playwright
    svg = (DOCS / "agent-graph.svg").read_text(encoding="utf-8")
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1100, "height": 1480}, device_scale_factor=scale)
        await page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
        await page.locator("svg").screenshot(path=str(DOCS / "agent-graph.png"))
        await b.close()
    print("wrote docs/agent-graph.png")


if __name__ == "__main__":
    mermaid()
    try:
        asyncio.run(png())
    except Exception as e:  # noqa: BLE001
        print(f"PNG skipped ({type(e).__name__}: {e}). Run: python -m playwright install chromium")
