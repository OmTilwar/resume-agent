"""CLI.

    python -m resume_agent run --jd "../../JD ML and Data Science Intern- KA.pdf" --company KA --role "ML & DS Intern"
    python -m resume_agent run --jd jd.txt --company Acme --role "AI Intern" --save-to-kit
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

KIT_RESUMES = Path(__file__).resolve().parents[3] / "resumes"   # Resume_Builder/resumes


def read_jd(arg: str) -> tuple[str, str]:
    p = Path(arg)
    if p.suffix.lower() == ".pdf" and p.exists():
        from pypdf import PdfReader
        text = "\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages).strip()
        if len(text) < 200:
            sys.exit(f"Could not extract text from {p} (scanned PDF?). Paste the JD into a .txt file instead.")
        return text, "pdf"
    if p.exists():
        return p.read_text(encoding="utf-8"), "file"
    return arg, "manual"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="resume_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="build a resume for one JD")
    run.add_argument("--jd", required=True, help="path to .pdf/.txt, or the JD text itself")
    run.add_argument("--company", required=True)
    run.add_argument("--role", required=True)
    run.add_argument("--slug", help="output name (default: company_role)")
    run.add_argument("--save-to-kit", action="store_true", help="also copy the .tex to Resume_Builder/resumes/")
    run.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    from .graph import build_graph
    from .state import JobPosting

    text, source = read_jd(args.jd)
    job = JobPosting(company=args.company, role=args.role, text=text, source=source)
    init = {"job": job}
    if args.slug:
        init["slug"] = args.slug
    import time
    t0, final = time.time(), dict(init)
    for chunk in build_graph().stream(init, config={"recursion_limit": 80}, stream_mode="updates"):
        for node, upd in chunk.items():
            final.update(upd or {})
            extra = ""
            if node == "validate":
                iss = upd.get("issues", [])
                extra = f" — {sum(i.severity == 'hard' for i in iss)} hard / {sum(i.severity == 'soft' for i in iss)} soft issue(s)"
            elif node == "layout_check" and upd.get("layout"):
                L = upd["layout"]
                extra = f" — {L.pages} page(s), {L.free_lines} blank lines, {len(L.orphans)} orphan(s) → {upd['layout_action']}"
            print(f"[{time.time() - t0:6.1f}s] {node}{extra}", flush=True)

    rep = final["report"]
    print(f"\n.tex   : {rep.tex_path}")
    print(f"pdf    : {rep.pdf_path or 'not compiled'}  (pages: {rep.pages})")
    print(f"report : {Path(final['out_dir']) / 'report.md'}")
    print(f"projects: {', '.join(final['selected'])}  | archetype: {final['analysis'].archetype}")
    if rep.honest_gaps:
        print(f"gaps   : {', '.join(rep.honest_gaps)}")
    if final.get("violations"):
        print(f"note   : {len(final['violations'])} model bullet(s)/items rejected and replaced — see report.md")
    if rep.pages and rep.pages != 1:
        print(f"WARNING: {rep.pages} pages — Phase 2 will auto-trim; trim manually for now (kit §11).")
    if args.save_to_kit:
        dest = KIT_RESUMES / f"{final['slug']}.tex"
        if dest.exists():
            print(f"NOT copied: {dest} already exists (won't overwrite)")
        else:
            shutil.copy(rep.tex_path, dest)
            print(f"copied : {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
