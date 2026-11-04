"""Seeded-error benchmark runner.

    python -m resume_agent.bench.run build                       # writes bench/data/seeded_v1.jsonl
    python -m resume_agent.bench.run judge --model nvidia/nemotron-3-ultra-550b-a55b:free
    python -m resume_agent.bench.run report                      # scores every cached model

Judge verdicts are cached per model in bench/cache/, so a run interrupted by
free-tier rate limits resumes where it stopped. Fallbacks are disabled: each
verdict is attributable to exactly one model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from ..knowledge import get_kb
from ..knowledge.checks import check_bullet
from ..validate import JUDGE_VARIANTS, Verdicts, evidence, judge_payload
from .seed import CLEAN_KINDS, KINDS, VERSION, Item, _ctx_for, build, to_rows

ROOT = Path(__file__).resolve().parents[2] / "bench"
DATA = ROOT / "data" / f"{VERSION}.jsonl"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
BATCH = 12
MIN_REPORTABLE = 50   # kit §2.4.4


def load_items() -> list[Item]:
    return [Item(**json.loads(l)) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def _facts(kb, it: Item):
    if it.section == "experience":
        return [kb.experience.fact(f) for f in it.fact_ids]
    p = kb.projects[it.section]
    return [p.fact(f) for f in it.fact_ids]


# ---- layers --------------------------------------------------------------

def deterministic_flags(items: list[Item]) -> dict[str, bool]:
    kb = get_kb()
    out = {}
    for it in items:
        facts = _facts(kb, it)
        v = check_bullet(it.text, it.id, facts, _ctx_for(kb, facts), kb, kb.projects.get(it.section))
        out[it.id] = bool(v)
    return out


def _cache_path(model: str, variant: str = "context") -> Path:
    return CACHE / (model.replace("/", "__").replace(":", "_") + f"@{variant}.jsonl")


def _parse_cache_name(stem: str) -> tuple[str, str]:
    name, _, variant = stem.partition("@")
    return name.replace("__", "/").replace("_free", ":free"), variant or "claims"


def load_cache(model: str, variant: str = "context") -> dict[str, dict]:
    p = _cache_path(model, variant)
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def run_judge(model: str, items: list[Item], limit: int | None = None, variant: str = "context") -> None:
    from ..llm import structured_for_model

    kb = get_kb()
    CACHE.mkdir(parents=True, exist_ok=True)
    done = load_cache(model, variant)
    todo = [it for it in items if it.id not in done][: limit or None]
    print(f"{model} [{variant}]: {len(done)} cached, {len(todo)} to judge")
    llm = structured_for_model(model, Verdicts)
    with _cache_path(model, variant).open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            rows = []
            for it in chunk:
                facts = _facts(kb, it)
                ev = evidence(it.section, it.fact_ids, {f.id: f for f in facts}, kb, _ctx_for(kb, facts), variant)
                rows.append((it.id, it.text, ev))
            t0 = time.time()
            try:
                res: Verdicts = llm.invoke(judge_payload(rows, variant))
            except Exception as e:   # noqa: BLE001 — rate limits etc.: stop, keep what we have
                print(f"  stopped at batch {i // BATCH + 1}: {type(e).__name__}: {str(e)[:200]}")
                break
            got = {v.id: v for v in (res.verdicts if res else [])}
            n = 0
            for it in chunk:
                v = got.get(it.id)
                if v is None:
                    continue                      # missing verdict → re-asked on next run
                fh.write(json.dumps({"id": it.id, "supported": v.supported, "issue": v.issue}) + "\n")
                n += 1
            fh.flush()
            print(f"  batch {i // BATCH + 1}: {n}/{len(chunk)} verdicts in {time.time() - t0:.0f}s", flush=True)


# ---- scoring --------------------------------------------------------------

def _rates(items, flagged: dict[str, bool]) -> dict:
    scored = [it for it in items if it.id in flagged]
    pos = [it for it in scored if it.label == 1]
    neg = [it for it in scored if it.label == 0]
    tp = sum(flagged[it.id] for it in pos)
    fp = sum(flagged[it.id] for it in neg)
    recall = tp / len(pos) if pos else None
    fpr = fp / len(neg) if neg else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    by_kind = {}
    for kind in list(KINDS) + sorted(CLEAN_KINDS):
        ks = [it for it in scored if it.kind == kind]
        if ks:
            hit = sum(flagged[it.id] for it in ks)
            by_kind[kind] = {"n": len(ks), "flagged": hit, "rate": round(hit / len(ks), 4),
                             "reportable": len(ks) >= MIN_REPORTABLE}
    r = lambda x: None if x is None else round(x, 4)
    return {"n_scored": len(scored), "n_corrupted": len(pos), "n_clean": len(neg),
            "catch_rate": r(recall), "false_positive_rate": r(fpr), "precision": r(precision), "f1": r(f1),
            "by_kind": by_kind}


def report(items: list[Item]) -> dict:
    det = deterministic_flags(items)
    out = {"benchmark": VERSION, "date": dt.date.today().isoformat(), "n_items": len(items),
           "layers": {"deterministic": _rates(items, det)}}
    for p in sorted(CACHE.glob("*.jsonl")) if CACHE.exists() else []:
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        model, variant = _parse_cache_name(p.stem)
        tag = f"{model.split('/')[-1]} | evidence={variant}"
        judge = {r["id"]: (not r["supported"]) for r in rows}
        combined = {i: det[i] or judge[i] for i in judge}
        out["layers"][f"judge[{tag}]"] = {**_rates(items, judge), "coverage": round(len(judge) / len(items), 4)}
        out["layers"][f"det+judge[{tag}]"] = _rates(items, combined)
    return out


def render_md(res: dict) -> str:
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"
    lines = [f"# Seeded-error benchmark ({res['benchmark']}, {res['date']})", "",
             f"{res['n_items']} items. Catch rate = corrupted bullets flagged; FPR = clean bullets wrongly flagged.", "",
             "| Layer | Scored | Catch rate | False-positive rate | Precision | F1 |", "|---|---|---|---|---|---|"]
    for name, L in res["layers"].items():
        cov = f" (coverage {pct(L['coverage'])})" if "coverage" in L else ""
        lines.append(f"| {name}{cov} | {L['n_scored']} | {pct(L['catch_rate'])} | {pct(L['false_positive_rate'])} | "
                     f"{pct(L['precision'])} | {pct(L['f1'])} |")
    kinds = list(KINDS) + sorted(CLEAN_KINDS)
    lines += ["", "## Flag rate by kind", "",
              "| Kind | n | " + " | ".join(res["layers"]) + " |", "|---|---|" + "---|" * len(res["layers"])]
    for k in kinds:
        n = next((L["by_kind"][k]["n"] for L in res["layers"].values() if k in L["by_kind"]), 0)
        if not n:
            continue
        cells = [f'{pct(L["by_kind"][k]["rate"])} ({L["by_kind"][k]["n"]})' if k in L["by_kind"] else "—"
                 for L in res["layers"].values()]
        flag = "" if n >= MIN_REPORTABLE else " *"
        lines.append(f"| {k}{flag} | {n} | " + " | ".join(cells) + " |")
    lines += ["", f"\\* fewer than {MIN_REPORTABLE} items — not quotable on a resume (kit §2.4.4).",
              "For clean kinds (original, benign_*), the rate is the false-positive rate; lower is better.",
              "scope_inflation and num_add use 5 template phrasings each — their rates partly measure those templates."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resume_agent.bench.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    j = sub.add_parser("judge")
    j.add_argument("--model", required=True)
    j.add_argument("--limit", type=int, help="judge at most N new items (to stay under daily limits)")
    j.add_argument("--trace", action="store_true", help="send judge calls to LangSmith (off by default: hundreds of calls)")
    j.add_argument("--variant", choices=sorted(JUDGE_VARIANTS), default="context",
                   help="judge evidence: 'claims' = cited fact claims only; 'context' = + project background + approved wording")
    sub.add_parser("report")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        DATA.parent.mkdir(parents=True, exist_ok=True)
        rows = to_rows(build(get_kb()))
        DATA.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} items to {DATA}")
        return 0
    items = load_items()
    if args.cmd == "judge":
        from ..tracing import configure_tracing, flush
        configure_tracing(enable=args.trace, project="resume-agent-bench")
        run_judge(args.model, items, args.limit, args.variant)
        flush()
        return 0
    res = report(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"{VERSION}_{res['date']}"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(render_md(res), encoding="utf-8")
    print(render_md(res))
    print(f"saved {stem}.json / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
