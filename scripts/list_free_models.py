"""List OpenRouter's current free models, and check config/models.yaml against them.

    python scripts/list_free_models.py           # table of free models
    python scripts/list_free_models.py --check   # exit 1 if a configured model is gone

No API key needed (the models endpoint is public).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import yaml

URL = "https://openrouter.ai/api/v1/models"
CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def fetch() -> list[dict]:
    req = urllib.request.Request(URL, headers={"User-Agent": "resume-agent"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["data"]


def is_free(m: dict) -> bool:
    p = m.get("pricing") or {}
    return m["id"].endswith(":free") or (p.get("prompt") == "0" and p.get("completion") == "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    free = sorted((m for m in fetch() if is_free(m)), key=lambda m: -m.get("created", 0))
    print(f"{len(free)} free models\n")
    print(f"{'id':<55} {'ctx':>8}  tools  struct  added")
    for m in free:
        sp = set(m.get("supported_parameters") or [])
        added = dt.date.fromtimestamp(m.get("created", 0))
        print(f"{m['id']:<55} {m.get('context_length') or 0:>8}  {'y' if 'tools' in sp else '-':^5}  "
              f"{'y' if 'structured_outputs' in sp else '-':^6}  {added}")

    if not args.check:
        return 0
    ids = {m["id"] for m in free}
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    missing = [(role, mid) for role, r in cfg["roles"].items()
               for mid in [r["model"], *r.get("fallbacks", [])] if mid not in ids]
    print()
    if missing:
        for role, mid in missing:
            print(f"MISSING  {role}: {mid}")
        return 1
    print("All configured models are still free and available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
