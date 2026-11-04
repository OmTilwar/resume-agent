"""Run the pipeline once and report wall time and the OpenRouter spend it caused.

    python scripts/timed_run.py --jd <path> --company X --role Y [--slug S]
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def usage() -> float:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.loads(r.read().decode())["data"]["usage"])


if __name__ == "__main__":
    from resume_agent.__main__ import main
    before, t0 = usage(), time.time()
    rc = main(["run", *sys.argv[1:]])
    elapsed = time.time() - t0
    time.sleep(5)                     # usage is updated asynchronously
    spent = usage() - before
    print(f"\nwall time: {elapsed / 60:.1f} min · OpenRouter spend: ${spent:.4f}")
    sys.exit(rc)
