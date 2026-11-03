"""Fail loudly when MASTER_RESUME_KIT.md changes but kb/ wasn't updated."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parents[1] / "MASTER_RESUME_KIT.md"   # Resume_Builder/MASTER_RESUME_KIT.md


@pytest.mark.skipif(not KIT.exists(), reason="kit not found next to projects/")
def test_kb_matches_kit_version():
    meta = yaml.safe_load((ROOT / "kb" / "meta.yaml").read_text(encoding="utf-8"))
    head = KIT.read_text(encoding="utf-8")[:2000]
    m = re.search(r"\*Last updated:\s*([^·*]+?)\s*[·*]", head)
    assert m, "could not find '*Last updated:' line in the kit"
    assert m.group(1).strip() == meta["kit_updated"], (
        f"kit updated to {m.group(1).strip()!r} but kb/meta.yaml says {meta['kit_updated']!r} — "
        "sync kb/*.yaml with the kit, then bump kit_updated")


@pytest.mark.skipif(not KIT.exists(), reason="kit not found next to projects/")
def test_every_kit_github_url_is_in_kb(kb):
    kit = KIT.read_text(encoding="utf-8")
    sec7 = kit[kit.index("## 7. Projects"): kit.index("## 8. Achievements")]
    urls = set(re.findall(r"\*\*GitHub:\*\*\s*(\S+)", sec7))
    assert urls == {p.github for p in kb.projects.values()}
