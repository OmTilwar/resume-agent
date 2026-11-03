"""Seeded-error benchmark: corrupt verified bank bullets in known ways.

Every item keeps the ORIGINAL bullet's cited facts, so a corrupted item is
unsupported by construction (label=1) and an untouched / benignly reworded one
is supported (label=0). Deterministic given the seed.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from ..knowledge import Context, KnowledgeBase
from ..knowledge.numbers import extract_numbers, normalize, normalize_values
from ..latex import _METRIC

VERSION = "seeded_v1"

# kind -> which layer *should* be able to catch it (for analysis only)
KINDS = {
    "num_fabricate": "deterministic",     # a real number changed
    "num_add": "deterministic",           # an invented metric clause appended
    "banned_phrase": "deterministic",     # novel / state-of-the-art inserted
    "pts_percent": "deterministic",       # (+5.92 pts) -> (+5.92%)
    "noun_swap": "judge",                 # instances -> images, queries -> users
    "scope_inflation": "judge",           # "deployed to production", "adopted by ..."
    "domain_swap": "judge",               # banking -> legal, chest X-rays -> brain MRIs
    "misattribution": "judge",            # leaderboard <-> validation, test set -> production
}
CLEAN_KINDS = {"original", "benign_verb", "benign_trim"}

NOUN_SWAPS = [
    ("annotated instances", "annotated images"), ("instances", "images"), ("records", "customer accounts"),
    ("banking queries", "user sessions"), ("queries", "users"), ("pairs", "documents"),
    ("chest X-rays", "patients"), ("stores", "customers"), ("texture fingerprints", "counterfeit detections"),
    ("DTD classes", "luxury brands"), ("faces", "clients"), ("identity documents", "passports in production"),
    ("unit and integration tests", "production deployments"), ("pytest tests", "beta testers"),
    ("GTest tests", "field trials"), ("thrusters", "vehicles"), ("reference entities", "verified customers"),
]
SCOPE_CLAUSES = [
    ", now running in production at a partner company",
    ", adopted by the research lab as its standard pipeline",
    ", leading a team of engineers through delivery",
    ", which became the default system for the whole department",
    ", deployed to live users after a successful pilot",
]
NUM_CLAUSES = [", serving 10,000+ daily users", ", cutting infrastructure cost by 35%",
               ", used by 2,000+ students", ", reducing manual effort by 60%", ", saving 120 hours per month"]
DOMAIN_SWAPS = {
    "finsight": [("banking", "legal"), ("account", "case")],
    "medvision": [("chest X-rays", "brain MRIs"), ("lung", "liver"), ("DICOM", "satellite")],
    "authnet": [("texture", "currency"), ("DTD", "banknote")],
    "amazon_er": [("business", "patient"), ("entity-resolution", "fraud-detection")],
    "retailforecast": [("store", "hospital"), ("promo", "drug")],
    "propvox": [("voice agent", "trading agent"), ("Hinglish", "Mandarin")],
    "photoretouch": [("CelebA faces", "medical scans"), ("photo", "radiology")],
    "voxpipeline": [("speech-data", "video-data"), ("TTS", "self-driving")],
    "submarine": [("underwater vehicles", "aircraft"), ("DVL", "GPS")],
    "vio": [("camera and IMU", "lidar and radar")],
    "id_vlm": [("identity documents", "medical records")],
    "resume_agent": [("resume", "legal contract"), ("job description", "court filing")],
    "experience": [("precision agriculture", "autonomous driving"), ("aerial and field imagery", "street-view imagery")],
}
MISATTRIBUTIONS = [
    ("on the Amazon ML Challenge 2026 leaderboard", "on the private test set"),
    ("leaderboard", "held-out test"), ("held-out test set", "training set"),
    ("cross-validation", "a held-out test set from another hospital"),
    ("on 762 banking queries", "on live customer traffic"), ("validation folds", "the final leaderboard"),
    ("on 47 DTD classes", "on real counterfeit products"), ("benchmark", "A/B test"),
]
BANNED_INSERTS = ["novel ", "state-of-the-art ", "cutting-edge "]
BENIGN_VERBS = {"Built": "Developed", "Developed": "Built", "Engineered": "Built", "Implemented": "Built",
                "Designed": "Built", "Shipped": "Delivered", "Deployed": "Shipped", "Benchmarked": "Evaluated",
                "Trained": "Trained", "Added": "Built", "Integrated": "Added", "Produced": "Generated"}


@dataclass
class Item:
    id: str
    source: str            # "<section>/<bullet id>"
    section: str
    text: str
    fact_ids: list[str]
    label: int             # 1 = corrupted (should be flagged), 0 = clean
    kind: str
    original: str


def _ctx_for(kb: KnowledgeBase, facts) -> Context:
    for c in [Context(), Context(archetype="healthcare"), Context(company="Entrupy"), Context(company="Sarvam AI")]:
        if all(kb.fact_allowed(f, c) for f in facts):
            return c
    return Context()


def bank(kb: KnowledgeBase):
    """(section, bullet, facts, ctx) for every verified bank bullet."""
    out = []
    for g in kb.experience.groups:
        for b in g.bullets:
            facts = [kb.experience.fact(f) for f in b.fact_ids]
            out.append(("experience", b, facts, Context()))
    for pid, p in kb.projects.items():
        for b in p.bullets:
            facts = [p.fact(f) for f in b.fact_ids]
            out.append((pid, b, facts, _ctx_for(kb, facts)))
    return out


def _insert_before_end(text: str, clause: str) -> str:
    t = text.rstrip()
    return (t[:-1] + clause + ".") if t.endswith(".") else t + clause


def _swap(text: str, pairs, rng) -> str | None:
    hits = [(a, b) for a, b in pairs if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", text)]
    if not hits:
        return None
    a, b = rng.choice(hits)
    return re.sub(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", b, text, count=1)


def _fabricate_number(text: str, allowed: set[str], rng) -> str | None:
    spans = [m for m in _METRIC.finditer(text)
             if re.search(r"\d", m.group(1)) and not re.match(r"^(19|20)\d{2}$", m.group(1))]
    spans = [m for m in spans if extract_numbers(m.group(1))]
    if not spans:
        return None
    m = rng.choice(spans)
    tok = m.group(1)
    num = re.search(r"\d[\d,]*(?:\.\d+)?", tok).group(0)
    raw = num.replace(",", "")
    for _ in range(20):
        if "." in raw:
            dec = len(raw.split(".")[1])
            new = f"{float(raw) * rng.uniform(1.04, 1.25):.{dec}f}"
            if float(raw) < 1 and float(new) >= 1:
                new = f"{min(float(raw) + rng.uniform(0.01, 0.05), 0.999):.{dec}f}"
        else:
            v = int(raw)
            new = str(v + max(1, round(v * rng.uniform(0.1, 0.4))))
            if "," in num:
                new = f"{int(new):,}"
        if normalize(new) not in allowed and normalize(new) != normalize(raw):
            return text[:m.start(1)] + tok.replace(num, new, 1) + text[m.end(1):]
    return None


def build(kb: KnowledgeBase, seed: int = 42) -> list[Item]:
    rng = random.Random(seed)
    items: list[Item] = []

    def add(section, b, text, label, kind):
        if text and (label == 0 or text != b.text):
            items.append(Item(id="", source=f"{section}/{b.id}", section=section, text=text,
                              fact_ids=list(b.fact_ids), label=label, kind=kind, original=b.text))

    for section, b, facts, ctx in bank(kb):
        allowed = set().union(*(normalize_values(f.values) for f in facts)) if facts else set()
        # clean controls
        add(section, b, b.text, 0, "original")
        verb = b.text.split(" ", 1)[0]
        if verb in BENIGN_VERBS and BENIGN_VERBS[verb] != verb:
            add(section, b, BENIGN_VERBS[verb] + b.text[len(verb):], 0, "benign_verb")
        trimmed = re.sub(r"\s*\((?![^)]*\d)[^)]*\)", "", b.text, count=1)   # drop a number-free parenthetical
        if trimmed != b.text:
            add(section, b, trimmed, 0, "benign_trim")
        # corruptions
        add(section, b, _fabricate_number(b.text, allowed, rng), 1, "num_fabricate")
        add(section, b, _insert_before_end(b.text, rng.choice(NUM_CLAUSES)), 1, "num_add")
        add(section, b, _swap(b.text, NOUN_SWAPS, rng), 1, "noun_swap")
        add(section, b, _insert_before_end(b.text, rng.choice(SCOPE_CLAUSES)), 1, "scope_inflation")
        add(section, b, _swap(b.text, DOMAIN_SWAPS.get(section, []), rng), 1, "domain_swap")
        add(section, b, _swap(b.text, MISATTRIBUTIONS, rng), 1, "misattribution")
        if re.search(r"\(\+[\d.]+ pts\)", b.text):
            add(section, b, re.sub(r"\(\+([\d.]+) pts\)", r"(+\1%)", b.text, count=1), 1, "pts_percent")
        m = re.search(r"\b(a|an) ", b.text)
        if m:
            ins = rng.choice(BANNED_INSERTS)
            art = "a" if m.group(1) == "an" and ins[0] not in "aeiou" else m.group(1)
            add(section, b, b.text[:m.start()] + f"{art} {ins}" + b.text[m.end():], 1, "banned_phrase")

    rng.shuffle(items)   # so judge batches mix kinds and sections
    for i, it in enumerate(items):
        it.id = f"s{i + 1:04d}"
    return items


def to_rows(items: list[Item]) -> list[dict]:
    return [asdict(i) for i in items]
