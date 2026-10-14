"""Common eval item schema and loaders for every dataset."""

import json
import random
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

HERE = Path(__file__).parent


class Item(BaseModel):
    id: str
    source: Literal["pubmedqa", "bioasq", "golden", "router"]
    question: str
    gold_pmids: list[str] = Field(default_factory=list)
    label: Literal["yes", "no", "maybe"] | None = None   # yes/no/maybe gold, when applicable
    answerable: bool = True
    key_facts: list[str] = Field(default_factory=list)
    multi_hop: bool = False
    expected_route: Literal["research", "personal_advice", "off_topic"] = "research"
    qtype: str = "yesno"


TEST_SIZE = 300  # the reported test set: random.Random(0).sample(all 1,000, 300)


def _pubmedqa_all() -> list[Item]:
    from datasets import load_dataset

    rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    return [
        Item(id=f"pqa-{r['pubid']}", source="pubmedqa", question=r["question"],
             gold_pmids=[str(r["pubid"])], label=r["final_decision"])
        for r in rows
    ]


def pubmedqa(limit: int | None = None, seed: int = 0) -> list[Item]:
    return _subsample(_pubmedqa_all(), limit, seed)


def pubmedqa_dev(limit: int = 100, seed: int = 1) -> list[Item]:
    """Prompt-tuning split, disjoint from the 300-item test set, so tuning can't leak into results."""
    items = _pubmedqa_all()
    test = {it.id for it in _subsample(items, TEST_SIZE, 0)}
    return _subsample([it for it in items if it.id not in test], limit, seed)


def golden(path: Path = HERE / "golden.jsonl", limit: int | None = None) -> list[Item]:
    """Hand-built set: gold PMIDs, key facts, ~15% unanswerable, some multi-hop."""
    if not path.exists():
        return []
    items = [Item.model_validate({"source": "golden", **json.loads(line)})
             for line in path.open() if line.strip()]
    return items[:limit] if limit else items


def router_probes(path: Path = HERE / "router_probes.jsonl") -> list[Item]:
    return [Item.model_validate({"source": "router", "answerable": False, **json.loads(line)})
            for line in path.open() if line.strip()]


def bioasq(path: Path, limit: int | None = None, seed: int = 0) -> list[Item]:
    """BioASQ training JSON (free registration at bioasq.org). Gold docs are PubMed URLs."""
    data = json.loads(path.read_text())
    items = []
    for q in data["questions"]:
        if q["type"] not in ("yesno", "factoid"):
            continue
        pmids = [m.group(1) for d in q.get("documents", []) if (m := re.search(r"(\d+)$", d))]
        label = q.get("exact_answer") if q["type"] == "yesno" else None
        items.append(Item(id=f"bioasq-{q['id']}", source="bioasq", question=q["body"],
                          gold_pmids=pmids, label=label if label in ("yes", "no") else None,
                          qtype=q["type"]))
    return _subsample(items, limit, seed)


def load(name: str, limit: int | None = None, path: Path | None = None) -> list[Item]:
    if name == "pubmedqa":
        return pubmedqa(limit)
    if name == "pubmedqa-dev":
        return pubmedqa_dev(limit or 100)
    if name == "golden":
        return golden(limit=limit)
    if name == "router":
        return router_probes()[:limit] if limit else router_probes()
    if name == "bioasq":
        return bioasq(path or HERE / "bioasq.json", limit)
    raise ValueError(f"unknown dataset {name!r}")


def _subsample(items: list[Item], limit: int | None, seed: int) -> list[Item]:
    if limit is None or limit >= len(items):
        return items
    return random.Random(seed).sample(items, limit)
