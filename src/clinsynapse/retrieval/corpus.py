"""PubMed abstract corpus: the single source of evidence every citation must point into."""

import json
import random
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel


class Doc(BaseModel):
    pmid: str
    title: str = ""
    text: str

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class Corpus:
    def __init__(self, docs: list[Doc]):
        self.docs = docs
        self._by_pmid = {d.pmid: d for d in docs}

    def __len__(self) -> int:
        return len(self.docs)

    def __contains__(self, pmid: str) -> bool:
        return pmid in self._by_pmid

    def get(self, pmid: str) -> Doc | None:
        return self._by_pmid.get(pmid)

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        with path.open() as f:
            return cls([Doc.model_validate_json(line) for line in f])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for d in self.docs:
                f.write(d.model_dump_json() + "\n")


def _pubmedqa_docs(config: str) -> Iterator[Doc]:
    from datasets import load_dataset

    for row in load_dataset("qiaojin/PubMedQA", config, split="train"):
        # Abstract body only: the title is phrased as the PubMedQA question and the conclusion
        # is the gold long answer, so both are excluded to keep retrieval and QA honest.
        yield Doc(pmid=str(row["pubid"]), text=" ".join(row["context"]["contexts"]))


def build_pubmedqa_corpus(max_docs: int | None = None, seed: int = 13) -> Corpus:
    """Labeled + unlabeled + artificial PubMedQA abstracts (~273k real PubMed articles).

    Labeled docs (the eval gold set) are always kept; `max_docs` subsamples the distractors.
    """
    gold = list(_pubmedqa_docs("pqa_labeled"))
    seen = {d.pmid for d in gold}
    rest = [d for cfg in ("pqa_unlabeled", "pqa_artificial") for d in _pubmedqa_docs(cfg)
            if d.pmid not in seen and not seen.add(d.pmid)]
    if max_docs is not None and len(gold) + len(rest) > max_docs:
        rest = random.Random(seed).sample(rest, max(0, max_docs - len(gold)))
    return Corpus(gold + rest)


def load_extra_docs(path: Path) -> list[Doc]:
    """Extra abstracts (e.g. golden-set or BioASQ gold docs) as JSONL of {pmid, title, text}."""
    if not path.exists():
        return []
    with path.open() as f:
        return [Doc.model_validate(json.loads(line)) for line in f if line.strip()]
