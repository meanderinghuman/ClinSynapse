"""Sparse retrieval with bm25s (in-process; swap for Elasticsearch at the 2.4M scale)."""

from pathlib import Path

import bm25s
import Stemmer

_stemmer = Stemmer.Stemmer("english")


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, pmids: list[str]):
        self.retriever = retriever
        self.pmids = pmids

    @classmethod
    def build(cls, pmids: list[str], texts: list[str]) -> "BM25Index":
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer, show_progress=False))
        return cls(retriever, pmids)

    def save(self, path: Path) -> None:
        self.retriever.save(path)
        (path / "pmids.txt").write_text("\n".join(self.pmids))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls(bm25s.BM25.load(path), (path / "pmids.txt").read_text().split("\n"))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer, show_progress=False)
        k = min(k, len(self.pmids))
        ids, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        return [(self.pmids[i], float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]
