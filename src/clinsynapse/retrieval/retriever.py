"""Retrieval pipeline: BM25 | dense | hybrid (RRF) -> optional MedCPT cross-encoder rerank -> top-k."""

import threading
from pathlib import Path

from pydantic import BaseModel

from clinsynapse.config import ROOT, Settings, get_settings
from clinsynapse.llm.cache import CacheMiss, ResponseCache, prompt_key
from clinsynapse.retrieval import medcpt
from clinsynapse.retrieval.bm25 import BM25Index
from clinsynapse.retrieval.corpus import Corpus
from clinsynapse.retrieval.dense import DenseIndex
from clinsynapse.retrieval.fusion import rrf


class Passage(BaseModel):
    pmid: str
    title: str = ""
    text: str
    score: float
    query: str = ""


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        bm25: BM25Index | None,
        dense: DenseIndex | None,
        settings: Settings | None = None,
    ):
        self.corpus = corpus
        self.bm25 = bm25
        self.dense = dense
        self.settings = settings or get_settings()

    @classmethod
    def from_disk(cls, settings: Settings | None = None) -> "Retriever":
        s = settings or get_settings()
        corpus = Corpus.load(s.index_dir / "corpus.jsonl")
        bm25 = BM25Index.load(s.index_dir / "bm25") if s.retrieval_mode != "dense" else None
        dense = (
            DenseIndex.open(s.index_dir / "qdrant", url=s.qdrant_url, device=s.device)
            if s.retrieval_mode != "bm25"
            else None
        )
        return cls(corpus, bm25, dense, s)

    def candidates(self, query: str) -> list[tuple[str, float]]:
        n = self.settings.candidates_per_retriever
        mode = self.settings.retrieval_mode
        if mode == "bm25":
            return self.bm25.search(query, n)
        if mode == "dense":
            return self.dense.search(query, n)
        rankings = [[p for p, _ in self.bm25.search(query, n)], [p for p, _ in self.dense.search(query, n)]]
        return rrf(rankings, k=self.settings.rrf_k)

    def retrieve(self, query: str, k: int | None = None) -> list[Passage]:
        k = k or self.settings.top_k
        cands = self.candidates(query)
        docs = [self.corpus.get(p) for p, _ in cands]
        pairs = [(d, s) for d, (_, s) in zip(docs, cands) if d is not None]

        if self.settings.rerank and pairs:
            scores = medcpt.cross_scores(query, [d.text for d, _ in pairs], self.settings.device)
            pairs = sorted(zip([d for d, _ in pairs], scores), key=lambda x: x[1], reverse=True)

        return [
            Passage(pmid=d.pmid, title=d.title, text=d.text, score=float(s), query=query)
            for d, s in pairs[:k]
        ]


META_VERSION = "meta:index_version"


def _rel(path: Path) -> str:
    p = path.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


class CachedRetriever:
    """Caches retrieval results; the index and models load only on the first cache miss."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.cache = ResponseCache(self.settings.cache_path)
        self._inner: Retriever | None = None
        self._lock = threading.Lock()
        self.version = self._index_version()

    def _index_version(self) -> str:
        """From the index's VERSION file; CI has no index, so fall back to the one the cache recorded."""
        f = self.settings.index_dir / "VERSION"
        if f.exists():
            version = f.read_text().strip()
            if not self.settings.replay_only:
                self.cache.put(META_VERSION, {"version": version})
            return version
        hit = self.cache.get(META_VERSION)
        return hit["version"] if hit else "unversioned"

    @property
    def inner(self) -> Retriever:
        with self._lock:  # eval workers race here on the first miss
            if self._inner is None:
                self._inner = Retriever.from_disk(self.settings)
        return self._inner

    def retrieve(self, query: str, k: int | None = None) -> list[Passage]:
        s = self.settings
        key = prompt_key({
            "kind": "retrieval", "query": query, "k": k or s.top_k, "mode": s.retrieval_mode,
            "rerank": s.rerank, "cands": s.candidates_per_retriever, "rrf_k": s.rrf_k,
            "index": _rel(s.index_dir), "version": self.version,
        })
        if hit := self.cache.get(key):
            return [Passage.model_validate(p) for p in hit["passages"]]
        if s.replay_only:
            raise CacheMiss(f"retrieval not in cache: {query!r}; re-record the smoke cache")
        passages = self.inner.retrieve(query, k)
        self.cache.put(key, {"passages": [p.model_dump() for p in passages]})
        return passages
