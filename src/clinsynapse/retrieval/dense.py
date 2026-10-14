"""Dense retrieval: MedCPT embeddings in Qdrant (embedded local mode, or a server via URL)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from clinsynapse.retrieval import medcpt

COLLECTION = "pubmed_medcpt"


@lru_cache
def _client(path: str | None, url: str | None) -> QdrantClient:
    # Local mode locks its folder, so every DenseIndex in a process shares one client per path.
    return QdrantClient(url=url) if url else QdrantClient(path=path)


class DenseIndex:
    def __init__(self, client: QdrantClient, device: str = "mps"):
        self.client = client
        self.device = device

    @classmethod
    def open(cls, path: Path | None = None, url: str | None = None, device: str = "mps") -> "DenseIndex":
        return cls(_client(str(path) if path else None, url), device)

    def create(self, dim: int = 768) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(COLLECTION, VectorParams(size=dim, distance=Distance.DOT))

    def upsert(self, pmids: list[str], vectors: np.ndarray) -> None:
        points = [
            PointStruct(id=int(p), vector=v.tolist(), payload={"pmid": p})
            for p, v in zip(pmids, vectors)
        ]
        self.client.upsert(COLLECTION, points, wait=True)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        vec = medcpt.encode_queries([query], self.device)[0]
        hits = self.client.query_points(COLLECTION, query=vec.tolist(), limit=k).points
        return [(h.payload["pmid"], float(h.score)) for h in hits]
