"""Per-item retrieval metrics against gold PMIDs."""

import math


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    for i, p in enumerate(ranked):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int = 10) -> float | None:
    if not gold:
        return None
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg
