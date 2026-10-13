"""Reciprocal rank fusion (Cormack et al., 2009)."""


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pmid in enumerate(ranking):
            scores[pmid] = scores.get(pmid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
