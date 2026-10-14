"""Find the sentence in a cited abstract that best supports a claim, for UI highlighting."""

import re

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {
    "the", "a", "an", "of", "in", "and", "or", "to", "for", "with", "on",
    "by", "was", "were", "is", "are", "be", "as", "at", "from", "that", "this",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


def best_sentence(claim: str, abstract: str) -> str:
    sentences = [s for s in _SPLIT.split(abstract) if s.strip()]
    if not sentences:
        return ""
    want = _tokens(claim)
    # Numbers are the most error-prone part of a claim, so matching them counts double.
    def score(s: str) -> float:
        have = _tokens(s)
        overlap = want & have
        return len(overlap) + sum(1 for t in overlap if t[0].isdigit())

    return max(sentences, key=score)
