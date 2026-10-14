"""Citation metrics.

validity  : share of cited PMIDs that were actually retrieved (catches invented sources)
precision : share of cited PMIDs that are gold documents
recall    : share of gold documents that were cited
"""


def citation_validity(cited: list[str], retrieved: list[str]) -> float | None:
    if not cited:
        return None
    r = set(retrieved)
    return sum(p in r for p in cited) / len(cited)


def citation_precision(cited: list[str], gold: set[str]) -> float | None:
    if not cited or not gold:
        return None
    return sum(p in gold for p in cited) / len(cited)


def citation_recall(cited: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(set(cited) & gold) / len(gold)
