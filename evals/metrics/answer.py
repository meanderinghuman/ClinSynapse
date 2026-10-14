"""Answer-quality metrics: PubMedQA accuracy / macro-F1, golden-set key-fact recall."""

from collections.abc import Sequence

import numpy as np

LABELS = ("yes", "no", "maybe")


def accuracy_item(pred: str | None, gold: str) -> float:
    return float(pred == gold)


def macro_f1(preds: Sequence[str | None], golds: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Macro-F1 over labels; abstentions (None) count as wrong for every class."""
    preds, golds = np.array([p or "_abstain" for p in preds]), np.array(golds)
    f1s = []
    for lab in labels:
        tp = np.sum((preds == lab) & (golds == lab))
        fp = np.sum((preds == lab) & (golds != lab))
        fn = np.sum((preds != lab) & (golds == lab))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def key_fact_recall(facts_found: Sequence[bool]) -> float | None:
    """Share of the golden item's key facts present in the answer (as judged per fact)."""
    if not facts_found:
        return None
    return float(np.mean(facts_found))
