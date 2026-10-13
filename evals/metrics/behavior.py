"""Behavior metrics: does the system decline when it should, and only then?"""


def refusal_correct(abstained: bool, answerable: bool) -> float:
    """1 if it abstained on an unanswerable item or answered an answerable one."""
    return float(abstained != answerable)


def false_refusal(abstained: bool, answerable: bool) -> float | None:
    """Defined only on answerable items: 1 if it abstained anyway."""
    return float(abstained) if answerable else None


def correct_abstention(abstained: bool, answerable: bool) -> float | None:
    """Defined only on unanswerable items: 1 if it abstained (refusal accuracy)."""
    return None if answerable else float(abstained)
