"""Bootstrap confidence intervals and paired permutation tests.

All functions take per-item score arrays so any metric that decomposes per item (accuracy,
recall@k, faithfulness, citation precision, ...) gets a CI and an A/B significance test.
Corpus-level metrics that don't decompose (macro-F1) take a `stat` callable over item indices.
"""

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

SEED = 0


class CI(BaseModel):
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


class Comparison(BaseModel):
    metric: str
    a: float
    b: float
    diff: float
    diff_lo: float
    diff_hi: float
    p_value: float
    n: int


def _clean(x: Sequence[float | None]) -> np.ndarray:
    return np.array([v for v in x if v is not None and not np.isnan(v)], dtype=float)


def bootstrap_ci(
    values: Sequence[float | None],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> CI:
    x = _clean(values)
    if len(x) == 0:
        return CI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(stat(x)), lo=float(lo), hi=float(hi), n=len(x))


def bootstrap_ci_indexed(
    n: int, stat: Callable[[np.ndarray], float], n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED
) -> CI:
    """CI for a corpus-level statistic computed from a set of item indices (e.g. macro-F1)."""
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(stat(np.arange(n))), lo=float(lo), hi=float(hi), n=n)


def _paired(a: Sequence[float | None], b: Sequence[float | None]) -> tuple[np.ndarray, np.ndarray]:
    if len(a) != len(b):
        raise ValueError("paired comparison needs aligned per-item arrays")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None
             and not np.isnan(x) and not np.isnan(y)]
    if not pairs:
        return np.array([]), np.array([])
    xa, xb = zip(*pairs)
    return np.array(xa, dtype=float), np.array(xb, dtype=float)


def paired_permutation_test(
    a: Sequence[float | None], b: Sequence[float | None], n_perm: int = 10_000, seed: int = SEED
) -> float:
    """Two-sided p-value for mean(b - a) != 0 by randomly sign-flipping per-item differences."""
    xa, xb = _paired(a, b)
    if len(xa) == 0:
        return 1.0
    d = xb - xa
    observed = abs(d.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (n_perm + 1))


def compare(metric: str, a: Sequence[float | None], b: Sequence[float | None], seed: int = SEED) -> Comparison:
    """A vs B on the same items: means, bootstrap CI of the difference, permutation p-value."""
    xa, xb = _paired(a, b)
    diff_ci = bootstrap_ci(xb - xa, seed=seed) if len(xa) else CI(mean=0, lo=0, hi=0, n=0)
    return Comparison(
        metric=metric,
        a=float(xa.mean()) if len(xa) else float("nan"),
        b=float(xb.mean()) if len(xb) else float("nan"),
        diff=diff_ci.mean,
        diff_lo=diff_ci.lo,
        diff_hi=diff_ci.hi,
        p_value=paired_permutation_test(a, b, seed=seed),
        n=len(xa),
    )


def cohens_kappa(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    t, p = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    po = np.mean(t == p)
    pe = np.mean(t) * np.mean(p) + np.mean(~t) * np.mean(~p)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
