import numpy as np

from evals.metrics.answer import macro_f1
from evals.metrics.citation import citation_precision, citation_recall, citation_validity
from evals.metrics.retrieval import mrr, ndcg_at_k, recall_at_k
from evals.stats import bootstrap_ci, cohens_kappa, compare, paired_permutation_test


def test_bootstrap_ci_contains_mean_and_narrows_with_n():
    small = bootstrap_ci([1, 0] * 10)
    big = bootstrap_ci([1, 0] * 500)
    assert small.lo <= 0.5 <= small.hi
    assert (big.hi - big.lo) < (small.hi - small.lo)


def test_permutation_detects_real_difference_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.6, 400).astype(float)
    b = np.where(rng.random(400) < 0.15, 1.0, a)  # b fixes ~15% of items
    assert paired_permutation_test(a, b) < 0.01
    assert paired_permutation_test(a, a.copy()) == 1.0


def test_compare_ignores_missing_items():
    c = compare("m", [1, None, 0, 1], [1, 1, 1, None])
    assert c.n == 2 and c.diff == 0.5


def test_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_retrieval_metrics():
    ranked, gold = ["a", "b", "c"], {"b", "z"}
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert mrr(ranked, gold) == 0.5
    assert 0 < ndcg_at_k(ranked, gold) < 1
    assert recall_at_k(ranked, set(), 2) is None


def test_citation_metrics():
    assert citation_validity(["a", "x"], ["a", "b"]) == 0.5
    assert citation_precision(["a", "b"], {"a"}) == 0.5
    assert citation_recall(["a"], {"a", "c"}) == 0.5
    assert citation_validity([], ["a"]) is None


def test_macro_f1_counts_abstain_as_wrong():
    assert macro_f1(["yes", "no", "maybe"], ["yes", "no", "maybe"]) == 1.0
    assert macro_f1([None, None, None], ["yes", "no", "maybe"]) == 0.0
