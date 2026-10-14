import random

from clinsynapse.config import Settings
from clinsynapse.graph.build import Deps, answer
from evals.datasets.items import Item
from evals.faults.injectors import (
    CANARY,
    CorruptNumbers,
    PromptInjection,
    RemoveGold,
    SwapCitations,
)
from tests.fakes import FakeLLM, FakeRetriever

ITEM = Item(id="x", source="golden", question="q", gold_pmids=["111"])


def base_result():
    return answer("q", Deps(FakeLLM(), FakeRetriever()), Settings(llm_cache=False))


def test_corrupt_numbers_changes_digits_only():
    r = CorruptNumbers().mutate(base_result(), ITEM, random.Random(0))
    assert "31%" not in r.claims[0].text and "%" in r.claims[0].text
    assert r.claims[1].text == "It acts via AMPK and mTOR inhibition."


def test_swap_citations_never_cites_gold():
    pool = [str(i) for i in range(1000, 1100)] + ["111"]
    r = SwapCitations(pool).mutate(base_result(), ITEM, random.Random(0))
    cited = {p for c in r.claims for p in c.pmids}
    assert cited and "111" not in cited and "222" not in cited


def test_remove_gold_filters_retrieval():
    wrapped = RemoveGold().wrap_retriever(FakeRetriever(), ITEM)
    assert "111" not in {p.pmid for p in wrapped.retrieve("q")}


def test_injection_lands_in_top_passage():
    wrapped = PromptInjection().wrap_retriever(FakeRetriever(), ITEM)
    hits = wrapped.retrieve("q")
    assert CANARY in hits[0].text and CANARY not in hits[1].text
