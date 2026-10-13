from clinsynapse.config import Settings
from clinsynapse.graph.build import Deps, answer
from clinsynapse.graph.state import RouteDecision
from evals.datasets.items import Item, router_probes
from evals.run import score_item, summarize
from tests.fakes import FakeLLM, FakeRetriever


class KeywordJudge:
    """Supported iff the claim's first word appears in the evidence."""

    name = "kw"

    def supported(self, claim, evidence):
        return claim.split()[0].lower() in evidence.lower().split()


def result(llm=None):
    return answer("q", Deps(llm or FakeLLM(), FakeRetriever()), Settings(llm_cache=False))


def test_score_item_on_answerable_pubmedqa_item():
    item = Item(id="1", source="pubmedqa", question="q", gold_pmids=["111"], label="yes")
    m = score_item(item, result(), KeywordJudge(), {})
    assert m["accuracy"] == 1.0
    assert m["citation_validity"] == 1.0
    assert m["citation_precision"] == 0.5  # cites 111 (gold) and 222 (not gold)
    assert m["citation_recall"] == 1.0
    assert m["faithfulness"] == 0.5        # "It acts via..." fails the keyword judge
    assert m["refusal_correct"] == 1.0 and m["false_refusal"] == 0.0


def test_score_item_router_probe():
    item = router_probes()[0]
    llm = FakeLLM(router=lambda m: RouteDecision(category="personal_advice", reason=""))
    m = score_item(item, result(llm), KeywordJudge(), {})
    assert m["route_correct"] == 1.0 and m["abstained"] is None


def test_summarize_reports_cis_and_ops():
    item = Item(id="1", source="pubmedqa", question="q", gold_pmids=["111"], label="yes")
    r = result()
    rec = {"item": item.model_dump(), "result": r.model_dump(), "metrics": score_item(item, r, KeywordJudge(), {})}
    s = summarize([rec, rec])
    assert s["metrics"]["accuracy"]["mean"] == 1.0
    assert s["metrics"]["macro_f1"]["n"] == 2
    assert {"p50_latency_s", "p95_latency_s", "usd_per_1k_queries"} <= set(s["ops"])


def test_router_probes_cover_both_refusal_types():
    routes = {p.expected_route for p in router_probes()}
    assert routes == {"personal_advice", "off_topic"}


def test_summarize_reports_majority_baseline():
    def rec(label):
        item = Item(id=label, source="pubmedqa", question="q", gold_pmids=["111"], label=label)
        r = result()
        return {"item": item.model_dump(), "result": r.model_dump(),
                "metrics": score_item(item, r, KeywordJudge(), {})}

    s = summarize([rec("yes"), rec("yes"), rec("no")])
    assert s["baselines"]["majority_label"] == "yes"
    assert abs(s["baselines"]["majority_accuracy"] - 2 / 3) < 1e-9
