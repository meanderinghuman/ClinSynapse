import json

from fastapi.testclient import TestClient

from clinsynapse.api import app as api
from clinsynapse.api.highlight import best_sentence
from clinsynapse.graph.build import Deps
from tests.fakes import FakeLLM, FakeRetriever


def client(monkeypatch):
    monkeypatch.setattr(api, "deps", lambda: Deps(FakeLLM(), FakeRetriever()))
    monkeypatch.setenv("CS_LLM_CACHE", "false")
    return TestClient(api.app)


def test_ask_returns_claims_with_support(monkeypatch):
    r = client(monkeypatch).post("/ask", json={"question": "metformin and cancer?"}).json()
    assert r["citations"][0]["url"].startswith("https://pubmed.ncbi.nlm.nih.gov/")
    assert "31%" in r["claims"][0]["support"]["111"]
    assert "passages" not in r and r["disclaimer"]


def test_stream_emits_nodes_then_result(monkeypatch):
    with client(monkeypatch).stream("POST", "/ask/stream", json={"question": "metformin?"}) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    nodes = [e["node"] for e in events if e["event"] == "node"]
    assert nodes[:2] == ["router", "planner"] and nodes[-1] == "finalize"
    assert events[-1]["event"] == "result" and events[-1]["result"]["citations"]


def test_best_sentence_prefers_numeric_match():
    abstract = "Background text here. Mortality fell by 12% in the treated arm. Funding was public."
    assert best_sentence("Treatment cut mortality by 12%", abstract).startswith("Mortality fell")
