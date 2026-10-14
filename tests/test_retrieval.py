from clinsynapse.llm.cache import ResponseCache, prompt_key
from clinsynapse.retrieval.bm25 import BM25Index
from clinsynapse.retrieval.fusion import rrf


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert [p for p, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["d"] < dict(fused)["c"] or dict(fused)["d"] == dict(fused)["c"]


def test_rrf_single_list_preserves_order():
    assert [p for p, _ in rrf([["x", "y", "z"]])] == ["x", "y", "z"]


def test_bm25_finds_matching_doc(tmp_path):
    idx = BM25Index.build(["1", "2"], ["metformin lowers glucose", "statins lower cholesterol"])
    assert idx.search("cholesterol statin", 2)[0][0] == "2"
    idx.save(tmp_path / "bm25")
    assert BM25Index.load(tmp_path / "bm25").search("metformin", 1)[0][0] == "1"


def test_cache_roundtrip_and_key_stability(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    k = prompt_key({"b": 1, "a": [1, 2]})
    assert k == prompt_key({"a": [1, 2], "b": 1})
    assert c.get(k) is None
    c.put(k, {"content": "hi"})
    assert c.get(k) == {"content": "hi"}


def test_json_fallback_and_fence_stripping(tmp_path, monkeypatch):
    import httpx
    from openai import BadRequestError
    from pydantic import BaseModel

    from clinsynapse.config import Settings
    from clinsynapse.llm.client import LLMClient

    class Out(BaseModel):
        ok: bool

    sent = []

    class FakeCompletions:
        def create(self, **params):
            sent.append(params)
            if "response_format" in params:
                req = httpx.Request("POST", "http://x")
                raise BadRequestError("unsupported", response=httpx.Response(400, request=req), body=None)
            msg = type("M", (), {"content": 'Sure:\n```json\n{"ok": true}\n```'})
            return type("R", (), {"choices": [type("C", (), {"message": msg})], "usage": None})

    client = LLMClient(Settings(llm_api_key="k", cache_file=tmp_path / "c.sqlite"))
    client._client = type("O", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})})
    assert client.structured([{"role": "user", "content": "hi"}], Out).ok is True
    assert "response_format" in sent[0] and "response_format" not in sent[1]
    assert client._json_mode is False


def test_permission_denied_is_fatal_not_retried(tmp_path):
    import httpx
    import pytest
    from openai import PermissionDeniedError

    from clinsynapse.config import Settings
    from clinsynapse.llm.client import LLMClient, LLMFatal

    calls = []

    class Denied:
        def create(self, **params):
            calls.append(1)
            req = httpx.Request("POST", "http://x")
            raise PermissionDeniedError("VPN", response=httpx.Response(403, request=req), body=None)

    client = LLMClient(Settings(llm_api_key="k", cache_file=tmp_path / "c.sqlite"))
    client._client = type("O", (), {"chat": type("Ch", (), {"completions": Denied()})})
    with pytest.raises(LLMFatal):
        client.chat([{"role": "user", "content": "hi"}])
    assert calls == [1]


def test_persistent_connection_error_is_fatal(tmp_path, monkeypatch):
    import httpx
    import pytest
    from openai import APIConnectionError

    import clinsynapse.llm.client as mod
    from clinsynapse.config import Settings
    from clinsynapse.llm.client import LLMClient, LLMFatal

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = []

    class Down:
        def create(self, **params):
            calls.append(1)
            raise APIConnectionError(request=httpx.Request("POST", "http://x"))

    client = LLMClient(Settings(llm_api_key="k", cache_file=tmp_path / "c.sqlite"))
    client._client = type("O", (), {"chat": type("Ch", (), {"completions": Down()})})
    with pytest.raises(LLMFatal):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 6
