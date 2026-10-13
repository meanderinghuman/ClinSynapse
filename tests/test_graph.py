from clinsynapse.config import Settings
from clinsynapse.graph.build import Deps, answer
from clinsynapse.graph.state import (
    AnswerDraft,
    Claim,
    ClaimCheck,
    Grading,
    PassageGrade,
    RouteDecision,
    Verification,
)
from clinsynapse.llm.client import LLMError
from tests.fakes import FakeLLM, FakeRetriever

Q = "Does metformin reduce cancer risk, and by what mechanism?"


def run(llm=None, retriever=None, **overrides):
    s = Settings(llm_cache=False, **overrides)
    return answer(Q, Deps(llm=llm or FakeLLM(), retriever=retriever or FakeRetriever()), s)


def test_happy_path_cites_every_claim():
    r = run()
    assert not r.abstained
    assert r.verdict == "yes"
    assert [c.pmid for c in r.citations] == ["111", "222"]
    assert "PMID:111" in r.answer and "PMID:222" in r.answer
    assert r.confidence > 0.9
    assert {"router", "planner", "retriever", "grader", "synthesizer", "verifier"} <= set(r.node_ms)


def test_personal_advice_is_refused_without_retrieval():
    retriever = FakeRetriever()
    llm = FakeLLM(router=lambda m: RouteDecision(category="personal_advice", reason="dosing"))
    r = run(llm, retriever)
    assert r.route == "personal_advice"
    assert not r.abstained and not r.citations
    assert "clinician" in r.answer
    assert retriever.calls == []


def test_insufficient_evidence_retries_then_abstains():
    retriever = FakeRetriever()
    llm = FakeLLM(
        grader=lambda m: Grading(grades=[], sufficient=False, rewritten_query="metformin neoplasm"),
    )
    r = run(llm, retriever, max_retrieval_loops=2)
    assert r.abstained
    assert r.retrieval_loops == 2  # capped at max_retrieval_loops retries
    assert retriever.calls.count("metformin neoplasm") == 2  # deduped across sub-queries
    assert "synthesizer" not in llm.calls


def test_verifier_drops_unsupported_claim():
    llm = FakeLLM(
        verifier=lambda m: Verification(
            checks=[ClaimCheck(index=0, supported=True), ClaimCheck(index=1, supported=False)]
        ),
        synthesizer=lambda m: AnswerDraft(
            claims=[
                Claim(text="Metformin reduced cancer incidence by 31%.", pmids=["111"]),
                Claim(text="Supported too.", pmids=["222"]),
                Claim(text="Also supported.", pmids=["111"]),
            ]
        ),
    )
    llm.handlers["verifier"] = lambda m: Verification(
        checks=[ClaimCheck(index=0, supported=True), ClaimCheck(index=1, supported=False),
                ClaimCheck(index=2, supported=True)]
    )
    r = run(llm)
    assert not r.abstained
    assert "Supported too." not in r.answer
    assert [c.supported for c in r.claims] == [True, False, True]


def test_abstains_when_most_claims_fail_verification():
    llm = FakeLLM(
        verifier=lambda m: Verification(
            checks=[ClaimCheck(index=0, supported=False), ClaimCheck(index=1, supported=False)]
        )
    )
    r = run(llm)
    assert r.abstained and r.abstain_reason == "unsupported_claims"
    assert r.citations == [] and r.verdict is None


def test_citation_to_unretrieved_pmid_is_unsupported():
    llm = FakeLLM(
        synthesizer=lambda m: AnswerDraft(
            claims=[
                Claim(text="Real.", pmids=["111"]),
                Claim(text="Real too.", pmids=["222"]),
                Claim(text="Hallucinated source.", pmids=["999"]),
            ]
        ),
        verifier=lambda m: Verification(checks=[ClaimCheck(index=i, supported=True) for i in range(3)]),
    )
    r = run(llm)
    assert [c.supported for c in r.claims] == [True, True, False]
    assert "999" not in {c.pmid for c in r.citations}


def test_grader_off_skips_grading():
    llm = FakeLLM()
    r = run(llm, use_grader=False, use_verifier=False, use_planner=False)
    assert "grader" not in llm.calls and "verifier" not in llm.calls and "planner" not in llm.calls
    assert not r.abstained


def test_router_failure_fails_open_and_counts_tool_error():
    def boom(m):
        raise LLMError("down")

    r = run(FakeLLM(router=boom))
    assert r.tool_errors == 1 and r.route == "research" and not r.abstained


def test_step_budget_degrades_to_abstain():
    r = run(max_steps=3)
    assert r.abstained and r.budget_hit == "steps"


def test_grader_ignores_pmids_it_was_not_shown():
    llm = FakeLLM(
        grader=lambda m: Grading(
            grades=[PassageGrade(pmid="111", relevant=True), PassageGrade(pmid="777", relevant=True),
                    PassageGrade(pmid="222", relevant=True)],
            sufficient=True,
        )
    )
    r = run(llm)
    assert "777" not in r.retrieved_pmids


def test_no_retry_counted_when_insufficient_without_rewrite():
    llm = FakeLLM(grader=lambda m: Grading(
        grades=[PassageGrade(pmid="111", relevant=True)], sufficient=False, rewritten_query=None))
    r = run(llm)
    assert r.retrieval_loops == 0 and not r.abstained
