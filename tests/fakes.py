"""Deterministic stand-ins for the LLM and retriever so graph logic is tested offline."""

from collections.abc import Callable

from clinsynapse.graph.state import (
    AnswerDraft,
    Claim,
    ClaimCheck,
    Grading,
    PassageGrade,
    Plan,
    RouteDecision,
    Verification,
)
from clinsynapse.retrieval.retriever import Passage

DOCS = {
    "111": "Metformin use was associated with a 31% reduction in cancer incidence in diabetics.",
    "222": "Metformin activates AMPK and inhibits mTOR signalling in tumour cells.",
    "333": "Statins lowered LDL cholesterol in a randomized trial.",
}


class FakeRetriever:
    def __init__(self, docs: dict[str, str] = DOCS):
        self.docs = docs
        self.calls: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> list[Passage]:
        self.calls.append(query)
        return [Passage(pmid=p, text=t, score=1.0, query=query) for p, t in self.docs.items()]


class FakeLLM:
    """Responds per node; override any handler to script a scenario."""

    def __init__(self, **handlers: Callable):
        self.calls: list[str] = []
        self.handlers = {
            "router": lambda m: RouteDecision(category="research", reason="ok"),
            "planner": lambda m: Plan(sub_questions=["metformin cancer risk", "metformin mechanism"]),
            "grader": lambda m: Grading(
                grades=[PassageGrade(pmid="111", relevant=True), PassageGrade(pmid="222", relevant=True)],
                sufficient=True,
            ),
            "synthesizer": lambda m: AnswerDraft(
                verdict="yes",
                claims=[
                    Claim(text="Metformin reduced cancer incidence by 31%.", pmids=["111"]),
                    Claim(text="It acts via AMPK and mTOR inhibition.", pmids=["222"]),
                ],
            ),
            "verifier": lambda m: Verification(
                checks=[ClaimCheck(index=0, supported=True), ClaimCheck(index=1, supported=True)]
            ),
        }
        self.handlers.update(handlers)

    def structured(self, messages, schema, *, node="unknown", **_):
        self.calls.append(node)
        return self.handlers[node](messages)
