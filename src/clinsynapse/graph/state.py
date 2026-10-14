"""Graph state, structured-output schemas, and the per-query runtime context."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from clinsynapse.config import Settings
from clinsynapse.llm.client import Usage

Route = Literal["research", "personal_advice", "off_topic"]
Verdict = Literal["yes", "no", "maybe"]


# ---- LLM output schemas ---------------------------------------------------------------

class RouteDecision(BaseModel):
    category: Route
    reason: str


class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=4)


class PassageGrade(BaseModel):
    pmid: str
    relevant: bool


class Grading(BaseModel):
    grades: list[PassageGrade]
    sufficient: bool = Field(description="True if the relevant passages together answer the query")
    rewritten_query: str | None = Field(
        None, description="A better PubMed search query if evidence is insufficient"
    )


class Claim(BaseModel):
    text: str = Field(description="One factual sentence")
    pmids: list[str] = Field(min_length=1, description="PMIDs whose abstracts support this sentence")


class AnswerDraft(BaseModel):
    verdict: Verdict | None = Field(
        None, description="yes/no/maybe if the question is a yes/no question, else null"
    )
    claims: list[Claim]
    insufficient_evidence: bool = Field(
        False, description="True if the abstracts do not contain enough evidence to answer"
    )


class ClaimCheck(BaseModel):
    index: int
    supported: bool
    reason: str = ""


class Verification(BaseModel):
    checks: list[ClaimCheck]


# ---- Graph state ----------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    question: str
    route: Route
    sub_questions: list[str]
    queries: list[str]
    passages: list[dict]          # Passage dicts, deduped by pmid
    relevant_pmids: list[str]
    sufficient: bool
    retrieval_passes: int         # 1 = no retry; retrieval_loops = passes - 1
    draft: dict | None            # AnswerDraft
    claims: list[dict]            # verified claims: {text, pmids, supported}
    verdict: Verdict | None
    abstained: bool
    abstain_reason: str | None
    refusal: str | None
    confidence: float
    final_text: str


# ---- Runtime context (passed via config["configurable"]["ctx"]) -----------------------

@dataclass
class TraceEvent:
    node: str
    ms: float
    detail: dict = field(default_factory=dict)


@dataclass
class QueryContext:
    settings: Settings
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    usage: Usage = field(default_factory=Usage)
    started: float = field(default_factory=time.perf_counter)
    steps: int = 0
    budget_hit: str | None = None
    tool_errors: int = 0
    events: list[TraceEvent] = field(default_factory=list)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def over_budget(self) -> bool:
        s = self.settings
        if self.steps >= s.max_steps:
            self.budget_hit = self.budget_hit or "steps"
        elif self.usage.total_tokens >= s.max_tokens:
            self.budget_hit = self.budget_hit or "tokens"
        elif self.elapsed() >= s.max_seconds:
            self.budget_hit = self.budget_hit or "time"
        return self.budget_hit is not None
