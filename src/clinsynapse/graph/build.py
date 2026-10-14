"""Graph assembly and the single entry point `answer()` used by the API and the evals.

router -> planner -> retriever -> grader --(insufficient, <= max loops)--> retriever
                                        \\-> synthesizer -> verifier -> finalize
                                        \\-> abstain
"""

from collections.abc import Iterator
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from clinsynapse.config import Settings, get_settings
from clinsynapse.graph import prompts
from clinsynapse.graph.common import traced
from clinsynapse.graph.grader import grader, route_after_grader
from clinsynapse.graph.planner import planner
from clinsynapse.graph.retrieve import retrieve
from clinsynapse.graph.router import route_after_router, router
from clinsynapse.graph.state import GraphState, QueryContext
from clinsynapse.graph.synthesizer import route_after_synth, synthesizer
from clinsynapse.graph.verifier import verifier
from clinsynapse.llm.client import LLMClient
from clinsynapse.observability import langfuse_handler
from clinsynapse.retrieval.retriever import Retriever


@dataclass
class Deps:
    llm: LLMClient
    retriever: Retriever


class Citation(BaseModel):
    pmid: str
    url: str
    text: str


class AnswerClaim(BaseModel):
    text: str
    pmids: list[str]
    supported: bool


class Result(BaseModel):
    question: str
    route: str
    answer: str
    verdict: str | None
    claims: list[AnswerClaim]
    citations: list[Citation]
    retrieved_pmids: list[str]
    passages: list[Citation] = []  # everything the synthesizer could see; used by the evals
    abstained: bool
    abstain_reason: str | None
    confidence: float
    trace_id: str
    # Agent and operations telemetry, consumed by the eval suite.
    steps: int
    retrieval_loops: int
    tool_errors: int
    budget_hit: str | None
    latency_s: float
    tokens: int
    cost_usd: float
    llm_calls: int
    cache_hits: int
    node_ms: dict[str, float]


@traced("abstain")
def abstain(state: GraphState, config) -> GraphState:
    return {"abstained": True, "abstain_reason": state.get("abstain_reason") or "no_relevant_evidence"}


@traced("finalize")
def finalize(state: GraphState, config) -> GraphState:
    claims = state.get("claims", [])
    kept = [c for c in claims if c["supported"]] if claims else []
    if state.get("abstained") or not kept:
        return {"final_text": prompts.ABSTAIN, "abstained": True}
    body = " ".join(f"{c['text']} [{', '.join('PMID:' + p for p in c['pmids'])}]" for c in kept)
    lead = f"**{state['verdict'].capitalize()}.** " if state.get("verdict") else ""
    return {"final_text": f"{lead}{body}"}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("router", router)
    g.add_node("planner", planner)
    g.add_node("retriever", retrieve)
    g.add_node("grader", grader)
    g.add_node("synthesizer", synthesizer)
    g.add_node("verifier", verifier)
    g.add_node("abstain", abstain)
    g.add_node("finalize", finalize)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {"planner": "planner", "__end__": END})
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "grader")
    g.add_conditional_edges("grader", route_after_grader, ["retriever", "synthesizer", "abstain"])
    g.add_conditional_edges("synthesizer", route_after_synth, ["verifier", "abstain"])
    g.add_edge("verifier", "finalize")
    g.add_edge("abstain", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def _run_config(ctx: QueryContext, deps: Deps) -> dict:
    config = {
        "configurable": {"ctx": ctx, "deps": deps},
        "recursion_limit": 50,
        "metadata": {"trace_id": ctx.trace_id},
        "run_name": "clinsynapse",
    }
    if handler := langfuse_handler():
        config["callbacks"] = [handler]
    return config


def _graph():
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    return _GRAPH


def answer(question: str, deps: Deps, settings: Settings | None = None) -> Result:
    ctx = QueryContext(settings=settings or get_settings())
    state = _graph().invoke({"question": question}, _run_config(ctx, deps))
    return _to_result(question, state, ctx)


def answer_stream(question: str, deps: Deps, settings: Settings | None = None) -> Iterator[dict]:
    """Yields {"event": "node", "node": ...} as each node finishes, then {"event": "result", ...}."""
    ctx = QueryContext(settings=settings or get_settings())
    state: dict = {}
    for mode, chunk in _graph().stream(
        {"question": question}, _run_config(ctx, deps), stream_mode=["updates", "values"]
    ):
        if mode == "values":
            state = chunk
        else:
            for node in chunk:
                yield {"event": "node", "node": node, "elapsed_s": round(ctx.elapsed(), 2)}
    yield {"event": "result", "result": _to_result(question, state, ctx).model_dump(exclude={"passages"})}


def _to_result(question: str, state: dict, ctx: QueryContext) -> Result:
    by_pmid = {p["pmid"]: p for p in state.get("passages", [])}
    kept = [c for c in state.get("claims", []) if c["supported"]] if not state.get("abstained") else []
    cited = list(dict.fromkeys(p for c in kept for p in c["pmids"] if p in by_pmid))
    node_ms: dict[str, float] = {}
    for e in ctx.events:
        node_ms[e.node] = node_ms.get(e.node, 0.0) + e.ms

    route = state.get("route", "research")
    return Result(
        question=question,
        route=route,
        answer=state.get("final_text", ""),
        verdict=None if state.get("abstained") else state.get("verdict"),
        claims=[AnswerClaim(text=c["text"], pmids=c["pmids"], supported=c["supported"])
                for c in state.get("claims", [])],
        citations=[Citation(pmid=p, url=f"https://pubmed.ncbi.nlm.nih.gov/{p}/", text=by_pmid[p]["text"])
                   for p in cited],
        retrieved_pmids=list(by_pmid),
        passages=[Citation(pmid=p, url=f"https://pubmed.ncbi.nlm.nih.gov/{p}/", text=d["text"])
                  for p, d in by_pmid.items()],
        abstained=route == "research" and bool(state.get("abstained")),
        abstain_reason=state.get("abstain_reason"),
        confidence=0.0 if state.get("abstained") else state.get("confidence", 0.0),
        trace_id=ctx.trace_id,
        steps=ctx.steps,
        retrieval_loops=max(0, state.get("retrieval_passes", 0) - 1),
        tool_errors=ctx.tool_errors,
        budget_hit=ctx.budget_hit,
        latency_s=round(ctx.elapsed(), 3),
        tokens=ctx.usage.total_tokens,
        cost_usd=ctx.usage.cost_usd,
        llm_calls=ctx.usage.calls,
        cache_hits=ctx.usage.cache_hits,
        node_ms=node_ms,
    )
