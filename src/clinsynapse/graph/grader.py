"""Corrective-RAG evidence grader: keep relevant passages, retry retrieval with rewritten queries."""

from concurrent.futures import ThreadPoolExecutor

from clinsynapse.graph import prompts
from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import Grading, GraphState
from clinsynapse.llm.client import LLMError


@traced("grader")
def grader(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    passages = state["passages"]

    if not ctx.settings.use_grader:
        return {"relevant_pmids": [p["pmid"] for p in passages], "sufficient": True}

    def grade(query: str) -> tuple[str, Grading | None]:
        mine = [p for p in passages if p["query"] == query]
        if not mine:
            return query, Grading(grades=[], sufficient=False, rewritten_query=None)
        msgs = [
            {"role": "system", "content": prompts.GRADER},
            {"role": "user", "content": f"Query: {query}\n\n{prompts.format_abstracts(mine, 1200)}"},
        ]
        try:
            return query, deps.llm.structured(msgs, Grading, usage=ctx.usage, node="grader")
        except LLMError:
            ctx.tool_errors += 1
            return query, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(grade, state["queries"]))

    known = {p["pmid"] for p in passages}
    relevant = set(state.get("relevant_pmids", []))
    retry_queries, all_sufficient = [], True
    for query, g in results:
        if g is None:  # grader failed: keep this query's passages ungraded rather than drop them
            relevant |= {p["pmid"] for p in passages if p["query"] == query}
            continue
        relevant |= {x.pmid for x in g.grades if x.relevant and x.pmid in known}
        if not g.sufficient:
            all_sufficient = False
            if g.rewritten_query and g.rewritten_query not in retry_queries:
                retry_queries.append(g.rewritten_query)

    enough = all_sufficient and len(relevant) >= ctx.settings.min_relevant_passages
    return {
        "relevant_pmids": sorted(relevant),
        "sufficient": enough,
        "queries": retry_queries,
    }


def route_after_grader(state: GraphState, config) -> str:
    ctx = ctx_of(config)
    can_retry = (
        not state["sufficient"]
        and state["queries"]
        and state["retrieval_passes"] <= ctx.settings.max_retrieval_loops  # retries so far < max
        and not ctx.over_budget()
    )
    if can_retry:
        return "retriever"
    return "synthesizer" if state["relevant_pmids"] else "abstain"
