from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import GraphState
from clinsynapse.llm.cache import CacheMiss


@traced("retriever")
def retrieve(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    passages = {p["pmid"]: p for p in state.get("passages", [])}
    for q in state["queries"]:
        try:
            hits = deps.retriever.retrieve(q)
        except CacheMiss:
            raise
        except Exception:  # noqa: BLE001 - any retriever failure is a counted tool error
            ctx.tool_errors += 1
            continue
        for p in hits:
            # Re-tag with the latest query so the grader judges it against the rewritten query;
            # earlier relevance judgements persist in relevant_pmids.
            prev = passages.get(p.pmid)
            passages[p.pmid] = {**p.model_dump(), "score": max(p.score, prev["score"] if prev else p.score)}
    return {"passages": list(passages.values()), "retrieval_passes": state.get("retrieval_passes", 0) + 1}
