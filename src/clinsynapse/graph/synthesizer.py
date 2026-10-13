from clinsynapse.graph import prompts
from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import AnswerDraft, GraphState
from clinsynapse.llm.client import LLMError


@traced("synthesizer")
def synthesizer(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    if ctx.over_budget():
        return {"abstained": True, "abstain_reason": f"budget:{ctx.budget_hit}", "draft": None}

    relevant = set(state["relevant_pmids"])
    evidence = [p for p in state["passages"] if p["pmid"] in relevant]
    evidence.sort(key=lambda p: p["score"], reverse=True)
    subs = "\n".join(f"- {q}" for q in state["sub_questions"])
    msgs = [
        {"role": "system", "content": prompts.SYNTHESIZER},
        {
            "role": "user",
            "content": f"Question: {state['question']}\nSub-questions:\n{subs}\n\n"
            f"{prompts.format_abstracts(evidence[:12])}",
        },
    ]
    try:
        draft = deps.llm.structured(msgs, AnswerDraft, usage=ctx.usage, node="synthesizer", max_tokens=4096)
    except LLMError:
        ctx.tool_errors += 1
        return {"abstained": True, "abstain_reason": "synthesis_failed", "draft": None}

    if draft.insufficient_evidence or not draft.claims:
        return {"abstained": True, "abstain_reason": "insufficient_evidence", "draft": draft.model_dump()}
    return {"draft": draft.model_dump(), "verdict": draft.verdict, "abstained": False}


def route_after_synth(state: GraphState) -> str:
    return "abstain" if state.get("abstained") else "verifier"
