from clinsynapse.graph import prompts
from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import GraphState, RouteDecision
from clinsynapse.llm.client import LLMError


@traced("router")
def router(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    try:
        decision = deps.llm.structured(
            [{"role": "system", "content": prompts.ROUTER}, {"role": "user", "content": state["question"]}],
            RouteDecision,
            model=ctx.settings.small_model,
            usage=ctx.usage,
            node="router",
            max_tokens=256,
        )
        route = decision.category
    except LLMError:
        ctx.tool_errors += 1
        route = "research"  # fail open to the guarded pipeline, which can still abstain
    if route != "research":
        return {"route": route, "refusal": prompts.REFUSALS[route], "final_text": prompts.REFUSALS[route]}
    return {"route": route}


def route_after_router(state: GraphState) -> str:
    return "planner" if state["route"] == "research" else "__end__"
