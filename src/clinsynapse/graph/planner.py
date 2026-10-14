from clinsynapse.graph import prompts
from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import GraphState, Plan
from clinsynapse.llm.client import LLMError


@traced("planner")
def planner(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    subs = [state["question"]]
    if ctx.settings.use_planner:
        try:
            plan = deps.llm.structured(
                [{"role": "system", "content": prompts.PLANNER}, {"role": "user", "content": state["question"]}],
                Plan,
                usage=ctx.usage,
                node="planner",
                max_tokens=1024,
            )
            subs = plan.sub_questions
        except LLMError:
            ctx.tool_errors += 1
    return {"sub_questions": subs, "queries": subs, "passages": [], "retrieval_passes": 0}
