"""Per-claim citation verification: drop unsupported claims; abstain if too many fail."""

from clinsynapse.graph import prompts
from clinsynapse.graph.common import ctx_of, deps_of, traced
from clinsynapse.graph.state import GraphState, Verification
from clinsynapse.llm.client import LLMError


@traced("verifier")
def verifier(state: GraphState, config) -> GraphState:
    ctx, deps = ctx_of(config), deps_of(config)
    by_pmid = {p["pmid"]: p for p in state["passages"]}
    # Citations to PMIDs that were never retrieved are unsupported regardless of the judge.
    claims = [{**c, "valid_citations": all(p in by_pmid for p in c["pmids"])}
              for c in state["draft"]["claims"]]

    if not ctx.settings.use_verifier or ctx.over_budget():
        checked = [{**c, "supported": c["valid_citations"], "verified": False} for c in claims]
        return _finish(state, checked, ctx)

    cited = sorted({p for c in claims for p in c["pmids"] if p in by_pmid})
    numbered = "\n".join(f"[{i}] {c['text']} (cites: {', '.join(c['pmids'])})" for i, c in enumerate(claims))
    msgs = [
        {"role": "system", "content": prompts.VERIFIER},
        {
            "role": "user",
            "content": f"Claims:\n{numbered}\n\n{prompts.format_abstracts([by_pmid[p] for p in cited])}",
        },
    ]
    try:
        result = deps.llm.structured(msgs, Verification, usage=ctx.usage, node="verifier", max_tokens=4096)
        verdicts = {c.index: c.supported for c in result.checks}
    except LLMError:
        ctx.tool_errors += 1
        verdicts = {}

    checked = [
        {**c, "supported": c["valid_citations"] and verdicts.get(i, False), "verified": i in verdicts}
        for i, c in enumerate(claims)
    ]
    return _finish(state, checked, ctx)


def _finish(state: GraphState, checked: list[dict], ctx) -> GraphState:
    n = len(checked)
    supported = [c for c in checked if c["supported"]]
    unsupported_ratio = 1 - len(supported) / n if n else 1.0
    if not supported or unsupported_ratio > ctx.settings.max_unsupported_ratio:
        return {"claims": checked, "abstained": True, "abstain_reason": "unsupported_claims"}

    # Confidence: share of claims that survived verification, scaled down when evidence is thin.
    evidence_factor = min(1.0, len(state["relevant_pmids"]) / max(1, ctx.settings.min_relevant_passages))
    verified_factor = 1.0 if all(c.get("verified") for c in checked) else 0.8
    confidence = round((len(supported) / n) * evidence_factor * verified_factor, 3)
    return {"claims": checked, "abstained": False, "confidence": confidence}
