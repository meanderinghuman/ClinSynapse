"""Helpers shared by nodes: runtime access and per-node tracing."""

import functools
import time

from langchain_core.runnables import RunnableConfig

from clinsynapse.graph.state import QueryContext, TraceEvent


def ctx_of(config: RunnableConfig) -> QueryContext:
    return config["configurable"]["ctx"]


def deps_of(config: RunnableConfig):
    return config["configurable"]["deps"]


def traced(name: str):
    """Count a step, time the node, and record a trace event."""

    def wrap(fn):
        @functools.wraps(fn)
        def inner(state, config: RunnableConfig):
            ctx = ctx_of(config)
            ctx.steps += 1
            start = time.perf_counter()
            tokens_before = ctx.usage.total_tokens
            out = fn(state, config)
            ctx.events.append(
                TraceEvent(
                    node=name,
                    ms=(time.perf_counter() - start) * 1000,
                    detail={"tokens": ctx.usage.total_tokens - tokens_before},
                )
            )
            return out

        return inner

    return wrap
