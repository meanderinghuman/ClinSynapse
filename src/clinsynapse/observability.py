"""Optional Langfuse tracing. Enabled when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set.

Every graph run is traced with per-node spans and the full state (including the retrieved
abstracts); each run carries our own trace_id in its metadata so an eval failure in
results.jsonl can be looked up in the Langfuse UI.
"""

import os
from functools import lru_cache


@lru_cache
def _handler_cls():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None
    return CallbackHandler


def langfuse_handler():
    cls = _handler_cls()
    return cls() if cls else None
