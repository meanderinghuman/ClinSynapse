"""FastAPI service.

    uv run uvicorn clinsynapse.api.app:app --reload

POST /ask         -> full result JSON
POST /ask/stream  -> server-sent events: one per finished node, then the result
"""

import asyncio
import json
from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from clinsynapse.api.highlight import best_sentence
from clinsynapse.config import get_settings
from clinsynapse.graph.build import Deps, answer, answer_stream
from clinsynapse.graph.prompts import DISCLAIMER
from clinsynapse.llm.client import LLMClient
from clinsynapse.retrieval.retriever import CachedRetriever

app = FastAPI(title="ClinSynapse", version="2.0.0", description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@lru_cache
def deps() -> Deps:
    s = get_settings()
    return Deps(llm=LLMClient(s), retriever=CachedRetriever(s))


def with_support(result: dict) -> dict:
    """Attach, per citation, the abstract sentence that best supports each claim citing it."""
    texts = {c["pmid"]: c["text"] for c in result["citations"]}
    for claim in result["claims"]:
        claim["support"] = {p: best_sentence(claim["text"], texts[p]) for p in claim["pmids"] if p in texts}
    result["disclaimer"] = DISCLAIMER
    return result


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask) -> dict:
    return with_support(answer(body.question, deps()).model_dump(exclude={"passages"}))


@app.post("/ask/stream")
async def ask_stream(body: Ask) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        try:
            for event in answer_stream(body.question, deps()):
                if event["event"] == "result":
                    event["result"] = with_support(event["result"])
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error event
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "detail": str(exc)})
        loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, produce)

    async def events():
        while (event := await queue.get()) is not None:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
