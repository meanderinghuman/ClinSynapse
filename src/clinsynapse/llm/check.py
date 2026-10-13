"""Verify the LLM endpoint: key works, configured models exist, JSON output parses.

    uv run python -m clinsynapse.llm.check
"""

from pydantic import BaseModel

from clinsynapse.config import get_settings
from clinsynapse.llm.client import LLMClient, Usage


class Ping(BaseModel):
    answer: str
    confident: bool


def main() -> None:
    s = get_settings().model_copy(update={"llm_cache": False})
    client = LLMClient(s)
    print(f"Endpoint: {s.llm_base_url}")
    available = sorted(m.id for m in client.client.models.list().data)
    print("Models:", ", ".join(available))
    for model in dict.fromkeys([s.main_model, s.small_model, s.alt_model]):
        if model not in available:
            print(f"  ✗ {model} is not served here; set CS_MAIN_MODEL / CS_SMALL_MODEL / CS_ALT_MODEL")
            continue
        usage = Usage()
        out = client.structured(
            [{"role": "user", "content": "Is aspirin an NSAID? Answer in one word."}],
            Ping, model=model, usage=usage, node="check", max_tokens=512,
        )
        mode = "json_object" if client._json_mode else "prompt-only JSON (endpoint rejected response_format)"
        print(f"  ✓ {model}: {out.answer!r} · {usage.total_tokens} tokens · {usage.seconds:.1f}s · {mode}")


if __name__ == "__main__":
    main()
