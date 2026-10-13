"""Central settings. Every knob an ablation flips lives here so eval configs are explicit."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="CS_", extra="ignore")

    # LLMs: any OpenAI-compatible endpoint. Defaults target Virginia Tech ARC's LLM API
    # (https://docs.arc.vt.edu/ai/011_llm_api_arc_vt_edu.html); Groq/Ollama/vLLM work via base URL.
    llm_base_url: str = Field(
        "https://llm-api.arc.vt.edu/api/v1",
        validation_alias=AliasChoices("CS_LLM_BASE_URL", "OPEN_AI_ENDPOINT"),
    )
    llm_api_key: str = Field(
        "", validation_alias=AliasChoices("CS_LLM_API_KEY", "OPEN_AI_API_KEY", "ARC_API_KEY")
    )
    main_model: str = Field(
        "gpt-oss-120b", validation_alias=AliasChoices("CS_MAIN_MODEL", "OPEN_AI_MODEL")
    )
    small_model: str = "gpt-oss-120b"  # router + LLM-small judge; ARC hosts no ~8B model
    alt_model: str = "DeepSeek-V4.1-Flash"  # head-to-head comparison model
    temperature: float = 0.0
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    json_mode: bool = True  # response_format=json_object; auto-disabled if the endpoint rejects it
    max_concurrency: int = 8  # ARC: 10 concurrent per model, 20 in flight per user
    llm_cache: bool = True

    # Data
    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"

    # Retrieval
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True
    candidates_per_retriever: int = 50
    top_k: int = 8
    rrf_k: int = 60
    device: str = "mps"
    qdrant_url: str | None = None  # None = embedded local mode under data/index/qdrant

    # Agent
    use_planner: bool = True
    use_grader: bool = True
    use_verifier: bool = True
    max_retrieval_loops: int = 2
    min_relevant_passages: int = 1  # one strong abstract can suffice (typical in PubMedQA)
    max_unsupported_ratio: float = 0.5

    # Budgets per query
    max_steps: int = 20
    max_tokens: int = 40_000
    max_seconds: float = 90.0

    # Web search is optional and always off during evals
    web_search: bool = False

    # One SQLite file caches LLM calls, retrieval results and judge verdicts.
    cache_file: Path | None = None
    # CI: never call models or load the index; any cache miss is a hard error.
    replay_only: bool = False

    @property
    def cache_path(self) -> Path:
        return self.cache_file or self.data_dir / "llm_cache.sqlite"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
