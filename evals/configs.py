"""Named ablation presets: each is a set of Settings overrides on top of the full system."""

PRESETS: dict[str, dict] = {
    "full": {},
    "single_shot": {"use_planner": False, "use_grader": False, "use_verifier": False},
    "bm25": {"retrieval_mode": "bm25"},
    "dense": {"retrieval_mode": "dense"},
    "no_rerank": {"rerank": False},
    "no_grader": {"use_grader": False},
    "no_verifier": {"use_verifier": False},
    "alt_model": {"main_model": "DeepSeek-V4.1-Flash"},
}

# Retrieval-only ablation grid (no LLM calls).
RETRIEVAL_PRESETS: dict[str, dict] = {
    "bm25": {"retrieval_mode": "bm25", "rerank": False},
    "dense": {"retrieval_mode": "dense", "rerank": False},
    "hybrid": {"retrieval_mode": "hybrid", "rerank": False},
    "bm25+rerank": {"retrieval_mode": "bm25", "rerank": True},
    "dense+rerank": {"retrieval_mode": "dense", "rerank": True},
    "hybrid+rerank": {"retrieval_mode": "hybrid", "rerank": True},
}
