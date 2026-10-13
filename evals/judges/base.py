"""A judge decides whether a claim is supported by evidence text."""

from typing import Protocol


class Judge(Protocol):
    name: str

    def supported(self, claim: str, evidence: str) -> bool: ...


def get_judge(name: str) -> Judge:
    if name == "nli":
        from evals.judges.nli import NLIJudge

        return NLIJudge()
    if name in ("llm-main", "llm-alt"):
        from evals.judges.llm import LLMJudge

        return LLMJudge(which=name.split("-")[1])
    raise ValueError(f"unknown judge {name!r}")
