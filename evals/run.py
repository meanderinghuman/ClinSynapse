"""Run the agent over a dataset and score it.

    uv run python -m evals.run --dataset pubmedqa --limit 200 --preset full
    uv run python -m evals.run --dataset pubmedqa --limit 200 --preset full --fault corrupt_numbers
    uv run python -m evals.run --dataset pubmedqa --retrieval-only           # retrieval ablation grid

Writes reports/runs/<name>/{results.jsonl, summary.json}.
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from clinsynapse.config import Settings, get_settings
from clinsynapse.graph.build import Deps, Result, answer
from clinsynapse.llm.cache import CacheMiss, ResponseCache, prompt_key
from clinsynapse.llm.client import LLMClient, LLMError
from clinsynapse.retrieval.retriever import CachedRetriever
from evals.configs import PRESETS, RETRIEVAL_PRESETS
from evals.datasets.items import Item, load
from evals.faults.injectors import Fault, attack_succeeded, get_fault
from evals.judges.base import Judge, get_judge
from evals.metrics import answer as am
from evals.metrics import behavior as bm
from evals.metrics import citation as cm
from evals.metrics import retrieval as rm
from evals.stats import bootstrap_ci, bootstrap_ci_indexed

PER_ITEM_METRICS = [
    "accuracy", "recall@8", "gold_in_context", "mrr", "ndcg@10",
    "citation_validity", "citation_precision", "citation_recall",
    "faithfulness", "faithfulness_draft", "key_fact_recall",
    "abstained", "refusal_correct", "false_refusal", "correct_abstention", "route_correct",
    "attack_success",
    "steps", "retrieval_loops", "retry_loop", "tool_error", "budget_hit",
    "latency_s", "tokens", "cost_usd",
]


class CachedJudge:
    """Caches judge verdicts in the shared cache so reruns and CI need no judge model."""

    def __init__(self, name: str, cache: ResponseCache):
        self.name, self.cache, self._judge = name, cache, None
        self._lock = threading.Lock()
        self.errors = 0

    @property
    def judge(self) -> Judge:
        with self._lock:
            self._judge = self._judge or get_judge(self.name)
        return self._judge

    def supported(self, claim: str, evidence: str) -> bool | None:
        """None if the judge failed; that claim is then excluded rather than scored."""
        key = prompt_key({"kind": "judge", "judge": self.name, "claim": claim, "evidence": evidence})
        if (hit := self.cache.get(key)) is not None:
            return hit["supported"]
        if get_settings().replay_only:
            raise CacheMiss(f"judge verdict not in cache ({self.name}); re-record the smoke cache")
        try:
            verdict = bool(self.judge.supported(claim, evidence))
        except LLMError:
            with self._lock:
                self.errors += 1
            return None
        self.cache.put(key, {"supported": verdict})
        return verdict


def score_item(item: Item, res: Result, judge: CachedJudge, extra_evidence: dict[str, str]) -> dict:
    gold = set(item.gold_pmids)
    evidence = {**extra_evidence, **{p.pmid: p.text for p in res.passages}}
    kept = [c for c in res.claims if c.supported] if not res.abstained else []
    cited = [p for c in kept for p in c.pmids]

    def judged(claims) -> float | None:
        if not claims:
            return None
        verdicts = []
        for c in claims:
            ev = "\n\n".join(evidence[p] for p in c.pmids if p in evidence)
            v = judge.supported(c.text, ev) if ev else False  # citing unseen text = unsupported
            if v is not None:
                verdicts.append(v)
        return float(np.mean(verdicts)) if verdicts else None

    m: dict = {
        "accuracy": am.accuracy_item(res.verdict, item.label) if item.label else None,
        "recall@8": rm.recall_at_k(res.retrieved_pmids, gold, 8),
        "gold_in_context": rm.recall_at_k(res.retrieved_pmids, gold, len(res.retrieved_pmids)),
        "citation_validity": cm.citation_validity(cited, res.retrieved_pmids),
        "citation_precision": cm.citation_precision(cited, gold),
        "citation_recall": cm.citation_recall(cited, gold) if not res.abstained else None,
        "faithfulness": judged(kept),
        "faithfulness_draft": judged(res.claims),
        "key_fact_recall": am.key_fact_recall(
            [v for f in item.key_facts if (v := judge.supported(f, res.answer)) is not None]
        ) if item.key_facts and not res.abstained else None,
        "abstained": float(res.abstained),
        "steps": res.steps,
        "retrieval_loops": res.retrieval_loops,
        "retry_loop": float(res.retrieval_loops > 0),
        "tool_error": float(res.tool_errors > 0),
        "budget_hit": float(res.budget_hit is not None),
        "latency_s": res.latency_s,
        "tokens": res.tokens,
        "cost_usd": res.cost_usd,
    }
    if item.source == "router":
        m["route_correct"] = float(res.route == item.expected_route)
        m["abstained"] = None
    else:
        declined = res.abstained or res.route != "research"
        m["refusal_correct"] = bm.refusal_correct(declined, item.answerable)
        m["false_refusal"] = bm.false_refusal(declined, item.answerable)
        m["correct_abstention"] = bm.correct_abstention(declined, item.answerable)
    return m


def summarize(records: list[dict]) -> dict:
    summary: dict = {"n": len(records), "metrics": {}}
    for name in PER_ITEM_METRICS:
        vals = [r["metrics"].get(name) for r in records]
        if any(v is not None for v in vals):
            summary["metrics"][name] = bootstrap_ci(vals).model_dump()

    labeled = [r for r in records if r["item"].get("label")]
    if labeled:
        preds = [r["result"]["verdict"] for r in labeled]
        golds = [r["item"]["label"] for r in labeled]
        f1 = bootstrap_ci_indexed(
            len(labeled), lambda idx: am.macro_f1([preds[i] for i in idx], [golds[i] for i in idx])
        )
        summary["metrics"]["macro_f1"] = f1.model_dump()
        # Always answering the most common label: the floor any accuracy must clear.
        majority = max(set(golds), key=golds.count)
        summary["baselines"] = {"majority_label": majority,
                                "majority_accuracy": golds.count(majority) / len(golds),
                                "majority_macro_f1": am.macro_f1([majority] * len(golds), golds)}

    lat = [r["metrics"]["latency_s"] for r in records]
    cost = [r["metrics"]["cost_usd"] for r in records]
    summary["ops"] = {
        "p50_latency_s": float(np.percentile(lat, 50)),
        "p95_latency_s": float(np.percentile(lat, 95)),
        "tokens_per_query": float(np.mean([r["metrics"]["tokens"] for r in records])),
        "usd_per_1k_queries": float(np.mean(cost) * 1000),
        "cache_hit_rate": float(np.mean([r.get("cache_hit", False) for r in records])),
    }
    return summary


def preflight(llm: LLMClient, settings: Settings) -> None:
    """One uncached call before a long run, so a dead endpoint fails in seconds, not hours."""
    if settings.replay_only:
        return
    try:
        llm.client.chat.completions.create(
            model=settings.main_model, max_tokens=16, messages=[{"role": "user", "content": "ok"}])
    except Exception as exc:
        raise SystemExit(f"LLM endpoint unusable, not starting the run: {exc}") from exc


def run_agent(items: list[Item], settings: Settings, fault: Fault, judge_name: str,
              workers: int, extra_evidence: dict[str, str]) -> list[dict]:
    llm = LLMClient(settings)
    preflight(llm, settings)
    retriever = CachedRetriever(settings)
    judge = CachedJudge(judge_name, ResponseCache(settings.cache_path))

    def one(item: Item) -> dict:
        deps = Deps(llm=llm, retriever=fault.wrap_retriever(retriever, item))
        res = answer(item.question, deps, settings)
        res = fault.mutate(res, item, random.Random(item.id))  # str seeds are deterministic
        metrics = score_item(item, res, judge, extra_evidence)
        if fault.name == "prompt_injection":
            metrics["attack_success"] = float(attack_succeeded(res))
        return {"item": item.model_dump(), "result": res.model_dump(exclude={"passages"}),
                "evidence": [p.model_dump() for p in res.passages],
                "metrics": metrics, "cache_hit": res.llm_calls > 0 and res.cache_hits == res.llm_calls}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(one, items), total=len(items), desc="items"))
    if judge.errors:
        print(f"warning: {judge.errors} judge call(s) failed; those claims are excluded from faithfulness")
    return records


def run_retrieval(items: list[Item], base: Settings) -> dict:
    out = {}
    for name, overrides in RETRIEVAL_PRESETS.items():
        s = base.model_copy(update=overrides)
        r = CachedRetriever(s)
        per = {"recall@1": [], "recall@8": [], "recall@50": [], "mrr": [], "ndcg@10": [], "ms": []}
        for it in tqdm(items, desc=name):
            t = time.perf_counter()
            ranked = [p.pmid for p in r.retrieve(it.question, k=50)]
            per["ms"].append((time.perf_counter() - t) * 1000)
            gold = set(it.gold_pmids)
            per["recall@1"].append(rm.recall_at_k(ranked, gold, 1))
            per["recall@8"].append(rm.recall_at_k(ranked, gold, 8))
            per["recall@50"].append(rm.recall_at_k(ranked, gold, 50))
            per["mrr"].append(rm.mrr(ranked, gold))
            per["ndcg@10"].append(rm.ndcg_at_k(ranked, gold, 10))
        out[name] = {"per_item": per, "summary": {k: bootstrap_ci(v).model_dump() for k, v in per.items()}}
        print(name, {k: str(bootstrap_ci(v)) for k, v in per.items() if k != "ms"})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pubmedqa", choices=["pubmedqa", "pubmedqa-dev", "golden", "router", "bioasq"])
    p.add_argument("--limit", type=int)
    p.add_argument("--preset", default="full", choices=list(PRESETS))
    p.add_argument("--fault", default="none",
                   choices=["none", "corrupt_numbers", "swap_citations", "remove_gold", "prompt_injection"])
    p.add_argument("--judge", default="llm-alt", choices=["nli", "llm-main", "llm-alt"],
                   help="faithfulness judge; llm-alt differs from the gpt-oss verifier by design")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=None,
                   help="per-query token budget override (reasoning models need more)")
    p.add_argument("--max-seconds", type=float, default=300.0,
                   help="per-query wall-clock budget; looser than serving (90s) so provider queueing "
                        "latency is not scored as answer quality. Latency is reported separately.")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--cache", type=Path, help="override cache file (e.g. evals/smoke/cache.sqlite)")
    p.add_argument("--name", help="output run name")
    p.add_argument("--out", type=Path, default=Path("reports/runs"))
    args = p.parse_args()

    base = get_settings().model_copy(
        update={"temperature": 0.0, "web_search": False, "max_seconds": args.max_seconds})
    if args.max_tokens:
        base = base.model_copy(update={"max_tokens": args.max_tokens})
    if args.cache:
        base = base.model_copy(update={"cache_file": args.cache})
    items = load(args.dataset, args.limit)
    if not items:
        raise SystemExit(f"Dataset {args.dataset!r} is empty")

    n = len(items)
    name = args.name or (f"{args.dataset}-retrieval-n{n}" if args.retrieval_only
                         else f"{args.dataset}-{args.preset}" + (f"-{args.fault}" if args.fault != "none" else "") + f"-n{n}")
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)

    if args.retrieval_only:
        res = run_retrieval(items, base)
        (out / "retrieval.json").write_text(json.dumps(res, indent=2))
        print(f"Wrote {out}/retrieval.json")
        return

    settings = base.model_copy(update=PRESETS[args.preset])
    extra: dict[str, str] = {}
    corpus_pmids = None
    if args.fault == "swap_citations":
        from clinsynapse.retrieval.corpus import Corpus

        corpus = Corpus.load(settings.index_dir / "corpus.jsonl")
        corpus_pmids = [d.pmid for d in corpus.docs]
        extra = {d.pmid: d.text for d in corpus.docs}
    fault = get_fault(args.fault, corpus_pmids)

    records = run_agent(items, settings, fault, args.judge, args.workers, extra)
    with (out / "results.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records)
    summary.update({"dataset": args.dataset, "preset": args.preset, "fault": args.fault,
                    "judge": args.judge, "settings": json.loads(settings.model_dump_json(
                        exclude={"llm_api_key", "data_dir", "reports_dir", "cache_file"}))})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{name}  (n={n})")
    for k, v in summary["metrics"].items():
        print(f"  {k:20s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
    print("  " + json.dumps(summary["ops"]))


if __name__ == "__main__":
    main()
