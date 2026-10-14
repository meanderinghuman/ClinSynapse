"""Re-score a run's faithfulness with a different judge (verdicts are cached).

    uv run python -m evals.judges.rescore reports/runs/<a> reports/runs/<b> --judges nli llm-main llm-alt

Needed for model head-to-heads: the default judge (llm-alt) is the same model as the alt_model
preset, so comparing models with one judge would let a model grade its own answers.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from clinsynapse.config import get_settings
from clinsynapse.llm.cache import ResponseCache
from evals.run import CachedJudge
from evals.stats import compare


def per_item_faithfulness(run_dir: Path, judge: CachedJudge, workers: int = 8) -> dict[str, float | None]:
    def score(line: str) -> tuple[str, float | None]:
        r = json.loads(line)
        res = r["result"]
        evidence = {p["pmid"]: p["text"] for p in r.get("evidence", [])}
        kept = [c for c in res["claims"] if c["supported"]] if not res["abstained"] else []
        verdicts = []
        for c in kept:
            ev = "\n\n".join(evidence[p] for p in c["pmids"] if p in evidence)
            v = judge.supported(c["text"], ev) if ev else False
            if v is not None:
                verdicts.append(v)
        return r["item"]["id"], float(np.mean(verdicts)) if verdicts else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(score, (run_dir / "results.jsonl").read_text().splitlines()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--judges", nargs="+", default=["nli", "llm-main", "llm-alt"])
    args = p.parse_args()

    cache = ResponseCache(get_settings().cache_path)
    print(f"| judge | {args.a.name} | {args.b.name} | Δ [95% CI] | p |\n|---|---|---|---|---|")
    rows = []
    for name in args.judges:
        judge = CachedJudge(name, cache)
        fa, fb = per_item_faithfulness(args.a, judge), per_item_faithfulness(args.b, judge)
        ids = sorted(set(fa) & set(fb))
        c = compare(f"faithfulness[{name}]", [fa[i] for i in ids], [fb[i] for i in ids])
        rows.append(c.model_dump())
        print(f"| {name} | {c.a:.3f} | {c.b:.3f} | {c.diff:+.3f} [{c.diff_lo:+.3f}, {c.diff_hi:+.3f}] | {c.p_value:.4f} |")
    out = Path("reports") / f"rescore_{args.a.name}_vs_{args.b.name}.json"
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
