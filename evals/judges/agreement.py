"""Inter-judge agreement on the claims of an eval run (no human labels needed).

    uv run python -m evals.judges.agreement reports/runs/pubmedqa-full-n300

Re-judges every claim with each judge (verdicts are cached) and reports pairwise raw agreement,
Cohen's kappa, and each judge's supported rate. Agreement is not accuracy: two judges can agree
and both be wrong. It shows how much the faithfulness number depends on the choice of judge.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from clinsynapse.config import get_settings
from clinsynapse.llm.cache import ResponseCache
from evals.run import CachedJudge
from evals.stats import cohens_kappa

JUDGES = ["nli", "llm-main", "llm-alt"]


def claims_of(run_dir: Path) -> list[tuple[str, str]]:
    out = []
    for line in (run_dir / "results.jsonl").open():
        r = json.loads(line)
        evidence = {p["pmid"]: p["text"] for p in r.get("evidence", [])}
        for c in r["result"]["claims"]:
            ev = "\n\n".join(evidence[p] for p in c["pmids"] if p in evidence)
            if ev:
                out.append((c["text"], ev))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--judges", nargs="+", default=JUDGES)
    args = p.parse_args()

    claims = claims_of(args.run_dir)
    cache = ResponseCache(get_settings().cache_path)
    verdicts = {}
    for name in args.judges:
        judge = CachedJudge(name, cache)
        verdicts[name] = [judge.supported(c, ev) for c, ev in claims]

    rows = {"n_claims": len(claims), "supported_rate": {}, "pairs": []}
    for name, v in verdicts.items():
        ok = [x for x in v if x is not None]
        rows["supported_rate"][name] = round(float(np.mean(ok)), 3) if ok else None
    for a, b in itertools.combinations(args.judges, 2):
        both = [(x, y) for x, y in zip(verdicts[a], verdicts[b]) if x is not None and y is not None]
        xa, xb = zip(*both) if both else ((), ())
        rows["pairs"].append({"a": a, "b": b, "n": len(both),
                              "agreement": round(float(np.mean(np.array(xa) == np.array(xb))), 3) if both else None,
                              "kappa": round(cohens_kappa(xa, xb), 3) if both else None})
    out = Path("reports/judge_agreement.json")
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
