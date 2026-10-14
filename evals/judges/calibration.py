"""Judge calibration against hand labels.

1. Export claims from an eval run for labeling:
     uv run python -m evals.judges.calibration export reports/<run> --n 150
   then fill in "label": true/false (supported?) in evals/judges/labels.jsonl.
2. Score the judges:
     uv run python -m evals.judges.calibration score --judges nli llm-main llm-alt

Hallucination precision/recall treat "unsupported" as the positive class: of the claims a judge
flags as hallucinated, how many really are (precision); of the real hallucinations, how many it
catches (recall).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from evals.judges.base import get_judge
from evals.stats import bootstrap_ci, cohens_kappa

LABELS = Path(__file__).parent / "labels.jsonl"


def export(run_dir: Path, n: int, seed: int = 0) -> None:
    rows = []
    for line in (run_dir / "results.jsonl").open():
        r = json.loads(line)
        evidence = {c["pmid"]: c["text"] for c in r.get("evidence", [])}
        for c in r["result"]["claims"]:
            ev = "\n\n".join(evidence[p] for p in c["pmids"] if p in evidence)
            if ev:
                rows.append({"item_id": r["item"]["id"], "claim": c["text"], "pmids": c["pmids"],
                             "evidence": ev, "label": None})
    random.Random(seed).shuffle(rows)
    existing = LABELS.read_text() if LABELS.exists() else ""
    with LABELS.open("a") as f:
        for row in rows[:n]:
            f.write(json.dumps(row) + "\n")
    print(f"Appended {min(n, len(rows))} claims to {LABELS} ({len(existing.splitlines())} already there)")


def score(judge_names: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in LABELS.open() if line.strip()]
    rows = [r for r in rows if r.get("label") is not None]
    if not rows:
        raise SystemExit(f"No labeled rows in {LABELS}")
    gold = np.array([bool(r["label"]) for r in rows])
    table = []
    for name in judge_names:
        judge = get_judge(name)
        start = time.perf_counter()
        pred = np.array([judge.supported(r["claim"], r["evidence"]) for r in rows])
        secs = time.perf_counter() - start
        halluc_true, halluc_pred = ~gold, ~pred
        tp = np.sum(halluc_true & halluc_pred)
        row = {
            "judge": name,
            "n": len(rows),
            "accuracy": str(bootstrap_ci((pred == gold).astype(float))),
            "kappa": round(cohens_kappa(gold, pred), 3),
            "halluc_precision": round(tp / max(1, halluc_pred.sum()), 3),
            "halluc_recall": round(tp / max(1, halluc_true.sum()), 3),
            "sec_per_claim": round(secs / len(rows), 3),
            "usd_per_1k_claims": round(getattr(judge, "usage", None).cost_usd / len(rows) * 1000, 4)
            if hasattr(judge, "usage") else 0.0,
        }
        table.append(row)
        print(row)
    out = Path("reports/judge_calibration.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("run_dir", type=Path)
    e.add_argument("--n", type=int, default=150)
    s = sub.add_parser("score")
    s.add_argument("--judges", nargs="+", default=["nli", "llm-main", "llm-alt"])
    args = p.parse_args()
    export(args.run_dir, args.n) if args.cmd == "export" else score(args.judges)


if __name__ == "__main__":
    main()
