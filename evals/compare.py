"""Paired A/B comparison of two eval runs on the same items.

    uv run python -m evals.compare reports/runs/pubmedqa-single_shot-n200 reports/runs/pubmedqa-full-n200
"""

import argparse
import json
from pathlib import Path

from evals.run import PER_ITEM_METRICS
from evals.stats import Comparison, compare


def load_records(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "results.jsonl").open() as f:
        return {(r := json.loads(line))["item"]["id"]: r for line in f}


def compare_runs(a_dir: Path, b_dir: Path, metrics: list[str] | None = None) -> list[Comparison]:
    a, b = load_records(a_dir), load_records(b_dir)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("runs share no items")
    out = []
    for m in metrics or PER_ITEM_METRICS:
        va = [a[i]["metrics"].get(m) for i in ids]
        vb = [b[i]["metrics"].get(m) for i in ids]
        if any(v is not None for v in va) and any(v is not None for v in vb):
            out.append(compare(m, va, vb))
    return out


def to_markdown(rows: list[Comparison], a_name: str, b_name: str) -> str:
    lines = [f"| metric | {a_name} | {b_name} | Δ [95% CI] | p |", "|---|---|---|---|---|"]
    for c in rows:
        sig = " **" if c.p_value < 0.05 else " "
        lines.append(
            f"| {c.metric} | {c.a:.3f} | {c.b:.3f} | {c.diff:+.3f} [{c.diff_lo:+.3f}, {c.diff_hi:+.3f}] |"
            f"{sig}{c.p_value:.4f}{sig.strip()} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--metrics", nargs="*")
    args = p.parse_args()
    rows = compare_runs(args.a, args.b, args.metrics)
    print(to_markdown(rows, args.a.name, args.b.name))


if __name__ == "__main__":
    main()
