"""CI quality gate: fail if a guarded metric drops beyond its threshold AND the drop is significant.

    uv run python -m evals.gate reports/baseline/smoke reports/runs/smoke

Requiring both a practical threshold and p < alpha means noise can't fail the build and a tiny
real regression can't either; a meaningful, real drop always does.
"""

import argparse
import sys
from pathlib import Path

from evals.compare import compare_runs, to_markdown

# metric -> maximum tolerated absolute drop
GUARDS = {
    "faithfulness": 0.05,
    "citation_validity": 0.02,
    "accuracy": 0.05,
    "refusal_correct": 0.05,
}
ALPHA = 0.05


def gate(baseline: Path, current: Path, guards: dict[str, float] = GUARDS, alpha: float = ALPHA):
    rows = compare_runs(baseline, current, list(guards))
    failures = [
        c for c in rows if c.diff < -guards[c.metric] and c.p_value < alpha
    ]
    return rows, failures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--summary-md", type=Path, help="append a markdown table (e.g. $GITHUB_STEP_SUMMARY)")
    args = p.parse_args()

    rows, failures = gate(args.baseline, args.current)
    table = to_markdown(rows, "baseline", "current")
    verdict = "FAIL" if failures else "PASS"
    report = f"### Eval gate: {verdict}\n\n{table}\n"
    if failures:
        report += "\nRegressed: " + ", ".join(
            f"{c.metric} ({c.diff:+.3f}, p={c.p_value:.4f})" for c in failures
        ) + "\n"
    print(report)
    if args.summary_md:
        with args.summary_md.open("a") as f:
            f.write(report)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
