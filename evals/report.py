"""Render every run under reports/runs into one HTML report (+ a markdown table for the README).

    uv run python -m evals.report
"""

import argparse
import html
import json
from pathlib import Path

HEADLINE = ["accuracy", "macro_f1", "recall@8", "citation_validity", "citation_precision",
            "faithfulness", "faithfulness_draft", "refusal_correct", "false_refusal",
            "attack_success", "route_correct"]
OPS = ["p50_latency_s", "p95_latency_s", "tokens_per_query", "usd_per_1k_queries"]


def fmt(ci: dict | None) -> str:
    if not ci or ci.get("n", 0) == 0:
        return "–"
    return f"{ci['mean']:.3f} <small>[{ci['lo']:.3f}, {ci['hi']:.3f}]</small>"


def load_runs(root: Path) -> list[tuple[str, dict]]:
    return [(p.parent.name, json.loads(p.read_text())) for p in sorted(root.glob("*/summary.json"))]


def load_retrieval(root: Path) -> list[tuple[str, dict]]:
    return [(p.parent.name, json.loads(p.read_text())) for p in sorted(root.glob("*/retrieval.json"))]


def markdown_table(runs: list[tuple[str, dict]]) -> str:
    cols = [m for m in HEADLINE if any(m in s["metrics"] for _, s in runs)]
    lines = ["| run | " + " | ".join(cols) + " | $/1k q | p95 s |",
             "|---" * (len(cols) + 3) + "|"]
    for name, s in runs:
        cells = []
        for m in cols:
            ci = s["metrics"].get(m)
            cells.append(f"{ci['mean']:.3f} [{ci['lo']:.2f}, {ci['hi']:.2f}]" if ci and ci["n"] else "–")
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {s['ops']['usd_per_1k_queries']:.2f} | {s['ops']['p95_latency_s']:.1f} |")
    floors = {s["dataset"]: s["baselines"] for _, s in runs if s.get("baselines")}
    for ds, b in floors.items():
        lines.append(f"\n_{ds}: always answering '{b['majority_label']}' scores accuracy "
                     f"{b['majority_accuracy']:.3f}, macro-F1 {b['majority_macro_f1']:.3f}._")
    return "\n".join(lines)


def render(root: Path) -> str:
    runs, retr = load_runs(root), load_retrieval(root)
    cols = [m for m in HEADLINE if any(m in s["metrics"] for _, s in runs)]
    rows = "".join(
        f"<tr><th>{html.escape(n)}</th><td>{s['n']}</td>"
        + "".join(f"<td>{fmt(s['metrics'].get(m))}</td>" for m in cols)
        + "".join(f"<td>{s['ops'][o]:.2f}</td>" for o in OPS) + "</tr>"
        for n, s in runs
    )
    retr_html = ""
    for name, r in retr:
        metrics = ["recall@1", "recall@8", "recall@50", "mrr", "ndcg@10", "ms"]
        body = "".join(
            f"<tr><th>{cfg}</th>" + "".join(f"<td>{fmt(v['summary'][m])}</td>" for m in metrics) + "</tr>"
            for cfg, v in r.items()
        )
        retr_html += (f"<h2>Retrieval ablation · {html.escape(name)}</h2><table><tr><th></th>"
                      + "".join(f"<th>{m}</th>" for m in metrics) + f"</tr>{body}</table>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ClinSynapse evals</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1b1f24;--muted:#667;--line:#e3e6ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111418;--fg:#e8eaed;--muted:#99a;--line:#2a2f36}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:24px 16px}}
table{{border-collapse:collapse;margin:12px 0 32px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}}
th:first-child{{text-align:left}} small{{color:var(--muted)}}
</style></head><body>
<h1>ClinSynapse eval report</h1><p>Means with bootstrap 95% CIs.</p>
<h2>Agent runs</h2><table><tr><th>run</th><th>n</th>{''.join(f'<th>{m}</th>' for m in cols)}
{''.join(f'<th>{o}</th>' for o in OPS)}</tr>{rows}</table>{retr_html}</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("reports/runs"))
    p.add_argument("--out", type=Path, default=Path("reports/report.html"))
    args = p.parse_args()
    args.out.write_text(render(args.runs))
    runs = load_runs(args.runs)
    if runs:
        (args.out.parent / "results.md").write_text(markdown_table(runs) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
