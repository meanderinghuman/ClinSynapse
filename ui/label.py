"""Click-through labeler for judge calibration (evals/judges/labels.jsonl).

    uv run python -m evals.judges.calibration export reports/runs/<run> --n 150
    uv run --group ui streamlit run ui/label.py

Label each claim against its cited evidence. Judges never see your labels until scoring.
"""

import html
import json
import re
from pathlib import Path

import streamlit as st

LABELS = Path(__file__).resolve().parents[1] / "evals" / "judges" / "labels.jsonl"
RULE = ("**Supported** = every part of the claim (numbers, direction of effect, population, "
        "intervention) is stated in or directly entailed by the evidence. Anything added, "
        "missing, or more specific than the evidence = **Unsupported**.")


def load() -> list[dict]:
    return [json.loads(line) for line in LABELS.open() if line.strip()] if LABELS.exists() else []


def save(rows: list[dict]) -> None:
    LABELS.write_text("".join(json.dumps(r) + "\n" for r in rows))


def mark_numbers(text: str) -> str:
    """Highlight numbers, the most common source of unsupported claims."""
    return re.sub(r"(\d+(?:\.\d+)?%?)", r"<mark>\1</mark>", html.escape(text))


st.set_page_config(page_title="Claim labeler", page_icon="🏷️", layout="wide")
rows = load()
if not rows:
    st.error(f"No claims in {LABELS}. Export some first (see module docstring).")
    st.stop()

done = sum(r.get("label") is not None for r in rows)
st.progress(done / len(rows), text=f"{done} / {len(rows)} labeled")
todo = [i for i, r in enumerate(rows) if r.get("label") is None]
if not todo:
    st.success("All claims labeled. Run `make calibration`.")
    st.stop()

i = todo[0]
row = rows[i]
st.caption(RULE)
st.subheader("Claim")
st.markdown(f"<div style='font-size:1.15rem'>{mark_numbers(row['claim'])}</div>", unsafe_allow_html=True)
st.caption("Cites " + ", ".join(f"[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in row["pmids"]))
st.subheader("Evidence")
st.markdown(mark_numbers(row["evidence"]), unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
if c1.button("✅ Supported", use_container_width=True, type="primary"):
    rows[i]["label"] = True
    save(rows)
    st.rerun()
if c2.button("❌ Unsupported", use_container_width=True):
    rows[i]["label"] = False
    save(rows)
    st.rerun()
if c3.button("Skip (remove)", use_container_width=True):
    rows.pop(i)
    save(rows)
    st.rerun()
