"""Streamlit UI over the streaming API.

    uv run --group ui streamlit run ui/app.py
"""

import html
import json
import os

import httpx
import streamlit as st

API = os.getenv("CS_API_URL", "http://localhost:8000")
NODE_LABELS = {
    "router": "Checking scope", "planner": "Planning sub-questions", "retriever": "Searching PubMed",
    "grader": "Grading evidence", "synthesizer": "Writing cited answer", "verifier": "Verifying claims",
    "abstain": "Abstaining", "finalize": "Done",
}

st.set_page_config(page_title="ClinSynapse", page_icon="🧬", layout="wide")
st.title("ClinSynapse")
st.caption("Answers from PubMed with verifiable citations. Research use only, not medical advice.")

question = st.text_input("Ask a biomedical research question",
                         placeholder="Does metformin reduce cancer risk in type 2 diabetes?")


def highlight(abstract: str, sentence: str) -> str:
    a, s = html.escape(abstract), html.escape(sentence)
    return a.replace(s, f"<mark>{s}</mark>", 1) if s else a


if st.button("Ask", type="primary") and question:
    status = st.status("Working…", expanded=True)
    result = None
    with httpx.stream("POST", f"{API}/ask/stream", json={"question": question}, timeout=180) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["event"] == "node":
                status.write(f"✓ {NODE_LABELS.get(event['node'], event['node'])} · {event['elapsed_s']}s")
            elif event["event"] == "error":
                status.update(label="Error", state="error")
                st.error(event["detail"])
            else:
                result = event["result"]
    if result:
        status.update(label=f"Finished in {result['latency_s']}s", state="complete", expanded=False)
        if result["route"] != "research":
            st.warning(result["answer"])
        elif result["abstained"]:
            st.info(result["answer"])
        else:
            if result["verdict"]:
                st.subheader(f"Answer: {result['verdict'].capitalize()}")
            st.metric("Confidence", f"{result['confidence']:.0%}")
            texts = {c["pmid"]: c for c in result["citations"]}
            for claim in result["claims"]:
                if not claim["supported"]:
                    continue
                links = " ".join(f"[PMID:{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in claim["pmids"])
                st.markdown(f"- {claim['text']} {links}")
                for pmid, sent in claim.get("support", {}).items():
                    with st.expander(f"Evidence · PMID {pmid}"):
                        st.markdown(highlight(texts[pmid]["text"], sent), unsafe_allow_html=True)
            dropped = [c for c in result["claims"] if not c["supported"]]
            if dropped:
                with st.expander(f"{len(dropped)} claim(s) removed by the verifier"):
                    for c in dropped:
                        st.markdown(f"- ~~{c['text']}~~")
        st.caption(f"trace {result['trace_id']} · {result['steps']} steps · {result['tokens']} tokens · "
                   f"${result['cost_usd']:.4f}")
