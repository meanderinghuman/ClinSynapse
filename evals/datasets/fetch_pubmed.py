"""Fetch abstracts from PubMed (NCBI E-utilities) into data/extra_docs.jsonl.

Used to put BioASQ / golden-set gold documents into the index:

    uv run python -m evals.datasets.fetch_pubmed --bioasq evals/datasets/bioasq.json
    uv run python -m evals.datasets.fetch_pubmed --pmids 12345 67890
    uv run python -m clinsynapse.retrieval.build --append      # embed only the new docs
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from clinsynapse.config import get_settings
from clinsynapse.retrieval.corpus import Doc, load_extra_docs
from evals.datasets.items import bioasq, golden

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_pubmed_xml(xml: str) -> list[Doc]:
    docs = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for node in art.iter("AbstractText"):
            text = "".join(node.itertext()).strip()
            if text:
                label = node.get("Label")
                parts.append(f"{label.capitalize()}: {text}" if label else text)
        if pmid and parts:
            docs.append(Doc(pmid=pmid, title=title.strip(), text=" ".join(parts)))
    return docs


def fetch(pmids: list[str], batch: int = 200) -> list[Doc]:
    docs = []
    for i in range(0, len(pmids), batch):
        query = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids[i : i + batch]), "retmode": "xml"})
        with urllib.request.urlopen(f"{EFETCH}?{query}", timeout=60) as resp:
            docs.extend(parse_pubmed_xml(resp.read().decode()))
        time.sleep(0.4)  # NCBI allows 3 requests/s without an API key
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pmids", nargs="*", default=[])
    p.add_argument("--bioasq", help="BioASQ training JSON")
    p.add_argument("--golden", action="store_true", help="gold PMIDs from evals/datasets/golden.jsonl")
    args = p.parse_args()

    wanted = list(args.pmids)
    if args.bioasq:
        from pathlib import Path

        wanted += [pm for it in bioasq(Path(args.bioasq)) for pm in it.gold_pmids]
    if args.golden:
        wanted += [pm for it in golden() for pm in it.gold_pmids]

    out = get_settings().data_dir / "extra_docs.jsonl"
    have = {d.pmid for d in load_extra_docs(out)}
    todo = sorted(set(wanted) - have)
    print(f"{len(set(wanted))} PMIDs requested, {len(todo)} new")
    docs = fetch(todo)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for d in docs:
            f.write(json.dumps(d.model_dump()) + "\n")
    missing = len(todo) - len(docs)
    print(f"Wrote {len(docs)} abstracts to {out}" + (f" ({missing} had no abstract)" if missing else ""))


if __name__ == "__main__":
    main()
