"""Build the local PubMed index.

    uv run python -m clinsynapse.retrieval.build              # full PubMedQA corpus (~273k)
    uv run python -m clinsynapse.retrieval.build --max-docs 50000
"""

import argparse

from tqdm import tqdm

from clinsynapse.config import get_settings
from clinsynapse.retrieval import medcpt
from clinsynapse.retrieval.bm25 import BM25Index
from clinsynapse.retrieval.corpus import Corpus, build_pubmedqa_corpus, load_extra_docs
from clinsynapse.retrieval.dense import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=512, help="docs per embed+upsert chunk")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    out = s.index_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpus ...")
    corpus = build_pubmedqa_corpus(args.max_docs)
    extra = [d for d in load_extra_docs(s.data_dir / "extra_docs.jsonl") if d.pmid not in corpus]
    corpus = Corpus(corpus.docs + extra)
    corpus.save(out / "corpus.jsonl")
    print(f"{len(corpus):,} docs ({len(extra)} extra)")

    pmids = [d.pmid for d in corpus.docs]
    print("Building BM25 ...")
    BM25Index.build(pmids, [f"{d.title} {d.text}" for d in corpus.docs]).save(out / "bm25")

    if args.skip_dense:
        return
    print(f"Embedding with MedCPT on {s.device} ...")
    dense = DenseIndex.open(out / "qdrant", url=s.qdrant_url, device=s.device)
    dense.create()
    for i in tqdm(range(0, len(corpus), args.batch), unit="chunk"):
        chunk = corpus.docs[i : i + args.batch]
        vecs = medcpt.encode_articles([d.title for d in chunk], [d.text for d in chunk], s.device)
        dense.upsert([d.pmid for d in chunk], vecs)
    print("Done.")


if __name__ == "__main__":
    main()
