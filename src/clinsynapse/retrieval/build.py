"""Build the local PubMed index.

    uv run python -m clinsynapse.retrieval.build              # full PubMedQA corpus (~273k)
    uv run python -m clinsynapse.retrieval.build --max-docs 50000
    uv run python -m clinsynapse.retrieval.build --append     # add data/extra_docs.jsonl only

Titles are stored for display but never indexed: PubMedQA abstracts have no usable title
(it is the question), so indexing titles only for fetched docs would bias retrieval toward them.
"""

import argparse
import hashlib

from tqdm import tqdm

from clinsynapse.config import get_settings
from clinsynapse.retrieval import medcpt
from clinsynapse.retrieval.bm25 import BM25Index
from clinsynapse.retrieval.corpus import Corpus, Doc, build_pubmedqa_corpus, load_extra_docs
from clinsynapse.retrieval.dense import DenseIndex


def index_version(corpus: Corpus) -> str:
    digest = hashlib.sha256("\n".join(sorted(d.pmid for d in corpus.docs)).encode()).hexdigest()
    return f"{len(corpus)}-{digest[:12]}"


def embed(dense: DenseIndex, docs: list[Doc], batch: int, device: str) -> None:
    for i in tqdm(range(0, len(docs), batch), unit="chunk"):
        chunk = docs[i : i + batch]
        vecs = medcpt.encode_articles([""] * len(chunk), [d.text for d in chunk], device)
        dense.upsert([d.pmid for d in chunk], vecs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=512, help="docs per embed+upsert chunk")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--append", action="store_true", help="add extra_docs.jsonl to the existing index")
    args = parser.parse_args()

    s = get_settings()
    out = s.index_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpus ...")
    base = Corpus.load(out / "corpus.jsonl") if args.append else build_pubmedqa_corpus(args.max_docs)
    extra = [d for d in load_extra_docs(s.data_dir / "extra_docs.jsonl") if d.pmid not in base]
    if args.append and not extra:
        print("Nothing new to append.")
        return
    corpus = Corpus(base.docs + extra)
    corpus.save(out / "corpus.jsonl")
    print(f"{len(corpus):,} docs ({len(extra)} extra)")

    print("Building BM25 ...")
    BM25Index.build([d.pmid for d in corpus.docs], [d.text for d in corpus.docs]).save(out / "bm25")

    if not args.skip_dense:
        print(f"Embedding with MedCPT on {s.device} ...")
        dense = DenseIndex.open(out / "qdrant", url=s.qdrant_url, device=s.device)
        if not args.append:
            dense.create()
        embed(dense, extra if args.append else corpus.docs, args.batch, s.device)

    # Retrieval-cache keys include this, so cached results never outlive an index change.
    (out / "VERSION").write_text(index_version(corpus))
    print(f"Done. Index version {index_version(corpus)}")


if __name__ == "__main__":
    main()
