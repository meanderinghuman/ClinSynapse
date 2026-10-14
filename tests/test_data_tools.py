from clinsynapse.config import Settings
from clinsynapse.retrieval.retriever import META_VERSION, CachedRetriever
from evals.datasets.fetch_pubmed import parse_pubmed_xml

XML = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
<ArticleTitle>Metformin and <i>cancer</i></ArticleTitle><Abstract>
<AbstractText Label="BACKGROUND">Why.</AbstractText><AbstractText Label="RESULTS">It worked.</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>456</PMID><Article><ArticleTitle>No abstract</ArticleTitle>
</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def test_parse_pubmed_xml_keeps_labels_and_skips_empty():
    docs = parse_pubmed_xml(XML)
    assert [d.pmid for d in docs] == ["123"]
    assert docs[0].title == "Metformin and cancer"
    assert docs[0].text == "Background: Why. Results: It worked."


def test_index_version_recorded_then_replayed_without_index(tmp_path):
    idx = tmp_path / "data" / "index"
    idx.mkdir(parents=True)
    (idx / "VERSION").write_text("10-abc")
    cache = tmp_path / "c.sqlite"
    rec = CachedRetriever(Settings(data_dir=tmp_path / "data", cache_file=cache))
    assert rec.version == "10-abc" and rec.cache.get(META_VERSION) == {"version": "10-abc"}

    (idx / "VERSION").unlink()  # CI: no index on disk
    replay = CachedRetriever(Settings(data_dir=tmp_path / "data", cache_file=cache, replay_only=True))
    assert replay.version == "10-abc"
