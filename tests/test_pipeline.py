import numpy as np
import pytest

from webdocs.chunker import chunk_text
from webdocs.crawler import crawl
from webdocs.database import Database
from webdocs.embedder import HashingEmbedder
from webdocs.search import search


def test_chunk_text_respects_size_and_overlap():
    text = "\n".join(f"Paragraph {i} " + "word " * 60 for i in range(8))
    chunks = chunk_text(text, chunk_size=400, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 60 for c in chunks)
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("hi", chunk_size=100, overlap=100)


def test_hashing_embedder_deterministic_unit_norm():
    embedder = HashingEmbedder(dimensions=128)
    v1, v2 = embedder.embed_one("duckdb vector search"), embedder.embed_one("duckdb vector search")
    assert np.allclose(v1, v2)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-6
    assert v1.shape == (128,)


def _index_fixture_site(fake_fetcher, db: Database, embedder: HashingEmbedder) -> None:
    for page in crawl("https://docs.example.com", fetcher=fake_fetcher):
        db.insert_page(page)
        chunks = chunk_text(page.text, 800, 100)
        texts = [c.text for c in chunks]
        db.insert_chunks(page.id, texts, embedder.embed_many(texts))


def test_database_roundtrip_and_hierarchy(tmp_path, fake_fetcher):
    db = Database(str(tmp_path / "t.duckdb"))
    _index_fixture_site(fake_fetcher, db, HashingEmbedder(64))

    assert db.counts()["pages"] == 4
    roots = db.root_pages()
    assert len(roots) == 1 and roots[0].url == "https://docs.example.com"
    child_urls = {c.url for c in db.children_of(roots[0].id)}
    assert child_urls == {"https://docs.example.com/install", "https://docs.example.com/api"}


def test_hybrid_search_finds_exact_identifier_and_semantic_match(tmp_path, fake_fetcher):
    db = Database(str(tmp_path / "t.duckdb"))
    embedder = HashingEmbedder(128)
    _index_fixture_site(fake_fetcher, db, embedder)

    # Exact identifier only present on the Docker page -> BM25 must surface it.
    results = search(db, embedder, "ERR_LOCK_TIMEOUT", top_k=3)
    assert results and "ERR_LOCK_TIMEOUT" in results[0].text

    # Token overlap with the install page.
    results = search(db, embedder, "refund policy enterprise licenses", top_k=3)
    assert results and "refund policy" in results[0].text.lower()


def test_search_empty_index_returns_empty(tmp_path):
    db = Database(str(tmp_path / "t.duckdb"))
    assert search(db, HashingEmbedder(64), "anything") == []
