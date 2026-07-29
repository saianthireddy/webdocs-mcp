"""Stable page identity and conditional re-crawling.

The regression these guard against is concrete: page ids used to be random
uuid4s, so ``INSERT OR REPLACE`` had nothing to replace and re-crawling a site
duplicated every page and every chunk. Identity is now derived from
(root_url, url), and once identity is stable ETag/Last-Modified can actually
save work.
"""
from __future__ import annotations

import duckdb
import pytest

from webdocs.crawler import crawl, page_id_for
from webdocs.database import Database
from webdocs.embedder import HashingEmbedder
from webdocs.fetching import FetchResult, as_conditional
from webdocs.jobs import JobManager

SITE = {
    "https://d.example.com": (
        "<html><title>Home</title><body><p>hello world</p>"
        "<a href='/a'>a</a></body></html>"
    ),
    "https://d.example.com/a": "<html><title>A</title><body><p>page a body text</p></body></html>",
}
ROOT = "https://d.example.com"


def plain_fetcher(url: str) -> str:
    key = url.rstrip("/") or url
    if key not in SITE:
        raise ValueError(f"404 for {url}")
    return SITE[key]


class RecordingFetcher:
    """Conditional fetcher over SITE that reports 304 when the ETag matches."""

    def __init__(self, pages: dict[str, str] | None = None) -> None:
        self.pages = dict(pages or SITE)
        self.etags = {url: f'W/"{i}"' for i, url in enumerate(self.pages)}
        self.bodies_served = 0
        self.not_modified_served = 0

    def bump(self, url: str, new_body: str) -> None:
        """Simulate the page changing: new body, new ETag."""
        self.pages[url] = new_body
        self.etags[url] = self.etags[url] + "-v2"

    def __call__(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        key = url.rstrip("/") or url
        if key not in self.pages:
            raise ValueError(f"404 for {url}")
        if etag is not None and etag == self.etags[key]:
            self.not_modified_served += 1
            return FetchResult(etag=etag, last_modified=last_modified, not_modified=True)
        self.bodies_served += 1
        return FetchResult(
            text=self.pages[key],
            etag=self.etags[key],
            last_modified="Mon, 27 Jul 2026 00:00:00 GMT",
        )


@pytest.fixture()
def manager():
    def _build(fetcher):
        db = Database(":memory:")
        return db, JobManager(db, HashingEmbedder(dimensions=32), fetcher=fetcher)

    return _build


# ------------------------------------------------------------------- identity


def test_page_id_is_deterministic_and_scoped_to_the_root():
    assert page_id_for(ROOT, ROOT) == page_id_for(ROOT, ROOT)
    assert page_id_for(ROOT, f"{ROOT}/a") != page_id_for(ROOT, ROOT)
    assert page_id_for("https://other.com", ROOT) != page_id_for(ROOT, ROOT)
    assert len(page_id_for(ROOT, ROOT)) == 32


def test_recrawling_does_not_duplicate_pages_or_chunks(manager):
    """The regression. Three crawls of a 2-page site used to leave 6 rows."""
    db, jobs = manager(plain_fetcher)
    for _ in range(3):
        jobs.submit(ROOT, synchronous=True)

    counts = db.counts()
    assert counts["pages"] == 2
    assert counts["chunks"] == 2
    assert len(db.root_pages()) == 1, "one root, not one per crawl"
    urls = [p.url for p in db.list_pages()]
    assert len(urls) == len(set(urls))


# ---------------------------------------------------------------- conditional


def test_unchanged_pages_are_not_reindexed(manager):
    fetcher = RecordingFetcher()
    db, jobs = manager(fetcher)

    first = jobs.submit(ROOT, synchronous=True).snapshot()
    assert first["pages_crawled"] == 2
    assert first["pages_unchanged"] == 0
    assert fetcher.bodies_served == 2

    second = jobs.submit(ROOT, synchronous=True).snapshot()
    assert second["pages_crawled"] == 0, "nothing changed, so nothing re-indexed"
    assert second["pages_unchanged"] == 2
    assert fetcher.not_modified_served == 2
    assert fetcher.bodies_served == 2, "no body re-downloaded"
    assert db.counts() == {"pages": 2, "chunks": 2}


def test_a_changed_page_is_reindexed_while_its_sibling_is_skipped(manager):
    fetcher = RecordingFetcher()
    db, jobs = manager(fetcher)
    jobs.submit(ROOT, synchronous=True)
    baseline_bodies = fetcher.bodies_served

    fetcher.bump(
        f"{ROOT}/a", "<html><title>A</title><body><p>completely rewritten content</p></body></html>"
    )
    snap = jobs.submit(ROOT, synchronous=True).snapshot()

    assert snap["pages_crawled"] == 1
    assert snap["pages_unchanged"] == 1
    assert fetcher.bodies_served == baseline_bodies + 1
    texts = {p.url: p.text for p in db.list_pages()}
    assert "completely rewritten" in texts[f"{ROOT}/a"]


def test_traversal_continues_past_a_304_using_known_links(manager):
    """A 304 has no body, so last crawl's links are the only way onward."""
    fetcher = RecordingFetcher()
    db, jobs = manager(fetcher)
    jobs.submit(ROOT, synchronous=True)

    snap = jobs.submit(ROOT, synchronous=True).snapshot()
    # The child is only reachable via the root's stored links; if traversal
    # stopped at the root's 304, this would be 1.
    assert snap["pages_unchanged"] == 2


def test_validators_and_known_links_are_persisted(manager):
    fetcher = RecordingFetcher()
    db, jobs = manager(fetcher)
    jobs.submit(ROOT, synchronous=True)

    validators = db.validators()
    assert set(validators) == {ROOT, f"{ROOT}/a"}
    assert validators[ROOT][0] == fetcher.etags[ROOT]
    assert validators[ROOT][1] == "Mon, 27 Jul 2026 00:00:00 GMT"

    assert db.known_links() == {ROOT: [f"{ROOT}/a"]}


def test_touch_page_does_not_change_content(manager):
    db, jobs = manager(plain_fetcher)
    jobs.submit(ROOT, synchronous=True)
    before = {p.url: p.text for p in db.list_pages()}
    db.touch_page(ROOT)
    assert {p.url: p.text for p in db.list_pages()} == before


# ------------------------------------------------------- backwards compatibility


def test_a_plain_fetcher_still_works_and_simply_never_reports_304(manager):
    """The original (url) -> html contract must keep working untouched."""
    db, jobs = manager(plain_fetcher)
    jobs.submit(ROOT, synchronous=True)
    snap = jobs.submit(ROOT, synchronous=True).snapshot()
    assert snap["pages_unchanged"] == 0
    assert snap["pages_crawled"] == 2, "no validators available, so everything is re-read"
    assert db.counts()["pages"] == 2, "but identity is still stable"


def test_as_conditional_detects_which_interface_it_was_given():
    def conditional(url, etag=None, last_modified=None):
        return FetchResult(text="x")

    assert as_conditional(conditional) is conditional
    adapted = as_conditional(plain_fetcher)
    assert adapted is not plain_fetcher
    result = adapted(ROOT, "some-etag", "some-date")
    assert result.text == SITE[ROOT]
    assert result.not_modified is False


def test_fetch_result_has_body_flag():
    assert FetchResult(text="hi").has_body is True
    assert FetchResult(not_modified=True).has_body is False


def test_crawl_reports_unchanged_urls_through_the_callback():
    fetcher = RecordingFetcher()
    crawl(ROOT, fetcher=fetcher)
    validators = {url: (fetcher.etags[url], None) for url in fetcher.pages}

    seen: list[str] = []
    pages = crawl(
        ROOT,
        fetcher=fetcher,
        validators=validators,
        known_links={ROOT: [f"{ROOT}/a"]},
        on_unchanged=seen.append,
    )
    assert pages == []
    assert sorted(seen) == [ROOT, f"{ROOT}/a"]


# ------------------------------------------------------------------- migration


def test_an_index_predating_conditional_recrawl_is_migrated(tmp_path):
    """Opening an older index file must add the new columns, not fail."""
    path = str(tmp_path / "old.duckdb")
    old = duckdb.connect(path)
    old.execute(
        "CREATE TABLE pages (id VARCHAR PRIMARY KEY, url VARCHAR NOT NULL, title VARCHAR,"
        " text VARCHAR, domain VARCHAR, depth INTEGER, parent_id VARCHAR, root_id VARCHAR,"
        " crawled_at TIMESTAMP)"
    )
    old.execute(
        "CREATE TABLE chunks (id VARCHAR PRIMARY KEY, page_id VARCHAR NOT NULL,"
        " seq INTEGER NOT NULL, text VARCHAR NOT NULL, embedding FLOAT[])"
    )
    old.execute(
        "INSERT INTO pages VALUES ('legacy', ?, 'Old', 'old body', 'd.example.com', 0, NULL, 'legacy', now())",
        [ROOT],
    )
    old.close()

    db = Database(path)
    columns = {row[1] for row in db._conn.execute("PRAGMA table_info('pages')").fetchall()}
    assert {"etag", "last_modified"} <= columns
    assert [p.url for p in db.list_pages()] == [ROOT], "existing rows survive the migration"
    assert db.validators() == {}, "legacy rows have no validators, so they re-fetch"
