"""Pruning pages that vanish from a site.

Deleting indexed content is the one operation here that loses data, so most of
these tests are about when pruning must *not* happen. The rule is that a URL is
only removed if the crawl can positively account for every other outcome: it was
not fetched, not 304'd, did not fail, was not disallowed, and the crawl ran to
completion. Anything less and absence is unexplained rather than meaningful.
"""
from __future__ import annotations

import pytest

from webdocs.crawler import crawl_detailed, page_id_for
from webdocs.database import Database
from webdocs.embedder import HashingEmbedder
from webdocs.jobs import JobManager

ROOT = "https://d.example.com"
FULL_SITE = {
    ROOT: '<html><title>Home</title><body>home<a href="/a">a</a><a href="/b">b</a></body></html>',
    f"{ROOT}/a": "<html><title>A</title><body>alpha content here</body></html>",
    f"{ROOT}/b": "<html><title>B</title><body>bravo content here</body></html>",
}
SHRUNK_SITE = {
    ROOT: '<html><title>Home</title><body>home<a href="/a">a</a></body></html>',
    f"{ROOT}/a": FULL_SITE[f"{ROOT}/a"],
}


def fetcher_for(site: dict[str, str]):
    def _fetch(url: str) -> str:
        key = url.rstrip("/") or url
        if key not in site:
            raise ValueError(f"404 for {url}")
        return site[key]

    return _fetch


@pytest.fixture()
def indexed():
    """A database with the full 3-page site already crawled."""
    db = Database(":memory:")
    JobManager(db, HashingEmbedder(dimensions=32), fetcher=fetcher_for(FULL_SITE)).submit(
        ROOT, synchronous=True
    )
    assert db.counts() == {"pages": 3, "chunks": 3}
    return db


def recrawl(db, site=None, fetcher=None, **kwargs):
    manager = JobManager(
        db, HashingEmbedder(dimensions=32), fetcher=fetcher or fetcher_for(site or FULL_SITE)
    )
    return manager.submit(ROOT, synchronous=True, **kwargs).snapshot()


def urls_in(db) -> set[str]:
    return {p.url for p in db.list_pages()}


# --------------------------------------------------------------- the happy path


def test_a_page_no_longer_linked_is_removed_with_its_chunks(indexed):
    snapshot = recrawl(indexed, SHRUNK_SITE)
    assert snapshot["pages_pruned"] == 1
    assert urls_in(indexed) == {ROOT, f"{ROOT}/a"}
    assert indexed.counts() == {"pages": 2, "chunks": 2}, "chunks must go with the page"


def test_pruning_is_idempotent(indexed):
    recrawl(indexed, SHRUNK_SITE)
    second = recrawl(indexed, SHRUNK_SITE)
    assert second["pages_pruned"] == 0
    assert indexed.counts() == {"pages": 2, "chunks": 2}


def test_nothing_is_pruned_when_the_site_is_unchanged(indexed):
    snapshot = recrawl(indexed)
    assert snapshot["pages_pruned"] == 0
    assert indexed.counts() == {"pages": 3, "chunks": 3}


# ------------------------------------------------------- when it must NOT prune


def test_a_truncated_crawl_prunes_nothing(indexed):
    """max_pages cut the crawl short, so unvisited URLs are unexplained."""
    snapshot = recrawl(indexed, max_pages=1)
    assert snapshot["pages_pruned"] == 0
    assert indexed.counts()["pages"] == 3


def test_a_page_that_failed_to_fetch_is_kept(indexed):
    """A 404 and a 503 are indistinguishable through the fetcher, so keep it.

    Deleting a page because of a transient error is far worse than leaving a
    dead one indexed, and the fetcher contract returns ``str`` with no status.
    """
    base = fetcher_for(FULL_SITE)

    def flaky(url: str) -> str:
        if url.endswith("/b"):
            raise ValueError("503 temporary")
        return base(url)

    snapshot = recrawl(indexed, fetcher=flaky)
    assert snapshot["pages_pruned"] == 0
    assert f"{ROOT}/b" in urls_in(indexed)


def test_a_page_newly_disallowed_by_robots_is_kept(indexed):
    """robots.txt saying no is not the same as the page being gone."""
    site = dict(FULL_SITE)
    site[f"{ROOT}/robots.txt"] = "User-agent: *\nDisallow: /b\n"
    snapshot = recrawl(indexed, site)
    assert snapshot["pages_pruned"] == 0
    assert f"{ROOT}/b" in urls_in(indexed)


def test_prune_can_be_disabled_per_job(indexed):
    snapshot = recrawl(indexed, SHRUNK_SITE, prune=False)
    assert snapshot["pages_pruned"] == 0
    assert indexed.counts()["pages"] == 3


# ------------------------------------------------------------------- isolation


def test_pruning_one_site_never_touches_another():
    """Scoped by root_id, so re-crawling site A cannot delete site B."""
    other_root = "https://other.example.com"
    other = {
        other_root: '<html><title>O</title><body>other<a href="/x">x</a></body></html>',
        f"{other_root}/x": "<html><title>X</title><body>x content</body></html>",
    }
    db = Database(":memory:")
    JobManager(db, HashingEmbedder(dimensions=32), fetcher=fetcher_for(FULL_SITE)).submit(
        ROOT, synchronous=True
    )
    JobManager(db, HashingEmbedder(dimensions=32), fetcher=fetcher_for(other)).submit(
        other_root, synchronous=True
    )
    assert db.counts()["pages"] == 5

    recrawl(db, SHRUNK_SITE)  # shrink only the first site

    remaining = {p.url for p in db.list_pages()}
    assert other_root in remaining and f"{other_root}/x" in remaining
    assert f"{ROOT}/b" not in remaining
    assert db.counts()["pages"] == 4


# ----------------------------------------------------------- CrawlResult itself


def test_crawl_result_classifies_every_outcome():
    site = dict(FULL_SITE)
    site[ROOT] = (
        '<html><title>Home</title><body>home'
        '<a href="/a">a</a><a href="/b">b</a><a href="/missing">m</a></body></html>'
    )
    site[f"{ROOT}/robots.txt"] = "User-agent: *\nDisallow: /b\n"
    outcome = crawl_detailed(ROOT, fetcher=fetcher_for(site))

    assert ROOT in outcome.indexed
    assert f"{ROOT}/a" in outcome.indexed
    assert f"{ROOT}/b" in outcome.disallowed
    assert f"{ROOT}/missing" in outcome.failed
    assert outcome.truncated is False and outcome.safe_to_prune is True
    # every non-indexed URL is still accounted for, so none of them is prunable
    assert outcome.accounted_for >= {f"{ROOT}/b", f"{ROOT}/missing"}


def test_truncated_crawl_is_flagged_and_blocks_pruning():
    outcome = crawl_detailed(ROOT, fetcher=fetcher_for(FULL_SITE), max_pages=1)
    assert outcome.truncated is True
    assert outcome.safe_to_prune is False


def test_prune_pages_returns_what_it_deleted(indexed):
    root_id = page_id_for(ROOT, ROOT)
    deleted = indexed.prune_pages(root_id, {ROOT, f"{ROOT}/a"})
    assert deleted == [f"{ROOT}/b"]
    assert indexed.prune_pages(root_id, {ROOT, f"{ROOT}/a"}) == []
