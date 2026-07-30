"""Breadth-first, same-domain web crawler with parent/child hierarchy tracking.

The fetcher is injected as a plain callable ``(url) -> html``, which keeps
the crawler fully unit-testable offline (tests inject a dict-backed fake)
and lets production swap in httpx, a headless browser, or a cache layer
without touching crawl logic.
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from webdocs.config import settings
from webdocs.fetching import FetchResult, as_conditional
from webdocs.html_utils import normalize_link, parse_page, same_domain
from webdocs.robots import RobotsPolicy, Throttle

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], str]


class SupportsFetch(Protocol):  # pragma: no cover - typing helper
    def __call__(self, url: str) -> str: ...


def page_id_for(root_url: str, url: str) -> str:
    """Deterministic page id, so a re-crawl updates rows instead of duplicating.

    Ids used to be random uuid4s, which meant ``INSERT OR REPLACE`` had nothing
    to replace: crawling a 2-page site three times left six page rows, three
    identical roots, and every chunk indexed three times. Hashing
    (root, url) makes identity a function of the thing being crawled.
    """
    digest = hashlib.sha256(f"{root_url}\n{url}".encode()).hexdigest()
    return digest[:32]


@dataclass
class CrawlResult:
    """Everything a caller needs to decide what is safe to delete.

    Pruning stale pages is only defensible if you can tell "this page is gone"
    apart from "this page was not reached", so the crawl reports *why* each URL
    it knew about did or did not end up in the index:

    ``indexed``     fetched with a body this crawl
    ``unchanged``   answered 304, still live, content reused
    ``failed``      the fetch raised — could be a 404 or a 503, and the fetcher
                    surfaces no status code, so we must not assume it is gone
    ``disallowed``  robots.txt said no; the page may well still exist
    ``truncated``   the crawl stopped on ``max_pages``, so anything not visited
                    is unexplained rather than absent

    A URL is only a pruning candidate if it appears in none of the first four
    sets *and* the crawl was not truncated.
    """

    pages: list[CrawledPage] = field(default_factory=list)
    indexed: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    disallowed: set[str] = field(default_factory=set)
    truncated: bool = False

    @property
    def accounted_for(self) -> set[str]:
        """URLs whose absence from the index would *not* mean they are gone."""
        return self.indexed | self.unchanged | self.failed | self.disallowed

    @property
    def safe_to_prune(self) -> bool:
        return not self.truncated


@dataclass
class CrawledPage:
    id: str
    url: str
    title: str
    text: str
    domain: str
    depth: int
    parent_id: str | None
    root_id: str
    outgoing_links: list[str] = field(default_factory=list)
    etag: str | None = None
    last_modified: str | None = None


def httpx_fetcher(url: str) -> str:
    """Default production fetcher: a plain GET with sane timeouts."""
    import httpx

    response = httpx.get(
        url,
        timeout=settings.request_timeout,
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent},
    )
    response.raise_for_status()
    return response.text


def crawl(
    root_url: str,
    fetcher: Fetcher | None = None,
    max_pages: int | None = None,
    max_depth: int | None = None,
    on_page: Callable[[CrawledPage], None] | None = None,
    respect_robots: bool | None = None,
    crawl_delay: float | None = None,
    sleep: Callable[[float], None] | None = None,
    validators: Mapping[str, tuple[str | None, str | None]] | None = None,
    known_links: Mapping[str, list[str]] | None = None,
    on_unchanged: Callable[[str], None] | None = None,
) -> list[CrawledPage]:
    """Backwards-compatible wrapper: returns just the pages.

    Callers that need to know what was skipped and why should use
    :func:`crawl_detailed`.
    """
    return crawl_detailed(
        root_url,
        fetcher=fetcher,
        max_pages=max_pages,
        max_depth=max_depth,
        on_page=on_page,
        respect_robots=respect_robots,
        crawl_delay=crawl_delay,
        sleep=sleep,
        validators=validators,
        known_links=known_links,
        on_unchanged=on_unchanged,
    ).pages


def crawl_detailed(
    root_url: str,
    fetcher: Fetcher | None = None,
    max_pages: int | None = None,
    max_depth: int | None = None,
    on_page: Callable[[CrawledPage], None] | None = None,
    respect_robots: bool | None = None,
    crawl_delay: float | None = None,
    sleep: Callable[[float], None] | None = None,
    validators: Mapping[str, tuple[str | None, str | None]] | None = None,
    known_links: Mapping[str, list[str]] | None = None,
    on_unchanged: Callable[[str], None] | None = None,
) -> CrawlResult:
    """Crawl *root_url* breadth-first, staying on the same domain.

    ``on_page`` fires after each successful page so callers (the job
    runner) can stream progress instead of waiting for the full crawl.

    Pass ``validators`` (url -> (etag, last_modified)) to make conditional
    requests on a re-crawl. When the server answers 304, ``on_unchanged`` fires
    instead of ``on_page`` and traversal continues through ``known_links`` —
    a 304 has no body, so the links that page yielded last time are the only
    way to keep walking the tree.

    The crawl is polite by default: ``robots.txt`` is fetched once up front
    and every candidate URL is checked against it, and requests to a host
    are spaced by ``crawl_delay`` seconds (raised to the site's own
    ``Crawl-delay`` if it asks for more). Pass ``respect_robots=False`` only
    for sites you control.
    """
    fetch = as_conditional(fetcher or httpx_fetcher)
    validators = validators or {}
    known_links = known_links or {}
    max_pages = max_pages or settings.max_pages
    max_depth = max_depth if max_depth is not None else settings.max_depth
    respect_robots = settings.respect_robots if respect_robots is None else respect_robots

    root_url = root_url.rstrip("/") or root_url
    root_id = page_id_for(root_url, root_url)
    domain = urlparse(root_url).netloc.lower()

    robots = RobotsPolicy(root_url, fetcher, settings.user_agent, enabled=respect_robots)
    delay = settings.crawl_delay if crawl_delay is None else crawl_delay
    if robots.crawl_delay is not None:
        delay = max(delay, robots.crawl_delay)
    throttle = Throttle(delay, **({"sleep": sleep} if sleep is not None else {}))

    queue: deque[tuple[str, int, str | None]] = deque([(root_url, 0, None)])
    seen: set[str] = {root_url}
    pages: list[CrawledPage] = []
    outcome = CrawlResult(pages=pages)

    while queue and len(pages) < max_pages:
        url, depth, parent_id = queue.popleft()
        if not robots.is_allowed(url):
            logger.info("robots.txt disallows %s - skipping", url)
            outcome.disallowed.add(url)
            continue
        throttle.wait(url)
        stored_etag, stored_last_modified = validators.get(url, (None, None))
        try:
            result: FetchResult = fetch(url, stored_etag, stored_last_modified)
        except Exception as exc:
            logger.warning("Skipping %s: %s", url, exc)
            outcome.failed.add(url)
            continue

        if result.not_modified:
            logger.debug("%s unchanged (304); reusing indexed content", url)
            outcome.unchanged.add(url)
            if on_unchanged is not None:
                on_unchanged(url)
            if depth < max_depth:
                for link in known_links.get(url, []):
                    if link not in seen and same_domain(link, root_url) and robots.is_allowed(link):
                        seen.add(link)
                        queue.append((link, depth + 1, page_id_for(root_url, url)))
            continue

        title, text, hrefs = parse_page(result.text)
        page = CrawledPage(
            id=page_id_for(root_url, url),
            url=url,
            title=title or url,
            text=text,
            domain=domain,
            depth=depth,
            parent_id=parent_id,
            root_id=root_id,
            etag=result.etag,
            last_modified=result.last_modified,
        )

        if depth < max_depth:
            for href in hrefs:
                link = normalize_link(url, href)
                if link and link not in seen and same_domain(link, root_url):
                    seen.add(link)
                    if not robots.is_allowed(link):
                        logger.info("robots.txt disallows %s - not queueing", link)
                        outcome.disallowed.add(link)
                        continue
                    page.outgoing_links.append(link)
                    queue.append((link, depth + 1, page.id))

        pages.append(page)
        outcome.indexed.add(url)
        if on_page is not None:
            on_page(page)

    # Anything still queued means max_pages cut us off, so the absence of a URL
    # from this crawl carries no information about whether it still exists.
    outcome.truncated = bool(queue)
    return outcome
