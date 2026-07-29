"""Breadth-first, same-domain web crawler with parent/child hierarchy tracking.

The fetcher is injected as a plain callable ``(url) -> html``, which keeps
the crawler fully unit-testable offline (tests inject a dict-backed fake)
and lets production swap in httpx, a headless browser, or a cache layer
without touching crawl logic.
"""
from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from webdocs.config import settings
from webdocs.html_utils import normalize_link, parse_page, same_domain
from webdocs.robots import RobotsPolicy, Throttle

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], str]


class SupportsFetch(Protocol):  # pragma: no cover - typing helper
    def __call__(self, url: str) -> str: ...


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
) -> list[CrawledPage]:
    """Crawl *root_url* breadth-first, staying on the same domain.

    ``on_page`` fires after each successful page so callers (the job
    runner) can stream progress instead of waiting for the full crawl.

    The crawl is polite by default: ``robots.txt`` is fetched once up front
    and every candidate URL is checked against it, and requests to a host
    are spaced by ``crawl_delay`` seconds (raised to the site's own
    ``Crawl-delay`` if it asks for more). Pass ``respect_robots=False`` only
    for sites you control.
    """
    fetcher = fetcher or httpx_fetcher
    max_pages = max_pages or settings.max_pages
    max_depth = max_depth if max_depth is not None else settings.max_depth
    respect_robots = settings.respect_robots if respect_robots is None else respect_robots

    root_url = root_url.rstrip("/") or root_url
    root_id = uuid.uuid4().hex
    domain = urlparse(root_url).netloc.lower()

    robots = RobotsPolicy(root_url, fetcher, settings.user_agent, enabled=respect_robots)
    delay = settings.crawl_delay if crawl_delay is None else crawl_delay
    if robots.crawl_delay is not None:
        delay = max(delay, robots.crawl_delay)
    throttle = Throttle(delay, **({"sleep": sleep} if sleep is not None else {}))

    queue: deque[tuple[str, int, str | None]] = deque([(root_url, 0, None)])
    seen: set[str] = {root_url}
    pages: list[CrawledPage] = []

    while queue and len(pages) < max_pages:
        url, depth, parent_id = queue.popleft()
        if not robots.is_allowed(url):
            logger.info("robots.txt disallows %s - skipping", url)
            continue
        throttle.wait(url)
        try:
            html = fetcher(url)
        except Exception as exc:
            logger.warning("Skipping %s: %s", url, exc)
            continue

        title, text, hrefs = parse_page(html)
        page = CrawledPage(
            id=root_id if not pages else uuid.uuid4().hex,
            url=url,
            title=title or url,
            text=text,
            domain=domain,
            depth=depth,
            parent_id=parent_id,
            root_id=root_id,
        )

        if depth < max_depth:
            for href in hrefs:
                link = normalize_link(url, href)
                if link and link not in seen and same_domain(link, root_url):
                    seen.add(link)
                    if not robots.is_allowed(link):
                        logger.info("robots.txt disallows %s - not queueing", link)
                        continue
                    page.outgoing_links.append(link)
                    queue.append((link, depth + 1, page.id))

        pages.append(page)
        if on_page is not None:
            on_page(page)

    return pages
