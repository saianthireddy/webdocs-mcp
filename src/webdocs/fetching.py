"""Fetch results that can carry cache validators.

The crawler's original fetcher contract was ``(url) -> html``, which is lovely
to test against but has no room for an ETag, a Last-Modified date, or a 304.
Rather than break that contract — every existing test and the robots policy
inject a plain callable — this module widens it:

* :class:`FetchResult` is what a conditional fetcher returns.
* :func:`as_conditional` wraps a plain ``(url) -> html`` callable so the crawler
  can call one interface either way. A plain fetcher simply never reports
  ``not_modified``, so a crawl using one behaves exactly as before.

That keeps the simple contract as the default and makes conditional requests an
opt-in capability of the fetcher, not a requirement of it.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass

from webdocs.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchResult:
    text: str = ""
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False

    @property
    def has_body(self) -> bool:
        return not self.not_modified


PlainFetcher = Callable[[str], str]
ConditionalFetcher = Callable[[str, str | None, str | None], FetchResult]


def _accepts_validators(fetcher: Callable) -> bool:
    """True if *fetcher* takes (url, etag, last_modified) rather than (url)."""
    try:
        params = list(inspect.signature(fetcher).parameters.values())
    except (TypeError, ValueError):  # builtins, C callables
        return False
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    if any(p.kind is p.VAR_POSITIONAL for p in positional):
        return True
    return len(positional) >= 3


def as_conditional(fetcher: Callable) -> ConditionalFetcher:
    """Normalise any fetcher to the conditional interface."""
    if _accepts_validators(fetcher):
        return fetcher  # type: ignore[return-value]

    def _wrapped(url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        # A plain fetcher cannot make a conditional request, so validators are
        # dropped and the body is always re-read. Correct, just not saving work.
        return FetchResult(text=fetcher(url))

    return _wrapped


def httpx_conditional_fetcher(
    url: str, etag: str | None = None, last_modified: str | None = None
) -> FetchResult:
    """Production fetcher: sends If-None-Match / If-Modified-Since.

    A 304 comes back as ``not_modified=True`` with the validators we sent, so
    the stored ones survive a re-crawl. Any other non-2xx still raises, which
    the crawler already treats as "skip this page and keep going".
    """
    import httpx

    headers = {"User-Agent": settings.user_agent}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    response = httpx.get(url, timeout=settings.request_timeout, follow_redirects=True, headers=headers)
    if response.status_code == 304:
        return FetchResult(etag=etag, last_modified=last_modified, not_modified=True)
    response.raise_for_status()
    return FetchResult(
        text=response.text,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
    )
