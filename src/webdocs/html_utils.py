"""Dependency-free HTML parsing helpers built on the stdlib ``html.parser``.

Deliberately avoids BeautifulSoup/lxml so the ingestion path has zero
native dependencies; a production deployment can swap in a richer
extractor behind the same three functions.
"""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse

_SKIP_CONTENT_TAGS = {"script", "style", "noscript", "template", "head"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote"}


class _Extractor(HTMLParser):
    """Single-pass extraction of title, visible text, and same-page links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._h1_parts: list[str] = []
        self._in_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_CONTENT_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "h1":
            self._in_h1 = True
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "h1":
            self._in_h1 = False
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_depth == 0:
            self.text_parts.append(data)
            if self._in_h1:
                self._h1_parts.append(data)


def parse_page(html: str) -> tuple[str, str, list[str]]:
    """Return ``(title, text, raw_hrefs)`` extracted from an HTML document."""
    parser = _Extractor()
    parser.feed(html)
    title = "".join(parser.title_parts).strip()
    if not title:
        title = "".join(parser._h1_parts).strip()
    lines = [ln.strip() for ln in "".join(parser.text_parts).splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return title, text, parser.links


def normalize_link(base_url: str, href: str) -> str | None:
    """Resolve *href* against *base_url*; return None for non-crawlable schemes."""
    href = href.strip()
    if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
        return None
    absolute, _fragment = urldefrag(urljoin(base_url, href))
    if urlparse(absolute).scheme not in {"http", "https"}:
        return None
    return absolute.rstrip("/") or absolute


def same_domain(url_a: str, url_b: str) -> bool:
    return urlparse(url_a).netloc.lower() == urlparse(url_b).netloc.lower()
