from webdocs.crawler import crawl
from webdocs.html_utils import normalize_link, parse_page, same_domain


def test_parse_page_extracts_title_text_and_links():
    title, text, links = parse_page(
        "<html><head><title>T</title><script>bad()</script></head>"
        "<body><h1>Heading</h1><p>Hello world</p><a href='/x'>x</a></body></html>"
    )
    assert title == "T"
    assert "Hello world" in text and "bad()" not in text
    assert links == ["/x"]


def test_normalize_link_rules():
    assert normalize_link("https://a.com/docs", "/install") == "https://a.com/install"
    assert normalize_link("https://a.com", "page#section") == "https://a.com/page"
    assert normalize_link("https://a.com", "mailto:x@y.com") is None
    assert normalize_link("https://a.com", "javascript:void(0)") is None
    assert same_domain("https://a.com/x", "https://A.COM/y")


def test_crawl_stays_on_domain_and_tracks_hierarchy(fake_fetcher):
    pages = crawl("https://docs.example.com", fetcher=fake_fetcher, max_pages=10, max_depth=3)
    urls = {p.url for p in pages}
    assert urls == {
        "https://docs.example.com",
        "https://docs.example.com/install",
        "https://docs.example.com/api",
        "https://docs.example.com/install/docker",
    }

    by_url = {p.url: p for p in pages}
    root = by_url["https://docs.example.com"]
    install = by_url["https://docs.example.com/install"]
    docker = by_url["https://docs.example.com/install/docker"]
    assert root.parent_id is None and root.depth == 0
    assert install.parent_id == root.id and install.depth == 1
    assert docker.parent_id == install.id and docker.depth == 2
    assert all(p.root_id == root.id for p in pages)


def test_crawl_respects_max_pages_and_survives_fetch_errors(fake_fetcher):
    assert len(crawl("https://docs.example.com", fetcher=fake_fetcher, max_pages=2)) == 2

    def flaky(url: str) -> str:
        if "install" in url:
            raise ValueError("boom")
        return fake_fetcher(url)

    pages = crawl("https://docs.example.com", fetcher=flaky, max_pages=10)
    assert {p.url for p in pages} == {"https://docs.example.com", "https://docs.example.com/api"}
