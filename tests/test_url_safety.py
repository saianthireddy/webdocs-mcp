"""SSRF guard: unsafe targets are refused at the API, the fetcher and every redirect hop."""
from __future__ import annotations

import httpx
import pytest

from webdocs import url_safety
from webdocs.config import settings
from webdocs.url_safety import UnsafeURLError, check_url, guarded_get

PUBLIC = "93.184.215.14"


def resolves_to(*addresses: str):
    return lambda host, port: list(addresses)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost.test/",  # resolved below to loopback
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://172.16.1.1/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",  # CGNAT
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://224.0.0.1/",
    ],
)
def test_non_public_targets_are_refused(url):
    resolver = resolves_to("127.0.0.1") if "localhost.test" in url else None
    with pytest.raises(UnsafeURLError):
        check_url(url, resolver=resolver)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gopher://example.com/", "ftp://example.com/", "example.com", "http://user:pw@example.com/", "http:///nohost"],
)
def test_bad_schemes_credentials_and_hostless_urls_are_refused(url):
    with pytest.raises(UnsafeURLError):
        check_url(url)


def test_host_with_any_private_address_is_refused():
    with pytest.raises(UnsafeURLError):
        check_url("https://mixed.example.com/", resolver=resolves_to(PUBLIC, "10.1.2.3"))


def test_unresolvable_host_is_refused():
    def fail(host, port):
        raise OSError("no such host")

    with pytest.raises(UnsafeURLError):
        check_url("https://nope.invalid/", resolver=fail)


def test_public_url_passes():
    assert check_url("https://docs.example.com/x", resolver=resolves_to(PUBLIC)) == "https://docs.example.com/x"


def test_private_networks_allowed_when_opted_in():
    assert check_url("http://10.0.0.5/", allow_private=True)
    with pytest.raises(UnsafeURLError):  # scheme is still enforced
        check_url("file:///etc/passwd", allow_private=True)


def test_redirect_to_internal_address_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "docs.example.com":
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
        raise AssertionError("the internal address must never be requested")

    with pytest.raises(UnsafeURLError):
        guarded_get("https://docs.example.com/", transport=httpx.MockTransport(handler))


def test_safe_redirects_are_followed():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "/new"})
        return httpx.Response(200, text="<html>moved here</html>")

    response = guarded_get("https://docs.example.com/old", transport=httpx.MockTransport(handler))
    assert response.status_code == 200 and "moved here" in response.text


def test_redirect_loop_is_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "/again"})

    with pytest.raises(UnsafeURLError):
        guarded_get("https://docs.example.com/", transport=httpx.MockTransport(handler))


def test_default_fetchers_use_the_guard(monkeypatch):
    from webdocs.crawler import httpx_fetcher
    from webdocs.fetching import httpx_conditional_fetcher

    for fetch in (httpx_fetcher, httpx_conditional_fetcher):
        with pytest.raises(UnsafeURLError):
            fetch("http://127.0.0.1:6379/")


# -- API ---------------------------------------------------------------

def test_fetch_url_rejects_metadata_endpoint(client):
    response = client.post("/fetch_url?sync=true", json={"url": "http://169.254.169.254/latest/meta-data/", "max_pages": 1, "max_depth": 0})
    assert response.status_code == 400
    assert client.get("/job_progress").json() == []  # no job was created


def test_fetch_url_rejects_host_resolving_to_loopback(client, monkeypatch):
    monkeypatch.setattr(url_safety, "_system_resolver", resolves_to("127.0.0.1"))
    response = client.post("/fetch_url", json={"url": "http://sneaky.example.com/"})
    assert response.status_code == 400


def test_fetch_url_rejects_non_http_scheme(client):
    assert client.post("/fetch_url", json={"url": "file:///etc/passwd"}).status_code == 400


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "s3cret")
    assert client.get("/health").status_code == 200  # health stays open for probes
    assert client.get("/list_doc_pages").status_code == 401
    assert client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).status_code == 401
    assert client.get("/list_doc_pages", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/list_doc_pages", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/list_doc_pages", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_no_api_key_configured_keeps_routes_open(client):
    assert client.get("/list_doc_pages").status_code == 200
