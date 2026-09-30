"""Outbound-request guard against server-side request forgery (SSRF).

``POST /fetch_url`` takes a URL from whoever can reach the API and makes the
server fetch it. Without a guard, that turns the crawler into a proxy into the
network it runs on: cloud metadata (``169.254.169.254``), the Redis container
next door, admin panels on ``localhost``. Everything fetched is then readable
back through ``/search_docs`` and ``/get_doc_page``.

Every outbound request goes through :func:`check_url`, which enforces:

* scheme is ``http`` or ``https`` (no ``file:``, ``gopher:``, ...);
* the URL has a hostname and carries no credentials;
* **every** address the hostname resolves to is a public, globally routable
  address — loopback, private (RFC 1918 / ULA), link-local, CGNAT, multicast,
  reserved and unspecified ranges are refused, including IPv4-mapped IPv6.

Redirects are followed by hand (:func:`guarded_get`) so each hop is re-checked;
a public page that 302s to ``http://127.0.0.1/`` is refused at the hop.

Residual risk: the address is checked, then httpx resolves the name again to
connect. A DNS-rebinding server with a near-zero TTL could answer differently
between the two lookups. The window is small, and closing it fully needs
IP-pinned connections; if the service is exposed to untrusted callers, also
run it with egress firewall rules that block internal ranges.

For crawling an intranet docs site on purpose, set
``WEBDOCS_ALLOW_PRIVATE_NETWORKS=true``. Scheme and credential checks still apply.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from urllib.parse import urljoin, urlparse

from webdocs.config import settings

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5

Resolver = Callable[[str, int], Iterable[str]]


class UnsafeURLError(ValueError):
    """The URL points somewhere the crawler must not go."""


def _system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None:
            ip = mapped
    # ``is_global`` excludes private, loopback, link-local, CGNAT (100.64/10),
    # reserved, documentation and unspecified ranges. Multicast is checked
    # separately because some Python versions report it as global.
    return ip.is_global and not ip.is_multicast


def check_url(url: str, *, allow_private: bool | None = None, resolver: Resolver | None = None) -> str:
    """Return *url* unchanged if it is safe to fetch, else raise :class:`UnsafeURLError`."""
    allow_private = settings.allow_private_networks if allow_private is None else allow_private

    try:
        parts = urlparse(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Only http and https URLs are allowed, got {scheme or 'none'!r}")
    if not parts.hostname:
        raise UnsafeURLError("URL has no hostname")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")

    if allow_private:
        return url

    host = parts.hostname
    port = port or (443 if scheme == "https" else 80)
    try:
        addresses = list((resolver or _system_resolver)(host, port))
    except (OSError, UnicodeError) as exc:
        raise UnsafeURLError(f"Could not resolve {host!r}: {exc}") from exc
    if not addresses:
        raise UnsafeURLError(f"{host!r} did not resolve to any address")

    for address in addresses:
        try:
            public = _is_public(address)
        except ValueError as exc:
            raise UnsafeURLError(f"{host!r} resolved to an unparseable address {address!r}") from exc
        if not public:
            raise UnsafeURLError(
                f"{host!r} resolves to non-public address {address}; "
                "set WEBDOCS_ALLOW_PRIVATE_NETWORKS=true to crawl internal sites"
            )
    return url


def guarded_get(url: str, headers: dict[str, str] | None = None, *, resolver: Resolver | None = None, transport=None):
    """``httpx.get`` that validates the URL and every redirect hop.

    Returns the final :class:`httpx.Response` (a 304 is returned as-is).
    ``transport`` exists so tests can plug in ``httpx.MockTransport``.
    """
    import httpx

    current = url
    with httpx.Client(timeout=settings.request_timeout, follow_redirects=False, transport=transport) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(current, resolver=resolver)
            response = client.get(current, headers=headers)
            if not response.is_redirect:
                return response
            location = response.headers.get("Location")
            if not location:
                return response
            current = urljoin(str(response.url), location)
    raise UnsafeURLError(f"Too many redirects fetching {url}")
