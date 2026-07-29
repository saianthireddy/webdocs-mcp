"""robots.txt compliance and rate limiting.

Every test here is offline: the policy takes the same injected fetcher the
crawler uses, and the throttle takes an injected clock and sleep, so timing
is asserted rather than waited out.
"""
from __future__ import annotations

import pytest

from webdocs.crawler import crawl
from webdocs.robots import RobotsPolicy, Throttle

UA = "webdocs-mcp/1.0"
ROBOTS_URL = "https://docs.example.com/robots.txt"


def _fetcher(pages: dict[str, str]):
    def _fetch(url: str) -> str:
        key = url.rstrip("/") or url
        if key not in pages:
            raise ValueError(f"404 for {url}")
        return pages[key]

    return _fetch


# --------------------------------------------------------------- RobotsPolicy


def test_missing_robots_txt_allows_everything():
    policy = RobotsPolicy("https://docs.example.com", _fetcher({}), UA)
    assert policy.is_allowed("https://docs.example.com/anything")
    assert policy.crawl_delay is None


def test_disabled_policy_never_fetches():
    calls: list[str] = []

    def exploding(url: str) -> str:
        calls.append(url)
        raise AssertionError("should not fetch robots.txt when disabled")

    policy = RobotsPolicy("https://docs.example.com", exploding, UA, enabled=False)
    assert policy.is_allowed("https://docs.example.com/private/")
    assert calls == []


def test_disallow_rules_are_enforced():
    robots = "User-agent: *\nDisallow: /private/\nDisallow: /tmp\n"
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert policy.is_allowed("https://docs.example.com/install")
    assert not policy.is_allowed("https://docs.example.com/private/keys")
    assert not policy.is_allowed("https://docs.example.com/tmp")


def test_allow_wins_when_listed_before_the_disallow():
    robots = "User-agent: *\nAllow: /docs/public/\nDisallow: /docs/\n"
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert not policy.is_allowed("https://docs.example.com/docs/secret")
    assert policy.is_allowed("https://docs.example.com/docs/public/guide")


def test_allow_after_disallow_is_conservatively_blocked():
    """Documents a known deviation from RFC 9309.

    The spec resolves conflicts by longest-match, so ``/docs/public/`` should
    win over ``/docs/``. ``urllib.robotparser`` is first-match-wins and
    blocks it. We assert the real behaviour rather than the ideal because the
    error is over-restrictive — we skip a page we were allowed to fetch,
    which is the safe direction — and because a test that lies about what the
    code does is worse than a documented limitation. See robots.py.
    """
    robots = "User-agent: *\nDisallow: /docs/\nAllow: /docs/public/\n"
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert not policy.is_allowed("https://docs.example.com/docs/public/guide")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3", 3.0), ("0.5", 0.5), ("2.5", 2.5)],
)
def test_crawl_delay_is_read_from_robots(value, expected):
    """Fractional delays included — stdlib drops those, see parse_crawl_delay."""
    robots = f"User-agent: *\nCrawl-delay: {value}\n"
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert policy.crawl_delay == pytest.approx(expected)


def test_crawl_delay_prefers_a_group_naming_our_agent():
    robots = (
        "User-agent: *\nCrawl-delay: 1\n\n"
        "User-agent: webdocs-mcp\nCrawl-delay: 7\n"
    )
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert policy.crawl_delay == pytest.approx(7.0)


def test_crawl_delay_ignores_other_agents():
    robots = "User-agent: googlebot\nCrawl-delay: 9\n"
    policy = RobotsPolicy(
        "https://docs.example.com", _fetcher({ROBOTS_URL: robots}), UA
    )
    assert policy.crawl_delay is None


def test_robots_url_is_built_from_origin_not_path():
    policy = RobotsPolicy(
        "https://docs.example.com/deep/path", _fetcher({}), UA
    )
    assert policy.robots_url == ROBOTS_URL


# -------------------------------------------------------------------- Throttle


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_throttle_spaces_requests_to_the_same_host():
    clock = FakeClock()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.now += seconds

    throttle = Throttle(1.0, clock=clock, sleep=sleep)
    assert throttle.wait("https://a.com/1") == 0.0  # first request is free
    assert throttle.wait("https://a.com/2") == pytest.approx(1.0)
    assert throttle.wait("https://a.com/3") == pytest.approx(1.0)
    assert slept == pytest.approx([1.0, 1.0])


def test_throttle_budgets_hosts_independently():
    clock = FakeClock()
    throttle = Throttle(1.0, clock=clock, sleep=lambda s: None)
    throttle.wait("https://a.com/1")
    assert throttle.wait("https://b.com/1") == 0.0


def test_throttle_credits_time_already_elapsed():
    clock = FakeClock()
    throttle = Throttle(1.0, clock=clock, sleep=lambda s: None)
    throttle.wait("https://a.com/1")
    clock.now += 5.0  # caller was slow; no need to sleep at all
    assert throttle.wait("https://a.com/2") == 0.0


def test_zero_delay_disables_throttling():
    throttle = Throttle(0.0, sleep=lambda s: pytest.fail("should not sleep"))
    assert throttle.wait("https://a.com/1") == 0.0


# ---------------------------------------------------------- crawler integration

SITE = {
    "https://docs.example.com": (
        "<html><head><title>Home</title></head><body>"
        "<a href='/install'>i</a><a href='/private/keys'>k</a></body></html>"
    ),
    "https://docs.example.com/install": "<html><title>Install</title><body>ok</body></html>",
    "https://docs.example.com/private/keys": "<html><title>Keys</title><body>secret</body></html>",
}


def test_crawl_skips_paths_disallowed_by_robots():
    pages = SITE | {ROBOTS_URL: "User-agent: *\nDisallow: /private/\n"}
    crawled = {p.url for p in crawl("https://docs.example.com", fetcher=_fetcher(pages))}
    assert crawled == {
        "https://docs.example.com",
        "https://docs.example.com/install",
    }


def test_crawl_fetches_disallowed_paths_when_robots_is_off():
    pages = SITE | {ROBOTS_URL: "User-agent: *\nDisallow: /private/\n"}
    crawled = {
        p.url
        for p in crawl(
            "https://docs.example.com", fetcher=_fetcher(pages), respect_robots=False
        )
    }
    assert "https://docs.example.com/private/keys" in crawled


def test_crawl_honours_the_larger_of_configured_and_robots_delay():
    """robots.txt asks for 3s, caller configured 0.5s — the site wins.

    Only the first sleep is asserted. The injected ``sleep`` records without
    advancing ``time.monotonic``, so the throttle sees its debt as unpaid and
    later waits compound; that is an artifact of the fake, not of Throttle,
    which is covered against a fake clock in the unit tests above.
    """
    pages = SITE | {ROBOTS_URL: "User-agent: *\nCrawl-delay: 3\n"}
    slept: list[float] = []
    crawl(
        "https://docs.example.com",
        fetcher=_fetcher(pages),
        crawl_delay=0.5,
        sleep=slept.append,
    )
    assert slept, "expected the crawler to throttle between pages"
    assert slept[0] == pytest.approx(3.0, abs=0.01)


def test_crawl_uses_the_configured_delay_when_robots_asks_for_less():
    pages = SITE | {ROBOTS_URL: "User-agent: *\nCrawl-delay: 1\n"}
    slept: list[float] = []
    crawl(
        "https://docs.example.com",
        fetcher=_fetcher(pages),
        crawl_delay=4.0,
        sleep=slept.append,
    )
    assert slept[0] == pytest.approx(4.0, abs=0.01)
