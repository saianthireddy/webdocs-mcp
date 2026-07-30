"""robots.txt compliance and per-domain rate limiting.

Both concerns are injected into the crawler rather than baked into it, so the
crawler stays offline-testable: :class:`RobotsPolicy` takes the same
``(url) -> html`` fetcher the crawler already uses, and :class:`Throttle`
takes clock and sleep callables so tests can assert on timing without
spending wall-clock seconds.

**Fail-open, and why.** A site with no ``robots.txt`` is treated as fully
allowed, which is what RFC 9309 requires for a 404. We also fail open when
the fetch raises for any other reason, which is a *deliberate deviation*:
the spec says an unreachable ``robots.txt`` (5xx) should be treated as a
full disallow, but the injected fetcher signature is ``(url) -> str`` and
surfaces no status code, so a 503 and a DNS failure are indistinguishable
here. Blocking every crawl on any transient network blip is the worse
failure mode for this tool. If the fetcher is ever given a richer return
type, tighten this to disallow on 5xx.

**Known deviation: Allow/Disallow precedence.** ``urllib.robotparser``
resolves conflicting rules first-match-wins, whereas RFC 9309 specifies
longest-match. So ``Disallow: /docs/`` followed by ``Allow: /docs/public/``
blocks ``/docs/public/`` here, even though a spec-compliant crawler would
fetch it. The error is always in the over-restrictive direction — we skip
pages we were permitted to read, never the reverse — which is the right way
to be wrong about this. Swap in a longest-match matcher if the missed
coverage ever matters more than the small dependency.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], str]


_UA_LINE = re.compile(r"^\s*user-agent\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DELAY_LINE = re.compile(r"^\s*crawl-delay\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _ua_matches(token: str, user_agent: str) -> bool:
    """robots.txt user-agent tokens match by case-insensitive prefix."""
    token = token.strip().lower()
    return token == "*" or user_agent.lower().startswith(token)


def parse_crawl_delay(body: str, user_agent: str) -> float | None:
    """Read ``Crawl-delay``, including fractional values.

    ``urllib.robotparser`` guards on ``str.isdigit()``, so it silently drops
    any non-integer delay — ``Crawl-delay: 0.5`` is read as "no delay
    requested". Discarding a site's politeness request over a decimal point
    is the wrong default, so we re-read it here and take whichever value is
    larger. A group naming our agent wins over the ``*`` group.
    """
    specific: float | None = None
    wildcard: float | None = None
    groups: list[str] = []

    for raw in body.splitlines():
        line = raw.split("#", 1)[0]
        ua = _UA_LINE.match(line)
        if ua:
            groups.append(ua.group(1))
            continue
        delay = _DELAY_LINE.match(line)
        if delay and groups:
            try:
                value = float(delay.group(1))
            except ValueError:  # pragma: no cover - regex already constrains this
                continue
            for token in groups:
                if token.strip() == "*":
                    wildcard = value if wildcard is None else max(wildcard, value)
                elif _ua_matches(token, user_agent):
                    specific = value if specific is None else max(specific, value)
            continue
        if line.strip() and not line.lstrip().lower().startswith(("allow", "disallow", "sitemap")):
            groups = []
        elif not line.strip():
            groups = []

    return specific if specific is not None else wildcard


def _origin(url: str) -> str:
    parts = urlparse(url)
    return f"{parts.scheme}://{parts.netloc}"


class RobotsPolicy:
    """Answers "may I fetch this URL?" for one origin.

    ``enabled=False`` short-circuits to allow-all without any fetch, for
    crawling sites you own or fixtures in tests.
    """

    def __init__(
        self,
        root_url: str,
        fetcher: Fetcher,
        user_agent: str,
        *,
        enabled: bool = True,
    ) -> None:
        self.user_agent = user_agent
        self.enabled = enabled
        self.robots_url = urljoin(_origin(root_url) + "/", "robots.txt")
        self.crawl_delay: float | None = None
        self._parser: RobotFileParser | None = None
        if enabled:
            self._load(fetcher)

    def _load(self, fetcher: Fetcher) -> None:
        try:
            body = fetcher(self.robots_url)
        except Exception as exc:
            logger.info(
                "No usable robots.txt at %s (%s) — treating as allow-all",
                self.robots_url,
                exc,
            )
            return

        parser = RobotFileParser()
        try:
            parser.parse(body.splitlines())
        except Exception as exc:  # pragma: no cover - robotparser is lenient
            logger.warning(
                "Could not parse %s (%s) — treating as allow-all", self.robots_url, exc
            )
            return

        self._parser = parser
        candidates = [
            float(value)
            for value in (parser.crawl_delay(self.user_agent), parse_crawl_delay(body, self.user_agent))
            if value is not None
        ]
        if candidates:
            self.crawl_delay = max(candidates)
            logger.info("robots.txt requests a %.2fs crawl delay", self.crawl_delay)

    def is_allowed(self, url: str) -> bool:
        if self._parser is None:
            return True
        try:
            return self._parser.can_fetch(self.user_agent, url)
        except Exception:  # pragma: no cover - defensive
            return True


class Throttle:
    """Spaces requests so consecutive fetches to a host are ``delay`` apart.

    Keyed per host so a crawl that follows a redirect to another netloc
    doesn't inherit the first host's budget. ``delay <= 0`` disables it.
    """

    def __init__(
        self,
        delay: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.delay = max(0.0, float(delay))
        self._clock = clock
        self._sleep = sleep
        self._next_allowed: dict[str, float] = {}

    def wait(self, url: str) -> float:
        """Block until *url*'s host is due. Returns seconds actually slept."""
        if self.delay <= 0:
            return 0.0
        host = urlparse(url).netloc.lower()
        now = self._clock()
        earliest = self._next_allowed.get(host, 0.0)
        slept = 0.0
        if now < earliest:
            slept = earliest - now
            self._sleep(slept)
            now = earliest
        self._next_allowed[host] = now + self.delay
        return slept
