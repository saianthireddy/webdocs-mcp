"""Background crawl-and-index jobs.

Jobs run on daemon threads with an in-process registry - the right
default for a single-node tool. The ``JobManager`` interface (submit /
get / list) is intentionally the same shape you would put in front of
Redis + a worker pool, so scaling out later is a swap, not a rewrite
(docker-compose already ships the Redis service for that path).
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field

from webdocs.chunker import chunk_text
from webdocs.config import settings
from webdocs.crawler import CrawledPage, Fetcher, crawl_detailed, page_id_for
from webdocs.database import Database
from webdocs.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    url: str
    status: str = "pending"  # pending | running | completed | failed
    pages_crawled: int = 0
    pages_unchanged: int = 0
    pages_pruned: int = 0
    chunks_indexed: int = 0
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.id,
                "url": self.url,
                "status": self.status,
                "pages_crawled": self.pages_crawled,
                "pages_unchanged": self.pages_unchanged,
                "pages_pruned": self.pages_pruned,
                "chunks_indexed": self.chunks_indexed,
                "error": self.error,
            }


class JobManager:
    def __init__(self, db: Database, embedder: Embedder, fetcher: Fetcher | None = None) -> None:
        self._db = db
        self._embedder = embedder
        self._fetcher = fetcher
        self._jobs: dict[str, Job] = {}
        self._registry_lock = threading.Lock()

    def submit(self, url: str, max_pages: int | None = None, max_depth: int | None = None,
               synchronous: bool = False, prune: bool | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url)
        with self._registry_lock:
            self._jobs[job.id] = job
        args = (job, max_pages, max_depth, prune)
        if synchronous:
            self._run(*args)
        else:
            thread = threading.Thread(target=self._run, args=args, daemon=True)
            thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._registry_lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._registry_lock:
            jobs = list(self._jobs.values())
        return [j.snapshot() for j in jobs]

    # -- internals -----------------------------------------------------

    def _index_page(self, job: Job, page: CrawledPage) -> None:
        self._db.insert_page(page)
        chunks = chunk_text(page.text, settings.chunk_size, settings.chunk_overlap)
        if chunks:
            texts = [c.text for c in chunks]
            self._db.insert_chunks(page.id, texts, self._embedder.embed_many(texts))
        with job._lock:
            job.pages_crawled += 1
            job.chunks_indexed += len(chunks)

    def _mark_unchanged(self, job: Job, url: str) -> None:
        self._db.touch_page(url)
        with job._lock:
            job.pages_unchanged += 1

    def _prune(self, job: Job, root_url: str, outcome) -> None:
        """Remove pages that are no longer linked from the site.

        Only runs when the crawl was complete: if ``max_pages`` cut it short,
        a URL's absence says nothing about whether it still exists. URLs that
        failed to fetch or were disallowed by robots.txt are kept, because the
        fetcher surfaces no status code — a 404 and a 503 are indistinguishable
        here, and deleting a page because of a transient error is much worse
        than leaving a dead one indexed.
        """
        if not outcome.safe_to_prune:
            logger.info("Crawl of %s hit its page limit; skipping prune", root_url)
            return
        pruned = self._db.prune_pages(page_id_for(root_url, root_url), outcome.accounted_for)
        with job._lock:
            job.pages_pruned = len(pruned)

    def _run(self, job: Job, max_pages: int | None, max_depth: int | None,
             prune: bool | None = None) -> None:
        prune = settings.prune_missing if prune is None else prune
        with job._lock:
            job.status = "running"
        try:
            outcome = crawl_detailed(
                job.url,
                fetcher=self._fetcher,
                max_pages=max_pages,
                max_depth=max_depth,
                on_page=lambda page: self._index_page(job, page),
                validators=self._db.validators(),
                known_links=self._db.known_links(),
                on_unchanged=lambda url: self._mark_unchanged(job, url),
            )
            if prune:
                self._prune(job, job.url.rstrip("/") or job.url, outcome)
            with job._lock:
                job.status = "completed"
        except Exception as exc:
            logger.exception("Crawl job %s failed", job.id)
            with job._lock:
                job.status = "failed"
                job.error = str(exc)
