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
from webdocs.crawler import CrawledPage, Fetcher, crawl
from webdocs.database import Database
from webdocs.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    url: str
    status: str = "pending"  # pending | running | completed | failed
    pages_crawled: int = 0
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
               synchronous: bool = False) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url)
        with self._registry_lock:
            self._jobs[job.id] = job
        if synchronous:
            self._run(job, max_pages, max_depth)
        else:
            thread = threading.Thread(target=self._run, args=(job, max_pages, max_depth), daemon=True)
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

    def _run(self, job: Job, max_pages: int | None, max_depth: int | None) -> None:
        with job._lock:
            job.status = "running"
        try:
            crawl(
                job.url,
                fetcher=self._fetcher,
                max_pages=max_pages,
                max_depth=max_depth,
                on_page=lambda page: self._index_page(job, page),
            )
            with job._lock:
                job.status = "completed"
        except Exception as exc:
            logger.exception("Crawl job %s failed", job.id)
            with job._lock:
                job.status = "failed"
                job.error = str(exc)
