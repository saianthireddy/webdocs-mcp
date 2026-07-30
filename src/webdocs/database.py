"""Unified DuckDB storage layer for pages, chunks, and embeddings.

DuckDB keeps the whole index in a single file (or in memory), which is
exactly the right weight for a self-hosted doc-search tool: no server
to run, columnar scans are fast enough for hybrid search over tens of
thousands of chunks, and the file is trivially portable.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import uuid
from dataclasses import dataclass

import duckdb
import numpy as np

from webdocs.crawler import CrawledPage

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id VARCHAR PRIMARY KEY,
    url VARCHAR NOT NULL,
    title VARCHAR,
    text VARCHAR,
    domain VARCHAR,
    depth INTEGER,
    parent_id VARCHAR,
    root_id VARCHAR,
    crawled_at TIMESTAMP,
    etag VARCHAR,
    last_modified VARCHAR
);
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR PRIMARY KEY,
    page_id VARCHAR NOT NULL,
    seq INTEGER NOT NULL,
    text VARCHAR NOT NULL,
    embedding FLOAT[]
);
"""


@dataclass
class PageRecord:
    id: str
    url: str
    title: str
    text: str
    domain: str
    depth: int
    parent_id: str | None
    root_id: str


class Database:
    """Thread-safe facade over one DuckDB connection.

    DuckDB connections are not safe for concurrent writes from multiple
    threads, and the crawl worker runs in a background thread, so every
    statement goes through one lock. Contention is negligible at this
    tool's scale.
    """

    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = duckdb.connect(path)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Add columns an older index file predates. Called with the lock held.

        Databases created before conditional re-crawling have no etag or
        last_modified column; adding them here means an existing index keeps
        working instead of failing on first insert.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info('pages')").fetchall()}
        for column in ("etag", "last_modified"):
            if column not in existing:
                self._conn.execute(f"ALTER TABLE pages ADD COLUMN {column} VARCHAR")
                logger.info("Migrated pages table: added %s", column)

    # -- ingestion -----------------------------------------------------

    def insert_page(self, page: CrawledPage) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [page.id, page.url, page.title, page.text, page.domain,
                 page.depth, page.parent_id, page.root_id, dt.datetime.now(),
                 page.etag, page.last_modified],
            )

    def insert_chunks(self, page_id: str, texts: list[str], embeddings: list[np.ndarray]) -> None:
        rows = [
            [uuid.uuid4().hex, page_id, seq, text, embedding.tolist()]
            for seq, (text, embedding) in enumerate(zip(texts, embeddings))
        ]
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE page_id = ?", [page_id])
            self._conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)

    def touch_page(self, url: str) -> None:
        """Record that *url* was re-checked and found unchanged."""
        with self._lock:
            self._conn.execute("UPDATE pages SET crawled_at = ? WHERE url = ?", [dt.datetime.now(), url])

    # -- incremental re-crawl ------------------------------------------

    def validators(self) -> dict[str, tuple[str | None, str | None]]:
        """url -> (etag, last_modified) for every indexed page.

        Read in one query rather than per-URL: a crawl asks about every page it
        visits, and the whole table is small enough that N queries would be the
        expensive choice.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT url, etag, last_modified FROM pages WHERE etag IS NOT NULL OR last_modified IS NOT NULL"
            ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def known_links(self) -> dict[str, list[str]]:
        """url -> child urls recorded last crawl, for traversing past a 304."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT parent.url, child.url FROM pages AS child "
                "JOIN pages AS parent ON child.parent_id = parent.id "
                "WHERE child.parent_id IS NOT NULL"
            ).fetchall()
        links: dict[str, list[str]] = {}
        for parent_url, child_url in rows:
            links.setdefault(parent_url, []).append(child_url)
        return links

    def urls_under_root(self, root_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT url FROM pages WHERE root_id = ?", [root_id]).fetchall()
        return [r[0] for r in rows]

    def prune_pages(self, root_id: str, keep_urls: set[str]) -> list[str]:
        """Delete indexed pages under *root_id* whose URL is not in *keep_urls*.

        Scoped to one root on purpose: crawling site A must never delete site
        B's pages, and root_id is derived from the root URL so the scope is
        exactly the site being re-crawled.

        Chunks are deleted first. If it were the other way round and the process
        died between the two statements, the index would keep orphan chunks with
        no page to attribute them to, which is worse than a page row with no
        chunks — search reads chunks.
        """
        stale = [url for url in self.urls_under_root(root_id) if url not in keep_urls]
        if not stale:
            return []
        with self._lock:
            placeholders = ", ".join("?" * len(stale))
            self._conn.execute(
                f"DELETE FROM chunks WHERE page_id IN ("
                f"  SELECT id FROM pages WHERE root_id = ? AND url IN ({placeholders})"
                f")",
                [root_id, *stale],
            )
            self._conn.execute(
                f"DELETE FROM pages WHERE root_id = ? AND url IN ({placeholders})",
                [root_id, *stale],
            )
        logger.info("Pruned %d stale page(s) under root %s", len(stale), root_id)
        return stale

    # -- reads ---------------------------------------------------------

    def list_pages(self) -> list[PageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, url, title, text, domain, depth, parent_id, root_id FROM pages ORDER BY crawled_at"
            ).fetchall()
        return [PageRecord(*row) for row in rows]

    def get_page(self, page_id: str) -> PageRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, url, title, text, domain, depth, parent_id, root_id FROM pages WHERE id = ?",
                [page_id],
            ).fetchone()
        return PageRecord(*row) if row else None

    def children_of(self, page_id: str) -> list[PageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, url, title, text, domain, depth, parent_id, root_id FROM pages "
                "WHERE parent_id = ? ORDER BY url",
                [page_id],
            ).fetchall()
        return [PageRecord(*row) for row in rows]

    def root_pages(self) -> list[PageRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, url, title, text, domain, depth, parent_id, root_id FROM pages "
                "WHERE parent_id IS NULL ORDER BY crawled_at"
            ).fetchall()
        return [PageRecord(*row) for row in rows]

    def all_chunks(self) -> tuple[list[str], list[str], list[str], np.ndarray]:
        """Return (chunk_ids, page_ids, texts, embedding_matrix)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, page_id, text, embedding FROM chunks ORDER BY page_id, seq"
            ).fetchall()
        if not rows:
            return [], [], [], np.zeros((0, 0), dtype=np.float32)
        ids = [r[0] for r in rows]
        page_ids = [r[1] for r in rows]
        texts = [r[2] for r in rows]
        matrix = np.array([r[3] for r in rows], dtype=np.float32)
        return ids, page_ids, texts, matrix

    def counts(self) -> dict[str, int]:
        with self._lock:
            pages = self._conn.execute("SELECT count(*) FROM pages").fetchone()[0]
            chunks = self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"pages": int(pages), "chunks": int(chunks)}


def open_database(path: str) -> Database:
    """Open *path*, falling back to in-memory if the filesystem refuses.

    Some sandboxed/readonly mounts reject DuckDB file locks; the tool
    should degrade to a session-scoped index instead of crashing.
    """
    try:
        return Database(path)
    except Exception:
        logger.warning("Could not open %s; falling back to in-memory database", path, exc_info=True)
        return Database(":memory:")
