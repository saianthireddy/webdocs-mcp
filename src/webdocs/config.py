"""Runtime configuration, sourced from environment variables.

Every setting has a safe default so the whole stack boots with zero
configuration and zero external API keys (offline-first). Set
``OPENAI_API_KEY`` + ``WEBDOCS_EMBEDDER=openai`` to switch to real
OpenAI embeddings in production.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    app_name: str = "webdocs-mcp"
    db_path: str = field(default_factory=lambda: os.environ.get("WEBDOCS_DB_PATH", "data/webdocs.duckdb"))
    embedder: str = field(default_factory=lambda: os.environ.get("WEBDOCS_EMBEDDER", "hashing"))
    embedding_dimensions: int = field(default_factory=lambda: _int_env("WEBDOCS_EMBED_DIM", 256))
    chunk_size: int = field(default_factory=lambda: _int_env("WEBDOCS_CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _int_env("WEBDOCS_CHUNK_OVERLAP", 150))
    max_pages: int = field(default_factory=lambda: _int_env("WEBDOCS_MAX_PAGES", 50))
    max_depth: int = field(default_factory=lambda: _int_env("WEBDOCS_MAX_DEPTH", 3))
    request_timeout: float = 15.0
    user_agent: str = "webdocs-mcp/1.0 (+https://github.com/saianthireddy/webdocs-mcp)"


settings = Settings()
