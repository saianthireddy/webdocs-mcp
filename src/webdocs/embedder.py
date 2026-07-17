"""Embedding backends behind one tiny interface.

``HashingEmbedder`` is deterministic and dependency-light so the entire
stack (crawl -> index -> search -> MCP) runs and tests with no API key.
``OpenAIEmbedder`` is the production drop-in; it is lazily imported so
the ``openai`` package is only required when actually selected.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimensions: int

    def embed_one(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: list[str]) -> list[np.ndarray]: ...


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedding, L2-normalised.

    Not a semantic model - it is the offline stand-in that keeps the
    pipeline runnable and testable anywhere. Token overlap still ranks
    sensibly for demo corpora, and the interface is identical to the
    OpenAI backend.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha1(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_one(t) for t in texts]


class OpenAIEmbedder:  # pragma: no cover - requires network + API key
    """text-embedding-3-small via the official client. Same interface."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self.dimensions = dimensions

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def build_embedder(kind: str, dimensions: int) -> Embedder:
    if kind == "openai":  # pragma: no cover - requires API key
        return OpenAIEmbedder()
    return HashingEmbedder(dimensions=dimensions)
