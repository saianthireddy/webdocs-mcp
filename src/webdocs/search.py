"""Hybrid retrieval: embedding cosine similarity blended with BM25.

Semantic search catches paraphrased questions; BM25 catches exact
identifiers (error codes, API names) that embeddings smear. Blending
both consistently beats either alone on doc-search workloads.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from webdocs.database import Database
from webdocs.embedder import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_K1 = 1.5  # BM25 term-frequency saturation
_B = 0.75  # BM25 length normalisation


@dataclass
class SearchResult:
    chunk_id: str
    page_id: str
    page_url: str
    page_title: str
    text: str
    score: float
    method: str


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]]) -> np.ndarray:
    n_docs = len(docs_tokens)
    scores = np.zeros(n_docs, dtype=np.float32)
    if n_docs == 0 or not query_tokens:
        return scores
    avg_len = sum(len(d) for d in docs_tokens) / n_docs
    doc_freq: dict[str, int] = {}
    for term in set(query_tokens):
        doc_freq[term] = sum(1 for d in docs_tokens if term in d)
    for i, doc in enumerate(docs_tokens):
        doc_len = len(doc) or 1
        for term in query_tokens:
            tf = doc.count(term)
            if tf == 0:
                continue
            idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            scores[i] += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * doc_len / avg_len))
    return scores


def _normalize(scores: np.ndarray) -> np.ndarray:
    span = scores.max() - scores.min()
    if span <= 1e-9:
        return np.zeros_like(scores)
    return (scores - scores.min()) / span


def search(
    db: Database,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
    semantic_weight: float = 0.6,
) -> list[SearchResult]:
    chunk_ids, page_ids, texts, matrix = db.all_chunks()
    if not chunk_ids:
        return []

    query_vec = embedder.embed_one(query)
    semantic = matrix @ query_vec  # embeddings are unit-norm -> dot == cosine
    keyword = _bm25_scores(_tokenize(query), [_tokenize(t) for t in texts])

    blended = semantic_weight * _normalize(semantic) + (1 - semantic_weight) * _normalize(keyword)
    order = np.argsort(-blended)[:top_k]

    pages = {p.id: p for p in db.list_pages()}
    results: list[SearchResult] = []
    for i in order:
        page = pages.get(page_ids[i])
        method = "hybrid"
        if keyword[i] <= 1e-9:
            method = "semantic"
        elif semantic[i] <= 1e-9:
            method = "keyword"
        results.append(
            SearchResult(
                chunk_id=chunk_ids[i],
                page_id=page_ids[i],
                page_url=page.url if page else "",
                page_title=page.title if page else "",
                text=texts[i],
                score=float(blended[i]),
                method=method,
            )
        )
    return results
