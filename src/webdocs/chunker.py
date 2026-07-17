"""Paragraph-aware text chunking with sliding-window overlap."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    seq: int
    text: str


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Paragraph boundaries are respected where possible; oversized
    paragraphs fall back to a character sliding window with *overlap*
    carried between windows so no sentence is stranded at a boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[Chunk] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(Chunk(seq=len(chunks), text=buffer.strip()))
        buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                window = para[start : start + chunk_size]
                chunks.append(Chunk(seq=len(chunks), text=window.strip()))
                start += chunk_size - overlap
            continue
        if len(buffer) + len(para) + 1 > chunk_size:
            tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
            flush()
            buffer = (tail + " " + para).strip() if tail else para
        else:
            buffer = f"{buffer}\n{para}".strip()

    flush()
    return chunks
