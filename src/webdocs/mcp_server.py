"""MCP server over streamable HTTP (JSON-RPC 2.0).

Implements the subset of the Model Context Protocol that tool-calling
clients (Cursor, VS Code, Claude Code) need from a search backend:
``initialize``, ``tools/list``, and ``tools/call``. Kept dependency-free
on purpose - the protocol core is just JSON-RPC dispatch, and owning it
keeps the whole stack runnable offline.
"""
from __future__ import annotations

import json
from typing import Any

from webdocs.database import Database
from webdocs.embedder import Embedder
from webdocs.search import search

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Hybrid (semantic + BM25) search over all crawled documentation. Returns the most relevant chunks with source URLs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question or keywords"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_doc_pages",
        "description": "List every indexed page (id, url, title) so an agent can pick one to read in full.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_doc_page",
        "description": "Fetch the full extracted text of one indexed page by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"page_id": {"type": "string"}},
            "required": ["page_id"],
        },
    },
]


def _result(request_id: Any, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": payload}


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text_content(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _call_tool(db: Database, embedder: Embedder, name: str, arguments: dict) -> dict:
    if name == "search_docs":
        results = search(db, embedder, arguments["query"], top_k=int(arguments.get("top_k", 5)))
        payload = [
            {"url": r.page_url, "title": r.page_title, "score": round(r.score, 4),
             "method": r.method, "text": r.text}
            for r in results
        ]
        return _text_content(json.dumps(payload, indent=2))
    if name == "list_doc_pages":
        pages = [{"page_id": p.id, "url": p.url, "title": p.title} for p in db.list_pages()]
        return _text_content(json.dumps(pages, indent=2))
    if name == "get_doc_page":
        page = db.get_page(arguments["page_id"])
        if page is None:
            return {"content": [{"type": "text", "text": "Page not found"}], "isError": True}
        return _text_content(page.text)
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def handle_message(db: Database, embedder: Embedder, message: dict) -> dict | None:
    """Dispatch one JSON-RPC message. Returns None for notifications."""
    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:  # notification (e.g. notifications/initialized)
        return None
    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "webdocs-mcp", "version": "1.0.0"},
        })
    if method == "ping":
        return _result(request_id, {})
    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params") or {}
        try:
            return _result(
                request_id,
                _call_tool(db, embedder, params.get("name", ""), params.get("arguments") or {}),
            )
        except KeyError as exc:
            return _error(request_id, -32602, f"Missing required argument: {exc}")
    return _error(request_id, -32601, f"Method not found: {method}")
