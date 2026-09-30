"""FastAPI application: crawl jobs, search, site maps, and the MCP endpoint."""
from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from webdocs import sitemap
from webdocs.config import settings
from webdocs.crawler import Fetcher
from webdocs.database import open_database
from webdocs.embedder import build_embedder
from webdocs.jobs import JobManager
from webdocs.mcp_server import handle_message
from webdocs.search import search as run_search
from webdocs.url_safety import UnsafeURLError, check_url


class FetchUrlRequest(BaseModel):
    url: str = Field(..., examples=["https://docs.example.com"])
    max_pages: int | None = Field(None, ge=1, le=500)
    max_depth: int | None = Field(None, ge=0, le=10)


def require_api_key(request: Request) -> None:
    """Enforce ``WEBDOCS_API_KEY`` when it is set; a no-op otherwise.

    Accepts ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.
    """
    expected = settings.api_key
    if not expected:
        return
    supplied = request.headers.get("X-API-Key")
    if supplied is None:
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        supplied = token if scheme.lower() == "bearer" else None
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})


def create_app(db_path: str | None = None, fetcher: Fetcher | None = None) -> FastAPI:
    """App factory. Tests pass a tmp ``db_path`` and a fake ``fetcher``."""
    db = open_database(db_path if db_path is not None else settings.db_path)
    embedder = build_embedder(settings.embedder, settings.embedding_dimensions)
    jobs = JobManager(db, embedder, fetcher=fetcher)

    app = FastAPI(
        title="webdocs-mcp",
        version="1.0.0",
        description="Crawl, index, and hybrid-search websites; exposed to LLM agents as an MCP server.",
    )
    app.state.db = db
    app.state.jobs = jobs
    protected = [Depends(require_api_key)]

    # -- core API ------------------------------------------------------

    @app.post("/fetch_url", tags=["crawl"], dependencies=protected)
    def fetch_url(body: FetchUrlRequest, sync: bool = False) -> dict:
        # Reject unsafe targets up front with a clear 400. The fetcher checks
        # again on every request and redirect hop, so this is not the only gate.
        try:
            check_url(body.url)
        except UnsafeURLError as exc:
            raise HTTPException(400, f"Refusing to crawl this URL: {exc}") from exc
        job = jobs.submit(body.url, body.max_pages, body.max_depth, synchronous=sync)
        return job.snapshot()

    @app.get("/job_progress", tags=["crawl"], dependencies=protected)
    def job_progress(job_id: str | None = None) -> dict | list:
        if job_id is None:
            return jobs.list()
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job_id")
        return job.snapshot()

    @app.get("/search_docs", tags=["search"], dependencies=protected)
    def search_docs(query: str, top_k: int = 5) -> list[dict]:
        results = run_search(db, embedder, query, top_k=top_k)
        return [
            {"url": r.page_url, "title": r.page_title, "score": round(r.score, 4),
             "method": r.method, "text": r.text}
            for r in results
        ]

    @app.get("/list_doc_pages", tags=["docs"], dependencies=protected)
    def list_doc_pages() -> list[dict]:
        return [{"page_id": p.id, "url": p.url, "title": p.title, "domain": p.domain}
                for p in db.list_pages()]

    @app.get("/get_doc_page", tags=["docs"], dependencies=protected)
    def get_doc_page(page_id: str) -> dict:
        page = db.get_page(page_id)
        if page is None:
            raise HTTPException(404, "Unknown page_id")
        return {"page_id": page.id, "url": page.url, "title": page.title, "text": page.text}

    # -- site maps (pure HTML, no JS) -----------------------------------

    @app.get("/map", response_class=HTMLResponse, tags=["map"], dependencies=protected)
    def map_index() -> str:
        return sitemap.render_index(db)

    @app.get("/map/site/{root_page_id}", response_class=HTMLResponse, tags=["map"], dependencies=protected)
    def map_site(root_page_id: str) -> str:
        rendered = sitemap.render_site_tree(db, root_page_id)
        if rendered is None:
            raise HTTPException(404, "Unknown site")
        return rendered

    @app.get("/map/page/{page_id}", response_class=HTMLResponse, tags=["map"], dependencies=protected)
    def map_page(page_id: str) -> str:
        rendered = sitemap.render_page(db, page_id)
        if rendered is None:
            raise HTTPException(404, "Unknown page")
        return rendered

    @app.get("/map/page/{page_id}/raw", response_class=PlainTextResponse, tags=["map"], dependencies=protected)
    def map_page_raw(page_id: str) -> str:
        page = db.get_page(page_id)
        if page is None:
            raise HTTPException(404, "Unknown page")
        return page.text

    # -- MCP (streamable HTTP JSON-RPC) ---------------------------------

    @app.post("/mcp", tags=["mcp"], dependencies=protected)
    async def mcp_endpoint(request: Request):
        message = await request.json()
        if isinstance(message, list):  # JSON-RPC batch
            responses = [r for r in (handle_message(db, embedder, m) for m in message) if r]
            return JSONResponse(responses)
        response = handle_message(db, embedder, message)
        if response is None:  # notification
            return JSONResponse(content=None, status_code=202)
        return JSONResponse(response)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "service": settings.app_name, **db.counts()}

    return app


app = create_app()
