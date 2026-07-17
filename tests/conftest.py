"""Shared fixtures: an in-memory fake website so no test touches the network."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from webdocs.api import create_app

FIXTURE_SITE: dict[str, str] = {
    "https://docs.example.com": """
        <html><head><title>Example Docs Home</title></head><body>
        <h1>Example Docs</h1>
        <p>Welcome to the documentation for ExampleDB, the fictional database.</p>
        <a href="/install">Install guide</a>
        <a href="/api">API reference</a>
        <a href="https://other-domain.com/off-site">Off-site link</a>
        <a href="mailto:team@example.com">Mail us</a>
        <script>console.log("should never appear in text")</script>
        </body></html>
    """,
    "https://docs.example.com/install": """
        <html><head><title>Installation</title></head><body>
        <h1>Installing ExampleDB</h1>
        <p>Run pip install exampledb to get started. The refund policy for
        enterprise licenses allows returns within 30 days.</p>
        <a href="/install/docker">Docker install</a>
        <a href="/">Home</a>
        </body></html>
    """,
    "https://docs.example.com/install/docker": """
        <html><head><title>Docker Install</title></head><body>
        <h1>Docker</h1>
        <p>Use docker compose up to start ExampleDB with the ERR_LOCK_TIMEOUT
        troubleshooting flag enabled.</p>
        </body></html>
    """,
    "https://docs.example.com/api": """
        <html><head><title>API Reference</title></head><body>
        <h1>API Reference</h1>
        <p>The query endpoint accepts SQL statements and returns JSON. Vector
        search is supported through the SIMILAR TO operator.</p>
        </body></html>
    """,
}


@pytest.fixture()
def fake_fetcher():
    def _fetch(url: str) -> str:
        key = url.rstrip("/") or url
        if key not in FIXTURE_SITE:
            raise ValueError(f"404 for {url}")
        return FIXTURE_SITE[key]

    return _fetch


@pytest.fixture()
def client(tmp_path, fake_fetcher) -> TestClient:
    app = create_app(db_path=str(tmp_path / "test.duckdb"), fetcher=fake_fetcher)
    return TestClient(app)


@pytest.fixture()
def indexed_client(client: TestClient) -> TestClient:
    """Client with the fixture site already crawled synchronously."""
    response = client.post("/fetch_url?sync=true", json={"url": "https://docs.example.com"})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    return client
