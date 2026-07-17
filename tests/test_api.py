def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pages"] == 0


def test_fetch_url_sync_indexes_site(indexed_client):
    body = indexed_client.get("/health").json()
    assert body["pages"] == 4 and body["chunks"] >= 4


def test_job_progress(indexed_client):
    jobs = indexed_client.get("/job_progress").json()
    assert len(jobs) == 1 and jobs[0]["status"] == "completed"
    job = indexed_client.get("/job_progress", params={"job_id": jobs[0]["job_id"]}).json()
    assert job["pages_crawled"] == 4
    assert indexed_client.get("/job_progress", params={"job_id": "nope"}).status_code == 404


def test_search_docs_endpoint(indexed_client):
    results = indexed_client.get("/search_docs", params={"query": "docker compose", "top_k": 3}).json()
    assert results and "docker" in results[0]["text"].lower()
    assert {"url", "title", "score", "method", "text"} <= set(results[0])


def test_list_and_get_doc_page(indexed_client):
    pages = indexed_client.get("/list_doc_pages").json()
    assert len(pages) == 4
    page = indexed_client.get("/get_doc_page", params={"page_id": pages[0]["page_id"]}).json()
    assert page["url"].startswith("https://docs.example.com")
    assert indexed_client.get("/get_doc_page", params={"page_id": "nope"}).status_code == 404


def test_map_views_render_html(indexed_client):
    index_html = indexed_client.get("/map").text
    assert "Crawled sites" in index_html and "docs.example.com" in index_html

    root_id = indexed_client.get("/list_doc_pages").json()[0]["page_id"]
    tree_html = indexed_client.get(f"/map/site/{root_id}").text
    assert "Site map" in tree_html and "Installation" in tree_html

    page_html = indexed_client.get(f"/map/page/{root_id}").text
    assert "Children" in page_html and "<script" not in page_html.lower().replace("noscript", "")

    raw = indexed_client.get(f"/map/page/{root_id}/raw").text
    assert "ExampleDB" in raw
    assert indexed_client.get("/map/page/nope").status_code == 404
