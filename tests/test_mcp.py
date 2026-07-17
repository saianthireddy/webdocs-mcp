import json


def _rpc(client, method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return client.post("/mcp", json=message)


def test_initialize_handshake(client):
    body = _rpc(client, "initialize", {"protocolVersion": "2024-11-05"}).json()
    assert body["result"]["serverInfo"]["name"] == "webdocs-mcp"
    assert "tools" in body["result"]["capabilities"]


def test_notification_returns_202(client):
    response = client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert response.status_code == 202


def test_tools_list(client):
    tools = _rpc(client, "tools/list").json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"search_docs", "list_doc_pages", "get_doc_page"}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"


def test_tools_call_search_docs(indexed_client):
    body = _rpc(indexed_client, "tools/call",
                {"name": "search_docs", "arguments": {"query": "vector search", "top_k": 2}}).json()
    result = body["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload and "url" in payload[0]


def test_tools_call_page_flow(indexed_client):
    pages = json.loads(
        _rpc(indexed_client, "tools/call", {"name": "list_doc_pages", "arguments": {}})
        .json()["result"]["content"][0]["text"]
    )
    assert len(pages) == 4
    page_text = (
        _rpc(indexed_client, "tools/call",
             {"name": "get_doc_page", "arguments": {"page_id": pages[0]["page_id"]}})
        .json()["result"]["content"][0]["text"]
    )
    assert "ExampleDB" in page_text


def test_unknown_tool_and_method(client):
    body = _rpc(client, "tools/call", {"name": "nope", "arguments": {}}).json()
    assert body["result"]["isError"] is True
    body = _rpc(client, "no/such/method").json()
    assert body["error"]["code"] == -32601


def test_batch_request(client):
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    responses = client.post("/mcp", json=batch).json()
    assert len(responses) == 2  # notification dropped
    assert {r["id"] for r in responses} == {1, 2}
