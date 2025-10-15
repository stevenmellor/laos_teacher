from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "vad_backend" in payload
    assert "whisper_loaded" in payload
    assert "llm_available" in payload


def test_index_serves_html_by_default():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"].lower()
    assert "Lao Tutor API" in response.text
    assert "Try the conversational tutor" in response.text


def test_index_returns_json_when_requested():
    client = TestClient(app)
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["service"] == "lao-tutor"


def test_conversation_endpoint_returns_reply():
    client = TestClient(app)
    response = client.post(
        "/api/v1/conversation",
        json={"message": "Hello", "history": []},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"]["role"] == "assistant"
    assert payload["reply"]["content"]
    assert isinstance(payload["history"], list)
    assert payload["history"], "History should include the new turn"