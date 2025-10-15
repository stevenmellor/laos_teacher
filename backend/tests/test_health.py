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


def test_index_serves_html_by_default():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"].lower()
    assert "Lao Tutor API" in response.text


def test_index_returns_json_when_requested():
    client = TestClient(app)
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["service"] == "lao-tutor"
