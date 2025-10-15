import base64
import struct

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
    assert "heard_text" in payload
    assert "spoken_text" in payload
    assert payload["utterance_feedback"] is None


def test_conversation_rejects_empty_payload():
    client = TestClient(app)
    response = client.post("/api/v1/conversation", json={"history": []})
    assert response.status_code == 422


def test_conversation_accepts_audio_only():
    client = TestClient(app)
    silence = struct.pack("<16h", *([0] * 16))
    payload = {
        "audio_base64": base64.b64encode(silence).decode("utf-8"),
        "sample_rate": 16000,
        "history": [],
    }
    response = client.post("/api/v1/conversation", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]["role"] == "assistant"
    assert body["utterance_feedback"] is not None
    assert "spoken_text" in body
