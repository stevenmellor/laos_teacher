import base64
import struct

import pytest

from fastapi.testclient import TestClient

from backend.app import main as main_module
from backend.app.logging_utils import get_logger
from backend.app.main import app
from backend.app.services.asr import AsrResult
from backend.app.services.tts import TtsResult

logger = get_logger(__name__)
logger.debug("Test module loaded")


def test_health_endpoint():
    logger.info("Running health endpoint test")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "vad_backend" in payload
    assert "whisper_loaded" in payload
    assert "llm_available" in payload
    assert "translation_available" in payload


def test_index_serves_html_by_default():
    logger.info("Running HTML index test")
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"].lower()
    assert "Lao Tutor API" in response.text
    assert "Try the conversational tutor" in response.text


def test_index_returns_json_when_requested():
    logger.info("Running JSON index test")
    client = TestClient(app)
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["service"] == "lao-tutor"


def test_conversation_endpoint_returns_reply():
    logger.info("Running conversation reply test")
    client = TestClient(app)
    response = client.post(
        "/api/v1/conversation",
        json={"message": "Hello", "history": []},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"]["role"] == "assistant"
    assert payload["reply"]["content"]
    assert "Let's practise" in payload["reply"]["content"]
    assert isinstance(payload["history"], list)
    assert payload["history"], "History should include the new turn"
    assert "heard_text" in payload
    assert "spoken_text" in payload
    assert payload["utterance_feedback"] is None


def test_conversation_rejects_empty_payload():
    logger.info("Running conversation validation test")
    client = TestClient(app)
    response = client.post("/api/v1/conversation", json={"history": []})
    assert response.status_code == 422


def test_conversation_accepts_audio_only():
    logger.info("Running conversation audio-only test")
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
    assert "Let's practise" in body["reply"]["content"]
    assert "spoken_text" in body


def test_conversation_audio_roundtrip_includes_teacher_audio(monkeypatch: pytest.MonkeyPatch):
    logger.info("Running conversation audio roundtrip test")
    client = TestClient(app)

    def fake_transcribe(audio, sample_rate):
        return AsrResult(text="ທົດລອງ", language="lo", confidence=0.9)

    fake_teacher_audio = base64.b64encode(struct.pack("<8h", *([1200] * 8))).decode("utf-8")

    def fake_synthesize(text):
        return TtsResult(audio_base64=fake_teacher_audio, sample_rate=16000)

    monkeypatch.setattr(main_module.tutor_engine.asr, "transcribe", fake_transcribe)
    monkeypatch.setattr(main_module.tutor_engine.tts, "synthesize", fake_synthesize)

    loud_audio = struct.pack("<32h", *([2000] * 32))
    payload = {
        "audio_base64": base64.b64encode(loud_audio).decode("utf-8"),
        "sample_rate": 16000,
        "history": [],
    }

    response = client.post("/api/v1/conversation", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["heard_text"] == "ທົດລອງ"
    assert data["teacher_audio_base64"] == fake_teacher_audio
    assert data["teacher_audio_sample_rate"] == 16000
    assert data["utterance_feedback"] is not None
    assert "I heard you say" in data["reply"]["content"]
