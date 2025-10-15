"""FastAPI entrypoint for the Lao tutor backend."""
from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import get_settings
from .models.schemas import (
    ChatMessage,
    ConversationRequest,
    ConversationResponse,
    HealthResponse,
    SegmentFeedback,
    UtteranceRequest,
    UtteranceResponse,
)
from .services.tutor import TutorEngine
from .services.llm import ConversationService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Lao Tutor API")
settings = get_settings()
tutor_engine = TutorEngine()
conversation_service = ConversationService(tutor_engine.export_phrase_banks())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    """Landing endpoint that serves HTML by default with a JSON fallback."""
    accept_header = (request.headers.get("accept") or "").lower()
    payload = {
        "service": "lao-tutor",
        "message": "Use /api/v1/utterance for audio interactions or /health for status",
        "docs": "/docs",
    }

    if "application/json" in accept_header and "text/html" not in accept_header:
        return JSONResponse(payload)

    html = """<!DOCTYPE html>
    <html lang=\"en\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Lao Tutor API</title>
        <style>
          :root {
            color-scheme: light dark;
            --bg: radial-gradient(circle at top, #f3f5ff 0%, #eef3ff 35%, #ffffff 100%);
            --card-bg: rgba(255, 255, 255, 0.85);
            --border: rgba(120, 136, 189, 0.35);
            --text-main: #1f273d;
            --text-muted: #4a5674;
            --accent: #4450d6;
          }

          body {
            margin: 0;
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 3rem 1.5rem 4rem;
          }

          main {
            width: min(860px, 100%);
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 20px 45px rgba(80, 98, 160, 0.18);
            backdrop-filter: blur(14px);
            padding: 2.5rem clamp(1.5rem, 3vw, 3rem);
            line-height: 1.7;
          }

          h1 {
            font-size: clamp(2.2rem, 4vw, 2.8rem);
            margin: 0 0 0.2rem;
          }

          h2 {
            font-size: clamp(1.2rem, 3vw, 1.5rem);
            margin-top: 2.2rem;
            margin-bottom: 0.6rem;
          }

          p.lead {
            margin: 0;
            color: var(--text-muted);
            font-size: 1.05rem;
          }

          section {
            border-top: 1px solid var(--border);
            padding-top: 1.6rem;
          }

          ul, ol {
            padding-left: 1.1rem;
            margin: 0.6rem 0;
          }

          li {
            margin-bottom: 0.35rem;
          }

          code {
            background: rgba(68, 80, 214, 0.08);
            color: var(--accent);
            padding: 0.15rem 0.4rem;
            border-radius: 6px;
            font-size: 0.95rem;
          }

          a {
            color: var(--accent);
            font-weight: 600;
            text-decoration: none;
          }

          a:hover {
            text-decoration: underline;
          }

          .cta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.2rem;
            margin-top: 1.4rem;
          }

          .chat-section {
            margin-top: 2rem;
          }

          .chat-card {
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.2rem clamp(1rem, 2vw, 1.6rem);
            background: rgba(255, 255, 255, 0.7);
            box-shadow: 0 10px 30px rgba(69, 86, 150, 0.12);
            display: flex;
            flex-direction: column;
            gap: 1rem;
          }

          .chat-log {
            max-height: 320px;
            overflow-y: auto;
            padding-right: 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
          }

          .msg {
            padding: 0.6rem 0.8rem;
            border-radius: 12px;
            line-height: 1.55;
          }

          .msg-user {
            align-self: flex-end;
            background: rgba(68, 80, 214, 0.12);
          }

          .msg-assistant {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(68, 80, 214, 0.1);
          }

          .msg-focus {
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: var(--text-muted);
          }

          .chat-form textarea {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem;
            font-size: 1rem;
            resize: vertical;
            min-height: 90px;
            font-family: inherit;
          }

          .chat-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.6rem;
          }

          .chat-actions button {
            background: var(--accent);
            color: #fff;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.1s ease;
          }

          .chat-actions button:disabled {
            opacity: 0.6;
            cursor: progress;
          }

          .chat-actions button:not(:disabled):hover {
            transform: translateY(-1px);
          }

          .chat-status {
            font-size: 0.9rem;
            color: var(--text-muted);
          }

          .lao {
            font-weight: 600;
            font-size: 1.05rem;
          }

          .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            border: 0;
          }

          .card {
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.65);
            box-shadow: 0 10px 25px rgba(94, 113, 178, 0.08);
          }

          footer {
            margin-top: 2.6rem;
            font-size: 0.92rem;
            color: var(--text-muted);
          }
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>Lao Tutor API</h1>
            <p class=\"lead\">Voice-first Lao language teaching backend.</p>
          </header>

          <section aria-labelledby=\"section-chat\" class=\"chat-section\">
            <h2 id=\"section-chat\">Try the conversational tutor</h2>
            <div class=\"chat-card\">
              <div id=\"chat-log\" class=\"chat-log\" aria-live=\"polite\" aria-label=\"Tutor conversation\"></div>
              <form id=\"chat-form\" class=\"chat-form\">\n                <label for=\"chat-input\" class=\"sr-only\">Your message</label>\n                <textarea id=\"chat-input\" name=\"message\" rows=\"3\" placeholder=\"Type in English or Lao...\" required></textarea>\n                <div class=\"chat-actions\">\n                  <button id=\"chat-send\" type=\"submit\">Send</button>\n                  <span id=\"chat-status\" class=\"chat-status\"></span>\n                </div>\n              </form>
            </div>
          </section>

          <section aria-labelledby=\"section-try\">
            <h2 id=\"section-try\">Try it out</h2>
            <div class=\"cta-grid\">
              <article class=\"card\">
                <h3>Health check</h3>
                <p>Confirm model availability and backend health.</p>
                <code>/health</code>
              </article>
              <article class=\"card\">
                <h3>Send audio</h3>
                <p>POST base64-encoded 16&nbsp;kHz mono PCM segments for analysis.</p>
                <code>/api/v1/utterance</code>
              </article>
              <article class=\"card\">
                <h3>Interactive docs</h3>
                <p>Explore request/response schemas and run sample calls.</p>
                <a href=\"/docs\">Swagger UI</a>
              </article>
            </div>
          </section>

          <section aria-labelledby=\"section-quickstart\">
            <h2 id=\"section-quickstart\">Quickstart</h2>
            <ol>
              <li>Base64-encode your Lao utterance audio (16&nbsp;kHz mono) and POST it to <code>/api/v1/utterance</code>.</li>
              <li>Receive transcription feedback, teaching prompts, and synthesized teacher audio.</li>
              <li>Poll <code>/health</code> to verify ASR/TTS availability.</li>
            </ol>
          </section>

          <footer>
            Prefer JSON? Send <code>Accept: application/json</code> with your request.
          </footer>
        </main>
        <script>
          (function () {
            const chatLog = document.getElementById('chat-log');
            const chatForm = document.getElementById('chat-form');
            const chatInput = document.getElementById('chat-input');
            const chatSend = document.getElementById('chat-send');
            const chatStatus = document.getElementById('chat-status');
            let history = [];
            let audioContext;

            function ensureAudioContext() {
              if (!audioContext) {
                const Ctx = window.AudioContext || window.webkitAudioContext;
                if (Ctx) {
                  audioContext = new Ctx();
                }
              }
              return audioContext;
            }

            function appendMessage(role, content, options = {}) {
              const entry = document.createElement('div');
              entry.className = role === 'assistant' ? 'msg msg-assistant' : 'msg msg-user';
              const roleLabel = role === 'assistant' ? 'Tutor' : 'You';
              entry.innerHTML = `<strong>${roleLabel}:</strong> ${content}`;
              if (options.focusPhrase) {
                const focus = document.createElement('div');
                focus.className = 'msg-focus';
                focus.innerHTML = `Focus phrase: <span class="lao">${options.focusPhrase}</span>${options.focusTranslation ? ` · <span>${options.focusTranslation}</span>` : ''}`;
                entry.appendChild(focus);
              }
              chatLog.appendChild(entry);
              chatLog.scrollTop = chatLog.scrollHeight;
            }

            function playTeacherAudio(base64, sampleRate) {
              if (!base64 || !sampleRate) return;
              const ctx = ensureAudioContext();
              if (!ctx) return;
              const binary = atob(base64);
              const buffer = new ArrayBuffer(binary.length);
              const view = new Uint8Array(buffer);
              for (let i = 0; i < binary.length; i += 1) {
                view[i] = binary.charCodeAt(i);
              }
              const floatView = new Float32Array(buffer);
              const audioBuffer = ctx.createBuffer(1, floatView.length, sampleRate);
              audioBuffer.copyToChannel(floatView, 0);
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(ctx.destination);
              source.start();
            }

            chatForm.addEventListener('submit', async (event) => {
              event.preventDefault();
              const message = chatInput.value.trim();
              if (!message) return;
              appendMessage('user', message);
              chatInput.value = '';
              chatInput.focus();
              chatSend.disabled = true;
              chatStatus.textContent = 'Thinking…';

              try {
                const response = await fetch('/api/v1/conversation', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ message, history }),
                });
                if (!response.ok) {
                  throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                history = data.history || [];
                const reply = data.reply?.content || 'I am still getting ready to chat.';
                appendMessage('assistant', reply, {
                  focusPhrase: data.focus_phrase,
                  focusTranslation: data.focus_translation,
                });
                if (data.teacher_audio_base64 && data.teacher_audio_sample_rate) {
                  playTeacherAudio(data.teacher_audio_base64, data.teacher_audio_sample_rate);
                }
              } catch (error) {
                appendMessage('assistant', 'I ran into a problem understanding that. Please try again after a moment.');
                console.error(error);
              } finally {
                chatSend.disabled = false;
                chatStatus.textContent = '';
              }
            });
          })();
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        whisper_loaded=tutor_engine.asr.is_ready,
        vad_backend=tutor_engine.vad.backend_name,
        tts_available=tutor_engine.tts.is_ready,
        llm_available=conversation_service.is_ready,
    )


def _decode_audio(audio_base64: str, expected_sample_rate: int) -> np.ndarray:
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc
    if len(audio_bytes) % 2 == 0:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    return audio


@app.post("/api/v1/utterance", response_model=UtteranceResponse)
def handle_utterance(payload: UtteranceRequest) -> UtteranceResponse:
    sample_rate = payload.sample_rate or settings.sample_rate
    audio = _decode_audio(payload.audio_base64, sample_rate)
    feedback: SegmentFeedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
    tts_result = tutor_engine.prepare_teacher_audio(feedback)
    teacher_audio_base64 = tts_result.audio_base64 if tts_result else None
    teacher_audio_sample_rate = tts_result.sample_rate if tts_result else None
    debug_info: dict[str, Any] = {
        "task": tutor_engine.state.current_task,
        "vad_backend": tutor_engine.vad.backend_name,
        "asr_ready": tutor_engine.asr.is_ready,
        "tts_ready": tutor_engine.tts.is_ready,
    }
    return UtteranceResponse(
        feedback=feedback,
        teacher_audio_base64=teacher_audio_base64,
        teacher_audio_sample_rate=teacher_audio_sample_rate,
        debug=debug_info,
    )


@app.post("/api/v1/conversation", response_model=ConversationResponse)
def handle_conversation(payload: ConversationRequest) -> ConversationResponse:
    history_payload = [message.dict() for message in payload.history]
    try:
        result = conversation_service.generate(history_payload, payload.message, payload.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tts_result = None
    if result.focus_phrase:
        tts_result = tutor_engine.prepare_teacher_audio(text_override=result.focus_phrase)

    response_history = [ChatMessage(**entry) for entry in result.history]
    reply_message = ChatMessage(role="assistant", content=result.reply_text)

    return ConversationResponse(
        reply=reply_message,
        history=response_history,
        focus_phrase=result.focus_phrase,
        focus_translation=result.focus_translation,
        teacher_audio_base64=tts_result.audio_base64 if tts_result else None,
        teacher_audio_sample_rate=tts_result.sample_rate if tts_result else None,
        debug=result.debug,
    )


__all__ = ["app"]
