"""FastAPI entrypoint for the Lao tutor backend."""
from __future__ import annotations

import base64
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import get_settings
from .logging_utils import configure_logging, get_logger
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

app = FastAPI(title="Lao Tutor API")
settings = get_settings()
configure_logging(settings.log_dir)
logger = get_logger(__name__)
logger.info("Application initialised", extra={"log_dir": str(settings.log_dir)})

tutor_engine = TutorEngine()
conversation_service = ConversationService(tutor_engine.export_phrase_banks())
logger.info(
    "Tutor services ready",
    extra={
        "vad": tutor_engine.vad.backend_name,
        "tts_ready": tutor_engine.tts.is_ready,
        "llm_ready": conversation_service.is_ready,
    },
)

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
    logger.debug("Serving index", extra={"accept": accept_header})
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

          .msg strong {
            margin-right: 0.3rem;
          }

          .msg-text {
            white-space: pre-wrap;
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
            display: inline-flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            align-items: baseline;
            background: rgba(68, 80, 214, 0.08);
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
          }

          .msg-spoken {
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 0.35rem;
          }

          .msg-spoken audio {
            margin-left: 0.35rem;
            height: 32px;
          }

          .msg-feedback {
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: var(--text-muted);
          }

          .msg-feedback ul {
            margin: 0.3rem 0 0;
            padding-left: 1.1rem;
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
            gap: 0.75rem;
            flex-wrap: wrap;
          }

          .chat-buttons {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
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

          .chat-actions button.secondary {
            background: rgba(68, 80, 214, 0.12);
            color: var(--accent);
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
              <form id=\"chat-form\" class=\"chat-form\">\n                <label for=\"chat-input\" class=\"sr-only\">Your message</label>\n                <textarea id=\"chat-input\" name=\"message\" rows=\"3\" placeholder=\"Type in English or Lao...\"></textarea>\n                <div class=\"chat-actions\">\n                  <div class=\"chat-buttons\">\n                    <button id=\"chat-record\" type=\"button\" class=\"secondary\">🎙️ Record phrase</button>\n                    <button id=\"chat-send\" type=\"submit\">Send</button>\n                  </div>\n                  <span id=\"chat-status\" class=\"chat-status\"></span>\n                </div>\n              </form>
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
                <p>Record a phrase above or POST base64-encoded 16&nbsp;kHz PCM for analysis.</p>
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
              <li>Click <strong>🎙️ Record phrase</strong> above to capture a Lao utterance or POST a base64 clip to <code>/api/v1/utterance</code>.</li>
              <li>Review the recognised Lao, romanisation, and corrections, then listen back to the teacher audio.</li>
              <li>Poll <code>/health</code> to verify ASR/TTS/LLM availability before integrating a client.</li>
            </ol>
          </section>

          <footer>
            Prefer JSON? Send <code>Accept: application/json</code> with your request.
          </footer>
        </main>
        <script>
          (function () {
            const TARGET_SAMPLE_RATE = 16000;
            const chatLog = document.getElementById('chat-log');
            const chatForm = document.getElementById('chat-form');
            const chatInput = document.getElementById('chat-input');
            const chatSend = document.getElementById('chat-send');
            const chatStatus = document.getElementById('chat-status');
            const recordBtn = document.getElementById('chat-record');
            let history = [];
            let audioContext;
            let mediaRecorder;
            let recordingStream;
            let audioChunks = [];
            let isRecording = false;
            let pendingUserEntry = null;
            const teacherAudioUrls = new Set();

            function cleanupTeacherAudioUrls() {
              teacherAudioUrls.forEach((url) => URL.revokeObjectURL(url));
              teacherAudioUrls.clear();
            }

            window.addEventListener('beforeunload', cleanupTeacherAudioUrls);

            function ensureAudioContext() {
              if (!audioContext) {
                const Ctx = window.AudioContext || window.webkitAudioContext;
                if (Ctx) {
                  audioContext = new Ctx();
                }
              }
              return audioContext;
            }

            function setStatus(message) {
              chatStatus.textContent = message || '';
            }

            function appendMessage(role, content, options = {}) {
              const entry = document.createElement('div');
              entry.className = role === 'assistant' ? 'msg msg-assistant' : 'msg msg-user';
              entry.dataset.role = role;
              const label = document.createElement('strong');
              label.textContent = role === 'assistant' ? 'Tutor:' : 'You:';
              entry.appendChild(label);
              const span = document.createElement('span');
              span.className = 'msg-text';
              span.textContent = ` ${content}`;
              entry.appendChild(span);
              if (options.focusPhrase) {
                appendFocus(entry, options.focusPhrase, options.focusTranslation);
              }
              if (options.spokenText || options.spokenAudioUrl) {
                appendSpoken(entry, options.spokenText, options.spokenAudioUrl);
              }
              chatLog.appendChild(entry);
              chatLog.scrollTop = chatLog.scrollHeight;
              return entry;
            }

            function appendFocus(entry, phrase, translation) {
              if (!phrase) return;
              const focus = document.createElement('div');
              focus.className = 'msg-focus';
              focus.textContent = 'Focus phrase: ';
              const laoSpan = document.createElement('span');
              laoSpan.className = 'lao';
              laoSpan.textContent = phrase;
              focus.appendChild(laoSpan);
              if (translation) {
                const translationSpan = document.createElement('span');
                translationSpan.textContent = ` · ${translation}`;
                focus.appendChild(translationSpan);
              }
              entry.appendChild(focus);
            }

            function appendSpoken(entry, spokenText, audioUrl) {
              if (!spokenText && !audioUrl) return;
              const spoken = document.createElement('div');
              spoken.className = 'msg-spoken';
              const icon = document.createElement('span');
              icon.setAttribute('aria-hidden', 'true');
              icon.textContent = '🎧';
              spoken.appendChild(icon);
              const text = document.createElement('span');
              text.textContent = spokenText ? `Spoken reply: ${spokenText}` : 'Spoken reply available';
              spoken.appendChild(text);
              if (audioUrl) {
                const audioEl = document.createElement('audio');
                audioEl.controls = true;
                audioEl.preload = 'auto';
                audioEl.src = audioUrl;
                audioEl.setAttribute('aria-label', 'Tutor pronunciation playback');
                spoken.appendChild(audioEl);
                setTimeout(() => {
                  audioEl.play().catch((error) => {
                    console.warn('Automatic tutor playback blocked', error);
                  });
                }, 0);
              }
              entry.appendChild(spoken);
            }

            function appendUtteranceFeedback(entry, feedback) {
              if (!feedback) return;
              const wrap = document.createElement('div');
              wrap.className = 'msg-feedback';
              const details = [];
              if (feedback.lao_text) {
                details.push(`Heard: ${feedback.lao_text}`);
              }
              if (feedback.romanised) {
                details.push(`Romanisation: ${feedback.romanised}`);
              }
              if (feedback.translation) {
                details.push(`Meaning: ${feedback.translation}`);
              }
              wrap.textContent = details.join(' · ');
              if (feedback.corrections && feedback.corrections.length) {
                const list = document.createElement('ul');
                feedback.corrections.forEach((hint) => {
                  const item = document.createElement('li');
                  item.textContent = hint;
                  list.appendChild(item);
                });
                wrap.appendChild(list);
              }
              if (feedback.praise) {
                const praise = document.createElement('div');
                praise.textContent = feedback.praise;
                wrap.appendChild(praise);
              }
              entry.appendChild(wrap);
            }

            function updateMessage(entry, content) {
              if (!entry) return;
              const span = entry.querySelector('.msg-text');
              if (span) {
                span.textContent = ` ${content}`;
              }
            }

            function base64ToFloat32(base64) {
              const binary = atob(base64);
              const bytes = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i += 1) {
                bytes[i] = binary.charCodeAt(i);
              }
              if (bytes.length % 4 !== 0) {
                throw new Error('Teacher audio payload is malformed');
              }
              return new Float32Array(bytes.buffer);
            }

            function float32ToWavBytes(floatArray, sampleRate) {
              const bytesPerSample = 2;
              const blockAlign = bytesPerSample;
              const dataSize = floatArray.length * bytesPerSample;
              const buffer = new ArrayBuffer(44 + dataSize);
              const view = new DataView(buffer);
              let offset = 0;

              const writeString = (str) => {
                for (let i = 0; i < str.length; i += 1) {
                  view.setUint8(offset + i, str.charCodeAt(i));
                }
                offset += str.length;
              };

              writeString('RIFF');
              view.setUint32(offset, 36 + dataSize, true);
              offset += 4;
              writeString('WAVE');
              writeString('fmt ');
              view.setUint32(offset, 16, true); // Subchunk1Size
              offset += 4;
              view.setUint16(offset, 1, true); // PCM format
              offset += 2;
              view.setUint16(offset, 1, true); // Mono channel
              offset += 2;
              view.setUint32(offset, sampleRate, true);
              offset += 4;
              view.setUint32(offset, sampleRate * blockAlign, true);
              offset += 4;
              view.setUint16(offset, blockAlign, true);
              offset += 2;
              view.setUint16(offset, bytesPerSample * 8, true);
              offset += 2;
              writeString('data');
              view.setUint32(offset, dataSize, true);
              offset += 4;

              const pcm = new Int16Array(buffer, offset, floatArray.length);
              for (let i = 0; i < floatArray.length; i += 1) {
                const sample = Math.max(-1, Math.min(1, floatArray[i]));
                pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
              }

              return new Uint8Array(buffer);
            }

            function prepareTeacherAudioClip(base64, sampleRate) {
              const floatData = base64ToFloat32(base64);
              if (!floatData.length) {
                throw new Error('Empty teacher audio payload');
              }
              const sr = Number(sampleRate) || TARGET_SAMPLE_RATE;
              const wavBytes = float32ToWavBytes(floatData, sr);
              const blob = new Blob([wavBytes], { type: 'audio/wav' });
              const url = URL.createObjectURL(blob);
              teacherAudioUrls.add(url);
              return { url, duration: floatData.length / sr };
            }

            function floatToPcm16Base64(floatArray) {
              const buffer = new ArrayBuffer(floatArray.length * 2);
              const view = new DataView(buffer);
              for (let i = 0; i < floatArray.length; i += 1) {
                const sample = Math.max(-1, Math.min(1, floatArray[i]));
                view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
              }
              const bytes = new Uint8Array(buffer);
              let binary = '';
              const chunkSize = 0x8000;
              for (let i = 0; i < bytes.length; i += chunkSize) {
                const chunk = bytes.subarray(i, i + chunkSize);
                binary += String.fromCharCode.apply(null, Array.from(chunk));
              }
              return btoa(binary);
            }

            function downsampleToTarget(audioBuffer) {
              const sourceRate = audioBuffer.sampleRate;
              const channelData = audioBuffer.getChannelData(0);
              if (sourceRate === TARGET_SAMPLE_RATE) {
                return new Float32Array(channelData);
              }
              const ratio = sourceRate / TARGET_SAMPLE_RATE;
              const length = Math.round(channelData.length / ratio);
              const result = new Float32Array(length);
              let offsetResult = 0;
              let offsetBuffer = 0;
              while (offsetResult < length) {
                const nextOffsetBuffer = Math.min(channelData.length, Math.round((offsetResult + 1) * ratio));
                let accum = 0;
                let count = 0;
                for (let i = offsetBuffer; i < nextOffsetBuffer; i += 1) {
                  accum += channelData[i];
                  count += 1;
                }
                result[offsetResult] = count > 0 ? accum / count : 0;
                offsetResult += 1;
                offsetBuffer = nextOffsetBuffer;
              }
              return result;
            }

            async function sendConversation(payload, { userEntry } = {}) {
              chatSend.disabled = true;
              if (recordBtn) {
                recordBtn.disabled = true;
              }
              setStatus('Thinking…');
              try {
                const response = await fetch('/api/v1/conversation', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ ...payload, history }),
                });
                if (!response.ok) {
                  throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                history = data.history || [];
                if (userEntry && data.heard_text) {
                  updateMessage(userEntry, `🎤 ${data.heard_text}`);
                }
                let spokenAudioUrl = null;
                if (data.teacher_audio_base64 && data.teacher_audio_sample_rate) {
                  try {
                    const clip = prepareTeacherAudioClip(
                      data.teacher_audio_base64,
                      data.teacher_audio_sample_rate,
                    );
                    spokenAudioUrl = clip.url;
                  } catch (err) {
                    console.warn('Failed to prepare tutor audio', err);
                  }
                }
                const replyEntry = appendMessage('assistant', data.reply?.content || 'I am still getting ready to chat.', {
                  focusPhrase: data.focus_phrase,
                  focusTranslation: data.focus_translation,
                  spokenText: data.spoken_text,
                  spokenAudioUrl,
                });
                appendUtteranceFeedback(replyEntry, data.utterance_feedback);
              } catch (error) {
                console.error(error);
                appendMessage('assistant', 'I ran into a problem understanding that. Please try again after a moment.');
              } finally {
                chatSend.disabled = false;
                if (recordBtn) {
                  recordBtn.disabled = false;
                }
                setStatus('');
              }
            }

            chatForm.addEventListener('submit', async (event) => {
              event.preventDefault();
              const message = chatInput.value.trim();
              if (!message) return;
              const userEntry = appendMessage('user', message);
              chatInput.value = '';
              chatInput.focus();
              await sendConversation({ message }, { userEntry });
            });

            async function startRecording() {
              if (!navigator.mediaDevices || !window.MediaRecorder) {
                appendMessage('assistant', 'Microphone recording is not supported in this browser.');
                return;
              }
              try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                recordingStream = stream;
                const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                  ? { mimeType: 'audio/webm;codecs=opus' }
                  : undefined;
                mediaRecorder = new MediaRecorder(stream, options);
                audioChunks = [];
                mediaRecorder.ondataavailable = (event) => {
                  if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                  }
                };
                mediaRecorder.onstart = () => {
                  pendingUserEntry = appendMessage('user', '🎤 Listening…');
                  setStatus('Recording… tap stop when finished.');
                  chatSend.disabled = true;
                };
                mediaRecorder.onstop = async () => {
                  setStatus('Processing audio…');
                  chatSend.disabled = false;
                  if (recordBtn) {
                    recordBtn.disabled = true;
                  }
                  try {
                    if (pendingUserEntry) {
                      updateMessage(pendingUserEntry, '🎤 Processing audio…');
                    }
                    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                    const arrayBuffer = await blob.arrayBuffer();
                    const ctx = ensureAudioContext();
                    if (!ctx) {
                      appendMessage('assistant', 'Audio playback is not supported in this environment.');
                      return;
                    }
                    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
                    const floatData = downsampleToTarget(audioBuffer);
                    const base64 = floatToPcm16Base64(floatData);
                    await sendConversation(
                      { audio_base64: base64, sample_rate: TARGET_SAMPLE_RATE },
                      { userEntry: pendingUserEntry }
                    );
                  } catch (error) {
                    console.error(error);
                    appendMessage('assistant', 'I could not process that audio clip. Please try again.');
                  } finally {
                    if (recordBtn) {
                      recordBtn.disabled = false;
                      recordBtn.textContent = '🎙️ Record phrase';
                    }
                    setStatus('');
                    pendingUserEntry = null;
                    if (recordingStream) {
                      recordingStream.getTracks().forEach((track) => track.stop());
                      recordingStream = null;
                    }
                  }
                };
                mediaRecorder.start();
                isRecording = true;
                if (recordBtn) {
                  recordBtn.textContent = '⏹️ Stop recording';
                }
              } catch (error) {
                console.error(error);
                appendMessage('assistant', 'Microphone permission was denied or not available.');
              }
            }

            function stopRecording() {
              if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
              }
            }

            if (recordBtn) {
              if (!navigator.mediaDevices || !window.MediaRecorder) {
                recordBtn.disabled = true;
                recordBtn.textContent = 'Recording unavailable';
              }
              recordBtn.addEventListener('click', async () => {
                if (isRecording) {
                  stopRecording();
                } else {
                  await startRecording();
                }
              });
            }
          })();
        </script>

      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    logger.info(
        "Health check invoked",
        extra={
            "vad_backend": tutor_engine.vad.backend_name,
            "asr_ready": tutor_engine.asr.is_ready,
            "tts_ready": tutor_engine.tts.is_ready,
            "llm_ready": conversation_service.is_ready,
        },
    )
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
        logger.warning("Invalid audio payload", exc_info=exc)
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc
    if len(audio_bytes) % 2 == 0:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio.size == 0:
        logger.warning("Empty audio payload received")
        raise HTTPException(status_code=400, detail="Empty audio payload")
    logger.debug(
        "Decoded audio payload",
        extra={"frames": int(audio.size), "expected_sample_rate": expected_sample_rate},
    )
    return audio


@app.post("/api/v1/utterance", response_model=UtteranceResponse)
def handle_utterance(payload: UtteranceRequest) -> UtteranceResponse:
    sample_rate = payload.sample_rate or settings.sample_rate
    logger.info(
        "Processing utterance",
        extra={
            "sample_rate": sample_rate,
            "task_id": payload.task_id,
            "has_audio": bool(payload.audio_base64),
        },
    )
    audio = _decode_audio(payload.audio_base64, sample_rate)
    feedback: SegmentFeedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
    logger.debug(
        "Utterance processed",
        extra={
            "heard_text": feedback.lao_text,
            "romanised": feedback.romanised,
            "corrections": len(feedback.corrections or []),
        },
    )
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

    sample_rate = payload.sample_rate or settings.sample_rate
    utterance_feedback: Optional[SegmentFeedback] = None
    heard_text: Optional[str] = None

    message_text = payload.message.strip() if payload.message else ""

    logger.info(
        "Conversation turn received",
        extra={
            "history_count": len(history_payload),
            "has_audio": bool(payload.audio_base64),
            "task_id": payload.task_id,
        },
    )

    if payload.audio_base64:
        audio = _decode_audio(payload.audio_base64, sample_rate)
        utterance_feedback = tutor_engine.process_audio(audio, sample_rate, payload.task_id)
        heard_text = utterance_feedback.lao_text or None
        if not message_text:
            message_text = utterance_feedback.lao_text or utterance_feedback.romanised or ""
        if not message_text and utterance_feedback.corrections:
            message_text = utterance_feedback.corrections[0]

    if not message_text.strip():
        message_text = "I could not speak clearly."

    try:
        result = conversation_service.generate(history_payload, message_text, payload.task_id)
    except ValueError as exc:
        logger.error("Conversation generation failed", exc_info=exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tts_result = None
    spoken_text = result.spoken_text
    if spoken_text:
        tts_result = tutor_engine.prepare_teacher_audio(text_override=spoken_text)
    elif result.focus_phrase:
        tts_result = tutor_engine.prepare_teacher_audio(text_override=result.focus_phrase)

    response_history = [ChatMessage(**entry) for entry in result.history]
    reply_message = ChatMessage(role="assistant", content=result.reply_text)

    logger.debug(
        "Conversation turn processed",
        extra={
            "reply_chars": len(result.reply_text),
            "heard_text": heard_text,
            "spoken_text": spoken_text,
            "teacher_audio": bool(tts_result),
        },
    )

    debug_payload: dict[str, Any] = dict(result.debug)
    if payload.audio_base64:
        debug_payload.update(
            {
                "audio_processed": True,
                "sample_rate": sample_rate,
                "vad_backend": tutor_engine.vad.backend_name,
                "asr_ready": tutor_engine.asr.is_ready,
            }
        )
    else:
        debug_payload.setdefault("audio_processed", False)

    return ConversationResponse(
        reply=reply_message,
        history=response_history,
        heard_text=heard_text,
        focus_phrase=result.focus_phrase,
        focus_translation=result.focus_translation,
        spoken_text=spoken_text,
        utterance_feedback=utterance_feedback,
        teacher_audio_base64=tts_result.audio_base64 if tts_result else None,
        teacher_audio_sample_rate=tts_result.sample_rate if tts_result else None,
        debug=debug_payload,
    )


__all__ = ["app"]
