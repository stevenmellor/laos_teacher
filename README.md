# Lao Teacher Voice Tutor

This repository contains an experimental, local-first Lao (ລາວ) language tutor. The initial milestone focuses on the backend voice-processing loop described in the architectural plan.

## Features

- FastAPI backend with `/health`, `/api/v1/utterance`, and `/api/v1/conversation` endpoints
- Interactive landing page featuring a chat log, Lao focus prompts, and in-browser microphone capture
- Voice activity detection prioritising WebRTC VAD, with Silero and energy fallbacks
- Whisper ASR wrapper for Lao transcription (gracefully degrades when models are absent)
- Lao segmentation and romanisation helper with LaoNLP integration when installed
- Meta MMS-based Lao TTS interface (defaulting to `facebook/mms-tts-lao`) with graceful fallback behaviour
- SQLite-backed spaced repetition (SM-2 style) store to track learner progress

## Getting started

1. Install [uv](https://github.com/astral-sh/uv) (fast Python/virtualenv manager) and sync dependencies pinned for Python 3.10+:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.10
   uv sync
   ```

   The `pyproject.toml` defines optional extras. Install the full speech stack (Whisper, Lao NLP helpers, MMS TTS, Torch, etc.) when you are ready to run inference locally:

   ```bash
   uv pip install '.[speech]'
   ```

   Add the conversational LLM components (TinyLlama by default) with:

   ```bash
   uv pip install '.[llm]'
   ```

   You can install both extras together when you want a fully voiced experience:

   ```bash
   uv pip install '.[speech,llm]'
   ```

   For development workflows that rely on the bundled pytest suite, install the dev tools extra (or combine it with `speech`).

   ```bash
   uv pip install '.[dev]'
   ```

   For lightweight API-only development you can continue to rely on `backend/requirements.txt`, which mirrors the base dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

2. Run the FastAPI application:

   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

3. From the landing page, click **🎙️ Record phrase** to capture a Lao utterance directly in the browser. The backend will transcribe it, surface romanisation/corrections, and (when models are available) play back teacher audio generated with Meta's MMS Lao voice.

4. Alternatively, send base64-encoded PCM audio to `/api/v1/utterance` programmatically for the same feedback pipeline.

5. Chat with the tutor over text and generated Lao speech via `/api/v1/conversation` or by using the interactive widget on the root page. Each tutor reply now includes a "spoken reply" snippet so you know exactly what the voice model is saying.

Environment variables such as `LAO_TUTOR_TTS_MODEL_NAME` and `LAO_TUTOR_TTS_DEVICE` can be used to switch voices or target a GPU/MPS runtime for faster synthesis.

## Tests

```bash
pytest
```

## Roadmap

This commit seeds the backend scaffolding. Next steps include hooking up real curriculum content, refining correction heuristics, adding pronunciation scoring, and building the WebRTC/Next.js frontend loop.
