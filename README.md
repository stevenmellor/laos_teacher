# Lao Teacher Voice Tutor

This repository contains an experimental, local-first Lao (ລາວ) language tutor. The initial milestone focuses on the backend voice-processing loop described in the architectural plan.

## Features

- FastAPI backend with `/health`, `/api/v1/utterance`, and `/api/v1/conversation` endpoints
- Voice activity detection prioritising WebRTC VAD, with Silero and energy fallbacks
- Whisper ASR wrapper for Lao transcription (gracefully degrades when models are absent)
- Lao segmentation and romanisation helper with LaoNLP integration when installed
- Meta MMS-based Lao TTS interface with fallback behaviour
- SQLite-backed spaced repetition (SM-2 style) store to track learner progress

## Getting started

1. Install [uv](https://github.com/astral-sh/uv) (fast Python/virtualenv manager) and sync dependencies pinned for Python 3.10+:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.10
   uv sync
   ```

   The `pyproject.toml` defines optional extras. Install the full speech stack when you are ready to run inference locally:

   ```bash
   uv pip install '.[speech]'
   ```

   Add the conversational LLM components (TinyLlama by default) with:

   ```bash
   uv pip install '.[llm]'
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

3. Send audio (base64 encoded PCM) to the `/api/v1/utterance` endpoint to receive feedback and optional synthesized audio.

4. Chat with the tutor over text (and optional generated audio) via `/api/v1/conversation` or by using the interactive widget on the root page.

## Tests

```bash
pytest
```

## Roadmap

This commit seeds the backend scaffolding. Next steps include hooking up real curriculum content, refining correction heuristics, adding pronunciation scoring, and building the WebRTC/Next.js frontend loop.
