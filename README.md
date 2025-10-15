# Lao Teacher Voice Tutor

This repository contains an experimental, local-first Lao (ລາວ) language tutor. The initial milestone focuses on the backend voice-processing loop described in the architectural plan.

## Features

- FastAPI backend with `/health`, `/api/v1/utterance`, and `/api/v1/conversation` endpoints
- Interactive landing page featuring a chat log, Lao focus prompts, and in-browser microphone capture
- Tutor replies include inline audio playback controls so you can rehear Lao pronunciations on demand
- Voice activity detection prioritising WebRTC VAD, with Silero and energy fallbacks
- Whisper ASR wrapper for Lao transcription (gracefully degrades when models are absent)
- Lao segmentation and romanisation helper with LaoNLP integration when installed
- Meta MMS-based Lao TTS interface (defaulting to `facebook/mms-tts-lao`) with graceful fallback behaviour
- SQLite-backed spaced repetition (SM-2 style) store to track learner progress
- Centralised logging with rotating file handlers writing to `logs/app.log`
- Optional Lao→English translation via Hugging Face NLLB so every tutor reply pairs Lao script with an English gloss

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

   The conversational + translation path relies on `transformers`, `torch`, and `sentencepiece`, all of which are covered by the
   combined extras above.

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

   Large models downloaded from Hugging Face are cached under `~/.cache/lao_tutor/models` by
   default so the auto-reloader does not thrash. If you override `LAO_TUTOR_MODEL_DIR` to a path
   within the repository, consider excluding it from the watcher:

   ```bash
   uvicorn backend.app.main:app --reload --port 8000 --reload-exclude "models/*"
   ```

3. From the landing page, click **🎙️ Record phrase** to capture a Lao utterance directly in the browser. The backend will transcribe it, surface romanisation/corrections, and (when models are available) play back teacher audio generated with Meta's MMS Lao voice.

4. Alternatively, send base64-encoded PCM audio to `/api/v1/utterance` programmatically for the same feedback pipeline.

5. Chat with the tutor over text and generated Lao speech via `/api/v1/conversation` or by using the interactive widget on the root page. Each tutor reply now includes an inline audio player and "spoken reply" snippet so you can immediately listen back or replay the teacher.

### Environment configuration

All runtime settings can be provided through environment variables (prefixed with `LAO_TUTOR_`) or a `.env` file. The table below lists the available options and their defaults.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `LAO_TUTOR_APP_NAME` | `Lao Tutor Backend` | Human-readable application name for logging/metadata. |
| `LAO_TUTOR_ENVIRONMENT` | `development` | Execution environment tag (e.g., `development`, `production`). |
| `LAO_TUTOR_DATA_DIR` | `data` | Directory for persisted learner data and lesson assets. |
| `LAO_TUTOR_MODEL_DIR` | `~/.cache/lao_tutor/models` | Location where downloaded ML model weights are stored. |
| `LAO_TUTOR_LOG_DIR` | `logs` | Directory where rotating log files (`app.log`) are written. |
| `LAO_TUTOR_WHISPER_MODEL_SIZE` | `small` | Whisper checkpoint family to load for ASR (`tiny`, `base`, `small`, `medium`, `large`). |
| `LAO_TUTOR_SQLITE_PATH` | `data/tutor.db` | Path to the SQLite database backing the SRS store. |
| `LAO_TUTOR_ENABLE_PITCH_FEEDBACK` | `false` | Enable pitch contour analysis (requires `librosa`). |
| `LAO_TUTOR_SAMPLE_RATE` | `16000` | Target sample rate (Hz) for audio capture and synthesis. |
| `LAO_TUTOR_VAD_THRESHOLD` | `0.35` | VAD probability cutoff shared across WebRTC/Silero/energy fallbacks. |
| `LAO_TUTOR_LLM_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face identifier for the conversational tutor model. |
| `LAO_TUTOR_LLM_DEVICE` | `cpu` | Device passed to the transformers pipeline (`cpu`, `cuda`, `mps`). |
| `LAO_TUTOR_LLM_MAX_NEW_TOKENS` | `256` | Maximum number of tokens generated per conversational turn. |
| `LAO_TUTOR_LLM_TEMPERATURE` | `0.7` | Sampling temperature for conversational responses. |
| `LAO_TUTOR_TTS_MODEL_NAME` | `facebook/mms-tts-lao` | Hugging Face identifier for Lao TTS voice synthesis. |
| `LAO_TUTOR_TTS_DEVICE` | `cpu` | Device used for MMS TTS inference (`cpu`, `cuda`, `mps`). |
| `LAO_TUTOR_TRANSLATION_MODEL_NAME` | `facebook/nllb-200-distilled-600M` | Translation checkpoint for Lao → English glosses. |
| `LAO_TUTOR_TRANSLATION_SOURCE_LANG` | `lo_Laoo` | Source language code passed to the translation model. |
| `LAO_TUTOR_TRANSLATION_TARGET_LANG` | `eng_Latn` | Target language code passed to the translation model. |
| `LAO_TUTOR_TRANSLATION_DEVICE` | `cpu` | Device identifier for translation inference (`cpu`, `cuda`, `mps`). |

Set any of these variables before launching Uvicorn (or define them in `.env`) to customise the tutor’s behaviour.

## Logging

- The backend configures structured logging on import and writes rotating files to `logs/app.log` (ignored by git).
- Each module obtains a named logger via `backend.app.logging_utils.get_logger` and emits contextual metadata (e.g., ASR results, VAD backend, conversation turn stats).
- Update `LAO_TUTOR_LOG_DIR` (or call `configure_logging` manually) if you want to stream logs to a custom directory for deployment.

## Tests

```bash
pytest
```

## Roadmap

This commit seeds the backend scaffolding. Next steps include hooking up real curriculum content, refining correction heuristics, adding pronunciation scoring, and building the WebRTC/Next.js frontend loop.
