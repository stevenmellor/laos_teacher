# Lao Language Teacher Voice Tutor — Local-First Architecture Plan

## 1. Product Vision & Success Criteria
- Deliver a voice-first tutor that carries learners from "alphabet & tones" to advanced fluency across months of study.
- Operate fully offline on a capable laptop (CPU/GPU) while remaining cloud-portable.
- Provide corrective feedback on pronunciation, tone, vocabulary, and grammar with transparent explanations.
- Persist long-term progress and schedule spaced repetition so the tutor "remembers" the learner.

## 2. End-to-End Pipeline (Conceptual)
```
Mic → VAD gate → ASR → Lao text tools → Tutor policy → Feedback planner → TTS → Speakers
                         ↓                    ↓
                 Pronunciation analyser   Progress store (SRS + logs)
```
Key decisions:
- **Silero VAD** (ONNX) or **WebRTC VAD** to segment user speech with millisecond latency.
- **Whisper** models (via `faster-whisper` GPU/CTranslate2 or `whisper.cpp` CPU) for Lao (`lo`) transcription.
- **Lao text tooling** (LaoNLP + Chamkho) for segmentation, part-of-speech, romanisation, tone hints.
- **Teacher policy**: deterministic state machine (LangGraph/Pydantic) orchestrating lessons, corrections, and review scheduling.
- **Feedback planner**: formats corrections, tone charts, and exercise prompts grounded in curated Lao content.
- **Meta MMS-TTS Lao** (Hugging Face `facebook/mms-tts-lo`) for local speech output; cache wavs for reuse.
- **Progress store**: SQLite + FSRS/SM-2 scheduler to drive spaced repetition and analytics.

## 3. Core Services & Stack Choices
| Layer | Recommended Stack | Offline viability | Notes |
| --- | --- | --- | --- |
| UI | Next.js 14 + React 18 + Tailwind (PWA) | Yes (service worker) | Web Audio capture, waveform visualisations, tone contour display. |
| Desktop packaging | Tauri (Rust) | Yes | Wraps web UI for native permissions, lightweight vs Electron. |
| Transport | FastAPI WebSocket server | Yes | Streams PCM chunks and TTS audio; uvicorn workers. |
| STT | `faster-whisper` w/ CTranslate2 GPU or `whisper.cpp` CPU | Yes | Use medium/small models; quantise (int8) for laptops. |
| VAD | WebRTC VAD (`webrtcvad-wheels`) + Silero ONNX fallback | Yes | WebRTC for deterministic gating, Silero for smarter detection. |
| TTS | `transformers` + MMS-TTS Lao; optional VITS fine-tune later | Yes | Provide normal and slow "teacher mode" voices. |
| Lao NLP | `laonlp`, `chamkho`, custom BGN/PCGN romaniser | Yes | Preload dictionaries; ensure Unicode normalization (NFC). |
| Tutor logic | LangGraph or custom FSM + Pydantic lesson schemas | Yes | Deterministic transitions; LLM optional for English explanations. |
| Retrieval | SQLite FTS5 + `faiss-cpu` or `sentence-transformers` LaBSE embeddings | Yes | Power contextual prompts and error lookup without external calls. |
| Pronunciation | `librosa` (YIN pitch), Montreal Forced Aligner (custom Lao lexicon) | Yes | Provide tone contour overlays & phoneme timing. |
| Persistence | SQLite + SQLModel (FastAPI integration) | Yes | Tables: learners, utterances, cards, reviews, sessions. |
| Packaging | Poetry for deps, Dockerfile (CUDA optional) | Partial | Docker enables later cloud move; offline `poetry export`. |

## 4. Data & Content Foundations
- **Curriculum backbone**: FSI Lao Basic Course dialogues/drills, NIU SEAsite lessons for alphabet & tone classes, curated beginner phrasebank.
- **Dictionary**: SEAlang Lao lexical database (word senses, audio if licensed) + custom JSON for quick lookup.
- **Audio pairs**: align each phrase with high-quality native audio (FSI/recorded). Store `wav` + metadata for cloning.
- **Spaced repetition cards**: YAML → SQLite importer with fields (`lo_text`, `romanised`, `ipa`, `audio_path`, `tone`, `tags`).
- **Licensing**: verify MMS-TTS & dataset terms; maintain manifest for attributions and redistribution constraints.

## 5. Tutor Policy & Pedagogy
1. **State Machine Design**
   - States: `onboarding`, `alphabet`, `tones`, `numbers`, `phrase_practice`, `review`, `free_conversation`.
   - Transitions triggered by learner proficiency metrics (accuracy %, pronunciation score, completion count).
2. **Error Handling Workflow**
   - Compare ASR transcript against target using Lao segmentation.
   - Identify tone or consonant class mismatches; generate minimal pairs from phrasebank.
   - Provide correction: highlight syllable, explain tone rule (consonant class + tone mark + syllable type), offer retry.
3. **Feedback Modalities**
   - Visual tone contour (target vs learner) using YIN F0 estimates.
   - Romanisation with BGN/PCGN and IPA to support non-native readers.
   - Immediate vs delayed correction toggle for advanced learners.
4. **Memory & Review**
   - After each session, log utterance + evaluation scores.
   - Run FSRS scheduler nightly to queue review prompts; surface in next session.

## 6. Real-Time Loop (Implementation Notes)
1. Frontend streams 16 kHz PCM via WebSocket; VAD client-side to reduce latency.
2. Backend buffers frames, applies Silero VAD for double-check, then feeds into Whisper streaming.
3. Receive partial transcripts + timestamps → push to UI for live captions.
4. On end-of-utterance, run Lao segmentation, romanisation, and policy evaluation.
5. Fetch exemplar audio & explanations, produce corrections.
6. Generate TTS (normal + slowed) and cache file path; stream to client.
7. Update SQLite: utterance record, SRS review item, pronunciation metrics (pitch RMSE, duration).

## 7. Roadmap (12+ Weeks)
### Phase 0 — Environment Setup (Week 0)
- Create mono-repo with `frontend/`, `backend/`, `data/` folders; configure Poetry & PNPM.
- Write smoke tests for audio streaming, database migrations, and TTS playback.

### Phase 1 — Voice MVP (Weeks 1–2)
- Implement FastAPI WebSocket for mic stream; integrate Silero VAD + Whisper small.
- Basic Next.js UI: start/stop recording, transcript display, audio playback.
- Hard-code Day 1 lesson (alphabet & greetings) in YAML; simple response templates.

### Phase 2 — Curriculum & Progress (Weeks 3–5)
- Model lessons/cards in SQLite; build admin script to import phrasebank.
- Implement SM-2/FSRS review queues; show daily goals in UI.
- Add romanisation/tone hints in feedback panel.

### Phase 3 — Pronunciation Insights (Weeks 6–8)
- Compute YIN pitch contours; render overlay graph in UI.
- Add mispronunciation detection: tone classification heuristic + tolerance.
- Introduce slowed TTS + minimal pair drills triggered by recurring tone errors.

### Phase 4 — Adaptive Tutor Brain (Weeks 9–12)
- Build deterministic policy graph with context-aware prompts (LangGraph).
- Add optional LLM (Qwen/Q4 quantised) for English explanations, with retrieval grounding using LaBSE embeddings.
- Persist per-learner knowledge graph (what vocabulary/grammar is mastered, needs work).

### Phase 5 — Extended Experiences (Months 4+)
- Role-play scenarios (travel, business) with dynamic branching.
- Introduce assessment mode (listening comprehension, dictation, composition).
- Sync SQLite to cloud PostgreSQL via replication for multi-device use.

## 8. Testing & Monitoring Strategy
- **Unit tests**: transcription accuracy evaluation (expected vs actual), tone rule engine, FSRS scheduling.
- **Integration tests**: end-to-end audio roundtrip (PCM → STT → response → TTS) using fixtures.
- **UX metrics**: track latency (<500 ms partial, <2 s response), learner retention, tone accuracy improvements.
- **Observability**: local logging with structlog, optional OpenTelemetry exporter when in cloud.

## 9. Future Enhancements
- Browser/WebRTC upgrade via LiveKit for ultra-low-latency streaming and group sessions.
- Train custom Lao acoustic models (fine-tune Whisper, VITS) once enough proprietary data collected.
- Montreal Forced Aligner-based phoneme scoring for granular pronunciation grading.
- Companion mobile app (React Native) that syncs spaced repetition deck and plays cached audio offline.

## 10. Risks & Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Limited Lao datasets | Medium | Partner with Lao educators, record bespoke corpus, augment with TTS for drilling. |
| Whisper latency on CPU | High | Quantise models, allow GPU (CUDA) path, offer offline text drills when hardware weak. |
| Tone evaluation false positives | Medium | Blend rule-based + statistical checks; allow user override and manual marking. |
| Licensing constraints | High | Maintain attribution manifest, restrict redistribution until permissions secured. |
| User fatigue | Medium | Adaptive pacing, varied exercise types, progress visualisations, weekly recap emails. |

