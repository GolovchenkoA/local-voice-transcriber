---
name: project-map
description: Map of the Local Voice Transcriber code (where each feature lives, function names, WebSocket and HTTP protocol). Use before exploring files for any change or question about the app's code, UI, server or models.
---

# Project map

Use this map to open only the code you need. Find a function with Grep (`def name` or `function name`), then Read with offset and limit. Don't read whole files.

## Where to change what

| Task | File and place |
|------|----------------|
| Button, layout, text on the page | `app/static/index.html`, `app/static/style.css` |
| UI behaviour, menus, transcript display | `app/static/app.js` (see below) |
| Microphone capture, resampling | `app.js` `beginCapture()`, `app/static/pcm-worklet.js` |
| New HTTP endpoint or WebSocket message | `app/server.py` `create_app()`, then `app.js`, then spec.md section 7 |
| Model list, labels, languages, modes | `app/engine.py` `KNOWN_MODELS`, `_LANGS` / `MULTILINGUAL` |
| Diarization models | `app/engine.py` `KNOWN_DIARIZERS`, `Nemotron3Diarizer`, `PyannoteDiarizer`, `DIARIZER_BACKENDS` |
| Live (streaming) transcription | `Engine._transcribe_streaming`, `Engine._set_latency` |
| Offline transcription after Stop | `Engine._transcribe_offline`, `_tdt_generate`, `_tdt_text`, `_tdt_words`, `_vibevoice` |
| Speaker turns | `words_from_timestamps`, `assign_speakers`, `merge_turns` (pure functions, easy to test) |
| Start-up arguments | `run.py` (mirrors spec.md section 6 and README "Parameters") |

## app/engine.py (~630 lines)
Constants: `SAMPLE_RATE = 16000`, `ALLOWED_LATENCIES_MS`, `DEFAULT_MAX_SECONDS`, `DEFAULT_MODEL`, `DEFAULT_LANGUAGE`.
Model kinds: `streaming` (AutoModelForRNNT, Nemotron), `offline` (AutoModelForTDT, Parakeet TDT), `vibevoice` (4-bit, speakers built in, needs bitsandbytes).
`Engine` methods: `available_models`, `available_diarizers`, `load`, `load_diarizer`, `unload_diarizer`, `gpu_name`, `info`, `resolve_language`, `check_mode`, `warm_up`, `transcribe` (dispatches to streaming or offline), `_collect`.
One model and at most one diarizer are loaded at a time. VibeVoice frees the diarizer.

## app/server.py (~170 lines)
All routes are inside `create_app(engine)`: `GET /api/info`, `POST /api/model`, `POST /api/diarizer` (`{"id": ...}`, 409 while recording, 404 if not downloaded), `GET /` (index.html), WebSocket `/ws/transcribe`. `run_exclusive` serialises model changes. One worker thread per session runs the engine; a lock allows one session at a time ("busy").

## WebSocket protocol (`/ws/transcribe`)
Client: `{"type":"start","mode":"single"|"multi","language":code|null}`, then binary 16 kHz mono int16 PCM (~80 ms frames), then `{"type":"stop"}`.
Server: `ready` (`streaming` true/false), `text` (`delta`), `turns` (list of `speaker`, `start`, `end`, `text`; multi mode, after Stop), `limit` (offline length cap), `done`, `error` (`message`).

## app/static/app.js (~520 lines, one closure)
State and UI: `updateControls`, `drawBars`, `renderDialogue`, `renderTranscript`, `setStatus`, `setRecordingUI`, `startTimer` / `stopTimer`.
Session: `start` (opens the WebSocket, `ws.onmessage` handles server messages), `beginCapture`, `stop`, `endSession`, `fail`, `cleanup`.
Menus: `renderMenus`, `renderBadges`, `applyInfo`, `post`, `loadModel`, `loadDiarizer`, `ensureSetup`. Event listeners are at the end of the file; `/api/info` is fetched on load.

## Docs (grep, don't read whole)
`spec.md`: 3 functional requirements, 5 architecture, 6 configuration, 7 protocol, 10 acceptance criteria. `README.md`: setup, models, parameters, "What has been tested", troubleshooting, project layout. Model research: `speech-to-text-models-overview.md`, `investigation/speaker-aware-models.md` (ask the doc-reader subagent).
