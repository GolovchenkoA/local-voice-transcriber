# Local Voice Transcriber: Specification

## 1. Purpose

A small web app that turns speech into text entirely on the user's own computer. It runs speech-recognition models on the GPU with CUDA, and the user chooses the model and the language in the web page. It supports several models: live (streaming) models that show text while the user speaks, and offline models that transcribe the whole recording when the user presses Stop. The models shipped with the project are NVIDIA's Nemotron ASR Streaming (English), Nemotron 3.5 ASR Streaming (multilingual) and Parakeet TDT v3 (25 languages); more can be added (see section 5). After the one-time setup it needs no internet connection.

## 2. Users and scenario

One user on one Windows 11 machine with an NVIDIA GeForce RTX 5060 (Laptop, 8 GB VRAM). The user opens the app in a browser on the same machine, chooses a model and a language, presses **Record**, and talks. With a live model the text appears while speaking; with an offline model it appears after **Stop**. Pressing **Stop** ends the session.

## 3. Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | The page shows one primary button. It reads **Record** when idle. When pressed it starts capturing the microphone and changes to **Stop**. Pressing **Stop** ends capture and returns the button to **Record**. |
| F2 | While recording, a voice visualizer (a row of bars) reacts to the real microphone level. When idle it shows a quiet idle animation. |
| F3 | A transcript window shows recognized text. With a live model it appears as it is produced, with no need to wait for Stop, and the newest word is highlighted until it is final. With an offline model the text appears after Stop (see F13). |
| F4 | Text from several recordings accumulates in the transcript window until the user presses **Clear**. |
| F5 | The user can copy the transcript to the clipboard with **Copy**. |
| F6 | The page shows a recording timer, a word count, and a status line (Ready, Connecting, Listening, Stopped, error messages). |
| F7 | The page shows what the engine is really using: GPU name (or CPU), the loaded model, and the mode (live with its latency, or text on Stop). This data comes from the server, not from fixed text. |
| F8 | If the microphone is blocked or missing, or the server is not reachable, the page shows a clear message and returns to idle. |
| F9 | The server accepts one active recording at a time. A second connection is refused with a "busy" message. |
| F10 | The design follows the approved mock-up: dark theme, green accent, red while recording, large pill button, bar visualizer, transcript panel. |
| F11 | The top bar has a **Model** menu. It lists the known models; models whose folder is missing in `models/` are shown greyed out as "not downloaded". Choosing a model loads it on the server (the previous one is freed from GPU memory). Switching is refused while a recording runs. |
| F12 | The top bar has a **Language** menu. It is enabled only when the chosen model has a language setting (the multilingual live model: Auto-detect, English, Polish, Ukrainian, Russian and others). Otherwise it is greyed out and states why ("English only", "Detected automatically"). The chosen language is sent with each recording and cannot change during a recording. |
| F13 | **Live (streaming) models** show text while the user speaks. **Offline models** (for example Parakeet TDT v3) buffer the recording and transcribe it when **Stop** is pressed. The page tells the user which mode applies, shows "Transcribing…" after Stop, and limits offline recordings to 5 minutes (at the limit the recording stops by itself and is transcribed). |

## 4. Non-functional requirements

| ID | Requirement |
|----|-------------|
| N1 | **Offline:** after setup (Python packages and model files downloaded once), the app makes no network requests. The page loads no fonts, scripts or images from the internet. The server sets `HF_HUB_OFFLINE=1` and loads every model from a local folder. |
| N2 | **GPU:** the model runs on CUDA in half precision (bfloat16). If CUDA is unavailable the app falls back to the CPU, prints a loud warning, and the header shows "CPU". |
| N3 | **Latency:** the delay between speech and text is set by the chosen lookahead (see 6). The default is 1120 ms of model latency (chosen for accuracy; 160 ms is available for a snappier feel), plus audio chunking and network on localhost. The server must keep up with real time (real-time factor below 1) on the RTX 5060. This is verified by measurement, not assumed. |
| N4 | **Privacy:** audio never leaves the machine. The server binds to `127.0.0.1` by default. |
| N5 | **Memory:** with live models the server keeps only the audio it still needs. Offline models must keep the whole recording, so it is limited to 5 minutes. Only one model is loaded at a time. |
| N6 | **Startup:** the chosen start model is loaded at server start, and every model is warmed up with a short silent clip after loading, so the first recording is not slow. |
| N7 | **Browsers:** current Chrome, Edge or Firefox on the same machine (`localhost` counts as a secure context, which microphone access requires). |

## 5. Architecture

```
Browser (localhost:8000)                        Python server (FastAPI + uvicorn)
┌────────────────────────────┐   WebSocket    ┌─────────────────────────────────────┐
│ Mic → AudioContext         │  binary PCM    │ /ws/transcribe                      │
│   ├─ AnalyserNode → bars   │ ─────────────▶ │   session queue ─▶ chunker          │
│   └─ AudioWorklet          │  16 kHz int16  │       ─▶ feature extractor (CPU)    │
│       (resample to 16 kHz) │                │       ─▶ speech model (CUDA),       │
│ Model menu, Language menu  │ ◀───────────── │          live or offline            │
│ Transcript panel           │   JSON text    │ /api/info, /api/model               │
└────────────────────────────┘                └─────────────────────────────────────┘
```

- **Models:** the server holds one model at a time and can switch between the models in `models/`. A model entry in `app/engine.py` has a name, a Hugging Face repository, a kind and a language list.

  | Kind | Example | Class | How it transcribes |
  |------|---------|-------|--------------------|
  | `streaming` | `nemotron-speech-streaming-en-0.6b`, `nemotron-3.5-asr-streaming-0.6b` | `AutoModelForRNNT` | Cache-aware FastConformer RNN-T. It processes only each new audio chunk and reuses cached context, so it does not re-decode overlapping windows. `generate()` runs with a generator that feeds feature chunks as audio arrives. |
  | `offline` | `parakeet-tdt-0.6b-v3` | `AutoModelForTDT` | The whole recording is collected, then transcribed with one `generate()` call after Stop. |

- **Adding a model:** download it into `models/<name>` and add an entry to `KNOWN_MODELS` in `app/engine.py`. Models in `models/` that are not listed there still appear in the menu, treated as live models without a language setting; this works only for models with the same architecture as the Nemotron streaming models.
- **Runtime:** Hugging Face `transformers` (version 5.13 or newer). This route needs no NeMo install, which keeps Windows setup simple. The streaming API is marked experimental upstream, so `requirements.txt` sets a minimum version (5.13). Pin the exact version that works once the app has been tested on the target machine.
- **Audio path:** the browser captures at its native rate (usually 48 kHz), an AudioWorklet resamples to 16 kHz mono and sends 16-bit PCM frames of about 80 ms. The server converts them to float32 and cuts them into model chunks.
- **Threads on the server:** the asyncio loop handles the socket. One worker thread per session runs `generate()`. A second thread reads the streamer and hands text back to the loop. A lock allows one session at a time.

## 6. Configuration

| Setting | Default | Meaning |
|---------|---------|---------|
| `--latency` | `1120` | Streaming latency in ms for live models. Allowed: 80, 160, 320, 560, 1120. Lower is snappier, higher is more accurate. The default favors accuracy. If a model lacks the mode, the nearest one is used. Ignored by offline models. |
| `--model` | `nemotron-speech-streaming-en-0.6b` | Model (folder name inside `--models-dir`) loaded at start. |
| `--models-dir` | `models` | Folder with one sub-folder per downloaded model. |
| `--host` / `--port` | `127.0.0.1` / `8000` | Where the server listens. |
| `--device` | `auto` | `auto` (CUDA if present), `cuda`, or `cpu`. |

## 7. WebSocket protocol (`/ws/transcribe`)

Client to server:
1. Text `{"type":"start","language":"pl-PL"}`. `language` is `null` when the model has no language setting.
2. Binary messages: raw 16 kHz mono signed 16-bit little-endian PCM.
3. Text `{"type":"stop"}`.

Server to client:
- `{"type":"ready","streaming":true}` after `start`, when the engine accepted the session. `streaming` is `false` for offline models.
- `{"type":"limit"}` when an offline model's recording reached the maximum length. The client then stops the recording; later audio is ignored.
- `{"type":"text","delta":"..."}` for each piece of newly recognized text.
- `{"type":"done"}` after `stop`, once the last text was sent.
- `{"type":"error","message":"..."}` on failure, including "busy".

`GET /api/info` returns the device (`device`, `cuda`), the loaded model id (`model`), whether it streams (`streaming`), `latency_ms` and `max_seconds` (null when not applicable), the list of `models` (`id`, `label`, `downloaded`, `streaming`), and for the loaded model its `languages` (`code`, `label`), `language_note` and `default_language`.

`POST /api/model` with `{"id":"<model id>"}` loads that model and returns the same data as `/api/info`. It answers 409 while a recording is running and 404 if the model is not downloaded.

## 8. Requirements on the machine

| Item | Requirement |
|------|-------------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5060 (Blackwell) or any CUDA-capable GPU with about 4 GB free VRAM |
| Driver | Current NVIDIA driver supporting CUDA 12.8 or newer |
| Python | 3.12 (64-bit) |
| PyTorch | A CUDA 12.8 (`cu128`) or newer build. Older builds do not support Blackwell GPUs. |
| Disk | About 6 GB (PyTorch with CUDA libraries and the model) |
| Other | A working microphone |

The full CUDA Toolkit is not needed. The PyTorch wheel bundles the CUDA runtime.

## 9. Project layout

```
spec.md
README.md
requirements.txt
run.py                      start the server
scripts/download_model.py   one-time model download (needs internet)
scripts/transcribe_file.py  test the engine on a WAV file, no browser
app/engine.py               model loading and switching, streaming and offline transcription
app/server.py               FastAPI app, model menu endpoints and WebSocket endpoint
app/static/                 index.html, style.css, app.js, pcm-worklet.js
```

## 10. Acceptance criteria

1. With the network disabled, `python run.py` starts and the page loads fully styled.
2. The header shows the RTX 5060 and "CUDA".
3. Pressing Record asks for microphone permission; the button becomes Stop and the bars follow the voice.
4. Speaking a sentence makes text appear within about two seconds of each word (default latency), or about half a second with `--latency 160`.
5. Pressing Stop flushes the last words, then the button becomes Record again.
6. `nvidia-smi` shows GPU memory use by the Python process during a session.
7. A second browser tab pressing Record while the first records shows a "busy" message.

## 11. Out of scope

Mixing several languages inside one recording, translation between languages, speaker separation, saving audio, user accounts, remote or multi-user access, and packaging as an installer.

## 12. Risks

- The `transformers` streaming API is experimental and may change. Mitigation: pin the working version after the first successful test.
- Blackwell GPUs need a recent PyTorch build. Mitigation: documented `cu128` install.
- Model license: NVIDIA Open Model License. The user should read it before any use beyond personal.
