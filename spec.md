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
| F8a | A round microphone button left of **Record** opens a menu upwards where the user ticks the audio inputs: every microphone the browser lists, plus **System audio** (video, music, calls; the browser's share dialog asks for a tab or screen with "Share audio" ticked, every time recording starts). The default microphone is ticked by default and the choice is remembered. Several ticked inputs are mixed into one signal. The menu is locked while recording. Echo cancellation and noise suppression apply to microphones only. |
| F9 | The server accepts one active recording at a time. A second connection is refused with a "busy" message. |
| F10 | The design follows the approved mock-up: dark theme, green accent, red while recording, large pill button, bar visualizer, transcript panel. |
| F11 | The top bar has a **Model** menu. It lists the known models; models whose folder is missing in `models/` are shown greyed out as "not downloaded". Choosing a model loads it on the server (the previous one is freed from GPU memory). Switching is refused while a recording runs. |
| F12 | The top bar has a **Language** menu. It is enabled only when the chosen model has a language setting (the multilingual live model: Auto-detect, English, Polish, Ukrainian, Russian and others). Otherwise it is greyed out and states why ("English only", "Detected automatically"). The chosen language is sent with each recording and cannot change during a recording. |
| F13 | **Live (streaming) models** show text while the user speaks. **Offline models** (for example Parakeet TDT v3) buffer the recording and transcribe it when **Stop** is pressed. The page tells the user which mode applies, shows "Transcribing…" after Stop, and limits offline recordings to 5 minutes (at the limit the recording stops by itself and is transcribed). |
| F14 | The top bar has a **Speakers** toggle (Single / Multiple), to the left of the Model menu. It decides which models the Model menu lists: *Single* lists every model; *Multiple* lists only models that can give speaker information (Parakeet TDT with a diarization model, and VibeVoice-ASR, which finds the speakers itself). Flipping the toggle loads a suitable downloaded model, and a diarization model if one is needed, when the current ones do not fit. |
| F15 | In *Multiple* mode a **Diarization model** menu appears. It lists the diarization models (Nemotron-3-Diarization, pyannote community-1), greyed out when not downloaded or when a needed Python package is missing. It is disabled and reads "Built into the model" for VibeVoice. |
| F16 | In *Multiple* mode the transcript is a **dialogue**: one row per turn, each with a coloured marker on the left, the label "Speaker N" and the start time. Consecutive text of the same speaker is one turn. Colours repeat after 8 speakers; the label is always shown, so colour is not the only cue. Rows from different recordings are separated by "New recording". Copy gives one line "Speaker N: text" per turn. |
| F17 | The badges row (`id="badges"`) shows the real setup: GPU, speaker mode, model, diarization model (Multiple only), live or on-Stop mode with latency or length limit, and language. |
| F19 | The models are not part of the repository. Every model or diarization model that is missing, or whose Python package is missing, is shown greyed out in its menu with the reason. `scripts/check_setup.py` reports the same items in the terminal with the command that fixes each, and `scripts/download_model.py --all` fetches all models at once. |
| F18 | The page layout fills the window: the transcript takes all free height and scrolls inside; the recorder (status, timer, visualizer, button) is a compact row at the bottom that is only as big as its content. |

## 4. Non-functional requirements

| ID | Requirement |
|----|-------------|
| N1 | **Offline:** after setup (Python packages and model files downloaded once), the app makes no network requests. The page loads no fonts, scripts or images from the internet. The server sets `HF_HUB_OFFLINE=1` and loads every model from a local folder. |
| N2 | **GPU:** the model runs on CUDA in half precision (bfloat16). If CUDA is unavailable the app falls back to the CPU, prints a loud warning, and the header shows "CPU". |
| N3 | **Latency:** the delay between speech and text is set by the chosen lookahead (see 6). The default is 1120 ms of model latency (chosen for accuracy; 160 ms is available for a snappier feel), plus audio chunking and network on localhost. The server must keep up with real time (real-time factor below 1) on the RTX 5060. This is verified by measurement, not assumed. |
| N4 | **Privacy:** audio never leaves the machine. The server binds to `127.0.0.1` by default. |
| N5 | **Memory:** with live models the server keeps only the audio it still needs. Offline models must keep the whole recording, so it is limited to 5 minutes. Only one model is loaded at a time. |
| N6 | **Startup and loading:** the model is read from disk into GPU memory at each server start, so only the model used last time (and its diarization model) is loaded, taken from `settings.json` unless `--model` is given. Every model is warmed up with a short silent clip after loading, so the first recording is not slow. The active model is never loaded again, and up to two recently used models stay in memory, so switching back to them is instant. The 4-bit VibeVoice model is never kept: loading it frees the others. If a load fails, the previous model is brought back. |
| N8 | **Quiet log:** the terminal shows one line when a model starts loading and one when it is ready (with the time). Known harmless warnings (triton, default generation length, RNN weight layout, "Redirects") are hidden. Other warnings and errors are shown. |
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
  | `offline` | `parakeet-tdt-0.6b-v3` | `AutoModelForTDT` | The whole recording is collected, then transcribed with one `generate()` call after Stop. In *Multiple* mode the call also returns token timestamps; they are turned into word times, and each word gets the speaker of the diarization segment that overlaps it most. |
  | `vibevoice` | `VibeVoice-ASR-HF-NF4` (4-bit community build) | `VibeVoiceAsrForConditionalGeneration` | The whole recording is collected and written to a temporary WAV file; the model returns speaker, start, end and text itself. Needs `bitsandbytes`. About 7 to 8 GB of GPU memory. |

  Diarization models (`KNOWN_DIARIZERS` in `app/engine.py`) take the whole recording and return speaker segments: **Nemotron-3-Diarization** through `AutoModelForAudioFrameClassification` (up to 8 speakers, 10 ms frames, threshold 0.5), and **pyannote community-1** through `pyannote.audio` (installed separately, loaded from a local folder). One diarization model is loaded at a time, and it is freed when VibeVoice is loaded.

- **Adding a model:** download it into `models/<name>` and add an entry to `KNOWN_MODELS` in `app/engine.py`. Models in `models/` that are not listed there still appear in the menu, treated as live models without a language setting; this works only for models with the same architecture as the Nemotron streaming models.
- **Runtime:** Hugging Face `transformers` (version 5.13 or newer). This route needs no NeMo install, which keeps Windows setup simple. The streaming API is marked experimental upstream, so `requirements.txt` sets a minimum version (5.13). Pin the exact version that works once the app has been tested on the target machine.
- **Audio path:** the browser captures at its native rate (usually 48 kHz), an AudioWorklet resamples to 16 kHz mono and sends 16-bit PCM frames of about 80 ms. The server converts them to float32 and cuts them into model chunks.
- **Threads on the server:** the asyncio loop handles the socket. One worker thread per session runs `generate()`. A second thread reads the streamer and hands text back to the loop. A lock allows one session at a time.

## 6. Configuration

| Setting | Default | Meaning |
|---------|---------|---------|
| `--latency` | `1120` | Streaming latency in ms for live models. Allowed: 80, 160, 320, 560, 1120. Lower is snappier, higher is more accurate. The default favors accuracy. If a model lacks the mode, the nearest one is used. Ignored by offline models. |
| `--model` | last used, else `nemotron-speech-streaming-en-0.6b` | Model (folder name inside `--models-dir`) loaded at start. |
| `--models-dir` | `models` | Folder with one sub-folder per downloaded model (speech and diarization models). |
| `--max-minutes` | `5` | Longest recording for offline models. |
| `--host` / `--port` | `127.0.0.1` / `8000` | Where the server listens. |
| `--device` | `auto` | `auto` (CUDA if present), `cuda`, or `cpu`. |

## 7. WebSocket protocol (`/ws/transcribe`)

Client to server:
1. Text `{"type":"start","mode":"multi","language":"pl-PL"}`. `mode` is `"single"` or `"multi"`. `language` is `null` when the model has no language setting. The server answers with an error if the loaded model cannot do `mode` or a diarization model is missing.
2. Binary messages: raw 16 kHz mono signed 16-bit little-endian PCM.
3. Text `{"type":"stop"}`.

Server to client:
- `{"type":"ready","streaming":true}` after `start`, when the engine accepted the session. `streaming` is `false` for offline models.
- `{"type":"turns","turns":[{"speaker":0,"start":0.0,"end":3.6,"text":"..."}]}` in *Multiple* mode, once after Stop. Speakers are numbered from 0 in order of first appearance in this recording.
- `{"type":"limit"}` when an offline model's recording reached the maximum length. The client then stops the recording; later audio is ignored.
- `{"type":"text","delta":"..."}` for each piece of newly recognized text.
- `{"type":"done"}` after `stop`, once the last text was sent.
- `{"type":"error","message":"..."}` on failure, including "busy".

`GET /api/info` returns the device (`device`, `cuda`), the loaded model id (`model`), whether it streams (`streaming`), `latency_ms` and `max_seconds` (null when not applicable), the list of `models` (`id`, `label`, `downloaded`, `streaming`), and for the loaded model its `languages` (`code`, `label`), `language_note` and `default_language`.

`/api/info` also returns `modes` and `diarization` ("external", "builtin" or null) of the loaded model, `model_label`, the loaded `diarizer` and the list `diarizers` (`id`, `label`, `downloaded`, `available`, `needs`). Models carry the same `available` and `needs` fields and a `modes` list.

`POST /api/model` with `{"id":"<model id>"}` loads that model and returns the same data as `/api/info`. It answers 409 while a recording is running and 404 if the model is not downloaded. `POST /api/diarizer` with `{"id":"<diarization model id>"}` does the same for a diarization model; `{"id":null}` unloads it.

## 8. Requirements on the machine

| Item | Requirement |
|------|-------------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5060 (Blackwell) or any CUDA-capable GPU. About 4 GB free VRAM for the Nemotron and Parakeet models; about 7 GB for the 4-bit VibeVoice model (measured: 6.95 GB in use on the 8 GB RTX 5060). |
| Driver | Current NVIDIA driver supporting CUDA 12.8 or newer |
| Python | 3.12 (64-bit) |
| PyTorch | A CUDA 12.8 (`cu128`) or newer build. Older builds do not support Blackwell GPUs. |
| Disk | About 17 GB: PyTorch with CUDA libraries (about 3 GB) plus all five models (about 14 GB; `download_model.py --all`). Fewer models need less. |
| Python packages | `requirements.txt` (includes `bitsandbytes` for VibeVoice and `transformers` 5.20 or newer). Optional: `requirements-multispeaker.txt` (`pyannote.audio`). |
| Other | A working microphone |

The full CUDA Toolkit is not needed. The PyTorch wheel bundles the CUDA runtime.

## 9. Project layout

```
spec.md
README.md
requirements.txt
requirements-multispeaker.txt  optional package for the pyannote diarization model (pyannote.audio)
run.py                      start the server
run.bat                     start the server with the virtual environment
investigation/              research on speaker-aware models
scripts/download_model.py   one-time model download (needs internet; --all for every model)
scripts/check_setup.py      reports missing models and packages with the command that fixes each
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
