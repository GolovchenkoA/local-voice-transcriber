# Local Voice Transcriber

Offline speech-to-text web app for Windows: browser microphone → WebSocket → FastAPI → Hugging Face `transformers` speech models on CUDA (RTX 5060, 8 GB). Nothing may reach the network at run time.

## Layout
- `app/engine.py`: models, loading and switching, streaming and offline transcription, diarization (`KNOWN_MODELS`, `KNOWN_DIARIZERS`, class `Engine`)
- `app/server.py`: FastAPI, `/api/info`, `/api/model`, `/api/diarizer`, WebSocket `/ws/transcribe`
- `app/static/`: the page (`index.html`, `style.css`, `app.js` = UI and WebSocket client, `pcm-worklet.js` = resample to 16 kHz)
- `run.py` / `run.bat`: start the server. `scripts/`: model download, WAV test
- `models/`: downloaded weights (several GB). Never list it recursively or read weight files.

## Working rules (keep token use low)
- Before exploring for architecture, protocol or "where is X" questions, load the `project-map` skill. To add a speech or diarization model, load `add-model`.
- Don't read `README.md`, `spec.md`, `speech-to-text-models-overview.md`, `investigation/` or model cards in full. Grep for the heading you need and read that part, or ask the `doc-reader` subagent.
- Read only the code you change: grep for the function name, then Read with offset and limit. `engine.py` (~630 lines) and `app.js` (~520 lines) are the big files.
- Verify with the `test-runner` subagent, not in the main conversation. It returns a short summary.
- Never start `run.py` or `run.bat` in the foreground: the server never exits. Run it in the background and stop it afterwards.
- Don't load a model to check a UI-only or server-only change. Syntax and API checks are enough.
- After a behaviour change, update only the matching section of `spec.md` and `README.md`.
- Keep answers short: report what changed and the check result, not the code you wrote.

## Environment
- Windows, Python 3.12 virtual environment: call `.venv/Scripts/python.exe` directly (no activation).
- No test suite yet. Quick checks: `.venv/Scripts/python.exe -m py_compile app/engine.py app/server.py run.py scripts/*.py` and `node --check app/static/app.js`.
- Engine test without a browser (loads a model, slow): `.venv/Scripts/python.exe scripts/transcribe_file.py <16 kHz mono 16-bit WAV>`.

# Compact instructions
Keep: files changed and why, the current task, open problems, the last failing check. Drop: file contents already read, full logs, exploration that led nowhere.
