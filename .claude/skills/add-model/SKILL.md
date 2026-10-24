---
name: add-model
description: Steps for adding a new speech-to-text or diarization model to the Local Voice Transcriber (download, KNOWN_MODELS or KNOWN_DIARIZERS entry, UI, docs, check). Use when asked to add, swap or support a new model.
---

# Add a model

1. Find the model's architecture and `transformers` class from its Hugging Face card. Ask the `doc-reader` subagent, or use the Hugging Face connector if it is available. Don't paste the whole model card into the conversation.
2. Decide the kind: `streaming` (cache-aware RNN-T like Nemotron), `offline` (whole recording after Stop, like Parakeet TDT), `vibevoice`, or a diarizer. A new architecture needs a new branch in `Engine.load` and in `Engine.transcribe`; read only those methods.
3. Add the entry next to the existing ones:
   - speech model: `KNOWN_MODELS` in `app/engine.py` with `repo`, `label`, `kind`, `languages`, `language_note`, `modes`, `diarization`, and `needs` for an extra pip package.
   - diarizer: `KNOWN_DIARIZERS` with `repo`, `label`, `backend`, `marker`, `needs`; a new backend also needs a class with `diarize(audio) -> list[dict]` registered in `DIARIZER_BACKENDS`.
4. Add the repo to the docstring list in `scripts/download_model.py`. Extra packages go into `requirements-multispeaker.txt`.
5. The UI reads everything from `/api/info`, so `app.js` usually needs no change.
6. Update the README sections "Models and languages" and "What has been tested" (mark it untested) and spec.md section 5.
7. Ask the `test-runner` subagent for the quick checks. Downloading (GBs) and real transcription only when the user asks.
