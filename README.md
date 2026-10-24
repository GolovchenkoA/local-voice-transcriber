# Local Voice Transcriber

A small web app that turns your speech into text on your own computer. It uses your NVIDIA GPU (CUDA) and works without internet after a one-time setup.

- **Several models:** you choose the speech-recognition model and the language in the web page. Live models show text while you speak. Offline models transcribe the whole recording when you press Stop. Included: Nemotron Streaming (English, live), Nemotron 3.5 Streaming (multilingual, live), Parakeet TDT v3 (25 languages, text on Stop) and VibeVoice-ASR (51 languages, speakers built in). See "Models and languages" below.
- **One or several speakers:** a toggle switches between plain text and a dialogue with one coloured row per speaker.
- **Interface:** choose the speakers mode, a model and a language, press **Record**, speak, press **Stop**.
- Details of the design and requirements are in [spec.md](spec.md).

## What you need

| Item | Notes |
|------|-------|
| Windows 11 with an NVIDIA GPU | Built for an RTX 5060 (8 GB). Any CUDA GPU with about 4 GB free should work. |
| Current NVIDIA driver | Check with `nvidia-smi`. It must support CUDA 12.8 or newer. |
| Python 3.12 (64-bit) | The newest Python (3.14) may not have PyTorch wheels yet, so use 3.12. |
| A microphone | And a browser: Chrome, Edge or Firefox. |
| Internet, once | To download PyTorch (about 3 GB) and the model (about 2.5 GB). |

You do not need to install the CUDA Toolkit. PyTorch brings its own CUDA runtime.

## 1. One-time setup (with internet)

Open PowerShell in this folder.

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py -3.12` says `No suitable Python runtime found`, your Python 3.12 is registered under another name. List what is installed:

```powershell
py --list
```

Then use the name shown for 3.12, for example:

```powershell
py -V:Astral/CPython3.12.15 -m venv .venv
```

If no 3.12 is listed, install it first with `winget install Python.Python.3.12`, then open a new PowerShell window.

If `Activate.ps1` is blocked, skip it and put `.\.venv\Scripts\` in front of the commands, for example `.\.venv\Scripts\python.exe -m pip install ...` and `.\.venv\Scripts\python.exe run.py`. Or allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

Install PyTorch with CUDA 12.8 first. The RTX 5060 needs this build or newer:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install the other packages:

```powershell
pip install -r requirements.txt
```

**Download the models.** The models are not part of this repository, and the app only offers models that are on the computer. Anything missing shows up **greyed out** in the menus. To get everything the app can use (about 14 GB, one time):

```powershell
python scripts/download_model.py --all
```

That downloads these five models into `models\`:

| Model | Menu | Size on disk |
|-------|------|--------------|
| `nvidia/nemotron-speech-streaming-en-0.6b` | Model: English, live | 2.3 GB |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | Model: multilingual, live | 2.4 GB |
| `nvidia/parakeet-tdt-0.6b-v3` | Model: 25 languages, text on Stop | 2.3 GB |
| `Dubedo/VibeVoice-ASR-HF-NF4` | Model: speakers built in (4-bit) | 6.5 GB |
| `nvidia/Nemotron-3-Diarization` | Diarization model | 0.4 GB |

You can also download only some of them, for example `python scripts/download_model.py nvidia/parakeet-tdt-0.6b-v3 nvidia/Nemotron-3-Diarization`. Without any argument only the English live model is downloaded. The optional pyannote diarization model is described under "Setting up Multiple speakers" below.

**Check the setup.** This tells you what is missing and the exact command for each item:

```powershell
python scripts/check_setup.py
```

It checks Python, PyTorch and CUDA (it should say `CUDA available: True` and show your GPU), the `transformers` version, and every model and Python package the menus need. Run it again after each fix until it says everything is in place. If CUDA is `False`, see Troubleshooting.

## 2. Run the program

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

Or simply double-click `run.bat` (or run `.\run.bat` in PowerShell). It uses the virtual environment without activating it, and passes extra options on, for example `.\run.bat --latency 160`.

**Start-up and loading.** Models are read from disk into GPU memory each time the app starts (a few seconds per model). To keep this short, the app remembers your last model and diarization model in `settings.json` (next to `run.py`; it belongs to this computer and is not committed) and starts with those, and it keeps the last few used models in memory so that switching back to one is instant. The 4-bit VibeVoice model needs almost all of the GPU memory, so loading it frees the other models, and they are loaded again from disk when you switch back. The Single / Multiple position is remembered by the browser. To forget the remembered choices, delete `settings.json`.

Wait for `Ready. Open http://127.0.0.1:8000`. Loading and warming up the model takes a short while the first time. Then open http://127.0.0.1:8000 in your browser, press **Record**, and allow microphone access.

The badge at the top should show your GPU name with "CUDA". If it says "CPU", the app is not using your GPU.

## Models and languages

The top of the page has a toggle and up to three menus:

- **Speakers (Single / Multiple):** *Single* gives one plain text. *Multiple* gives a dialogue: one row per turn, with a coloured marker and a label ("Speaker 1", "Speaker 2") for each person, and the start time. The dialogue appears after you press Stop. Speaker labels are anonymous and numbered per recording; a new recording starts again at "Speaker 1".
- **Model:** the models available for the chosen toggle position. Models that are not downloaded (or need a missing Python package) are listed greyed out. Switching takes a few seconds while the new model loads, and it is not possible during a recording. When you flip the toggle and the loaded model does not fit, the page loads the first suitable downloaded model for you.
- **Language:** shown when the chosen model has a language setting. Otherwise it is greyed out and says why.
- **Diarization model:** only for *Multiple*. Diarization means finding who spoke when. It is greyed out for models that find the speakers themselves (VibeVoice).

The badges under the title show the real setup: GPU, speaker mode, model, diarization model, live or on-Stop mode, and language.

| Model | Languages | When the text appears |
|-------|-----------|-----------------------|
| Nemotron Streaming 0.6B | English only | While you speak (live) |
| Nemotron 3.5 Streaming 0.6B | About 30 languages in the Language menu, including Polish, Ukrainian and Russian, plus Auto-detect | While you speak (live) |
| Parakeet TDT 0.6B v3 | 25 European languages, detected automatically | **When you press Stop.** The whole recording is transcribed at once, so the Language menu is not used. Recordings are limited to 5 minutes (`--max-minutes`). Single and Multiple speakers. |
| VibeVoice-ASR 4-bit | 51 languages including Polish, Ukrainian and Russian, detected automatically | **When you press Stop.** Finds the speakers itself, so no diarization model is needed. Single and Multiple speakers. |

The live Nemotron models are available only with *Single*, because they do not give word timestamps, which are needed to match words to speakers.

Diarization models (for *Multiple* with Parakeet TDT):

| Diarization model | Speakers | Extra setup |
|-------------------|----------|-------------|
| Nemotron-3-Diarization | Up to 8 | None. Runs through `transformers`. |
| pyannote community-1 | No fixed limit | `pip install pyannote.audio`, and a Hugging Face token because the download is gated (see below). |

More detail and accuracy numbers are in [speech-to-text-models-overview.md](speech-to-text-models-overview.md). The research behind the speaker models is in [investigation/speaker-aware-models.md](investigation/speaker-aware-models.md).

### Setting up Multiple speakers

If you ran `python scripts/download_model.py --all` (step 1), Multiple speakers already works with **Parakeet TDT + Nemotron-3-Diarization** and with **VibeVoice**, because `requirements.txt` includes `bitsandbytes`. Run `python scripts/check_setup.py` to confirm. The only extra, optional piece is the pyannote diarization model:

```powershell
pip install -r requirements-multispeaker.txt                              # installs pyannote.audio
hf auth login                                                             # once; paste a Hugging Face token
python scripts/download_model.py pyannote/speaker-diarization-community-1
```

- pyannote community-1 is **gated**: create a Hugging Face token, accept the conditions on https://huggingface.co/pyannote/speaker-diarization-community-1, and log in once. After the download it works offline.
- `pyannote.audio` pulls in many packages and may change your PyTorch version. Run `python scripts/check_setup.py` afterwards to check that CUDA still works.
- The 4-bit VibeVoice build is a community upload. On the RTX 5060 laptop GPU (8 GB) it held 6.95 GB of GPU memory in use during the test and worked. Close other programs that use the GPU (games, video editors, other models), because Windows itself also uses some GPU memory. A GPU with less than 8 GB will probably not fit it; use Parakeet TDT with a diarization model there. The app frees the diarization model while VibeVoice is loaded.
- VibeVoice and Nemotron-3-Diarization need `transformers` 5.20 or newer (`requirements.txt` already says so).

### What has been tested

All tests used an RTX 5060 Laptop GPU (8 GB), Windows 11, Python 3.12, torch 2.11 (CUDA 12.8), transformers 5.20.2, bitsandbytes 0.50.2.

| Part | Status |
|------|--------|
| English live model | Tested, including the browser. |
| Parakeet TDT with word timestamps and Nemotron-3-Diarization | Tested on a 19-second recording of two voices (Windows speech voices): all four turns were attributed to the right speaker. The diarization step took 0.6 s. Also tested in the browser (with a stand-in diarizer for the timing). |
| VibeVoice 4-bit | Tested on the same recording: four correct turns with the right speakers, 7.7 s for 18.6 s of audio. A pure sound tag such as "[Silence]" is removed from the result. |
| pyannote community-1 | Written from the model documentation but **not run** (needs the gated download). |
| Nemotron 3.5 multilingual (language menu) | Downloaded, but **not run** by me. |
| Real conversations, other languages, 3 or more speakers | **Not tested.** The recording above is synthetic and has two speakers. |

## Parameters

All parameters are optional. `python run.py --help` lists them.

### `run.py`

| Parameter | Default | Values | What it does |
|-----------|---------|--------|--------------|
| `--latency` | `1120` | `80`, `160`, `320`, `560`, `1120` | Live models only: how many milliseconds of upcoming audio the model waits for before it writes a word. This is the most important setting. A longer wait gives the model more context, so it makes fewer mistakes, but the text appears later. `1120` is the most accurate and the default. `160` feels almost instant. `80` is the fastest and the least accurate. If a model does not have the chosen mode, the nearest one is used and a warning is printed. Ignored by offline models (Parakeet TDT). |
| `--max-minutes` | `5` | number | Offline models (Parakeet TDT, VibeVoice) keep the whole recording in memory and work on it after Stop. This is the longest recording; at the limit the recording stops by itself and is transcribed. Raise it with care on an 8 GB GPU: long recordings need more memory. |
| `--device` | `auto` | `auto`, `cuda`, `cpu` | Where the model runs. `auto` uses the NVIDIA GPU (CUDA) when it is available and otherwise the CPU, with a warning. `cuda` stops with an error if CUDA is missing. `cpu` is too slow for live use. |
| `--model` | the model used last time, else `nemotron-speech-streaming-en-0.6b` | folder name in `models\` | Model loaded at start. Without this option the app starts with the model you used last time (and its diarization model). You can switch models later in the web page. |
| `--models-dir` | `models` | folder path | Folder that holds one sub-folder per downloaded model. Change it only if you stored the models somewhere else. |
| `--host` | `127.0.0.1` | IP address | Address the web server listens on. The default makes the app reachable only from this computer. Do not change it unless you understand the privacy impact: other computers on your network could then use your microphone page and GPU. |
| `--port` | `8000` | number | Port of the web page. Change it if 8000 is already used by another program. |

Examples:

```powershell
python run.py                    # accurate (1120 ms), default
python run.py --latency 160      # snappier, a little less accurate
python run.py --port 8080        # use another port
python run.py --device cuda      # refuse to run without the GPU
```

### `scripts/transcribe_file.py`

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `wav` (required) | none | Path to a 16 kHz, mono, 16-bit WAV file. |
| `--latency` | `1120` | Same as above. |
| `--model-dir` | `models\nemotron-speech-streaming-en-0.6b` | Folder of one model, for example `models\parakeet-tdt-0.6b-v3`. |
| `--language` | none | Multilingual live model only, for example `pl-PL`, `uk-UA`, `ru-RU` or `auto`. |

### `scripts/download_model.py`

| Argument | What it does |
|----------|--------------|
| none | Downloads the English live model only. |
| `--all` | Downloads all five models of step 1 (everything except the gated pyannote model). |
| one or more Hugging Face names, for example `nvidia/parakeet-tdt-0.6b-v3` | Downloads just those models. |

The files go to `models\<model name>`. Gated models (pyannote) need a Hugging Face login first.

### `scripts/check_setup.py`

No arguments. Reports Python, PyTorch, CUDA, `transformers`, and every model and package the menus need, with the command that fixes each missing item.

### Settings that are not parameters

Microphone noise suppression and echo cancellation are switched on in the browser. To transcribe video or music from the computer, use the microphone button left of **Record** and tick **System audio**: the browser asks you to share a tab or screen, and you must tick **Share audio** in that dialog. This asks every time you press Record, and sharing a tab captures only that tab (share the entire screen to get the sound of all apps). The speakers toggle, the model, the language and the diarization model are chosen in the web page.

## 3. Using it on a computer with no internet

1. On a computer with internet, do all of step 1 (the same Python version and the same type of Windows).
2. Copy the whole project folder, including `.venv` and `models`, to the offline computer. If a copied `.venv` does not start, instead download the wheels with `pip download -r requirements.txt -d wheels` (and the PyTorch wheel with `--index-url https://download.pytorch.org/whl/cu128`), copy the `wheels` folder, and install with `pip install --no-index --find-links wheels -r requirements.txt`.
3. Run `python scripts/check_setup.py` there to make sure every model you want is present. A model that is missing is greyed out in the menus.
4. Run step 2. The app sets `HF_HUB_OFFLINE=1` itself and the web page loads nothing from the internet.

## Test without a browser

You can check the engine and measure its speed on a 16 kHz, mono, 16-bit WAV file:

```powershell
python scripts/transcribe_file.py path\to\speech.wav
```

It prints the text and the real-time factor. A value below 1.0 means it keeps up with live speech.

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `py -3.12` says `No suitable Python runtime found` | Run `py --list` and use the exact name it shows for 3.12 (see step 1), or install Python 3.12 with `winget install Python.Python.3.12`. |
| `Activate.ps1` is blocked by the execution policy | Run `.\.venv\Scripts\python.exe run.py` without activating, or run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once. |
| `torch.cuda.is_available()` is `False` | Update the NVIDIA driver, and reinstall PyTorch from the `cu128` index shown above. |
| `no kernel image is available` or `sm_120` errors | Your PyTorch is too old for the RTX 50 series. Reinstall with the `cu128` (or newer) index. |
| `Model not found` at start | Run `python scripts/download_model.py` once with internet. |
| A model or diarization model in a menu is greyed out, or a menu looks empty | The app offers only what is on this computer. Run `python scripts/check_setup.py`. It lists every missing model or Python package with the command that fixes it. Quickest fix: `python scripts/download_model.py --all`, then reload the page. (The models are not part of the repository, so a new computer always starts with none.) |
| Choosing the multilingual or the offline (Parakeet TDT) model gives an error | These two are untested here. Send me the message. The installed `transformers` version may not support them yet, so try `pip install -U transformers`. |
| Nothing appears while recording with an offline model (Parakeet TDT, VibeVoice) | That is expected. It transcribes after you press Stop. |
| "Choose a diarization model first" | Multiple speakers with Parakeet TDT needs a diarization model. Download one (see "Setting up Multiple speakers") and pick it in the menu. |
| A model shows "(needs: pip install ...)" | A Python package is missing. For VibeVoice run `pip install -r requirements.txt` (it contains `bitsandbytes`); for pyannote run `pip install -r requirements-multispeaker.txt`. Then restart the app. |
| VibeVoice runs out of GPU memory | The 4-bit build holds about 7 GB. Close other programs that use the GPU, record less, or use Parakeet TDT with a diarization model. |
| VibeVoice stops with `Some modules are dispatched on the CPU` or a `bitsandbytes` assertion | You are running an older copy of the app. Update to the current version, which loads the whole model on the GPU and fixes the layers that the 4-bit checkpoint stores in full precision. |
| The terminal prints `Loading <model> ...` and `<model> is ready (loaded in N s)` | That is normal. The model is being read from disk into GPU memory. Harmless warnings (triton, generation length, RNN weights) are hidden. If you see other warnings or errors, they are real. |
| Starting the app takes several seconds | A model must be read from disk into GPU memory every time the app starts, because GPU memory is emptied when the program exits. Measured on an RTX 5060 laptop: English live model 5 s, Parakeet TDT 4 s, Nemotron-3-Diarization 1 s, VibeVoice 11 s. The app loads only the model you used last time, and keeps recently used models in memory, so switching back to one is instant (about 0.2 s). |
| Wrong speaker for a few words at a change of speaker | The diarization model and the word timestamps are only accurate to a fraction of a second, and overlapping speech is hard. A different diarization model may do better. |
| Import or attribute errors about `AutoModelForRNNT` or the processor | Your `transformers` is too old. It must be 5.13 or newer: `pip install -U "transformers>=5.13.0"`. The streaming API is experimental and may change in later versions. |
| Browser says the microphone is blocked | Click the lock icon in the address bar and allow the microphone. Use `http://127.0.0.1:8000` or `http://localhost:8000`; other addresses are not allowed to use the microphone. |
| The page says "busy" | Another tab or window is already recording. Stop it first. |
| Text appears too late | Use a lower delay: `--latency 560` or `--latency 160`. Accuracy drops a little. |
| Text is slow for another reason | Close other GPU-heavy programs. |

## Project layout

```
spec.md                      what the app does and why
run.py                       starts the server
app/engine.py                model loading, model switching, streaming and offline transcription (CUDA)
app/server.py                web server, model menu endpoints and WebSocket
app/static/                  the web page (HTML, CSS, JavaScript, audio worklet)
scripts/download_model.py    one-time model download (--all for every model)
scripts/check_setup.py       shows what is missing on this computer and how to fix it
scripts/transcribe_file.py   test the engine on a WAV file
```

## License note

The model is released by NVIDIA under the NVIDIA Open Model License. Read it before using the model beyond personal use.
