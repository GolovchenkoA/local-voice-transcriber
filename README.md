# Local Voice Transcriber

A small web app that turns your speech into text on your own computer. It uses your NVIDIA GPU (CUDA) and works without internet after a one-time setup.

- **Several models:** you choose the speech-recognition model and the language in the web page. Live models show text while you speak. Offline models transcribe the whole recording when you press Stop. Included: Nemotron Streaming (English, live), Nemotron 3.5 Streaming (multilingual, live) and Parakeet TDT v3 (25 languages, text on Stop). See "Models and languages" below.
- **Interface:** choose a model and a language, press **Record**, speak, press **Stop**.
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

Download the English live model into `models\nemotron-speech-streaming-en-0.6b` (2.4 GB):

```powershell
python scripts/download_model.py
```

Other models are optional. Each one appears in the page's Model menu once it is downloaded (see "Models and languages" below):

```powershell
python scripts/download_model.py nvidia/nemotron-3.5-asr-streaming-0.6b   # multilingual, live
python scripts/download_model.py nvidia/parakeet-tdt-0.6b-v3              # 25 languages, text appears on Stop
```

Check that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and your GPU name. If it prints `False`, see Troubleshooting.

## 2. Run the program

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

Wait for `Ready. Open http://127.0.0.1:8000`. Loading and warming up the model takes a short while the first time. Then open http://127.0.0.1:8000 in your browser, press **Record**, and allow microphone access.

The badge at the top should show your GPU name with "CUDA". If it says "CPU", the app is not using your GPU.

## Models and languages

The top of the page has two menus:

- **Model:** every model whose folder exists in `models\` can be chosen. Models that are not downloaded are listed greyed out. Switching takes a few seconds while the new model loads, and it is not possible during a recording.
- **Language:** shown when the chosen model has a language setting. Otherwise it is greyed out and says why.

| Model | Languages | When the text appears |
|-------|-----------|-----------------------|
| Nemotron Streaming 0.6B | English only | While you speak (live) |
| Nemotron 3.5 Streaming 0.6B | About 30 languages in the Language menu, including Polish, Ukrainian and Russian, plus Auto-detect | While you speak (live) |
| Parakeet TDT 0.6B v3 | 25 European languages, detected automatically | **When you press Stop.** The whole recording is transcribed at once, so the Language menu is not used. Recordings are limited to 5 minutes. |

More detail and accuracy numbers are in [speech-to-text-models-overview.md](speech-to-text-models-overview.md).

Status of each model in this project: the English model is tested on an RTX 5060. The other two are implemented from NVIDIA's documentation but **have not been tested here yet**.

## Parameters

All parameters are optional. `python run.py --help` lists them.

### `run.py`

| Parameter | Default | Values | What it does |
|-----------|---------|--------|--------------|
| `--latency` | `1120` | `80`, `160`, `320`, `560`, `1120` | Live models only: how many milliseconds of upcoming audio the model waits for before it writes a word. This is the most important setting. A longer wait gives the model more context, so it makes fewer mistakes, but the text appears later. `1120` is the most accurate and the default. `160` feels almost instant. `80` is the fastest and the least accurate. If a model does not have the chosen mode, the nearest one is used and a warning is printed. Ignored by offline models (Parakeet TDT). |
| `--device` | `auto` | `auto`, `cuda`, `cpu` | Where the model runs. `auto` uses the NVIDIA GPU (CUDA) when it is available and otherwise the CPU, with a warning. `cuda` stops with an error if CUDA is missing. `cpu` is too slow for live use. |
| `--model` | `nemotron-speech-streaming-en-0.6b` | folder name in `models\` | Model loaded at start. You can switch models later in the web page. |
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

Optional first argument: the Hugging Face name of the model, for example `nvidia/parakeet-tdt-0.6b-v3`. Without it, the English live model is downloaded. The files go to `models\<model name>`.

### Settings that are not parameters

Microphone noise suppression and echo cancellation are switched on in the browser. The model and the language are chosen in the web page.

## 3. Using it on a computer with no internet

1. On a computer with internet, do all of step 1 (the same Python version and the same type of Windows).
2. Copy the whole project folder, including `.venv` and `models`, to the offline computer. If a copied `.venv` does not start, instead download the wheels with `pip download -r requirements.txt -d wheels` (and the PyTorch wheel with `--index-url https://download.pytorch.org/whl/cu128`), copy the `wheels` folder, and install with `pip install --no-index --find-links wheels -r requirements.txt`.
3. Run step 2 there. The app sets `HF_HUB_OFFLINE=1` itself and the web page loads nothing from the internet.

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
| A model in the menu is greyed out "(not downloaded)" | Download it with `python scripts/download_model.py <name>` (see step 1), then reload the page. |
| Choosing the multilingual or the offline (Parakeet TDT) model gives an error | These two are untested here. Send me the message. The installed `transformers` version may not support them yet, so try `pip install -U transformers`. |
| Nothing appears while recording with an offline model (Parakeet TDT) | That is expected. It transcribes after you press Stop. |
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
scripts/download_model.py    one-time model download
scripts/transcribe_file.py   test the engine on a WAV file
```

## License note

The model is released by NVIDIA under the NVIDIA Open Model License. Read it before using the model beyond personal use.
