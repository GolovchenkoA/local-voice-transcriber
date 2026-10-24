"""One-time model download. Needs internet; afterwards the app runs fully offline.

Usage
  python scripts/download_model.py                 the English live model (default)
  python scripts/download_model.py --all           every model the app offers, except the gated pyannote one
  python scripts/download_model.py REPO [REPO ...] only the named models, for example nvidia/parakeet-tdt-0.6b-v3

Models the app offers in its menus
  Speech models
    nvidia/nemotron-speech-streaming-en-0.6b   English, live
    nvidia/nemotron-3.5-asr-streaming-0.6b     multilingual, live
    nvidia/parakeet-tdt-0.6b-v3                25 languages, text on Stop (single and multiple speakers)
    Dubedo/VibeVoice-ASR-HF-NF4                51 languages, speakers built in, about 6.5 GB on disk
  Diarization models (for multiple speakers with Parakeet TDT)
    nvidia/Nemotron-3-Diarization              up to 8 speakers
    pyannote/speaker-diarization-community-1   gated: needs a Hugging Face login and accepted conditions

The files go to models/<name after the slash>. Run `python scripts/check_setup.py` to see what is missing.
For the gated pyannote model, log in first:  hf auth login   (or set the HF_TOKEN environment variable).
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
ALL_REPOS = [
    "nvidia/nemotron-speech-streaming-en-0.6b",
    "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "nvidia/parakeet-tdt-0.6b-v3",
    "Dubedo/VibeVoice-ASR-HF-NF4",
    "nvidia/Nemotron-3-Diarization",
]
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Skip duplicate or unused formats of the same weights.
IGNORE = ["*.nemo", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "*.tflite"]


def download(repo: str) -> None:
    target = MODELS_DIR / repo.split("/")[-1]
    print(f"Downloading {repo} to {target} ...")
    snapshot_download(repo_id=repo, local_dir=target, ignore_patterns=IGNORE)


def main() -> None:
    args = sys.argv[1:]
    repos = ALL_REPOS if "--all" in args else [a for a in args if not a.startswith("--")] or [DEFAULT_REPO]
    for repo in repos:
        download(repo)
    print("Done. Copy the models folder to the offline computer if you downloaded elsewhere.")
    print("Check the result with:  python scripts/check_setup.py")


if __name__ == "__main__":
    main()
