"""One-time model download. Needs internet; afterwards the app runs fully offline.

Usage:  python scripts/download_model.py [repo_id]

repo_id defaults to nvidia/nemotron-speech-streaming-en-0.6b. Models this app offers in its menu:
  nvidia/nemotron-speech-streaming-en-0.6b   English, live
  nvidia/nemotron-3.5-asr-streaming-0.6b     multilingual, live
  nvidia/parakeet-tdt-0.6b-v3                25 languages, transcribed when you press Stop
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "nvidia/nemotron-speech-streaming-en-0.6b"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    target = MODELS_DIR / repo.split("/")[-1]
    print(f"Downloading {repo} to {target} ...")
    # Only the files the app needs: skips the duplicate .nemo and .gguf files.
    snapshot_download(repo_id=repo, local_dir=target, allow_patterns=[
        "*.json", "*.safetensors", "*.txt", "*.model", "*.md",
    ])
    print("Done. Copy the models folder to the offline computer if you downloaded elsewhere.")


if __name__ == "__main__":
    main()
