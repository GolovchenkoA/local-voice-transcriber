"""Check this computer's setup and say exactly what to run for anything that is missing.

Usage:  python scripts/check_setup.py

Why: the app's menus grey out every model that is not downloaded, or whose Python package is missing.
This script shows the same information in the terminal, with the command that fixes each item.
"""

import importlib.metadata as metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, MISSING = "[ok]     ", "[MISSING]"


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    problems = 0
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        print("  note: this project was built and tested with Python 3.12.")

    try:
        import torch
    except ImportError:
        print(f"{MISSING} PyTorch is not installed. Run:")
        print("          pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__}, CUDA available: {cuda}")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory / 1e9:.1f} GB memory")
    else:
        problems += 1
        print(f"{MISSING} CUDA is not available, so the app would run on the CPU (too slow for live use).")
        print("          Update the NVIDIA driver and reinstall PyTorch from the cu128 index (see README).")

    tf = version("transformers")
    print(f"transformers {tf}")
    if tf is None:
        problems += 1
        print(f"{MISSING} Run: pip install -r requirements.txt")

    from app.engine import KNOWN_DIARIZERS, KNOWN_MODELS, package_available

    print("\nSpeech models (Model menu):")
    for mid, meta in KNOWN_MODELS.items():
        folder = ROOT / "models" / mid
        need = meta.get("needs")
        if not (folder / "config.json").is_file():
            problems += 1
            print(f"{MISSING} {mid}\n          python scripts/download_model.py {meta['repo']}")
        elif not package_available(need):
            problems += 1
            print(f"{MISSING} {mid}: Python package '{need}' is not installed\n          pip install -r requirements.txt")
        else:
            print(f"{OK} {mid}")

    print("\nDiarization models (Diarization model menu, Multiple speakers):")
    for did, meta in KNOWN_DIARIZERS.items():
        folder = ROOT / "models" / did
        need = meta.get("needs")
        if not (folder / meta["marker"]).is_file():
            hint = f"python scripts/download_model.py {meta['repo']}"
            if need:
                hint += "   (gated: hf auth login first)"
            print(f"{MISSING} {did}\n          {hint}")
            problems += 1 if not need else 0  # the gated one is optional
        elif not package_available(need):
            print(f"{MISSING} {did}: Python package '{need}' is not installed\n"
                  "          pip install -r requirements-multispeaker.txt")
        else:
            print(f"{OK} {did}")

    if cuda:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gb < 8:
            print(f"\nnote: this GPU has {gb:.1f} GB. VibeVoice-ASR 4-bit needs about 7 GB, so it will probably not fit;"
                  " use Parakeet TDT with a diarization model for multiple speakers.")

    print("\nEverything the app needs is in place." if not problems else
          f"\n{problems} thing(s) to fix. Models that are not ready appear greyed out in the app's menus.")
    print("Quickest fix for all models:  python scripts/download_model.py --all")


if __name__ == "__main__":
    main()
