"""Start the Local Voice Transcriber (local, offline)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = "nemotron-speech-streaming-en-0.6b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                        help="folder that contains one sub-folder per downloaded model")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model (sub-folder name) loaded at start; you can switch in the web page")
    parser.add_argument("--latency", type=int, default=1120, choices=[80, 160, 320, 560, 1120],
                        help="streaming models: delay in ms (lower = snappier, higher = more accurate)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not (Path(args.models_dir) / args.model / "config.json").is_file():
        sys.exit(f"Model not found: {Path(args.models_dir) / args.model}\n"
                 "Download it once with internet:  python scripts/download_model.py")

    # Never reach out to the network: everything is loaded from local files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import uvicorn
    from app.engine import Engine
    from app.server import create_app

    print(f"Loading model {args.model} ...")
    engine = Engine(args.models_dir, args.model, latency_ms=args.latency, device=args.device)
    print(f"Device: {engine.gpu_name}. Ready. Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
