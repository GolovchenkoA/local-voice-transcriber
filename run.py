"""Start the Local Voice Transcriber (local, offline)."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULT_MODELS_DIR = ROOT / "models"
SETTINGS_PATH = ROOT / "settings.json"  # remembers the last model and diarization model
DEFAULT_MODEL = "nemotron-speech-streaming-en-0.6b"


def remembered() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def quiet_logs() -> None:
    """Hide a few harmless messages that would print again and again."""
    import warnings

    warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")
    warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
    # Windows has no triton; torch notes that once when it is imported
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

    def drop_length_notice(record: logging.LogRecord) -> bool:
        return "Both `max_new_tokens`" not in record.getMessage()

    for name in ("transformers.generation.utils", "transformers.generation.configuration_utils"):
        logging.getLogger(name).addFilter(drop_length_notice)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                        help="folder that contains one sub-folder per downloaded model")
    parser.add_argument("--model", default=None,
                        help="model (sub-folder name) loaded at start; default: the model used last time, "
                             "or the English live model. You can switch in the web page.")
    parser.add_argument("--latency", type=int, default=1120, choices=[80, 160, 320, 560, 1120],
                        help="streaming models: delay in ms (lower = snappier, higher = more accurate)")
    parser.add_argument("--max-minutes", type=float, default=5,
                        help="offline models (Parakeet TDT, VibeVoice): longest recording in minutes")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    saved = remembered()
    model = args.model or saved.get("model") or DEFAULT_MODEL
    if not (Path(args.models_dir) / model / "config.json").is_file():
        if args.model:
            sys.exit(f"Model not found: {Path(args.models_dir) / model}\n"
                     "Download models once with internet:  python scripts/download_model.py --all")
        model = DEFAULT_MODEL  # the remembered model is gone; fall back
        if not (Path(args.models_dir) / model / "config.json").is_file():
            sys.exit(f"No model found in {args.models_dir}.\n"
                     "Download models once with internet:  python scripts/download_model.py --all\n"
                     "Then check the setup with:  python scripts/check_setup.py")

    # Never reach out to the network: everything is loaded from local files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    quiet_logs()

    import transformers
    import uvicorn
    from app.engine import Engine
    from app.server import create_app

    transformers.utils.logging.disable_progress_bar()  # the app prints one clear line per model instead

    engine = Engine(args.models_dir, model, latency_ms=args.latency, device=args.device,
                    max_seconds=int(args.max_minutes * 60), settings_path=SETTINGS_PATH)

    # Bring back the diarization model used last time, if the loaded model needs one.
    last_diarizer = saved.get("diarizer")
    if last_diarizer and engine.meta.get("diarization") == "external":
        try:
            engine.load_diarizer(last_diarizer)
        except Exception as exc:
            print(f"Note: could not load the diarization model {last_diarizer}: {exc}")

    print(f"Device: {engine.gpu_name}. Ready. Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
